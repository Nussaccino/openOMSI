//! DLSS's depth prepass: the ordinary prepass writing each pixel's motion beside its depth,
//! the sky's motion laid first, and the rain films' drops (see `crate::dlss` and
//! `dlss_motion.wgsl`).

use super::common::*;
use super::scene::SceneBase;
use crate::*;

/// DLSS's motion vectors: pixels of the render size from each pixel to where its point was
/// last frame.
pub(crate) const DLSS_MOTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;
/// DLSS's result at the window's size (a storage texture: sRGB formats cannot be one).
pub(crate) const DLSS_OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// DLSS's `BiasCurrentColor` mask: 1 where the current picture goes before the history (the
/// rain films' drops).
pub(crate) const DLSS_BIAS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// The DLSS depth prepass: [plain, alpha-tested, transmap] x [two-sided, culled] as the
/// ordinary prepass, writing motion beside depth, and the sky's motion.
pub(crate) struct DlssPipelines {
    pub prepass: [wgpu::RenderPipeline; 6],
    pub sky: wgpu::RenderPipeline,
    /// The drops of the rain films: their motion (`fs_motion_rain`) and the bias mask
    /// (`fs_bias_rain`).
    pub rain: wgpu::RenderPipeline,
    pub rain_bias: wgpu::RenderPipeline,
    pub motion_layout: wgpu::BindGroupLayout,
}

pub(crate) fn build(device: &wgpu::Device, scene: &SceneBase) -> DlssPipelines {
    let motion_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("dlss motion"),
        entries: &[
            uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("dlss prepass"),
        // (group 2 is the vehicle reflection's in shader.wgsl: DLSS's motion is group 3)
        bind_group_layouts: &[Some(&scene.camera_layout), Some(&scene.material_layout), None, Some(&motion_layout)],
        immediate_size: 0,
    });
    let colour = |format, write_mask| Some(wgpu::ColorTargetState { format, blend: None, write_mask });
    // (the prepass's two targets: the motion, and the bias mask only the drops write)
    let motion_target = [colour(DLSS_MOTION_FORMAT, wgpu::ColorWrites::ALL), colour(DLSS_BIAS_FORMAT, wgpu::ColorWrites::empty())];
    let bias_target = [colour(DLSS_MOTION_FORMAT, wgpu::ColorWrites::empty()), colour(DLSS_BIAS_FORMAT, wgpu::ColorWrites::ALL)];
    let depth = |write: bool, compare| wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(write),
        depth_compare: Some(compare),
        stencil: Default::default(),
        bias: Default::default(),
    };
    // the scene's meshes, by `vs_motion` (the main pass's vertex to the last bit)
    let mesh = |label, fragment, cull: bool, write: bool, targets: &[Option<wgpu::ColorTargetState>]| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &scene.shader,
                entry_point: Some("vs_motion"),
                buffers: &[scene.vertex_layout.clone()],
                compilation_options: Default::default(),
            },
            primitive: one_sided_primitive(cull),
            depth_stencil: Some(depth(write, wgpu::CompareFunction::GreaterEqual)),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &scene.shader,
                entry_point: Some(fragment),
                targets,
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    let make = |kind: u8, cull: bool| {
        let fragment = match kind {
            0 => "fs_motion",
            1 => "fs_motion_test",
            2 => "fs_motion_transmap",
            _ => unreachable!(),
        };
        mesh("dlss depth prepass", fragment, cull, true, &motion_target)
    };
    let sky_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("dlss sky motion"),
        bind_group_layouts: &[None, None, None, Some(&motion_layout)],
        immediate_size: 0,
    });
    let sky = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("dlss sky motion"),
        layout: Some(&sky_pl),
        vertex: wgpu::VertexState {
            module: &scene.shader,
            entry_point: Some("vs_motion_sky"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(depth(false, wgpu::CompareFunction::Always)),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &scene.shader,
            entry_point: Some("fs_motion_sky"),
            targets: &motion_target,
            compilation_options: Default::default(),
        }),
        multiview_mask: None,
        cache: None,
    });
    // the rain films' drops: their pane's motion over what the prepass left, the depth
    // DLSS reads untouched
    let rain = mesh("dlss rain motion", "fs_motion_rain", false, false, &motion_target);
    let rain_bias = mesh("dlss rain bias", "fs_bias_rain", false, false, &bias_target);
    DlssPipelines {
        prepass: [make(0, false), make(0, true), make(1, false), make(1, true), make(2, false), make(2, true)],
        sky,
        rain,
        rain_bias,
        motion_layout,
    }
}
