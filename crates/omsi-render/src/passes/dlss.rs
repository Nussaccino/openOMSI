//! DLSS's part of a frame (see `crate::dlss`): Streamline started at the first window frame
//! that asks for it, the projection's sub-pixel jitter and the motion uniforms, the inputs'
//! states at the end of the scene's commands, and after the scene's submit DLSS itself with
//! the HUD drawn over its result.

use super::*;
use crate::pipelines::dlss::{DLSS_BIAS_FORMAT, DLSS_MOTION_FORMAT, DLSS_OUTPUT_FORMAT};

/// `Motion` in dlss_motion.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MotionUniform {
    vp: [[f32; 4]; 4],
    vp_prev: [[f32; 4]; 4],
    vp_prev_here: [[f32; 4]; 4],
    vp_inv: [[f32; 4]; 4],
    params: [f32; 4],
}

/// What DLSS reads and writes for one window size and mode.
pub(crate) struct DlssTargets {
    render: (u32, u32),
    output: (u32, u32),
    /// The 3D picture at the render size (the renderer's format), with the upscale bind group
    /// the rain on the glass reads it by.
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    color_bg: wgpu::BindGroup,
    pub motion: wgpu::Texture,
    pub motion_view: wgpu::TextureView,
    /// DLSS's `BiasCurrentColor` mask.
    pub bias: wgpu::Texture,
    pub bias_view: wgpu::TextureView,
    /// DLSS's result, and the bind group that draws it into the window.
    output_tex: wgpu::Texture,
    output_view: wgpu::TextureView,
    output_bg: wgpu::BindGroup,
}

/// Last frame's camera, for the motion vectors and DLSS's `clipToPrevClip`.
#[derive(Clone, Copy)]
struct DlssPrev {
    camera: Camera,
    aspect: f32,
    origin: DVec3,
    render: (u32, u32),
    output: (u32, u32),
    mode: DlssMode,
}

/// DLSS once Streamline is up: its runtime, targets, and what the last frame left for the
/// next one's motion vectors.
pub(crate) struct DlssState {
    runtime: crate::dlss::Runtime,
    pub targets: Option<DlssTargets>,
    /// Textures DLSS read in the last frames: a texture wgpu lets go of when its last submit is
    /// done, and DLSS's command list comes after that submit.
    keep: std::collections::VecDeque<Vec<wgpu::Texture>>,
    motion_buf: wgpu::Buffer,
    /// Blit parameters of the result into the window (`upscale.wgsl` at the window's size).
    blit_buf: wgpu::Buffer,
    /// Last frame's model matrices (copied at the end of each frame) and its bind group with
    /// `motion_buf`.
    pub prev_models: Option<(wgpu::Buffer, wgpu::BindGroup)>,
    /// The scene's model layout (`Scene::model_epoch`) and entry count the copy was made with.
    prev_epoch: u64,
    prev_entries: u32,
    prev: Option<DlssPrev>,
    frame: u64,
}

impl DlssState {
    /// The state carried into a renderer built anew (a fallback after a GPU error): Streamline
    /// starts once a process, so it is kept; what was made with the old pipelines is not, and
    /// the history starts again.
    pub(crate) fn rebuilt(self) -> DlssState {
        DlssState { prev_models: None, prev_epoch: u64::MAX, prev: None, ..self }
    }
}

/// What a DLSS frame hands on from its camera to DLSS (after the scene's submit).
pub(crate) struct DlssFrame {
    /// This frame's view-projection without the jitter, and last frame's relative to this
    /// frame's render origin.
    vp: Mat4,
    vp_prev_here: Mat4,
    /// The offset the projection was moved by, in pixels.
    jitter: [f32; 2],
    reset: bool,
}

/// The Halton sequence's `i`th value (from 1) in `base`, 0..1.
fn halton(mut i: u32, base: u32) -> f32 {
    let (mut f, mut r) = (1.0f32, 0.0f32);
    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
}

/// Matrices as DLSS takes them: Streamline's rows are glam's columns (row vectors).
fn sl_matrix(m: Mat4) -> [[f32; 4]; 4] {
    m.to_cols_array_2d()
}

impl Renderer {
    /// Another DLSS quality while the game runs: the next frame is drawn at its size (DLSS
    /// on or off is not changed here - its pipelines are made when the renderer is). True
    /// when it was taken.
    pub fn set_dlss_mode(&mut self, mode: DlssMode) -> bool {
        if !(self.options.dlss.is_on() && mode.is_on() && self.dlss_pipes.is_some()) {
            return false;
        }
        if self.options.dlss != mode {
            log::info!("DLSS: {} from now on", mode.as_str());
            self.options.dlss = mode;
        }
        true
    }

    /// Whether DLSS draws this frame (the window's own picture: not a mirror's, an eye's or
    /// one for a texture), and its render size for the window's. Streamline is started at
    /// the first frame that asks (once; when it cannot, the game says why and draws as
    /// without DLSS).
    pub(crate) fn dlss_frame_size(&mut self, with_overlays: bool, projection: Option<Mat4>, full: (u32, u32)) -> Option<(u32, u32)> {
        if !(with_overlays && projection.is_none() && self.dlss_window && self.texture_aspect.is_none() && self.dlss_start()) {
            return None;
        }
        let size = self.dlss.as_mut().expect("dlss state").runtime.render_size(self.options.dlss, full);
        self.dlss_targets(size, full);
        Some(size)
    }

    fn dlss_start(&mut self) -> bool {
        if self.dlss.is_some() {
            return true;
        }
        if self.dlss_failed || self.dlss_pipes.is_none() || !self.options.dlss.is_on() {
            return false;
        }
        match crate::dlss::Runtime::load(&self.device, &self.queue) {
            Ok(runtime) => {
                let buf = |label: &str, size: u64| {
                    self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(label),
                        size,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                };
                self.dlss = Some(DlssState {
                    runtime,
                    targets: None,
                    keep: Default::default(),
                    motion_buf: buf("dlss motion", std::mem::size_of::<MotionUniform>() as u64),
                    blit_buf: buf("dlss blit", 16),
                    prev_models: None,
                    prev_epoch: u64::MAX,
                    prev_entries: 0,
                    prev: None,
                    frame: 0,
                });
                log::info!("DLSS {} on: {}", self.options.dlss.as_str(), self.adapter_name);
                true
            }
            Err(e) => {
                log::warn!("DLSS cannot start: {e:#}; drawing without it");
                self.dlss_failed = true;
                false
            }
        }
    }

    /// DLSS's targets for this render and window size (made anew when either changes).
    fn dlss_targets(&mut self, render: (u32, u32), output: (u32, u32)) {
        let Some(blit_buf) = self.dlss.as_ref().filter(|d| !d.targets.as_ref().is_some_and(|t| t.render == render && t.output == output)).map(|d| d.blit_buf.clone()) else {
            return;
        };
        let tex = |label: &str, (w, h): (u32, u32), format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let targets = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        // (copied from by the rain on the glass, as the render scale's target)
        let color = tex("dlss scene", render, self.format, targets | wgpu::TextureUsages::COPY_SRC);
        let motion = tex("dlss motion vectors", render, DLSS_MOTION_FORMAT, targets);
        let bias = tex("dlss bias mask", render, DLSS_BIAS_FORMAT, targets);
        let output_tex = tex("dlss output", output, DLSS_OUTPUT_FORMAT, targets | wgpu::TextureUsages::STORAGE_BINDING);
        let color_view = color.create_view(&Default::default());
        let motion_view = motion.create_view(&Default::default());
        let bias_view = bias.create_view(&Default::default());
        let output_view = output_tex.create_view(&Default::default());
        let color_bg = self.picture_group(&color_view);
        let output_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dlss upscale"),
            layout: &self.upscale_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: blit_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&output_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
            ],
        });
        let d = self.dlss.as_mut().expect("dlss state");
        // (the old ones may still be read by DLSS's last command list; the history belongs to
        // the old size)
        if let Some(old) = d.targets.take() {
            d.keep.push_back(vec![old.color, old.motion, old.bias, old.output_tex]);
        }
        d.prev = None;
        d.targets = Some(DlssTargets { render, output, color, color_view, color_bg, motion, motion_view, bias, bias_view, output_tex, output_view, output_bg });
    }

    /// The scene target of a DLSS frame: its picture at the render size, and the bind group
    /// that reads it.
    pub(crate) fn dlss_scene_target(&self) -> Option<(wgpu::TextureView, wgpu::BindGroup)> {
        self.dlss.as_ref().and_then(|d| d.targets.as_ref()).map(|t| (t.color_view.clone(), t.color_bg.clone()))
    }

    /// The projection of a DLSS frame: every frame moved by another sub-pixel offset (Halton
    /// 2, 3; the more DLSS scales up, the more phases), from which DLSS puts the finer
    /// picture together. The motion vectors are worked out without it (`vp_plain`).
    pub(crate) fn dlss_jitter(&mut self, scene: &Scene, f: &FrameCtx, vp_plain: Mat4) -> (Mat4, Option<DlssFrame>) {
        if !f.dlss_frame {
            return (vp_plain, None);
        }
        let (camera, ro, (width, height), (full_w, full_h)) = (f.camera, f.ro, (f.width, f.height), (f.full_w, f.full_h));
        let mode = self.options.dlss;
        let motion_layout = &self.dlss_pipes.as_ref().expect("dlss pipelines").motion_layout;
        let d = self.dlss.as_mut().expect("dlss state");
        d.frame += 1;
        let ratio = (full_w as f32 / width.max(1) as f32).max(1.0);
        let phases = ((8.0 * ratio * ratio).round() as u32).clamp(8, 64);
        let i = (d.frame % phases as u64) as u32 + 1;
        // (OMSI_DLSS_JITTER_SIGN=-1 tells DLSS the offset the other way round: for trying
        // which way this driver reads it)
        let sign = omsi_cfg::flags::OMSI_DLSS_JITTER_SIGN.parse::<f32>().unwrap_or(1.0).signum();
        let jitter = [halton(i, 2) - 0.5, halton(i, 3) - 0.5];
        // (in pixels, x right and y down)
        let shift = Mat4::from_translation(Vec3::new(2.0 * jitter[0] / width as f32, -2.0 * jitter[1] / height as f32, 0.0));
        // last frame's camera, when it is this picture's history
        let prev = d.prev.filter(|p| p.render == (width, height) && p.output == (full_w, full_h) && p.mode == mode);
        let jumped = prev.is_some_and(|p| (p.camera.position - camera.position).length() > 30.0 || p.camera.forward().dot(camera.forward()) < 0.8);
        let (vp_prev, vp_prev_here) = match prev {
            Some(p) => (p.camera.view_proj(p.aspect, p.origin), p.camera.view_proj(p.aspect, ro)),
            None => (vp_plain, vp_plain),
        };
        // last frame's model matrices: a buffer as big as the models buffer, copied into at
        // the end of each frame
        // (a buffer: DLSS draws on DirectX 12, never on the chips that read the models from a
        // texture)
        if let Some(GpuArray::Buffer(models)) = scene.model_buf.as_ref() {
            if d.prev_models.as_ref().is_none_or(|(b, _)| b.size() < models.size()) {
                let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("dlss last models"),
                    size: models.size(),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("dlss motion"),
                    layout: motion_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: d.motion_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: buf.as_entire_binding() },
                    ],
                });
                d.prev_models = Some((buf, bg));
                d.prev_epoch = u64::MAX;
            }
        }
        let models_ok = prev.is_some() && d.prev_epoch == scene.model_epoch && d.prev_models.is_some();
        let mu = MotionUniform {
            vp: vp_plain.to_cols_array_2d(),
            vp_prev: vp_prev.to_cols_array_2d(),
            vp_prev_here: vp_prev_here.to_cols_array_2d(),
            vp_inv: vp_plain.inverse().to_cols_array_2d(),
            params: [width as f32, height as f32, if models_ok { 1.0 } else { 0.0 }, d.prev_entries as f32],
        };
        self.queue.write_buffer(&d.motion_buf, 0, bytemuck::bytes_of(&mu));
        let frame = DlssFrame { vp: vp_plain, vp_prev_here, jitter: [sign * jitter[0], sign * jitter[1]], reset: prev.is_none() || jumped };
        (shift * vp_plain, Some(frame))
    }

    /// The prepass of a DLSS frame: its pipelines, targets and the motion bind group (None
    /// until the first frame made them).
    pub(crate) fn dlss_prepass(&self, f: &FrameCtx) -> Option<(&DlssPipelines, &DlssTargets, &wgpu::BindGroup)> {
        if !f.dlss_frame {
            return None;
        }
        let d = self.dlss.as_ref()?;
        Some((self.dlss_pipes.as_ref()?, d.targets.as_ref()?, &d.prev_models.as_ref()?.1))
    }

    /// The end of a DLSS frame's scene commands: this frame's model matrices kept for the
    /// next one's motion vectors, and each texture DLSS reads left in the state DLSS is told
    /// (`crate::dlss::Runtime::evaluate`): wgpu leaves a texture in the state of its last use, and
    /// these passes of nothing but their attachments are that last use - colour targets,
    /// the depth written.
    pub(crate) fn dlss_end_of_scene(&mut self, encoder: &mut wgpu::CommandEncoder, scene: &Scene) {
        let Some(d) = self.dlss.as_mut() else { return };
        // (the prepass read last frame's copy before this: its commands come first)
        if let (Some(GpuArray::Buffer(models)), Some((prev, _))) = (scene.model_buf.as_ref(), d.prev_models.as_ref()) {
            let bytes = (scene.uploaded_entries as u64 * 64).min(models.size()).min(prev.size());
            if bytes > 0 {
                encoder.copy_buffer_to_buffer(models, 0, prev, 0, bytes);
            }
            d.prev_epoch = scene.model_epoch;
            d.prev_entries = (bytes / 64) as u32;
        }
        let (Some(t), Some(ao)) = (d.targets.as_ref(), self.ao.as_ref()) else { return };
        let keep = |view| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })
        };
        drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dlss inputs"),
            color_attachments: &[keep(&t.color_view), keep(&t.motion_view), keep(&t.bias_view)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &ao.depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        }));
        drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dlss output"),
            color_attachments: &[keep(&t.output_view)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        }));
    }

    /// After a DLSS frame's scene submit: DLSS on wgpu's queue, then its result drawn into
    /// the window with the HUD over it. When DLSS fails, the picture is scaled up as without
    /// it, and DLSS stays off from then on.
    pub(crate) fn dlss_present(&mut self, scene: &Scene, f: &FrameCtx, frame: DlssFrame) {
        let (camera, ro, render, full) = (f.camera, f.ro, (f.width, f.height), (f.full_w, f.full_h));
        let mode = self.options.dlss;
        let (Some(d), Some(ao)) = (self.dlss.as_mut(), self.ao.as_ref()) else { return };
        let Some(t) = d.targets.as_ref() else { return };
        let fov = camera.fov_deg.to_radians();
        let proj = Mat4::perspective_rh(fov, f.aspect, camera.far, camera.near);
        let constants = crate::dlss::FrameConstants {
            view_to_clip: sl_matrix(proj),
            clip_to_view: sl_matrix(proj.inverse()),
            clip_to_prev_clip: sl_matrix(frame.vp_prev_here * frame.vp.inverse()),
            prev_clip_to_clip: sl_matrix(frame.vp * frame.vp_prev_here.inverse()),
            jitter: frame.jitter,
            camera_pos: (camera.position - ro).as_vec3().to_array(),
            camera_up: camera.up().to_array(),
            camera_right: camera.right().to_array(),
            camera_fwd: camera.forward().to_array(),
            near: camera.near,
            far: camera.far,
            fov,
            aspect: f.aspect,
            reset: frame.reset,
        };
        let inputs = crate::dlss::Inputs { color: &t.color, depth: &ao.depth, motion: &t.motion, bias: &t.bias, output: &t.output_tex, render_size: render, output_size: full };
        let ok = match d.runtime.evaluate(&self.device, &self.queue, mode, &inputs, &constants) {
            Ok(()) => true,
            Err(e) => {
                log::error!("DLSS failed: {e:#}; drawing without it from now on");
                false
            }
        };
        d.keep.push_back(vec![t.color.clone(), ao.depth.clone(), t.motion.clone(), t.bias.clone(), t.output_tex.clone()]);
        while d.keep.len() > 4 {
            d.keep.pop_front();
        }
        d.prev = Some(DlssPrev { camera: *camera, aspect: f.aspect, origin: ro, render, output: full, mode });
        // (2: DLSS hands back the sRGB-encoded values it read: the blit decodes them, or the
        // window's sRGB target encoded them a second time - a grey veil over everything; a
        // window of a plain format stores what it is given, so they go in as they are)
        let decode = if self.format.is_srgb() { 2.0 } else { 0.0 };
        self.queue.write_buffer(&d.blit_buf, 0, bytemuck::cast_slice(&[full.0 as f32, full.1 as f32, 0.0, decode]));
        // (without DLSS's result: the picture scaled up as the render scale does it)
        let bg = if ok {
            t.output_bg.clone()
        } else {
            self.queue.write_buffer(&self.upscale_buf, 0, bytemuck::cast_slice(&[render.0 as f32, render.1 as f32, 0.0, 0.0]));
            t.color_bg.clone()
        };
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dlss hud") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dlss hud"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: f.target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.upscale_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
            if !f.overlays.is_empty() {
                pass.set_pipeline(&self.overlay_pipeline_1x);
                for k in 0..f.overlays.len() {
                    if let Some((_, _, bg, _)) = scene.overlay_res.get(k) {
                        pass.set_bind_group(0, bg, &[]);
                        pass.draw(0..6, 0..1);
                    }
                }
            }
        }
        self.queue.submit(Some(encoder.finish()));
        if !ok {
            self.dlss_failed = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dlss_jitter_is_halton() {
        assert_eq!(halton(1, 2), 0.5);
        assert_eq!(halton(2, 2), 0.25);
        assert_eq!(halton(3, 2), 0.75);
        assert!((halton(1, 3) - 1.0 / 3.0).abs() < 1e-6);
        assert!((halton(2, 3) - 2.0 / 3.0).abs() < 1e-6);
        // Streamline's rows are glam's columns: a translation sits in the last row
        let rows = sl_matrix(Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(rows[3], [1.0, 2.0, 3.0, 1.0]);
    }
}
