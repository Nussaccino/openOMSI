# Third-party crates carried with a change

## wgpu-hal 29.0.4

The crates.io release with one change in `src/dx12/instance.rs`: the DX12 instance does
not create an `IDXGIFactoryMedia`. wgpu only uses it for surfaces made from
DirectComposition handles, which openOMSI never makes, and ReShade 6.8 (as `dxgi.dll` or
`d3d12.dll`) takes the object `CreateDXGIFactory1` hands out for it for an ordinary DXGI
factory: openOMSI went down in `dxgi.dll` before its window opened, other games (which never
ask for that factory) ran.

Drop this copy and the `[patch.crates-io]` entry in the workspace's `Cargo.toml` once
wgpu or ReShade no longer need it.
