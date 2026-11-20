//! Phase 3B — optional GPU backend. The CPU path in `d2d` is authoritative; this crate only
//! accelerates operations it can prove byte-identical to that path, and every caller keeps a CPU
//! fallback ([`Gpu::new`] returns `None` with no adapter — FreeBSD, headless, or a browser without
//! WebGPU).
//!
//! **Why integer-only.** WGSL explicitly permits floating-point reassociation, contraction (FMA)
//! and accuracy variation, so identical float WGSL on two conformant devices need not produce
//! identical bits. Integer (`u32`) arithmetic
//! has no such licence: `+`, `min`, `max`, shifts and masks are defined exactly and reproduce
//! bit-for-bit on every backend. So the accelerated set is exactly the premultiplied-BGRA8
//! operations that reduce to integer arithmetic matching `d2d::pixels`:
//!
//! * **Clear / solid fill** of a pixel-aligned (full-coverage) region, and **image composite /
//!   bitmap blit** at an integer offset, under the integer primitive-blend modes
//!   [`Blend::Copy`]/`Min`/`Max`/`Add`. At full coverage `d2d::pixels::composite` reduces to:
//!   Copy → `src`; Min → per-channel `min(dst, src + 255 − src.a)` (the source over white, as
//!   native's MIN); Max → per-channel `max`; Add → per-channel saturating add
//!   (`min(a+b,255)`). `SourceOver` with an **opaque** source is `src`, i.e. `Copy`
//!   (see [`pick`]); translucent `SourceOver`, anti-aliased fringe pixels (partial coverage),
//!   gradient-LUT sampling and bilinear bitmap filtering stay on the CPU — their arithmetic is
//!   floating-point and cannot be proven identical here.
//!
//! Not built: a DXBC→WGSL path for arbitrary `d2d-shader` programs. Those shaders are floating
//! point, so a WGSL translation cannot claim parity; phase 3C's CPU executor remains the authority.
//!
//! The dispatcher is [`pick`] (routing policy) plus the fact that the only methods here implement
//! proven-exact kernels. Verification: `tests/parity.rs` runs the same inputs through
//! `d2d::pixels::Target` and this backend and asserts byte-identical output.

// Only the native (synchronous) composite methods touch pixel buffers; the browser host drives
// WebGPU readback asynchronously itself, so this is native-only.
#[cfg(not(target_arch = "wasm32"))]
use d2d_types::contracts::PixelBuffer;

/// Integer-exact blend mode (`D2D1_PRIMITIVE_BLEND` minus the floating `SourceOver`). Values are
/// the `mode` the shader switches on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Blend {
    /// Replace the destination with the source (`D2D1_PRIMITIVE_BLEND_COPY`, and opaque over).
    Copy = 0,
    /// Per-channel `min(dst, src + 255 − src.a)` (`D2D1_PRIMITIVE_BLEND_MIN`: the source over white).
    Min = 1,
    /// Per-channel `max` (`D2D1_PRIMITIVE_BLEND_MAX`).
    Max = 2,
    /// Per-channel saturating add (`D2D1_PRIMITIVE_BLEND_ADD`).
    Add = 3,
}

/// `D2D1_PRIMITIVE_BLEND`, for the [`pick`] routing decision.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PrimBlend {
    SourceOver,
    Copy,
    Min,
    Add,
    Max,
}

/// Below this many pixels the host↔device transfer costs more than the kernel saves, so the CPU
/// wins; the dispatcher keeps small targets on the CPU. Measured on lavapipe (`tests/parity.rs`).
pub const MIN_GPU_PIXELS: usize = 1 << 18; // 256k px (~512×512)

/// Routing policy for a solid primitive fill: return the GPU [`Blend`] only when the operation is
/// provably integer-exact and large enough to be worth offloading, else `None` (⇒ CPU). `aligned`
/// means the fill covers only whole pixels (no anti-aliased fringe); `src_opaque` means the fill
/// colour has alpha 255.
pub fn pick(mode: PrimBlend, src_opaque: bool, aligned: bool, pixels: usize) -> Option<Blend> {
    if !aligned || pixels < MIN_GPU_PIXELS {
        return None;
    }
    Some(match mode {
        PrimBlend::Copy => Blend::Copy,
        PrimBlend::Min => Blend::Min,
        PrimBlend::Max => Blend::Max,
        PrimBlend::Add => Blend::Add,
        PrimBlend::SourceOver if src_opaque => Blend::Copy,
        PrimBlend::SourceOver => return None, // translucent over is floating point ⇒ CPU
    })
}

/// Shader uniform. `#[repr(C)]` of 13 `u32` + padding to a 16-byte-aligned 64-byte block; field
/// order matches the WGSL `Params` struct exactly.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    tw: u32,
    th: u32,
    rx0: u32,
    ry0: u32,
    rx1: u32,
    ry1: u32,
    mode: u32,
    is_image: u32,
    color: u32,
    ox: u32,
    oy: u32,
    sw: u32,
    sh: u32,
    _pad: [u32; 3],
}

const SHADER: &str = r#"
struct Params {
    tw: u32, th: u32,
    rx0: u32, ry0: u32, rx1: u32, ry1: u32,
    mode: u32, is_image: u32, color: u32,
    ox: u32, oy: u32, sw: u32, sh: u32,
};
@group(0) @binding(0) var<storage, read_write> tgt: array<u32>;
@group(0) @binding(1) var<storage, read> srcbuf: array<u32>;
@group(0) @binding(2) var<uniform> p: Params;

fn blend(dst: u32, src: u32, mode: u32) -> u32 {
    if (mode == 0u) { return src; }          // Copy (and opaque source-over)
    var o: u32 = 0u;
    let sa = (src >> 24u) & 0xffu;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let sh = i * 8u;
        let d = (dst >> sh) & 0xffu;
        let s = (src >> sh) & 0xffu;
        var r: u32;
        switch mode {
            case 1u: { r = min(d, min(s + 255u - sa, 255u)); } // Min: the source over white, then min
            case 2u: { r = max(d, s); }       // Max
            default: { r = min(d + s, 255u); } // Add
        }
        o = o | (r << sh);
    }
    return o;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.tw || y >= p.th) { return; }
    if (x < p.rx0 || x >= p.rx1 || y < p.ry0 || y >= p.ry1) { return; }
    var src: u32;
    if (p.is_image == 1u) {
        if (x < p.ox || x >= p.ox + p.sw || y < p.oy || y >= p.oy + p.sh) { return; }
        src = srcbuf[(y - p.oy) * p.sw + (x - p.ox)];
    } else {
        src = p.color;
    }
    let idx = y * p.tw + x;
    tgt[idx] = blend(tgt[idx], src, p.mode);
}
"#;

/// A GPU device plus the single integer compute pipeline. Cheap to keep for a process; create once.
/// (On wasm only `new_async` exists — the fields are read by the native composite methods.)
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

#[cfg(not(target_arch = "wasm32"))]
fn pack(px: [u8; 4]) -> u32 {
    px[0] as u32 | (px[1] as u32) << 8 | (px[2] as u32) << 16 | (px[3] as u32) << 24
}

impl Gpu {
    /// Async device init — the entry point a browser/WebGPU host awaits (WebGPU has no blocking
    /// poll). Native callers use [`Gpu::new`], which blocks on this.
    pub async fn new_async() -> Option<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .ok()?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("d2d-wgpu"),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .ok()?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("d2d-wgpu blend"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let storage = |ro: bool| wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: ro },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("d2d-wgpu bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, ..storage(false) },
                wgpu::BindGroupLayoutEntry { binding: 1, ..storage(true) },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("d2d-wgpu pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("d2d-wgpu pipeline"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Some(Gpu { device, queue, pipeline, layout })
    }

    /// Create the backend, or `None` if no GPU adapter is available (caller falls back to CPU).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Option<Gpu> {
        pollster::block_on(Self::new_async())
    }

    /// Fill the pixel-aligned device-space rectangle `rect` (`[x0,y0,x1,y1]`, integer pixels) of
    /// `buf` with premultiplied BGRA `color` under `blend`. Equivalent to `d2d::pixels::Target`'s
    /// full-coverage fill of the same rectangle. Only route here via [`pick`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn fill_solid(&self, buf: &mut PixelBuffer, rect: [i32; 4], color: [u8; 4], blend: Blend) {
        self.run(buf, None, rect, pack(color), blend, [0, 0]);
    }

    /// Composite premultiplied-BGRA `src` onto `buf` with its top-left at integer `offset`, under
    /// `blend` (`Copy` is a plain blit). Equivalent to the CPU path's integer-mode image composite.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn composite_image(&self, buf: &mut PixelBuffer, src: &PixelBuffer, offset: [i32; 2], blend: Blend) {
        assert_eq!(src.stride, src.desc.width as usize * 4, "source must be tightly packed BGRA8");
        let rect = [0, 0, buf.desc.width as i32, buf.desc.height as i32];
        self.run(buf, Some(src), rect, 0, blend, offset);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn run(&self, buf: &mut PixelBuffer, src: Option<&PixelBuffer>, rect: [i32; 4], color: u32, blend: Blend, offset: [i32; 2]) {
        use wgpu::util::DeviceExt;
        let (w, h) = (buf.desc.width, buf.desc.height);
        assert_eq!(buf.stride, w as usize * 4, "target must be tightly packed BGRA8");
        let clamp = |v: i32, hi: u32| v.clamp(0, hi as i32) as u32;
        let params = Params {
            tw: w,
            th: h,
            rx0: clamp(rect[0], w),
            ry0: clamp(rect[1], h),
            rx1: clamp(rect[2], w),
            ry1: clamp(rect[3], h),
            mode: blend as u32,
            is_image: src.is_some() as u32,
            color,
            ox: offset[0].max(0) as u32,
            oy: offset[1].max(0) as u32,
            sw: src.map_or(0, |s| s.desc.width),
            sh: src.map_or(0, |s| s.desc.height),
            _pad: [0; 3],
        };
        let dev = &self.device;
        let target = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("target"),
            contents: &buf.data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let source = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("source"),
            contents: src.map_or(&[0u8; 4][..], |s| &s.data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let ubo = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let readback = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: buf.data.len() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: target.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: source.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ubo.as_entire_binding() },
            ],
        });
        let mut enc = dev.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
        enc.copy_buffer_to_buffer(&target, 0, &readback, 0, buf.data.len() as u64);
        self.queue.submit([enc.finish()]);

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        dev.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        rx.recv().unwrap().unwrap();
        buf.data.copy_from_slice(&slice.get_mapped_range().unwrap());
        readback.unmap();
    }
}
