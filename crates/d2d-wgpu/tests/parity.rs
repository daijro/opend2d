//! Phase 3B acceptance: the GPU path must produce output byte-identical to the authoritative CPU
//! rasterizer (`d2d::pixels`), which `d2d`'s own conformance suite proved byte-identical to the
//! WARP oracle — so GPU == CPU here means GPU == WARP transitively. Runs on whatever adapter wgpu
//! finds (a real GPU, or lavapipe software Vulkan); with no adapter it exercises the CPU-fallback
//! branch and passes, matching the "no-GPU runs" requirement.

use d2d::bitmap_target;
use d2d::pixels::{Aa, Blend as CpuBlend, Target};
use d2d_types::contracts::PixelBuffer;
use d2d_wgpu::{pick, Blend, Gpu, PrimBlend, MIN_GPU_PIXELS};

/// A target whose every pixel's 4 BGRA bytes span the full 0..255 range (B=x, G=y, R=255−x, and a
/// mixed alpha), so one fill exercises the per-channel kernel against thousands of (dst,src) pairs.
fn patterned(w: u32, h: u32) -> PixelBuffer {
    let mut buf = bitmap_target(w, h, 96.0);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize * 4;
            buf.data[i..i + 4].copy_from_slice(&[
                x as u8,
                y as u8,
                255u8.wrapping_sub(x as u8),
                (x.wrapping_mul(7).wrapping_add(y * 3)) as u8,
            ]);
        }
    }
    buf
}

fn first_diff(a: &PixelBuffer, b: &PixelBuffer) -> Option<(usize, [u8; 4], [u8; 4])> {
    a.data
        .chunks_exact(4)
        .zip(b.data.chunks_exact(4))
        .enumerate()
        .find(|(_, (p, q))| p != q)
        .map(|(i, (p, q))| (i, p.try_into().unwrap(), q.try_into().unwrap()))
}

const MODES: &[(CpuBlend, Blend)] =
    &[(CpuBlend::Copy, Blend::Copy), (CpuBlend::Min, Blend::Min), (CpuBlend::Max, Blend::Max), (CpuBlend::Add, Blend::Add)];
const COLORS: &[[u8; 4]] = &[[0, 0, 0, 0], [255, 255, 255, 255], [200, 50, 250, 255], [17, 240, 3, 128]];

#[test]
fn pick_routes_only_exact_ops() {
    let big = MIN_GPU_PIXELS;
    // Integer primitive-blend modes always route (when aligned + large enough).
    assert_eq!(pick(PrimBlend::Copy, false, true, big), Some(Blend::Copy));
    assert_eq!(pick(PrimBlend::Add, false, true, big), Some(Blend::Add));
    assert_eq!(pick(PrimBlend::Min, false, true, big), Some(Blend::Min));
    assert_eq!(pick(PrimBlend::Max, false, true, big), Some(Blend::Max));
    // Source-over is exact only when the source is opaque (then it is Copy); translucent ⇒ CPU.
    assert_eq!(pick(PrimBlend::SourceOver, true, true, big), Some(Blend::Copy));
    assert_eq!(pick(PrimBlend::SourceOver, false, true, big), None);
    // Fringe (non-aligned) and small targets stay on the CPU.
    assert_eq!(pick(PrimBlend::Copy, false, false, big), None);
    assert_eq!(pick(PrimBlend::Copy, false, true, big - 1), None);
}

#[test]
fn gpu_matches_cpu_byte_for_byte() {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU adapter — CPU fallback path (Gpu::new()==None) exercised; GPU parity skipped");
        return;
    };

    let (w, h) = (256u32, 256u32);
    let rect = [0.0, 0.0, w as f32, h as f32];
    let irect = [0, 0, w as i32, h as i32];

    // 1. Solid fills: every integer blend mode × a spread of source colours, over a patterned
    //    destination. This is the full-coverage path where pixels::composite reduces to integer
    //    arithmetic — the GPU blend() must reproduce it exactly.
    for &(cpu_mode, gpu_mode) in MODES {
        for &color in COLORS {
            let mut cpu = patterned(w, h);
            Target::new(&mut cpu).fill_rect(rect, color, Aa::PerPrimitive, cpu_mode);
            let mut gpu_buf = patterned(w, h);
            gpu.fill_solid(&mut gpu_buf, irect, color, gpu_mode);
            assert!(
                cpu == gpu_buf,
                "solid {gpu_mode:?}/{color:?}: first diff {:?}",
                first_diff(&cpu, &gpu_buf)
            );
        }
    }

    // 2. Opaque source-over reduces to Copy (the pick() routing claim), byte-identical to the CPU
    //    path's source-over with an opaque brush.
    for &color in &[[200, 50, 250, 255], [255, 255, 255, 255], [0, 0, 0, 255]] {
        let mut cpu = patterned(w, h);
        Target::new(&mut cpu).fill_rect(rect, color, Aa::PerPrimitive, CpuBlend::SourceOver);
        let mut gpu_buf = patterned(w, h);
        gpu.fill_solid(&mut gpu_buf, irect, color, pick(PrimBlend::SourceOver, true, true, MIN_GPU_PIXELS).unwrap());
        assert!(cpu == gpu_buf, "opaque over {color:?}: first diff {:?}", first_diff(&cpu, &gpu_buf));
    }

    // 3. Image composite / bitmap blit. Copy is an exact blit (dst := src where covered); Add/Min/Max
    //    reuse the same blend() proven in (1). Check placement/indexing at a non-zero offset against a
    //    direct CPU reference computed with the identical integer formula.
    let src = patterned(64, 64);
    for &(_, gpu_mode) in MODES {
        let mut gpu_buf = patterned(w, h);
        let off = [40i32, 24i32];
        gpu.composite_image(&mut gpu_buf, &src, off, gpu_mode);
        let mut want = patterned(w, h);
        ref_composite(&mut want, &src, off, gpu_mode);
        assert!(gpu_buf == want, "image {gpu_mode:?}: first diff {:?}", first_diff(&gpu_buf, &want));
    }

    bench(&gpu);
}

/// CPU reference for the integer image composite, using the same per-channel rule as the shader.
fn ref_composite(dst: &mut PixelBuffer, src: &PixelBuffer, off: [i32; 2], mode: Blend) {
    let (dw, dh) = (dst.desc.width as i32, dst.desc.height as i32);
    let (sw, sh) = (src.desc.width as i32, src.desc.height as i32);
    for sy in 0..sh {
        for sx in 0..sw {
            let (x, y) = (sx + off[0], sy + off[1]);
            if x < 0 || x >= dw || y < 0 || y >= dh {
                continue;
            }
            let di = (y * dw + x) as usize * 4;
            let si = (sy * sw + sx) as usize * 4;
            let sa = src.data[si + 3] as u16;
            for c in 0..4 {
                let (d, s) = (dst.data[di + c] as u16, src.data[si + c] as u16);
                dst.data[di + c] = match mode {
                    Blend::Copy => s,
                    // The source over white, then the min (d2d::pixels::blend_cov, native MIN).
                    Blend::Min => d.min((s + 255 - sa).min(255)),
                    Blend::Max => d.max(s),
                    Blend::Add => (d + s).min(255),
                } as u8;
            }
        }
    }
}

/// Report end-to-end throughput of one accelerated op vs the CPU rasterizer. On a software adapter
/// (lavapipe) the "GPU" runs on the CPU with extra buffer copies, so this measures the floor, not a
/// real GPU; the dispatcher's MIN_GPU_PIXELS threshold + CPU fallback keep small/transfer-bound ops
/// on the CPU regardless.
fn bench(gpu: &Gpu) {
    use std::time::Instant;
    let (w, h) = (2048u32, 2048u32);
    let rect = [0.0, 0.0, w as f32, h as f32];
    let irect = [0, 0, w as i32, h as i32];
    let color = [200, 50, 250, 255];
    let iters = 30;

    let mut cpu = bitmap_target(w, h, 96.0);
    let t = Instant::now();
    for _ in 0..iters {
        Target::new(&mut cpu).fill_rect(rect, color, Aa::PerPrimitive, CpuBlend::Add);
    }
    let cpu_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

    let mut g = bitmap_target(w, h, 96.0);
    gpu.fill_solid(&mut g, irect, color, Blend::Add); // warm up (pipeline/alloc)
    let t = Instant::now();
    for _ in 0..iters {
        gpu.fill_solid(&mut g, irect, color, Blend::Add);
    }
    let gpu_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    eprintln!("bench {w}x{h} Add fill: CPU {cpu_ms:.2} ms/op, GPU(end-to-end) {gpu_ms:.2} ms/op, speedup {:.2}x", cpu_ms / gpu_ms);
}
