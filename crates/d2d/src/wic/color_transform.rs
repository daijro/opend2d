//! `IWICColorTransform` for app-supplied ICC profiles: the legacy "Win " CMM — the ICC colour-
//! management engine the system ICM layer dispatches to. For a matrix/shaper src→dst
//! pair it builds a device-link `Combi` — a 32³ cube resampled by two integer matrix passes, with the
//! TRCs in per-pixel ELUT/ALUT — and interpolates it tetrahedrally per pixel (8-bit [`Combi::apply`]
//! and 16-bit [`Combi::apply16`] kernels). Byte-exact vs the suite's `fixtures/wic_colortransform/`
//! oracle for {ss,sw,ws} × {BGRA8,RGB8,RGBA16}.
//! The `crates/wic-ffi` `ColorTransformObj` wires this to `CreateColorTransformer`/`Initialize`.
pub mod icm {
    /// A parsed matrix/TRC ICC profile (the subset the CMM needs): colorant matrix columns, white
    /// point, and a per-channel tone curve.
    pub struct IccProfile {
        /// Colorant matrix columns rXYZ/gXYZ/bXYZ (s15Fixed16 → f64), device-linear → PCS(XYZ, D50).
        pub mtx: [[f64; 3]; 3],
        /// Per-channel TRC.
        pub trc: [Curve; 3],
        pub data_space: u32,
        pub pcs: u32,
        /// ICC header version ≥ 4.0: a V4 profile's CLUT PCS is referenced to the perceptual-intent
        /// reference-medium black point, which the CMM strips when it builds the device-link.
        pub v4: bool,
        /// A `lut8Type`/`lut16Type` ('mft1'/'mft2') or `lutAToBType` ('mAB ') A2B CLUT, when the
        /// profile carries an A2B tag (the CMM uses it over the matrix/TRC path). 3→3 only.
        pub lut: Option<Lut>,
    }

    /// V4 perceptual→legacy black-point adjustment: strip the ICC V4 perceptual
    /// reference-medium black point from the resampled PCS cube (f32, per channel, clamp+truncate).
    /// Applied to a V4 CLUT profile's cube before the destination matrix; lut16/lut8
    /// (V2) profiles skip it. BP = (0.00336, 0.0034731, 0.00287)·32768, scale = 1 − BP/mediaWhite.
    fn adjust_black_v4_to_legacy(cube: &mut [u16], nodes: usize) {
        // f32 bit-exact (f32 constants): `min(65535, max(0, f32(v) − BP) / SC)`, trunc→u16.
        const BP: [u32; 3] = [0x42dc_3372, 0x42e3_9cf3, 0x42bc_169c];
        const SC: [u32; 3] = [0x3f7f_1ba0, 0x3f7f_1c63, 0x3f7f_1bfd];
        for n in 0..nodes {
            for c in 0..3 {
                let x = (cube[n * 3 + c] as f32 - f32::from_bits(BP[c])).max(0.0) / f32::from_bits(SC[c]);
                cube[n * 3 + c] = 65535.0f32.min(x) as u16;
            }
        }
    }

    /// `phConnectionSpace == 'Lab '` — the source profile connects through CIELAB, not XYZ.
    const PCS_LAB: u32 = 0x4C61_6220;

    /// CIE Lab f⁻¹ companding table (the engine's static companding table's role), generated from the
    /// standard formula. Reproduces the engine's offline build in **f32**:
    /// sampled at `t = i·256` (`L* = t·100/65536`), `fy = (L* + 16)·(1/116)`, `f⁻¹(fy) = fy³` above the
    /// 6/29 knee else the linear `3·(6/29)²·(fy − 4/29)`, then `round(f⁻¹·16383)` clamped to u16 (the
    /// `·2` in [`lab2xyz_cube16`] and the half-scale PCS give the final `·0.5·65535`). f32 throughout ⇒
    /// bit-identical on every target (no browser fingerprint from the companding). Matches the native
    /// static table EXCEPT 4/432 entries that are ≤1 LSB off — a sub-ULP difference in the engine's
    /// offline float chain not yet reproduced (the companding exact-table wall).
    static LAB_FINV: std::sync::LazyLock<[u16; 432]> = std::sync::LazyLock::new(|| {
        const D: f32 = 6.0 / 29.0;
        const INV116: f32 = 1.0 / 116.0;
        const LSTEP: f32 = 25600.0 / 65536.0; // i·256·100/65536
        std::array::from_fn(|i| {
            let fy = (i as f32 * LSTEP + 16.0) * INV116;
            let finv = if fy > D { (fy * fy) * fy } else { 3.0 * D * D * (fy - 4.0 / 29.0) };
            (finv * 16383.0 + 0.5).floor().clamp(0.0, 65535.0) as u16
        })
    });

    /// Lab→XYZ cube conversion: a Lab-PCS source profile's CLUT emits 16-bit CIELAB,
    /// so the "Win " CMM converts every cube node to XYZ before the destination colorant matrix (which
    /// is XYZ-PCS). Integer-exact: each channel folds into an [`LAB_FINV`] lookup (the a*/b* channels
    /// add a step-offset to L first), then scales by the D50 white point — X·31595/32768, Z·13515/16384,
    /// Y·1. The L-only Y index is always ≤ 255 (in range); X/Z clamp on their wider domain.
    fn lab2xyz_cube16(cube: &mut [u16], nodes: usize) {
        let finv = |t: i32| -> i32 {
            if t < 0 {
                return 0;
            }
            let i = (t >> 8) as usize;
            if i >= 431 {
                return 131070;
            }
            let v = LAB_FINV[i] as i32;
            (((t & 0xFF) * (LAB_FINV[i + 1] as i32 - v) + 63) >> 7) + 2 * v
        };
        for n in 0..nodes {
            let (l, a, b) = (cube[n * 3] as i32, cube[n * 3 + 1] as i32, cube[n * 3 + 2] as i32);
            let x = finv(((19 * a + 15) >> 5) + l - 19456);
            let z = finv(l - ((95 * b + 31) >> 6) + 48640);
            let yi = (l >> 8) as usize; // L is u16 ⇒ yi ≤ 255, always in range
            let yv = LAB_FINV[yi] as i32;
            let y = (((l & 0xFF) * (LAB_FINV[yi + 1] as i32 - yv) + 63) >> 7) + 2 * yv;
            cube[n * 3] = ((31595 * x + 0x4000) >> 15) as u16;
            cube[n * 3 + 1] = y as u16;
            cube[n * 3 + 2] = ((13515 * z + 0x2000) >> 14).min(0xFFFF) as u16;
        }
    }

    /// A parsed lut8/lut16 A2B, prebuilt into the combi's CLUT inputs: grid points, the device→PCS
    /// CLUT (gp³ nodes × 3, promoted to u16), its bit width (8 for 'mft1', 16 for 'mft2'), the
    /// per-pixel input ELUT (input tables resampled then grid-descaled), and the output ALUT.
    pub struct Lut {
        pub gp: usize,
        pub clut: Vec<u16>,
        pub clut_bits: u32,
        pub pp_elut: [[u16; E]; 3],
        pub out_alut: [[u16; N]; 3],
        /// `lutAToBType` M-matrix (colorant·0.5), applied to the CLUT output before the dst matrix;
        /// `None` for lut8/lut16 and for mAB profiles without a matrix element.
        pub mab_matrix: Option<[[f64; 3]; 3]>,
    }

    #[derive(Clone)]
    pub enum Curve {
        /// 'curv' count 0 — identity (linear).
        Identity,
        /// 'curv' count 1 — pure gamma; exponent is u8Fixed8 (value/256).
        Gamma(f64),
        /// 'curv' count>1 — sampled table of u16.
        Table(Vec<u16>),
        /// 'para' parametric curve: ICC function type 0–4, params `[g,a,b,c,d,e,f]` (s15Fixed16).
        Para { ty: u8, p: [f64; 7] },
    }

    /// `para` forward device→linear (the ICC parameterized curve, types Gamma/1..4): the ICC piecewise
    /// power form for `ty`, clamped to [0,1] (negative → 0).
    fn para_fwd(ty: u8, p: &[f64; 7], x: f64) -> f64 {
        let [g, a, b, c, d, e, f] = *p;
        let y = match ty {
            0 => crate::crt::pow(x, g),
            1 => if x < -b / a { 0.0 } else { crate::crt::pow(a * x + b, g) },
            2 => if x < -b / a { c } else { crate::crt::pow(a * x + b, g) + c },
            3 => if x < d { c * x } else { crate::crt::pow(a * x + b, g) },
            4 => if x < d { c * x + f } else { crate::crt::pow(a * x + b, g) + e },
            _ => x,
        };
        if y >= 0.0 { y.min(1.0) } else { 0.0 }
    }

    /// `para` inverse linear→device (inverse of the ICC parameterized curve): only the power segment is
    /// inverted (`(X^(1/g)−b)/a`, with the `−c`/`−e` offset for types 2/4), clamped to [0,1].
    fn para_inv(ty: u8, p: &[f64; 7], y: f64) -> f64 {
        let g = p[0];
        if ty == 0 {
            let v = crate::crt::pow(y, 1.0 / g);
            return if v >= 0.0 { v.min(1.0) } else { 0.0 };
        }
        let base = match ty {
            2 => y - p[3], // −c
            4 => y - p[5], // −e
            _ => y,        // types 1, 3
        };
        let v = (crate::crt::pow(base, 1.0 / g) - p[2]) / p[1];
        if v >= 0.0 { v.min(1.0) } else { 0.0 }
    }

    impl Curve {
        /// Forward device→linear, input and output in [0,1].
        pub fn fwd(&self, x: f64) -> f64 {
            match self {
                Curve::Identity => x,
                Curve::Para { ty, p } => para_fwd(*ty, p, x),
                Curve::Gamma(g) => crate::crt::pow(x, *g),
                Curve::Table(t) => {
                    let n = t.len();
                    let p = x.clamp(0.0, 1.0) * (n - 1) as f64;
                    let i = p.floor() as usize;
                    let f = p - i as f64;
                    let a = t[i] as f64;
                    let b = t[(i + 1).min(n - 1)] as f64;
                    (a * (1.0 - f) + b * f) / 65535.0
                }
            }
        }
        /// Inverse linear→device, input and output in [0,1].
        pub fn inv(&self, y: f64) -> f64 {
            match self {
                Curve::Identity => y,
                Curve::Para { ty, p } => para_inv(*ty, p, y),
                Curve::Gamma(g) => crate::crt::pow(y.clamp(0.0, 1.0), 1.0 / g),
                Curve::Table(_) => {
                    // numeric inverse of the forward table
                    let y = y.clamp(0.0, 1.0);
                    let mut lo = 0.0f64;
                    let mut hi = 1.0f64;
                    for _ in 0..40 {
                        let m = 0.5 * (lo + hi);
                        if self.fwd(m) < y {
                            lo = m;
                        } else {
                            hi = m;
                        }
                    }
                    0.5 * (lo + hi)
                }
            }
        }
    }

    fn be32(b: &[u8], o: usize) -> u32 {
        u32::from_be_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
    }
    fn s15f16(b: &[u8], o: usize) -> f64 {
        (be32(b, o) as i32) as f64 / 65536.0
    }

    /// Find a top-level tag's byte slice by signature.
    fn tag<'a>(p: &'a [u8], sig: &[u8; 4]) -> Option<&'a [u8]> {
        let n = be32(p, 128) as usize;
        for i in 0..n {
            let e = 132 + i * 12;
            if &p[e..e + 4] == sig {
                let off = be32(p, e + 4) as usize;
                let len = be32(p, e + 8) as usize;
                return p.get(off..off + len);
            }
        }
        None
    }

    fn xyz_tag(p: &[u8], sig: &[u8; 4]) -> Option<[f64; 3]> {
        let t = tag(p, sig)?;
        Some([s15f16(t, 8), s15f16(t, 12), s15f16(t, 16)])
    }

    fn be16(b: &[u8], o: usize) -> u16 {
        u16::from_be_bytes([b[o], b[o + 1]])
    }

    /// Parse the first present A2B tag as a lut8/lut16 ('mft1'/'mft2') CLUT, 3→3 (other channel
    /// counts, mAB/mBA not yet reproduced). Prebuilds the per-pixel input ELUT (input tables → the
    /// 256-entry ELUT via the matching resampler, grid-descaled) and the output ALUT.
    fn a2b_lut(p: &[u8]) -> Option<Lut> {
        let t = [b"A2B0", b"A2B1", b"A2B2"].iter().find_map(|s| tag(p, s))?;
        if &t[0..4] == b"mAB " {
            return a2b_mab(t);
        }
        if t[8] != 3 || t[9] != 3 {
            return None;
        }
        let (gp, nc) = (t[10] as usize, (t[10] as usize).pow(3) * 3);
        match &t[0..4] {
            b"mft2" => {
                let (ie, oe) = (be16(t, 48) as usize, be16(t, 50) as usize);
                let (clut_off, out_off) = (52 + 3 * ie * 2, 52 + 3 * ie * 2 + nc * 2);
                let clut = (0..nc).map(|i| be16(t, clut_off + i * 2)).collect();
                let pp_elut = std::array::from_fn(|c| {
                    let it: Vec<u16> = (0..ie).map(|e| be16(t, 52 + (c * ie + e) * 2)).collect();
                    let mut e = fill_ushort_elut_lut16(&it, ie);
                    descale_elut(&mut e);
                    e
                });
                let out_alut = std::array::from_fn(|c| clut_output_alut(&(0..oe).map(|e| be16(t, out_off + (c * oe + e) * 2)).collect::<Vec<_>>(), oe));
                Some(Lut { gp, clut, clut_bits: 16, pp_elut, out_alut, mab_matrix: None })
            }
            b"mft1" => {
                let (clut_off, out_off) = (48 + 3 * 256, 48 + 3 * 256 + nc);
                let clut = (0..nc).map(|i| t[clut_off + i] as u16).collect();
                let pp_elut = std::array::from_fn(|c| {
                    let it: [u8; 256] = std::array::from_fn(|e| t[48 + c * 256 + e]);
                    let mut e = fill_ushort_elut_lut8(&it);
                    descale_elut(&mut e);
                    e
                });
                let out_alut = std::array::from_fn(|c| clut_output_alut_lut8(&std::array::from_fn(|e| t[out_off + c * 256 + e])));
                Some(Lut { gp, clut, clut_bits: 8, pp_elut, out_alut, mab_matrix: None })
            }
            _ => None,
        }
    }

    /// Parse a `curv`/`para` curve element at `d[0..]`; returns the curve and its 4-byte-padded length
    /// (for walking the consecutive A-/B-curve sections of an `mAB ` tag).
    fn curve_at(d: &[u8]) -> (Curve, usize) {
        let raw = if &d[0..4] == b"para" {
            let ty = u16::from_be_bytes([d[8], d[9]]) as u8;
            let n = match ty { 0 => 1, 1 => 3, 2 => 4, 3 => 5, 4 => 7, _ => 1 };
            let mut pr = [0.0f64; 7];
            for (i, v) in pr.iter_mut().take(n).enumerate() {
                *v = s15f16(d, 12 + i * 4);
            }
            (Curve::Para { ty, p: pr }, 12 + n * 4)
        } else {
            let n = be32(d, 8) as usize;
            let c = match n {
                0 => Curve::Identity,
                1 => Curve::Gamma(u16::from_be_bytes([d[12], d[13]]) as f64 / 256.0),
                _ => Curve::Table((0..n).map(|i| u16::from_be_bytes([d[12 + i * 2], d[13 + i * 2]])).collect()),
            };
            (c, 12 + n * 2)
        };
        (raw.0, (raw.1 + 3) & !3)
    }

    /// Parse a `lutAToBType` ('mAB ', ICC v4) A2B: A-curves → the per-pixel input ELUT (device→CLUT
    /// position, descaled), the CLUT (device→PCS), B-curves → the CLUT output ALUT, and the optional
    /// M-matrix (colorant·0.5). device → A → CLUT → [M-curves →] matrix → B → PCS; 3→3. The CMM then
    /// strips the V4 perceptual black point ([`adjust_black_v4_to_legacy`]) in `Combi::new`.
    fn a2b_mab(t: &[u8]) -> Option<Lut> {
        if t[8] != 3 || t[9] != 3 {
            return None;
        }
        let off = |o| be32(t, o) as usize;
        let (off_b, off_mat, off_clut, off_a) = (off(12), off(16), off(24), off(28));
        // CLUT: 16 grid-dim bytes, then a precision byte (1 or 2), then (pad to) the CLUT data.
        let gp = t[off_clut] as usize;
        let prec = t[off_clut + 16] as usize;
        let nc = gp.pow(3) * 3;
        let data = off_clut + 20;
        let (clut, clut_bits): (Vec<u16>, u32) = if prec == 2 {
            ((0..nc).map(|i| be16(t, data + i * 2)).collect(), 16)
        } else {
            ((0..nc).map(|i| t[data + i] as u16).collect(), 8)
        };
        // A-curves → per-pixel input ELUT (shaper fill, grid-descaled), consecutive & 4-byte-padded.
        let mut ao = off_a;
        let pp_elut = std::array::from_fn(|_| {
            let (c, len) = curve_at(&t[ao..]);
            ao += len;
            let mut e = shaper(&c);
            descale_elut(&mut e);
            e
        });
        // B-curves → CLUT output ALUT (identity 'curv' → a linear 0..0xFFFF table, as lut16's).
        let mut bo = off_b;
        let out_alut = std::array::from_fn(|_| {
            let (c, len) = curve_at(&t[bo..]);
            bo += len;
            match c {
                Curve::Identity => clut_output_alut(&[0, 0xFFFF], 2),
                other => clut_output_alut(&(0..256).map(|i| (other.fwd(i as f64 / 255.0) * 65535.0).round() as u16).collect::<Vec<_>>(), 256),
            }
        });
        // Optional M-matrix: 9 s15Fixed16 (colorant·0.5) + 3 offsets (ignored, always 0 for our v4 set).
        let mab_matrix = (off_mat != 0).then(|| {
            std::array::from_fn(|i| std::array::from_fn(|j| s15f16(t, off_mat + (i * 3 + j) * 4)))
        });
        Some(Lut { gp, clut, clut_bits, pp_elut, out_alut, mab_matrix })
    }

    fn curve_tag(p: &[u8], sig: &[u8; 4]) -> Curve {
        tag(p, sig).map_or(Curve::Identity, |t| curve_at(t).0)
    }

    impl IccProfile {
        pub fn parse(p: &[u8]) -> Option<IccProfile> {
            if p.len() < 132 || &p[36..40] != b"acsp" {
                return None;
            }
            // A CLUT/A2B profile (e.g. a Lab-PCS 'scnr' lut16) carries no rXYZ/gXYZ/bXYZ matrix — it is
            // unused when `lut` is Some, so default it. A matrix/shaper profile always has these.
            let lut = a2b_lut(p);
            if lut.is_none() && xyz_tag(p, b"rXYZ").is_none() {
                return None; // matrix profile with no matrix and no CLUT → not usable
            }
            let r = xyz_tag(p, b"rXYZ").unwrap_or([0.0; 3]);
            let g = xyz_tag(p, b"gXYZ").unwrap_or([0.0; 3]);
            let b = xyz_tag(p, b"bXYZ").unwrap_or([0.0; 3]);
            Some(IccProfile {
                mtx: [[r[0], g[0], b[0]], [r[1], g[1], b[1]], [r[2], g[2], b[2]]],
                trc: [curve_tag(p, b"rTRC"), curve_tag(p, b"gTRC"), curve_tag(p, b"bTRC")],
                data_space: be32(p, 16),
                pcs: be32(p, 20),
                v4: be32(p, 8) >= 0x0400_0000,
                lut,
            })
        }
    }

    /// Exact port of the 3×3 matrix inverse: adjugate × (1/det) with reciprocal-multiply and the
    /// native product/accumulation order (matching the last f64 ULP, which decides a rounded coeff).
    fn mat_inv(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let a = [m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]];
        let p04 = a[0] * a[4];
        let p15 = a[1] * a[5];
        let p24 = a[2] * a[4];
        let p50 = a[5] * a[0];
        let p23 = a[2] * a[3];
        let p31 = a[3] * a[1];
        let det = a[8] * p04 + a[6] * p15 + a[7] * p23 - a[6] * p24 - a[8] * p31 - a[7] * p50;
        let r = 1.0 / det;
        [
            [(a[8] * a[4] - a[7] * a[5]) * r, (a[7] * a[2] - a[8] * a[1]) * r, (p15 - p24) * r],
            [(a[5] * a[6] - a[3] * a[8]) * r, (a[8] * a[0] - a[6] * a[2]) * r, (p23 - p50) * r],
            [(a[7] * a[3] - a[6] * a[4]) * r, (a[6] * a[1] - a[7] * a[0]) * r, (p04 - p31) * r],
        ]
    }

    /// A built combi device-link: per-channel input ELUT (device byte → grid word cell<<11|frac),
    /// the 32³ cube (3×u16/node, dst-linear·65535), per-channel output ALUT (1024×u16, dst device).
    pub struct Combi {
        elut: [[u16; 256]; 3],
        cube: Vec<u16>, // 32*32*32*3, idx = (R*1024 + G*32 + B)*3 + ch
        alut: [[u16; 1024]; 3],
    }

    const GP: usize = 32; // grid points per axis (2^5)

    /// Bit-replicate a `bits`-wide grid index to a full 16-bit value (x|x>>b|…). For
    /// bits=5 this maps 0→0 and 31→65535.
    fn expand16(k: u32, bits: u32) -> u32 {
        let mut x = k << (16 - bits);
        let mut v = x;
        let mut sh = bits;
        while sh < 16 {
            x = v >> sh;
            v |= x;
            sh *= 2;
        }
        v & 0xFFFF
    }

    const E: usize = 256; // input ELUT / shaper entries
    const N: usize = 1024; // output ALUT entries

    /// Linear input ELUT: `E`-entry linear ramp, device byte → 16-bit (`(0xFFFF·i+(E-1)/2−1)/(E−1)`).
    fn linear_elut16() -> [u16; E] {
        let mut e = [0u16; E];
        for (i, s) in e.iter_mut().enumerate() {
            *s = ((((E as u32 - 1) >> 1) + 0xFFFF * i as u32).wrapping_sub(1) / (E as u32 - 1)).min(0xFFFF) as u16;
        }
        e
    }

    /// Rescale a 16-bit ELUT to the cube grid so cell∈[0,GP-1], frac∈[0,2047]
    /// (byte 255 → cell GP-1, frac 0 → no halo needed). `scale·in/(scale−1)·(GP−1)/GP`, `scale = 1<<ELUTbits`.
    fn descale_elut(e: &mut [u16; E]) {
        descale_elut_gp(e, GP)
    }
    /// `FillDescaledInputLut` to an arbitrary combi grid `gp` (32 for the 3→3 cube, 16 for the 4→3
    /// CMYK cube): a device byte maps to a grid coord in the top `log2(gp)` bits (top → frac 0).
    fn descale_elut_gp(e: &mut [u16; E], gp: usize) {
        for v in e.iter_mut() {
            *v = ((0x10000u32 * *v as u32 / 0xFFFF) * (gp as u32 - 1) / gp as u32).min(0xFFFF) as u16;
        }
    }

    /// `CreateLinearAlut16`: `N`-entry linear ramp, grid → 16-bit (`(2047+262396·k)>>12`).
    fn linear_alut16() -> [u16; N] {
        let mut a = [0u16; N];
        for (k, s) in a.iter_mut().enumerate() {
            *s = (((2047u32 + 262396 * k as u32) >> 12)).min(0xFFFF) as u16;
        }
        a
    }

    /// Input shaper S (the per-channel input ELUT from the source TRC, 16-bit):
    /// `E`-entry gamma curve sampled at stride 4 with linear-interpolated in-betweens; identity/table
    /// fall back to the exact-ish float sample. Only `Gamma` is byte-exact (the oracle's curve type).
    fn shaper(curve: &Curve) -> [u16; E] {
        match curve {
            Curve::Identity => linear_elut16(),
            Curve::Gamma(g) => fill_param_elut16(|x| crate::crt::pow(x, *g)),
            Curve::Para { ty, p } => fill_param_elut16(|x| para_fwd(*ty, p, x)),
            Curve::Table(_) => {
                let mut s = [0u16; E];
                for (i, v) in s.iter_mut().enumerate() {
                    *v = (curve.fwd(i as f64 / (E - 1) as f64) * 65535.0).round() as u16;
                }
                s
            }
        }
    }

    /// Build the input ELUT from a parameterized curve (`E`=256, 16-bit): truncated samples at
    /// stride `1<<(8-6)`=4 (plus the endpoint), linearly interpolated between, and — when the curve
    /// is steep at 0 (gamma<1) — the first stride recomputed directly.
    fn fill_param_elut16(f: impl Fn(f64) -> f64) -> [u16; E] {
        let stride = 4usize;
        let samp = |k: usize| (f(k as f64 / (E - 1) as f64).clamp(0.0, 1.0) * 65535.0) as i64;
        let mut s = [0i64; E];
        let mut k = 0;
        while k < E - 1 {
            s[k] = samp(k);
            k += stride;
        }
        s[E - 1] = samp(E - 1);
        let mut base = 0;
        while base < E - stride {
            let d = s[base + stride] - s[base];
            for j in 1..stride {
                s[base + j] = s[base] + ((((j as i64 * d) << 8) / stride as i64 + 128) >> 8);
            }
            base += stride;
        }
        let d = s[E - 1] - s[E - stride]; // last cell spans to the endpoint: divisor stride-1
        for j in 1..stride - 1 {
            s[E - stride + j] = s[E - stride] + ((((j as i64 * d) << 8) / (stride as i64 - 1) + 128) >> 8);
        }
        let mut out = [0u16; E];
        for (o, v) in out.iter_mut().zip(s) {
            *o = v as u16;
        }
        out
    }

    /// Inverse ALUT A (the per-channel output ALUT from the dst inverse TRC,
    /// 16-bit): `N`-entry inverse-gamma sampled at stride 16, interpolated, first stride recomputed
    /// when 1/gamma<1. Only `Gamma` is byte-exact.
    fn inv_alut(curve: &Curve) -> [u16; N] {
        match curve {
            Curve::Identity => linear_alut16(),
            Curve::Gamma(g) => fill_param_inv_alut16(|x| crate::crt::pow(x, 1.0 / *g), *g > 1.0),
            Curve::Para { ty, p } => fill_param_inv_alut16(|x| para_inv(*ty, p, x), p[0] > 1.0),
            Curve::Table(_) => {
                let mut a = [0u16; N];
                for (k, v) in a.iter_mut().enumerate() {
                    *v = (curve.inv(k as f64 / (N - 1) as f64) * 65535.0).round() as u16;
                }
                a
            }
        }
    }

    /// Build the output ALUT from the inverse of a parameterized curve: rounded samples of the inverse curve
    /// `f` at stride 16 (plus the endpoint), linear-interpolated between (tail cell divisor 15); when
    /// `steep` (native `1/params[0] < 1`, i.e. g>1) the first stride is recomputed directly (no +0.5).
    fn fill_param_inv_alut16(f: impl Fn(f64) -> f64, steep: bool) -> [u16; N] {
        let stride = 16usize;
        let samp = |k: usize| f(k as f64 / (N - 1) as f64).clamp(0.0, 1.0);
        let mut a = [0i64; N];
        let mut k = 0;
        while k < N - 1 {
            a[k] = (samp(k) * 65535.0 + 0.5) as i64;
            k += stride;
        }
        a[N - 1] = (samp(N - 1) * 65535.0 + 0.5) as i64;
        let mut base = 0;
        while base + stride < N {
            let d = a[base + stride] - a[base];
            for j in 1..stride {
                a[base + j] = a[base] + ((16 * (j as i64 * d + 8)) >> 8);
            }
            base += stride;
        }
        let d = a[N - 1] - a[N - stride];
        for j in 1..stride - 1 {
            a[N - stride + j] = a[N - stride] + ((((j as i64 * d) << 8) / (stride as i64 - 1) + 128) >> 8);
        }
        if steep {
            for j in 0..stride {
                a[j] = (samp(j) * 65535.0) as i64;
            }
        }
        let mut out = [0u16; N];
        for (o, v) in out.iter_mut().zip(a) {
            *o = v as u16;
        }
        out
    }

    /// `m × s`.
    fn scale_matrix(m: &[[f64; 3]; 3], s: f64) -> [[f64; 3]; 3] {
        let mut o = *m;
        for row in &mut o {
            for v in row {
                *v *= s;
            }
        }
        o
    }

    /// 16-bit matrix cube pass (E=256/ELUTbits=16/N=1024/ALUTbits=16): resample `nodes`
    /// cube nodes in place through per-channel input shaper `s`, the integer colorant `matrix`, and
    /// per-channel output ALUT `a`.
    fn do_matrix_pass(cube: &mut [u16], s: &[[u16; E]; 3], matrix: &[[f64; 3]; 3], a: &[[u16; N]; 3], nodes: usize) {
        // matrix coeff = round((N-1)·16·m), rounding away from zero (±0.5 by sign).
        let mut m = [[0i64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let c = (N as f64 - 1.0) * 16.0 * matrix[i][j];
                m[i][j] = (c + if c <= 0.0 { -0.5 } else { 0.5 }) as i64;
            }
        }
        for node in 0..nodes {
            let b = node * 3;
            // shaper lerp: t = v−(v>>8), gi=t>>8, fr=t&255; shaped = lerp(S[gi],S[gi+1]) >> (ELUTbits−8=8).
            let mut x = [0i64; 3];
            for ch in 0..3 {
                let v = cube[b + ch] as i64;
                let t = v - (v >> 8);
                let gi = (t >> 8) as usize;
                let fr = t & 0xFF;
                let shaped = if gi >= E - 1 {
                    s[ch][E - 1] as i64
                } else {
                    (fr * s[ch][gi + 1] as i64 + (256 - fr) * s[ch][gi] as i64) >> 8
                };
                x[ch] = (shaped + 2) >> 2;
            }
            for i in 0..3 {
                let out = ((m[i][0] * x[0] + m[i][1] * x[1] + m[i][2] * x[2] + 511) >> 10).max(0);
                let oi = (out >> 8) as usize;
                let of = out & 0xFF;
                let v = if oi >= N - 1 {
                    a[i][N - 1] as i64
                } else {
                    (of * a[i][oi + 1] as i64 + (256 - of) * a[i][oi] as i64) >> 8
                };
                cube[b + i] = v as u16;
            }
        }
    }

    /// Resample a lut16 output table (`entries` × u16) to the 1024-entry
    /// output ALUT with the native fixed-point (`((entries-1)<<20)/0x3FF` step, 15-bit frac lerp).
    fn clut_output_alut(table: &[u16], entries: usize) -> [u16; N] {
        let step = (((entries as i64 - 1) << 20) / 0x3FF) as i64;
        let mut a = [0u16; N];
        for (k, out) in a.iter_mut().enumerate() {
            let s = (step * k as i64 + 16) >> 5;
            let (idx, fr) = ((s >> 15) as usize, s & 0x7FFF);
            let t0 = table[idx.min(entries - 1)] as i64;
            *out = if fr != 0 {
                let t1 = table[(idx + 1).min(entries - 1)] as i64;
                (t0 + ((fr * (t1 - t0) + 0x3FFF) >> 15)) as u16
            } else {
                t0 as u16
            };
        }
        a
    }

    /// lut16 input ELUT (ELUTbits=16): resample an input table (`entries`×u16) to the
    /// 256-entry per-channel input ELUT with the native fixed-point (differs from the linear input ELUT
    /// even for an identity table — it is the lut16 per-pixel ELUT, before the grid descale).
    fn fill_ushort_elut_lut16(table: &[u16], entries: usize) -> [u16; E] {
        let step = (((entries as i64) << 18) - 0x40000) / 255;
        let mut mult = (((1i64 << 16) - 1) as f64 / 65535.0 * 4194304.0) as i64;
        let (mut shift, mut bias) = (22i64, 0x1FFFFFi64);
        while mult & 0xFFF8000 != 0 {
            mult >>= 1;
            shift -= 1;
            bias >>= 1;
        }
        let mut e = [0u16; E];
        for (i, out) in e.iter_mut().enumerate() {
            let s = (i as i64 * step + 4) >> 3;
            let (idx, frac) = ((s >> 15) as usize, s & 0x7FFF);
            let t0 = table[idx.min(entries - 1)] as i64;
            let scaled0 = t0 * mult;
            let (acc, out_shift) = if frac != 0 {
                let t1 = table[(idx + 1).min(entries - 1)] as i64;
                ((bias >> 1) + ((frac * ((mult * (t1 - t0)) >> 15)) >> 1) + (scaled0 >> 1), shift - 1)
            } else {
                (bias + scaled0, shift)
            };
            *out = (acc >> out_shift) as u16;
        }
        e
    }

    /// lut8 input ELUT (ELUTbits=16): the lut8 input table is a fixed 256-entry u8
    /// ramp, promoted to the 16-bit per-pixel ELUT by `(((4096<<16)−4096)/255·t + 2048) >> 12`.
    fn fill_ushort_elut_lut8(table: &[u8; 256]) -> [u16; E] {
        let mult = (((4096i64 << 16) - 4096) / 255) as i64;
        std::array::from_fn(|k| ((mult * table[k] as i64 + 2048) >> 12) as u16)
    }

    /// lut8 output ALUT: lut8 output table (256×u8) → 1024-entry output ALUT, promoted
    /// ×257 and resampled (`idx=(1021·k)>>12`, 12-bit frac lerp); the last entry is the table endpoint.
    fn clut_output_alut_lut8(table: &[u8; 256]) -> [u16; N] {
        let mut a = [0u16; N];
        for (k, out) in a.iter_mut().enumerate().take(N - 1) {
            let s = 1021 * k as i64;
            let (idx, frac) = ((s >> 12) as usize, s & 0xFFF);
            let t0 = 257 * table[idx] as i64;
            *out = if frac != 0 {
                let t1 = 257 * table[idx + 1] as i64;
                (t0 + ((frac * (t1 - t0) + 2048) >> 12)) as u16
            } else {
                t0 as u16
            };
        }
        a[N - 1] = 257 * table[255] as u16;
        a
    }

    /// `FillDescaledInputLut` for a `gp`-point CLUT grid (ELUTbits=16): descale a linear 256-entry ELUT
    /// so a device byte maps to a grid coord (cell in the top `log2(gp)` bits, frac below), top = frac 0.
    fn clut_input_elut(gp: usize) -> [u16; E] {
        let lin = linear_elut16();
        let mut e = [0u16; E];
        for (i, out) in e.iter_mut().enumerate() {
            let v = lin[i] as u64;
            *out = ((0x10000 * v / 0xFFFF) * (gp as u64 - 1) / gp as u64) as u16;
        }
        e
    }

    /// 3→3 CLUT resample (pow-of-2 grid): resample the `nodes` ramp nodes of `cube` in
    /// place through the lut16/lut8 A2B — `in_elut` (descaled linear) → cell/frac → tetrahedral CLUT
    /// interpolation → the CLUT value replicated `clut_bits`→16-bit → per-channel output ALUT. 16-bit
    /// output ALUT (16-bit, the high-quality path).
    fn do_clut_pass(cube: &mut [u16], in_elut: &[u16; E], lut: &Lut, nodes: usize) {
        let (gp, cb) = (lut.gp, lut.clut_bits as i64);
        // pow-of-2 grids use shift/mask off the 16-bit descaled ELUT (the fast path); other
        // grids multiply by gp and take the top 16 bits (frac range 2^16) — the general branch.
        let pow2 = gp.is_power_of_two();
        let frac_shift = if pow2 { 16 - gp.trailing_zeros() as i64 } else { 16 };
        let (fracrange, fracmask) = (1i64 << frac_shift, (1i64 << frac_shift) - 1);
        let stride = [gp * gp, gp, 1];
        let shift_total = frac_shift + cb - 10; // (frac_shift) + CLUTbits − 10
        let (shift_out, pre_shift) = (shift_total.min(16), (shift_total - 16).max(0));
        let unit = 1i64 << shift_out;
        for node in 0..nodes {
            let bb = node * 3;
            let (mut cell, mut frac) = ([0usize; 3], [0i64; 3]);
            for c in 0..3 {
                let v = cube[bb + c] as i64;
                let t = v - (v >> 8);
                let (hi, lo) = ((t >> 8) as usize, t & 0xFF);
                let d = (lo * in_elut[(hi + 1).min(E - 1)] as i64 + (256 - lo) * in_elut[hi.min(E - 1)] as i64) >> 8;
                let scaled = if pow2 { d } else { gp as i64 * d };
                cell[c] = (scaled >> frac_shift) as usize;
                frac[c] = scaled & fracmask;
            }
            let base = cell[0] * stride[0] + cell[1] * stride[1] + cell[2];
            let mut ax = [0usize, 1, 2];
            ax.sort_by(|&a, &b| frac[b].cmp(&frac[a]));
            let (a0, a1, a2) = (ax[0], ax[1], ax[2]);
            let w = [fracrange - frac[a0], frac[a0] - frac[a1], frac[a1] - frac[a2], frac[a2]];
            // tetrahedral corners; a +1 neighbour is dropped at the grid top (native's boundary clamp).
            let nb = |a: usize| if cell[a] < gp - 1 { stride[a] } else { 0 };
            let off = [0, nb(a0), nb(a0) + nb(a1), nb(a0) + nb(a1) + nb(a2)];
            for i in 0..3 {
                let sum: i64 = (0..4).map(|k| w[k] * lut.clut[(base + off[k]) * 3 + i] as i64).sum();
                // replicate the clut_bits-wide value to 16-bit precision (S + S>>cb + S>>2cb + …).
                let (mut term, mut sh) = (sum, cb);
                while sh <= 16 {
                    term += sum >> sh;
                    sh += cb;
                }
                let pos = (term - (term >> 10)) >> pre_shift;
                let (idx, fr) = ((pos >> shift_out) as usize, pos & (unit - 1));
                let a = &lut.out_alut[i];
                cube[bb + i] = (((unit - fr) * a[idx.min(N - 1)] as i64 + fr * a[(idx + 1).min(N - 1)] as i64) >> shift_out) as u16;
            }
        }
    }

    impl Combi {
        /// Build the device-link the native "Win " CMM builds for a src→dst matrix/shaper pair
        /// (the pure-matrix path): an identity ramp resampled by two
        /// *pure-matrix* cube passes (linear shaper + linear ALUT) — src colorant·0.5
        /// → 16-bit PCS, then (dst colorant·0.5)⁻¹ → linear dst-RGB. The TRCs live in the **per-pixel**
        /// LUTs: input ELUT = srcTRC (descaled to the grid), output ALUT = dst inverse TRC.
        pub fn new(src: &IccProfile, dst: &IccProfile) -> Combi {
            let nodes = GP * GP * GP;
            // identity ramp: node (R,G,B) = bit-replicated grid index, order R·1024+G·32+B. The 1057
            // halo nodes stay 0 (unprocessed); the descaled per-pixel ELUT keeps cells off them.
            let mut cube = vec![0u16; (nodes + GP * GP + GP + 1) * 3];
            for r in 0..GP {
                for g in 0..GP {
                    for bl in 0..GP {
                        let idx = (r * GP * GP + g * GP + bl) * 3;
                        cube[idx] = expand16(r as u32, 5) as u16;
                        cube[idx + 1] = expand16(g as u32, 5) as u16;
                        cube[idx + 2] = expand16(bl as u32, 5) as u16;
                    }
                }
            }
            let lin_elut = [linear_elut16(); 3];
            let lin_alut = [linear_alut16(); 3];
            // src pass → 16-bit PCS. A CLUT (A2B) src resamples through the lut16 CLUT;
            // a matrix/shaper src applies srcColorant·0.5 (the matrix cube pass). The colorant is just
            // ×(1/65536)·0.5: its wtpt/D50 row-scale is gated on absolute intent,
            // which the relative-colorimetric transform WIC uses never hits.
            let elut = match &src.lut {
                Some(lut) => {
                    // CLUT resample (lut16/lut8/mAB); the per-pixel input ELUT is prebuilt from the A2B
                    // input tables (the CLUT path's own input ELUT is the descaled linear `clut_input_elut`).
                    do_clut_pass(&mut cube, &clut_input_elut(lut.gp), lut, nodes);
                    // mAB M-matrix (colorant·0.5) on the CLUT output, then the V4 perceptual black-point
                    // strip — both only for `lutAToBType` (V4) profiles; lut16/lut8 (V2) skip them.
                    if let Some(m) = lut.mab_matrix {
                        do_matrix_pass(&mut cube, &lin_elut, &m, &lin_alut, nodes);
                    }
                    if src.v4 {
                        adjust_black_v4_to_legacy(&mut cube, nodes);
                    }
                    lut.pp_elut
                }
                None => {
                    let src_m = scale_matrix(&src.mtx, 0.5);
                    do_matrix_pass(&mut cube, &lin_elut, &src_m, &lin_alut, nodes);
                    // per-pixel ELUT = srcTRC (descaled to the grid so the top cell needs no halo).
                    let mut e = [shaper(&src.trc[0]), shaper(&src.trc[1]), shaper(&src.trc[2])];
                    for c in &mut e {
                        descale_elut(c);
                    }
                    e
                }
            };
            // A Lab-PCS source (a scanner A2B profile) emits 16-bit CIELAB from its CLUT; the "Win "
            // CMM converts each node to XYZ before the destination colorant matrix (which is XYZ-PCS).
            if src.pcs == PCS_LAB {
                lab2xyz_cube16(&mut cube, nodes);
            }
            // dst pass: PCS → (dstColorant·0.5)⁻¹ → linear dst-RGB.
            let dst_m = mat_inv(&scale_matrix(&dst.mtx, 0.5));
            do_matrix_pass(&mut cube, &lin_elut, &dst_m, &lin_alut, nodes);
            // per-pixel ALUT = dst inverse TRC.
            let alut = [inv_alut(&dst.trc[0]), inv_alut(&dst.trc[1]), inv_alut(&dst.trc[2])];
            Combi { elut, cube, alut }
        }

        /// Tetrahedral cube lookup: sorted-barycentric over the cell of the grid word `word[c]`
        /// (`cell=word>>11`, `frac=word&0x7FF`), 11-bit weights summing to 2048. Returns the per-output
        /// `Σ wᵢ·nodeᵢ` the kernels then map through the output ALUT. Shared by the 8- and 16-bit paths.
        fn tetra_sums(&self, word: [u16; 3]) -> [i64; 3] {
            let frac = [(word[0] & 0x7FF) as i64, (word[1] & 0x7FF) as i64, (word[2] & 0x7FF) as i64];
            let step = [GP * GP, GP, 1]; // R, G, B node strides
            let base = (word[0] >> 11) as usize * step[0] + (word[1] >> 11) as usize * step[1] + (word[2] >> 11) as usize * step[2];
            let node = |off: usize, ch: usize| self.cube[(base + off) * 3 + ch] as i64;
            // sort axes by frac descending (stable: R>G>B on ties; equal-frac corners carry weight 0).
            let mut ax = [0usize, 1, 2];
            ax.sort_by(|&a, &b| frac[b].cmp(&frac[a]));
            let (a0, a1, a2) = (ax[0], ax[1], ax[2]);
            let (s0, s1, s2) = (frac[a0], frac[a1], frac[a2]);
            let w = [2048 - s0, s0 - s1, s1 - s2, s2];
            let off = [0usize, step[a0], step[a0] + step[a1], step[0] + step[1] + step[2]];
            std::array::from_fn(|ch| (0..4).map(|i| w[i] * node(off[i], ch)).sum())
        }

        /// 8-bit apply (3→3, grid 32): per-channel ELUT byte lookup → cube → `>>17`
        /// → output ALUT's high byte. `rgb` in R,G,B order; returns R,G,B.
        pub fn apply(&self, rgb: [u8; 3]) -> [u8; 3] {
            let word = [self.elut[0][rgb[0] as usize], self.elut[1][rgb[1] as usize], self.elut[2][rgb[2] as usize]];
            let sums = self.tetra_sums(word);
            std::array::from_fn(|ch| (self.alut[ch][((sums[ch] >> 17) as usize).min(1023)] >> 8) as u8)
        }

        /// 16-bit apply (3→3, grid 32): the ELUT is interpolated by the 16-bit
        /// input (high byte indexes, low byte lerps — same fixed-point as the shaper), and the cube
        /// sum maps through a 9-bit output-ALUT interpolation with the `S+(S>>16)`, `u−(u>>10)` rounding.
        pub fn apply16(&self, rgb: [u16; 3]) -> [u16; 3] {
            let word = std::array::from_fn(|c| {
                let v = rgb[c] as i64;
                let t = v - (v >> 8);
                let (hi, lo) = ((t >> 8) as usize, t & 0xFF);
                ((lo * self.elut[c][(hi + 1).min(255)] as i64 + (256 - lo) * self.elut[c][hi] as i64) >> 8) as u16
            });
            let sums = self.tetra_sums(word);
            std::array::from_fn(|ch| {
                let s = sums[ch];
                let acc = s + (s >> 16);
                let u = acc - (acc >> 10);
                let idx = (u >> 17) as usize;
                let f = (u >> 8) & 0x1FF;
                let a = &self.alut[ch];
                ((f * a[(idx + 1).min(1023)] as i64 + (512 - f) * a[idx.min(1023)] as i64) >> 9) as u16
            })
        }
    }

    /// A gray (`GRAY`/XYZ) source profile's `kTRC` tone curve, if the blob is a gray device profile.
    pub fn gray_ktrc(p: &[u8]) -> Option<Curve> {
        if p.len() < 132 || &p[36..40] != b"acsp" || be32(p, 16) != 0x47524159 {
            return None; // not a GRAY data-space profile
        }
        Some(curve_tag(p, b"kTRC"))
    }

    const GGP: usize = 128; // gray grid points

    /// A built gray→RGB device-link (1→3): the source
    /// `kTRC` as the per-pixel input ELUT (descaled to the 128-point grid), a 128-node cube = the gray
    /// PCS ramp (black→the CMM's fixed D50 gray white-point, ×0.5) resampled through the destination's
    /// colorant-inverse matrix, and the destination's inverse TRC as the per-channel output ALUT.
    pub struct GrayCombi {
        elut: [u16; E],
        cube: Vec<u16>, // GGP * 3, dst-linear·65535
        alut: [[u16; N]; 3],
    }

    impl GrayCombi {
        /// `gray` = the source `kTRC`; `dst` = the (matrix/shaper) destination profile.
        pub fn new(gray: &Curve, dst: &IccProfile) -> GrayCombi {
            // Hardcoded PCS gray endpoints: node0 = black, node1 = the D50
            // gray white (×0.5, 16-bit). The 128-node ramp is the linear interpolation black→white.
            const WHITE: [u16; 3] = [31594, 32767, 27105];
            // 128-node gray PCS ramp (black→white, the 2-node source CLUT resampled over the grid).
            let mut cube = vec![0u16; GGP * 3];
            for k in 0..GGP {
                let p = ((k as u64) << 16) / (GGP as u64 - 1); // node 127 → 65536 (white exact)
                for (c, w) in WHITE.iter().enumerate() {
                    cube[k * 3 + c] = ((*w as u64 * p) >> 16) as u16;
                }
            }
            // dst pass: PCS → (dstColorant·0.5)⁻¹ → linear dst-RGB (same as the 3→3 dst matrix pass).
            let dst_m = mat_inv(&scale_matrix(&dst.mtx, 0.5));
            do_matrix_pass(&mut cube, &[linear_elut16(); 3], &dst_m, &[linear_alut16(); 3], GGP);
            // per-pixel input ELUT = srcTRC descaled to the 128-grid (device byte → cell<<9 | frac).
            let shaped = shaper(gray);
            let elut = std::array::from_fn(|i| ((0x10000u64 * shaped[i] as u64 / 0xFFFF) * (GGP as u64 - 1) / GGP as u64) as u16);
            let alut = [inv_alut(&dst.trc[0]), inv_alut(&dst.trc[1]), inv_alut(&dst.trc[2])];
            GrayCombi { elut, cube, alut }
        }

        /// 8-bit apply (1→3, grid 128): gray byte → ELUT → `cell=w>>9`, `frac=w&0x1FF`
        /// → per-channel 2-node lerp `(frac·next+(512−frac)·cur)>>15` → output ALUT high byte. → R,G,B.
        pub fn apply(&self, g: u8) -> [u8; 3] {
            let w = self.elut[g as usize] as i64;
            let (cell, frac) = ((w >> 9) as usize, w & 0x1FF);
            let node = |k: usize, c: usize| self.cube[k.min(GGP - 1) * 3 + c] as i64;
            std::array::from_fn(|c| {
                let idx = ((frac * node(cell + 1, c) + (512 - frac) * node(cell, c)) >> 15) as usize;
                (self.alut[c][idx.min(N - 1)] >> 8) as u8
            })
        }

        /// 16-bit-output apply: 8-bit gray input, 16-bit RGB output. Same 2-node lerp sum as [`apply`]
        /// (its `>>15` high byte is byte-exact in rgb8/bgra8); the output runs the generic
        /// 16-bit output stage (same as [`Combi4::apply16`] / [`do_clut_pass4`]). For the 9-bit grid frac
        /// (GGP=128) and `cb=16`: `shift_total = 9+16−10 = 15 → shift_out=15, pre_shift=0`, so `term = s + s>>16`,
        /// `pos = term − term>>10`, idx `pos>>15`, 15-bit frac `pos & 0x7FFF`,
        /// `((2¹⁵−fr)·A[idx] + fr·A[idx+1]) >> 15`.
        pub fn apply16(&self, g: u8) -> [u16; 3] {
            let w = self.elut[g as usize] as i64;
            let (cell, frac) = ((w >> 9) as usize, w & 0x1FF);
            let node = |k: usize, c: usize| self.cube[k.min(GGP - 1) * 3 + c] as i64;
            std::array::from_fn(|c| {
                let s = frac * node(cell + 1, c) + (512 - frac) * node(cell, c);
                let term = s + (s >> 16);
                let pos = term - (term >> 10);
                let idx = (pos >> 15) as usize;
                let fr = pos & 0x7FFF;
                let a = &self.alut[c];
                (((0x8000 - fr) * a[idx.min(N - 1)] as i64 + fr * a[(idx + 1).min(N - 1)] as i64) >> 15) as u16
            })
        }
    }

    const GP4: usize = 16; // 4-input combi grid (2^4)

    /// A parsed CMYK (`mft2`, 4→3) A2B, prebuilt for the combi: grid points, the device→PCS CLUT
    /// (gp⁴·3, u16), its bit width (16), the per-pixel 4-channel input ELUT (input tables resampled
    /// then combi-grid-descaled) and the 3-channel output ALUT. The only non-3→3 CLUT WIC's transform
    /// runs (a 'prtr'/'CMYK' profile); the mft2 extractors already accept inCh=4.
    pub struct Lut4 {
        pub gp: usize,
        pub clut: Vec<u16>,
        pub clut_bits: u32,
        pub pp_elut: [[u16; E]; 4],
        pub out_alut: [[u16; N]; 3],
    }

    /// Parse a CMYK `lut16Type` ('mft2') A2B (4 input, 3 output); `None` for any other shape. CLUT at
    /// tag off 52+4·ie·2 (`inCh=4`), gp⁴·3 u16; input tables (4×ie) at 52, output tables (3×oe) after.
    pub fn cmyk_a2b(p: &[u8]) -> Option<Lut4> {
        if p.len() < 132 || &p[36..40] != b"acsp" || be32(p, 16) != 0x434D_594B {
            return None; // not a CMYK data-space profile
        }
        let t = [b"A2B0", b"A2B1", b"A2B2"].iter().find_map(|s| tag(p, s))?;
        if &t[0..4] != b"mft2" || t[8] != 4 || t[9] != 3 {
            return None;
        }
        let gp = t[10] as usize;
        let (ie, oe) = (be16(t, 48) as usize, be16(t, 50) as usize);
        let nc = gp.pow(4) * 3;
        let (clut_off, out_off) = (52 + 4 * ie * 2, 52 + 4 * ie * 2 + nc * 2);
        let clut = (0..nc).map(|i| be16(t, clut_off + i * 2)).collect();
        let pp_elut = std::array::from_fn(|c| {
            let it: Vec<u16> = (0..ie).map(|e| be16(t, 52 + (c * ie + e) * 2)).collect();
            let mut e = fill_ushort_elut_lut16(&it, ie);
            descale_elut_gp(&mut e, GP4);
            e
        });
        let out_alut = std::array::from_fn(|c| clut_output_alut(&(0..oe).map(|e| be16(t, out_off + (c * oe + e) * 2)).collect::<Vec<_>>(), oe));
        Some(Lut4 { gp, clut, clut_bits: 16, pp_elut, out_alut })
    }

    /// 4→3 CLUT resample (4-input simplex extension of [`do_clut_pass`]'s 3-tetra):
    /// fill the `GP4⁴` ramp nodes of `cube` from the per-axis grid ramp through `lut`'s gp⁴ CLUT — each
    /// axis descaled-linear → cell/frac (16-bit frac, non-pow-of-2 gp general branch) → 5-corner simplex
    /// → the CLUT value replicated `clut_bits`→16-bit → per-channel output ALUT.
    fn do_clut_pass4(cube: &mut [u16], in_elut: &[u16; E], lut: &Lut4, nodes: usize) {
        let (gp, cb) = (lut.gp, lut.clut_bits as i64);
        let frac_shift = 16i64; // gp (=9) non-pow-of-2 → the general branch (frac range 2^16)
        let (fracrange, fracmask) = (1i64 << frac_shift, (1i64 << frac_shift) - 1);
        let stride = [gp * gp * gp, gp * gp, gp, 1];
        let shift_total = frac_shift + cb - 10;
        let (shift_out, pre_shift) = (shift_total.min(16), (shift_total - 16).max(0));
        let unit = 1i64 << shift_out;
        for node in 0..nodes {
            let (mut cell, mut frac) = ([0usize; 4], [0i64; 4]);
            for c in 0..4 {
                // input coord = this axis's grid ramp value (bit-replicated digit).
                let digit = (node >> (4 * (3 - c))) & (GP4 - 1);
                let v = expand16(digit as u32, 4) as i64;
                let t = v - (v >> 8);
                let (hi, lo) = ((t >> 8) as usize, t & 0xFF);
                let d = (lo * in_elut[(hi + 1).min(E - 1)] as i64 + (256 - lo) * in_elut[hi.min(E - 1)] as i64) >> 8;
                let scaled = gp as i64 * d;
                cell[c] = (scaled >> frac_shift) as usize;
                frac[c] = scaled & fracmask;
            }
            let base = cell[0] * stride[0] + cell[1] * stride[1] + cell[2] * stride[2] + cell[3];
            let mut ax = [0usize, 1, 2, 3];
            ax.sort_by(|&a, &b| frac[b].cmp(&frac[a]));
            let w = [fracrange - frac[ax[0]], frac[ax[0]] - frac[ax[1]], frac[ax[1]] - frac[ax[2]], frac[ax[2]] - frac[ax[3]], frac[ax[3]]];
            let nb = |a: usize| if cell[a] < gp - 1 { stride[a] } else { 0 };
            let mut off = [0usize; 5];
            for k in 1..5 {
                off[k] = off[k - 1] + nb(ax[k - 1]);
            }
            for i in 0..3 {
                let sum: i64 = (0..5).map(|k| w[k] * lut.clut[(base + off[k]) * 3 + i] as i64).sum();
                let (mut term, mut sh) = (sum, cb);
                while sh <= 16 {
                    term += sum >> sh;
                    sh += cb;
                }
                let pos = (term - (term >> 10)) >> pre_shift;
                let (idx, fr) = ((pos >> shift_out) as usize, pos & (unit - 1));
                let a = &lut.out_alut[i];
                cube[node * 3 + i] = (((unit - fr) * a[idx.min(N - 1)] as i64 + fr * a[(idx + 1).min(N - 1)] as i64) >> shift_out) as u16;
            }
        }
    }

    /// A built CMYK→RGB device-link (4→3): per-pixel 4-channel input ELUT, a GP4⁴
    /// cube (3×u16/node = dst-linear·65535, the gp⁴ A2B CLUT resampled then the dst colorant-inverse
    /// matrix), and the destination's inverse TRC as the 3-channel output ALUT.
    pub struct Combi4 {
        elut: [[u16; E]; 4],
        cube: Vec<u16>, // GP4⁴ * 3
        alut: [[u16; N]; 3],
    }

    impl Combi4 {
        /// `cmyk` = the parsed CMYK A2B; `dst` = the (matrix/shaper) destination profile.
        pub fn new(cmyk: &Lut4, dst: &IccProfile) -> Combi4 {
            let nodes = GP4.pow(4);
            let mut cube = vec![0u16; nodes * 3];
            do_clut_pass4(&mut cube, &clut_input_elut(cmyk.gp), cmyk, nodes);
            // dst pass: PCS → (dstColorant·0.5)⁻¹ → linear dst-RGB (same 3-channel matrix pass as 3→3).
            let dst_m = mat_inv(&scale_matrix(&dst.mtx, 0.5));
            do_matrix_pass(&mut cube, &[linear_elut16(); 3], &dst_m, &[linear_alut16(); 3], nodes);
            let alut = [inv_alut(&dst.trc[0]), inv_alut(&dst.trc[1]), inv_alut(&dst.trc[2])];
            Combi4 { elut: cmyk.pp_elut, cube, alut }
        }

        /// 4-input simplex over the GP4⁴ cube: per-channel ELUT word → `cell=word>>12`, `frac=word&0xFFF`
        /// → sorted-barycentric 5-corner sum `Σ wᵢ·nodeᵢ` (12-bit weights summing to 4096).
        fn simplex_sums(&self, word: [u16; 4]) -> [i64; 3] {
            let fracbits = 16 - GP4.trailing_zeros() as i64; // 12 (grid 16)
            let fracrange = 1i64 << fracbits;
            let frac: [i64; 4] = std::array::from_fn(|c| word[c] as i64 & (fracrange - 1));
            let cell: [usize; 4] = std::array::from_fn(|c| (word[c] >> fracbits) as usize);
            let stride = [GP4 * GP4 * GP4, GP4 * GP4, GP4, 1];
            let base: usize = (0..4).map(|c| cell[c] * stride[c]).sum();
            let mut ax = [0usize, 1, 2, 3];
            ax.sort_by(|&a, &b| frac[b].cmp(&frac[a]));
            let w = [fracrange - frac[ax[0]], frac[ax[0]] - frac[ax[1]], frac[ax[1]] - frac[ax[2]], frac[ax[2]] - frac[ax[3]], frac[ax[3]]];
            let nb = |a: usize| if cell[a] < GP4 - 1 { stride[a] } else { 0 };
            let mut off = [0usize; 5];
            for k in 1..5 {
                off[k] = off[k - 1] + nb(ax[k - 1]);
            }
            std::array::from_fn(|ch| (0..5).map(|k| w[k] * self.cube[(base + off[k]) * 3 + ch] as i64).sum())
        }

        /// 8-bit apply (4→3, grid 16): CMYK byte → ELUT → simplex → `>>18` → output
        /// ALUT high byte. Returns R,G,B.
        pub fn apply(&self, cmyk: [u8; 4]) -> [u8; 3] {
            let word = std::array::from_fn(|c| self.elut[c][cmyk[c] as usize]);
            let sums = self.simplex_sums(word);
            std::array::from_fn(|ch| (self.alut[ch][((sums[ch] >> 18) as usize).min(N - 1)] >> 8) as u8)
        }

        /// 16-bit-output apply: 8-bit CMYK input, 16-bit RGB output. Same simplex sum as [`apply`]
        /// (the `>>18` high byte is byte-exact in rgb8/bgra8); the per-pixel output runs the SAME generic
        /// 16-bit output stage as [`do_clut_pass4`], not the 10-bit output stage. For the
        /// 12-bit simplex frac and `cb=16`: `shift_total = 12+16−10 = 18 → shift_out=16, pre_shift=2`, so `term =
        /// s + s>>16`, `pos = (term − term>>10) >> 2`, idx `pos>>16`, **16-bit** frac `pos & 0xFFFF`,
        /// `((2¹⁶−fr)·A[idx] + fr·A[idx+1]) >> 16`. The extra 6 fraction bits over the 10-bit stage are
        /// the CMYK rgba16 ≤3-LSB residual.
        pub fn apply16(&self, cmyk: [u8; 4]) -> [u16; 3] {
            let word = std::array::from_fn(|c| self.elut[c][cmyk[c] as usize]);
            let sums = self.simplex_sums(word);
            std::array::from_fn(|ch| {
                let term = sums[ch] + (sums[ch] >> 16);
                let pos = (term - (term >> 10)) >> 2;
                let idx = (pos >> 16) as usize;
                let fr = pos & 0xFFFF;
                let a = &self.alut[ch];
                (((0x10000 - fr) * a[idx.min(N - 1)] as i64 + fr * a[(idx + 1).min(N - 1)] as i64) >> 16) as u16
            })
        }
    }

    #[cfg(test)]
    mod oracle {
        use super::*;
        use std::path::PathBuf;

        fn dir() -> Option<PathBuf> {
            d2d_reference::suite_dir().map(|s| s.join("fixtures/wic_colortransform"))
        }

        /// Byte-exact against the native `wic-dump colortransform` captures for every direction and
        /// every integer pixel format: BGRA8/RGB8 via the 8-bit kernel, RGBA16 via the 16-bit kernel.
        /// Alpha is passed through by WIC (not the CMM), so the colour channels alone are compared.
        /// Build the combi for `src`→`dst` and assert it reproduces the native capture
        /// `oracle/{tag}_{fmt}.f0.bin` byte-for-byte for all three integer formats.
        fn check(d: &std::path::Path, tag: &str, src: &str, dst: &str) {
            let prof = |n: &str| IccProfile::parse(&std::fs::read(d.join(n)).unwrap()).unwrap();
            let combi = Combi::new(&prof(src), &prof(dst));
            let bgra8 = std::fs::read(d.join("in_bgra8.bin")).unwrap();
            let rgb8 = std::fs::read(d.join("in_rgb8.bin")).unwrap();
            let rgba16: Vec<u16> = std::fs::read(d.join("in_rgba16.bin")).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let n = bgra8.len() / 4;

            let o = std::fs::read(d.join(format!("oracle/{tag}_bgra8.f0.bin"))).unwrap();
            for i in 0..n {
                let p = &bgra8[i * 4..i * 4 + 4];
                let got = combi.apply([p[2], p[1], p[0]]);
                assert_eq!([got[2], got[1], got[0], p[3]], o[i * 4..i * 4 + 4], "{tag}/bgra8 px {i}");
            }
            let o = std::fs::read(d.join(format!("oracle/{tag}_rgb8.f0.bin"))).unwrap();
            for i in 0..n {
                let p = &rgb8[i * 3..i * 3 + 3];
                assert_eq!(combi.apply([p[0], p[1], p[2]]), o[i * 3..i * 3 + 3], "{tag}/rgb8 px {i}");
            }
            let o: Vec<u16> = std::fs::read(d.join(format!("oracle/{tag}_rgba16.f0.bin"))).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            for i in 0..n {
                let p = &rgba16[i * 4..i * 4 + 4];
                let got = combi.apply16([p[0], p[1], p[2]]);
                assert_eq!([got[0], got[1], got[2], p[3]], o[i * 4..i * 4 + 4], "{tag}/rgba16 px {i}");
            }
        }

        /// The generated `LAB_FINV` implements the CIE Lab f⁻¹ companding: 0 at the black knee,
        /// 65535 at the top, monotonic non-decreasing, and the lower tail is the exact linear segment
        /// `3·(6/29)²·(fy−4/29)` (independently recomputed). `lab2xyz_cube16` then wires it per the
        /// the reference arithmetic: a node with L*≈0 (L16=0) maps to XYZ black.
        #[test]
        fn lab_finv_is_cie_companding() {
            assert_eq!(LAB_FINV[0], 0);
            assert_eq!(LAB_FINV[431], 65535);
            assert!(LAB_FINV.windows(2).all(|w| w[1] >= w[0]), "monotonic");
            // Linear-segment knee: below fy = 6/29 the companding is the exact affine ramp (f32).
            const D: f32 = 6.0 / 29.0;
            for i in 0..18 {
                let fy = (i as f32 * (25600.0 / 65536.0) + 16.0) * (1.0 / 116.0);
                let want = (3.0 * D * D * (fy - 4.0 / 29.0) * 16383.0 + 0.5).floor() as u16;
                assert_eq!(LAB_FINV[i], want, "knee entry {i}");
            }
            let mut node = [0u16, 32768, 32768]; // L*=0, a*=b*=0 → XYZ black
            lab2xyz_cube16(&mut node, 1);
            assert_eq!(node, [0, 0, 0]);
        }

        /// Matrix/shaper 'curv'-gamma profiles: srgb↔wide in every direction.
        #[test]
        fn cube_vs_oracle() {
            let Some(d) = dir() else { return };
            for (tag, s, t) in [("ss", "srgb.icc", "srgb.icc"), ("sw", "srgb.icc", "wide.icc"), ("ws", "wide.icc", "srgb.icc")] {
                check(&d, tag, s, t);
            }
        }

        /// Lab-PCS end-to-end **measurement** (WIP): lab.icc ('scnr'/Lab) → srgb.icc vs the native
        /// `labs_*` oracle. Run with `cargo test -- --ignored --nocapture` and D2D_SUITE pointing at a
        /// suite that has the labs fixtures. Current residual is ~15 LSB (bgra8/rgb8) / ~4139 (rgba16),
        /// and it is the SAME with the exact native companding table swapped in — so the error is the
        /// Lab CLUT→XYZ **integration** in `Combi::new`, not the (formula-generated) companding
        /// table. Needs a VM `colortransforminterm` trace to localise; `#[ignore]` until byte-exact.
        #[test]
        #[ignore = "Lab WIP: ~15 LSB vs oracle = the generated companding table (PROVEN by interm_cubes_vs_vm: lab_pre byte-exact, native companding table → lab_post byte-exact); exact-table generator open"]
        fn lab_end_to_end() {
            let Some(d) = dir() else { return };
            if !d.join("lab.icc").exists() {
                return; // labs fixtures only in the merged suite / main checkout
            }
            let prof = |n: &str| IccProfile::parse(&std::fs::read(d.join(n)).unwrap()).unwrap();
            let combi = Combi::new(&prof("lab.icc"), &prof("srgb.icc"));
            let bgra8 = std::fs::read(d.join("in_bgra8.bin")).unwrap();
            let rgb8 = std::fs::read(d.join("in_rgb8.bin")).unwrap();
            let rgba16: Vec<u16> = std::fs::read(d.join("in_rgba16.bin")).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let n = bgra8.len() / 4;
            let mut worst = 0u32;
            // 8-bit (bgra8 + rgb8)
            for (fmt, o) in [("bgra8", "labs_bgra8.f0.bin"), ("rgb8", "labs_rgb8.f0.bin")] {
                let o = std::fs::read(d.join(format!("oracle/{o}"))).unwrap();
                let (mut maxd, mut miss) = (0u32, 0usize);
                for i in 0..n {
                    let got = if fmt == "bgra8" {
                        let p = &bgra8[i * 4..i * 4 + 3];
                        let g = combi.apply([p[2], p[1], p[0]]);
                        [g[2], g[1], g[0]]
                    } else {
                        let p = &rgb8[i * 3..i * 3 + 3];
                        combi.apply([p[0], p[1], p[2]])
                    };
                    let base = if fmt == "bgra8" { i * 4 } else { i * 3 };
                    for c in 0..3 {
                        let diff = (got[c] as i32 - o[base + c] as i32).unsigned_abs();
                        maxd = maxd.max(diff);
                        if diff != 0 {
                            miss += 1;
                        }
                    }
                }
                worst = worst.max(maxd);
                println!("lab/{fmt}: max diff {maxd} LSB, {miss}/{} channels off", n * 3);
            }
            // 16-bit
            let o: Vec<u16> = std::fs::read(d.join("oracle/labs_rgba16.f0.bin")).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let (mut maxd, mut miss) = (0u32, 0usize);
            for i in 0..n {
                let p = &rgba16[i * 4..i * 4 + 3];
                let got = combi.apply16([p[0], p[1], p[2]]);
                for c in 0..3 {
                    let diff = (got[c] as i32 - o[i * 4 + c] as i32).unsigned_abs();
                    maxd = maxd.max(diff);
                    if diff != 0 {
                        miss += 1;
                    }
                }
            }
            println!("lab/rgba16: max diff {maxd} LSB, {miss}/{} channels off", n * 3);
            worst = worst.max(maxd / 256); // 16-bit diffs scaled to 8-bit-equivalent for the bound
            // WIP gross-regression guard only (current residual ~15 is the integration bug, not a pass).
            assert!(worst <= 24, "Lab end-to-end residual grossly regressed: {worst}");
        }

        /// 'para' parametric TRC profiles (ICC func types 0–4): srgb→paraN (exercises paraN inverse)
        /// and paraN→srgb (exercises paraN forward).
        #[test]
        fn para_vs_oracle() {
            let Some(d) = dir() else { return };
            for ty in 0..5 {
                check(&d, &format!("s_p{ty}"), "srgb.icc", &format!("para{ty}.icc"));
                check(&d, &format!("p{ty}_s"), &format!("para{ty}.icc"), "srgb.icc");
            }
        }

        /// lut16 ('mft2') / lut8 ('mft1') A2B0 CLUT profiles → srgb (the CLUT interpolation engine).
        #[test]
        fn lut16_vs_oracle() {
            let Some(d) = dir() else { return };
            check(&d, "l16s", "lut16.icc", "srgb.icc");
        }

        #[test]
        fn lut8_vs_oracle() {
            let Some(d) = dir() else { return };
            check(&d, "l8s", "lut8.icc", "srgb.icc");
        }

        /// lut16 with a 17³ (non-power-of-2) grid → srgb: the CLUT engine's general-grid branch.
        #[test]
        fn lut17_vs_oracle() {
            let Some(d) = dir() else { return };
            check(&d, "l17s", "lut17.icc", "srgb.icc");
        }

        /// CMYK (`mft2`, 4→3, 9⁴ grid) → srgb (4→3 simplex CLUT resample): CMYK8 input. The
        /// 8-bit RGB outputs (rgb8/bgra8, alpha 0xFF) are **byte-exact**, validating the 4-input simplex
        /// cube build + per-pixel kernel. 16-bit output (rgba16) is near-exact (±1-LSB on ~11% of
        /// pixels): native has no specialised 4→3 16-bit kernel, so CMYK→RGBA16 goes through the generic
        /// generic per-pixel path whose rounding differs sub-LSB from the 16-bit
        /// output stage this port reuses.
        /// Diff `do_clut_pass4`'s CMYK PCS cube against the native `colortransforminterm` dump (the cube
        /// at the matrix cube pass, before the dst matrix). Env CMYK_INTERM = the `.interm.bin`
        /// (`[u32 nodes][nodes·3 u16]`). Prints the diff histogram to localise the simplex
        /// resample's low-bit rounding (the CMYK rgba16 residual). `#[ignore]`: needs the VM dump.
        #[test]
        #[ignore = "needs a VM colortransforminterm dump (CMYK_INTERM=...)"]
        fn cmyk_pcs_vs_interm() {
            let Some(d) = dir() else { return };
            let Ok(path) = std::env::var("CMYK_INTERM") else { return };
            let cmyk = cmyk_a2b(&std::fs::read(d.join("cmyk.icc")).unwrap()).unwrap();
            let nodes = GP4.pow(4);
            let mut cube = vec![0u16; nodes * 3];
            do_clut_pass4(&mut cube, &clut_input_elut(cmyk.gp), &cmyk, nodes);
            let raw = std::fs::read(path).unwrap();
            let nat: Vec<u16> = raw[4..].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            assert_eq!(nat.len(), nodes * 3, "interm node count");
            let (mut maxd, mut off, mut hist) = (0i32, 0usize, [0usize; 9]);
            for i in 0..nodes * 3 {
                let dd = (cube[i] as i32 - nat[i] as i32).abs();
                maxd = maxd.max(dd);
                if dd != 0 {
                    off += 1;
                    hist[(dd as usize).min(8)] += 1;
                }
            }
            println!("cmyk PCS cube: max diff {maxd}, {off}/{} words off; hist(1..8+)={:?}", nodes * 3, &hist[1..]);
            // Show the first few diffs with their cube index (to spot a pattern).
            for i in 0..nodes * 3 {
                if cube[i] != nat[i] {
                    println!("  first diff at word {i} (node {}, ch {}): mine {} native {}", i / 3, i % 3, cube[i], nat[i]);
                    break;
                }
            }
        }

        /// gray/mAB/Lab pre-dst-matrix cube vs the native `colortransforminterm` dumps (gray_i = gray
        /// PCS ramp at the matrix cube pass; mabc_i = mAB CLUT+black cube; lab_pre = the Lab-PCS cube at
        /// the Lab→XYZ conversion entry; lab_post = the XYZ cube after it). Localises the gray/mAB cube-
        /// resample rounding and whether the Lab residual is upstream (pre-Lab→XYZ) or the dst matrix.
        /// Env VMDUMP = the dir holding `{gray_i,mabc_i,lab_pre,lab_post}.interm.bin` (default target/vmdump).
        #[test]
        #[ignore = "needs VM colortransforminterm dumps in $VMDUMP"]
        fn interm_cubes_vs_vm() {
            let Some(d) = dir() else { return };
            let vm = std::path::PathBuf::from(std::env::var("VMDUMP").unwrap_or_else(|_| "target/vmdump".into()));
            let read_interm = |name: &str| -> Option<Vec<u16>> {
                let raw = std::fs::read(vm.join(name)).ok()?;
                Some(raw[4..].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
            };
            let hist = |mine: &[u16], nat: &[u16], tag: &str| {
                assert_eq!(mine.len(), nat.len(), "{tag} len");
                let (mut maxd, mut off, mut h, mut first) = (0i32, 0usize, [0usize; 9], None);
                for i in 0..mine.len() {
                    let dd = (mine[i] as i32 - nat[i] as i32).abs();
                    maxd = maxd.max(dd);
                    if dd != 0 {
                        off += 1;
                        h[(dd as usize).min(8)] += 1;
                        if first.is_none() { first = Some(i); }
                    }
                }
                println!("{tag}: max {maxd}, {off}/{} off; hist(1..8+)={:?}", mine.len(), &h[1..]);
                if let Some(i) = first {
                    let lo = i.saturating_sub(i % 3);
                    println!("  first diff node {} ch {}: mine {:?} nat {:?}", i / 3, i % 3, &mine[lo..lo + 3], &nat[lo..lo + 3]);
                }
            };
            // gray: the 128-node WHITE ramp (pre dst matrix).
            if let Some(nat) = read_interm("gray_i.interm.bin") {
                const WHITE: [u16; 3] = [31594, 32767, 27105];
                let mut mine = vec![0u16; GGP * 3];
                for k in 0..GGP {
                    let p = ((k as u64) << 16) / (GGP as u64 - 1);
                    for c in 0..3 { mine[k * 3 + c] = ((WHITE[c] as u64 * p) >> 16) as u16; }
                }
                hist(&mine, &nat, "gray ramp");
            }
            // 3→3 pre-dst-matrix builder: identity ramp → src CLUT pass (+ mAB M-matrix + V4 black) →
            // optional Lab→XYZ — i.e. `Combi::new` up to just before the dst matrix.
            let build33 = |prof: &str, lab: bool| -> Vec<u16> {
                let src = IccProfile::parse(&std::fs::read(d.join(prof)).unwrap()).unwrap();
                let nodes = GP * GP * GP;
                let mut cube = vec![0u16; (nodes + GP * GP + GP + 1) * 3];
                for r in 0..GP { for g in 0..GP { for bl in 0..GP {
                    let idx = (r * GP * GP + g * GP + bl) * 3;
                    cube[idx] = expand16(r as u32, 5) as u16;
                    cube[idx + 1] = expand16(g as u32, 5) as u16;
                    cube[idx + 2] = expand16(bl as u32, 5) as u16;
                }}}
                let (lin_e, lin_a) = ([linear_elut16(); 3], [linear_alut16(); 3]);
                let lut = src.lut.as_ref().unwrap();
                do_clut_pass(&mut cube, &clut_input_elut(lut.gp), lut, nodes);
                if let Some(m) = lut.mab_matrix { do_matrix_pass(&mut cube, &lin_e, &m, &lin_a, nodes); }
                if src.v4 { adjust_black_v4_to_legacy(&mut cube, nodes); }
                if lab { lab2xyz_cube16(&mut cube, nodes); }
                cube.truncate(nodes * 3);
                cube
            };
            if let Some(nat) = read_interm("mabc_i.interm.bin") { hist(&build33("mabc.icc", false), &nat, "mAB mabc cube"); }
            if let Some(nat) = read_interm("lab_pre.interm.bin") { hist(&build33("lab.icc", false), &nat, "lab pre-Lab2XYZ"); }
            if let Some(nat) = read_interm("lab_post.interm.bin") { hist(&build33("lab.icc", true), &nat, "lab post-Lab2XYZ"); }
        }

        #[test]
        fn cmyk_vs_oracle() {
            let Some(d) = dir() else { return };
            if !d.join("oracle/cmyks_rgb8.f0.bin").exists() {
                return; // cmyk oracle only in the merged suite / main checkout
            }
            let cmyk = cmyk_a2b(&std::fs::read(d.join("cmyk.icc")).unwrap()).unwrap();
            let dst = IccProfile::parse(&std::fs::read(d.join("srgb.icc")).unwrap()).unwrap();
            let combi = Combi4::new(&cmyk, &dst);
            let inp = std::fs::read(d.join("in_cmyk32.bin")).unwrap();
            let n = inp.len() / 4;
            let o = std::fs::read(d.join("oracle/cmyks_rgb8.f0.bin")).unwrap();
            for i in 0..n {
                let p = &inp[i * 4..i * 4 + 4];
                assert_eq!(combi.apply([p[0], p[1], p[2], p[3]]), o[i * 3..i * 3 + 3], "cmyk/rgb8 px {i}");
            }
            let o = std::fs::read(d.join("oracle/cmyks_bgra8.f0.bin")).unwrap();
            for i in 0..n {
                let p = &inp[i * 4..i * 4 + 4];
                let g = combi.apply([p[0], p[1], p[2], p[3]]);
                assert_eq!([g[2], g[1], g[0], 0xFF], o[i * 4..i * 4 + 4], "cmyk/bgra8 px {i}");
            }
            let o: Vec<u16> = std::fs::read(d.join("oracle/cmyks_rgba16.f0.bin")).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let (mut diffs, mut maxd) = (0, 0i32);
            for i in 0..n {
                let p = &inp[i * 4..i * 4 + 4];
                let g = combi.apply16([p[0], p[1], p[2], p[3]]);
                for c in 0..3 {
                    let dd = (g[c] as i32 - o[i * 4 + c] as i32).abs();
                    maxd = maxd.max(dd);
                    diffs += (dd != 0) as usize;
                }
                assert_eq!(o[i * 4 + 3], 0xFFFF);
            }
            eprintln!("cmyk/rgba16: {diffs} channel diffs, max {maxd} LSB over {} channels", n * 3);
            assert_eq!(diffs, 0, "cmyk/rgba16 not byte-exact: {diffs} diffs, max {maxd}");
        }

        /// `mAB ` (lutAToBType, ICC v4) A2B profiles → srgb: `mabc` (identity A, gamma-CLUT), `mab`
        /// (gamma A, linear·colorant CLUT), `mabm` (gamma A, identity CLUT + colorant·0.5 M-matrix).
        /// All V4 ⇒ the CMM strips the perceptual black point ([`adjust_black_v4_to_legacy`]), the key
        /// that takes the family from ~3× low-end over-attenuation to near-exact. Residual: ≤1-LSB on
        /// ≲1% of channels — a sub-LSB rounding in the engine's mAB CLUT resample shaper vs lut16's; the
        /// black-point, per-pixel ELUT and output ALUT are byte-exact.
        #[test]
        fn mab_vs_oracle() {
            let Some(d) = dir() else { return };
            for (tag, prof) in [("mabcs", "mabc.icc"), ("mabs", "mab.icc"), ("mabms", "mabm.icc")] {
                let src = IccProfile::parse(&std::fs::read(d.join(prof)).unwrap()).unwrap();
                let dst = IccProfile::parse(&std::fs::read(d.join("srgb.icc")).unwrap()).unwrap();
                let combi = Combi::new(&src, &dst);
                let bgra8 = std::fs::read(d.join("in_bgra8.bin")).unwrap();
                let rgb8 = std::fs::read(d.join("in_rgb8.bin")).unwrap();
                let rgba16: Vec<u16> = std::fs::read(d.join("in_rgba16.bin")).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                let ob = std::fs::read(d.join(format!("oracle/{tag}_bgra8.f0.bin"))).unwrap();
                let orgb = std::fs::read(d.join(format!("oracle/{tag}_rgb8.f0.bin"))).unwrap();
                let o16: Vec<u16> = std::fs::read(d.join(format!("oracle/{tag}_rgba16.f0.bin"))).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                let (mut diffs, mut d16, mut m16) = (0usize, 0usize, 0i32);
                for i in 0..bgra8.len() / 4 {
                    let p = &bgra8[i * 4..i * 4 + 4];
                    let g = combi.apply([p[2], p[1], p[0]]);
                    for c in 0..3 { let dd = (g[2 - c] as i32 - ob[i * 4 + c] as i32).abs(); assert!(dd <= 1, "{tag}/bgra8 px {i} ch {c}"); diffs += (dd != 0) as usize; }
                    let p = &rgb8[i * 3..i * 3 + 3];
                    let g = combi.apply([p[0], p[1], p[2]]);
                    for c in 0..3 { let dd = (g[c] as i32 - orgb[i * 3 + c] as i32).abs(); assert!(dd <= 1, "{tag}/rgb8 px {i} ch {c}"); diffs += (dd != 0) as usize; }
                    let p = &rgba16[i * 4..i * 4 + 3];
                    let g = combi.apply16([p[0], p[1], p[2]]);
                    for c in 0..3 { let dd = (g[c] as i32 - o16[i * 4 + c] as i32).abs(); m16 = m16.max(dd); d16 += (dd != 0) as usize; }
                }
                assert!(diffs <= 70, "{tag}: {diffs} ±1-LSB channel diffs (regressed)");
                // rgba16 goes through the same Combi::apply16 that cube_vs_oracle passes byte-exact on, so
                // this residual is the mAB CUBE-resample rounding (the ≤1-LSB 8-bit error ×256), not apply16.
                eprintln!("{tag}/rgba16: {d16} channel diffs, max {m16} LSB");
                assert!(m16 <= 90, "{tag}/rgba16 regressed beyond the cube-resample residual: max {m16}");
            }
        }

        /// Gray (`GRAY`/XYZ) 1→3 source → srgb (the gray→RGB device-link). Near-exact:
        /// 779/784 byte-exact, the ≤5 residuals are ±1-LSB in the 2-node→128 cube resample rounding (the
        /// exact 1→3 resample — a generic sub-path — is not yet reproduced; see the note).
        #[test]
        fn gray_vs_oracle() {
            let Some(d) = dir() else { return };
            let gray = gray_ktrc(&std::fs::read(d.join("gray.icc")).unwrap()).unwrap();
            let dst = IccProfile::parse(&std::fs::read(d.join("srgb.icc")).unwrap()).unwrap();
            let combi = GrayCombi::new(&gray, &dst);
            let o = std::fs::read(d.join("oracle/grays_rgb8.f0.bin")).unwrap();
            let gin = std::fs::read(d.join("in_gray8.bin")).unwrap();
            let mut diffs = 0;
            for (i, &g) in gin.iter().enumerate() {
                let got = combi.apply(g);
                for c in 0..3 {
                    let d = got[c] as i32 - o[i * 3 + c] as i32;
                    assert!(d.abs() <= 1, "gray {g} ch {c}: {} vs {}", got[c], o[i * 3 + c]);
                    diffs += (d != 0) as usize;
                }
            }
            assert!(diffs <= 5, "gray regressed: {diffs} channel diffs (was 5)");
            // 16-bit output (grays_rgba16, 8-bit gray input → 16-bit RGB via the generic 1→3 output stage).
            let o16: Vec<u16> = std::fs::read(d.join("oracle/grays_rgba16.f0.bin")).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let (mut d16, mut m16) = (0usize, 0i32);
            for (i, &g) in gin.iter().enumerate() {
                let got = combi.apply16(g);
                for c in 0..3 {
                    let dd = (got[c] as i32 - o16[i * 4 + c] as i32).abs();
                    m16 = m16.max(dd);
                    d16 += (dd != 0) as usize;
                }
                assert_eq!(o16[i * 4 + 3], 0xFFFF);
            }
            // 16-bit output exposes the gray CUBE-build residual (the ≤1-LSB 8-bit cube error, ×256 at
            // 16-bit): the generic 1→3 output stage above is byte-exact for an exact cube (cf. CMYK), so
            // this ≤54-LSB residual is the 128-node ramp / do_matrix_pass resample rounding, not apply16.
            eprintln!("gray/rgba16: {d16} channel diffs, max {m16} LSB over {} channels", gin.len() * 3);
            assert!(m16 <= 56, "gray/rgba16 regressed beyond the cube-build residual: max {m16}");
        }
    }
}
