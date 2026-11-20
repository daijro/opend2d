//! Variable-font instancing: `gvar` glyph-point interpolation at a normalized location.
//!
//! A variable font's named instance (or an explicit axis-value set) renders the `gvar`-interpolated
//! outline, not the default `glyf` master. We parse the tuple-variation data with `read-fonts` but do
//! the delta arithmetic ourselves in **f32**, byte-exactly matching DirectWrite's variation
//! interpolator (`dwrite.dll` `GlyphOutlineVariationInterpolator` / `TupleVariationInterpolator`,
//! win11-26100): the per-tuple scalar, the IUP interpolation of points a sparse tuple omits, and the
//! accumulation are all `f32`, where `read-fonts`' own `simple_deltas` works in `Fixed` (16.16) —
//! that precision difference was the instanced-outline residual vs native. The deltas are added to
//! the raw `glyf` points and rounded to integer font units, so the byte-exact contour builder in
//! [`super::raster`] (and the hinting scaler) still sees an integer glyf-like outline. `avar` (axis
//! value remapping) is honoured via `fvar::user_to_normalized`.

use skrifa::raw::tables::glyf::{SimpleGlyph, PHANTOM_POINT_COUNT};
use skrifa::raw::tables::gvar::GlyphDelta;
use skrifa::raw::tables::variations::TupleVariation;
use skrifa::raw::types::{F2Dot14, Fixed, GlyphId, Tag};
use skrifa::raw::TableProvider;
use skrifa::FontRef;

/// `fvar` user axis coordinates `(tag, value)` → the normalized F2Dot14 coordinates `gvar` consumes
/// (through the `avar` segment mapping when present). Empty when the font has no `fvar` table.
pub fn normalized_coords(font: &FontRef, user: &[(Tag, f32)]) -> Vec<F2Dot14> {
    let Ok(fvar) = font.fvar() else { return Vec::new() };
    let mut out = vec![F2Dot14::ZERO; fvar.axis_count() as usize];
    fvar.user_to_normalized(
        font.avar().ok().as_ref(),
        user.iter().map(|&(t, v)| (t, Fixed::from_f64(v as f64))),
        &mut out,
    );
    out
}

/// Normalized coords for a glyph run whose font face reports DWRITE axis values — `(packed_tag,
/// value)` with `packed_tag` DWRITE's little-endian 4-char tag — building the `FontRef` from the
/// run's bytes. Empty when the font is not variable or the values are the default (no instancing).
/// The FFI uses this to turn `IDWriteFontFace5::GetFontAxisValues` into the core `GlyphRun.coords`.
pub fn normalized_from_dwrite(font_bytes: &[u8], face_index: u32, axes: &[(u32, f32)]) -> Vec<F2Dot14> {
    let Ok(font) = FontRef::from_index(font_bytes, face_index) else { return Vec::new() };
    let user: Vec<(Tag, f32)> = axes.iter().map(|&(t, v)| (Tag::new(&t.to_le_bytes()), v)).collect();
    let n = normalized_coords(&font, &user);
    if n.iter().all(|c| *c == F2Dot14::ZERO) { Vec::new() } else { n }
}

/// A simple glyph's contour **and** phantom points `gvar`-interpolated at the normalized `coords`, as
/// integer font units in `s.points()` order followed by the four phantom points. `phantom` are the
/// base phantom points (x-side-bearing, advance, top, bottom) in font units, so the varied advance/
/// side bearings come out too (what the hinting path needs). `None` when the glyph has no `gvar` data
/// or `coords` is the default instance (the caller renders the unvaried base outline + base phantoms).
/// An outline-less glyph's four phantom points (font units) at the instance `coords`: its `gvar` phantom deltas
/// (a variable face's space narrows with its width axis), or `None` for the default instance or no variation.
pub fn instanced_phantoms_fu(font: &FontRef, gid: GlyphId, coords: &[F2Dot14], phantom: [[i32; 2]; 4]) -> Option<[[i32; 2]; 4]> {
    if coords.iter().all(|c| *c == F2Dot14::ZERO) {
        return None;
    }
    let gvd = font.gvar().ok()?.glyph_variation_data(gid).ok().flatten()?;
    let bx: Vec<f32> = phantom.iter().map(|p| p[0] as f32).collect();
    let by: Vec<f32> = phantom.iter().map(|p| p[1] as f32).collect();
    let cf: Vec<f32> = coords.iter().map(|c| c.to_f32()).collect();
    let (mut dx, mut dy) = (vec![0f32; PHANTOM_POINT_COUNT], vec![0f32; PHANTOM_POINT_COUNT]);
    for tuple in gvd.tuples() {
        let scalar = tuple_scalar_f32(&tuple, &cf);
        if scalar > 0.0 {
            accumulate_tuple_f32(&mut dx, &mut dy, &tuple, scalar, &bx, &by, &[], PHANTOM_POINT_COUNT);
        }
    }
    Some(std::array::from_fn(|i| [phantom[i][0] + round_dw(dx[i]), phantom[i][1] + round_dw(dy[i])]))
}

pub fn instanced_points_fu(font: &FontRef, gid: GlyphId, coords: &[F2Dot14], s: &SimpleGlyph, phantom: [[i32; 2]; 4]) -> Option<Vec<[i32; 2]>> {
    if coords.iter().all(|c| *c == F2Dot14::ZERO) {
        return None; // the default instance is the base outline
    }
    let gvar = font.gvar().ok()?;
    let gvd = gvar.glyph_variation_data(gid).ok().flatten()?;
    let npts = s.num_points();
    // `gvar`/IUP work over the glyph's points plus its four phantom points (advance/side bearings).
    // The base coordinates (font units, as f32) let IUP interpolate an omitted point's delta between
    // its neighbours' true positions; the phantom deltas vary the metrics.
    let count = npts + PHANTOM_POINT_COUNT;
    let mut bx = vec![0f32; count];
    let mut by = vec![0f32; count];
    for (i, p) in s.points().enumerate() {
        bx[i] = p.x as f32;
        by[i] = p.y as f32;
    }
    for (i, p) in phantom.iter().enumerate() {
        bx[npts + i] = p[0] as f32;
        by[npts + i] = p[1] as f32;
    }
    let contours: Vec<u16> = s.end_pts_of_contours().iter().map(|e| e.get()).collect();
    // DirectWrite accumulates the gvar deltas in f32 (see the module note); match it bit-for-bit.
    let cf: Vec<f32> = coords.iter().map(|c| c.to_f32()).collect();
    let mut dx = vec![0f32; count];
    let mut dy = vec![0f32; count];
    let mut any = false;
    for tuple in gvd.tuples() {
        let scalar = tuple_scalar_f32(&tuple, &cf);
        if scalar <= 0.0 {
            continue; // native keeps only strictly-positive region scalars
        }
        any = true;
        accumulate_tuple_f32(&mut dx, &mut dy, &tuple, scalar, &bx, &by, &contours, count);
    }
    if !any {
        return None;
    }
    // Final round of the accumulated f32 delta onto the integer outline (native `ApplyVariation`):
    // truncate toward zero, and only the exact +0.5 fractional case rounds up.
    Some((0..count).map(|i| [bx[i] as i32 + round_dw(dx[i]), by[i] as i32 + round_dw(dy[i])]).collect())
}

/// A composite glyph's per-component offset deltas plus the four phantom-point deltas,
/// `gvar`-interpolated at `coords`, as integer font-unit deltas (not absolute). `count` is the number
/// of components plus four. Composite `gvar` carries one delta per component offset and the phantom
/// points, with **no** IUP (interpolation is meaningless for component offsets, so an omitted
/// component's delta stays zero) — native `composite_deltas`. `None` for no `gvar` data / the default
/// instance. The caller adds `[..n_components]` to the component offsets and `[n_components..]` to the
/// composite's phantom points, in f32 like the simple path.
pub fn instanced_composite_deltas_fu(font: &FontRef, gid: GlyphId, coords: &[F2Dot14], count: usize) -> Option<Vec<[i32; 2]>> {
    if coords.iter().all(|c| *c == F2Dot14::ZERO) {
        return None;
    }
    let gvar = font.gvar().ok()?;
    let gvd = gvar.glyph_variation_data(gid).ok().flatten()?;
    let cf: Vec<f32> = coords.iter().map(|c| c.to_f32()).collect();
    let mut dx = vec![0f32; count];
    let mut dy = vec![0f32; count];
    let mut any = false;
    for tuple in gvd.tuples() {
        let scalar = tuple_scalar_f32(&tuple, &cf);
        if scalar <= 0.0 {
            continue;
        }
        any = true;
        // Dense covers every component; sparse references a subset — either way the delta is indexed
        // by component/phantom position, and the full `delta * scalar` f32 product accumulates.
        for d in tuple.deltas() {
            let i = d.position as usize;
            if i < count {
                dx[i] += d.x_delta as f32 * scalar;
                dy[i] += d.y_delta as f32 * scalar;
            }
        }
    }
    if !any {
        return None;
    }
    Some((0..count).map(|i| [round_dw(dx[i]), round_dw(dy[i])]).collect())
}

/// Native `ApplyVariation`'s final delta round: truncate toward zero, bump only the exact +0.5 frac
/// (`v = (int)f; if ((float)v - f == -0.5) ++v`). So 3.5→4 but 3.6→3, −3.5→−3.
fn round_dw(f: f32) -> i32 {
    let v = f as i32; // f32 as i32 truncates toward zero (saturating, which never trips for glyph deltas)
    if (v as f32) - f == -0.5 { v + 1 } else { v }
}

/// The OpenType tuple region scalar in f32 (native `TupleVariationInterpolator::ComputeTupleScalar`).
/// `coords` are the instance's normalized axis coordinates. 1.0 when no axis participates; 0.0 when
/// the location is outside the tuple's region (so the caller drops the tuple).
fn tuple_scalar_f32(tuple: &TupleVariation<GlyphDelta>, coords: &[f32]) -> f32 {
    let peak = tuple.peak();
    let istart = tuple.intermediate_start();
    let iend = tuple.intermediate_end();
    let has_inter = istart.is_some() && iend.is_some();
    let axis = |t: &Option<_>, i: usize| -> f32 {
        t.as_ref().and_then(|t: &skrifa::raw::tables::variations::Tuple| t.get(i)).map(|v| v.to_f32()).unwrap_or(0.0)
    };
    let mut scalar = 1.0f32;
    for i in 0..coords.len() {
        let pk = peak.get(i).map(|v| v.to_f32()).unwrap_or(0.0); // peak coord
        let c = coords[i]; // instance coord
        let (s0, e0) = if has_inter { (axis(&istart, i), axis(&iend, i)) } else { (0.0, 0.0) };
        // Sign-agreement flags native checks before a tuple's intermediate region applies.
        let mut end_sign = pk >= 0.0;
        if e0 < 0.0 { end_sign = false; } else if e0 > 0.0 { end_sign = true; }
        let mut start_sign = if s0 < 0.0 { false } else { pk >= 0.0 };
        if s0 > 0.0 { start_sign = true; }
        let participates = pk != 0.0 && (!has_inter || (start_sign == end_sign && pk >= s0 && e0 >= pk));
        if participates {
            if c == 0.0 {
                return 0.0; // peak set but instance at default on this axis → no contribution
            }
            let factor = if has_inter {
                if c < s0 || e0 < c {
                    return 0.0; // outside the intermediate region
                }
                if pk <= c {
                    if c > pk { (e0 - c) / (e0 - pk) } else { 1.0 }
                } else {
                    (c - s0) / (pk - s0)
                }
            } else {
                if (c < 0.0 && pk > c) || (c > 0.0 && c > pk) {
                    return 0.0; // opposite side of the peak, or beyond it
                }
                c / pk
            };
            scalar *= factor;
        }
    }
    scalar
}

/// Accumulate one active tuple's deltas into the f32 x/y accumulators, native-exactly. A dense tuple
/// (deltas for every point) and a sparse tuple's *referenced* points accumulate the full `delta *
/// scalar` product; a sparse tuple's *omitted* points are IUP-interpolated and their contribution is
/// truncated to an integer per tuple (native `AccumulateInterpolatedTupleDeltas[Internal]`).
#[allow(clippy::too_many_arguments)]
fn accumulate_tuple_f32(dx: &mut [f32], dy: &mut [f32], tuple: &TupleVariation<GlyphDelta>, scalar: f32, bx: &[f32], by: &[f32], contours: &[u16], n: usize) {
    if tuple.has_deltas_for_all_points() {
        for d in tuple.deltas() {
            let i = d.position as usize;
            if i < n {
                dx[i] += d.x_delta as f32 * scalar;
                dy[i] += d.y_delta as f32 * scalar;
            }
        }
        return;
    }
    let refs: Vec<GlyphDelta> = tuple.deltas().collect();
    if refs.is_empty() {
        return;
    }
    // IUP: fill the points this tuple does not reference, per contour (truncated per tuple).
    iup_f32(dx, bx, &refs, |d| d.x_delta as f32, contours, n, scalar);
    iup_f32(dy, by, &refs, |d| d.y_delta as f32, contours, n, scalar);
    // The referenced points take their explicit delta * scalar (no per-tuple truncation).
    for d in &refs {
        let p = d.position as usize;
        if p < n {
            dx[p] += d.x_delta as f32 * scalar;
            dy[p] += d.y_delta as f32 * scalar;
        }
    }
}

/// IUP for one axis: interpolate the points a sparse tuple omits, bracketed by the tuple's referenced
/// points within each contour (phantom points are in no contour, so they are only varied when the
/// tuple references them). `refs` are in ascending point order.
#[allow(clippy::too_many_arguments)]
fn iup_f32(out: &mut [f32], coord: &[f32], refs: &[GlyphDelta], sel: impl Fn(&GlyphDelta) -> f32, contours: &[u16], n: usize, scalar: f32) {
    let mut cstart = 0usize;
    for &cend_u in contours {
        let cend = cend_u as usize;
        if cend >= n {
            break;
        }
        // The referenced points that fall in this contour, in ascending order.
        let cref: Vec<usize> = (0..refs.len()).filter(|&k| { let p = refs[k].position as usize; p >= cstart && p <= cend }).collect();
        let m = cref.len();
        if m != 0 {
            for a in 0..m {
                interp_gap(out, coord, &refs[cref[a]], &refs[cref[(a + 1) % m]], &sel, cstart, cend, scalar);
            }
        }
        cstart = cend + 1;
    }
}

/// Interpolate the omitted points strictly between two consecutive reference points `r1`/`r2`
/// (cyclically, within the contour `[cstart, cend]`), native `AccumulateInterpolatedTupleDeltasInternal`:
/// a fused f32 mul/div between the references ordered by base coordinate, each contribution truncated.
#[allow(clippy::too_many_arguments)]
fn interp_gap(out: &mut [f32], coord: &[f32], r1: &GlyphDelta, r2: &GlyphDelta, sel: &impl Fn(&GlyphDelta) -> f32, cstart: usize, cend: usize, scalar: f32) {
    let (p1, p2) = (r1.position as usize, r2.position as usize);
    let (d1, d2) = (sel(r1), sel(r2));
    // Order the two references by base coordinate; the interpolation runs low→high.
    let (in_lo, in_hi, d_lo, d_hi) = if coord[p1] > coord[p2] { (coord[p2], coord[p1], d2, d1) } else { (coord[p1], coord[p2], d1, d2) };
    let degenerate = (in_hi - in_lo).abs() < 1e-8;
    let mut idx = p1;
    loop {
        idx = if idx == cend { cstart } else { idx + 1 };
        if idx == p2 {
            break; // reached the far reference (for a single-reference contour, p2 == p1 → full loop)
        }
        let c = coord[idx];
        let d = if degenerate {
            // References share a coordinate: shift by the common delta, else leave the point alone.
            if d_lo == d_hi { d_lo } else { 0.0 }
        } else if c <= in_lo {
            d_lo
        } else if c >= in_hi {
            d_hi
        } else {
            d_lo + (c - in_lo) * (d_hi - d_lo) / (in_hi - in_lo)
        };
        out[idx] += ((d * scalar) as i32 as i16) as f32;
    }
}

/// A simple glyph's contour points `gvar`-interpolated at `coords`, as `(x, y)` fractional (but
/// integer-valued) font-unit floats in `s.points()` order — the unhinted raster path. `None` for a
/// glyph with no `gvar` data / the default instance. See [`instanced_points_fu`].
pub fn instanced_simple(font: &FontRef, gid: GlyphId, coords: &[F2Dot14], s: &SimpleGlyph) -> Option<Vec<[f64; 2]>> {
    let npts = s.num_points();
    let pts = instanced_points_fu(font, gid, coords, s, [[0; 2]; 4])?;
    Some(pts[..npts].iter().map(|p| [p[0] as f64, p[1] as f64]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use skrifa::raw::tables::glyf::Glyph;
    use skrifa::{GlyphId, MetadataProvider};

    // A local variable font (not shipped): `D2D_UNAGEO` names its file (Unageo[wght].ttf), and the test
    // skips without it. Byte-exactness vs native is the VM step (bahnschrift); here we only prove gvar
    // interpolation activates and moves the outline.
    fn unageo() -> Option<Vec<u8>> {
        std::fs::read(std::env::var_os("D2D_UNAGEO")?).ok()
    }

    #[test]
    fn gvar_instancing_moves_the_outline() {
        let Some(bytes) = unageo() else { return };
        let font = FontRef::new(&bytes).unwrap();
        let glyf = font.glyf().unwrap();
        let loca = font.loca(None).unwrap();
        // The wght axis at its maximum (normalized +1.0).
        let coords = vec![F2Dot14::from_f32(1.0); font.fvar().unwrap().axis_count() as usize];
        // First glyph with gvar data and an outline that the heaviest weight actually varies.
        let mut moved = false;
        for gid in 1..font.maxp().unwrap().num_glyphs() {
            let gid = GlyphId::from(gid);
            let Some(Glyph::Simple(s)) = loca.get_glyf(gid, &glyf).ok().flatten() else { continue };
            let base: Vec<[f64; 2]> = s.points().map(|p| [p.x as f64, p.y as f64]).collect();
            if let Some(inst) = instanced_simple(&font, gid, &coords, &s) {
                assert_eq!(inst.len(), base.len());
                let max_delta = base.iter().zip(&inst).map(|(b, i)| (b[0] - i[0]).abs().max((b[1] - i[1]).abs())).fold(0.0f64, f64::max);
                if max_delta > 1.0 {
                    moved = true;
                    break;
                }
            }
        }
        assert!(moved, "gvar instancing at max wght must move at least one glyph's outline");
    }

    #[test]
    fn default_instance_is_the_base_outline() {
        let Some(bytes) = unageo() else { return };
        let font = FontRef::new(&bytes).unwrap();
        let glyf = font.glyf().unwrap();
        let loca = font.loca(None).unwrap();
        let zero = vec![F2Dot14::ZERO; font.fvar().unwrap().axis_count() as usize];
        let gid = GlyphId::from(5u16);
        if let Some(Glyph::Simple(s)) = loca.get_glyf(gid, &glyf).ok().flatten() {
            // All-zero coords = the default instance: no interpolation, so the base outline is used.
            assert!(instanced_simple(&font, gid, &zero, &s).is_none());
        }
    }
}
