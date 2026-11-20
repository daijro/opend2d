//! Phase 2D-2: DirectWrite's TrueType scan converter and `IDWriteGlyphRunAnalysis::CreateAlphaTexture`,
//! ported from DirectWrite. This turns a positioned glyph run's (hinted)
//! outlines into the coverage alpha texture Direct2D's text blend samples.
//!
//! [`grayscale`] produces the `DWRITE_TEXTURE_ALIASED_1x1` texture (one coverage byte per device
//! pixel, `TEXT_ANTIALIAS_MODE_GRAYSCALE` / the natural analysis modes); [`cleartype`] produces the
//! `DWRITE_TEXTURE_CLEARTYPE_3x1` texture (three colour-subpixel bytes per device pixel). Both are
//! verified byte-for-byte against native `dwrite-dump` captures (grid-fit disabled, so the input is
//! the unhinted `skrifa` outline); see `crates/d2d/tests/text_raster`.
//!
//! ## The native engine
//!
//! The font scaler rasterizes each glyph into a 1-bit bitmap at an oversampled resolution — **4× in
//! x and y** for grayscale (the gray-map box resolve handles each 4×4 block, `round(255·count/16)`), **6×
//! horizontal × 5× vertical** for ClearType (the classic Platt horizontal filter + the 5-tap vertical
//! filter resolve to three colour subpixels). The 1-bit bitmap is the classic Apple/Microsoft TrueType
//! scan converter, ported whole: outline → oversampled 26.6 → decompose → DDA → sweep.
//!
//! * **Coordinates.** Native rounds the transformed outline to 1/64 **device** pixel, then overscales
//!   ×factor (the outline overscale: `<<2` for grayscale), so oversampled coords are multiples of the
//!   factor ([`to_2614`]). Getting this grid right (not rounding at the finer oversampled grid) is what
//!   places edges where native does.
//! * **Decompose** ([`split_quad`]): split each quadratic at its x/y extrema
//!   (`CompDiv` round-half-away divide, [`compdiv_n`]) and midpoint-bisect until each chord ≤ 3200
//!   units, yielding monotone pieces (collinear ones become lines).
//! * **DDA** ([`calc_line`], [`calc_spline`]): a Bresenham / second-
//!   order forward-difference cell walk that marks, per oversampled scanline the piece spans, the
//!   crossing column — with the native entry/exit rounding (`(x+32)>>6` / `(x−33)`) and the precision
//!   shift (a native lookup table, [`prec_shift`]) that keeps the differences inside 32 bits.
//! * **Vertex topology** ([`check_horiz_topology`]): a piece endpoint exactly on
//!   a scanline centre is covered by neither adjacent piece (the entry/exit rounding leaves a one-row
//!   gap); the fill crossing is added from the 3-point window (the on-crossing `(x+31)>>6` / the off-crossing
//!   `(x+32)>>6`). A horizontal edge on a centre fills its span only through that window at its corners,
//!   which depends on its direction; the edge itself marks no row crossing.
//! * **Sweep** ([`scan_bitmap`]): per row, nonzero-winding the crossings and fill
//!   columns `[on, off−1]`.
//! * **Bounds** ([`extrema_box`] + `GetAlphaTextureBounds` [`device_bounds`]): the
//!   oversampled box of *all* control points, `left/top = (min+31)>>6`, `right/bottom = (max+32)>>6`
//!   (zero-width axis bumped +1), divided to device pixels (`⌊/factor⌋` … `⌈/factor⌉`, ClearType
//!   widening x by ±2 first). This control-point box — not a tight coverage box — reproduces native's
//!   bounds exactly, including the zero-coverage border column `a` keeps at 13 px.
//!
//! ## Dropout control
//!
//! The fill-bitmap pass finishes an antialiased glyph with dropout recovery
//! unless the mode disables it. A *dropout* is a zero-width rank-paired span — an
//! on-crossing column equal to its off-crossing column in a row (a dropped vertical stem) or row==row
//! in a column (a curve bowl tangent to a scanline) — and it is recovered by setting one sample. [`Dropout`]
//! ports both passes: the horizontal and vertical dropout, the smart
//! neighbour-crossing test, and the already-set-bit guard.
//! The per-column scan the vertical pass needs is the horizontal scan run on the `transpose`d
//! contours (native's vertical scan is the x↔y transpose). The mode comes from
//! [`dropout_mode`]: the `SCANTYPE` value when `SCANCTRL` enables dropout, else "no dropout" — for
//! hinted text the final glyph program state includes composite child mode merging;
//! grid-fit-disabled glyphs use [`DEFAULT_DROPOUT`].
//!
//! Position: modes with `scan_type & 4` recover at the mean exact edge intercept. The native
//! `CalcHorizLineSubpix`/`CalcVertLineSubpix` callbacks use integer division; spline callbacks bisect
//! the monotone quadratic in 26.6; topology endpoints use their coordinate directly. Other modes
//! recover the preceding sample (`on−1`).
//!
//! ## Residual — what the ≤1-sample gaps actually are
//!
//! Bounds are byte-exact on every fixture; `H`/`e20`/`g20`/`R40`/`o20`/`Q20` are byte-exact in the
//! hinted path. Two tracked floors remain, both outside the fill arithmetic (verified against native
//! with this port in place):
//!
//! * **The curve-tangent floor (hinted `g20`/`e20`/`R40`) — FIXED.** It was the *implied on-curve
//!   midpoint* between two consecutive off-curve controls. Native synthesises it as `(a+b+1)>>1` in
//!   the overscaled integer grid (the glyph scan converter, after the outline overscale's `<<2`); averaging in
//!   device f64 and re-rounding to 1/64 device px instead loses the half-oversample-unit native keeps
//!   (an odd os-coordinate sum lands at `≡2 mod 4`, not `≡0`), shifting a tangent crossing by one 4×4
//!   sub-sample. Computing it in os-integer space ([`outline_contour_segs`]) made those exact and took
//!   textx-glyph from 81 to 176 byte-exact. (The transform is already bit-exact: the scaler
//!   does `(M₁₆.₁₆·font + 0x8000) >> 16`, identical to [`to_2614`] for the upem-2048
//!   fixtures; and [`calc_line`]/[`calc_spline`] match the native line/spline fill marks —
//!   hand-traced crossing-for-crossing on the right arc, where our DDA reproduces the sample-centre
//!   rule exactly for the given geometry.)
//! * **The 2-sample gaps were the fill — FIXED by [`x_on_center`].** Verified against
//!   native's exact glyph-scan-converter contour list (grid-fit oracle, from native's contour dump): fed native's
//!   own hinted points, our coverage was still 1 sample short at a vertical stem whose edge is a `Quad`
//!   with the control on the line (`p0.x==c.x==p1.x`, `x ≡ 32 mod 64`). The X-on-centre rule matched
//!   only `Seg::Line`, missing that quad-encoded stem; matching it too makes them byte-exact.
//! * **Thin-stem scenes (e.g. `textx-glyph-512`, Tahoma `#` at 27 px).** Dropout runs with the right
//!   mode (`SCANTYPE 5`) but finds no dropped span: our hinted stem edges don't round to the same
//!   oversampled column as native's — again the hinting placement, upstream.
//!
//! The hinting (grid-fitting) that adjusts the outline before scan conversion is done by
//! [`super::hinting`]; the hinted path hands its hinted **quadratics** to [`coverage_outline`], which
//! decomposes them with the same `split_quad`/`calc_spline` as the unhinted [`grayscale`]/[`cleartype`].
//! (Flattening those curves to polylines first — the earlier hinted glue — was the dominant
//! grayscale-coverage residual on curved glyphs; feeding quadratics took textx-glyph from 21 to 81.)

use super::GlyphRun;
mod fillsubrect; // d2d1 GlyphRunFiller::FillSubrect coverage-resolve tables (port in progress)
use skrifa::instance::{LocationRef, Size};
use skrifa::outline::{DrawSettings, OutlinePen};
use skrifa::raw::tables::glyf::{Anchor, CompositeGlyphFlags, Glyf, Glyph};
use skrifa::raw::tables::loca::Loca;
use skrifa::raw::TableProvider;
use skrifa::{FontRef, GlyphId, MetadataProvider};

/// A grayscale coverage alpha texture: a device-pixel rect `[left, top, right, bottom]` and one
/// coverage byte per pixel, row-major (`width · height` bytes), matching native `CreateAlphaTexture`
/// for `DWRITE_TEXTURE_ALIASED_1x1`. ClearType uses 3 bytes/pixel (`data.len() = 3·width·height`).
#[derive(Clone, Debug)]
pub struct AlphaTexture {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
    pub data: Vec<u8>,
}

impl AlphaTexture {
    pub fn width(&self) -> i32 {
        self.right - self.left
    }
    pub fn height(&self) -> i32 {
        self.bottom - self.top
    }
    pub(crate) fn empty() -> Self {
        AlphaTexture { left: 0, top: 0, right: 0, bottom: 0, data: Vec::new() }
    }
}

/// Synthetic-bold embolden amount for `DWRITE_FONT_SIMULATIONS_BOLD` at an integer device `ppem`
/// (upright, no rotation), from the scaler's initialization: the per-mille
/// bold strength is 20, `boldY = (ppem·20 − 10) / 1000` (truncated, ≥ 0 here) and `boldX = boldY + 1`.
/// Verified against native regular-vs-bold Impact aliased captures at em 16/24/48/52/56/58/60/64/96/128.
pub fn bold_amount(ppem: i32) -> (i32, i32) {
    let bold_y = ((ppem * 20 - 10) / 1000).max(0);
    (bold_y + 1, bold_y)
}

/// The bitmap embolden turned with the glyph, as the scaler's emboldening rotation factor turns it: a flip
/// mirrors each axis's growth, a quarter turn moves `boldX` onto device y and `boldY` onto device x. Returns
/// `(x, y)` signed device pixels, x right and y up positive; any other matrix keeps the upright directions.
/// Only the aliased bitmap dilates under a quarter turn; ClearType and grayscale embolden the outline there.
pub fn bold_dir(dt: &[f64; 6], (bx, by): (i32, i32)) -> (i32, i32) {
    let sign = |v: f64| if v < 0.0 { -1 } else { 1 };
    if dt[1] == 0.0 && dt[2] == 0.0 {
        (sign(dt[0]) * bx, sign(dt[3]) * by)
    } else if dt[0] == 0.0 && dt[3] == 0.0 {
        // The scaler's quarter-turn factors on the aliased bitmap: boldX down when m12 > 0 (up otherwise), boldY left
        // when m21 > 0.
        // ponytail: VM-checked on device y only (r25 30351: Leelawadee bold vertical text, 71 -> 37 px); the x arm
        // is mirrored to match.
        (-sign(dt[2]) * by, -sign(dt[1]) * bx)
    } else {
        (bx, by)
    }
}

/// Mirror a `w`×`h` cell grid left-right and/or top-bottom.
fn mirror<T: Copy>(v: &[T], w: usize, h: usize, fx: bool, fy: bool) -> Vec<T> {
    (0..w * h).map(|i| v[(if fy { h - 1 - i / w } else { i / w }) * w + if fx { w - 1 - i % w } else { i % w }]).collect()
}

/// Embolden the aliased mono texture (the aliased embolden): dilate the glyph right by `bold_x` and
/// up by `bold_y` — each output pixel is the OR of the `bold_x` pixels to its left and the `bold_y`
/// pixels below it (native's y-down device bitmap; "up" = toward smaller y). The texture grows to fit
/// (`right += bold_x`, `top -= bold_y`; left/bottom fixed), matching native's margined glyph bitmap. One
/// coverage byte per pixel (aliased is 0/255); any non-zero source pixel emboldens to 255.
pub fn embolden_aliased(tex: &AlphaTexture, bold_x: i32, bold_y: i32) -> AlphaTexture {
    // Left or down (a flipped or turned glyph, [`bold_dir`]): embolden the mirrored texture and mirror it back.
    if (bold_x < 0 || bold_y < 0) && !tex.data.is_empty() {
        let (w, h) = (tex.width() as usize, tex.height() as usize);
        let (fx, fy) = (bold_x < 0, bold_y < 0);
        let m = AlphaTexture { data: mirror(&tex.data, w, h, fx, fy), ..tex.clone() };
        let e = embolden_aliased(&m, bold_x.abs(), bold_y.abs());
        let (gx, gy) = (bold_x.abs(), bold_y.abs());
        let (l, r) = if fx { (tex.left - gx, tex.right) } else { (tex.left, tex.right + gx) };
        let (t, b) = if fy { (tex.top, tex.bottom + gy) } else { (tex.top - gy, tex.bottom) };
        return AlphaTexture { left: l, top: t, right: r, bottom: b, data: mirror(&e.data, e.width() as usize, e.height() as usize, fx, fy) };
    }
    let (bx, by) = (bold_x.max(0), bold_y.max(0));
    if bx == 0 && by == 0 || tex.data.is_empty() {
        return tex.clone();
    }
    let (w, h) = (tex.width(), tex.height());
    let (nw, nh) = (w + bx, h + by);
    let mut data = vec![0u8; (nw * nh) as usize];
    for orow in 0..nh {
        for ocol in 0..nw {
            // Source box: input cols [ocol-bx, ocol], input rows [orow-by, orow] shifted up by `by`
            // (new top is `by` rows above the old top, so new row `orow` is old row `orow - by`).
            let mut ink = false;
            'box_: for dy in 0..=by {
                let irow = orow - by + dy;
                if irow < 0 || irow >= h {
                    continue;
                }
                for dx in 0..=bx {
                    let icol = ocol - dx;
                    if icol < 0 || icol >= w {
                        continue;
                    }
                    if tex.data[(irow * w + icol) as usize] != 0 {
                        ink = true;
                        break 'box_;
                    }
                }
            }
            if ink {
                data[(orow * nw + ocol) as usize] = 255;
            }
        }
    }
    AlphaTexture { left: tex.left, top: tex.top - by, right: tex.right + bx, bottom: tex.bottom, data }
}

/// Grayscale coverage (`DWRITE_TEXTURE_ALIASED_1x1`): scan-convert the outline at 4× oversample and
/// resolve each 4×4 block, `round(255·count/16)`. `dt` is the device transform (world ∘ DPI,
/// `[m11,m12,m21,m22,dx,dy]`); the run's baseline origin, advances and offsets are DIPs.
///
/// Grid-fit-disabled glyphs carry no `prep`-set `SCANCTRL`/`SCANTYPE`, so the scan converter runs with
/// the scaler's [`DEFAULT_DROPOUT`] mode (the scaler's default and
/// confirmed against the oracle captures).
pub fn grayscale(dt: &[f64; 6], run: &GlyphRun, font: &FontRef, dropout: u16) -> AlphaTexture {
    unhinted_yup(dt, 1, |up| resolve_grayscale(glyph_run_arcs(up, run, font, 4, 4, 4), dropout))
}

/// Scan an unhinted run in native's y-up frame: `resolve` gets the y-negated device transform, and its
/// texture (`bpp` bytes per pixel) is flipped back to device rows. The implied-midpoint rounding, the
/// scan's entry/exit column rounding and the on-centre topology are all y-asymmetric, so a tangent or
/// edge exactly on a sample centre resolves as native only in that frame (as in [`coverage_outline`]).
fn unhinted_yup(dt: &[f64; 6], bpp: usize, resolve: impl FnOnce(&[f64; 6]) -> AlphaTexture) -> AlphaTexture {
    let mut tex = resolve(&[dt[0], -dt[1], dt[2], -dt[3], dt[4], -dt[5]]);
    flip_rows(&mut tex, bpp);
    tex
}

/// Flip a texture scanned in the y-up frame back to device rows (`bpp` bytes per pixel).
fn flip_rows(tex: &mut AlphaTexture, bpp: usize) {
    let (w, h) = (tex.width() as usize, tex.height() as usize);
    tex.data = (0..h).flat_map(|r| tex.data[(h - 1 - r) * w * bpp..(h - r) * w * bpp].to_vec()).collect();
    (tex.top, tex.bottom) = (-tex.bottom, -tex.top);
}

/// The dropout-control mode word the scan converter uses (native's per-glyph dropout
/// word): the `SCANTYPE` value 0-6 when `SCANCTRL` enables dropout, else `2` ("no dropout
/// control"). Bits, read from the native dropout routines: bit1 = no
/// dropout, bit0 = smart (neighbour-crossing test), bit2 = vertical pass + exact-intercept position.
pub fn dropout_mode(enabled: bool, scan_type: i32) -> u16 {
    if enabled { (scan_type & 0xFFFF) as u16 } else { 2 }
}

/// Dropout mode for grid-fit-disabled glyphs (no `prep` runs, so no `SCANCTRL`). The captures show
/// native applies **no** dropout here: enabling any dropout mode only adds samples the `o`/`g`/`R`
/// fixtures don't have (a regression), so the "no dropout" mode is what matches — their ≤1-sample
/// residual is scan precision at curve tangents, not dropout (see the module "Residual" note).
pub const DEFAULT_DROPOUT: u16 = 2;

/// Which coverage an analysis mode needs: grayscale (`ALIASED_1x1`, 4× resolve), ClearType
/// (`CLEARTYPE_3x1`, 6×5 resolve) or aliased (1 sample/pixel, binary).
#[derive(Clone, Copy)]
pub enum Kind {
    Grayscale,
    ClearType,
    /// ClearType without vertical antialiasing (6×1): the GDI measuring modes and plain NATURAL.
    /// Same horizontal Platt filter, single device-row vertical sample — native's horizontal-only CT.
    ClearTypeH,
    Aliased,
}

/// Scan-convert a hinted glyph's **quadratic** contours — device-space points with their on-curve
/// flags, as [`super::hinting::HintedOutline`] exposes them — keeping the curves so the scan converter
/// decomposes them with the same native `split_quad`/`calc_spline` path the unhinted [`grayscale`]/
/// [`cleartype`] use. This is the hinted back end of `text::render_glyph_run`: unlike [`coverage`],
/// which takes polylines already flattened off the native grid, the only difference from native then
/// is the hinted point positions. Each contour is `(points, on_curve)`, parallel and in device pixels.
pub fn coverage_outline(contours: &[(Vec<[f64; 2]>, Vec<bool>)], mode: u16, kind: Kind, bold: (i32, i32)) -> AlphaTexture {
    // EVERY mode scans in native's y-up frame (§10e–§10h): the implied-midpoint rounding `(a+b+1)>>1`,
    // the scan's y-asymmetric entry/exit column rounding, and `check_horiz_topology` are only correct
    // when the glyph is scanned y-up (native `fs_ContourScan`), and a hinted outline puts through-
    // vertices and edges exactly on sample centres where the y orientation flips the On/Off decision.
    // So negate y before building segs (forming the midpoint on native's coordinates) and flip the
    // resolved rows back — one root for all modes. Grayscale proved it first (commit 2cc120d); CT and
    // aliased share the same scan, so they get the same framing (native covers ~1 sub-sample more at an
    // on-centre edge in the y-down frame, never shifted). No-op wherever the scan is y-symmetric, so the
    // byte-exact CT/aliased fixtures are unchanged.
    // ClearType bold (the sub-pixel embolden) and aliased bold (post-pass `embolden_aliased`) are handled
    // elsewhere; `bold` here drives only the grayscale count-buffer embolden.
    match kind {
        Kind::Grayscale => coverage_box(contours, mode, 4, 4, bold, false),
        Kind::ClearType => yup_scan(contours, 6, 5, 3, |segs| resolve_cleartype(segs, mode, &CT_VWEIGHTS, bold)),
        Kind::ClearTypeH => yup_scan(contours, 6, 1, 3, |segs| resolve_cleartype(segs, mode, &[1], bold)),
        Kind::Aliased => yup_scan(contours, 1, 1, 1, |segs| resolve_aliased(segs, mode)),
    }
}

/// CPU glyph-cache leaf scan. DirectWrite's fsc boundary rules differ from WARP's text fill.
/// Reproduces DirectWrite's glyph-fill boundary rules, verified byte-identical.
pub(crate) fn coverage_colr(contours: &[(Vec<[f64; 2]>, Vec<bool>)], mode: u16, kind: Kind, nudge: bool) -> AlphaTexture {
    // D2D_LEAFCONTOURS: dump the (device-space, post-hint+embold) leaf outline for the outline-vs-scan
    // split (30066); pair with D2D_LEAFLIST by call order. Inert without the env var.
    if std::env::var_os("D2D_LEAFCONTOURS").is_some() {
        let ends: Vec<usize> = contours.iter().map(|(p, _)| p.len()).collect();
        let pts: Vec<[f64; 2]> = contours.iter().flat_map(|(p, _)| p.iter().copied()).collect();
        let on: Vec<u8> = contours.iter().flat_map(|(_, o)| o.iter().map(|&b| b as u8)).collect();
        eprintln!("LEAFCONTOURS n={} ends={:?} on={:?} pts={:?}", contours.len(), ends, on, pts);
    }
    if !matches!(kind, Kind::Grayscale) { return coverage_outline(contours, mode, kind, (0, 0)); }
    let contours: Vec<_> = contours.iter().map(|(pts, on)| {
        let mut out: Vec<([f64; 2], bool)> = Vec::new();
        for (&p, &on) in pts.iter().zip(on) {
            if out.last().is_some_and(|v| v.0 == p) {
                out.last_mut().unwrap().1 = true;
            } else { out.push((p, on)); }
        }
        if out.len() > 1 && out.first().unwrap().0 == out.last().unwrap().0 {
            out.remove(0); out.last_mut().unwrap().1 = true;
        }
        out.into_iter().unzip()
    }).collect();
    coverage_box_inner(&contours, mode, 4, 4, (0, 0), false, false, nudge)
}

/// Scan `contours` in native's y-up frame: negate y, build the oversampled segs, run `resolve`, then
/// flip the resolved rows back into the device texture (`bpp` bytes per pixel). The y-up counterpart of
/// the inline negate/flip in [`coverage_box`], shared by the ClearType and aliased resolves (§10h).
fn yup_scan(
    contours: &[(Vec<[f64; 2]>, Vec<bool>)],
    sx: i32,
    sy: i32,
    bpp: usize,
    resolve: impl FnOnce(Vec<Vec<Seg>>) -> AlphaTexture,
) -> AlphaTexture {
    let neg: Vec<Vec<Seg>> = contours
        .iter()
        .map(|(pts, on)| {
            let np: Vec<[f64; 2]> = pts.iter().map(|p| [p[0], -p[1]]).collect();
            // `fine`: hinted points are 1/64-device so the shape is unchanged, and ClearType's 1/6-px
            // sub-pixel phase stays exact instead of truncating to 1/64 device (see `to_2614`).
            outline_contour_segs(&np, on, sx, sy, true)
        })
        .collect();
    let mut tex = resolve(neg);
    flip_rows(&mut tex, bpp);
    tex
}

/// Each glyph's oversampled bitmap rect `[l, t, r, b]` (y-up, oversampled units) on the `sx`×`sy` scan grid,
/// `None` for an empty glyph.
pub(crate) fn glyph_os_boxes(contours: &[(Vec<[f64; 2]>, Vec<bool>)], glyphs: &[(std::ops::Range<usize>, u16)], sx: i32, sy: i32, fine: bool) -> Vec<Option<[i32; 4]>> {
    glyphs.iter().map(|(r, _)| {
        let segs: Vec<_> = contours.get(r.clone()).unwrap_or(&[]).iter().map(|(pts, on)| {
            let pts: Vec<_> = pts.iter().map(|p| [p[0], -p[1]]).collect();
            outline_contour_segs(&pts, on, sx, sy, fine)
        }).collect();
        extrema_box(&segs)
    }).collect()
}

// Native's 2×2 downsample: checkerboard majority of cached 4×4 bits.
pub(crate) fn coverage_downsampled(
    contours: &[(Vec<[f64; 2]>, Vec<bool>)], glyphs: &[(std::ops::Range<usize>, u16)], phases: &[bool], align_to_run: bool, widen: bool,
) -> AlphaTexture {
    let segs: Vec<_> = contours.iter().map(|(pts,on)| {
        let pts: Vec<_> = pts.iter().map(|p| [p[0],-p[1]]).collect();
        outline_contour_segs(&pts,on,4,4,false)
    }).collect();
    downsample_segs(&segs, glyphs, phases, align_to_run, widen)
}

/// [`coverage_downsampled`] on y-up 4×4 segs, `glyphs` indexing them.
fn downsample_segs(segs: &[Vec<Seg>], glyphs: &[(std::ops::Range<usize>, u16)], phases: &[bool], align_to_run: bool, widen: bool) -> AlphaTexture {
    let Some(os) = extrema_box(segs) else { return AlphaTexture::empty() };
    let [l,t,r,b] = device_bounds(os,4,4,false);
    if r <= l || b <= t { return AlphaTexture::empty(); }
    let (w,h)=((r-l) as usize,(b-t) as usize);
    // The merged brush path (`align_to_run`) scans every glyph into the shared run frame and
    // re-aligns each glyph's 2×2 dither checkerboard by its device-pixel extrema parity. This applies
    // ONLY to a MULTI-glyph run: with one glyph the run frame IS the glyph's own analysis frame (as the
    // solid per-glyph path, which passes `align_to_run=false`), so no re-alignment runs and a lone glyph
    // whose extrema parity is odd must NOT be flipped (fuzz 22837 'x' over-covered 3 stroke tips; the
    // flip is right for the multi-glyph runs — recap9 1483 'fjord', r8 9089 'The quick br').
    let multi = glyphs.len() > 1;
    let mut reduced=vec![false;w*h*4];
    for (gi,(range,mode)) in glyphs.iter().enumerate() {
        let Some(c)=segs.get(range.clone()) else { continue };
        let Some([glyph_left,_,_,glyph_top])=extrema_box(c) else { continue };
        let x_shift=(glyph_left & 1) as usize;
        let phase=phases.get(gi).copied().unwrap_or(false)
            ^ (align_to_run && multi && ((glyph_left>>1)+((glyph_top+1)>>1))&1!=0);
        let mut bits=scan_glyph_bitmaps(c,4*l,4*r,4*t,4*b,*mode,Some(&[(0..c.len(),*mode)]),true,0,false);
        x_on_center(&mut bits,c,4*l,4*r,4*t,4*b);
        sloped_on_center(&mut bits,c,4*l,4*r,4*t,4*b);
        for y in 0..h*2 {
            for x in 0..w*2 {
                let i=y*2*w*4+x*2+x_shift;
                let bit=|off: usize| bits.get(off).copied().unwrap_or(false) as u8;
                let count=bit(i)+bit(i+1)+bit(i+w*4)+bit(i+w*4+1);
                let threshold=2+(((x+y)&1!=0) ^ phase) as u8;
                reduced[y*w*2+x] |= count>=threshold;
            }
        }
    }
    // A thin face's cached bitmap packs every filled span one sample longer ([`resolve_box_glyphs`]' merge dilate),
    // here on the reduced 2×2 grid (VM: Courier New italic at ppem 93, NATURAL_SYMMETRIC_DOWNSAMPLED; on the 4×4
    // grid before the reduce it falls a level short at every right edge).
    if widen {
        for row in reduced.chunks_mut(w * 2) {
            for x in (1..row.len()).rev() {
                if row[x - 1] {
                    row[x] = true;
                }
            }
        }
    }
    let mut data=vec![0;w*h];
    for y in 0..h {
        for x in 0..w {
            let i=y*2*w*2+x*2;
            let count=reduced[i] as u32+reduced[i+1] as u32+reduced[i+w*2] as u32+reduced[i+w*2+1] as u32;
            data[(h-1-y)*w+x]=((count*255+2)/4) as u8;
        }
    }
    AlphaTexture {left:l,top:-b,right:r,bottom:-t,data}
}

/// Scan glyphs with their post-program dropout modes, then merge coverage before filtering.
/// Native dwrite's contour scan / run-bitmap fill retain this per-glyph state.
pub(crate) fn coverage_glyphs(
    contours: &[(Vec<[f64; 2]>, Vec<bool>)],
    glyphs: &[(std::ops::Range<usize>, u16)],
    kind: Kind, sx: i32, sy: i32, bold: (i32, i32), dilate: bool,
) -> AlphaTexture {
    let ct = matches!(kind, Kind::ClearType | Kind::ClearTypeH);
    let neg = contours.iter().map(|(pts, on)| {
        let pts: Vec<_> = pts.iter().map(|p| [p[0], -p[1]]).collect();
        outline_contour_segs(&pts, on, sx, sy, ct)
    }).collect();
    let mut tex = if ct {
        resolve_cleartype_glyphs(neg, 2, if sy == 1 { &[1] } else { &CT_VWEIGHTS }, Some(glyphs), bold, dilate)
    } else if matches!(kind, Kind::Aliased) {
        resolve_aliased_glyphs(neg, 2, Some(glyphs))
    } else {
        resolve_box_glyphs(neg, 2, sx, sy, bold, dilate, true, Some(glyphs))
    };
    let (w, h, bpp) = (tex.width() as usize, tex.height() as usize, if ct { 3 } else { 1 });
    tex.data = (0..h).flat_map(|r| tex.data[(h - 1 - r) * w * bpp..(h - r) * w * bpp].to_vec()).collect();
    (tex.top, tex.bottom) = (-tex.bottom, -tex.top);
    tex
}

/// The raw oversampled coverage bitmap of hinted device-space quadratic contours at an arbitrary
/// `sx`×`sy`: the device-pixel bounds `[l, t, r, b]` and the `sx·width × sy·height` boolean sample
/// grid (row-major), using the same scan converter [`coverage_outline`] feeds. The block resolve is
/// the caller's — exposed for the GDI bitmap render target's 8×1 / 6×1 oversamples
/// ([`crate::text::gdi_raster`]), which the fixed `Kind` resolves don't cover. `None` if empty.
pub fn coverage_bitmap(contours: &[(Vec<[f64; 2]>, Vec<bool>)], mode: u16, sx: i32, sy: i32) -> Option<([i32; 4], Vec<bool>)> {
    let segs: Vec<Vec<Seg>> = contours.iter().map(|(pts, on)| outline_contour_segs(pts, on, sx, sy, false)).collect();
    let dev = device_bounds(extrema_box(&segs)?, sx, sy, false);
    let bits = scan_bitmap(&segs, dev[0] * sx, dev[2] * sx, dev[1] * sy, dev[3] * sy, mode);
    Some((dev, bits))
}

/// Grayscale coverage at an arbitrary `sx`×`sy` oversample grid — the generalisation of
/// [`resolve_grayscale`]'s 4×4 box. DirectWrite picks the grid per rendering mode from
/// A native mode table `[8·aa + mode]` (low nibble = `sx`, next = `sy`): grayscale NATURAL and the GDI
/// modes use **8×1** (horizontal-only, no vertical AA), NATURAL_SYMMETRIC the 4×4 of
/// [`coverage_outline`]. The resolve is the same unweighted box, `round(255·count/(sx·sy))`.
pub fn coverage_box(contours: &[(Vec<[f64; 2]>, Vec<bool>)], mode: u16, sx: i32, sy: i32, bold: (i32, i32), dilate: bool) -> AlphaTexture {
    coverage_box_inner(contours, mode, sx, sy, bold, dilate, true, false)
}

#[allow(clippy::too_many_arguments)]
fn coverage_box_inner(contours: &[(Vec<[f64; 2]>, Vec<bool>)], mode: u16, sx: i32, sy: i32, bold: (i32, i32), dilate: bool, boundary: bool, nudge: bool) -> AlphaTexture {
    // Native scan-converts the glyph in its y-up frame (`fs_ContourScan`) and flips the 1-bit bitmap
    // into the device texture (the row-sweep bitmap fill writes row 0 = top). The implied on-curve midpoint
    // `(a+b+1)>>1` (round half up) and the scan's y-asymmetric entry/exit rounding do not commute with
    // a y-flip, so to reproduce native we negate y *before* building segs (forming the midpoint on
    // native's y-up coordinates) and flip the resolved rows back. No-op wherever the scan is
    // y-symmetric; with the `SCANTYPE & 4` exact-intercept dropout ([`Dropout`]) this makes the 8×1
    // curve glyphs (Consolas '0'/'6'/'R', the e-acute class) byte-exact — see the raster task notes.
    let neg: Vec<Vec<Seg>> = contours
        .iter()
        .map(|(pts, on)| {
            let np: Vec<[f64; 2]> = pts.iter().map(|p| [p[0], -p[1]]).collect();
            let mut segs = outline_contour_segs(&np, on, sx, sy, false);
            translate_segs(&mut segs, [nudge as i32, 0]);
            segs
        })
        .collect();
    let mut tex = resolve_box(neg, mode, sx, sy, bold, dilate, boundary);
    let (w, h) = (tex.width() as usize, tex.height() as usize);
    let flipped: Vec<u8> = (0..h).flat_map(|r| tex.data[(h - 1 - r) * w..(h - r) * w].to_vec()).collect();
    tex.data = flipped;
    (tex.top, tex.bottom) = (-tex.bottom, -tex.top);
    tex
}

/// Resolve an `sx`×`sy` oversampled 1-bit fill to grayscale coverage (`round(255·count/(sx·sy))`),
/// the box resolve native applies at any grid. `sx==sy==4` reproduces [`resolve_grayscale`].
fn resolve_box(contours: Vec<Vec<Seg>>, mode: u16, sx: i32, sy: i32, bold: (i32, i32), dilate: bool, boundary: bool) -> AlphaTexture {
    resolve_box_glyphs(contours, mode, sx, sy, bold, dilate, boundary, None)
}

#[allow(clippy::too_many_arguments)]
fn resolve_box_glyphs(contours: Vec<Vec<Seg>>, mode: u16, sx: i32, sy: i32, bold: (i32, i32), dilate: bool, boundary: bool, glyphs: Option<&[(std::ops::Range<usize>, u16)]>) -> AlphaTexture {
    let Some(os) = extrema_box(&contours) else { return AlphaTexture::empty() };
    let [l, mut t, mut r, b] = device_bounds(os, sx, sy, false);
    if r <= l || b <= t {
        return AlphaTexture::empty();
    }
    // The old `if boundary { x_on_center; sloped_on_center }` 8×1 column-fill recovery is retired: those
    // heuristics faked native's vertical recovery for the transpose vscan, which the single-pass both-axis
    // scan's real vertical-dropout pass now does (keeping them would double-count against the dropout).
    // A one-pixel bitmap bold at a small size runs per glyph inside the scan ([`bold_runs`]), on a grid one pixel wider.
    let runs = sy == 1 && run_darkening(bold);
    let mut bits = scan_glyph_bitmaps(&contours, sx * l, sx * (r + runs as i32), sy * t, sy * b, mode, glyphs, boundary, if runs { sx as usize } else { 0 }, false);
    if runs {
        r += 1;
    }
    if std::env::var_os("D2D_GRAYBITS_GO").is_some() {
        // Raw sx×sy scan grid before the box resolve (y-up — these segs are negated, row 0 = bottom).
        let (gbw, grows) = ((sx * (r - l)) as usize, (sy * (b - t)) as usize);
        eprintln!("GRAYBITS dev=[{l},{t},{r},{b}] sx={sx} sy={sy} bw={gbw} rows={grows}");
        for rr in 0..grows {
            let s: String = (0..gbw).map(|c| if bits[rr * gbw + c] { '#' } else { '.' }).collect();
            eprintln!("GB{rr:>2} {s}");
        }
    }
    let (sxu, syu) = (sx as usize, sy as usize);
    // Synthetic-bold embolden (the grayscale embolden on the overscaled gray map): a mono OR-dilate
    // of the OVERSAMPLED 1-bit grid — right by boldX·sx and up by boldY·sy sub-samples (= boldX/boldY
    // device px). Dilating the sub-sample grid (not the device-pixel count buffer) preserves interior
    // coverage — a solid stem's inner columns keep their sub-row count, only the right/top edge grows —
    // which the device-count-additive form over-filled. The bitmap grows right += boldX / up += boldY.
    let (bx, by) = bold;
    let (mut osw, mut osh) = ((sxu * (r - l) as usize), (syu * (b - t) as usize));
    if (bx > 0 || by > 0) && !runs {
        // The grid grows by boldX·sx / boldY·sy sub-samples so the device bounds stay pixel-aligned
        // (right += boldX, top -= boldY); the OR dilation reaches the same distance.
        let (gx, gy) = (bx.max(0) as usize * sxu, by.max(0) as usize * syu);
        (bits, osw, osh) = embolden_bits(&bits, osw, osh, gx, gy, gx, gy);
        r += bx;
        t -= by;
    }
    let bw = osw;
    let (w, h) = ((r - l) as usize, (b - t) as usize);
    let total = (sx * sy) as u32;
    // Native's DWrite `GlyphRunAnalysis::MergeGrayscaleGlyphBitmaps`/`MergeUncompressedGlyph4x4` packs the
    // overscaled 1-bit fill into the run bitmap with a horizontal OR-dilate-left by one sub-sample
    // (`out[x] = in[x] | in[x-1]`) before the box resolve. This is the no-DOGRAY NATURAL_SYMMETRIC path;
    // verified byte-exact against native's captured merged grid (see memory `nonlatin-textx-glyph-is-scan-converter`).
    if dilate {
        for oy in 0..osh {
            let base = oy * bw;
            for ox in (1..bw).rev() {
                if bits[base + ox - 1] {
                    bits[base + ox] = true;
                }
            }
        }
    }
    // Per-device-pixel subsample count (0..total), the gray-map box count, then the ×255 resolve.
    //
    // This plain box average is NOT native's `GlyphRunFiller::FillSubrect` in general. For the 8×1 grid
    // native computes a per-pixel coverage INDEX `idx = OVERSCALE_8X1[window]` (window = a sliding 11-bit
    // rolling value starts at 0; per device-col byte, LSB-first: `val >>= 8; val |= byte << 7; idx = OVERSCALE_8X1[val & 0x7FF]`;
    // output col X → device col origin+X−1 — validated reproducing native's own fillbm→out_b64, all rows),
    // then `COV_8X1[189·phase + idx]·255/8` in `TextLookupTableAtlas::FillRows`, finished by the blend's
    // enhanced-contrast gamma. The overscale reaches 3 sub-samples into the left neighbour, so a 1.5-px
    // stem resolves to a translation-invariant 4/8/4 the box average cannot: at a 7/8 sub-sample frac the
    // window gives 64/255/64, the box average 32/255/96. (The old comment here claimed "phase 3 == box
    // average, byte-exact over the suite"; that was WRONG — the 'd' stem matches at COV phase 0, not 3,
    // and the box average diverges at fractional fracs.) The box average nonetheless reproduces native
    // byte-for-byte here because the glyphs that reach this path are positioned where the two coincide:
    // GDI measuring modes place the origin on WHOLE pixels (text/mod.rs `subpixel` false), and grid-fit
    // pins the stem frac, so the overscale window and the box count land on the same bytes. The ~48-seed
    // r12 8×1 "coverage" class was NOT this resolve — it was our sub-pixel origin drift for GDI_NATURAL
    // (now whole-pixel), proved in memory `vm-cold-capture-method` (fillsubrect-cap/PROOF.md). Should a
    // genuinely sub-pixel-x grayscale mode ever need it, the window+COV path lives in `fillsubrect.rs`.
    let mut data = vec![0u8; w * h];
    for py in 0..h {
        for px in 0..w {
            let mut count = 0u32;
            for j in 0..syu {
                let rowbase = (syu * py + j) * bw + sxu * px;
                for i in 0..sxu {
                    count += bits[rowbase + i] as u32;
                }
            }
            data[py * w + px] = ((255 * count + total / 2) / total) as u8;
        }
    }
    // Bold shifts r/t off the extrema, so `drop_empty_tangent`'s os-based speculative-cell test no longer
    // maps to the grown box. Native still reports the tight ink box, so for the emboldened result trim any
    // fully-empty border row/column directly (frame-agnostic, and a no-op when every edge is inked — the
    // CT bold path's box is already tight). Matches native's gray sim-bold bounds on the ladder.
    if bx == 0 && by == 0 {
        drop_empty_tangent(l, t, r, b, data, os, sx, sy, false, 1)
    } else {
        trim_empty_border(l, t, r, b, data, 1)
    }
}

/// Trim fully-empty border rows/columns from a `bpp`-byte-per-pixel coverage texture, tightening the
/// device bounds to the ink box (native's `GetAlphaTextureBounds` reports the tight box). `data` is
/// row-major `(r-l)×(b-t)` pixels.
fn trim_empty_border(mut l: i32, mut t: i32, mut r: i32, mut b: i32, mut data: Vec<u8>, bpp: usize) -> AlphaTexture {
    let row_empty = |data: &[u8], w: usize, y: usize| data[y * w * bpp..(y + 1) * w * bpp].iter().all(|&v| v == 0);
    let col_empty = |data: &[u8], w: usize, h: usize, x: usize| (0..h).all(|y| data[(y * w + x) * bpp..(y * w + x + 1) * bpp].iter().all(|&v| v == 0));
    // Top / bottom rows.
    while b - t > 1 && row_empty(&data, (r - l) as usize, 0) {
        let w = (r - l) as usize;
        data.drain(0..w * bpp);
        t += 1;
    }
    while b - t > 1 && row_empty(&data, (r - l) as usize, (b - t - 1) as usize) {
        let w = (r - l) as usize;
        data.truncate((b - t - 1) as usize * w * bpp);
        b -= 1;
    }
    // Left / right columns.
    let (mut w, h) = ((r - l) as usize, (b - t) as usize);
    while r - l > 1 && col_empty(&data, w, h, w - 1) {
        let mut nd = Vec::with_capacity((w - 1) * h * bpp);
        for y in 0..h { nd.extend_from_slice(&data[y * w * bpp..(y * w + w - 1) * bpp]); }
        data = nd; r -= 1; w -= 1;
    }
    while r - l > 1 && col_empty(&data, w, h, 0) {
        let mut nd = Vec::with_capacity((w - 1) * h * bpp);
        for y in 0..h { nd.extend_from_slice(&data[(y * w + 1) * bpp..(y * w + w) * bpp]); }
        data = nd; l += 1; w -= 1;
    }
    AlphaTexture { left: l, top: t, right: r, bottom: b, data }
}

/// Synthetic-bold embolden of the OVERSAMPLED 1-bit sub-sample grid (the grayscale embolden on the
/// overscaled gray map): an OR-dilate right by `bx` and up by `by` sub-samples (= boldX/boldY device px
/// at an sx×sy grid). A sub-sample turns on if itself or a neighbour within the box is on, so a solid
/// stem's interior keeps its sub-row count and only the right/top edge grows — the count-additive form
/// over-filled the interior. The grid grows `right += bx`, and `up += by` (new rows above the old top).
fn embolden_bits(bits: &[bool], osw: usize, osh: usize, grow_x: usize, grow_y: usize, reach_x: usize, reach_y: usize) -> (Vec<bool>, usize, usize) {
    let (nw, nh) = (osw + grow_x, osh + grow_y);
    // The box `out[ox,oy] = OR bits[ox-reach_x..=ox, oy-grow_y..=oy-grow_y+reach_y]` is a separable
    // OR-dilation: do a horizontal pass then a vertical one, O(area) instead of the box's O(area·reach²).
    // The naive box blew the fuzz deadline on a huge-em (344 px) synthetic-bold glyph, where `reach` is
    // ~40 sub-samples over a ~450k-cell grid (≈1e8 ops/glyph); native's embolden is O(area). Output is
    // byte-identical to the box form.
    // Pass 1: right-dilate each input row by `reach_x` into the grown width (sliding "cols since last set";
    // a column turns on if a set bit lies within `reach_x` to its left, extending ink into the new cols).
    let mut h = vec![false; nw * osh];
    for iy in 0..osh {
        let (hrow, brow) = (iy * nw, iy * osw);
        let mut since = reach_x + 1;
        for ox in 0..nw {
            if ox < osw && bits[brow + ox] { since = 0; } else { since += 1; }
            h[hrow + ox] = since <= reach_x;
        }
    }
    // Pass 2: OR each column's rows over the window `[oy-grow_y, oy-grow_y+reach_y]` via a difference
    // array — a set `h[ox,iy]` turns on the output rows `oy ∈ [iy+grow_y-reach_y, iy+grow_y]`.
    let mut out = vec![false; nw * nh];
    let mut diff = vec![0i32; nh + 1];
    for ox in 0..nw {
        diff.iter_mut().for_each(|d| *d = 0);
        for iy in 0..osh {
            if h[iy * nw + ox] {
                let lo = (iy + grow_y).saturating_sub(reach_y);
                let hi = iy + grow_y; // inclusive last output row
                if lo < nh { diff[lo] += 1; }
                diff[(hi + 1).min(nh)] -= 1;
            }
        }
        let mut acc = 0i32;
        for oy in 0..nh {
            acc += diff[oy];
            out[oy * nw + ox] = acc > 0;
        }
    }
    (out, nw, nh)
}

thread_local! {
    static BOLD_PPEM: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };
}

/// While alive, a one-pixel bitmap bold (`bold == (1, 0)`) on the 8×1 and 6×1 grids takes the scaler's small-size
/// bitmap bold per glyph ([`bold_glyph`]) when `ppem`, the rounded y ppem, is at most 24. 0 keeps the plain dilation.
pub(crate) struct BoldPpem(i32);

impl BoldPpem {
    pub(crate) fn set(ppem: i32) -> Self {
        BoldPpem(BOLD_PPEM.with(|c| c.replace(ppem)))
    }
}

impl Drop for BoldPpem {
    fn drop(&mut self) {
        BOLD_PPEM.with(|c| c.set(self.0));
    }
}

fn run_darkening(bold: (i32, i32)) -> bool {
    bold == (1, 0) && BOLD_PPEM.with(|c| (1..=24).contains(&c.get()))
}

/// One glyph's one-pixel bitmap bold at a small size, on its `w`×`rows` overscaled grid (row 0 the bottom), `os`
/// sub-samples a pixel. A glyph taller than 48 rows takes the plain dilation, every sample `os` to the right of ink;
/// otherwise each run's right edge grows by [`bold_runs`].
fn bold_glyph(bits: &mut [bool], w: usize, rows: usize, os: usize) {
    if rows > 48 {
        for row in bits.chunks_mut(w) {
            let mut since = os + 1;
            for x in 0..w {
                if row[x] { since = 0 } else { since += 1 }
                row[x] = since <= os;
            }
        }
    } else {
        bold_runs(bits, w, rows, os);
    }
}

/// The scaler's small-size bitmap bold: every run's right edge, taken top row first and right to left, grows by
/// the average over the edges connected below it ([`next_edge`]) of an amount set by each edge's gap to the next ink
/// on its right in the scanned bits: nothing up to `2·os/3`, the excess over that up to `5·os/3`, `os` past it (or
/// at the row end). The average rounds half up and is at least `os/3`; a grown edge stops short of ink.
fn bold_runs(bits: &mut [bool], w: usize, rows: usize, os: usize) {
    let orig = bits.to_vec();
    let (lo, mid) = ((85 * os + 128) >> 8, (170 * os + 128) >> 8);
    let hi = mid + os;
    for r in (0..rows).rev() {
        for x in (0..w).rev() {
            if !orig[r * w + x] || (x + 1 < w && bits[r * w + x + 1]) || x + 1 == w {
                continue;
            }
            let (mut chain, mut sum) = (Vec::new(), 0);
            let (mut cr, mut cx) = (r, x);
            loop {
                let gap = (cx + 1..w).position(|i| orig[cr * w + i]).unwrap_or(hi).min(hi);
                sum += if gap <= mid { 0 } else if gap < hi { gap - mid } else { os };
                chain.push((cr, cx));
                match next_edge(&orig, w, cr, cx, os) {
                    Some(nx) => (cr, cx) = (cr - 1, nx),
                    None => break,
                }
            }
            let n = chain.len();
            let grow = ((sum + n / 2) / n).max(lo);
            for (cr, cx) in chain {
                for i in cx + 1..(cx + 1 + grow).min(w) {
                    if bits[cr * w + i] {
                        break;
                    }
                    bits[cr * w + i] = true;
                }
            }
        }
    }
}

/// The right edge in the row below (`r − 1`) connected to the run edge at `x` in row `r`, in [`bold_runs`]: straight
/// down; else the run below continues right, at most `os − 2` samples (1 below `os` 3) with row `r` empty up to two
/// past its end; else it ends left of `x`, at most `os − 1` samples, under row `r`'s run. Samples outside the row are
/// empty.
fn next_edge(orig: &[bool], w: usize, r: usize, x: usize, os: usize) -> Option<usize> {
    if r == 0 {
        return None;
    }
    let at = |row: usize, i: isize| i >= 0 && (i as usize) < w && orig[row * w + i as usize];
    let (below, here) = (|i: isize| at(r - 1, i), |i: isize| at(r, i));
    let x = x as isize;
    if below(x + 1) {
        let run = (0..8).take_while(|&j| below(x + 1 + j)).count() as isize;
        let reach = if os < 3 { 1 } else { os as isize - 2 };
        (run < 8 && run - 1 <= reach && (1..=run + 1).all(|j| !here(x + j))).then_some((x + run) as usize)
    } else if below(x) {
        Some(x as usize)
    } else {
        let i = (x - 7..x).rev().find(|&i| below(i))?;
        (i >= x + 1 - os.min(8) as isize && (i + 1..=x).all(here)).then_some(i as usize)
    }
}

/// Reconstruct one TrueType contour (device-space points + on-curve flags) into oversampled-26.6
/// [`Seg`]s, emitting a [`Seg::Quad`] per off-curve control point (inserting the implied on-curve
/// midpoint between two consecutive off-curve points) and a [`Seg::Line`] per on-curve step — the same
/// walk the font scaler does before spline decomposition. Mirrors the unhinted [`Builder`] pen, so both
/// paths reach [`split_quad`] identically.
fn outline_contour_segs(pts: &[[f64; 2]], on_in: &[bool], sx: i32, sy: i32, fine: bool) -> Vec<Seg> {
    let n = pts.len();
    // A one-point contour (an anchor, e.g. Consolas 'f') has no edges, and native's extrema pass skips any
    // contour whose start and end points coincide, so it never widens the glyph bounds either (r12 17700:
    // it pinned our NAT_SYM_DOWNSAMPLED 'f' left at 0 instead of native's 3, flipping the 2×2 dither).
    if n < 2 {
        return Vec::new();
    }
    // Skip a degenerate glyph whose device points are non-finite or astronomically far (a huge/negative
    // font size or an extreme transform): the 26.6 overscale (`to_2614`, `·sx·64`) saturates and the
    // os-midpoint `(a+b+1)>>1` then overflows i32 — a panic / page crash (fuzz seed 5335). 1e6 device px
    // is far past any target (D3D caps a bitmap at 16384) yet keeps `·sx·64` and the sums within i32.
    if pts.iter().any(|p| !p[0].is_finite() || !p[1].is_finite() || p[0].abs() > 1.0e6 || p[1].abs() > 1.0e6) {
        return Vec::new();
    }
    // Overscale every point to 26.6 first, then walk the contour in integer oversampled space exactly
    // like the native scaler: an implied on-curve point between two off-curve controls is their
    // os-integer midpoint `(a+b+1)>>1` (round half up — the scan converter), run *after* `<<2` overscale
    // (the outline overscale). Averaging in device f64 and re-rounding to the 1/64 device grid instead
    // quantizes off the half-os-unit native keeps (an odd os-sum lands at `≡2 mod 4`, not `≡0`), which
    // flips one 4×4 sub-sample at a curve tangent.
    if std::env::var_os("D2D_PTSDUMP").is_some() {
        let v: Vec<(i32, i32, bool)> = pts.iter().zip(on_in).map(|(&p, &o)| ((p[0] * 64.0).round() as i32, (p[1] * 64.0).round() as i32, o)).collect();
        eprintln!("PTS {v:?}");
    }
    // The scan converter first drops a point equal to the next one, the one kept turning on-curve, then the first
    // point when the last equals it, also turning the last on-curve (VM: a hinted Segoe UI Emoji mouth whose two
    // collapsed controls sit on their on-curve corners; kept, they moved a flat edge's sample-centre fill a sub-row).
    let mut q: Vec<P> = Vec::with_capacity(n);
    let mut on: Vec<bool> = Vec::with_capacity(n);
    for (&p, &o) in pts.iter().zip(on_in) {
        let p = to_2614(p, sx, sy, fine);
        if q.last() == Some(&p) {
            *on.last_mut().unwrap() = true;
        } else {
            q.push(p);
            on.push(o);
        }
    }
    if q.len() > 1 && q.last() == q.first() {
        q.remove(0);
        on.remove(0);
        *on.last_mut().unwrap() = true;
    }
    let n = q.len();
    if n < 2 {
        return Vec::new();
    }
    let mid = |a: P, b: P| [(a[0] + b[0] + 1) >> 1, (a[1] + b[1] + 1) >> 1];
    let first_on = (0..n).find(|&i| on[i]);
    let s0 = first_on.unwrap_or(0);
    let start = match first_on {
        Some(i) => q[i],
        None => mid(q[0], q[n - 1]),
    };
    let mut segs = Vec::new();
    let mut last = start;
    let mut i = 1;
    while i <= n {
        let idx = (s0 + i) % n;
        if on[idx] {
            segs.push(Seg::Line(last, q[idx]));
            last = q[idx];
            i += 1;
        } else {
            let nidx = (s0 + i + 1) % n;
            let end = if on[nidx] { q[nidx] } else { mid(q[idx], q[nidx]) };
            segs.push(Seg::Quad(last, q[idx], end));
            last = end;
            i += if on[nidx] { 2 } else { 1 };
        }
    }
    if last != start {
        segs.push(Seg::Line(last, start));
    }
    segs
}

/// Shift every segment point by `d` (os 26.6) — positions a glyph-relative contour at its pen origin.
/// dwrite's 45° outline nudge. Native's transform setup flags a scaler matrix `m`
/// (device px per em, 16.16) that is not a positive uniform square when either row's entries match
/// in magnitude to within 33 (native's max-of-45 trick); native's bitmap-size step then adds one oversampled
/// 26.6 unit to every TrueType outline x. Pinned by native's glyph-scan-converter contour lists (rot45 captures:
/// grayscale, ClearType and aliased all `sx·ours + 1`; 12 units off 45° nudges, 70 does not).
pub(crate) fn max45_nudge(m: [f64; 4]) -> bool {
    let f = m.map(|v| (v * 65536.0).round().clamp(-1e15, 1e15) as i64);
    let max45 = |a: i64, b: i64| (a.abs() - b.abs()).abs() <= 33;
    let square = f[0] == f[3] && f[1] == 0 && f[2] == 0 && f[3] >= 0;
    !square && (max45(f[2], f[3]) || max45(f[0], f[1]))
}

fn translate_segs(segs: &mut [Seg], d: P) {
    let mv = |p: P| [p[0] + d[0], p[1] + d[1]];
    for s in segs.iter_mut() {
        *s = match *s {
            Seg::Line(a, b) => Seg::Line(mv(a), mv(b)),
            Seg::Quad(a, c, b) => Seg::Quad(mv(a), mv(c), mv(b)),
        };
    }
}

/// ClearType coverage (`DWRITE_TEXTURE_CLEARTYPE_3x1`): scan-convert at 6× horizontal × 5× vertical,
/// apply the classic Platt horizontal filter (three width-6 boxes over a 10-sub-pixel window, offset
/// 2 apart) and the 5-tap vertical filter `{4,9,10,9,4}`, resolving to three colour subpixels
/// (`round(255·s/216)`, byte 0/1/2 = left/centre/right). Both filter tables are derived from the
/// algorithm reproduced, not copied — and verified byte-identical to native (its
/// ClearType 3×1 alpha-texture filter: the classic Platt RGB filter = the
/// width-6 boxes at offsets 0/2/4, `{4,9,10,9,4}` rows, and `g_AlphaNormalizationTable6x5[0..=216]` is
/// exactly this `round(255·s/216)`).
pub fn cleartype(dt: &[f64; 6], run: &GlyphRun, font: &FontRef, dropout: u16) -> AlphaTexture {
    unhinted_yup(dt, 3, |up| resolve_cleartype(glyph_run_arcs(up, run, font, 6, 5, 6), dropout, &CT_VWEIGHTS, (0, 0)))
}

/// Aliased coverage (`DWRITE_TEXTURE_ALIASED_1x1`, bi-level): scan-convert at the device resolution
/// (one sample at each pixel centre) and map a covered sample to 255, else 0 — the hard edges of
/// `TEXT_ANTIALIAS_MODE_ALIASED` / `DWRITE_RENDERING_MODE_ALIASED`.
pub fn aliased(dt: &[f64; 6], run: &GlyphRun, font: &FontRef, dropout: u16) -> AlphaTexture {
    unhinted_yup(dt, 1, |up| resolve_aliased(glyph_run_arcs(up, run, font, 1, 1, 1), dropout))
}

/// DWRITE_FONT_SIMULATIONS_OBLIQUE's slant: the TrueType scaler shears its transform before hinting
/// (`m01 += FixMul(m00, 22272)`, `m11 += FixMul(m10, 22272)`), i.e. design x += 22272/65536 · y.
pub const OBLIQUE_SHEAR: f64 = 22272.0 / 65536.0;

/// The oblique-simulated transform as per-font-unit 16.16 entries `[x←X, x←Y, y←X, y←Y]` (font Y up) for
/// `dt`·`em`: each entry rounded to 16.16, the shear `FixMul(·, 22272)` added to the Y column, then
/// `FixDiv(t, upem<<16)` (round half away from zero).
fn oblique_entries(dt: &[f64; 6], em: f64, upem: f64) -> [i32; 4] {
    let e = |c: f64| (c * 65536.0).round().clamp(i32::MIN as f64, i32::MAX as f64) as i64;
    let fixmul = |a: i64, b: i64| { let v = a * b; (v + 0x8000 - i64::from(v < 0)) >> 16 };
    let u = (upem as i64) << 16;
    let fixdiv = |a: i64| { let q = ((a.abs() << 16) + u / 2) / u; (if a < 0 { -q } else { q }) as i32 };
    let (t00, t10) = (e(dt[0] * em), e(dt[1] * em));
    let (t01, t11) = (e(-dt[2] * em) + fixmul(t00, 22272), e(-dt[3] * em) + fixmul(t10, 22272));
    [fixdiv(t00), fixdiv(t01), fixdiv(t10), fixdiv(t11)]
}

/// A font-unit point through [`oblique_entries`]: each product rounded to 1/64 px on its own.
fn oblique_map(e: &[i32; 4], fx: f64, fy: f64) -> [f64; 2] {
    let m = |f: f64, c: i32| (f * 64.0 * c as f64 / 65536.0).round();
    [(m(fx, e[0]) + m(fy, e[1])) / 64.0, (m(fx, e[2]) + m(fy, e[3])) / 64.0]
}

/// Unhinted (grid-fit-disabled) coverage for `kind`. DirectWrite oversamples by rendering mode as it
/// does for hinted glyphs: the GDI and NATURAL modes (`h_only`) scan grayscale on the 8×1 grid and
/// ClearType on 6×1; the symmetric modes on 4×4 and 6×5. The GDI modes (`whole_px`) place each glyph
/// origin on a whole pixel, as for hinted glyphs.
pub fn unhinted(dt: &[f64; 6], run: &GlyphRun, font: &FontRef, kind: Kind, h_only: bool, whole_px: bool, dropout: u16) -> AlphaTexture {
    unhinted_ds(dt, run, font, kind, h_only, whole_px, None, dropout, None)
}

/// [`unhinted`]; `ds` = NATURAL_SYMMETRIC_DOWNSAMPLED, whose grayscale scans each glyph at 4×4 from a
/// half-pixel pen and reduces it 2×2 as the hinted path does ([`coverage_downsampled`]); `Some(true)`
/// re-aligns each glyph's checkerboard to the run, as the hinted brush path (VM r24 29703: a solid run
/// under a generic matrix is byte-exact unaligned).
pub fn unhinted_ds(dt: &[f64; 6], run: &GlyphRun, font: &FontRef, kind: Kind, h_only: bool, whole_px: bool, ds: Option<bool>, dropout: u16, design_bold: Option<(i32, i32)>) -> AlphaTexture {
    unhinted_grouped(dt, run, font, kind, h_only, whole_px, ds, dropout, design_bold, false, None).pop().unwrap_or_else(AlphaTexture::empty)
}

/// The glyph groups a run draws as separate textures, from each glyph's oversampled box (`None` for no ink) and the
/// row slack two boxes may be apart and still meet (one where y is oversampled), in glyph order ([`unhinted_grouped`]).
pub type GroupFn<'a> = &'a dyn Fn(&[Option<[i32; 4]>], i32) -> Vec<Vec<usize>>;

/// [`unhinted_ds`] drawn as d2d1's glyph cache draws it: `group` splits the run into the glyph groups that scan
/// together, one texture each, in glyph order, on every grid but the aliased one, whose 0/255 coverage composites as
/// its union (VM: joined Segoe UI Arabic under a transpose with grid fit disabled, ClearType 6×1 and 6×5, grayscale
/// 8×1 and 4×4).
#[allow(clippy::too_many_arguments)]
pub fn unhinted_grouped(dt: &[f64; 6], run: &GlyphRun, font: &FontRef, kind: Kind, h_only: bool, whole_px: bool, ds: Option<bool>, dropout: u16, design_bold: Option<(i32, i32)>, widen: bool, group: Option<GroupFn>) -> Vec<AlphaTexture> {
    let dt = &scaler_matrix(dt);
    let scale = dt[0].abs().max(dt[3].abs()) as f32;
    let rounded = GlyphRun { em_size: (run.em_size * scale).round() / scale, ..*run };
    // The GDI modes scale the outline at a whole-pixel em only on the axes; off-axis native scales the
    // exact em (r20 25908: a sheared GDI_CLASSIC run drew 0.5% small, its 48.24 px em rounded to 48).
    let axis = dt[1] == 0.0 && dt[2] == 0.0;
    // Grid-fit disabled, native still rescales a uniform axis-aligned or quarter-turn transform to a whole
    // device ppem (round half up) when the face asks for integer ppems: the face flag it tests is set from
    // head.flags FORCE_INTEGER_PPEM (bit 3). VM: Arial NAT_SYM em 4.45, grid-fit disabled, scales at ppem 4.
    let (a, d) = if axis { (dt[0] as f32 * run.em_size, dt[3] as f32 * run.em_size) } else if dt[0] == 0.0 && dt[3] == 0.0 { (dt[1] as f32 * run.em_size, dt[2] as f32 * run.em_size) } else { (0.0, 0.0) };
    let integer_ppem = font.head().is_ok_and(|h| h.flags().bits() & 8 != 0);
    // Aliased text keeps the whole ppem under a uniform scale for any face, as on the hinted path ([`super::fractional_em`])
    // (VM: Segoe UI Emoji aliased with grid fit disabled at em 47.5 under the identity, flips and a quarter turn
    // scales at ppem 48, under a shear at the exact em; grayscale keeps the exact em throughout).
    let ppem_run = if (integer_ppem || matches!(kind, Kind::Aliased)) && a != 0.0 && (a.abs() - d.abs()).abs() < 1.0 / 65536.0 && a.abs() < 32768.0 {
        let v = a.abs();
        let mut n = v.round_ties_even();
        if n - v == -0.5 {
            n += 1.0;
        }
        Some(GlyphRun { em_size: run.em_size * (n / v), ..*run })
    } else {
        None
    };
    // The downsampled grid scales at that whole ppem too (VM: Malgun and Trebuchet bold, grid fit disabled, ppem 4.3 to 32).
    if let (Some(align_to_run), Kind::Grayscale) = (ds, kind) {
        let run = ppem_run.as_ref().unwrap_or(run);
        let up = [dt[0], -dt[1], dt[2], -dt[3], dt[4], -dt[5]];
        let mut gl = Vec::new();
        let segs = glyph_run_arcs_at(&up, run, font, 4, 4, 2, 2, Some(&mut gl), design_bold);
        let modes: Vec<_> = gl.iter().map(|(r, _)| (r.clone(), dropout)).collect();
        let phases: Vec<bool> = gl.iter().map(|g| g.1).collect();
        // Drawn through the glyph cache, the run splits into its groups, each downsampled on its own with the
        // checkerboard re-aligned within it, as the hinted path does. A design-bold glyph's box grows right and up by
        // the embolden, strength(upm) font units rounded out to whole 4×4 units, so bold glyphs whose bitmaps only
        // touch draw as one texture while regular ones stay apart (VM: Comic Sans 'abo' and .notdef runs at ppem 3
        // with gasp hinting off, upright and under a y-shear, bold and regular).
        if let Some(group) = group {
            let (gx, gy) = design_bold.map_or((0, 0), |(bx, by)| {
                let em = run.em_size as f64 / font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f64;
                let grow = |b: i32, a: f64| (b as f64 * em * a * 4.0).ceil() as i32;
                (grow(bx, dt[0].abs().max(dt[1].abs())), grow(by, dt[2].abs().max(dt[3].abs())))
            });
            let boxes: Vec<_> = gl.iter().map(|(r, _)| extrema_box(&segs[r.clone()]).map(|[l, t, r, b]| [l, t, r + gx, b + gy])).collect();
            let mut segs: Vec<Option<Vec<Seg>>> = segs.into_iter().map(Some).collect();
            return group(&boxes, 1)
                .iter()
                .map(|ids| {
                    let (mut sub, mut gm, mut gp) = (Vec::new(), Vec::new(), Vec::new());
                    for &i in ids {
                        let s = sub.len();
                        sub.extend(segs[gl[i].0.clone()].iter_mut().filter_map(Option::take));
                        gm.push((s..sub.len(), dropout));
                        gp.push(phases[i]);
                    }
                    downsample_segs(&sub, &gm, &gp, true, widen)
                })
                .collect();
        }
        return vec![downsample_segs(&segs, &modes, &phases, align_to_run, widen)];
    }
    let run = if whole_px && axis && scale > 0.0 { &rounded } else if let Some(r) = &ppem_run { r } else { run };
    // DWRITE_FONT_SIMULATIONS_BOLD on the grid-fit-disabled (`IDWriteGlyphRunAnalysis` / disabled DrawText)
    // surface: native emboldens the rasterized glyph bitmap per AA mode (not the outline). This path was
    // previously bold-blind (hardcoded `(0,0)`), so disabled bold runs rendered un-bolded; threading the
    // amount here closes that gap for both `render_analysis`'s disabled branch and the dwrite-ffi shim.
    // The amount is the full `bold_amount(ppem)` — the hinted path's `(1,0)→(0,0)` grayscale skip is
    // specific to its `horizontal_one_pixel` embolden and does NOT apply here (native widens even an
    // em-16 grayscale glyph, gray bounds +2).
    //
    // Only for an UPRIGHT positive-scale axis-aligned transform: native emboldens the glyph in its own
    // (pre-transform) space, while this embolden grows the resolved grid in device right/up. Those agree
    // only when the glyph's local right/up map to device right/up; under a rotation, shear or an
    // axis-flip (negative `dt[0]`/`dt[3]`, e.g. a 180° turn) the grid dilate grows the wrong way and
    // over-draws (r25 30299 Tahoma 900 under a 180° flip, 30425 Malgun 700 under rotation+shear — the
    // bitmap embolden is negligible at those ~4 px sizes, so native reads as un-bolded there and the grid
    // dilate spuriously inks a pixel). An off-axis matrix takes the design-outline bold (`design_bold`) instead.
    // ponytail: a flip embolds along the scaler's quarter-turn factor, not ported; no corpus run there is
    // unhinted, bold and big enough to show it.
    let bold = if run.simulations & 1 != 0 && design_bold.is_none() && axis && dt[0] > 0.0 && dt[3] > 0.0 {
        bold_amount((run.em_size as f64 * dt[0].abs().max(dt[3].abs())).round() as i32)
    } else {
        (0, 0)
    };
    // `design_bold`: the bold is baked into the design outline, so the scans run un-bolded.
    let arcs = |up: &[f64; 6], sx: i32, sy: i32, px: i32| glyph_run_arcs_at(up, run, font, sx, sy, px, sy, None, design_bold);
    // The grid each kind scans on (x and y oversample, the x pen grid) and its bytes per pixel.
    let grid = match kind {
        Kind::ClearType | Kind::ClearTypeH if h_only => Some((6, 1, if whole_px { 1 } else { 6 }, 3)),
        Kind::ClearType | Kind::ClearTypeH => Some((6, 5, 6, 3)),
        Kind::Grayscale if h_only => Some((8, 1, if whole_px { 1 } else { 8 }, 1)),
        Kind::Grayscale => Some((4, 4, 4, 1)),
        Kind::Aliased => None,
    };
    if let (Some(group), Some((sx, sy, px, bpp))) = (group, grid) {
        let up = [dt[0], -dt[1], dt[2], -dt[3], dt[4], -dt[5]];
        let mut gl = Vec::new();
        let segs = glyph_run_arcs_at(&up, run, font, sx, sy, px, sy, Some(&mut gl), design_bold);
        // A glyph's bitmap is its emboldened one, so it meets its neighbours by that: the scaler widens the measured
        // box by the embolden, boldX oversampled columns on the right and boldY rows on top (this frame is y-up),
        // as the hinted path does (r25 30345, a tiny bold ClearType run native draws as one texture).
        let (gx, gy) = (bold.0 * sx, bold.1 * sy);
        let boxes: Vec<_> = gl
            .iter()
            .map(|(r, _)| extrema_box(&segs[r.clone()]).map(|[l, t, r, b]| [l + gx.min(0), t + gy.min(0), r + gx.max(0), b + gy.max(0)]))
            .collect();
        // Each glyph's contours go to the one group that holds it.
        let mut segs: Vec<Option<Vec<Seg>>> = segs.into_iter().map(Some).collect();
        return group(&boxes, (sy > 1) as i32)
            .iter()
            .map(|ids| {
                let (mut sub, mut glyphs) = (Vec::new(), Vec::new());
                for &i in ids {
                    let s = sub.len();
                    sub.extend(segs[gl[i].0.clone()].iter_mut().filter_map(Option::take));
                    glyphs.push((s..sub.len(), dropout));
                }
                let mut tex = match (kind, sy) {
                    // Inside a group each glyph still scans on its own before the merge, as on the ungrouped 6×1 path.
                    (Kind::ClearType | Kind::ClearTypeH, 1) => resolve_cleartype_glyphs(sub, dropout, &[1], Some(&glyphs), bold, widen),
                    (Kind::ClearType | Kind::ClearTypeH, _) => resolve_cleartype_glyphs(sub, dropout, &CT_VWEIGHTS, Some(&glyphs), bold, widen),
                    (_, 1) => resolve_box_glyphs(sub, dropout, 8, 1, bold, widen, true, Some(&glyphs)),
                    _ => resolve_box_glyphs(sub, dropout, 4, 4, bold, widen, true, Some(&glyphs)),
                };
                flip_rows(&mut tex, bpp);
                tex
            })
            .collect();
    }
    vec![match kind {
        // Each glyph scans into its own bitmap before the merge, as on the grayscale path below: one scan of the whole
        // run puts a glyph's rows in a taller frame, where a near-horizontal curve bottom crosses a row centre the
        // glyph's own scan does not (VM: Courier New 'a' beside a 'g' at 17.3 px, GDI CT, grid fit disabled).
        Kind::ClearType | Kind::ClearTypeH if h_only => unhinted_yup(dt, 3, |up| {
            let mut gl = Vec::new();
            let segs = glyph_run_arcs_at(up, run, font, 6, 1, if whole_px { 1 } else { 6 }, 1, Some(&mut gl), design_bold);
            let glyphs: Vec<(std::ops::Range<usize>, u16)> = gl.iter().map(|(r, _)| (r.clone(), dropout)).collect();
            resolve_cleartype_glyphs(segs, dropout, &[1], Some(&glyphs), bold, widen)
        }),
        Kind::ClearType | Kind::ClearTypeH => unhinted_yup(dt, 3, |up| resolve_cleartype_glyphs(arcs(up, 6, 5, 6), dropout, &CT_VWEIGHTS, None, bold, widen)),
        Kind::Aliased => {
            // Each glyph scans on its own with its dropout control before the union, as on the antialiased grids (VM:
            // Malgun Gothic 'fjord' at ppem 6.5, aliased, grid fit disabled, where a joint scan loses junction pixels).
            let tex = unhinted_yup(dt, 1, |up| {
                let mut gl = Vec::new();
                let segs = glyph_run_arcs_at(up, run, font, 1, 1, 1, 1, Some(&mut gl), design_bold);
                let glyphs: Vec<(std::ops::Range<usize>, u16)> = gl.iter().map(|(r, _)| (r.clone(), dropout)).collect();
                resolve_aliased_glyphs(segs, dropout, Some(&glyphs))
            });
            let (bx, by) = bold;
            if bx > 0 || by > 0 { embolden_aliased(&tex, bx, by) } else { tex }
        }
        // Native scan-converts each glyph of the run SEPARATELY into its own oversampled 1-bit bitmap
        // (one row-sweep bitmap fill per glyph, each with its own dropout control), then OR-merges those
        // per-glyph bitmaps into the shared run grid before the box resolve. Scanning the whole run's
        // merged outline in one pass instead drops a thin corner where one glyph's dropout-recovered
        // staircase abuts the next glyph's on the oversampled grid: under a glyph-collapsing transform a
        // run like "Th" leaves the junction pixel's 2/8 sub-column coverage unrecovered. Pass the
        // per-glyph contour ranges so each glyph scans on its own, as native does.
        Kind::Grayscale if h_only => unhinted_yup(dt, 1, |up| {
            let mut gl = Vec::new();
            let segs = glyph_run_arcs_at(up, run, font, 8, 1, if whole_px { 1 } else { 8 }, 1, Some(&mut gl), design_bold);
            let glyphs: Vec<(std::ops::Range<usize>, u16)> = gl.iter().map(|(r, _)| (r.clone(), dropout)).collect();
            resolve_box_glyphs(segs, dropout, 8, 1, bold, widen, true, Some(&glyphs))
        }),
        Kind::Grayscale if bold != (0, 0) || widen => unhinted_yup(dt, 1, |up| resolve_box(arcs(up, 4, 4, 4), dropout, 4, 4, bold, widen, true)),
        Kind::Grayscale => unhinted_yup(dt, 1, |up| resolve_grayscale(arcs(up, 4, 4, 4), dropout)),
    }]
}

/// The device matrix as the TrueType scaler holds it: a 2×2 component below 2^-17, zero in the scaler's 16.16 fixed
/// point, is exactly zero, so a quarter turn computed in floats (cos 90° = −4.37e-8) scales and rescales its ppem as a
/// quarter turn (VM: 16 replayed unhinted runs under such turns are byte-exact only with the flush).
pub(crate) fn scaler_matrix(dt: &[f64; 6]) -> [f64; 6] {
    let z = |v: f64| if (v * 65536.0).round() == 0.0 { 0.0 } else { v };
    [z(dt[0]), z(dt[1]), z(dt[2]), z(dt[3]), dt[4], dt[5]]
}

/// Vertical 5-tap ClearType filter weights (`g_AlphaNormalizationTable6x5` indexing): centre row 10,
/// inner 9, outer 4 — sum 36, so a fully-covered pixel's box sum is 36·6 = 216.
const CT_VWEIGHTS: [u32; 5] = [4, 9, 10, 9, 4];

// ---- oversampled 26.6 outline ---------------------------------------------------------------

type P = [i32; 2]; // a point in oversampled 26.6 fixed point (1 oversampled pixel = 64 units)

/// An outline segment in oversampled 26.6: a line, or a quadratic (control point `c`).
enum Seg {
    Line(P, P),
    Quad(P, P, P),
}

/// Round a device-space point to oversampled 26.6 (×overscale, ×64).
fn to_2614(p: [f64; 2], sx: i32, sy: i32, fine: bool) -> P {
    // Native rounds the outline to 1/64 DEVICE pixel (device 26.6) first, then overscales ×factor
    // (the outline overscale: `<<2` for grayscale), so oversampled coords are multiples of the factor.
    // Clamp the 26.6 coord to ±2²⁷ (NaN → 0) before the ×overscale: an extreme/NaN device point (e.g. a
    // huge text layout rect) otherwise saturates `as i32` and overflows the multiply — a panic that
    // aborts across the COM ABI (a public DrawText must never crash). Real glyph coords are ≤~2²⁰, far
    // inside the bound (no-op); an out-of-range glyph lands far off-canvas and scan-converts to nothing,
    // matching native (which draws nothing for such a degenerate placement).
    // `fine`: round once at the OVERSAMPLED grid (×64·s) instead of 1/64 device then ×s. For a hinted
    // point (an exact 1/64-device F26Dot6 value) the two are identical — `round(n/64·64·s) = n·s =
    // round(n/64·64)·s` — so the glyph SHAPE is unchanged; they differ only for the sub-pixel pen
    // PHASE, which ClearType places on 1/6 px (not a 1/64 multiple): round-to-1/64-first truncates 2/6
    // px to 21/64 (→ 126 os units) instead of the exact 128 (2 sub-columns), leaving a few curve
    // crossings one sub-column short at the 2/6 and 5/6 phases. `fine` is set only for the hinted CT
    // scan; the unhinted (grid-fit-disabled) path keeps the coarse 1/64-device round native applies to
    // a raw transformed outline (`texte2e-03-gfdisabled-curve` needs it). Clamp the device coord to
    // ±2²¹ px first (real glyph coords ≤~2²⁰; NaN → 0) so the ×64·s multiply can't overflow `as i32`.
    let q = |v: f64, s: i32| -> i32 {
        let vc = if v.is_finite() { v.clamp(-(1i64 << 21) as f64, (1i64 << 21) as f64) } else { 0.0 };
        if fine {
            (vc * 64.0 * s as f64).round() as i32
        } else {
            (vc * 64.0).round() as i32 * s
        }
    };
    [q(p[0], sx), q(p[1], sy)]
}

/// Flatten every glyph's (unhinted) outline into oversampled-26.6 segments under `dt`, each glyph
/// translated so its device origin lands on the `sx`×`sy` sub-pixel grid (native scan-converts each
/// glyph into the oversampled bitmap at an integer sub-pixel offset — the `CreateAlphaTexture*` merge).
/// Phase 0.75 px is on the 1/4 grid but not 1/6, so this only shifts glyphs under ClearType.
/// `px` is the pen phases per pixel in x: `sx` (the sub-pixel grid), or 1 for the GDI modes'
/// whole-pixel origins.
fn glyph_run_arcs(dt: &[f64; 6], run: &GlyphRun, font: &FontRef, sx: i32, sy: i32, px: i32) -> Vec<Vec<Seg>> {
    glyph_run_arcs_at(dt, run, font, sx, sy, px, sy, None, None)
}

/// Assemble a TrueType composite glyph into font-unit contours (points, on-curve flags, per-contour end
/// indices) by resolving each component's simple outline and applying the component's font-unit offset —
/// so the unhinted scan path builds a composite's implied on-curve midpoints in os-integer space
/// (`outline_contour_segs` `(a+b+1)>>1`, the native scan converter), exactly like a simple glyph. Skrifa's
/// `OutlinePen` instead inserts those midpoints in float font units and the Builder re-rounds them, landing
/// 1 os unit off at a curve apex — the café/é accented-Latin ClearType residual. A component's 2×2 applies to
/// its font-unit points before the offset (VM: Segoe UI Emoji U+1F642, whose eyes are one swapped and scaled
/// component, matches native's scan contours point for point at ppem 12 to 57). Returns `None` for a
/// point-anchored component or excessive nesting; the caller then falls back to skrifa. Variable instances are
/// gated out upstream.
/// A static TrueType glyph's raw points in design units (positions, on-curve flags, contour ends), a composite
/// assembled from its components; `None` for a CFF face or a composite with an anchored component.
pub fn glyf_points_fu(font: &FontRef, gid: GlyphId) -> Option<(Vec<[f64; 2]>, Vec<bool>, Vec<usize>)> {
    composite_contours_fu(&font.glyf().ok()?, &font.loca(None).ok()?, gid, 0)
}

/// A glyph as GetGlyphRunOutline returns it: hinted by its glyph program at ppem = upem (one pixel per design unit),
/// in design units with the hinter's 26.6 fractions (VM sb39: Calibri 'O' at em 150, the counter's right stem edge
/// one unit in from its control value, the rest of the counter interpolated). `None` when the face has no TrueType
/// hinting tables.
pub fn outline_points_fu(inst: &crate::text::hinting::Instance, gid: GlyphId) -> Option<(Vec<[f64; 2]>, Vec<bool>, Vec<usize>)> {
    let h = inst.hint(gid.to_u32() as u16)?;
    let pts = h.points.iter().map(|p| [p.x as f64 / 64.0, p.y as f64 / 64.0]).collect();
    Some((pts, h.on_curve, h.contour_ends.iter().map(|&e| e as usize).collect()))
}

fn composite_contours_fu(gf: &Glyf, lc: &Loca, gid: GlyphId, depth: u8) -> Option<(Vec<[f64; 2]>, Vec<bool>, Vec<usize>)> {
    if depth > 4 {
        return None;
    }
    match lc.get_glyf(gid, gf).ok().flatten()? {
        Glyph::Simple(s) => {
            let pts: Vec<[f64; 2]> = s.points().map(|p| [p.x as f64, p.y as f64]).collect();
            let on: Vec<bool> = s.points().map(|p| p.on_curve).collect();
            let ends: Vec<usize> = s.end_pts_of_contours().iter().map(|e| e.get() as usize).collect();
            Some((pts, on, ends))
        }
        Glyph::Composite(c) => {
            let (mut pts, mut on, mut ends) = (Vec::new(), Vec::new(), Vec::new());
            for comp in c.components() {
                // A font-unit offset after the component's 2×2 (through it too when the component says its offset is
                // scaled); a point-anchored component falls back to skrifa. The offset is added in font units (native
                // scales the assembled outline uniformly for a grid-fit-disabled glyph), matching the hinting path.
                let (x, y) = match comp.anchor {
                    Anchor::Offset { x, y } => (x as f64, y as f64),
                    Anchor::Point { .. } => return None,
                };
                let t = &comp.transform;
                let [xx, yx, xy, yy] = [t.xx, t.yx, t.xy, t.yy].map(|v| v.to_f32() as f64);
                let (dx, dy) = if comp.flags.contains(CompositeGlyphFlags::SCALED_COMPONENT_OFFSET) { (x * xx + y * xy, x * yx + y * yy) } else { (x, y) };
                let (cp, con, cends) = composite_contours_fu(gf, lc, GlyphId::from(comp.glyph), depth + 1)?;
                let base = pts.len();
                for p in &cp {
                    pts.push([p[0] * xx + p[1] * xy + dx, p[0] * yx + p[1] * yy + dy]);
                }
                on.extend_from_slice(&con);
                for e in &cends {
                    ends.push(e + base);
                }
            }
            Some((pts, on, ends))
        }
    }
}

/// [`glyph_run_arcs`] with `py` pen phases per pixel in y, recording each glyph's contour range and its
/// 2×2 dither phase (the parity of its half-pixel device origin, as the hinted path's) in `glyphs`.
#[allow(clippy::too_many_arguments)]
fn glyph_run_arcs_at(dt: &[f64; 6], run: &GlyphRun, font: &FontRef, sx: i32, sy: i32, px: i32, py: i32, mut glyphs: Option<&mut Vec<(std::ops::Range<usize>, bool)>>, design_bold: Option<(i32, i32)>) -> Vec<Vec<Seg>> {
    let upem = font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f64;
    let scale = run.em_size as f64 / upem;
    let outlines = font.outline_glyphs();
    // Raw `glyf` points let a TrueType simple glyph build its contours with the same os-integer
    // implied-midpoint walk ([`outline_contour_segs`]) as the hinted path. Our raw+scaled points are
    // byte-identical to native's scan-converter contour list; skrifa's pen instead inserts the implied
    // midpoint between two off-curve points in float and re-rounds, losing the half-oversample unit —
    // 1 sub-sample off on curved stems (the `texte2e-03`/unhinted-`o`/`g` residual).
    let raw = font.glyf().ok().zip(font.loca(None).ok());
    let em = run.em_size as f64;
    let nudge = (raw.is_some() && max45_nudge([dt[0] * em, dt[1] * em, dt[2] * em, dt[3] * em])) as i32;
    // An unhinted outline is sheared in design space, as the scaler's transform is ([`OBLIQUE_SHEAR`]):
    // the rasterizer adds FixMul(t00, 22272) to t01 and FixMul(t10, 22272) to t11, and the unscaled scaler path
    // (MPPEM = upem) divides every entry by upem (`FixDiv(t, upem<<16)`), so each term is its own rounded
    // product of a per-font-unit 16.16 entry (VM: Arial 'ca' em 4.45–20 contours, shear 44/128 at ppem 4).
    let shear = if run.simulations & 2 != 0 && !run.is_sideways { OBLIQUE_SHEAR } else { 0.0 };
    let ob = (shear != 0.0).then(|| oblique_entries(dt, em, upem));
    // Native's unscaled pass runs the glyph program at ppem = upem before the outline is scaled (VM r25 30215: Georgia
    // 'i' at 239.5 degrees, grid-fit disabled; the program's y moves of points 5-15 reach the device contour). Its pass
    // flags come from the real transform and scan grid, as on the hinted path: an x-subpixel pass that is not symmetric
    // under a zero m11 toggles vertical LCD (r25 30700: sideways Georgia on the GDI 8x1 grid, whose program then leaves
    // 'a', 'e', 'r' unmoved, while the 4x4 grid moves them).
    // ponytail: the program's GETINFO mode follows the scan grid; a variable instance keeps its raw outline.
    let unscaled = run.coords.is_empty().then(|| {
        let mode = match (sx, sy) { (6, _) => super::hinting::Mode::ClearType, (1, 1) => super::hinting::Mode::Aliased, _ => super::hinting::Mode::Grayscale };
        let axes_swapped = (dt[0] * em * 65536.0).round() == 0.0;
        super::hinting::Instance::new(font, upem, super::hinting::Config { mode, symmetric: sy > 1, cleartype: sx == 6, axes_swapped, ..Default::default() })
    }).flatten();
    let mut contours: Vec<Vec<Seg>> = Vec::new();
    let mut pen_x = run.baseline_origin[0] as f64;
    let origin_y = run.baseline_origin[1] as f64;
    for g in run.glyphs {
        let gid = GlyphId::new(g.id as u32);
        let pen = [pen_x + g.offset[0] as f64, origin_y - g.offset[1] as f64];
        // Rigid shift placing this glyph's device origin on the sub-pixel grid (round half away).
        let ox = dt[0] * pen[0] + dt[2] * pen[1] + dt[4];
        let oy = dt[1] * pen[0] + dt[3] * pen[1] + dt[5];
        // A glyph whose device origin is non-finite or far beyond any possible target (D3D caps a
        // bitmap at 16384 px) contributes nothing and can't be placed: skip it so the os-26.6 pen
        // (`round(o·s)·64`) can't overflow i32 (a degenerate/huge text rect otherwise panics — fuzz
        // seed 1226). Still advance the pen so later glyphs keep their spacing.
        if !ox.is_finite() || !oy.is_finite() || ox.abs() > 1.0e6 || oy.abs() > 1.0e6 {
            pen_x += g.advance as f64;
            continue;
        }
        let qx = (ox * px as f64).round() / px as f64;
        let qy = (oy * py as f64).round() / py as f64;
        let shift = [qx - ox, qy - oy];
        let first = contours.len();
        // Native rounds the scaled outline to os glyph-relative (its scan-converter contour list), then
        // positions it by the baseline origin — not `round(pen+glyph)`, whose round-half-away tie an integer
        // pen would flip at a .5-fraction point. So round the glyph outline with the pen at the origin and
        // translate the os segments by the (snapped) pen position.
        //
        // The pen snaps to a whole **oversampled pixel** (`CreateAlphaTexture*` merges each glyph into the run
        // bitmap at an integer sub-pixel offset): os-26.6 origin `round(ox·sx)·64`, *not*
        // `to_2614(round(ox·sx)/sx)`. Those agree only when `round(ox·sx)` is a multiple of `sx` (so the
        // device-1/64 round `to_2614` does first is exact); for ClearType (`sx=6`, 64⁄6 non-integer) the old
        // form landed 2 os-26.6 units (~0.005 px) off at phases 1/6·2/6·4/6·5/6, shifting the CT filter and
        // leaving a per-glyph residual that grew with run length (`textmodes-04`). Grayscale/aliased (`sx`
        // divides 64) were already exact.
        let pen_os = [(qx * sx as f64).round() as i32 * 64, (qy * sy as f64).round() as i32 * 64];
        let dev = |fx: f64, fy: f64| -> [f64; 2] {
            if let Some(e) = ob {
                return oblique_map(&e, fx, fy);
            }
            // The unscaled scaler maps font units through the transform scaled to 16.16 per font unit
            // and rounds each product to 1/64 device before adding the two axes, on the axes as well as
            // off them (r25 30913: aliased Georgia at 42 ppem lost a pixel to the exact float scale).
            let q = |m: f64, f: f64| ((m * scale * 65536.0).round() * f / 1024.0).round() / 64.0;
            [q(dt[0], fx) + q(dt[2], -fy), q(dt[1], fx) + q(dt[3], -fy)]
        };
        // Font-unit contours for this glyph: a simple glyf's points (gvar-interpolated for a variable
        // instance), or a static composite assembled from its components in font units. Native builds a
        // composite's implied on-curve midpoints in os-integer space just like a simple glyph; skrifa's
        // float Builder rounds them in float (1 os off at curve apexes — the café/é accent residual), so
        // only a CFF glyph, a variable composite, or a composite with a point-anchored / non-identity
        // component falls back to the Builder below.
        let fu_contours = raw.as_ref().and_then(|(gf, lc)| match lc.get_glyf(gid, gf).ok().flatten()? {
            Glyph::Simple(s) => {
                // A variable-font instance renders the gvar-interpolated outline (deltas added to the raw
                // glyf points; unchanged for the default instance / a static face).
                let inst = super::variations::instanced_simple(font, gid, run.coords, &s);
                let pts: Vec<[f64; 2]> = match &inst {
                    Some(ip) => ip.iter().map(|xy| [xy[0], xy[1]]).collect(),
                    None => s.points().map(|p| [p.x as f64, p.y as f64]).collect(),
                };
                let on: Vec<bool> = s.points().map(|p| p.on_curve).collect();
                let ends: Vec<usize> = s.end_pts_of_contours().iter().map(|e| e.get() as usize).collect();
                Some((pts, on, ends))
            }
            Glyph::Composite(_) if run.coords.is_empty() => composite_contours_fu(gf, lc, gid, 0),
            _ => None,
        }).map(|(pts, on, ends)| {
            let hinted = unscaled.as_ref().and_then(|u| u.hint(g.id)).filter(|h| run.coords.is_empty() && h.points.len() == pts.len());
            (hinted.map_or(pts, |h| h.points.iter().map(|p| [p.x as f64 / 64.0, p.y as f64 / 64.0]).collect()), on, ends)
        });
        match fu_contours {
            Some((pts_fu, on, ends)) => {
                // DWRITE_FONT_SIMULATIONS_BOLD on the unscaled-scaler (grid-fit-disabled) path: native
                // embolds the UNHINTED DESIGN outline (font units ×64 = glyph-local 26.6, y-up) with the
                // constant design strength via the per-point embolden, THEN scales to device and scans (the
                // overscale-bold path, captured byte-exact). Same edge-offset-and-intersect embold
                // (`apply_26dot6`, flip on, no re-anchor) the hinted device path uses; the amount is the
                // design strength (not the device `bold_amount`), so it does not scale with ppem.
                let pts_fu = match design_bold {
                    Some((bx, by)) => {
                        use super::outline_embold::Pt;
                        let mut cs: Vec<Vec<Pt>> = Vec::new();
                        let mut s0 = 0usize;
                        for &e in &ends {
                            let e = e.min(pts_fu.len().saturating_sub(1));
                            if e >= s0 && !pts_fu.is_empty() {
                                cs.push((s0..=e).map(|i| Pt { x: (pts_fu[i][0] * 64.0).round() as i32, y: (pts_fu[i][1] * 64.0).round() as i32, on: on[i] }).collect());
                            }
                            s0 = e + 1;
                        }
                        super::outline_embold::apply_26dot6(&mut cs, bx, by);
                        let mut flat = pts_fu;
                        let mut idx = 0usize;
                        for c in &cs { for p in c { if idx < flat.len() { flat[idx] = [p.x as f64 / 64.0, p.y as f64 / 64.0]; idx += 1; } } }
                        flat
                    }
                    None => pts_fu,
                };
                let pts: Vec<[f64; 2]> = pts_fu.iter().map(|p| dev(p[0], p[1])).collect();
                let mut start = 0usize;
                for e in ends {
                    if e >= start && e < pts.len() {
                        // Coarse 1/64-device round (native rounds a raw transformed outline to 1/64
                        // device); the sub-pixel pen phase is applied exactly below in os units.
                        let mut segs = outline_contour_segs(&pts[start..=e], &on[start..=e], sx, sy, false);
                        translate_segs(&mut segs, [pen_os[0] + nudge, pen_os[1]]);
                        contours.push(segs);
                    }
                    start = e + 1;
                }
            }
            // CFF, a variable composite, or a composite with a point-anchored / non-identity component:
            // skrifa resolves it (its pen's float midpoints are the fallback).
            None => {
                if let Some(glyph) = outlines.get(gid) {
                    let mut b = Builder { dt, scale, ob, pen, shift, sx, sy, nudge, contours: &mut contours, cur: Vec::new(), start: None, last: [0, 0] };
                    let loc = LocationRef::new(run.coords);
                    let _ = glyph.draw(DrawSettings::unhinted(Size::unscaled(), loc), &mut b);
                    b.close_contour();
                }
            }
        }
        if let Some(gl) = glyphs.as_mut() {
            gl.push((first..contours.len(), ((qx * 2.0).round() as i64 ^ (qy * 2.0).round() as i64) & 1 != 0));
        }
        pen_x += g.advance as f64;
    }
    contours
}

struct Builder<'a> {
    dt: &'a [f64; 6],
    scale: f64,
    /// The oblique simulation's per-font-unit transform entries ([`oblique_entries`]).
    ob: Option<[i32; 4]>,
    pen: [f64; 2],
    shift: [f64; 2],
    sx: i32,
    sy: i32,
    nudge: i32,
    contours: &'a mut Vec<Vec<Seg>>,
    cur: Vec<Seg>,
    start: Option<P>,
    last: P,
}

impl Builder<'_> {
    /// Font-unit point → oversampled 26.6 (y-up design → y-down device, world∘DPI, origin shift, ×os).
    fn pt(&self, x: f32, y: f32) -> P {
        let d = self.dt;
        let dev = if let Some(e) = self.ob {
            let g = oblique_map(&e, x as f64, y as f64);
            let (px, py) = (self.pen[0], self.pen[1]);
            [d[0] * px + d[2] * py + d[4] + self.shift[0] + g[0], d[1] * px + d[3] * py + d[5] + self.shift[1] + g[1]]
        } else {
            let xd = self.pen[0] + x as f64 * self.scale;
            let yd = self.pen[1] - y as f64 * self.scale;
            [d[0] * xd + d[2] * yd + d[4] + self.shift[0], d[1] * xd + d[3] * yd + d[5] + self.shift[1]]
        };
        let p = to_2614(dev, self.sx, self.sy, false); // unhinted outline: coarse 1/64-device round
        [p[0] + self.nudge, p[1]]
    }
    fn close_contour(&mut self) {
        if let Some(s) = self.start {
            if s != self.last {
                self.cur.push(Seg::Line(self.last, s));
            }
        }
        self.start = None;
        if !self.cur.is_empty() {
            self.contours.push(std::mem::take(&mut self.cur));
        }
    }
}

impl OutlinePen for Builder<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        self.close_contour();
        self.last = self.pt(x, y);
        self.start = Some(self.last);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.pt(x, y);
        self.cur.push(Seg::Line(self.last, p));
        self.last = p;
    }
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        let (c, p1) = (self.pt(cx, cy), self.pt(x, y));
        self.cur.push(Seg::Quad(self.last, c, p1));
        self.last = p1;
    }
    fn curve_to(&mut self, ax: f32, ay: f32, bx: f32, by: f32, x: f32, y: f32) {
        // ponytail: cubics (CFF) → two quads (midpoint); TrueType text fonts have none. Exact only
        // for quadratic outlines; upgrade to native CFF conversion if a CFF fixture needs it.
        let (p0, a, b, p1) = (self.last, self.pt(ax, ay), self.pt(bx, by), self.pt(x, y));
        let q0c = [(p0[0] + 3 * a[0]) / 4, (p0[1] + 3 * a[1]) / 4];
        let q1c = [(p1[0] + 3 * b[0]) / 4, (p1[1] + 3 * b[1]) / 4];
        let mid = [(q0c[0] + q1c[0]) / 2, (q0c[1] + q1c[1]) / 2];
        self.cur.push(Seg::Quad(p0, q0c, mid));
        self.cur.push(Seg::Quad(mid, q1c, p1));
        self.last = p1;
    }
    fn close(&mut self) {
        self.close_contour();
    }
}

// ---- bounds (the extrema pass + GetAlphaTextureBounds) -------------------------------------------

/// Oversampled control-point box `[left, top, right, bottom]` (os pixels), native's extrema pass:
/// `left/top = (min+31)>>6`, `right/bottom = (max+32)>>6`, a zero-width axis bumped +1.
fn extrema_box(contours: &[Vec<Seg>]) -> Option<[i32; 4]> {
    let (mut lo_x, mut lo_y, mut hi_x, mut hi_y) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    let mut pt = |p: P| {
        lo_x = lo_x.min(p[0]);
        lo_y = lo_y.min(p[1]);
        hi_x = hi_x.max(p[0]);
        hi_y = hi_y.max(p[1]);
    };
    for s in contours.iter().flatten() {
        match *s {
            Seg::Line(p0, p1) => {
                pt(p0);
                pt(p1);
            }
            Seg::Quad(p0, c, p1) => {
                pt(p0);
                pt(c);
                pt(p1);
            }
        }
    }
    if lo_x == i32::MAX {
        return None;
    }
    let (l, t) = ((lo_x + 31) >> 6, (lo_y + 31) >> 6);
    let (mut r, mut b) = ((hi_x + 32) >> 6, (hi_y + 32) >> 6);
    if l == r {
        r += 1;
    }
    if t == b {
        b += 1;
    }
    Some([l, t, r, b])
}

/// Native's extrema pass sets the glyph-bitmap collapse flag that the scan converter tests to mask
/// off the smart-dropout bit: it is set exactly when the extrema box **collapses** in a column axis —
/// `left == right` or `top == bottom` BEFORE the `+1` bump (a sub-column-wide or sub-row-tall sliver).
/// This is stricter than a final width/height of 1: a feature that is naturally one column wide
/// (`left + 1 == right`, e.g. a tall 1px vertical stroke) does NOT collapse, so native keeps smart
/// there. A collapsed sliver has no neighbour for the smart neighbour test, so native falls to simple
/// dropout to recover it.
fn extrema_collapsed(contours: &[Vec<Seg>]) -> bool {
    let (mut lo_x, mut lo_y, mut hi_x, mut hi_y) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for s in contours.iter().flatten() {
        let mut pt = |p: P| {
            lo_x = lo_x.min(p[0]);
            lo_y = lo_y.min(p[1]);
            hi_x = hi_x.max(p[0]);
            hi_y = hi_y.max(p[1]);
        };
        match *s {
            Seg::Line(p0, p1) => { pt(p0); pt(p1); }
            Seg::Quad(p0, c, p1) => { pt(p0); pt(c); pt(p1); }
        }
    }
    if lo_x == i32::MAX {
        return false;
    }
    ((lo_x + 31) >> 6) == ((hi_x + 32) >> 6) || ((lo_y + 31) >> 6) == ((hi_y + 32) >> 6)
}

// Signed floor/ceil division (`i32::div_floor`/`div_ceil` are still unstable — `int_roundings`).
fn div_floor(a: i32, b: i32) -> i32 {
    let q = a / b;
    if a % b != 0 && (a < 0) != (b < 0) { q - 1 } else { q }
}
fn div_ceil(a: i32, b: i32) -> i32 {
    let q = a / b;
    if a % b != 0 && (a < 0) == (b < 0) { q + 1 } else { q }
}

/// Divide the oversampled box down to device pixels (native `GetAlphaTextureBounds`): `left = ⌊/sx⌋`,
/// `right = ⌈/sx⌉` (same for y with `sy`). ClearType widens the os x box by ±2 sub-pixels first.
fn device_bounds(os: [i32; 4], sx: i32, sy: i32, cleartype: bool) -> [i32; 4] {
    let (mut l, mut r) = (os[0], os[2]);
    if cleartype {
        l -= 2;
        r += 2;
    }
    // Right/bottom bound = floor(max)+1, NOT ceil: native's glyph bitmap includes the device column/row
    // that a max coordinate lands *on* when it sits exactly on a cell boundary (a right/bottom vertical/
    // horizontal tangent, max ≡ 0 mod s). The per-row fill is `[on, off-1]` so it drops that boundary
    // cell, and native's vertical-dropout pass recovers the tangent sample into it — but only if the bitmap holds
    // the column. `floor+1 == ceil` off a boundary, so this only widens the exact-tangent case (verified:
    // Courier New '0'@16 right edge os[2]=96 → native bounds [5,5,13,15], not our old [5,5,12,15]).
    let out = [div_floor(l, sx), div_floor(os[1], sy), div_floor(r, sx) + 1, div_floor(os[3], sy) + 1];
    // ponytail: a device glyph box this large is never a real glyph — a huge em (native refuses it at
    // device em ≥ 32768 with E_INVALIDARG) or an extreme world transform. Rasterising it would size
    // the oversampled `scan_bitmap` bool grid off the box, so a 1e5-px device em (e.g. em -1e9 under a
    // 1e-4 transform) allocates ~176 GB and aborts the process — a DoS fingerprint. Collapse to empty
    // so every caller's `r <= l` guard draws nothing (native draws nothing for the refused op too).
    // (Huge on-screen glyphs don't reach this path: ppem ≥ 100/350 routes to `fill_outline`, the
    // geometry-fill outline realisation, whose triangles `fill_triangle` clips to the target.)
    const MAX_AREA: i64 = 1 << 24; // 16.7M device px (≈4096²); any real glyph is orders of magnitude smaller
    let (dw, dh) = (out[2] as i64 - out[0] as i64, out[3] as i64 - out[1] as i64);
    if dw > 0 && dh > 0 && dw * dh > MAX_AREA {
        return [0, 0, 0, 0];
    }
    out
}

/// Report `ceil(max)` instead of the `floor(max)+1` [`device_bounds`] *allocates* on a max edge whose
/// coordinate sits exactly on a cell boundary (`floor+1 > ceil`), when the scan left that speculative
/// row/column empty. Native allocates the extra cell so a curve's vertical/horizontal tangent can be
/// recovered into it by the vertical/horizontal dropout pass (Courier '0'@16's col-12 right edge), but its
/// reported `GetAlphaTextureBounds` is `ceil` when no sample landed there — a straight on-boundary edge
/// (`H`@20 bottom, the aliased stems) or, via the resolve's y-negated frame, the device *top* of a
/// grid-fit curve (`o`/`e`@20). Off-boundary edges (`floor+1 == ceil`, e.g. a sub-threshold right
/// column that holds the curve's real extremum) are not speculative and are always kept. Operates in the
/// resolve's own frame (`bpp` bytes/px); the caller's flip then carries the trimmed edge to device space.
fn drop_empty_tangent(l: i32, t: i32, mut r: i32, mut b: i32, mut data: Vec<u8>, os: [i32; 4], sx: i32, sy: i32, cleartype: bool, bpp: usize) -> AlphaTexture {
    let rx = if cleartype { os[2] + 2 } else { os[2] }; // device_bounds widens the CT x box by +2 first
    let spec_r = div_floor(rx, sx) + 1 > div_ceil(rx, sx);
    let spec_b = div_floor(os[3], sy) + 1 > div_ceil(os[3], sy);
    let w = (r - l) as usize;
    let h = (b - t) as usize;
    if spec_b && h > 1 && data[(h - 1) * w * bpp..h * w * bpp].iter().all(|&v| v == 0) {
        data.truncate((h - 1) * w * bpp);
        b -= 1;
    }
    let h = (b - t) as usize;
    if spec_r && w > 1 && (0..h).all(|y| data[(y * w + w - 1) * bpp..(y * w + w) * bpp].iter().all(|&v| v == 0)) {
        let mut nd = Vec::with_capacity((w - 1) * h * bpp);
        for y in 0..h {
            nd.extend_from_slice(&data[y * w * bpp..(y * w + w - 1) * bpp]);
        }
        data = nd;
        r -= 1;
    }
    AlphaTexture { left: l, top: t, right: r, bottom: b, data }
}

// ---- the scan converter (decompose the spline, then the line / spline DDA) ----------------------

/// Per-row crossing columns collected from the scan converter. `rows[r - row0]` holds the
/// `(column, winding-direction, subpixel-intercept)` of each edge crossing scanline row `r`, exactly
/// as the native horizontal-mark lists do; the sweep ([`scan_bitmap`]) pairs them `[on, off−1]` by nonzero
/// winding. `subpix` is the exact x-intercept (subcol·64) from native's line, spline, or endpoint
/// callback, used only to position a dropout under `SCANTYPE & 4`. [`NO_SUBPIX`] permits synthetic
/// crossings without an intercept to use the simple recovery position.
struct Scan {
    rows: Vec<Vec<(i32, i32, i32)>>,
    row0: i32,
}

/// Sentinel for a synthetic crossing without a sub-pixel intercept.
const NO_SUBPIX: i32 = i32::MIN;

impl Scan {
    fn new(row0: i32, nrows: usize) -> Scan {
        Scan { rows: (0..nrows).map(|_| Vec::new()).collect(), row0 }
    }
    fn mark(&mut self, col: i32, row: i32, dir: i32, subpix: i32) {
        let i = row - self.row0;
        if i >= 0 && (i as usize) < self.rows.len() {
            self.rows[i as usize].push((col, dir, subpix));
        }
    }
}

/// The exact x-intercept (oversampled 26.6, subcol·64) of
/// the straight edge `p0→p1` at scanline `row`'s sample centre (`row·64 + 32`). The transpose
/// (`transpose`) makes the same code serve the vertical dropout (native `CalcVertLineSubpix`).
fn line_subpix(p0: P, p1: P, row: i32) -> i32 {
    let dy = (p1[1] - p0[1]) as i64;
    if dy == 0 {
        return p0[0]; // horizontal edge never crosses a scanline; value unused
    }
    p0[0] + compdiv_n(dy, (p1[0] - p0[0]) as i64 * (row as i64 * 64 - p0[1] as i64 + 32))
}

/// Native dwrite's horizontal subpixel bisect: bisect the monotone quadratic in
/// integer 26.6 until its midpoint reaches the requested sample centre. The transpose
/// also supplies CalcVertSpSubpix for exact-intercept dropout modes.
fn spline_subpix(mut a: P, mut c: P, mut b: P, row: i32) -> i32 {
    if a[1] > b[1] { std::mem::swap(&mut a, &mut b); }
    let y = row * 64 + 32;
    loop {
        let m = [(a[0] + b[0] + 2 * c[0] + 1) >> 2, (a[1] + b[1] + 2 * c[1] + 1) >> 2];
        if m[1] == y { return m[0]; }
        if m[1] > y {
            c = [(a[0] + c[0]) >> 1, (a[1] + c[1]) >> 1];
            b = m;
        } else {
            c = [(b[0] + c[0]) >> 1, (b[1] + c[1]) >> 1];
            a = m;
        }
    }
}

/// Floor to a multiple of 64 (native `x & 0xFFFFFFC0`).
#[inline]
fn floor64(x: i32) -> i32 {
    x & !63
}

/// Scan-convert a straight segment (the line DDA, fill/horizontal marks only): emit, for every
/// oversampled scanline row the edge spans, the column where it crosses — a Bresenham cell walk whose
/// row/column entry rounding is the asymmetric `(x+32)>>6` / `(x−33)` the native engine uses.
fn calc_line(scan: &mut Scan, p0: P, p1: P) {
    let (x0, y0, x1, y1) = (p0[0], p0[1], p1[0], p1[1]);
    if y1 == y0 {
        return; // horizontal edge: no fill crossing
    }
    let dir = if y1 < y0 { -1 } else { 1 };
    // y setup: first_row first row, abs_dy |Δy|, subpix_y sub-pixel y offset, row_step row step, row_count row count.
    let (row_step, first_row, abs_dy, subpix_y, y_parity);
    let row_count;
    if y1 < y0 {
        row_step = -1;
        let y_snap = floor64(y0 - 33) + 32;
        first_row = y_snap >> 6;
        abs_dy = y0 - y1;
        subpix_y = y0 - y_snap;
        y_parity = 1;
        row_count = first_row - ((y1 + 32) >> 6) + 1;
    } else {
        row_step = 1;
        let base = floor64(y0 + 32) + 32;
        first_row = base >> 6;
        abs_dy = y1 - y0;
        subpix_y = base - y0;
        y_parity = 0;
        row_count = ((y1 - 33) >> 6) - first_row + 1;
    }
    if x1 == x0 {
        // vertical edge: constant column, one mark per spanned row.
        let col = if y1 <= y0 { (x0 + 32) >> 6 } else { (x0 + 31) >> 6 };
        let mut r = first_row;
        for _ in 0..row_count {
            scan.mark(col, r, dir, x0); // vertical edge: x-intercept is the constant x
            r += row_step;
        }
        return;
    }
    // x setup: first_col first column, abs_dx |Δx|, subpix_x sub-pixel x offset, col_step col step, col_count col count.
    let (col_step, col_entry, first_col, subpix_x, x_parity, abs_dx, col_count);
    if x1 < x0 {
        col_step = -1;
        col_entry = 1;
        let x_snap = floor64(x0 - 33) + 32;
        first_col = x_snap >> 6;
        subpix_x = x0 - x_snap;
        abs_dx = x0 - x1;
        x_parity = 1 - y_parity;
        col_count = first_col - ((x1 + 32) >> 6) + 1;
    } else {
        col_step = 1;
        col_entry = 0;
        let base = floor64(x0 + 32) + 32;
        first_col = base >> 6;
        subpix_x = base - x0;
        abs_dx = x1 - x0;
        x_parity = y_parity;
        col_count = ((x1 - 33) >> 6) - first_col + 1;
    }
    // Diagonal Bresenham: mark the current column on each row step.
    let mut col = col_entry + first_col;
    let mut err = subpix_y * abs_dx - subpix_x * abs_dy + x_parity;
    let mut r = first_row;
    for _ in 0..(row_count + col_count) {
        if err > 0 {
            col += col_step;
            err -= 64 * abs_dy;
        } else {
            scan.mark(col, r, dir, line_subpix([x0, y0], [x1, y1], r));
            r += row_step;
            err += 64 * abs_dx;
        }
    }
}

/// Both-axis line DDA (the dropout-enabled branch, mode bit 1 clear): the SAME single DDA walk emits a
/// HORIZONTAL crossing into `h` when the error favours a row step and a VERTICAL crossing into `v` when it
/// favours a column step — native marks both the per-row and per-column lists
/// in one pass, where our [`calc_line`] (the dropout-disabled branch) only marks `h`. `v` is indexed by column
/// (its `mark(value, index)` stores the row value under the column), the transpose `collect_crossings`'s
/// vertical pass used to build separately — but this reproduces native's crossing lists, which the
/// transpose does not (native marks vert on the ORIGINAL `err>0`, the transpose on the TRANSPOSED `err<=0`).
fn calc_line_bi(h: &mut Scan, v: &mut Scan, p0: P, p1: P) {
    let (x0, y0, x1, y1) = (p0[0], p0[1], p1[0], p1[1]);
    // y setup (first_row first row, abs_dy |Δy|, subpix_y sub-pixel y, row_step row step, vrow_off vert sub-pixel row offset, row_count).
    let (row_step, vrow_off, first_row, abs_dy, subpix_y, y_parity, row_count);
    if y1 < y0 {
        row_step = -1; vrow_off = 1;
        let y_snap = floor64(y0 - 33) + 32;
        first_row = y_snap >> 6; abs_dy = y0 - y1; subpix_y = y0 - y_snap; y_parity = 1;
        row_count = (first_row - ((y1 + 32) >> 6)) + 1;
    } else {
        row_step = 1; vrow_off = 0;
        let base = floor64(y0 + 32) + 32;
        first_row = base >> 6; abs_dy = y1 - y0; subpix_y = base - y0; y_parity = 0;
        row_count = (((y1 - 33) >> 6) - first_row) + 1;
    }
    let dir_h = if y1 < y0 { -1 } else { 1 };
    // x setup (first_col first column, abs_dx |Δx|, subpix_x sub-pixel x, col_step col step, col_entry entry offset, col_count).
    let (col_step, col_entry, first_col, subpix_x, x_parity, abs_dx, col_count);
    if x1 < x0 {
        col_step = -1; col_entry = 1;
        let x_snap = floor64(x0 - 33) + 32;
        first_col = x_snap >> 6; subpix_x = x0 - x_snap; abs_dx = x0 - x1; x_parity = 1 - y_parity;
        col_count = (first_col - ((x1 + 32) >> 6)) + 1;
    } else {
        col_step = 1; col_entry = 0;
        let base = floor64(x0 + 32) + 32;
        first_col = base >> 6; subpix_x = base - x0; abs_dx = x1 - x0; x_parity = y_parity;
        col_count = (((x1 - 33) >> 6) - first_col) + 1;
    }
    let dir_v = col_step;
    if y1 == y0 {
        // Horizontal edge: one VERTICAL crossing per spanned column.
        let h_edge_y = if x1 >= x0 { y0 } else { y0 - 1 };
        let row = (h_edge_y + 32) >> 6;
        let mut col = first_col;
        for _ in 0..col_count {
            v.mark(row, col, dir_v, y0); // horizontal edge: y-intercept is the constant y
            col += col_step;
        }
        return;
    }
    if x1 == x0 {
        // Vertical edge: one HORIZONTAL crossing per spanned row.
        let v_edge_x = if y1 <= y0 { x0 } else { x0 - 1 };
        let col = (v_edge_x + 32) >> 6;
        let mut r = first_row;
        for _ in 0..row_count {
            h.mark(col, r, dir_h, x0);
            r += row_step;
        }
        return;
    }
    // Diagonal: err<=0 ⇒ horiz mark (col+col_entry, row) + step row; err>0 ⇒ vert mark (col, row+vrow_off) + step col.
    let mut col = first_col;
    let mut r = first_row;
    let mut err = subpix_y * abs_dx - subpix_x * abs_dy + x_parity;
    for _ in 0..(row_count + col_count) {
        if err <= 0 {
            h.mark(col + col_entry, r, dir_h, line_subpix(p0, p1, r));
            r += row_step;
            err += 64 * abs_dx;
        } else {
            // Vert mark's exact-intercept subpix = the edge's y-intercept at this column (line_subpix on
            // the x↔y-swapped edge), for the vertical-dropout SCANTYPE&4 recovery.
            v.mark(r + vrow_off, col, dir_v, line_subpix([p0[1], p0[0]], [p1[1], p1[0]], col));
            col += col_step;
            err -= 64 * abs_dy;
        }
    }
}

/// Validation helper (M2): scan a contour set (lines AND quadratics) through the both-axis
/// [`calc_line_bi`] / [`calc_spline_bi`] and return the native `(col, row)` H and V crossing marks
/// (matching dw_oracle's adder record) for comparison against the native fsc oracle. Points are in the
/// fsc frame (os-26.6: device·sx·64 in x, device·64 in y); `on` flags mark on-curve points (consecutive
/// off-curve points get the implied `(a+b+1)>>1` midpoint). `bounds` = `[row0, nrows, col0, ncols]`.
pub fn biscan_marks(xs: &[i32], ys: &[i32], on: &[bool], ends: &[usize], bounds: [i32; 4]) -> (Vec<(i32, i32)>, Vec<(i32, i32)>) {
    let [r0, nr, c0, nc] = bounds;
    let mid = |a: P, b: P| [(a[0] + b[0] + 1) >> 1, (a[1] + b[1] + 1) >> 1];
    // The points are already os-26.6 (fsc frame); build each contour's Segs directly with native's integer
    // implied midpoint `(a+b+1)>>1` (the `outline_contour_segs` walk without the device round-trip).
    let mut contours: Vec<Vec<Seg>> = Vec::new();
    let mut s = 0usize;
    for &e in ends {
        let n = e - s + 1;
        let q: Vec<P> = (s..=e).map(|i| [xs[i], ys[i]]).collect();
        let onc: Vec<bool> = (s..=e).map(|i| on[i]).collect();
        let first_on = (0..n).find(|&i| onc[i]);
        let s0 = first_on.unwrap_or(0);
        let start = match first_on { Some(i) => q[i], None => mid(q[0], q[n - 1]) };
        let mut segs: Vec<Seg> = Vec::new();
        let mut last = start;
        let mut i = 1;
        while i <= n {
            let idx = (s0 + i) % n;
            if onc[idx] {
                segs.push(Seg::Line(last, q[idx])); last = q[idx]; i += 1;
            } else {
                let nidx = (s0 + i + 1) % n;
                let end = if onc[nidx] { q[nidx] } else { mid(q[idx], q[nidx]) };
                segs.push(Seg::Quad(last, q[idx], end)); last = end;
                i += if onc[nidx] { 2 } else { 1 };
            }
        }
        contours.push(segs);
        s = e + 1;
    }
    let (h, v) = collect_crossings_bi(&contours, r0, nr as usize, c0, nc as usize, true);
    let hm = h.rows.iter().enumerate().flat_map(|(i, row)| row.iter().map(move |c| (c.0, r0 + i as i32))).collect();
    let vm = v.rows.iter().enumerate().flat_map(|(i, row)| row.iter().map(move |c| (c0 + i as i32, c.0))).collect();
    (hm, vm)
}

/// Bit length of `n` (native lookup): `⌊log2 n⌋ + 1`, 0 for 0.
#[inline]
fn bitlen(n: u32) -> i32 {
    (32 - n.leading_zeros()) as i32
}

/// Forward-difference precision shift (from a native table indexed by the span and determinant bit-lengths): 0 for small arcs, up to 3.
#[inline]
fn prec_shift(i: i32) -> i32 {
    if i < 30 {
        0
    } else if i < 33 {
        1
    } else if i < 36 {
        2
    } else {
        3
    }
}

/// Scan-convert a y-monotone, ≤3200-chord quadratic arc (the spline DDA, fill/horizontal marks):
/// a second-order forward-difference DDA that marks the crossing column of every spanned row. The
/// `v*` names mirror the native routine; the precision shift keeps the differences inside 32 bits.
#[allow(clippy::too_many_arguments)]
fn calc_spline(scan: &mut Scan, p0: P, pc: P, p1: P) {
    let (x0, y0, cx, cy, x1, y1) = (p0[0], p0[1], pc[0], pc[1], p1[0], p1[1]);
    if std::env::var_os("D2D_ARCDUMP").is_some() {
        eprintln!("ARC p0=[{},{}] pc=[{},{}] p1=[{},{}]", x0, y0, cx, cy, x1, y1);
    }
    let dir = if y1 <= y0 { -1 } else { 1 };
    // y setup.
    let (row_step, mut first_row, last_row, mut subpix_y, mut dy0, mut dy);
    if y1 <= y0 {
        row_step = -1;
        let y_snap = floor64(y0 - 33) + 32;
        subpix_y = y0 - y_snap;
        dy0 = y0 - cy;
        first_row = y_snap >> 6;
        last_row = ((y1 + 32) >> 6) - 1;
        dy = y0 - y1;
    } else {
        row_step = 1;
        let base = floor64(y0 + 32) + 32;
        first_row = base >> 6;
        last_row = ((y1 - 33) >> 6) + 1;
        dy0 = cy - y0;
        subpix_y = base - y0;
        dy = y1 - y0;
    }
    let y_parity = if y1 <= y0 { 1 } else { 0 };
    // x setup. `err_seed` (the parity seed of the initial error term `err`) is `y_parity` for an x-increasing
    // arc but `1 - y_parity` for an x-decreasing one (`x1 <= x0`) — native flips it in the `x1 <= x0`
    // branch. Dropping that flip leaves the initial `err` off by one on a borderline x-decreasing arc,
    // stepping a column one row early/late at the tangent.
    let (err_seed, col_step, first_col, last_col, mut subpix_x, mut dx0, mut dx, col_entry);
    if x1 <= x0 {
        err_seed = 1 - y_parity;
        col_step = -1;
        col_entry = 1;
        let x_snap = floor64(x0 - 33) + 32;
        last_col = ((x1 + 32) >> 6) - 1;
        subpix_x = x0 - x_snap;
        first_col = x_snap >> 6;
        dx0 = x0 - cx;
        dx = x0 - x1;
    } else {
        err_seed = y_parity;
        col_step = 1;
        col_entry = 0;
        first_col = (floor64(x0 + 32) + 32) >> 6;
        subpix_x = (floor64(x0 + 32) + 32) - x0;
        last_col = ((x1 - 33) >> 6) + 1;
        dx0 = cx - x0;
        dx = x1 - x0;
    }
    // Vertical-only (fill) early outs.
    if first_row == last_row {
        return;
    }
    if first_col == last_col {
        let col = col_entry + first_col;
        let mut r = first_row;
        while r != last_row {
            scan.mark(col, r, dir, spline_subpix(p0, pc, p1, r));
            r += row_step;
        }
        return;
    }
    // Determinant (its sign selects the loop); then the precision shift.
    let det = dx0 * dy - dy0 * dx;
    let det_bits = bitlen((2i64 * det as i64).unsigned_abs() as u32);
    let span_bits = bitlen(dy.max(dx) as u32);
    let prec = prec_shift(span_bits + det_bits);
    let shift = 6 - prec;
    if prec > 0 {
        let add = 1 << (prec - 1);
        subpix_x = (add + subpix_x) >> prec;
        dx = (add + dx) >> prec;
        subpix_y = (add + subpix_y) >> prec;
        dy = (add + dy) >> prec;
        dx0 = (add + dx0) >> prec;
        dy0 = (add + dy0) >> prec;
    }
    let curv2 = 2 * (dx0 * dy - dy0 * dx);
    let ddx = dx - 2 * dx0;
    let ddy = dy - 2 * dy0;
    let dy0_curv = dy0 * curv2;
    let ddy_sq = ddy * ddy;
    let ddx_ddy = ddx * ddy;
    let ddx_sq = ddx * ddx;
    let dy0_curv2 = dy0 * curv2;
    let unit = 1 << shift;
    let neg_cross = -ddx_ddy;
    let dx0_curv = dx0 * curv2;
    let (incr_c, incr_cross, mut acc_c, init_c, mut acc_r, incr_r, init_r);
    if span_bits <= 7 {
        let cross2 = 2 * neg_cross;
        let subpix_cross = subpix_y * 2 * neg_cross;
        acc_c = (subpix_cross + 2 * dy0_curv + ddy_sq * (unit + 2 * subpix_x)) << shift;
        acc_r = (-2 * dx0_curv + ddx_sq * (unit + 2 * subpix_y) + subpix_x * cross2) << shift;
        incr_c = ddy_sq << (2 * shift);
        incr_cross = cross2 << (2 * shift);
        init_c = subpix_cross + 2 * dy0_curv + subpix_x * ddy_sq;
        incr_r = ddx_sq << (2 * shift);
        init_r = subpix_y * (-2 * dx0_curv + subpix_y * ddx_sq);
    } else {
        let half = unit >> 1;
        incr_c = ddy_sq << (shift - 1);
        incr_cross = neg_cross << shift;
        acc_c = dy0_curv2 + subpix_y * neg_cross + ddy_sq * (half + subpix_x);
        incr_r = ddx_sq << (shift - 1);
        init_c = (dy0_curv2 + subpix_y * neg_cross + subpix_x * (ddy_sq >> 1)) >> shift;
        acc_r = subpix_x * neg_cross + ddx_sq * (subpix_y + half) - dx0_curv;
        init_r = subpix_y * ((subpix_y * (ddx_sq >> 1) - dx0_curv) >> shift);
    }
    let incr_cross_c = incr_cross;
    let mut thr_c = incr_c;
    let mut thr_r = incr_r;
    let mut step_c = 2 * incr_c;
    let mut err = err_seed + init_r + subpix_x * init_c;
    let step_r = 2 * incr_r;
    let mut cur_col = col_entry + first_col;
    let col_end = col_entry + last_col;
    if curv2 <= 0 {
        while cur_col != col_end {
            if first_row == last_row {
                return;
            }
            if err < 0 || acc_c > thr_c {
                scan.mark(cur_col, first_row, dir, spline_subpix(p0, pc, p1, first_row));
                err += acc_r;
                first_row += row_step;
                thr_c = incr_c;
                acc_c += incr_cross_c;
                acc_r += step_r;
                step_c = 2 * incr_c;
            } else {
                cur_col += col_step;
                err += acc_c;
                thr_c = incr_c;
                acc_c += step_c;
                acc_r += incr_cross_c;
                step_c = 2 * incr_c;
            }
        }
        while first_row != last_row {
            scan.mark(cur_col, first_row, dir, spline_subpix(p0, pc, p1, first_row));
            first_row += row_step;
        }
    } else {
        loop {
            if cur_col == col_end {
                break;
            }
            if first_row == last_row {
                return;
            }
            if err < 0 || acc_r > thr_r {
                cur_col += col_step;
                let tmp = acc_c;
                acc_c += step_c;
                err += tmp;
                thr_r = incr_r;
                acc_r += incr_cross_c;
                step_c = 2 * incr_c;
            } else {
                scan.mark(cur_col, first_row, dir, spline_subpix(p0, pc, p1, first_row));
                err += acc_r;
                first_row += row_step;
                acc_c += incr_cross_c;
                thr_r = incr_r;
                acc_r += step_r;
                step_c = 2 * incr_c;
            }
        }
        while first_row != last_row {
            scan.mark(cur_col, first_row, dir, spline_subpix(p0, pc, p1, first_row));
            first_row += row_step;
        }
    }
}

/// Both-axis spline DDA (the dropout-enabled branch, mode bit 1 clear): the SAME second-order
/// forward-difference DDA that [`calc_spline`] runs, but it marks a HORIZONTAL crossing into `h` on a row
/// step and a VERTICAL crossing into `v` on a column step (native's per-row / per-column adders) — where
/// [`calc_spline`] (the dropout-disabled branch) marks only `h` on row steps and steps columns silently. `v` is
/// indexed by column (`mark(value, index)`). Setup is identical to [`calc_spline`]; only the loop and the
/// two leftover tails differ. Verified against the native fsc oracle (dw_oracle.py) on curved glyphs.
#[allow(clippy::too_many_arguments)]
fn calc_spline_bi(h: &mut Scan, v: &mut Scan, p0: P, pc: P, p1: P) {
    let (x0, y0, cx, cy, x1, y1) = (p0[0], p0[1], pc[0], pc[1], p1[0], p1[1]);
    let dir_h = if y1 <= y0 { -1 } else { 1 };
    let (row_step, mut first_row, last_row, mut subpix_y, mut dy0, mut dy);
    if y1 <= y0 {
        row_step = -1;
        let y_snap = floor64(y0 - 33) + 32;
        subpix_y = y0 - y_snap; dy0 = y0 - cy; first_row = y_snap >> 6; last_row = ((y1 + 32) >> 6) - 1; dy = y0 - y1;
    } else {
        row_step = 1;
        let base = floor64(y0 + 32) + 32;
        first_row = base >> 6; last_row = ((y1 - 33) >> 6) + 1; dy0 = cy - y0; subpix_y = base - y0; dy = y1 - y0;
    }
    let y_parity = if y1 <= y0 { 1 } else { 0 };
    let vmark_row_off = y_parity; // vertical mark's row offset (native vmark_row_off = y_parity)
    let (err_seed, col_step, mut first_col, last_col, mut subpix_x, mut dx0, mut dx, col_entry);
    if x1 <= x0 {
        err_seed = 1 - y_parity; col_step = -1; col_entry = 1;
        let x_snap = floor64(x0 - 33) + 32;
        last_col = ((x1 + 32) >> 6) - 1; subpix_x = x0 - x_snap; first_col = x_snap >> 6; dx0 = x0 - cx; dx = x0 - x1;
    } else {
        err_seed = y_parity; col_step = 1; col_entry = 0;
        first_col = (floor64(x0 + 32) + 32) >> 6; subpix_x = (floor64(x0 + 32) + 32) - x0;
        last_col = ((x1 - 33) >> 6) + 1; dx0 = cx - x0; dx = x1 - x0;
    }
    let dir_v = col_step;
    // Vert mark's exact-intercept subpix = the arc's y-intercept at the column (spline_subpix on the
    // x↔y-swapped arc), for the vertical-dropout SCANTYPE&4 recovery.
    let vsub = |col: i32| spline_subpix([p0[1], p0[0]], [pc[1], pc[0]], [p1[1], p1[0]], col);
    // Native checks `first_col==last_col` first (vertical arc → horizontal marks), then `first_row==last_row` (horizontal arc →
    // vertical marks), in the enabled branch.
    if first_col == last_col {
        let col = col_entry + first_col;
        let mut r = first_row;
        while r != last_row {
            h.mark(col, r, dir_h, spline_subpix(p0, pc, p1, r));
            r += row_step;
        }
        return;
    }
    if first_row == last_row {
        let row_val = vmark_row_off + first_row; // the vertical mark's row, held constant
        let mut col = first_col;
        while col != last_col {
            v.mark(row_val, col, dir_v, vsub(col));
            col += col_step;
        }
        return;
    }
    let det = dx0 * dy - dy0 * dx;
    let det_bits = bitlen((2i64 * det as i64).unsigned_abs() as u32);
    let span_bits = bitlen(dy.max(dx) as u32);
    let prec = prec_shift(span_bits + det_bits);
    let shift = 6 - prec;
    if prec > 0 {
        let add = 1 << (prec - 1);
        subpix_x = (add + subpix_x) >> prec; dx = (add + dx) >> prec; subpix_y = (add + subpix_y) >> prec;
        dy = (add + dy) >> prec; dx0 = (add + dx0) >> prec; dy0 = (add + dy0) >> prec;
    }
    let curv2 = 2 * (dx0 * dy - dy0 * dx);
    let ddx = dx - 2 * dx0;
    let ddy = dy - 2 * dy0;
    let dy0_curv = dy0 * curv2;
    let ddy_sq = ddy * ddy;
    let ddx_ddy = ddx * ddy;
    let ddx_sq = ddx * ddx;
    let dy0_curv2 = dy0 * curv2;
    let unit = 1 << shift;
    let neg_cross = -ddx_ddy;
    let dx0_curv = dx0 * curv2;
    let (incr_c, incr_cross, mut acc_c, init_c, mut acc_r, incr_r, init_r);
    if span_bits <= 7 {
        let cross2 = 2 * neg_cross;
        let subpix_cross = subpix_y * 2 * neg_cross;
        acc_c = (subpix_cross + 2 * dy0_curv + ddy_sq * (unit + 2 * subpix_x)) << shift;
        acc_r = (-2 * dx0_curv + ddx_sq * (unit + 2 * subpix_y) + subpix_x * cross2) << shift;
        incr_c = ddy_sq << (2 * shift); incr_cross = cross2 << (2 * shift); init_c = subpix_cross + 2 * dy0_curv + subpix_x * ddy_sq;
        incr_r = ddx_sq << (2 * shift); init_r = subpix_y * (-2 * dx0_curv + subpix_y * ddx_sq);
    } else {
        let half = unit >> 1;
        incr_c = ddy_sq << (shift - 1); incr_cross = neg_cross << shift; acc_c = dy0_curv2 + subpix_y * neg_cross + ddy_sq * (half + subpix_x);
        incr_r = ddx_sq << (shift - 1); init_c = (dy0_curv2 + subpix_y * neg_cross + subpix_x * (ddy_sq >> 1)) >> shift;
        acc_r = subpix_x * neg_cross + ddx_sq * (subpix_y + half) - dx0_curv; init_r = subpix_y * ((subpix_y * (ddx_sq >> 1) - dx0_curv) >> shift);
    }
    let incr_cross_c = incr_cross;
    let mut thr_c = incr_c;
    let mut thr_r = incr_r;
    let mut err = err_seed + init_r + subpix_x * init_c;
    let step_r = 2 * incr_r;
    // Both-axis forward-difference walk (native enabled branch): HORIZ on a row step, VERT on a column step.
    if curv2 <= 0 {
        loop {
            if first_col == last_col { break; }
            if first_row == last_row { break; }
            if err < 0 || acc_c > thr_c {
                h.mark(first_col + col_entry, first_row, dir_h, spline_subpix(p0, pc, p1, first_row));
                first_row += row_step; err += acc_r; acc_c += incr_cross_c; thr_c = incr_c; acc_r += step_r;
            } else {
                v.mark(first_row + vmark_row_off, first_col, dir_v, vsub(first_col));
                first_col += col_step; err += acc_c; acc_c += 2 * incr_c; thr_c = incr_c; acc_r += incr_cross_c;
            }
        }
        while first_col != last_col { v.mark(first_row + vmark_row_off, first_col, dir_v, vsub(first_col)); first_col += col_step; }
        while first_row != last_row { h.mark(first_col + col_entry, first_row, dir_h, spline_subpix(p0, pc, p1, first_row)); first_row += row_step; }
    } else {
        'outer: loop {
            loop {
                if first_col == last_col { break 'outer; }
                if first_row == last_row {
                    while first_col != last_col { v.mark(first_row + vmark_row_off, first_col, dir_v, vsub(first_col)); first_col += col_step; }
                    break 'outer;
                }
                if err >= 0 && acc_r <= thr_r { break; }
                v.mark(first_row + vmark_row_off, first_col, dir_v, vsub(first_col));
                let tmp = acc_c; first_col += col_step; acc_c += 2 * incr_c; err += tmp; thr_r = incr_r; acc_r += incr_cross_c;
            }
            h.mark(first_col + col_entry, first_row, dir_h, spline_subpix(p0, pc, p1, first_row));
            first_row += row_step; acc_c += incr_cross_c; err += acc_r; thr_r = incr_r; acc_r += step_r;
        }
        while first_row != last_row { h.mark(first_col + col_entry, first_row, dir_h, spline_subpix(p0, pc, p1, first_row)); first_row += row_step; }
    }
}

/// A monotone, ≤3200-chord outline piece in oversampled 26.6, ready to scan: a line or an arc.
#[derive(Clone, Copy)]
enum Piece {
    Line(P, P),
    Arc(P, P, P),
}

impl Piece {
    fn start(self) -> P {
        match self {
            Piece::Line(a, _) | Piece::Arc(a, _, _) => a,
        }
    }
    fn end(self) -> P {
        match self {
            Piece::Line(_, b) | Piece::Arc(_, _, b) => b,
        }
    }
    fn render(self, scan: &mut Scan) {
        match self {
            Piece::Line(a, b) => calc_line(scan, a, b),
            Piece::Arc(a, c, b) => calc_spline(scan, a, c, b),
        }
    }
}

/// Split a quadratic at its x/y extrema and bisect until each chord ≤ 3200 (1/64) units, pushing each
/// resulting monotone piece in contour order — native's spline decomposition (collinear
/// pieces become lines). Recursion matches the native midpoint `(a+b)>>1` and extremum `CompDiv` splits,
/// so each piece is exactly the arc native rasterizes, and the pieces' shared endpoints are the vertices
/// the topology pass ([`check_horiz_topology`]) inspects.
fn split_quad(out: &mut Vec<Piece>, mut x0: i32, mut y0: i32, mut cx: i32, mut cy: i32, x2: i32, y2: i32) {
    loop {
        let dxce = x2 - cx;
        let dxsc = cx - x0;
        let dysc = cy - y0;
        let dyce = y2 - cy;
        if (dysc > 0 && dyce < 0) || (dysc < 0 && dyce > 0) {
            // y-extremum inside: split there.
            let n = (dysc - dyce) as i64;
            let lc_x = x0 + compdiv_n(n, dysc as i64 * dxsc as i64);
            let rc_x = cx + compdiv_n(n, dysc as i64 * dxce as i64);
            let split_x = compdiv_n(n, dysc as i64 * (rc_x - lc_x) as i64) + lc_x;
            let split_y = y0 + compdiv_n(n, dysc as i64 * dysc as i64);
            split_quad(out, x0, y0, lc_x, split_y, split_x, split_y);
            y0 = split_y;
            x0 = split_x;
            cx = rc_x;
            cy = split_y;
            continue;
        }
        if (dxsc > 0 && dxce < 0) || (dxsc < 0 && dxce > 0) {
            // x-extremum inside: split there.
            let n = (dxsc - dxce) as i64;
            let lc_y = y0 + compdiv_n(n, dxsc as i64 * dysc as i64);
            let rc_y = cy + compdiv_n(n, dxsc as i64 * dyce as i64);
            let split_y = lc_y + compdiv_n(n, dxsc as i64 * (rc_y - lc_y) as i64);
            let split_dx = compdiv_n(n, dxsc as i64 * dxsc as i64);
            split_quad(out, x0, y0, x0 + split_dx, lc_y, x0 + split_dx, split_y);
            x0 += split_dx;
            y0 = split_y;
            cx = x0;
            cy = rc_y;
            continue;
        }
        if (x0 - x2).abs() <= 3200 && (y0 - y2).abs() <= 3200 {
            if dxsc * dyce == dxce * dysc {
                out.push(Piece::Line([x0, y0], [x2, y2]));
            } else {
                out.push(Piece::Arc([x0, y0], [cx, cy], [x2, y2]));
            }
            return;
        }
        // bisect
        let my0 = (cy + y0) >> 1;
        let mx0 = (cx + x0) >> 1;
        let my2 = (cy + y2) >> 1;
        let mx2 = (cx + x2) >> 1;
        let mid_y = (my2 + my0) >> 1;
        let mid_x = (mx2 + mx0) >> 1;
        split_quad(out, x0, y0, mx0, my0, mid_x, mid_y);
        x0 = mid_x;
        y0 = mid_y;
        cx = mx2;
        cy = my2;
    }
}

/// Native's horizontal vertex-topology rule: a contour vertex `s` that lands exactly on a scanline centre is covered
/// by neither adjacent piece's row range (the asymmetric entry/exit rounding leaves a one-row gap), so
/// the fill crossing is added here from the 3-point window (`pp` = previous piece start, `s` = the
/// vertex, `e` = next piece start). A monotone through-vertex adds one crossing (On going down / Off
/// going up); a local extremum adds both (the touched pixel). The on-crossing rounds the column
/// `(x+31)>>6`, the off-crossing `(x+32)>>6`; the row is `s.y>>6`.
/// Vertical analogue of [`check_horiz_topology`] (native's inlined vertical topology,
/// port verified 0/20000 in checktop.py): a piece start `s` on a COLUMN centre (os x ≡ 32 mod
/// 64) adds its per-column fill crossing from the 3-point window `(pp, s, e)`. Marks into the per-column
/// `v` scan: Von at `(row=(s.y+31)>>6, col=s.x>>6)`, Voff at `(row=(s.y+32)>>6, col=s.x>>6)` (native
/// the vertical on/off crossings). NOT a plain x↔y transpose of the horizontal rule — the branch order differs.
fn check_vert_topology(v: &mut Scan, pp: P, s: P, e: P) {
    // Vertical winding is OPPOSITE the horizontal (native's vertical topology ≠ x↔y transpose): Von ⇒ dir
    // −1, Voff ⇒ dir +1. Subpix = the vertex y (an on-column vertex's exact y-intercept is its own y).
    let von = |v: &mut Scan| v.mark((s[1] + 31) >> 6, s[0] >> 6, -1, s[1]);
    let voff = |v: &mut Scan| v.mark((s[1] + 32) >> 6, s[0] >> 6, 1, s[1]);
    let (ppx, ppy, sx, sy, nx, ny) = (pp[0], pp[1], s[0], s[1], e[0], e[1]);
    let vx = sx; // prev-x
    if nx < vx {
        if vx >= ppx {
            if vx > ppx {
                von(v);
                voff(v);
                return;
            }
            if sy >= ppy {
                return;
            }
        }
        von(v);
    } else if nx == vx {
        let vy = sy;
        if vx >= ppx {
            if vx > ppx || vy > ppy {
                if ny < vy {
                    voff(v);
                }
                return;
            }
            if vy >= ppy {
                return;
            }
        }
        if ny > vy {
            von(v);
        }
    } else {
        // nx > vx
        if vx < ppx {
            von(v);
            voff(v);
        } else if vx > ppx || sy > ppy {
            voff(v);
        }
    }
}

fn check_horiz_topology(scan: &mut Scan, pp: P, s: P, e: P) {
    let on = |scan: &mut Scan| scan.mark((s[0] + 31) >> 6, s[1] >> 6, 1, s[0]);
    let off = |scan: &mut Scan| scan.mark((s[0] + 32) >> 6, s[1] >> 6, -1, s[0]);
    let (sy, ppy) = (s[1], pp[1]); // prev y, prev-prev y
    let (ex, ey) = (e[0], e[1]);
    if ey > sy {
        if sy > ppy {
            on(scan);
        } else if sy == ppy {
            if s[0] < pp[0] {
                on(scan);
            }
        } else {
            on(scan);
            off(scan);
        }
    } else if ey == sy {
        if sy <= ppy {
            if sy < ppy || s[0] > pp[0] {
                if ex < s[0] {
                    off(scan);
                }
                return;
            }
            if s[0] >= pp[0] {
                return;
            }
        }
        if ex > s[0] {
            on(scan);
        }
    } else if sy > ppy {
        on(scan);
        off(scan);
    } else if sy < ppy || s[0] > pp[0] {
        off(scan);
    }
}

/// `compdiv` taking the divisor as `i64` (native `CompDiv` with a 64-bit dividend; the divisor here
/// is the extremum denominator, which can exceed `i32`).
fn compdiv_n(d: i64, n: i64) -> i32 {
    if d == 0 {
        return if n < 0 { -0x7FFF_FFFF - 1 } else { 0x7FFF_FFFF };
    }
    let half = d / 2;
    let num = if (d < 0) == (n < 0) { n + half } else { n - half };
    (num / d) as i32
}

// ---- sweep + resolve -----------------------------------------------------------------------------

/// Single-pass BOTH-AXIS crossing collection (native's scan-converter dropout-enabled contour loop):
/// one walk of each piece marks the per-row list `h` (fill + horizontal dropout) AND the per-column list
/// `v` (vertical dropout) via [`calc_line_bi`] / [`calc_spline_bi`], instead of `collect_crossings`'s
/// horizontal-only pass + a separate `transpose`d pass. The transpose does NOT reproduce native's `v`
/// list (it marks vert on the TRANSPOSED `err<=0`, native on the ORIGINAL `err>0`); this does, verified
/// against the fsc oracle. `h` is indexed by row (`rmin`, `nrows`), `v` by column (`cmin`, `ncols`). A vertex
/// on a row centre gets its crossing from `check_horiz_topology`; a horizontal edge there marks none of its own.
fn collect_crossings_bi(contours: &[Vec<Seg>], rmin: i32, nrows: usize, cmin: i32, ncols: usize, boundary: bool) -> (Scan, Scan) {
    let mut h = Scan::new(rmin, nrows);
    let mut v = Scan::new(cmin, ncols);
    let mut pieces: Vec<Piece> = Vec::new();
    for contour in contours {
        pieces.clear();
        for s in contour {
            match *s {
                Seg::Line(p0, p1) => pieces.push(Piece::Line(p0, p1)),
                Seg::Quad(p0, c, p1) => split_quad(&mut pieces, p0[0], p0[1], c[0], c[1], p1[0], p1[1]),
            }
        }
        // A zero-length piece (a repeated point, or a quad that collapses to one) is no piece: the scan's vertex window
        // must see the pieces either side of it, as native's scan input without it does (VM: Segoe UI Emoji's woman at
        // em 40, a doubled on-curve point on a sample row, and the man bold and oblique at em 78, a collapsed curve
        // there: each left that row's crossing out and inverted the row across the glyph).
        pieces.retain(|p| !matches!(*p, Piece::Line(a, b) if a == b) && !matches!(*p, Piece::Arc(a, c, b) if a == c && c == b));
        for p in &pieces {
            match *p {
                Piece::Line(p0, p1) => calc_line_bi(&mut h, &mut v, p0, p1),
                Piece::Arc(p0, c, p1) => calc_spline_bi(&mut h, &mut v, p0, c, p1),
            }
            if let Piece::Line(a, b) = *p {
                // Vertical analogue: a vertical edge on a column centre fills its per-column span.
                if boundary && a[0] == b[0] && a[0] & 0x3F == 0x20 {
                    let (lo, hi) = (a[1].min(b[1]), a[1].max(b[1]));
                    v.mark((lo + 31) >> 6, a[0] >> 6, 1, lo);
                    v.mark((hi + 32) >> 6, a[0] >> 6, -1, hi);
                }
            }
        }
        let n = pieces.len();
        for i in 0..n {
            let s = pieces[i].start();
            let pp = pieces[(i + n - 1) % n].start();
            let e = pieces[i].end();
            if s != e {
                if s[1] & 0x3F == 0x20 {
                    check_horiz_topology(&mut h, pp, s, e);
                }
                if s[0] & 0x3F == 0x20 {
                    check_vert_topology(&mut v, pp, s, e);
                }
            }
        }
    }
    (h, v)
}

#[allow(clippy::too_many_arguments)]
/// `bold_os` > 0 emboldens each glyph's own bitmap, `bold_os` sub-samples a pixel, with the small-size bitmap bold
/// ([`bold_glyph`]) before the merge; `on_center` first adds the glyph's [`x_on_center`] samples, as the ClearType scan does.
#[allow(clippy::too_many_arguments)]
fn scan_glyph_bitmaps(contours: &[Vec<Seg>], cmin: i32, cmax: i32, rmin: i32, rmax: i32,
    mode: u16, glyphs: Option<&[(std::ops::Range<usize>, u16)]>, boundary: bool, bold_os: usize, on_center: bool) -> Vec<bool> {
    let Some(glyphs) = glyphs else {
        let mut bits = scan_bitmap_inner(contours, cmin, cmax, rmin, rmax, mode, boundary);
        if bold_os > 0 && !bits.is_empty() {
            if on_center {
                x_on_center(&mut bits, contours, cmin, cmax, rmin, rmax);
            }
            bold_glyph(&mut bits, (cmax - cmin) as usize, (rmax - rmin) as usize, bold_os);
        }
        return bits;
    };
    let mut bits = vec![false; ((cmax - cmin) * (rmax - rmin)) as usize];
    for (range, mode) in glyphs {
        let outline = &contours[range.clone()];
        let Some([l, t, r, b]) = extrema_box(outline) else { continue };
        let (l, t, r, b) = (l.max(cmin), t.max(rmin), (r + bold_os as i32).min(cmax), b.min(rmax));
        if r <= l || b <= t { continue; }
        // Native rasterizes each glyph within its own extrema bounds before OR-merging it.
        let mut glyph = scan_bitmap_inner(outline, l, r, t, b, *mode, boundary);
        if bold_os > 0 && !glyph.is_empty() {
            if on_center {
                x_on_center(&mut glyph, outline, l, r, t, b);
            }
            bold_glyph(&mut glyph, (r - l) as usize, (b - t) as usize, bold_os);
        }
        for y in t..b {
            for x in l..r {
                bits[((y - rmin) * (cmax - cmin) + x - cmin) as usize] |=
                    glyph[((y - t) * (r - l) + x - l) as usize];
            }
        }
    }
    bits
}

/// Build the oversampled 1-bit bitmap over os columns `[cmin,cmax)` × rows `[rmin,rmax)`: scan every
/// segment, per row nonzero-winding-fill `[on, off−1]`, then run TrueType dropout control.
fn scan_bitmap(contours: &[Vec<Seg>], cmin: i32, cmax: i32, rmin: i32, rmax: i32, mode: u16) -> Vec<bool> {
    scan_bitmap_inner(contours, cmin, cmax, rmin, rmax, mode, true)
}

fn scan_bitmap_inner(contours: &[Vec<Seg>], cmin: i32, cmax: i32, rmin: i32, rmax: i32, mode: u16, boundary: bool) -> Vec<bool> {
    // An empty or inverted device range (a degenerate zero-width/height glyph outline, or a collapsed
    // box) has no cells. Guard it: `(cmax - cmin) as usize` would wrap to a huge allocation when
    // cmax < cmin, and the dropout pass's `col.clamp(cmin, cmax - 1)` would panic (min > max) when
    // cmax == cmin. Return no bits — the caller's `r <= l` check then draws nothing, as native does for
    // a degenerate glyph. (Crash-safety for crafted fonts; a real glyph always has cmax > cmin.)
    if cmax <= cmin || rmax <= rmin {
        return Vec::new();
    }
    let w = (cmax - cmin) as usize;
    // The scan converter masks off the smart-dropout bit for a glyph whose extrema
    // box collapsed in a column axis: the mode's dropout bit is cleared when the glyph-bitmap collapse
    // flag is set. That flag is `left==right || top==bottom` (see [`extrema_collapsed`]). A VM single-step
    // of the scan converter (seguiemj em14): gid 16022 (collapsed 1×2 triangle) → bm10=1, mode 0; gid 32432
    // (naturally 1-col × 14-row stroke, no collapse) → bm10=0, mode 1.
    let mode = if extrema_collapsed(contours) { mode & 0xFFFE } else { mode };
    // Build the per-row (h, fill + horizontal dropout) and per-column (v, vertical dropout) crossing
    // lists in ONE native both-axis pass (the native scan converter works from one consistent set): the
    // horizontal-only scan + a separate transpose vscan did NOT reproduce native's vertical list. `v` is
    // only consumed when dropout is enabled (`mode & 2 == 0`), so skip it otherwise.
    let (h, v) = collect_crossings_bi(contours, rmin, (rmax - rmin) as usize, cmin, w, boundary);
    let (mut scan, bi_v) = (h, (mode & 2 == 0).then_some(v));
    if std::env::var_os("D2D_CROSS_GO").is_some() {
        let mut v: Vec<(i32, i32, i32)> = Vec::new();
        for (ri, xs) in scan.rows.iter().enumerate() {
            for &(col, d, _) in xs { v.push((col, scan.row0 + ri as i32, d)); }
        }
        v.sort();
        for (col, row, d) in v { eprintln!("XC col={col} row={row} dir={d}"); }
    }
    let mut bits = vec![false; w * scan.rows.len()];
    for (ri, xs) in scan.rows.iter_mut().enumerate() {
        xs.sort_by_key(|c| c.0);
        let row = &mut bits[ri * w..][..w];
        let mut wind = 0;
        let mut on = 0;
        for &(col, d, _) in xs.iter() {
            let prev = wind;
            wind += d;
            if prev == 0 && wind != 0 {
                on = col;
            } else if prev != 0 && wind == 0 {
                for c in on.max(cmin)..col.min(cmax) {
                    row[(c - cmin) as usize] = true;
                }
            }
        }
    }
    // Dropout control (the row-sweep fill's dropout search): `mode & 2` disables it; `(mode & 6) == 4`
    // additionally builds the per-column scan so the vertical pass runs.
    if mode & 2 == 0 {
        // Native runs the vertical-dropout pass whenever dropout is enabled (the dropout search is gated only by
        // mode bit 1), not just under SCANTYPE&4 — the `(mode&6)==4` block in the scan converter only records the
        // EXACT subpixel positions, it does not gate the vertical pass itself. Build the per-column scan
        // whenever dropout is on so the vertical dropout runs for the 8×1 grayscale case too.
        // The single-pass both-axis vertical list (built alongside the fill h scan above), sorted for the
        // dropped-span pairing. Always present here (`mode & 2 == 0`).
        let mut vscan = bi_v.expect("both-axis v scan is built whenever dropout is enabled");
        for xs in vscan.rows.iter_mut() {
            xs.sort_by_key(|c| c.0);
        }
        if std::env::var_os("D2D_VSCANDUMP").is_some() {
            let mut vc: Vec<(i32, i32, i32)> = Vec::new();
            for (ki, xs) in vscan.rows.iter().enumerate() {
                for &(row, d, _) in xs { vc.push((vscan.row0 + ki as i32, row, d)); }
            }
            vc.sort();
            for (col, row, d) in &vc { eprintln!("VC col={col} row={row} dir={d}"); }
        }
        let pre_drop: Vec<bool> = if std::env::var_os("D2D_DROPDUMP").is_some() { bits.clone() } else { Vec::new() };
        Dropout { bits: &mut bits, w, cmin, cmax, rmin, rmax, h: &scan, v: Some(&vscan), smart: mode & 1 != 0, exact: mode & 4 != 0 }.run();
        if std::env::var_os("D2D_DROPDUMP").is_some() {
            let rows = bits.len() / w.max(1);
            eprintln!("DROPDUMP cmin={cmin} cmax={cmax} rmin={rmin} rmax={rmax} w={w} rows={rows}");
            for i in 0..bits.len() {
                if pre_drop.get(i) != Some(&bits[i]) {
                    eprintln!("DROPSET row={} col={}", rmin + (i / w) as i32, cmin + (i % w) as i32);
                }
            }
        }
    }
    if std::env::var_os("D2D_SCANDUMP").is_some() {
        let rows = bits.len() / w.max(1);
        eprintln!("SCANDUMP cmin={cmin} cmax={cmax} rmin={rmin} rmax={rmax} w={w} rows={rows} mode={mode}");
        let mut xc: Vec<(i32, i32, i32)> = Vec::new();
        for (ri, xs) in scan.rows.iter().enumerate() {
            for &(col, d, _) in xs { xc.push((col, scan.row0 + ri as i32, d)); }
        }
        xc.sort();
        for (col, row, d) in &xc { eprintln!("XC col={col} row={row} dir={d}"); }
        for rr in 0..rows {
            let s: String = (0..w).map(|c| if bits[rr * w + c] { '#' } else { '.' }).collect();
            eprintln!("SB{rr:>2} {s}");
        }
    }
    bits
}

/// TrueType dropout control: recover a thin stem
/// (horizontal pass) or a curve bowl tangent to a scanline (vertical pass) that the fill dropped — a
/// zero-width rank-paired on/off span — by setting one sample. `h`/`v` are the per-row/per-column
/// crossing lists (sorted). `smart` (`mode & 1`) runs the neighbour-crossing test that skips stubs;
/// `exact` (`mode & 4`, e.g. Consolas/Tahoma `SCANTYPE 5`) places the recovered sample at the mean of
/// the two edges' sub-pixel intercepts (native's horizontal/vertical dropout `(on+off−1)>>7`),
/// not the simple one-step-back position.
struct Dropout<'a> {
    bits: &'a mut [bool],
    w: usize,
    cmin: i32,
    cmax: i32,
    rmin: i32,
    rmax: i32,
    h: &'a Scan,
    v: Option<&'a Scan>,
    smart: bool,
    exact: bool,
}

/// Rank-paired zero-width spans in a sorted crossing line: pair the k-th "on" (dir > 0) with the k-th
/// "off" (dir < 0); equal coordinate ⟹ a dropped span there. Returns `(coord, on_subpix, off_subpix)`
/// so the caller can place the recovered sample at the exact sub-pixel mean under `SCANTYPE & 4`.
fn dropped_spans(xs: &[(i32, i32, i32)]) -> Vec<(i32, i32, i32)> {
    let on: Vec<(i32, i32)> = xs.iter().filter(|c| c.1 > 0).map(|c| (c.0, c.2)).collect();
    let off: Vec<(i32, i32)> = xs.iter().filter(|c| c.1 < 0).map(|c| (c.0, c.2)).collect();
    (0..on.len().min(off.len())).filter(|&k| on[k].0 == off[k].0).map(|k| (on[k].0, on[k].1, off[k].1)).collect()
}

/// The recovered sample's coordinate for a dropped span at `coord` with on/off sub-pixel intercepts
/// `on`/`off`: the native exact position `(on + off − 1) >> 7` under `SCANTYPE & 4` when both
/// intercepts are known, else the simple `coord − 1`.
fn recover_at(coord: i32, on: i32, off: i32, exact: bool) -> i32 {
    if exact && on != NO_SUBPIX && off != NO_SUBPIX {
        (on + off - 1) >> 7
    } else {
        coord - 1
    }
}

impl Dropout<'_> {
    fn get(&self, col: i32, row: i32) -> bool {
        col >= self.cmin
            && col < self.cmax
            && row >= self.rmin
            && row < self.rmax
            && self.bits[(row - self.rmin) as usize * self.w + (col - self.cmin) as usize]
    }
    fn set(&mut self, col: i32, row: i32) {
        if col >= self.cmin && col < self.cmax && row >= self.rmin && row < self.rmax {
            self.bits[(row - self.rmin) as usize * self.w + (col - self.cmin) as usize] = true;
        }
    }
    /// Crossings in row `row` whose column == `col` (native's horizontal-crossing count).
    fn hc(&self, col: i32, row: i32) -> i32 {
        line_count(self.h, row, col)
    }
    /// Crossings in column `col` whose row == `row` (native's vertical-crossing count; 0 without the vertical scan).
    fn vc(&self, col: i32, row: i32) -> i32 {
        self.v.map_or(0, |v| line_count(v, col, row))
    }

    fn run(&mut self) {
        // Horizontal pass: a dropped span at column `c` in row `r` recovers one sample (native
        // the horizontal dropout pass). Collect candidates first (the lists borrow `self.h`), then fill.
        let mut hits: Vec<(i32, i32, i32, i32)> = Vec::new();
        for row in self.rmin..self.rmax {
            for (c, on, off) in dropped_spans(&self.h.rows[(row - self.rmin) as usize]) {
                hits.push((c, row, on, off));
            }
        }
        for (c, r, on, off) in hits {
            if self.smart
                && (self.hc(c, r + 1) + self.vc(c - 1, r + 1) + self.vc(c, r + 1) < 2
                    || self.hc(c, r - 1) + self.vc(c - 1, r) + self.vc(c, r) < 2)
            {
                continue;
            }
            if (c > self.cmin && self.get(c - 1, r)) || (c < self.cmax && self.get(c, r)) {
                continue;
            }
            let col = recover_at(c, on, off, self.exact);
            self.set(col.clamp(self.cmin, self.cmax - 1), r);
        }
        // Vertical pass (native's vertical dropout): a dropped span at row `rr` in column `k`.
        let Some(v) = self.v else { return };
        let mut vhits: Vec<(i32, i32, i32, i32)> = Vec::new();
        for k in self.cmin..self.cmax {
            // Native's dropout search walks each column's dropped spans BACKWARD (decrementing by the row stride), so a
            // higher-row span fills first and the already-set-bit guard then skips a lower-row span whose
            // recovered sample lands on the just-filled cell. Processing forward instead double-fills a
            // column with two stacked zero-height spans (a thin glyph aperture like Consolas/Times 'e' at
            // ~7 px closes its eye; the horizontal pass keeps native's forward within-row order).
            for (rr, on, off) in dropped_spans(&v.rows[(k - self.cmin) as usize]).into_iter().rev() {
                vhits.push((k, rr, on, off));
            }
        }
        for (k, rr, on, off) in vhits {
            if self.smart
                && (self.hc(k, rr - 1) + self.hc(k, rr) + self.vc(k - 1, rr) < 2
                    || self.hc(k + 1, rr - 1) + self.hc(k + 1, rr) + self.vc(k + 1, rr) < 2)
            {
                continue;
            }
            if (rr > self.rmin && self.get(k, rr - 1)) || (rr < self.rmax && self.get(k, rr)) {
                continue;
            }
            let row = recover_at(rr, on, off, self.exact);
            self.set(k, row.clamp(self.rmin, self.rmax - 1));
        }
    }
}

/// Count crossings in `scan`'s line `line` whose value == `val` (shared by the horizontal-crossing count on the
/// per-row scan and the vertical-crossing count on the per-column scan).
fn line_count(scan: &Scan, line: i32, val: i32) -> i32 {
    let i = line - scan.row0;
    if i < 0 || i as usize >= scan.rows.len() {
        return 0;
    }
    scan.rows[i as usize].iter().filter(|c| c.0 == val).count() as i32
}

/// Grayscale: 4× bitmap → `round(255·count/16)` per 4×4 block.
fn resolve_grayscale(contours: Vec<Vec<Seg>>, mode: u16) -> AlphaTexture {
    let Some(os) = extrema_box(&contours) else { return AlphaTexture::empty() };
    let [l, t, r, b] = device_bounds(os, 4, 4, false);
    if r <= l || b <= t {
        return AlphaTexture::empty();
    }
    let mut bits = scan_bitmap(&contours, 4 * l, 4 * r, 4 * t, 4 * b, mode);
    x_on_center(&mut bits, &contours, 4 * l, 4 * r, 4 * t, 4 * b);
    // Recover a straight sloped edge sitting on a sample centre (both no-ops here in practice: this path
    // serves the unhinted `raster::grayscale` entry, whose points land on `≡0 mod 4`, never a centre).
    // The hinted 4×4 grayscale goes through the y-negated `coverage_outline`→`coverage_box` path instead.
    sloped_on_center(&mut bits, &contours, 4 * l, 4 * r, 4 * t, 4 * b);
    let bw = (4 * (r - l)) as usize;
    let (w, h) = ((r - l) as usize, (b - t) as usize);
    let mut data = vec![0u8; w * h];
    for py in 0..h {
        for px in 0..w {
            let mut count = 0u32;
            for j in 0..4 {
                let rowbase = (4 * py + j) * bw + 4 * px;
                for i in 0..4 {
                    count += bits[rowbase + i] as u32;
                }
            }
            data[py * w + px] = ((255 * count + 8) >> 4) as u8;
        }
    }
    drop_empty_tangent(l, t, r, b, data, os, 4, 4, false, 1)
}

/// X-on-centre topology (native's vertical vertex-topology rule, the x↔y transpose of
/// [`check_horiz_topology`]): a straight vertical edge lying exactly on a sample-centre column
/// (`x ≡ 32 mod 64`) covers that column over the rows it spans — its centre is on the boundary ⇒
/// inside, which the per-row horizontal sweep drops with its asymmetric exit rounding (`calc_line`
/// marks `(x+32)>>6`, one column past the centre). Fill the centre column `x>>6` over the edge's
/// rows `[(ymin+32)>>6, (ymax-33)>>6]` (the same row rounding `calc_line` uses for a vertical edge).
/// `[cmin,cmax)×[rmin,rmax)` are the bitmap's sample columns/rows (device at ×1, oversampled at ×4/×6).
///
/// Native runs this in every mode, not only bi-level: a hinted outline's grid-fitted points land on
/// the oversampled half-column (`x ≡ 32 mod 64`), unlike an unhinted point (`≡ 0 mod 4`), so the
/// oversampled grayscale/ClearType fill needs it too (the Arial hinted residual). A hinted
/// stem edge is often a `Quad` with its control on the line (p0.x==c.x==p1.x), so match that as well
/// as a straight `Line` or the stem is missed. A genuinely curved vertical tangent (control off the
/// line) is the vertical-dropout case, not a column fill.
fn x_on_center(bits: &mut [bool], contours: &[Vec<Seg>], cmin: i32, cmax: i32, rmin: i32, rmax: i32) {
    // A flipping transform reverses the outline's winding, so a vertical edge on a sample centre is an edge whose
    // column rule leaves that centre out ((x+32)>>6 on, (x+31)>>6 off), and native leaves it out too: only the
    // upright winding's edges keep it (VM: Malgun Gothic 'j' stem at x = 91.5 subpixels under a y-flip, native
    // crossings move one column and the filled stem is one sample narrower; ClearType GDI natural and classic,
    // regular and bold, under x and y flips).
    if FLIPPED.with(|c| c.get()) {
        return;
    }
    let bw = (cmax - cmin) as usize;
    for s in contours.iter().flatten() {
        // A vertical edge: a straight `Line`, or a `Quad` whose control lies on it (p0.x==c.x==p1.x) —
        // a hinted stem's edge is often encoded that way, so match both or the stem is missed.
        let vert = match *s {
            Seg::Line(p0, p1) if p0[0] == p1[0] => Some((p0[0], p0[1], p1[1])),
            Seg::Quad(p0, c, p1) if p0[0] == c[0] && c[0] == p1[0] => Some((p0[0], p0[1], p1[1])),
            _ => None,
        };
        if let Some((x, y0, y1)) = vert {
            if x & 0x3F == 0x20 {
                let col = x >> 6;
                if col < cmin || col >= cmax {
                    continue;
                }
                let (ymin, ymax) = (y0.min(y1), y0.max(y1));
                for row in ((ymin + 32) >> 6).max(rmin)..=((ymax - 33) >> 6).min(rmax - 1) {
                    bits[(row - rmin) as usize * bw + (col - cmin) as usize] = true;
                }
            }
        }
    }
}

/// The sloped analogue of [`x_on_center`]: a non-vertical *straight* edge whose crossing at a scanline's
/// sample centre (`y ≡ 32 mod 64`) lands exactly on a sample-centre column (`x ≡ 32 mod 64`) puts that
/// sample on the fill boundary. Native rounds the on/off column `(x+31)>>6`
/// / `(x+32)>>6` — identical except at an exact centre, where both leave column `x>>6` filled; our
/// single-column `calc_line` mark drops the *off* side there. Set it, matching native. Only an *exact*
/// integer grid hit counts, so this is a rare recovery; a vertical edge is handled by [`x_on_center`].
///
/// Curves (`Quad`) are NOT recovered here — their on-centre crossings are generated by the
/// `calc_spline` DDA and resolved by [`check_horiz_topology`] at piece vertices, byte-matching native's
/// the spline DDA / horizontal topology rule (provided the scan runs in native's y-up frame; see
/// [`coverage_box`]). The former analytic quadratic-root patch here over-filled curve crossings native
/// dropouts (Comic Sans 'B'/'e'@27) and has been retired.
fn sloped_on_center(bits: &mut [bool], contours: &[Vec<Seg>], cmin: i32, cmax: i32, rmin: i32, rmax: i32) {
    let bw = (cmax - cmin) as usize;
    let mut set = |x: i128, row: i32| {
        if x & 0x3F == 0x20 {
            let col = (x >> 6) as i32;
            if col >= cmin && col < cmax {
                bits[(row - rmin) as usize * bw + (col - cmin) as usize] = true;
            }
        }
    };
    for s in contours.iter().flatten() {
        match *s {
            Seg::Line(p0, p1) => {
                let (dy, dx) = ((p1[1] - p0[1]) as i128, (p1[0] - p0[0]) as i128);
                if dy == 0 || dx == 0 {
                    continue; // horizontal (no crossing) or vertical (handled by x_on_center)
                }
                for row in rmin..rmax {
                    let sy = (row * 64 + 32) as i128; // scanline sample centre (oversampled 26.6)
                    // Strictly inside the edge's y-span, or it is a vertex rather than a crossing.
                    if (sy - p0[1] as i128).signum() * (sy - p1[1] as i128).signum() >= 0 {
                        continue;
                    }
                    let num = dx * (sy - p0[1] as i128);
                    if num % dy != 0 {
                        continue; // not an exact integer crossing ⇒ cannot sit on a column centre
                    }
                    set(p0[0] as i128 + num / dy, row);
                }
            }
            Seg::Quad(..) => {} // curve on-centre crossings come from calc_spline + check_horiz_topology
        }
    }
}

/// Aliased: device-resolution 1-bit bitmap → 0/255 per pixel (one sample at the pixel centre, the
/// same scan converter + dropout control as grayscale, only without oversampling).
fn resolve_aliased(contours: Vec<Vec<Seg>>, mode: u16) -> AlphaTexture {
    resolve_aliased_glyphs(contours, mode, None)
}

fn resolve_aliased_glyphs(contours: Vec<Vec<Seg>>, mode: u16, glyphs: Option<&[(std::ops::Range<usize>, u16)]>) -> AlphaTexture {
    let Some(os) = extrema_box(&contours) else { return AlphaTexture::empty() };
    let [l, t, r, b] = device_bounds(os, 1, 1, false);
    if r <= l || b <= t {
        return AlphaTexture::empty();
    }
    let mut bits = scan_glyph_bitmaps(&contours, l, r, t, b, mode, glyphs, true, 0, false);
    x_on_center(&mut bits, &contours, l, r, t, b);
    let data = bits.iter().map(|&bit| if bit { 255 } else { 0 }).collect();
    drop_empty_tangent(l, t, r, b, data, os, 1, 1, false, 1)
}

/// ClearType: 6×5 bitmap → Platt horizontal (three width-6 boxes offset 2) + 5-tap vertical filter,
/// `round(255·s/216)`, 3 bytes/pixel (left/centre/right colour boxes = the bytes native stores).
thread_local! {
    /// The ClearType level the ClearType coverage resolved now is drawn at ([`CtLevel`]); 1 outside a draw.
    static CT_LEVEL: std::cell::Cell<f32> = const { std::cell::Cell::new(1.0) };
}

/// While alive, ClearType coverage resolves at ClearType level `level` (the rendering params'): Direct2D mixes the red
/// and blue sub-pixel windows toward the green one, `count·L + green·(1 − L)`, before the coverage is rounded. The
/// analysis API (no draw) keeps level 1.
pub(crate) struct CtLevel(f32);

thread_local! {
    /// The run draws under a transform with a negative determinant (a flip), whose outline winding is reversed.
    pub(crate) static FLIPPED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Marks the scans while it lives as drawn under a flipping transform ([`FLIPPED`]), restoring the old state on drop.
pub(crate) struct Flipped(bool);

impl Flipped {
    pub(crate) fn set(flipped: bool) -> Self {
        Flipped(FLIPPED.with(|c| c.replace(flipped)))
    }
}

impl Drop for Flipped {
    fn drop(&mut self) {
        FLIPPED.with(|c| c.set(self.0));
    }
}
impl CtLevel {
    pub(crate) fn set(level: f32) -> Self {
        CtLevel(CT_LEVEL.with(|c| c.replace(level.abs())))
    }
}

impl Drop for CtLevel {
    fn drop(&mut self) {
        CT_LEVEL.with(|c| c.set(self.0));
    }
}

fn resolve_cleartype(contours: Vec<Vec<Seg>>, mode: u16, vweights: &[u32], bold: (i32, i32)) -> AlphaTexture {
    resolve_cleartype_glyphs(contours, mode, vweights, None, bold, false)
}

fn resolve_cleartype_glyphs(contours: Vec<Vec<Seg>>, mode: u16, vweights: &[u32], glyphs: Option<&[(std::ops::Range<usize>, u16)]>, bold: (i32, i32), dilate: bool) -> AlphaTexture {
    // `vweights` is the vertical filter: `{4,9,10,9,4}` (5× oversample, NATURAL_SYMMETRIC) or `[1]`
    // (no vertical AA, the GDI/NATURAL horizontal-only modes — pure 6× box per device row, so every
    // sub-pixel lands on a clean `k·255/6`, verified byte-exact vs native GDI-classic 'o'@13). The box
    // sum normalises by `6·Σw` (horizontal box 6 × vertical weight sum): 216 for 6×5, 6 for 6×1.
    let sy = vweights.len() as i32;
    let norm: u32 = 6 * vweights.iter().sum::<u32>();
    let Some(os) = extrema_box(&contours) else { return AlphaTexture::empty() };
    let [mut l, mut t, mut r, mut b] = device_bounds(os, 6, sy, true);
    if r <= l || b <= t {
        return AlphaTexture::empty();
    }
    let (cmin, rmin) = (6 * l - 2, sy * t); // window at output column `l` starts at 6·l−2
    let runs = sy == 1 && run_darkening(bold);
    if runs {
        r += 1;
    }
    let mut bits = scan_glyph_bitmaps(&contours, cmin, 6 * r + 2, rmin, sy * b, mode, glyphs, true, if runs { 6 } else { 0 }, true);
    x_on_center(&mut bits, &contours, cmin, 6 * r + 2, rmin, sy * b);
    let mut bw = (6 * (r - l) + 4) as usize; // sub-columns spanning every pixel's 10-wide window
    // Synthetic-bold embolden (the sub-pixel embolden): OR-dilate the 6× sub-pixel grid right by
    // `boldX·6` sub-columns and up by `boldY·sy`, as the grayscale path dilates its 8× grid — the stroke
    // grows right by boldX device px / up by boldY. Extend the device bounds to match (`right += boldX`,
    // `top -= boldY`); the window padding (±2) is carried by the dilated grid.
    // A negative amount ([`bold_dir`]) grows left / down: the mirrored grid grows right / up.
    let (bx, by) = bold;
    if (bx != 0 || by != 0) && !runs {
        let rows = (sy * (b - t)) as usize;
        let (fx, fy) = (bx < 0, by < 0);
        let (gx, gy) = (bx.unsigned_abs() as usize * 6, by.unsigned_abs() as usize * sy as usize);
        let (nb, nbw, nrows) = embolden_bits(&mirror(&bits, bw, rows, fx, fy), bw, rows, gx, gy, gx, gy);
        bits = mirror(&nb, nbw, nrows, fx, fy);
        bw = nbw;
        if fx { l += bx } else { r += bx }
        if fy { b -= by } else { t -= by }
    }
    // The thin-font merge widens the 6× grid by one sub-column to the right, as the grayscale merge does its
    // 8× / 4× grid (VM: Courier New 'W' GDI and NATURAL_SYMMETRIC ClearType, every stroke edge one sub-column wider).
    if dilate {
        for row in bits.chunks_mut(bw) {
            for ox in (1..row.len()).rev() {
                row[ox] |= row[ox - 1];
            }
        }
    }
    if std::env::var_os("D2D_CTBITS").is_some() {
        // Raw 6×sy scan grid before the Platt/vertical resolve (y-up frame — these segs are negated;
        // row 0 is the glyph's bottom). cmin/rmin are the os-origin of column 0 / row 0.
        let rows = (sy * (b - t)) as usize;
        eprintln!("CTBITS dev=[{l},{t},{r},{b}] cmin={cmin} rmin={rmin} bw={bw} rows={rows}");
        for rr in 0..rows {
            let s: String = (0..bw).map(|c| if bits[rr * bw + c] { '#' } else { '.' }).collect();
            eprintln!("CT{rr:>2} {s}");
        }
    }
    let (w, h) = ((r - l) as usize, (b - t) as usize);
    let mut data = vec![0u8; w * h * 3];
    let level = CT_LEVEL.with(|c| c.get());
    for py in 0..h {
        for px in 0..w {
            if level != 1.0 {
                // Below level 1 the red and blue windows mix toward the green one. 6×5 (vertical AA): each row's
                // mixed count is unpacked to a byte, `round(mix·42.5)`, the vertical filter sums the bytes back to
                // the 0..216 count scale, and that count indexes the coverage `round(count·255/216)`. 6×1: the
                // mixed count of the one row straight to `round(mix·255/6)`.
                let rb = |v: f32| (v + 0.5).floor();
                let mut acc = [0f32; 3];
                for (j, &wt) in vweights.iter().enumerate() {
                    let rowbase = (sy as usize * py + j) * bw + 6 * px;
                    let on = |k: usize| bits[rowbase + k] as u32;
                    let g = (2..8).map(&on).sum::<u32>() as f32;
                    let mix = |c: u32| c as f32 * level + g * (1.0 - level);
                    let m = [mix((0..6).map(&on).sum::<u32>()), g, mix((4..10).map(&on).sum::<u32>())];
                    for k in 0..3 {
                        acc[k] += if sy == 1 { m[k] } else { wt as f32 * rb(m[k] * 42.5) };
                    }
                }
                for k in 0..3 {
                    let v = if sy == 1 { acc[k] * 255.0 / 6.0 } else { (acc[k] / 42.5).round() * (255.0 / 216.0) };
                    data[(py * w + px) * 3 + k] = (rb(v) as u32).min(255) as u8;
                }
                continue;
            }
            let mut acc = [0u32; 3];
            for (j, &wt) in vweights.iter().enumerate() {
                let rowbase = (sy as usize * py + j) * bw + 6 * px;
                let on = |k: usize| bits[rowbase + k] as u32;
                let b0 = (0..6).map(&on).sum::<u32>();
                let b1 = (2..8).map(&on).sum::<u32>();
                let b2 = (4..10).map(&on).sum::<u32>();
                acc[0] += wt * b0;
                acc[1] += wt * b1;
                acc[2] += wt * b2;
            }
            for k in 0..3 {
                data[(py * w + px) * 3 + k] = ((255 * acc[k] + norm / 2) / norm) as u8;
            }
        }
    }
    drop_empty_tangent(l, t, r, b, data, os, 6, sy, true, 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(row0: i32, rows: Vec<Vec<(i32, i32, i32)>>) -> Scan {
        Scan { rows, row0 }
    }

    /// The small-size bitmap bold grows a stem by its edges' averaged gap amount: a stem one pixel from its
    /// neighbour in half its rows grows 6 of the 8 sub-samples in every row, the free neighbour all 8.
    #[test]
    fn bold_runs_average_the_gap_down_the_stem() {
        let (w, rows) = (40, 4);
        let mut bits = vec![false; w * rows];
        for r in 0..rows {
            bits[r * w + 4..r * w + 8].fill(true);
        }
        for r in 0..2 {
            bits[r * w + 16..r * w + 20].fill(true);
        }
        bold_runs(&mut bits, w, rows, 8);
        let row = |r: usize| (0..w).filter(|&x| bits[r * w + x]).collect::<Vec<_>>();
        assert_eq!(row(3), (4..14).collect::<Vec<_>>());
        assert_eq!(row(0), (4..14).chain(16..28).collect::<Vec<_>>());
    }

    #[test]
    fn curved_dropout_intercepts_survive_one_column_spans() {
        // The DDA has one quantized column, but the horizontal subpixel pass still bisects the curve.
        let (a, c, b) = ([0, 0], [16, 64], [24, 128]);
        assert_eq!(spline_subpix(a, c, b, 0), 7);
        assert_eq!(spline_subpix(b, c, a, 0), 7);
        let mut crossings = Scan::new(0, 2);
        calc_spline(&mut crossings, a, c, b);
        assert_eq!(crossings.rows[0], vec![(0, 1, 7)]);
        assert_eq!(crossings.rows[1], vec![(0, 1, 19)]);
    }

    /// A rank-paired zero-width span (`on == off`) is a dropout: the horizontal pass recovers the
    /// sample one column left (native's horizontal dropout, simple position), the vertical pass one row up.
    #[test]
    fn dropout_recovers_dropped_span() {
        // Row 1 of a 6×3 bitmap has a dropped stem at column 3 (on == off == 3), nothing filled.
        let h = scan(0, vec![vec![], vec![(3, 1, NO_SUBPIX), (3, -1, NO_SUBPIX)], vec![]]);
        let mut bits = vec![false; 6 * 3];
        Dropout { bits: &mut bits, w: 6, cmin: 0, cmax: 6, rmin: 0, rmax: 3, h: &h, v: None, smart: false, exact: false }.run();
        assert!(bits[1 * 6 + 2], "horizontal dropout fills column c-1 = 2 in the dropped row");
        assert_eq!(bits.iter().filter(|&&b| b).count(), 1, "exactly one recovered sample");

        // Neighbour-set guard: with the recovered column already filled, the pass must not fill again.
        let h = scan(0, vec![vec![(3, 1, NO_SUBPIX), (3, -1, NO_SUBPIX)]]);
        let mut bits = vec![false; 6];
        bits[2] = true;
        Dropout { bits: &mut bits, w: 6, cmin: 0, cmax: 6, rmin: 0, rmax: 1, h: &h, v: None, smart: false, exact: false }.run();
        assert_eq!(bits.iter().filter(|&&b| b).count(), 1, "neighbour already set ⇒ no second fill");

        // Vertical pass: transpose of the above — a dropped span down a column recovers one row up.
        let v = scan(0, vec![vec![], vec![(1, 1, NO_SUBPIX), (1, -1, NO_SUBPIX)], vec![]]);
        let h = scan(0, vec![vec![], vec![]]); // 2 empty rows (smart off, so unused by the vert pass)
        let mut bits = vec![false; 3 * 2];
        Dropout { bits: &mut bits, w: 3, cmin: 0, cmax: 3, rmin: 0, rmax: 2, h: &h, v: Some(&v), smart: false, exact: false }.run();
        assert!(bits[1], "vertical dropout fills row rr-1 = 0 in column 1");

        // Exact position (`SCANTYPE & 4`): recovered column is `(on_subpix + off_subpix − 1) >> 7`,
        // not `c − 1`. A thin stem whose both edges intercept subcol·64 = 2·64 and 3·64 recovers at
        // `(128 + 192 − 1) >> 7 = 2`.
        let h = scan(0, vec![vec![(3, 1, 2 * 64), (3, -1, 3 * 64)]]);
        let mut bits = vec![false; 6];
        Dropout { bits: &mut bits, w: 6, cmin: 0, cmax: 6, rmin: 0, rmax: 1, h: &h, v: None, smart: false, exact: true }.run();
        assert!(bits[2], "exact dropout fills the sub-pixel-mean column 2");
        assert_eq!(bits.iter().filter(|&&b| b).count(), 1, "exactly one recovered sample");
    }

    /// A sloped edge whose crossing at a scanline sample centre lands exactly on a sample-centre column
    /// sets that sample ([`sloped_on_center`]); a crossing off the centre leaves the bitmap untouched.
    #[test]
    fn sloped_on_center_fills_exact_centre() {
        // 45° line from (0,0) to (64,64): at row 0's sample y=32 it crosses x=32 ≡ 32 mod 64 (col 0's
        // centre), so col 0 of row 0 is recovered; row 1's sample y=96 is past the edge, so untouched.
        let contours = vec![vec![Seg::Line([0, 0], [64, 64])]];
        let mut bits = vec![false; 4 * 2];
        sloped_on_center(&mut bits, &contours, 0, 4, 0, 2);
        assert!(bits[0], "exact centre crossing at row 0 col 0 is set");
        assert_eq!(bits.iter().filter(|&&b| b).count(), 1, "only the exact-centre sample is set");

        // A line that crosses off-centre (x=16 at row 0) sets nothing.
        let off = vec![vec![Seg::Line([0, 0], [32, 64])]];
        let mut bits = vec![false; 4 * 2];
        sloped_on_center(&mut bits, &off, 0, 4, 0, 2);
        assert_eq!(bits.iter().filter(|&&b| b).count(), 0, "off-centre crossing recovers nothing");

        // Curves are NOT recovered here: their on-centre crossings come from the calc_spline DDA +
        // check_horiz_topology (see the fn doc). A quad through (32,32) leaves the bitmap untouched.
        let quad = vec![vec![Seg::Quad([0, 0], [32, 32], [64, 64])]];
        let mut bits = vec![false; 4 * 2];
        sloped_on_center(&mut bits, &quad, 0, 4, 0, 2);
        assert_eq!(bits.iter().filter(|&&b| b).count(), 0, "sloped_on_center ignores curves");
    }

    /// A degenerate/collapsed glyph box (empty or inverted device range) must scan nothing without a
    /// panic or a wrapped allocation. Before the guard, `cmax < cmin` made `(cmax - cmin) as usize`
    /// wrap to a ~18-exabyte `vec!` (process abort), and `cmax == cmin` with a dropped span hit the
    /// dropout clamp `col.clamp(cmin, cmax - 1)` (min > max). A crafted font can present such a glyph.
    #[test]
    fn empty_device_range_scans_nothing() {
        let stem = vec![vec![Seg::Line([64, 0], [64, 256]), Seg::Line([128, 256], [128, 0])]];
        let m = 1u16; // smart dropout on (mode & 2 == 0 ⇒ the dropout pass, and thus the clamp, runs)
        assert!(scan_bitmap_inner(&stem, 0, 0, 0, 4, m, true).is_empty(), "cmax == cmin");
        assert!(scan_bitmap_inner(&stem, 5, 0, 0, 4, m, true).is_empty(), "cmax < cmin (no exabyte vec)");
        assert!(scan_bitmap_inner(&stem, 0, 4, 2, 2, m, true).is_empty(), "rmax == rmin");
    }
}

