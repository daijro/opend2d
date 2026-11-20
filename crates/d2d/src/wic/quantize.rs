//! windowscodecs' median-cut colour quantiser — the engine behind
//! `IWICPalette::InitializeFromBitmap`. Confirmed byte-for-byte against native WIC (`wic_palette/`),
//! including the per-call box array of the best-box split.
//!
//! The quantiser converts the source to 24bpp BGR
//! through the scan pipeline, feeds every row into the histogram, then runs the cut.
//! Model, exactly as native:
//! * a joint 32×32×32 RGB histogram, 5 bits/channel (low 3 dropped), **every bin initialised to 1**
//!   each pixel increments its bin by 1 capped at `0x1FFFFF`. So a box's "population" is its
//!   bin volume plus its pixels — empty bins count 1, which makes every box non-empty (bounds never
//!   need shrinking) and pulls box statistics toward the geometric centre.
//! * the distinct 8-bit colours are tracked up to `max_colors + 1`; if the image has `≤ max_colors`
//!   distinct colours the palette is exactly those colours **in first-seen order**.
//! * otherwise median cut over the full `[0,31]³` cube: repeatedly split the box maximising
//!   `population × longest-axis-extent` (axis ties R>G>B, first box wins) at that axis's
//!   population-weighted mean `sum/population`; the lower half becomes a new box, the upper half stays.
//!   Each box's colour is its population-weighted average per channel, scaled 5-bit→8-bit as
//!   `(15 + 255·avg)/31`; boxes are emitted by descending population (stable).
//!
//! Because every bin starts at 1, the average is dominated by the uniform background and lands at (or
//! near) the box centre — e.g. a pure-grey box emits G=B=0x7b even though its pixels are grey.
//!
//! The format converter's palette-less indexed path (`GetOptimalPalette`) reuses this quantiser with
//! error-diffusion dithering; Direct2D never converts to an indexed destination, so that dithered
//! path is out of scope here (see `convert.rs`).

/// Histogram resolution: 5 bits per channel → 32 levels, 32768 bins.
const BITS: u32 = 5;
const LEVELS: u32 = 1 << BITS; // 32
const DROP: u32 = 8 - BITS; // 3
/// Per-bin population cap (a bin saturates at `0x1FFFFF`).
const CAP: u32 = 0x1F_FFFF;

/// A colour box: inclusive 5-bit bounds per channel (0=R, 1=G, 2=B), the population-weighted channel
/// sums and the total population (bin volume + pixels), i.e. native's `Box` dword2..5.
#[derive(Clone)]
struct Box {
    lo: [u8; 3],
    hi: [u8; 3],
    sum: [u32; 3],
    count: u32,
}

/// Histogram bin index `R5 | G5<<5 | B5<<10`.
fn hist_index(r5: u32, g5: u32, b5: u32) -> usize {
    (r5 | g5 << BITS | b5 << (2 * BITS)) as usize
}

/// Native `ShrinkBox`: the box's population and weighted per-channel sums over its bounds (32-bit
/// wrapping, as native). Bounds stay as the splits set them — every bin is ≥1, so there is nothing to
/// tighten.
// ponytail: O(boxes × cube) like native's per-box ShrinkBox; fine for the small images WIC quantises.
fn stats(bx: &mut Box, hist: &[u32]) {
    let mut count = 0u32;
    let mut sum = [0u32; 3];
    for b5 in bx.lo[2]..=bx.hi[2] {
        for g5 in bx.lo[1]..=bx.hi[1] {
            for r5 in bx.lo[0]..=bx.hi[0] {
                let c = hist[hist_index(r5 as u32, g5 as u32, b5 as u32)];
                count = count.wrapping_add(c);
                let idx = [r5, g5, b5];
                for k in 0..3 {
                    sum[k] = sum[k].wrapping_add(c.wrapping_mul(idx[k] as u32));
                }
            }
        }
    }
    bx.sum = sum;
    bx.count = count;
}

/// Split the box maximising `population × longest-axis extent` at that axis's population-weighted mean;
/// the new (lower) box is appended. `false` if no box can be split.
fn split_best(boxes: &mut Vec<Box>, hist: &[u32]) -> bool {
    let mut best: Option<(usize, usize, u32)> = None; // (box, axis, score)
    for (i, bx) in boxes.iter().enumerate() {
        if bx.count <= 1 {
            continue;
        }
        let ext = [bx.hi[0] - bx.lo[0], bx.hi[1] - bx.lo[1], bx.hi[2] - bx.lo[2]];
        // Longest axis, ties R>G>B (native's branch order).
        let axis = if ext[0] >= ext[1] && ext[0] >= ext[2] {
            0
        } else if ext[1] >= ext[2] {
            1
        } else {
            2
        };
        if ext[axis] == 0 {
            continue;
        }
        let score = bx.count.wrapping_mul(ext[axis] as u32);
        if score > best.map_or(0, |b| b.2) {
            best = Some((i, axis, score));
        }
    }
    let Some((bi, axis, _)) = best else { return false };
    let mean = (boxes[bi].sum[axis] / boxes[bi].count) as u8;
    let mut lower = boxes[bi].clone();
    lower.hi[axis] = mean;
    boxes[bi].lo[axis] = mean + 1;
    stats(&mut boxes[bi], hist);
    stats(&mut lower, hist);
    boxes.push(lower);
    true
}

/// Reduce the 24bpp BGR image (tightly packed, `B,G,R` per pixel) to at most
/// `max_colors` colours, returning the palette as `0xFFRRGGBB` in native's order. `max_colors` is the
/// quantiser target (the FFI decrements it for the transparent-entry option).
pub fn median_cut(bgr: &[u8], max_colors: usize) -> Vec<u32> {
    if max_colors == 0 {
        return Vec::new();
    }
    // Every bin starts at 1; pixels add on top (capped at 0x1FFFFF).
    let mut hist = vec![1u32; 1 << (3 * BITS)];
    let mut uniq: Vec<u32> = Vec::new();
    let mut overflow = false;
    for px in bgr.chunks_exact(3) {
        let (b, g, r) = (px[0] as u32, px[1] as u32, px[2] as u32);
        let bin = &mut hist[hist_index(r >> DROP, g >> DROP, b >> DROP)];
        if *bin != CAP {
            *bin += 1;
        }
        if !overflow {
            let key = r | g << 8 | b << 16;
            if !uniq.contains(&key) {
                uniq.push(key);
                overflow = uniq.len() > max_colors;
            }
        }
    }

    // Few colours: the exact distinct colours in first-seen order.
    if !overflow {
        return uniq
            .iter()
            .map(|&k| 0xFF00_0000 | (k & 0xFF) << 16 | (k & 0xFF00) | (k >> 16) & 0xFF)
            .collect();
    }

    // Median cut over the occupied cube.
    let mut boxes = vec![Box { lo: [0; 3], hi: [(LEVELS - 1) as u8; 3], sum: [0; 3], count: 0 }];
    stats(&mut boxes[0], &hist);
    while boxes.len() < max_colors && split_best(&mut boxes, &hist) {}

    // Emit each box's weighted-average colour; boxes by descending population.
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| boxes[b].count.cmp(&boxes[a].count));
    order
        .iter()
        .map(|&i| {
            let bx = &boxes[i];
            let chan = |axis: usize| ((LEVELS - 1) / 2 + 255 * (bx.sum[axis] / bx.count)) / (LEVELS - 1);
            0xFF00_0000 | chan(0) << 16 | chan(1) << 8 | chan(2)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bgr(pixels: &[(u8, u8, u8)]) -> Vec<u8> {
        pixels.iter().flat_map(|&(r, g, b)| [b, g, r]).collect()
    }

    #[test]
    fn few_colours_are_exact_in_first_seen_order() {
        // ≤ max_colors distinct → exact 8-bit colours, insertion order, no quantisation.
        let px = bgr(&[(10, 20, 30), (40, 50, 60), (10, 20, 30)]);
        assert_eq!(median_cut(&px, 4), [0xFF0A_141E, 0xFF28_323C]);
    }

    #[test]
    fn median_cut_weighted_average_over_unit_background() {
        // 3 greys > 2 colours → one R split at the weighted mean (15): box {R5 0..15} (more pixels)
        // and {R5 16..31}. With every bin at 1, each box's G/B average is the box centre → 0x7b, and R
        // is the half's centre (7→0x3a, 23→0xbd). Boxes by population. Matches native `skew_c2`.
        let px = bgr(&[(0, 0, 0), (0, 0, 0), (0, 0, 0), (128, 128, 128), (255, 255, 255)]);
        assert_eq!(median_cut(&px, 2), [0xFF3A_7B7B, 0xFFBD_7B7B]);
    }
}
