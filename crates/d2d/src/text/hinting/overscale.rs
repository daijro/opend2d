//! DirectWrite's GDI-classic x-overscale subsystem, ported from its reference
//! rasterizer. Two halves:
//!
//! * **Link memory** ([`add_distance`]/[`add_proportion`], via [`double_check_link_color`]) — the
//!   x-direction MIRP/MDRP/IP point-movers record, per glyph point, which stem edge or reference
//!   points it is tied to (native's link-building steps: distance, proportion, link-colour,
//!   indirect dependency, inter-align). The engine calls these mid-program.
//! * **Phase-control post-pass** ([`execute_phase_control`]) — after the glyph program, native's
//!   phase-control post-pass (`PhaseShift`/`CalcAvgXPhaseShift`) squeezes each x by the global overscale
//!   factor, moving linked stem edges coherently so stems keep their width while the glyph fits the
//!   GDI-compatible advance grid.
//!
//! All of it is gated on GDI-classic ([`state::GraphicsState::gdi_overscale`]); other modes leave the
//! link memory empty and skip the post-pass, so this file is inert for them.

use super::state::{LinkRec, Zone};

thread_local! {
    // The GDI-classic link memory after the last glyph program (before the phase post-pass), snapshot
    // for unit-testing against native's phase dump. `None` unless capture is enabled.
    static CAPTURE: std::cell::RefCell<Option<Vec<LinkRec>>> = const { std::cell::RefCell::new(None) };
}

/// Start capturing the link memory on this thread; pair with [`take_capture`] after `Instance::hint`.
pub fn enable_capture() {
    CAPTURE.with(|c| *c.borrow_mut() = Some(Vec::new()));
}
/// Take the captured link memory (the last glyph hinted while capture was enabled).
pub fn take_capture() -> Option<Vec<LinkRec>> {
    CAPTURE.with(|c| c.borrow_mut().take())
}
/// Record the post-glyph-program link memory (called by `Instance::execute`).
pub(super) fn record(links: &[LinkRec]) {
    CAPTURE.with(|c| {
        if let Some(p) = c.borrow_mut().as_mut() {
            *p = links.to_vec();
        }
    });
}

/// Contour index of point `p` (native `ContNum`): the first contour whose last point index is `>= p`,
/// or `-1` for an out-of-range point (the four phantoms have no contour). `ends` is one past-the-last
/// index per contour (the TrueType `endPtsOfContours`).
fn cont_num(ends: &[u16], p: i32) -> i32 {
    for (c, &e) in ends.iter().enumerate() {
        if p <= e as i32 {
            return c as i32;
        }
    }
    -1
}

fn contour_start(ends: &[u16], c: usize) -> i32 {
    if c == 0 { 0 } else { ends[c - 1] as i32 + 1 }
}

/// Does `a` (transitively, through its `p1`/`p2` links) depend on `b`? Native `IndirectlyDependsOn`,
/// a depth-limited recursion (the cycle guard before adding a link). The depth bottoms out as a
/// conservative "yes" to break cycles, exactly as native does.
fn indirectly_depends_on(links: &[LinkRec], a: i32, b: i32, depth: i32) -> bool {
    let v4 = depth - 1;
    if v4 < 0 {
        return true;
    }
    let a = a as usize;
    let l = links[a];
    if l.p1 == -1 {
        return false;
    }
    if l.p2 == -1 {
        if l.p1 == b {
            return true;
        }
        indirectly_depends_on(links, l.p1, b, v4 - 1)
    } else {
        if l.p1 == b || l.p2 == b {
            return true;
        }
        if indirectly_depends_on(links, l.p1, b, v4 - 1) {
            return true;
        }
        indirectly_depends_on(links, l.p2, b, v4 - 1)
    }
}

/// Whether `a3`'s original x lies between `a2`'s and `a4`'s (native `InterAlign`, on `elem+32` = the
/// unscaled/orus x). Used by MDRP to decide a two-reference "proportion" link vs a one-reference
/// "distance" link.
pub fn inter_align(z: &Zone, a2: i32, a3: i32, a4: i32) -> bool {
    let ox = |i: i32| z.unscaled(i as usize).x;
    let (p2, p4, p3) = (ox(a2), ox(a4), ox(a3));
    let lo = p2.min(p4);
    if lo > p3 {
        return false;
    }
    p3 <= p2.max(p4)
}

/// The link "colour" of a candidate near-horizontal stem edge `a`→`b` (native `DoubleCheckLinkColor`):
/// `0` no link, `default` (the caller's `op&3`) for a cross-contour / non-adjacent pair, else `1`/`2`
/// keyed off the local contour orientation XOR the contour fill bit. Colour `1` triggers the average
/// cross-link in [`add_distance`].
pub fn double_check_link_color(z: &Zone, fill_bits: &[bool], a: i32, b: i32, default: i32) -> i32 {
    let ca = cont_num(&z.contours, a);
    if ca < 0 {
        return 0;
    }
    let cb = cont_num(&z.contours, b);
    if cb < 0 {
        return 0;
    }
    if ca != cb {
        return default;
    }
    let c = ca as usize;
    let start = contour_start(&z.contours, c);
    let end = z.contours[c] as i32;
    let next_a = if a != end { a + 1 } else { start };
    let prev_a = if a != start { a - 1 } else { end };
    if b != next_a && b != prev_a {
        return default;
    }
    let ox = |i: i32| z.unscaled(i as usize).x;
    let oy = |i: i32| z.unscaled(i as usize).y;
    // Cross product of the (prev,cur,next) edges at `a`, 32-bit wrapping like native.
    let cross = |cur: i32, prev: i32, next: i32| -> bool {
        let l = (oy(next).wrapping_sub(oy(cur))).wrapping_mul(ox(cur).wrapping_sub(ox(prev)));
        let r = (ox(next).wrapping_sub(ox(cur))).wrapping_mul(oy(cur).wrapping_sub(oy(prev)));
        l < r
    };
    let orient_a = cross(a, prev_a, next_a);
    let prev_b = if b != start { b - 1 } else { end };
    let next_b = if b != end { b + 1 } else { start };
    let orient_b = cross(b, prev_b, next_b);
    if orient_a != orient_b {
        return 0;
    }
    let dx = ox(b).wrapping_sub(ox(a));
    let dy = oy(b).wrapping_sub(oy(a));
    if 2 * dx.abs() >= dy.abs() {
        (orient_a ^ !fill_bits[c]) as i32 + 1
    } else {
        0
    }
}

/// Native's distance link (and the equivalent block inlined in MIRP): link point `p` to the stem
/// edge anchored at `rp0`, walking `rp0`'s parent chain while the unscaled x is unchanged, and — for
/// colour 1 — recording the average cross-link.
pub fn add_distance(z: &Zone, links: &mut [LinkRec], rp0: i32, p: i32, color: i32) {
    if rp0 < 0 {
        return;
    }
    let n = links.len() as i32;
    if !(rp0 < n && p >= 0 && p < n && rp0 != p) {
        return;
    }
    let (rp0u, pu) = (rp0 as usize, p as usize);
    let mut chain_end = rp0;
    if indirectly_depends_on(links, rp0, p, 100) {
        links[pu].flags |= 1;
    } else if links[pu].p1 == -1 {
        let mut i = links[rp0u].p1;
        while i != -1 {
            if z.unscaled(chain_end as usize).x != z.unscaled(i as usize).x {
                break;
            }
            chain_end = i;
            i = links[i as usize].p1;
        }
        links[pu].p1 = chain_end;
        links[pu].p2 = -1;
    }
    if color == 1 {
        let ce = chain_end as usize;
        if links[ce].child == -1 && links[pu].child != chain_end {
            let ce_p1 = links[ce].p1;
            if ce_p1 != -1 && links[ce_p1 as usize].child == chain_end {
                links[pu].flags |= 1;
            } else {
                links[ce].child = p;
            }
        }
    }
}

/// Native `AddProportion`: link point `p` to two references `r1`/`r2` (IP, or MDRP's aligned case) so
/// the post-pass interpolates its shift between them.
pub fn add_proportion(links: &mut [LinkRec], r1: i32, p: i32, r2: i32) {
    if r1 < 0 {
        return;
    }
    let n = links.len() as i32;
    if !(r1 < n && r2 >= 0 && r2 < n && p >= 0 && p < n && r1 != p && r2 != p && r1 != r2) {
        return;
    }
    let pu = p as usize;
    if indirectly_depends_on(links, r1, p, 100) || indirectly_depends_on(links, r2, p, 100) {
        links[pu].flags |= 1;
    } else if links[pu].p1 == -1 && links[pu].p2 == -1 {
        links[pu].p1 = r1;
        links[pu].p2 = r2;
    }
}

/// Per-contour fill-direction bits (native `elem+88`), from the sign of each contour's signed area in
/// the unscaled/orus coordinates. `DoubleCheckLinkColor` XORs this with the local orientation.
pub fn fill_bits(z: &Zone) -> Vec<bool> {
    let mut bits = Vec::with_capacity(z.contours.len());
    for c in 0..z.contours.len() {
        let start = contour_start(&z.contours, c) as usize;
        let end = z.contours[c] as usize;
        let mut area: i64 = 0;
        for i in start..=end {
            let a = z.unscaled(i);
            let b = z.unscaled(if i == end { start } else { i + 1 });
            area += a.x as i64 * b.y as i64 - b.x as i64 * a.y as i64;
        }
        bits.push(area > 0);
    }
    bits
}

/// Round-half-away-from-zero divide (native `CompDiv(divisor, dividend)`): `round(dividend / divisor)`.
/// `divisor` is the post-pass constant `0x20000` here, so the native divide-by-zero saturation is moot.
fn comp_div(divisor: i32, dividend: i64) -> i32 {
    let half = (divisor / 2) as i64;
    let v = if (divisor < 0) == (dividend < 0) { dividend + half } else { dividend - half };
    (v / divisor as i64) as i32
}

/// Linear-interpolate a point's x-shift between its two parents by original x (native
/// `CalcAvgXPhaseShift`). `org_x` is the scaled original x (native `elem+16`).
fn calc_avg(org_x: &[i32], p1: i32, cur: i32, p2: i32, shift1: i32, shift2: i32) -> i32 {
    if shift1 == shift2 {
        return shift1;
    }
    let mid = org_x[cur as usize];
    // Order the two parents by original x; `shift_hi`/`shift_lo` follow their parent.
    let (hi, lo, shift_hi, shift_lo) = if org_x[p1 as usize] >= org_x[p2 as usize] {
        (org_x[p1 as usize], org_x[p2 as usize], shift1, shift2)
    } else {
        (org_x[p2 as usize], org_x[p1 as usize], shift2, shift1)
    };
    if lo == hi {
        return (shift_hi + shift_lo) / 2;
    }
    (shift_lo * (hi - mid) + shift_hi * (mid - lo)) / (hi - lo)
}

/// Recursive per-point shift resolution (native `PhaseShift`). Modifies `x` (the current x array) in
/// place and memoises each point's computed shift in `links[i].flags`/`.shift`-equivalent via the
/// returned value; `shift` carries the cached result. `last` is the last contour point index.
#[allow(clippy::too_many_arguments)]
fn phase_shift(
    x: &mut [i32],
    org_x: &[i32],
    links: &mut [LinkRec],
    shift: &mut [i32],
    overscale: i32,
    has_broken: bool,
    last: i32,
    p: i32,
) -> i32 {
    if p == -1 {
        return 0;
    }
    let pu = p as usize;
    if links[pu].flags & 4 != 0 {
        return 0;
    }
    links[pu].flags |= 4;
    if links[pu].flags & 2 != 0 {
        links[pu].flags &= !4;
        return shift[pu];
    }
    let os = (overscale - 0x10000) as i64;
    let mut delta;
    // Phantom lsb/advance (indices last+1, last+2) get the raw overscale shift; every other point
    // (real points and the top/bottom phantoms) resolves through its links.
    if last >= p || p > last + 2 {
        delta = if links[pu].p1 == -1 {
            if !has_broken {
                0
            } else {
                comp_div(0x20000, os * (2 * x[pu] as i64))
            }
        } else if links[pu].p2 == -1 {
            phase_shift(x, org_x, links, shift, overscale, has_broken, last, links[pu].p1)
        } else {
            let p1 = links[pu].p1;
            let p2 = links[pu].p2;
            let s2 = phase_shift(x, org_x, links, shift, overscale, has_broken, last, p2);
            let s1 = phase_shift(x, org_x, links, shift, overscale, has_broken, last, p1);
            calc_avg(org_x, p1, p, p2, s1, s2)
        };
    } else {
        delta = comp_div(0x20000, os * (2 * x[pu] as i64));
    }
    // apply the shift (and the average cross-link / same-x-parent coalescing).
    if links[pu].flags & 2 == 0 {
        let child = links[pu].child;
        let mut handled = false;
        if child != -1 && links[child as usize].flags & 2 == 0 {
            if links[pu].p1 == -1 || links[pu].p2 == -1 {
                delta = comp_div(0x20000, os * (x[pu] as i64 + x[child as usize] as i64));
            }
            x[pu] += delta;
            x[child as usize] += delta;
            links[child as usize].flags |= 2;
            shift[child as usize] = delta;
            handled = true;
        }
        if !handled {
            if has_broken && links[pu].p2 == -1 {
                if links[pu].p1 != -1 && org_x[pu] == org_x[links[pu].p1 as usize] {
                    // same original x as parent → apply the inherited shift to x directly.
                    x[pu] += delta;
                } else {
                    delta = comp_div(0x20000, os * (2 * x[pu] as i64));
                    if links[pu].p1 != -1 {
                        x[pu] += delta;
                    }
                }
            } else if links[pu].p1 != -1 {
                x[pu] += delta;
            }
        }
        links[pu].flags |= 2;
        shift[pu] = delta;
    }
    links[pu].flags &= !4;
    shift[pu]
}

/// Native's GDI-classic x-overscale post-pass. Squeezes `x` (current x, one
/// per glyph-zone point incl. phantoms) in place using the link memory and the global `overscale`
/// (16.16). `org_x` is the scaled original x; `last` is the last real contour point index (phantoms
/// follow at `last+1..=last+4`). Returns each point's cached shift (unused by the renderer).
pub fn execute_phase_control(x: &mut [i32], org_x: &[i32], links: &mut [LinkRec], overscale: i32, last: i32) {
    let npts = (last + 5).max(0) as usize;
    if npts == 0 || npts > x.len() {
        return;
    }
    let has_broken = links[..npts].iter().any(|l| l.flags & 1 != 0);
    let mut shift = vec![0i32; npts];
    for p in 0..npts as i32 {
        phase_shift(x, org_x, links, &mut shift, overscale, has_broken, last, p);
    }
}
