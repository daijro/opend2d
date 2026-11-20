//! Interpreter graphics state: unit vectors, zones, reference/zone pointers and the
//! point-movement primitives. Mirrors the reference rasterizer's state exactly, with
//! one deliberate omission: there is **no** FreeType-style "backward compatibility"
//! mode. The classic `dwrite.dll` interpreter has no such concept — points
//! move freely and ClearType behaviour comes from the font branching on GETINFO, not
//! from an interpreter-level clamp.

use super::math::{self, RoundMode, RoundState};

/// A point or vector as raw `i32` bits: F26Dot6 for coordinates/distances, 2.14 for
/// unit vectors (1.0 = 0x4000).
#[derive(Copy, Clone, Default, PartialEq, Eq, Debug)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

impl core::ops::Add for Point {
    type Output = Point;
    fn add(self, o: Point) -> Point {
        Point::new(self.x.wrapping_add(o.x), self.y.wrapping_add(o.y))
    }
}
impl core::ops::Sub for Point {
    type Output = Point;
    fn sub(self, o: Point) -> Point {
        Point::new(self.x.wrapping_sub(o.x), self.y.wrapping_sub(o.y))
    }
}

/// Axis a vector or movement is aligned to; `Both` means an arbitrary direction.
#[derive(Copy, Clone, PartialEq, Eq, Default, Debug)]
pub enum CoordAxis {
    #[default]
    Both,
    X,
    Y,
}

// Point flag bits (our own layout; only the three the interpreter needs).
pub const ON_CURVE: u8 = 0x01;
pub const TOUCHED_X: u8 = 0x02;
pub const TOUCHED_Y: u8 = 0x04;

impl CoordAxis {
    fn touch_mask(self) -> u8 {
        match self {
            CoordAxis::X => TOUCHED_X,
            CoordAxis::Y => TOUCHED_Y,
            CoordAxis::Both => TOUCHED_X | TOUCHED_Y,
        }
    }
}

/// GDI-classic overscale "link memory": one record per glyph point (phantoms included), mirroring
/// native's `elem+104` 12-byte record. The x-direction MIRP/MDRP/IP point-movers fill it (which stem
/// edge / reference points each point is linked to) so the phase-control post-pass shifts linked
/// edges coherently instead of squeezing every point independently. `-1` is native's `0xFFFF`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LinkRec {
    /// Chain parent (distance link) or first reference (proportion link); `-1` = none.
    pub p1: i32,
    /// Second reference (AddProportion / two-point IP); `-1` = none.
    pub p2: i32,
    /// Average partner (DoubleCheckLinkColor colour-1 cross-link); `-1` = none.
    pub child: i32,
    /// 0x1 dependency-broken, 0x2 shift-computed, 0x4 visiting (post-pass).
    pub flags: u16,
}

impl Default for LinkRec {
    fn default() -> Self {
        Self { p1: -1, p2: -1, child: -1, flags: 0 }
    }
}

/// Selects the twilight (0) or glyph (1) zone.
#[derive(Copy, Clone, PartialEq, Eq, Default, Debug)]
pub enum ZonePointer {
    Twilight = 0,
    #[default]
    Glyph = 1,
}

impl ZonePointer {
    pub fn is_twilight(self) -> bool {
        self == ZonePointer::Twilight
    }
    pub fn from_stack(v: i32) -> Option<Self> {
        match v {
            0 => Some(ZonePointer::Twilight),
            1 => Some(ZonePointer::Glyph),
            _ => None,
        }
    }
}

/// Point buffers for one zone. `unscaled` is font units for a simple glyph and
/// already-scaled coordinates for a composite (see [`GraphicsState::unscaled_to_pixels`]);
/// empty for the twilight zone, which reads as the origin.
#[derive(Default, Clone)]
pub struct Zone {
    pub unscaled: Vec<Point>,
    pub original: Vec<Point>,
    pub cur: Vec<Point>,
    pub flags: Vec<u8>,
    pub contours: Vec<u16>,
}

impl Zone {
    pub fn point(&self, i: usize) -> Point {
        self.cur.get(i).copied().unwrap_or_default()
    }
    pub fn original(&self, i: usize) -> Point {
        self.original.get(i).copied().unwrap_or_default()
    }
    pub fn unscaled(&self, i: usize) -> Point {
        self.unscaled.get(i).copied().unwrap_or_default()
    }
    pub fn len(&self) -> usize {
        self.cur.len()
    }
    pub fn contour(&self, i: usize) -> u16 {
        self.contours.get(i).copied().unwrap_or(0)
    }
    pub fn touch(&mut self, i: usize, axis: CoordAxis) {
        if let Some(f) = self.flags.get_mut(i) {
            *f |= axis.touch_mask();
        }
    }
    pub fn untouch(&mut self, i: usize, axis: CoordAxis) {
        if let Some(f) = self.flags.get_mut(i) {
            *f &= !axis.touch_mask();
        }
    }
    pub fn is_touched(&self, i: usize, axis: CoordAxis) -> bool {
        self.flags.get(i).map_or(false, |f| f & axis.touch_mask() != 0)
    }
    pub fn flip_on_curve(&mut self, i: usize) {
        if let Some(f) = self.flags.get_mut(i) {
            *f ^= ON_CURVE;
        }
    }
    pub fn set_on_curve(&mut self, start: usize, end: usize, on: bool) {
        if let Some(fs) = self.flags.get_mut(start..end) {
            for f in fs {
                if on {
                    *f |= ON_CURVE;
                } else {
                    *f &= !ON_CURVE;
                }
            }
        }
    }

    /// Interpolate untouched points along `axis` (IUP). Reproduces the reference
    /// rasterizer's contour walk.
    pub fn iup(&mut self, axis: CoordAxis) {
        let mut point = 0usize;
        for c in 0..self.contours.len() {
            let mut end = self.contour(c) as usize;
            let first = point;
            if end >= self.cur.len() {
                end = self.cur.len().saturating_sub(1);
            }
            while point <= end && !self.is_touched(point, axis) {
                point += 1;
            }
            if point <= end {
                let first_touched = point;
                let mut cur_touched = point;
                point += 1;
                while point <= end {
                    if self.is_touched(point, axis) {
                        self.iup_interp(axis, cur_touched + 1, point - 1, cur_touched, point);
                        cur_touched = point;
                    }
                    point += 1;
                }
                if cur_touched == first_touched {
                    self.iup_shift(axis, first, end, cur_touched);
                } else {
                    self.iup_interp(axis, cur_touched + 1, end, cur_touched, first_touched);
                    if first_touched > 0 {
                        self.iup_interp(axis, first, first_touched - 1, cur_touched, first_touched);
                    }
                }
            }
        }
    }

    fn coord(p: Point, axis: CoordAxis) -> i32 {
        if axis == CoordAxis::X { p.x } else { p.y }
    }
    fn set_coord(p: &mut Point, axis: CoordAxis, v: i32) {
        if axis == CoordAxis::X {
            p.x = v;
        } else {
            p.y = v;
        }
    }

    fn iup_shift(&mut self, axis: CoordAxis, p1: usize, p2: usize, p: usize) {
        if p1 > p2 || p1 > p || p > p2 {
            return;
        }
        let delta = Self::coord(self.point(p), axis).wrapping_sub(Self::coord(self.original(p), axis));
        if delta == 0 {
            return;
        }
        for i in p1..=p2 {
            if i == p {
                continue;
            }
            if let Some(pt) = self.cur.get_mut(i) {
                let v = Self::coord(*pt, axis).wrapping_add(delta);
                Self::set_coord(pt, axis, v);
            }
        }
    }

    fn iup_interp(&mut self, axis: CoordAxis, p1: usize, p2: usize, mut r1: usize, mut r2: usize) {
        if p1 > p2 {
            return;
        }
        let n = self.cur.len();
        if r1 >= n || r2 >= n {
            return;
        }
        let mut orus1 = Self::coord(self.unscaled(r1), axis);
        let mut orus2 = Self::coord(self.unscaled(r2), axis);
        if orus1 > orus2 {
            core::mem::swap(&mut orus1, &mut orus2);
            core::mem::swap(&mut r1, &mut r2);
        }
        let org1 = Self::coord(self.original(r1), axis);
        let org2 = Self::coord(self.original(r2), axis);
        let cur1 = Self::coord(self.point(r1), axis);
        let cur2 = Self::coord(self.point(r2), axis);
        let delta1 = cur1.wrapping_sub(org1);
        let delta2 = cur2.wrapping_sub(org2);
        let cur_range = cur2.wrapping_sub(cur1);
        let orus_range = orus2.wrapping_sub(orus1); // >= 0 after the sort above
        // For wide ranges native (the IUP handler) falls back to a 16.16 scale (FixDiv then FixMul, two
        // roundings); for the small ranges small text always hits it uses one rounded integer
        // muldiv `cur1 + (cur_range*(o-orus1) + orus_range/2) / orus_range`. Reproduce both, gated on
        // native's signed `< 0x8000` threshold on both components.
        let scale = (orus_range >= 0x8000 || cur_range >= 0x8000)
            .then(|| math::fixdiv(cur_range, orus_range));
        for i in p1..=p2 {
            if i >= n {
                break;
            }
            let a = Self::coord(self.original[i], axis);
            let new = if a <= org1 {
                a.wrapping_add(delta1)
            } else if a >= org2 {
                a.wrapping_add(delta2)
            } else if orus_range == 0 {
                // orus1 == orus2: native breaks before scaling and shifts the span by the lower
                // reference's delta.
                a.wrapping_add(delta1)
            } else if let Some(scale) = scale {
                let o = Self::coord(self.unscaled(i), axis);
                cur1.wrapping_add(math::fixmul(o.wrapping_sub(orus1), scale))
            } else {
                let o = Self::coord(self.unscaled(i), axis) as i64;
                let num = cur_range as i64 * (o - orus1 as i64) + orus_range as i64 / 2;
                cur1.wrapping_add((num / orus_range as i64) as i32)
            };
            Self::set_coord(&mut self.cur[i], axis, new);
        }
    }
}

/// The GETINFO result bytes the reference rasterizer configures per rendering mode.
/// Version is 42 (`0x2A`, read from the GETINFO handler); the smooth flags map to the
/// standard Microsoft result bits. Values are set by [`super::Config`]'s `getinfo`.
#[derive(Copy, Clone, Debug)]
pub struct GetInfoFlags {
    pub version: i32,
    pub subpixel_hinting: bool,
    pub compatible_width: bool,
    pub vertical_lcd: bool,
    pub bgr: bool,
    pub subpixel_positioned: bool,
    pub symmetric: bool,
    pub grayscale_cleartype: bool,
}

impl Default for GetInfoFlags {
    fn default() -> Self {
        Self {
            version: 42,
            subpixel_hinting: false,
            compatible_width: false,
            vertical_lcd: false,
            bgr: false,
            subpixel_positioned: false,
            symmetric: false,
            grayscale_cleartype: false,
        }
    }
}

/// Graphics-state fields that persist from `prep` across every glyph in an instance.
#[derive(Copy, Clone, Debug)]
pub struct Retained {
    pub auto_flip: bool,
    pub control_value_cutin: i32,
    pub delta_base: u16,
    pub delta_shift: u16,
    pub instruct_control: u8,
    /// FDEF bytecode recognition: function 58 constructs a temporary phantom-point slope vector.
    pub slope_function: bool,
    /// Which recognised font-tool functions the font program defined (`engine::KNOWN_*` bits).
    pub known_functions: u8,
    pub min_distance: i32,
    /// The raw `SCANCTRL` word, evaluated per glyph as native's glyph setup does ([`Self::dropout_on`]):
    /// the last SCANCTRL wins. Native's pre-program setup seeds it with 0x120 (dropout at ppem <= 32).
    pub scan_word: i32,
    pub scan_type: i32,
    pub single_width_cutin: i32,
    pub single_width: i32,
    pub scale: i32,
    /// Per-axis FUnit scales, as native's scaler keeps them; `scale` is the larger, which the CVT,
    /// WCVTF and SSW use.
    pub x_scale: i32,
    pub y_scale: i32,
    /// The CVT stretch ratios: the smaller axis scale over the larger, 0x10000 on the larger
    /// axis. Both 0x10000 = square, where native reads the CVT, MPPEM and MDRP's original
    /// distance without them.
    pub x_ratio: i32,
    pub y_ratio: i32,
    pub ppem: i32,
    pub is_rotated: bool,
    pub is_stretched: bool,
    /// How the glyph→device matrix rotates, selecting a composite's component-offset rounding
    /// ([`super::Instance::hint_composite`]): `0` axis-aligned (base rule — x on the subpixel bank iff
    /// not vertical-LCD, y the reverse); `1` a pure quarter turn (both diagonals collapse — native swaps
    /// the two axes' tests); `2` a general rotation or skew (an off-diagonal AND a diagonal both live —
    /// both axes round on the subpixel bank).
    pub rotation_kind: u8,
    pub getinfo: GetInfoFlags,
    /// GDI-classic render pass: enables x-subpixel hinting even under
    /// compatible width. See [`super::Config::gdi_render`].
    pub gdi_render: bool,
    /// GDI-natural antialiased: snap the advance phantom's **current** x to the 1/16-px subpixel grid
    /// (round_sp) instead of the whole-pixel grid, because native keeps that phantom sub-pixel and
    /// advance-relative MIRPs position stems off it. Set from [`super::Config::subpixel_advance`]
    /// (aa-gated by the caller: aliased GDI-natural keeps the whole-pixel phantom).
    pub compatible_advance: bool,
    pub font_cache: bool,
}

impl Default for Retained {
    fn default() -> Self {
        Self {
            auto_flip: true,
            control_value_cutin: 68, // 17/16 px in F26Dot6
            delta_base: 9,
            delta_shift: 3,
            instruct_control: 0,
            slope_function: false,
            known_functions: 0,
            min_distance: 64, // 1 px
            scan_word: 0x120,
            scan_type: 0,
            single_width_cutin: 0,
            single_width: 0,
            scale: 0,
            x_scale: 0,
            y_scale: 0,
            x_ratio: 0x10000,
            y_ratio: 0x10000,
            ppem: 0,
            is_rotated: false,
            is_stretched: false,
            rotation_kind: 0,
            getinfo: GetInfoFlags::default(),
            gdi_render: false,
            compatible_advance: false,
            font_cache: false,
        }
    }
}

impl Retained {
    /// Native's square flag: the x and y scales are equal.
    pub fn square(&self) -> bool {
        self.x_ratio == 0x10000 && self.y_ratio == 0x10000
    }

    /// Native's glyph setup: dropout control is on when the SCANCTRL word asks for it at this ppem
    /// (bit 8 with the ppem, clamped to 255, at most the low-byte threshold; or threshold 0xFF with bit 8),
    /// or for a rotated (bit 9) or stretched (bit 10) glyph. The "off" bits 11-13 play no part.
    pub fn dropout_on(&self) -> bool {
        let (w, ppem) = (self.scan_word, self.ppem.clamp(0, 255));
        let on = w & 0x100 != 0;
        (on && ppem <= w & 0xFF) || (on && w & 0xFF == 0xFF) || (w & 0x200 != 0 && self.is_rotated) || (w & 0x400 != 0 && self.is_stretched)
    }
}

/// Full interpreter graphics state for one program run.
pub struct GraphicsState {
    pub ret: Retained,
    pub proj: Point,
    pub proj_axis: CoordAxis,
    pub dual: Point,
    pub dual_axis: CoordAxis,
    pub free: Point,
    pub free_axis: CoordAxis,
    pub fdotp: i32,
    pub round: RoundState,
    pub rp0: usize,
    pub rp1: usize,
    pub rp2: usize,
    pub loop_counter: u32,
    pub zp0: ZonePointer,
    pub zp1: ZonePointer,
    pub zp2: ZonePointer,
    pub zones: [Zone; 2],
    pub is_composite: bool,
    /// True only while running the glyph program. The 1/16-px subpixel round bank (`subpixel_round`)
    /// is selected in native's interpreter for the glyph program only; `fpgm`/`prep` run with the
    /// default whole-pixel round function, so CVT rounding there is whole-pixel. Gating on this keeps
    /// prep's CVT cut-in values on the whole grid (a stem CVT rounds to 2.0 px, not 2.125).
    pub is_glyph: bool,
    /// Cached in [`Self::update_projection`]: the current projection runs along the DirectWrite
    /// "subpixel axis" (x for horizontal ClearType/grayscale, y for vertical LCD) in a pure smooth
    /// mode, so point-moving instructions round to a 1/16-px grid instead of the whole-pixel grid.
    pub subpixel_round: bool,
    /// GDI-classic overscale mode (native's global GDI-classic + subpixel flags): the x-direction movers
    /// build [`links`](Self::links) and a phase-control post-pass squeezes x. Only GDI-classic
    /// (compatible width, whole-pixel positioning); GDI-natural and the smooth modes leave it false.
    pub gdi_overscale: bool,
    /// The GDI-classic link memory (one [`LinkRec`] per glyph-zone point), built during the glyph
    /// program and consumed by the post-pass. Empty unless [`gdi_overscale`](Self::gdi_overscale).
    pub links: Vec<LinkRec>,
    /// Per-contour fill-direction bit (native `elem+88[c]&1`), for `DoubleCheckLinkColor`'s colour.
    pub fill_bits: Vec<bool>,
    /// The two reference points of the last SPVTL/SDPVTL projection line, for
    /// MDRP's `InterAlign`/`AddProportion` branch; `(-1, -1)` when none.
    pub proj_ref: (i32, i32),
    /// Native's dual projection is the generic projection: SPVTL and SPVFS set it whatever direction the
    /// vector lands on; SVTCA/SPVTCA set the axis projections and SDPVTL the old-points projection.
    pub proj_generic: bool,
    /// GDI-classic: the phase-control post-pass has run (native's IUP handler calls it once, on the subpixel
    /// IUP axis). Guards against a second application.
    pub phase_done: bool,
    /// GDI-classic render-pass x-overscale factor (16.16); the phase-control pass (run
    /// during the subpixel `IUP`) squeezes x by `(overscale-0x10000)`. `0x10000` (identity) otherwise.
    pub overscale: i32,
}

impl GraphicsState {
    pub fn new(ret: Retained, twilight: Zone, glyph: Zone) -> Self {
        let mut gs = Self {
            ret,
            proj: Point::new(0x4000, 0),
            proj_axis: CoordAxis::Both,
            dual: Point::new(0x4000, 0),
            dual_axis: CoordAxis::Both,
            free: Point::new(0x4000, 0),
            free_axis: CoordAxis::Both,
            fdotp: 0x4000,
            round: RoundState::default(),
            rp0: 0,
            rp1: 0,
            rp2: 0,
            loop_counter: 1,
            zp0: ZonePointer::Glyph,
            zp1: ZonePointer::Glyph,
            zp2: ZonePointer::Glyph,
            zones: [twilight, glyph],
            is_composite: false,
            is_glyph: false,
            subpixel_round: false,
            gdi_overscale: ret.getinfo.compatible_width && !ret.getinfo.subpixel_positioned,
            links: Vec::new(),
            fill_bits: Vec::new(),
            proj_ref: (-1, -1),
            proj_generic: false,
            phase_done: false,
            overscale: 0x10000,
        };
        gs.update_projection();
        gs
    }

    /// Reset the non-retained part of the state (vectors, round state, reference and
    /// zone pointers, loop counter) to defaults at the start of each program.
    pub fn reset_volatile(&mut self) {
        self.proj = Point::new(0x4000, 0);
        self.dual = Point::new(0x4000, 0);
        self.free = Point::new(0x4000, 0);
        self.fdotp = 0x4000;
        self.round = RoundState::default();
        self.rp0 = 0;
        self.rp1 = 0;
        self.rp2 = 0;
        self.loop_counter = 1;
        self.zp0 = ZonePointer::Glyph;
        self.zp1 = ZonePointer::Glyph;
        self.zp2 = ZonePointer::Glyph;
        self.proj_ref = (-1, -1);
        self.update_projection();
    }

    #[inline]
    pub fn zone(&self, zp: ZonePointer) -> &Zone {
        &self.zones[zp as usize]
    }
    #[inline]
    pub fn zone_mut(&mut self, zp: ZonePointer) -> &mut Zone {
        &mut self.zones[zp as usize]
    }

    /// Scale factor to turn a zone's "unscaled" coordinate into pixels: the font-unit
    /// scale for a simple glyph, identity for a composite (whose points are pre-scaled).
    pub fn unscaled_to_pixels(&self) -> i32 {
        if self.is_composite {
            0x10000
        } else {
            self.ret.scale
        }
    }

    pub fn in_bounds(&self, pairs: &[(ZonePointer, usize)]) -> bool {
        pairs.iter().all(|&(zp, i)| i < self.zone(zp).len())
    }

    /// Recompute `fdotp`, the per-vector fast axes and the small-`fdotp` clamp after any
    /// vector change (reference `Compute_Funcs`).
    pub fn update_projection(&mut self) {
        // A vector is an axis only when its other component is zero: a near-axis line normalizes to e.g.
        // (0x4000, 68), and native's generic project/move use both components.
        const X: Point = Point { x: 0x4000, y: 0 };
        const Y: Point = Point { x: 0, y: 0x4000 };
        const ONE: i32 = 0x4000;
        if self.free == X {
            self.fdotp = self.proj.x;
        } else if self.free == Y {
            self.fdotp = self.proj.y;
        } else {
            // Native's projection compute-and-check rounds each component before adding.
            self.fdotp = project_diagonal(self.proj, self.free) as i16 as i32;
        }
        self.proj_axis = CoordAxis::Both;
        if self.proj == X {
            self.proj_axis = CoordAxis::X;
        } else if self.proj == Y {
            self.proj_axis = CoordAxis::Y;
        }
        self.dual_axis = CoordAxis::Both;
        if self.dual == X {
            self.dual_axis = CoordAxis::X;
        } else if self.dual == Y {
            self.dual_axis = CoordAxis::Y;
        }
        self.free_axis = CoordAxis::Both;
        if self.fdotp == ONE {
            if self.free == X {
                self.free_axis = CoordAxis::X;
            } else if self.free == Y {
                self.free_axis = CoordAxis::Y;
            }
        }
        if self.fdotp.abs() < 0x400 {
            self.fdotp = if self.fdotp < 0 { -ONE } else { ONE };
        }
        // DirectWrite subpixel hinting (SVTCA/SPVTCA): on the subpixel axis the round
        // function switches to the 1/16-px bank. The subpixel axis is x for horizontal
        // ClearType/grayscale and y for vertical LCD, enabled only when the per-mode subpixel
        // bit is set (our `subpixel_hinting`) and not in compatible-width (GDI) mode, which uses the
        // overscale path instead.
        let g = &self.ret.getinfo;
        self.subpixel_round = self.is_glyph
            && g.subpixel_hinting
            // The GDI-classic render pass does x-subpixel hinting even under compatible width (hint-13);
            // every other compatible-width path (metric pass, GDI-natural) keeps whole-pixel x.
            && (!g.compatible_width || self.ret.gdi_render)
            && match self.proj_axis {
                // The subpixel round bank applies to the subpixel axis (x for horizontal
                // ClearType/grayscale, y for vertical LCD) AND to diagonal projections: native rounds
                // a diagonal stem's distance on the 1/16-px grid too, so a diagonal MDRP/MIRP along an
                // x freedom vector barely nudges x instead of snapping to the whole-pixel grid (read
                // from the oracle: Verdana/Courier 'A' legs stay ~unhinted in x — their diagonal MIRPs
                // round on the subpixel bank). Native's SPVTL keeps a diagonal on the bank under vertical
                // LCD too: only the exact positive non-subpixel axis leaves it (VM: a quarter-turned Verdana
                // 'W' MIRPs its diagonal stem tops to the 1/16-px bank).
                CoordAxis::X => !g.vertical_lcd,
                CoordAxis::Y => g.vertical_lcd,
                CoordAxis::Both => true,
            };
    }

    /// Round a projected distance to the grid, using the 1/16-px subpixel grid on the subpixel axis
    /// (see [`Self::subpixel_round`]) and the whole-pixel grid otherwise.
    pub fn round_dist(&self, distance: i32) -> i32 {
        // Native's round-down-to-grid (sub-pixel) delegates to the WHOLE-pixel round-down when the prep has
        // not set INSTCTRL selector 3 (bit 2 of the INSTCTRL word) and the projection was last set by SPVTL or
        // SPVFS, even when that vector is an axis (Times Bold 'x' MDAP[r] after a fpgm SPVTL, any mode). Only this round mode delegates; the
        // round-to-grid SP function always stays on the 1/16px bank.
        let diag_down = self.proj_generic && self.round.mode == RoundMode::DownToGrid && self.ret.instruct_control & 4 == 0;
        if self.subpixel_round && !diag_down {
            self.round.round_sp(distance)
        } else {
            self.round.round(distance)
        }
    }

    /// `MINIMUM_DISTANCE`, halved on the subpixel axis (native MDRP/MIRP `v3+140 / 2`).
    pub fn min_dist(&self) -> i32 {
        if self.subpixel_round {
            self.ret.min_distance / 2
        } else {
            self.ret.min_distance
        }
    }

    /// Projection of `a - b` along the projection vector.
    pub fn project(&self, a: Point, b: Point) -> i32 {
        match self.proj_axis {
            CoordAxis::X => a.x.wrapping_sub(b.x),
            CoordAxis::Y => a.y.wrapping_sub(b.y),
            CoordAxis::Both => {
                project_diagonal(a - b, self.proj)
            }
        }
    }
    /// Projection of `a - b` along the dual projection vector.
    pub fn dual_project(&self, a: Point, b: Point) -> i32 {
        match self.dual_axis {
            CoordAxis::X => a.x.wrapping_sub(b.x),
            CoordAxis::Y => a.y.wrapping_sub(b.y),
            CoordAxis::Both => {
                project_diagonal(a - b, self.dual)
            }
        }
    }

    /// Move an original (pre-hint) point along the freedom vector so its projection
    /// changes by `distance`.
    pub fn move_original(&mut self, zp: ZonePointer, i: usize, distance: i32) {
        let (fv, fdotp, axis) = (self.free, self.fdotp, self.free_axis);
        let p = match self.zones[zp as usize].original.get_mut(i) {
            Some(p) => p,
            None => return,
        };
        match axis {
            CoordAxis::X => p.x = p.x.wrapping_add(distance),
            CoordAxis::Y => p.y = p.y.wrapping_add(distance),
            CoordAxis::Both => {
                if fv.x != 0 {
                    p.x = p.x.wrapping_add(math::mul_div(distance, fv.x, fdotp));
                }
                if fv.y != 0 {
                    p.y = p.y.wrapping_add(math::mul_div(distance, fv.y, fdotp));
                }
            }
        }
    }

    /// Move a current point along the freedom vector by `distance` (and mark it touched).
    pub fn move_point(&mut self, zp: ZonePointer, i: usize, distance: i32) {
        let (fv, fdotp, axis) = (self.free, self.fdotp, self.free_axis);
        let zone = &mut self.zones[zp as usize];
        match axis {
            CoordAxis::X => {
                if let Some(p) = zone.cur.get_mut(i) {
                    p.x = p.x.wrapping_add(distance);
                }
                zone.touch(i, CoordAxis::X);
            }
            CoordAxis::Y => {
                if let Some(p) = zone.cur.get_mut(i) {
                    p.y = p.y.wrapping_add(distance);
                }
                zone.touch(i, CoordAxis::Y);
            }
            CoordAxis::Both => {
                if fv.x != 0 {
                    if let Some(p) = zone.cur.get_mut(i) {
                        // Native's point-move handling.
                        // Its x branch adds |fdotp / 2| even for a negative numerator.
                        // FRESH-EVERYWHERE (owner decision 2026-11-07): a diagonal font_cache MovePoint uses
                        // the dwrite in-proc interpreter x branch `+|fdotp/2|` truncate — a FRESH
                        // (service-cache-miss) render, which is what any Windows machine draws for a glyph its
                        // apps have not drawn with GDI. The fntcache-service CompDiv variant appears only for
                        // GDI-pre-warmed glyphs (cache history, not portable — new-cache-key experiment
                        // 2026-11-07), so OpenD2D targets fresh. This subsumes the a1c819e SPVTL-DownToGrid
                        // special case (that case already truncated). The whole-pixel ROUND gate in
                        // `round_dist` stays (fresh RoundDownToGridSP, verified textx-glyph-576).
                        let dx = if fdotp == 0x4000 {
                            ((distance as i64 * fv.x as i64 + 0x2000) >> 14) as i32
                        } else if fdotp == fv.x {
                            distance
                        } else {
                            ((distance as i64 * fv.x as i64 + (fdotp / 2).abs() as i64) / fdotp as i64) as i32
                        };
                        p.x = p.x.wrapping_add(dx);
                    }
                    zone.touch(i, CoordAxis::X);
                }
                if fv.y != 0 {
                    if let Some(p) = zone.cur.get_mut(i) {
                        let dy = if fdotp == 0x4000 {
                            ((distance as i64 * fv.y as i64 + 0x2000) >> 14) as i32
                        } else {
                            math::mul_div(distance, fv.y, fdotp)
                        };
                        p.y = p.y.wrapping_add(dy);
                    }
                    zone.touch(i, CoordAxis::Y);
                }
            }
        }
    }

    /// Move a point in the zp2 zone by an explicit delta (SHP/SHC/SHZ/SHPIX helper).
    pub fn move_zp2(&mut self, i: usize, dx: i32, dy: i32, do_touch: bool) {
        let fv = self.free;
        let zone = &mut self.zones[self.zp2 as usize];
        if fv.x != 0 {
            if let Some(p) = zone.cur.get_mut(i) {
                p.x = p.x.wrapping_add(dx);
            }
            if do_touch {
                zone.touch(i, CoordAxis::X);
            }
        }
        if fv.y != 0 {
            if let Some(p) = zone.cur.get_mut(i) {
                p.y = p.y.wrapping_add(dy);
            }
            if do_touch {
                zone.touch(i, CoordAxis::Y);
            }
        }
    }

    /// Displacement of the reference point along the freedom vector (SHP/SHC/SHZ).
    pub fn point_displacement(&self, opcode: u8) -> (ZonePointer, usize, i32, i32) {
        let (zp, pi) = if opcode & 1 != 0 { (self.zp0, self.rp1) } else { (self.zp1, self.rp2) };
        let z = self.zone(zp);
        let distance = self.project(z.point(pi), z.original(pi));
        let dx = math::mul_div(distance, self.free.x, self.fdotp);
        let dy = math::mul_div(distance, self.free.y, self.fdotp);
        (zp, pi, dx, dy)
    }
}

// Native's projection (current and legacy).
fn project_diagonal(delta: Point, vector: Point) -> i32 {
    let x = (delta.x as i64 * vector.x as i64 + 0x2000) >> 14;
    let y = (delta.y as i64 * vector.y as i64 + 0x2000) >> 14;
    (x as i32).wrapping_add(y as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_projection_rounds_components_separately() {
        let mut gs = GraphicsState::new(Retained::default(), Zone::default(), Zone::default());
        gs.proj = Point::new(8192, 8192);
        gs.dual = gs.proj;
        gs.update_projection();
        assert_eq!(gs.project(Point::new(1, 1), Point::default()), 2);
        assert_eq!(gs.dual_project(Point::new(-1, -1), Point::default()), 0);
    }

    #[test]
    fn diagonal_negative_x_move_uses_native_division_bias() {
        let zone = Zone { cur: vec![Point::new(671, 1798)], flags: vec![0], ..Zone::default() };
        let mut gs = GraphicsState::new(Retained::default(), Zone::default(), zone);
        gs.free = Point::new(16384, 0);
        gs.proj = Point::new(15923, -3860);
        gs.update_projection();
        gs.move_point(ZonePointer::Glyph, 0, -13);
        assert_eq!(gs.zones[1].cur[0], Point::new(659, 1798));
    }

    /// FRESH-EVERYWHERE (owner decision 2026-11-07): a diagonal `font_cache` MovePoint truncates its x
    /// move exactly like the non-font_cache path (`diagonal_negative_x_move_uses_native_division_bias`
    /// above → 659), i.e. the dwrite in-proc itrp `+|fdotp/2|` — a fresh (service-cache-miss) render. The
    /// old fntcache CompDiv variant (→ 658) only appears for GDI-pre-warmed glyphs (cache history, not a
    /// portable per-glyph property — new-cache-key experiment), which OpenD2D does not target.
    #[test]
    fn font_cache_diagonal_x_move_is_fresh_truncate() {
        let zone = Zone { cur: vec![Point::new(671, 1798)], flags: vec![0], ..Zone::default() };
        let ret = Retained { font_cache: true, ..Default::default() };
        let mut gs = GraphicsState::new(ret, Zone::default(), zone);
        gs.free = Point::new(16384, 0);
        gs.proj = Point::new(15923, -3860);
        gs.update_projection();
        gs.move_point(ZonePointer::Glyph, 0, -13);
        assert_eq!(gs.zones[1].cur[0], Point::new(659, 1798));
    }

    /// IUP interpolates an untouched span with native's IUP single rounded integer muldiv
    /// `cur1 + (cur_range*(orus-orus1) + orus_range/2) / orus_range` (truncating division), not a
    /// two-step FixDiv/FixMul. One contour, endpoints touched & moved, two interior points free.
    #[test]
    fn iup_x_in_range_single_muldiv() {
        let mut z = Zone {
            unscaled: vec![Point::new(0, 0), Point::new(100, 0), Point::new(300, 0), Point::new(400, 0)],
            original: vec![Point::new(0, 0), Point::new(50, 0), Point::new(150, 0), Point::new(200, 0)],
            cur: vec![Point::new(0, 0), Point::new(50, 0), Point::new(150, 0), Point::new(260, 0)],
            flags: vec![TOUCHED_X, 0, 0, TOUCHED_X],
            contours: vec![3],
        };
        z.iup(CoordAxis::X);
        // cur1=0, cur_range=260, orus_range=400. p1: (260*100+200)/400 = 65; p2: (260*300+200)/400 = 195.
        assert_eq!(z.cur[1].x, 65);
        assert_eq!(z.cur[2].x, 195);
    }
}
