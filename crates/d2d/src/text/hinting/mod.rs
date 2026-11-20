//! DirectWrite's TrueType hinting interpreter, reproduced from its classic
//! reference rasterizer (the interpreter, scaler and scan-control routines).
//!
//! An [`Instance`] parses a font's `fpgm`/`prep`/`cvt`/`maxp`/`head` once for a given
//! device em size and rendering mode, runs the font and CVT programs, and retains the
//! resulting graphics state, CVT and functions. [`Instance::hint`] then grid-fits one
//! glyph: it scales the outline, appends the four phantom points, runs the glyph program
//! and returns the hinted outline plus the hinted advance.
//!
//! What this matches, and how it differs from a generic (FreeType-style) hinter:
//! * `GETINFO` reports **rasterizer version 42** (`0x2A`, read from the GETINFO handler) and
//!   its ClearType/subpixel result bits come from the configured [`Mode`], not hardcoded.
//! * There is **no** FreeType "backward compatibility" clamp on ordinary point moves — the classic
//!   interpreter moves points freely; ClearType behaviour is whatever the font's own GETINFO
//!   branches produce. The one subpixel-specific suppression native *does* apply is in the DELTA
//!   engine: under subpixel positioning an x-direction `DELTAP` is skipped, so a glyph's sub-pixel
//!   left bearing and stems stay unhinted in x (read from the delta engine; see [`engine`] and
//!   its `deltap_allowed`). See [`engine`]/[`state`].
//! * Fixed-point arithmetic, the SuperRound engine and the instruction semantics are the
//!   TrueType spec's, confirmed against the native routines.
//!
//! The smooth modes (grayscale / ClearType / natural) grid-fit y to the whole-pixel grid and hint x
//! on a **1/16-px subpixel grid**: the interpreter's x-direction `MDRP`/`MIRP`/`MDAP`/`ROUND` use the
//! subpixel round bank (round-to-grid/round-off, period 4 not 64) with the min distance
//! halved, so a vertical stem takes a CVT-controlled sub-pixel width instead of its raw scaled width
//! (DirectWrite's x-stem widening — the SVTCA/MIRP interpreter handling, gated by the
//! per-mode subpixel bit, [`state::GraphicsState::subpixel_round`]).
//! [`Mode::GdiClassic`]/[`Mode::GdiNatural`] expose the GDI-compatible grid-fitting the layout code
//! measures with.
//! Both GDI modes carry whole-pixel ("compatible") advances — [`Instance::hint`] snaps the advance
//! to the pixel grid, matching native `GetGdiCompatibleGlyphMetrics` for 99.7% (classic) / 95.4%
//! (natural) of the 15 textx fonts.
//!
//! Compatible-width proportional spacing (the distance/proportion/inter-align link steps +
//! per-point dependency graph + a horizontal-overscale → scale-down pass, native's MDRP/
//! MIRP gated by the global GDI-classic + subpixel flags) is **not** reproduced — it is a large
//! subsystem spanning the inner gridfit, the gridfit pass and the scaler on a native element memory
//! model this single-glyph interpreter does not share (log and stop). Reading the transform shows the subpixel flag is set for **GDI-classic only**
//! (rendering-flags byte = 3; GDI-natural = 1, smooth = 0x51/0x71), so that model accounts only for
//! the ~0.3% GDI-classic advance residual (e.g. Arial 'î' @ 8px: 3 vs native 2) — not GDI-natural
//! advances (those are `round(scaled)`) nor the smooth-mode stem widening (a separate x-stem control
//! plus `raster`'s scan-converter curve tie-break). Verification lives in `tests/text_hinting`.

mod engine;
mod math;
mod overscale;
mod state;

pub use engine::{enable_trace, take_trace, MoverRec};
pub(crate) use math::{fixdiv, fixmul};
pub use overscale::{enable_capture, execute_phase_control, take_capture};
pub use state::{LinkRec, Point};

use engine::{Definition, Engine};
use skrifa::raw::tables::glyf::{Anchor, CompositeGlyph, CompositeGlyphFlags, Glyf, Glyph, SimpleGlyph};
use skrifa::raw::tables::gvar::Gvar;
use skrifa::raw::tables::hmtx::Hmtx;
use skrifa::raw::tables::loca::Loca;
use skrifa::raw::types::{F2Dot14, GlyphId};
use skrifa::raw::{FontRef, TableProvider};
use state::{GraphicsState, GetInfoFlags, Retained, Zone};

/// Text anti-alias mode, selecting the GETINFO ClearType/subpixel flags the font sees.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Mode {
    /// Bi-level: classic full (x and y) grid-fitting; no subpixel/ClearType flags.
    Aliased,
    /// Grayscale antialiased: subpixel positioning + "grayscale ClearType".
    Grayscale,
    /// ClearType: subpixel positioning; horizontal RGB structure (optionally BGR).
    ClearType,
    /// GDI-compatible classic (`CLEARTYPE_GDI_CLASSIC`, measuring mode GDI_CLASSIC): full x+y
    /// grid-fitting with integer ("compatible") advance widths. Used for GDI-classic glyph metrics.
    GdiClassic,
    /// GDI-compatible natural (`CLEARTYPE_GDI_NATURAL`, measuring mode GDI_NATURAL): integer advance
    /// widths but natural (subpixel) x positioning. Used for GDI-natural glyph metrics.
    GdiNatural,
}

/// Native rasterizer implementation: system font-cache service or in-process DirectWrite.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Rasterizer {
    #[default]
    DirectWrite,
    FontCache,
}

/// How the glyph is being rendered. `symmetric` selects NATURAL_SYMMETRIC (the small-size
/// default); `bgr` sets the ClearType BGR pixel-geometry bit.
#[derive(Copy, Clone, Debug)]
pub struct Config {
    pub rasterizer: Rasterizer,
    pub mode: Mode,
    pub symmetric: bool,
    pub bgr: bool,
    pub is_rotated: bool,
    pub is_stretched: bool,
    /// The glyph→device matrix has no device-x part on the glyph's x axis (m11 = 0, a quarter turn). Native's
    /// transform setup then toggles the vertical-LCD pass flag for an x-subpixel, non-symmetric pass:
    /// ClearType's sub-pixel direction is the glyph's y (VM, Arial 'H' GDI_CLASSIC under [0,1,-1,0]: pass flags 0x85).
    /// Its other matrix rule, m12 ≠ 0 drops compatible widths, is the caller's GDI-classic → GDI-natural mode.
    pub axes_swapped: bool,
    /// How the glyph→device matrix rotates (`0` axis-aligned, `1` a pure quarter turn, `2` a general
    /// rotation or skew — an off-diagonal AND a diagonal both live). It selects a composite's
    /// component-offset rounding: kind 1 swaps the two axes' subpixel-bank tests, kind 2 rounds both on
    /// the subpixel bank, kind 3 (a synthetic oblique, otherwise axis-aligned) rounds both on the subpixel bank for
    /// a component with its own 2×2 only (see [`Instance::hint_composite`]).
    pub rotation_kind: u8,
    /// The device em along the glyph's y axis when it differs from the x axis (`device_em_px`): native's
    /// scaler scales x and y by their own axis scales, with MPPEM the larger.
    pub em_y: Option<f64>,
    /// Scaler version GETINFO reports. DirectWrite is 42 (read from the GETINFO handler);
    /// fonts branch on it, so it is exposed for cross-checking against another
    /// interpreter (e.g. 40 for FreeType) and for oracle tuning.
    pub version: i32,
    /// GDI **render** pass — selects the rendered-outline GETINFO rather than the metric (advance) pass.
    /// `false` (default) = the metric pass, so every existing caller (incl. the GDI advance measurer and
    /// the `gdi_advances` oracle) is unchanged.
    ///
    /// * [`Mode::GdiClassic`]: native hints the glyph twice — a metric
    ///   pass (global `g448==0`) for the whole-pixel compatible advance, and a render pass (`g448==0x83` =
    ///   subpixel-hinting|compatible|gray-CT) for the outline + the links the phase post-pass consumes.
    ///   The flag selects the render GETINFO (`storage[2]==262`) and x-subpixel hinting under compat width.
    /// * [`Mode::GdiNatural`]: the metric pass keeps compatible/whole-pixel advances, but the render
    ///   outline drops `compatible_width`/`subpixel_positioned` to reproduce native's GDI-natural outline
    ///   (GDI-classic vertical grid-fit + natural x-subpixel stems — see the `getinfo` branch).
    pub gdi_render: bool,
    /// The GDI-classic render pass targets a ClearType (RGB sub-pixel) texture rather than grayscale:
    /// its GETINFO drops `grayscale_cleartype` (storage[2]==6 instead of 262). Native's GDI-classic CT
    /// outline differs from its grayscale-CT one. Only meaningful with `gdi_render` on `GdiClassic`.
    pub cleartype: bool,
    /// GDI-natural, antialiased (CT/grayscale) only: snap the advance phantom's current x to the 1/16-px
    /// subpixel grid instead of the whole-pixel grid. Native keeps the GDI-natural advance phantom sub-pixel
    /// (Arial 'H'@13 phantom cur=600/orig=601, not whole 576; 'n' 464/463) so advance-relative MIRPs land
    /// right (§10l). Aliased GDI-natural keeps the whole-pixel phantom (17-text), hence the aa gate.
    pub subpixel_advance: bool,
    /// Whether to run the glyph program (grid-fit). `false` = scale the outline only, skipping the
    /// interpreter — native's `DWRITE_GRID_FIT_MODE_DISABLED`. DirectWrite's recommended rendering mode
    /// disables grid-fit for the smooth modes NATURAL/NATURAL_SYMMETRIC when the
    /// font's `gasp` range at the round-half-up device ppem lacks SYMMETRIC_GRIDFIT (0x04) — the toggle
    /// array `ReadRenderingModeRanges` builds from the 0x04 bit — and always for NAT_SYM_DOWNSAMPLED;
    /// GDI/aliased modes force it on. Default `true` (every existing caller grid-fits as before).
    pub grid_fit: bool,
    /// Synthetic bold on the GDI-classic render pass: the design strength (font units) the bold simulation adds to
    /// the advance, 0 for none. The x-overscale becomes the compatible advance plus the bold pixel over the scaled
    /// emboldened advance, at every size; the outline is emboldened after hinting (by the caller) only above a
    /// one-pixel boldX (VM contour captures, Trebuchet, Tahoma, Georgia, Leelawadee UI and Comic Sans, ppem 13 to 58).
    pub bold_fu: i32,
}

impl Default for Config {
    fn default() -> Self {
        Self { rasterizer: Rasterizer::DirectWrite, mode: Mode::Grayscale, symmetric: true, bgr: false, is_rotated: false, is_stretched: false, axes_swapped: false, rotation_kind: 0, em_y: None, version: 42, gdi_render: false, cleartype: false, subpixel_advance: false, grid_fit: true, bold_fu: 0 }
    }
}

impl Config {
    /// The GETINFO result flags the reference rasterizer configures for this mode.
    /// Version is always 42. The smooth modes advertise subpixel hinting + subpixel
    /// positioning (so ClearType-aware fonts skip x-direction grid-fitting); aliased
    /// advertises nothing, so fonts do full classic hinting.
    // ponytail: per-mode smooth flags are the standard ClearType model (confirmed bit
    // layout + version); pin the exact booleans against the oracle.
    fn getinfo(&self) -> GetInfoFlags {
        let mut g = GetInfoFlags::default();
        g.version = self.version;
        match self.mode {
            Mode::Aliased => {}
            Mode::Grayscale => {
                g.subpixel_hinting = true;
                g.subpixel_positioned = true;
                g.symmetric = self.symmetric;
                g.grayscale_cleartype = true;
            }
            Mode::ClearType => {
                g.subpixel_hinting = true;
                g.subpixel_positioned = true;
                g.symmetric = self.symmetric;
                g.bgr = self.bgr;
            }
            Mode::GdiClassic if self.gdi_render => {
                // Render pass: subpixel-x hinting under compatible width (native `g448` bits 0|1). The
                // grayscale render adds `grayscale_cleartype` (bit 7 → `g448==0x83`, hint-13); the
                // ClearType render adds **neither** `grayscale_cleartype` **nor** the sel-0x20 bit
                // (native CT glyph g448==0x3, g404==0 — GETINFO detour §10m).
                g.subpixel_hinting = true;
                g.compatible_width = true;
                if !self.cleartype {
                    g.grayscale_cleartype = true;
                }
            }
            Mode::GdiClassic => {
                // The metric pass (and aliased GDI) runs with pass flags 0: GETINFO reports neither the
                // grayscale bit (sel 0x20) nor compatible widths (sel 0x80) (VM dumps, Arial 'H' identity and
                // Consolas Italic 'h' aliased GDI_NATURAL; Consolas Italic's prep inhibits its glyph programs
                // when sel 0x20 answers). `compatible_width` stays as the pass's internal whole-pixel advance.
                g.compatible_width = true;
            }
            Mode::GdiNatural if self.gdi_render => {
                // GDI-natural **render** outline. The vertical grid-fit is identical to GDI-classic —
                // native's contour dump shows GDI-natural `y` == GDI-classic `y` byte-for-byte
                // (crossbar *and* apex curve points) for Arial 'A'/'V'@21 px — with x-subpixel hinting for
                // the natural (fractional) stems. The decisive bit is `subpixel_positioned`: the 'A' glyph
                // program branches on GETINFO[0x400] and, when it is set, shifts the crossbar 1 px down, so
                // the smooth/natural modes set it (their crossbar *is* 1 px lower) but GDI-natural must NOT.
                // The metric pass below keeps `compatible_width`/`subpixel_positioned` for the whole-pixel
                // GDI advance (`gdi_advances` oracle: GDI-natural advances are compatible/whole-pixel).
                // GETINFO (§10l, dwrite-dump cold font_file measuring=2 Arial 'H'@13): native's GDI-natural
                // glyph program is like GDI-classic's render branch but WITHOUT compatible_width — grayscale
                // = g448=0x81 (subpixel_hinting + grayscale_cleartype), ClearType = g448=0x1 (subpixel_hinting
                // only). NEITHER answers sel 0x20 (→0). Our old ClearType-enabled bit made sel
                // 0x20 return 0x1000, so the glyph program branched the wrong way (text-05 GDI-natural CT).
                g.subpixel_hinting = true;
                if !self.cleartype {
                    g.grayscale_cleartype = true;
                }
            }
            Mode::GdiNatural => {
                // Metric pass (advances, gdi_render off): native's GetGdiCompatibleGlyphMetrics(useGdiNatural)
                // runs the glyph program with **g448=0x1 — subpixel_hinting ONLY** (dwrite-dump metric capture
                // of 'b': sel 0x20→0 cleartype, 0x80→0 compat-width, 0x1000→0 gray-CT, 0x400→0 subpixel-pos).
                // No compatible_width ⇒ the advance stays SUB-PIXEL/natural (Arial 'b' 7.998, not whole 7),
                // which is the whole point of GDI-natural vs GDI-classic (§10l). Our old compatible_width +
                // subpixel_positioned + the sel-0x20 bit (g448=0x13) whole-rounded the advance → ~1px short
                // per glyph → run-pen drift (text-05 GDI-natural line).
                g.subpixel_hinting = true;
            }
        }
        // Native's transform setup: `(flags & 0x21) == 1 && t00 == 0` → flags ^= 4 (vertical LCD).
        if self.axes_swapped && g.subpixel_hinting && !g.symmetric {
            g.vertical_lcd = !g.vertical_lcd;
        }
        g
    }
}

/// A hinted glyph outline in device F26Dot6 coordinates (y-up), with the hinted advance.
#[derive(Clone, Debug, Default)]
pub struct HintedOutline {
    /// Grid-fitted points, phantom points removed, shifted so the pen origin is x=0.
    pub points: Vec<Point>,
    pub on_curve: Vec<bool>,
    /// Last point index of each contour.
    pub contour_ends: Vec<u16>,
    /// Hinted advance width in F26Dot6 device pixels.
    pub advance: i32,
    /// Final glyph-program dropout state, including SCANCTRL / SCANTYPE overrides.
    pub scan_control: (bool, i32),
    /// The outline in font units (composites assembled in font units), kept only when grid-fit is
    /// disabled: native's unscaled scaler path (MPPEM = upem) maps these through the whole transform.
    pub fu: Vec<Point>,
    /// GDI-classic x-overscale factor (16.16): the ratio of the
    /// whole-pixel "compatible" advance to the scaled design advance, `|FixDiv(advance,
    /// fixmul(design_advance_fu, scale))|` (native's contour setup). The phase-control
    /// post-pass squeezes x by `(overscale − 0x10000)·2·x`. `0x10000` (identity) outside GDI-classic
    /// or when the glyph has no advance/outline.
    pub overscale: i32,
}

thread_local! {
    /// The outermost composite's GDI-classic overscale while its components are hinted ([`RootOverscale`]).
    static ROOT_OVERSCALE: std::cell::Cell<Option<i32>> = const { std::cell::Cell::new(None) };
}

/// Holds the outermost composite's overscale for its components; a nested composite leaves it as set.
struct RootOverscale(Option<i32>);

impl RootOverscale {
    fn set(v: i32) -> Self {
        RootOverscale(ROOT_OVERSCALE.with(|c| c.replace(Some(c.get().unwrap_or(v)))))
    }
}

impl Drop for RootOverscale {
    fn drop(&mut self) {
        ROOT_OVERSCALE.with(|c| c.set(self.0));
    }
}

impl HintedOutline {
    /// Hinted advance in device pixels.
    pub fn advance_px(&self) -> f32 {
        self.advance as f32 / 64.0
    }

    /// The hinted outline as flattened, closed polygons (F26Dot6, y-up), one per contour, honouring
    /// the on-curve flags: quadratic segments are subdivided and implied on-curve midpoints inserted.
    /// Consumers that scan-convert the glyph (the raster coverage path) must build contours from this
    /// rather than reading [`points`](Self::points) directly — a quadratic's off-curve control point
    /// is not a polygon vertex, so using the raw points distorts every curved glyph.
    pub fn contours(&self) -> Vec<Vec<Point>> {
        let mut out = Vec::with_capacity(self.contour_ends.len());
        let mut start = 0usize;
        for &end in &self.contour_ends {
            let end = end as usize;
            if end >= start && end < self.points.len() {
                out.push(flatten_contour(&self.points[start..=end], &self.on_curve[start..=end]));
            }
            start = end + 1;
        }
        out
    }
}

/// Reconstruct one TrueType contour into a flattened closed polygon. `pts`/`on` are parallel (a
/// point and whether it is on-curve). Off-curve points are quadratic control points; two consecutive
/// off-curve points imply an on-curve midpoint between them.
fn flatten_contour(pts: &[Point], on: &[bool]) -> Vec<Point> {
    let n = pts.len();
    if n == 0 {
        return Vec::new();
    }
    let first_on = (0..n).find(|&i| on[i]);
    let s0 = first_on.unwrap_or(0);
    let start = match first_on {
        Some(i) => pts[i],
        None => midpoint(pts[0], pts[n - 1]),
    };
    let mut out = vec![start];
    let mut last_on = start;
    let mut i = 1;
    while i <= n {
        let (p, is_on) = (pts[(s0 + i) % n], on[(s0 + i) % n]);
        if is_on {
            out.push(p);
            last_on = p;
            i += 1;
        } else {
            let (np, np_on) = (pts[(s0 + i + 1) % n], on[(s0 + i + 1) % n]);
            let end = if np_on { np } else { midpoint(p, np) };
            flatten_quad(last_on, p, end, &mut out);
            last_on = end;
            i += if np_on { 2 } else { 1 };
        }
    }
    // The walk closes back onto the start point; drop that duplicate (the polygon is closed by the
    // caller wrapping the last vertex to the first).
    if out.len() > 1 && out[0] == out[out.len() - 1] {
        out.pop();
    }
    out
}

fn midpoint(a: Point, b: Point) -> Point {
    Point::new((a.x + b.x) / 2, (a.y + b.y) / 2)
}

/// Subdivide a quadratic (F26Dot6) until the control point's deviation from the chord is within
/// ~1/8 px, appending the on-curve points (not `p0`) to `out`.
fn flatten_quad(p0: Point, c: Point, p1: Point, out: &mut Vec<Point>) {
    let dx = (p0.x + p1.x) / 2 - c.x;
    let dy = (p0.y + p1.y) / 2 - c.y;
    if dx * dx + dy * dy <= 64 {
        out.push(p1);
        return;
    }
    let m0 = midpoint(p0, c);
    let m1 = midpoint(c, p1);
    let mm = midpoint(m0, m1);
    flatten_quad(p0, m0, mm, out);
    flatten_quad(mm, m1, p1, out);
}

/// A hinting instance: the font plus its prepared (post-`prep`) interpreter state for one
/// device em size and rendering mode.
pub struct Instance<'a> {
    metric: Option<Box<Instance<'a>>>,
    font: FontRef<'a>,
    /// The `gvar` table and the normalized variation coords, when this instance renders a
    /// variable-font instance: the glyph outline is `gvar`-interpolated in font units before scaling
    /// and hinting (so grid-fit `DrawText`/`DrawTextLayout` render the instance, as native does).
    /// Empty `coords` (the default/static instance) leaves the base outline untouched.
    gvar: Option<Gvar<'a>>,
    coords: Vec<F2Dot14>,
    glyf: Glyf<'a>,
    loca: Loca<'a>,
    hmtx: Hmtx<'a>,
    fpgm: &'a [u8],
    prep: &'a [u8],
    cvt: Vec<i32>,
    storage: Vec<i32>,
    functions: Vec<Definition>,
    instructions: Vec<Definition>,
    retained: Retained,
    max_stack: usize,
    max_twilight: usize,
    upem: i32,
    scale: i32,
    /// Per-axis FUnit scales (`scale` is the larger): outline points, phantoms and component offsets.
    x_scale: i32,
    y_scale: i32,
    /// Config's grid-fit gate: `false` skips the glyph program (scaled outline only).
    grid_fit: bool,
    /// Native skips prep and every glyph program at an integer ppem <= 1 (non-ClearType), see `new_instanced`.
    skip_programs: bool,
    /// [`Config::bold_fu`] on the GDI-classic render pass, else 0.
    bold_fu: i32,
    /// DWRITE_FONT_SIMULATIONS_OBLIQUE's shear for an unhinted glyph, as a 16.16 per-font-unit entry
    /// (0 = none). See [`Instance::set_oblique`].
    oblique: i32,
    device_em: f64,
    /// Grid-fit disabled: the instance at ppem = upem, native's unscaled pass, which runs prep and the glyph
    /// program in font units before the outline is scaled to the device.
    unscaled: Option<Box<Instance<'a>>>,
    /// The unscaled pass's per-font-unit 16.16 transform entries (x, y): FixDiv(em, upem << 16).
    unit_entry: [i32; 2],
}

impl<'a> Instance<'a> {
    /// The integer ppem (`MPPEM`) of this instance.
    pub fn ppem(&self) -> i32 {
        self.retained.ppem
    }

    /// Whether glyph programs run (grid-fit on).
    pub fn grid_fit(&self) -> bool {
        self.grid_fit
    }

    /// Shear unhinted glyphs for DWRITE_FONT_SIMULATIONS_OBLIQUE under an upright transform. Native's
    /// TrueType rasterizer adds `FixMul(t00, 22272)` to the 16.16 transform's t01 when it sets the transform; with grid-fit
    /// off the scaler keeps the outline in font units (MPPEM = upem) and maps it through that transform
    /// divided by upem (`FixDiv(t, upem<<16)`), one rounded product per term. So the shear entry is
    /// `FixDiv(ppem·22272, upem<<16)`, not the exact 22272/65536: 44/128 at ppem 4 (VM, Arial 'café' em
    /// 4–7 and 20: all 251 contour points). Hinted glyphs keep the exact post-hint shear (the caller's).
    pub fn set_oblique(&mut self) {
        let t00 = (self.device_em * 65536.0).round() as i32;
        self.oblique = math::fixdiv(math::fixmul(t00, 22272), self.upem << 16);
    }

    #[doc(hidden)]
    pub fn dbg_storage(&self) -> Vec<i32> {
        self.storage.clone()
    }

    /// The post-`prep` dropout-control state for the scan converter: `(enabled, scan_type)` where
    /// `enabled` is the resolved `SCANCTRL` flag and `scan_type` the `SCANTYPE` mode 0-6. The native
    /// scaler hands the rasterizer the scan
    /// type when dropout is enabled, else the "no dropout" mode — see [`raster::dropout_mode`].
    pub fn scan_control(&self) -> (bool, i32) {
        (self.retained.dropout_on(), self.retained.scan_type)
    }

    /// Build an instance for `device_em_px` (the em size in device pixels) and `config`,
    /// running the font and CVT programs. Returns `None` if the font lacks the TrueType
    /// scaler tables.
    pub fn new(font: &FontRef<'a>, device_em_px: f64, config: Config) -> Option<Self> {
        Self::new_instanced(font, device_em_px, config, &[])
    }

    /// As [`new`](Self::new), rendering the variable-font instance at the normalized `coords`
    /// (empty = the default/static instance, identical to `new`).
    pub fn new_instanced(font: &FontRef<'a>, device_em_px: f64, config: Config, coords: &[F2Dot14]) -> Option<Self> {
        let head = font.head().ok()?;
        let upem = head.units_per_em() as i32;
        if upem == 0 {
            return None;
        }
        let maxp = font.maxp().ok()?;
        let max_storage = maxp.max_storage().unwrap_or(0) as usize;
        let max_func = maxp.max_function_defs().unwrap_or(0) as usize;
        let max_instr = maxp.max_instruction_defs().unwrap_or(0) as usize;
        let max_stack = (maxp.max_stack_elements().unwrap_or(0) as usize).max(32) + 32;
        let max_twilight = maxp.max_twilight_points().unwrap_or(0) as usize;

        let fpgm = font.table_data(skrifa::raw::types::Tag::new(b"fpgm")).map(|d| d.as_bytes()).unwrap_or(&[]);
        let prep = font.table_data(skrifa::raw::types::Tag::new(b"prep")).map(|d| d.as_bytes()).unwrap_or(&[]);

        // Scale (font units -> F26Dot6, 16.16). The scale factor is one rounded divide over the
        // device em at its full 16.16 precision: round(em_px * 64 * 65536 / upem). Rounding
        // em_px*64 to an integer F26Dot6 first (the old fixdiv(round(em*64), upem)) quantized the em
        // to 1/64 px, which at fractional ppem (generic/anisotropic transforms) dropped up to half a
        // 1/64 unit and left right-edge points one unit low (30673 Leelawadee 'e', 30790). For
        // integer ppem — and any em already on the 1/64 grid — the two forms are identical, so only
        // off-grid ppem moves. Matches the native outline scaler's integer matrix-reduction setup.
        let em_y = config.em_y.unwrap_or(device_em_px);
        let scale_of = |em: f64| (em * 64.0 * 65536.0 / upem as f64).round() as i32;
        let (x_scale, y_scale) = (scale_of(device_em_px), scale_of(em_y));
        let scale = x_scale.max(y_scale);
        let ppem = (device_em_px.max(em_y).round() as i32).max(0);

        // Scale the CVT (font units -> F26Dot6).
        let cvt: Vec<i32> = match font.table_data(skrifa::raw::types::Tag::new(b"cvt ")) {
            Some(d) => d
                .as_bytes()
                .chunks_exact(2)
                .map(|c| math::scale(i16::from_be_bytes([c[0], c[1]]) as i32, scale))
                .collect(),
            None => Vec::new(),
        };

        let mut retained = Retained::default();
        retained.scale = scale;
        retained.x_scale = x_scale;
        retained.y_scale = y_scale;
        // Native's scaler keeps the smaller axis' ratio to the larger.
        let (ex, ey) = ((device_em_px * 65536.0).round() as i32, (em_y * 65536.0).round() as i32);
        if ex < ey {
            retained.x_ratio = math::fixdiv(ex, ey);
        } else if ey < ex {
            retained.y_ratio = math::fixdiv(ey, ex);
        }
        retained.font_cache = config.rasterizer == Rasterizer::FontCache;
        retained.ppem = ppem;
        retained.is_rotated = config.is_rotated;
        retained.is_stretched = config.is_stretched;
        retained.rotation_kind = config.rotation_kind;
        retained.getinfo = config.getinfo();
        retained.gdi_render = config.gdi_render && config.mode == Mode::GdiClassic;
        // GDI-natural antialiased: subpixel-round the advance phantom's current x (native keeps it sub-pixel;
        // §10l). Gated on aa by the caller (aliased GDI-natural keeps whole-pixel).
        retained.compatible_advance = config.subpixel_advance;

        let twilight = make_twilight(max_twilight);
        let gs = GraphicsState::new(retained, twilight, Zone::default());
        let mut engine = Engine::new(
            gs,
            cvt,
            vec![0; max_storage],
            vec![Definition::default(); max_func],
            vec![Definition::default(); max_instr],
            max_stack,
            fpgm,
            prep,
        );
        // Run fpgm then prep; ignore interpreter errors (the glyph still hints with
        // whatever state was established, matching a lenient scaler).
        if !fpgm.is_empty() {
            let _ = engine.run_program(0, &[]);
        }
        // Native's interpreter runs neither prep nor a glyph program while the integer ppem
        // (`(max(xscale, yscale) + 0x8000) >> 16`) is <= 1 (fpgm still runs). With no prep, a 1-ppem glyph
        // keeps the pre-program's seed SCANCTRL word 0x120 and scan type 0, so it
        // scans with dropout mode 0 (VM: Times/Arial/Consolas/Segoe UI at em 0.5-1.49, grid-fit on,
        // NATURAL/NAT_SYM → only fpgm executes, fsc a4 = 0). The disabled path never meets the gate.
        let tiny = config.grid_fit && ppem <= 1;
        let mut unscaled = None;
        if !config.grid_fit {
            // Grid-fit disabled: native runs prep in its scaler's unscaled path (x = y = upem;
            // VM: MPPEM = 2048 for Segoe UI at em 0.5) and only its SCANCTRL word and SCANTYPE reach the
            // fill, where the glyph setup evaluates the word at the device ppem.
            if let Some(up) = Self::new_instanced(font, upem as f64, Config { grid_fit: true, em_y: None, ..config }, coords) {
                engine.gs.ret.scan_word = up.retained.scan_word;
                engine.gs.ret.scan_type = up.retained.scan_type;
                unscaled = Some(Box::new(up));
            }
        } else if !prep.is_empty() && !tiny {
            let _ = engine.run_program(1, &[]);
        }

        let metric = if config.mode == Mode::GdiClassic && config.gdi_render {
            Some(Box::new(Self::new_instanced(font, device_em_px, Config { gdi_render: false, cleartype: false, ..config }, coords)?))
        } else {
            None
        };
        Some(Self {
            metric,
            font: font.clone(),
            gvar: font.gvar().ok(),
            coords: coords.to_vec(),
            glyf: font.glyf().ok()?,
            loca: font.loca(None).ok()?,
            hmtx: font.hmtx().ok()?,
            fpgm,
            prep,
            cvt: engine.cvt,
            storage: engine.storage,
            functions: engine.functions,
            instructions: engine.instructions,
            retained: engine.gs.ret,
            max_stack,
            max_twilight,
            upem,
            scale,
            x_scale,
            y_scale,
            grid_fit: config.grid_fit,
            skip_programs: tiny,
            bold_fu: if config.gdi_render && config.mode == Mode::GdiClassic { config.bold_fu } else { 0 },
            oblique: 0,
            device_em: device_em_px,
            unscaled,
            unit_entry: [math::fixdiv(ex, upem << 16), math::fixdiv(ey, upem << 16)],
        })
    }

    /// Grid-fit `glyph_id` and return its hinted outline. On any interpreter error the
    /// scaled outline is returned instead (a safe fallback, like a lenient scaler).
    pub fn hint(&self, glyph_id: u16) -> Option<HintedOutline> {
        let h = self.hint_rec(GlyphId::new(glyph_id as u32), 0)?;
        // Advance and lsb come from the phantom points; shift so the pen origin is x=0.
        let lsb = h.phantom[0].x;
        let mut advance = h.phantom[1].x.wrapping_sub(lsb);
        // GDI (compatible-width) modes carry whole-pixel advances (module docs); a no-op for
        // GDI-classic, which already grid-fits the advance phantom. GDI-natural's sub-pixel metric advance
        // is rounded by its consumer (`gdi`), where the transform decides it (native rounds it only for an axis-aligned transform).
        if self.retained.getinfo.compatible_width {
            advance = math::round(advance);
        }
        // GDI-classic overscale factor: |FixDiv(compatible advance, scaled design
        // advance)|. Only GDI-classic (compatible width + whole-pixel positioning) uses it; identity
        // otherwise, or when the glyph has no advance/outline (native's no-outline guard).
        let design_adv = self.hmtx.advance(GlyphId::new(glyph_id as u32)).unwrap_or(0) as i32;
        // GDI-classic (compatible width, whole-pixel positioning) is the only mode with the overscale
        // phase-control subsystem; GDI-natural and the smooth modes leave it identity.
        let g = &self.retained.getinfo;
        let overscale = if g.compatible_width && !g.subpixel_positioned && design_adv != 0 && advance != 0 {
            math::fixdiv(advance, math::fixmul(design_adv, self.x_scale)).abs()
        } else {
            0x10000
        };
        // Native's grid-fit (the GDI-classic render pass, subpixel) translates the outline by
        // `(pen - lsb + 2) & ~3`: the shift is rounded half-up to the 1/16-px grid, so an lsb phantom the
        // phase squeeze moved off that grid (Arial Bold 'A' @68: -14) leaves the outline 2/64 px off the pen.
        // ponytail: native rounds the shift in every grid-fit mode (whole px when not x-subpixel); the other
        // modes pre-round the phantom to that grid and rarely move it, so only the render pass does it here.
        let shift = if self.retained.gdi_render && self.grid_fit { (2i32.wrapping_sub(lsb)) & !3 } else { lsb.wrapping_neg() };
        // Unhinted, native's unscaled pass first runs the glyph program at ppem = upem (in font units), then the
        // scaler maps that outline (composites assembled in font units) through the transform's per-font-unit
        // 16.16 entries, each product rounded on its own half away from zero, not half toward +inf like the
        // hinter's own scale (VM r25 30181, Impact em 34 disabled: fu y -24 scales to -25.5 and lands at -26, and
        // the program's 1/4-unit nudge of 'a' point 13 lifts it from 125 to 126).
        let points = if !self.grid_fit && h.fu.len() == h.points.len() {
            let up = self.unscaled.as_ref().and_then(|u| u.hint_rec(GlyphId::new(glyph_id as u32), 0)).map(|u| u.points).filter(|p| p.len() == h.fu.len());
            let src = up.unwrap_or_else(|| h.fu.iter().map(|p| Point::new(p.x.wrapping_mul(64), p.y.wrapping_mul(64))).collect());
            let [ex, ey] = self.unit_entry;
            src.iter().map(|p| Point::new(math::fixmul(p.x, ex).wrapping_add(math::fixmul(p.y, self.oblique)).wrapping_add(shift), math::fixmul(p.y, ey))).collect()
        } else {
            h.points.iter().map(|p| Point::new(p.x.wrapping_add(shift), p.y)).collect()
        };
        Some(HintedOutline { points, on_curve: h.on_curve, contour_ends: h.contours, advance, overscale, scan_control: h.scan_control, fu: h.fu })
    }

    /// Hint one glyph (recursively hinting components of a composite), returning its
    /// hinted contour points and the four hinted phantom points.
    fn hint_rec(&self, gid: GlyphId, depth: u8) -> Option<Hinted> {
        if depth > 8 {
            return None;
        }
        let phantom = self.phantom_funits(gid);
        let compatible_advance = self.metric.as_ref().and_then(|m| engine::without_trace(|| m.hint(gid.to_u32() as u16))).map(|h| h.advance);
        match self.loca.get_glyf(gid, &self.glyf).ok()? {
            None => Some(Hinted {
                points: Vec::new(),
                on_curve: Vec::new(),
                contours: Vec::new(),
                // A variable face's outline-less glyph takes its phantoms at the instance (VM r25 30460: Bahnschrift's
                // condensed bold space is 6 px at ppem 28, not the default instance's 8).
                phantom: self.scale_phantoms(self.gvar.as_ref().and_then(|_| {
                    let ph = phantom.map(|p| [p.x, p.y]);
                    super::variations::instanced_phantoms_fu(&self.font, gid, &self.coords, ph).map(|v| v.map(|p| Point::new(p[0], p[1])))
                }).unwrap_or(phantom)),
                scan_control: self.scan_control(),
                fu: Vec::new(),
            }),
            Some(Glyph::Simple(s)) => Some(self.hint_simple(gid, &s, phantom, compatible_advance)),
            Some(Glyph::Composite(c)) => self.hint_composite(gid, &c, phantom, depth, compatible_advance),
        }
    }

    fn hint_simple(&self, gid: GlyphId, s: &SimpleGlyph<'a>, phantom_fu: [Point; 4], compatible_advance: Option<i32>) -> Hinted {
        let n = s.num_points();
        // Variable-font instancing: `gvar`-interpolate the contour + phantom points in font units
        // before scaling/hinting (native applies the variation deltas to the master outline, then the
        // scaler/interpreter runs on the instanced points). `None` = default/static instance → base.
        let inst = self.gvar.as_ref().and_then(|_| {
            let ph = [[phantom_fu[0].x, phantom_fu[0].y], [phantom_fu[1].x, phantom_fu[1].y], [phantom_fu[2].x, phantom_fu[2].y], [phantom_fu[3].x, phantom_fu[3].y]];
            super::variations::instanced_points_fu(&self.font, gid, &self.coords, s, ph)
        });
        let mut scaled = Vec::with_capacity(n + 4);
        let mut unscaled = Vec::with_capacity(n + 4);
        let mut on_curve = Vec::with_capacity(n + 4);
        for (i, p) in s.points().enumerate() {
            let (x, y) = inst.as_ref().map_or((p.x as i32, p.y as i32), |v| (v[i][0], v[i][1]));
            unscaled.push(Point::new(x, y));
            scaled.push(Point::new(math::scale(x, self.x_scale), math::scale(y, self.y_scale)));
            on_curve.push(p.on_curve);
        }
        for (j, p) in phantom_fu.iter().enumerate() {
            let pt = inst.as_ref().map_or(*p, |v| Point::new(v[n + j][0], v[n + j][1]));
            unscaled.push(pt);
            scaled.push(Point::new(math::scale(pt.x, self.x_scale), math::scale(pt.y, self.y_scale)));
            on_curve.push(true);
        }
        let contours: Vec<u16> = s.end_pts_of_contours().iter().map(|e| e.get()).collect();
        // Grid-fit disabled (gasp has no SYMMETRIC_GRIDFIT at this ppem): native skips the glyph program
        // and rasterizes the scaled outline directly, so return the scaled points unhinted.
        let fu = if self.grid_fit { Vec::new() } else { unscaled[..n].to_vec() };
        let (hinted, scan_control) = if self.grid_fit {
            self.execute(scaled, unscaled, &on_curve, contours.clone(), s.instructions(), false, compatible_advance)
        } else {
            (scaled, self.scan_control())
        };
        Hinted {
            phantom: [hinted[n], hinted[n + 1], hinted[n + 2], hinted[n + 3]],
            points: hinted[..n].to_vec(),
            on_curve: on_curve[..n].to_vec(),
            contours,
            scan_control,
            fu,
        }
    }

    fn hint_composite(&self, gid: GlyphId, c: &CompositeGlyph<'a>, mut phantom_fu: [Point; 4], depth: u8, compatible_advance: Option<i32>) -> Option<Hinted> {
        // Each component is hinted on its own (its glyph program runs), then transformed
        // by the component's 2x2 matrix and offset and merged; finally the composite's
        // own instructions (if any) hint the assembly.
        // Variable-font instancing: a composite's `gvar` varies its component offsets and phantom
        // points (no IUP — one delta each), applied in font units before scaling, like the simple
        // path. `None` for a static font / the default instance (then this is a no-op).
        let ncomp = c.components().count();
        let cdelta = self.gvar.as_ref().and_then(|_| super::variations::instanced_composite_deltas_fu(&self.font, gid, &self.coords, ncomp + 4));
        if let Some(cd) = &cdelta {
            phantom_fu = [
                Point::new(phantom_fu[0].x + cd[ncomp][0], phantom_fu[0].y + cd[ncomp][1]),
                Point::new(phantom_fu[1].x + cd[ncomp + 1][0], phantom_fu[1].y + cd[ncomp + 1][1]),
                Point::new(phantom_fu[2].x + cd[ncomp + 2][0], phantom_fu[2].y + cd[ncomp + 2][1]),
                Point::new(phantom_fu[3].x + cd[ncomp + 3][0], phantom_fu[3].y + cd[ncomp + 3][1]),
            ];
        }
        // The GDI-classic render pass squeezes every component by the overscale of the glyph drawn, the outermost
        // composite, not by each component's own advance (VM r25 30250: Segoe UI U+0628's dot component, advance 513,
        // keeps its 1/16-px x where its own 384/385 squeeze left it 1/64 px left).
        // The design advance is scaled once, not as the difference of two scaled phantoms: a composite whose lsb is not
        // its base glyph's (Times New Roman Bold U+00C5, lsb 17 over A's 16) takes the base's factor.
        let _root = self.retained.gdi_render.then(|| {
            let adv = phantom_fu[1].x.wrapping_sub(phantom_fu[0].x);
            let sadv = math::scale(adv, self.x_scale);
            let compat = compatible_advance.unwrap_or_else(|| math::round(sadv));
            let (compat, sadv) = if self.bold_fu != 0 { (compat + 64, math::scale(adv + self.bold_fu, self.x_scale)) } else { (compat, sadv) };
            RootOverscale::set(if sadv != 0 { math::fixdiv(compat, sadv).abs() } else { 0x10000 })
        });
        let mut points: Vec<Point> = Vec::new();
        let mut fu: Vec<Point> = Vec::new();
        let mut on_curve: Vec<bool> = Vec::new();
        let mut contours: Vec<u16> = Vec::new();
        // USE_MY_METRICS: the composite takes its advance/lsb (phantom points) from this component's
        // already-hinted phantoms, not from re-scaling the composite's own hmtx — matching native
        // (Arial 'á' = 'a's hinted advance 17, not the design-rounded 18). Last such component wins.
        let mut my_metrics: Option<[Point; 4]> = None;
        let mut merged_scan_type: Option<i32> = None;
        // The SMOOTH x-subpixel regime (NATURAL / NATURAL_SYMMETRIC grayscale/ClearType): the composite
        // component x-offset grid-fits on the 1/16-px bank (`ROUND_XY_TO_GRID`), not the whole-pixel grid.
        // Oracle: Arial 'ñ' tilde offset 255fu@20px → 159 F26Dot6 → round-to-grid-SP 160 (2.5px) not whole
        // 128; 'é' acute 243fu@16 → 122 → 124. Native's component-offset rounding keys this on the GETINFO
        // subpixel pass bit alone, so the GDI-classic *render* pass (0x83) rounds on the 1/16-px bank
        // too (Consolas 'έ' tonos 41fu@12 → 15 → 16) and only its metric pass (0) is whole-pixel.
        // Vertical-LCD ClearType is the same with the axes swapped: y on the 1/16-px bank, x whole-pixel.
        let gi = &self.retained.getinfo;
        // Which glyph axis grid-fits a component offset on the 1/16-px subpixel bank, which on the whole
        // grid. The base (axis-aligned) rule keys x on the subpixel bank when the pass is not vertical-LCD
        // and y the reverse, with the compatible-width / gdi_render terms the untransformed path matches
        // native with. Native keys the choice on how the glyph→device matrix rotates: under a pure quarter
        // turn it SWAPS the two axes' tests (the vertical-LCD bit is already toggled by the transform setup,
        // and native's own swap applies on top — so the net is the untoggled test, read here from the
        // already-toggled bit as its negation); under a general rotation or skew it rounds BOTH axes on the
        // subpixel bank. Without the turn swap a turned composite accent lands up to 1/4 px off on one axis
        // (Lucida Console 'é'); without the general-rotation both-axes a sheared or turned-and-obliqued one
        // does (r25 30514 30900 and the Arabic/CJK composite marks).
        let sub = gi.subpixel_hinting;
        let (xv, yv) = if self.retained.rotation_kind == 1 { (gi.vertical_lcd, !gi.vertical_lcd) } else { (!gi.vertical_lcd, gi.vertical_lcd) };
        let p2 = self.retained.rotation_kind == 2 && sub;
        let x_subpixel = (sub && xv && (!gi.compatible_width || self.retained.gdi_render)) || p2;
        let y_subpixel = (sub && yv && !gi.compatible_width) || p2;
        for (ci, comp) in c.components().enumerate() {
            let child = self.hint_rec(GlyphId::from(comp.glyph), depth + 1)?;
            // Native's composite-glyph merge combines
            // resolved component dropout modes, including disabled components (mode 2).
            let mode = if child.scan_control.0 { child.scan_control.1 } else { 2 };
            merged_scan_type = Some(merged_scan_type.map_or(mode, |previous| previous & ((mode & 3) | 4)));
            if comp.flags.contains(CompositeGlyphFlags::USE_MY_METRICS) {
                my_metrics = Some(child.phantom);
            }
            let base = points.len();
            // gvar component-offset delta (font units), added before scaling — native order.
            let (odx, ody) = cdelta.as_ref().map_or((0, 0), |cd| (cd[ci][0], cd[ci][1]));
            let (fdx, fdy) = match comp.anchor {
                Anchor::Offset { x, y } => ((x as i32).wrapping_add(odx), (y as i32).wrapping_add(ody)),
                Anchor::Point { .. } => (0, 0),
            };
            let (dx, dy) = match comp.anchor {
                Anchor::Offset { x, y } => {
                    let mut dx = math::fixmul((x as i32).wrapping_add(odx), self.x_scale);
                    let mut dy = math::fixmul((y as i32).wrapping_add(ody), self.y_scale);
                    // Under a synthetic oblique (`rotation_kind` 3) every component sits in a skewed frame, so its offset rounds
                    // on the subpixel bank on both axes as under a general matrix (VM: Segoe UI Emoji U+1F642's turned eye
                    // components, oblique at em 14.6, 12/64 px higher than whole-pixel y; Tahoma and Verdana U+00C9 and U+00D1
                    // oblique in GDI classic, the accent's y offset 175 at ppem 15 on 176, not 192).
                    let skewed = self.retained.rotation_kind == 3 && sub;
                    let (x_subpixel, y_subpixel) = (x_subpixel || skewed, y_subpixel || skewed);
                    if comp.flags.contains(CompositeGlyphFlags::ROUND_XY_TO_GRID) {
                        // Native's component-offset calc rounds each
                        // component offset with a DIRECT half-toward-+∞ bit mask (NOT the interpreter's
                        // sign-symmetric round_sp): the subpixel axis `(off + 2) & ~3` (1/16 px), the other
                        // `(off + 32) & ~63` (whole px). The mask rounds a negative tie toward +∞, so Lucida
                        // 'ñ' tilde argx −3 → fixmul −2 → (−2+2)&~3 = 0 (tilde stays on its design anchor, not
                        // shifted −4 as round_sp gave); Arial 'ñ' tilde 255fu → 159 → (159+2)&~3 = 160 still.
                        dx = if x_subpixel { (dx + 2) & !3 } else { (dx + 32) & !63 };
                        dy = if y_subpixel { (dy + 2) & !3 } else { (dy + 32) & !63 };
                    }
                    (dx, dy)
                }
                // ponytail: point-anchored components are rare in Latin fonts; treated
                // as no offset. Add matched-point placement if an oracle glyph needs it.
                Anchor::Point { .. } => (0, 0),
            };
            let t = &comp.transform;
            let (xx, yx, xy, yy) =
                (t.xx.to_bits() as i32, t.yx.to_bits() as i32, t.xy.to_bits() as i32, t.yy.to_bits() as i32);
            let identity = xx == 0x4000 && yy == 0x4000 && yx == 0 && xy == 0;
            // A transformed component with no glyph program of its own maps its font-unit points through the
            // component's 2×2 folded into the scale, one rounding, rather than scaling then transforming the
            // scaled points (VM: Segoe UI Emoji U+1F642's turned eye components at em 11.8 to 14.6, every point
            // exact where the two roundings left the eye 1/64 px off).
            let raw = (!identity && self.gvar.is_none())
                .then(|| match self.loca.get_glyf(GlyphId::from(comp.glyph), &self.glyf).ok().flatten() {
                    Some(Glyph::Simple(cs)) if cs.instructions().is_empty() && cs.num_points() == child.points.len() => Some(cs.points().map(|q| (q.x as i32, q.y as i32)).collect::<Vec<_>>()),
                    _ => None,
                })
                .flatten();
            if let Some(raw) = &raw {
                let (cxx, cxy, cyx, cyy) = (math::mul14(self.x_scale, xx), math::mul14(self.x_scale, xy), math::mul14(self.y_scale, yx), math::mul14(self.y_scale, yy));
                for &(fx, fy) in raw {
                    points.push(Point::new(math::scale(fx, cxx).wrapping_add(math::scale(fy, cxy)).wrapping_add(dx), math::scale(fx, cyx).wrapping_add(math::scale(fy, cyy)).wrapping_add(dy)));
                }
            }
            for p in child.points.iter().filter(|_| raw.is_none()) {
                let np = if identity {
                    Point::new(p.x.wrapping_add(dx), p.y.wrapping_add(dy))
                } else {
                    let x = math::mul14(p.x, xx).wrapping_add(math::mul14(p.y, xy));
                    let y = math::mul14(p.x, yx).wrapping_add(math::mul14(p.y, yy));
                    Point::new(x.wrapping_add(dx), y.wrapping_add(dy))
                };
                points.push(np);
            }
            // Unhinted: the same assembly in font units (a font-unit offset is already on any rounding grid).
            for p in &child.fu {
                let np = if identity {
                    Point::new(p.x, p.y)
                } else {
                    Point::new(math::mul14(p.x, xx).wrapping_add(math::mul14(p.y, xy)), math::mul14(p.x, yx).wrapping_add(math::mul14(p.y, yy)))
                };
                fu.push(Point::new(np.x.wrapping_add(fdx), np.y.wrapping_add(fdy)));
            }
            on_curve.extend_from_slice(&child.on_curve);
            for e in &child.contours {
                contours.push(e + base as u16);
            }
        }
        let n = points.len();
        // The assembled (already-hinted) points plus the composite's phantom points. With
        // USE_MY_METRICS the phantoms are the component's hinted (device-space) ones; otherwise they
        // are the composite's own hmtx phantoms scaled to device units.
        let mut scaled = points;
        let mut on_curve_all = on_curve;
        let phantom_dev = my_metrics
            .unwrap_or_else(|| phantom_fu.map(|p| Point::new(math::fixmul(p.x, self.x_scale), math::fixmul(p.y, self.y_scale))));
        for p in phantom_dev {
            scaled.push(p);
            on_curve_all.push(true);
        }
        // For composites the "unscaled" buffer is the assembled scaled points (identity
        // scale), so relative distances are measured in pixels directly.
        let unscaled = scaled.clone();
        // Grid-fit disabled: components were already scaled unhinted (hint_simple skipped the program);
        // skip the composite's own assembly program too and keep the assembled scaled outline.
        let (hinted, scan_control) = if self.grid_fit {
            self.execute(scaled, unscaled, &on_curve_all, contours.clone(), c.instructions().unwrap_or(&[]), true, compatible_advance)
        } else {
            (scaled, self.scan_control())
        };
        let scan_control = if c.instructions().unwrap_or(&[]).is_empty() {
            merged_scan_type.map_or(scan_control, |mode| (mode != 2, mode))
        } else {
            scan_control
        };
        Some(Hinted {
            phantom: [hinted[n], hinted[n + 1], hinted[n + 2], hinted[n + 3]],
            points: hinted[..n].to_vec(),
            on_curve: on_curve_all[..n].to_vec(),
            contours,
            scan_control,
            fu: if self.grid_fit || fu.len() != n { Vec::new() } else { fu },
        })
    }

    /// Build the glyph zone from `scaled` (contour points followed by the four phantom
    /// points), run the glyph program and return the hinted current points. The phantom
    /// points' current positions are rounded to the grid first; their originals are not.
    fn execute(&self, scaled: Vec<Point>, unscaled: Vec<Point>, on_curve: &[bool], contours: Vec<u16>, instructions: &'a [u8], is_composite: bool, compatible_advance: Option<i32>) -> (Vec<Point>, (bool, i32)) {
        let n = scaled.len().saturating_sub(4);
        let mut flags = vec![0u8; scaled.len()];
        for (i, &oc) in on_curve.iter().enumerate() {
            if oc {
                flags[i] |= state::ON_CURVE;
            }
        }
        // Phantom-point grid-fit: y (vertical advance/metrics) always snaps to the whole-pixel grid.
        // x (the lsb/advance points) snaps to the whole grid under non-subpixel positioning (aliased /
        // GDI-classic); under the smooth x-subpixel regime it snaps to the **1/16-px subpixel grid**,
        // because the interpreter's x-direction MDRP/MIRP position stems *relative to these phantoms*
        // and native hands them already rounded to that grid (confirmed by the native scan-converter
        // oracle: Arial 'o'@16px right edge MIRPs off the advance phantom, which native rounds 570→572).
        // GDI-natural also snaps the phantom x to the **1/16-px subpixel grid**, not the whole grid: native
        // keeps the advance phantom sub-pixel (Arial 'H'@13 GDI-natural phantom cur=600, orig=601 = scaled
        // 600.84 round_sp/round, NOT whole 576; 'n' cur=464 orig=463). Whole-rounding it (the old GDI-natural
        // behaviour) mislanded every advance-relative MIRP stem — §10l. The GDI-classic *render* pass
        // (hint-13) likewise snaps the phantom x to the 1/16-px grid (pt23 off phantom 26).
        let gi = &self.retained.getinfo;
        // GDI-natural's metric pass (`compatible_advance`) keeps the sub-pixel phantom only while the glyph
        // program runs; with hinting inhibited (INSTCTRL bit 0, e.g. small sizes) its phantom x stays as scaled
        // (VM r25 30008: Segoe UI at ppem 4 under a shear advances by its 26.6 design widths, 'l' 62/64 px).
        let inhibited = (self.retained.instruct_control & 1) != 0;
        let keep_x = inhibited && self.retained.compatible_advance && !self.retained.gdi_render;
        let round_x = !gi.subpixel_positioned && !self.retained.gdi_render && (!self.retained.compatible_advance || inhibited);
        let x_subpixel = gi.subpixel_hinting && !gi.vertical_lcd && (!gi.compatible_width || self.retained.gdi_render);
        // A simple glyph's whole outline, original points included, shifts so its lsb phantom lands on the
        // grid (native's simple-glyph grid-fit, `(v + 2) & ~3` on the subpixel bank, else `(v + 32) & ~63`).
        // The other phantoms carry the same shift, so an hmtx lsb that differs from the glyph's xMin moves the
        // ink instead of its phantom (Times New Roman Italic 'j'). Inhibited hinting skips it.
        let shift_lsb = !is_composite && !inhibited && n < scaled.len();
        let mut scaled = scaled;
        if shift_lsb {
            let v = scaled[n].x;
            let d = if !round_x && x_subpixel { (v.wrapping_add(2) & !3).wrapping_sub(v) } else { math::round(v).wrapping_sub(v) };
            for p in scaled.iter_mut() {
                p.x = p.x.wrapping_add(d);
            }
        }
        let mut cur = scaled.clone();
        // Native rounds the side-bearing phantom x on the 1/16-px grid only for an
        // x-subpixel pass (pass flags `& 5 == 1`); otherwise, vertical LCD included, on whole pixels.
        for (k, p) in cur[n..].iter_mut().enumerate() {
            if (!shift_lsb || k < 2) && !keep_x {
                p.x = if !round_x && x_subpixel { math::RoundState::default().round_sp(p.x) } else { math::round(p.x) };
            }
            p.y = math::round(p.y);
        }
        if (self.retained.instruct_control & 1) != 0 || instructions.is_empty() || self.skip_programs {
            return (cur, self.scan_control());
        }
        // Pure smooth modes (grayscale / ClearType / natural) grid-fit y to the whole-pixel grid and
        // keep x hinted on the 1/16-px subpixel grid: the interpreter runs x-direction MDRP/MIRP with
        // the subpixel round bank (see `GraphicsState::subpixel_round`), so vertical stems take a
        // sub-pixel width controlled by the CVT rather than the raw scaled width. The non-subpixel
        // modes, and x under vertical LCD, keep full x grid-fit.
        // GDI-classic render pass (hint-13): the x-overscale factor native stores for the render pass
        // = |FixDiv(compat advance, scaled design advance)|. The compatible advance comes from the
        // metric hinting pass, not rounding the design advance (Arial f@13: 192, not 256).
        // Native's cold MSIRP/IUP traces retain this metric-pass factor.
        // Only the render pass applies it; other modes leave it identity.
        let overscale = if let Some(root) = ROOT_OVERSCALE.with(|c| c.get()).filter(|_| self.retained.gdi_render && n + 1 < scaled.len()) {
            root
        } else if self.retained.gdi_render && n + 1 < scaled.len() {
            let sadv = scaled[n + 1].x.wrapping_sub(scaled[n].x);
            let compat = compatible_advance.unwrap_or_else(|| math::round(sadv));
            // Synthetic bold: the bold pixel on the compatible advance, over the advance scaled with the bold
            // strength added in font units.
            let (compat, sadv) = if self.bold_fu != 0 {
                (compat + 64, math::scale(unscaled[n + 1].x + self.bold_fu, self.x_scale).wrapping_sub(scaled[n].x))
            } else {
                (compat, sadv)
            };
            if sadv != 0 { math::fixdiv(compat, sadv).abs() } else { 0x10000 }
        } else {
            0x10000
        };
        let glyph_zone = Zone { unscaled, original: scaled, cur: cur.clone(), flags, contours };
        let mut gs = GraphicsState::new(self.retained, make_twilight(self.max_twilight), glyph_zone);
        gs.is_composite = is_composite;
        gs.is_glyph = true; // the 1/16-px subpixel round bank is the glyph program's, not prep/fpgm's
        gs.overscale = overscale;
        // GDI-classic builds the x-overscale link memory during the glyph program (inert in every
        // other mode); the phase-control pass (during `IUP`, render pass) consumes it.
        if gs.gdi_overscale {
            gs.links = vec![state::LinkRec::default(); gs.zones[1].len()];
            gs.fill_bits = overscale::fill_bits(&gs.zones[1]);
        }
        let mut engine = Engine::new(
            gs,
            self.cvt.clone(),
            self.storage.clone(),
            self.functions.clone(),
            self.instructions.clone(),
            self.max_stack,
            self.fpgm,
            self.prep,
        );
        match engine.run_program(2, instructions) {
            Ok(()) => {
                engine.finish_glyph();
                let mut out = engine.gs.zones[1].cur.clone();
                // GDI-classic: snapshot the link memory the glyph program built (for the phase-control
                // post-pass / its verification). Inert otherwise.
                if engine.gs.gdi_overscale {
                    overscale::record(&engine.gs.links);
                }
                // Native's simple / composite inner gridfit expose final glyph state;
                // both begin with prep state, also used when the glyph has no instructions.
                (out, (engine.gs.ret.dropout_on(), engine.gs.ret.scan_type))
            }
            Err(_) => (cur, self.scan_control()),
        }
    }

    /// The four phantom points in font units: left side bearing, advance, top, bottom.
    fn phantom_funits(&self, gid: GlyphId) -> [Point; 4] {
        let advance = self.hmtx.advance(gid).unwrap_or(0) as i32;
        let lsb = self.hmtx.side_bearing(gid).unwrap_or(0) as i32;
        let (x_min, y_max) = match self.loca.get_glyf(gid, &self.glyf).ok().flatten() {
            Some(Glyph::Simple(s)) => (s.x_min() as i32, s.y_max() as i32),
            Some(Glyph::Composite(c)) => (c.x_min() as i32, c.y_max() as i32),
            None => (0, 0),
        };
        [
            Point::new(x_min - lsb, 0),
            Point::new(x_min - lsb + advance, 0),
            Point::new(0, y_max),
            Point::new(0, y_max - self.upem),
        ]
    }

    /// An outline-less glyph's scaled phantoms, x rounded as a hinted glyph's are: on the 1/16-px grid in an
    /// x-subpixel pass (GDI-natural's metric pass included: VM, Arial's space at ppem 29 under a shear is
    /// 8.0625 px, not 8), else whole.
    fn scale_phantoms(&self, p: [Point; 4]) -> [Point; 4] {
        let gi = &self.retained.getinfo;
        let inhibited = (self.retained.instruct_control & 1) != 0;
        let round_x = !gi.subpixel_positioned && !self.retained.gdi_render && (!self.retained.compatible_advance || inhibited);
        let sp = !round_x && gi.subpixel_hinting && !gi.vertical_lcd && (!gi.compatible_width || self.retained.gdi_render);
        let keep_x = inhibited && self.retained.compatible_advance && !self.retained.gdi_render;
        let rx = |x: i32| if keep_x { x } else if sp { math::RoundState::default().round_sp(x) } else { math::round(x) };
        p.map(|q| Point::new(rx(math::fixmul(q.x, self.x_scale)), math::round(math::fixmul(q.y, self.y_scale))))
    }
}

/// A hinted glyph: contour points and the four hinted phantom points (all F26Dot6).
struct Hinted {
    points: Vec<Point>,
    on_curve: Vec<bool>,
    contours: Vec<u16>,
    phantom: [Point; 4],
    scan_control: (bool, i32),
    /// Font-unit points, unhinted only (empty when grid-fitted).
    fu: Vec<Point>,
}

fn make_twilight(n: usize) -> Zone {
    Zone {
        unscaled: Vec::new(),
        original: vec![Point::default(); n],
        cur: vec![Point::default(); n],
        flags: vec![0; n],
        contours: Vec::new(),
    }
}
