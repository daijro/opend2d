//! When Direct2D renders what a device context has drawn so far: the batching between the API and the target.
//!
//! The device context records each call into a command batch: a 1536-byte buffer, switched once to a 32 KiB one,
//! that every command first checks for 496 bytes of headroom (a glyph run for its glyph data too). When the large
//! buffer runs short, the batch ends as at Flush: the target renders everything queued, the deferred image context's
//! memory is freed and its atlas set closes, and a fresh batch starts.
//!
//! Rendering a batch queues the target's primitives. The target holds at most 200 entries, a budget of 235 units (a
//! solid fill takes 1, a bitmap 2, a glyph run 4) and 128 geometry references (one per fill); a primitive that would
//! pass a limit first renders the queue. That closes the atlas set too, and frees the memory of every DrawImage of an
//! effect already queued, which is all of it unless the limit falls inside a DrawImage's own queueing (its memory then
//! stays until the next flush). A primitive like the one before it (consecutive fills, bitmaps, glyph runs or
//! DrawImages) merges into that entry.
//!
//! A gradient or bitmap brush's fill takes the constant rows its brush writes (2, a radial 3, one more for a radial's
//! focal offset and for an opacity below 1) and merges only with the same brush; an image brush's fill is an operation
//! of the deferred image context, as a DrawImage is. A clip is an entry at its push and at its pop; an antialiased one
//! also copies each fractional edge at the push and redraws its strips and corners at the pop. A layer is four entries
//! at each end: its begin, a copy and its clip, then its clear (unless it starts from the background); its clip, its
//! draw-back (two fills under an opacity or a mask), its end and the clip restored.
//!
//! ponytail: meshes, ink, sprites, opacity masks, command lists, DrawImages of bitmaps, a bitmap brush's extra rows,
//! tiled layers and opacity-brush layers queue no entries here; reference slots (4096 a batch) are not counted.
//!
//! A tessellated fill also needs room in the device's vertex buffer, a 12024-byte ring a flush only unlocks: the next
//! lock starts where the last one stopped, unless 5184 bytes or fewer are left, which discards it. A fill that finds less
//! than 2592 bytes free renders the queue first. A solid fill WARP draws itself writes nothing, a brushed one or a
//! layer's draw-back a placeholder triangle (36 bytes), an antialiased clip's strip or corner a quad (72).
//! ponytail: one ring per context, not per device; DrawBitmap's and the other kinds' vertices are not counted.

/// A target primitive as the queue counts it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Prim {
    Clear,
    /// A fill or stroke of a solid brush (a rectangle, ellipse, rounded rectangle, geometry or line).
    Fill,
    /// A fill or stroke of a gradient or bitmap brush: the brush (its identity) and the constant rows it writes.
    Brush(usize, u32),
    Bitmap,
    /// A glyph run.
    Text,
    /// A DrawImage of an effect, or a fill with an image brush of one: an operation of the deferred image context.
    Image,
    /// An antialiased clip's edge strip redrawn when it pops (`true`: a left or right one) ...
    Strip(bool),
    /// ... and a corner between two strips.
    Corner,
    /// An entry of its own: a clip, a copy of a clip's edge, a layer's begin or end.
    Mark,
    /// A layer's own fill (its clear, or drawing it back), of these constant rows.
    LayerFill(u32),
}

/// What a command or primitive made the target render.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Flush {
    None,
    /// The target rendered its queue.
    Target,
    /// The target rendered its queue inside a DrawImage's own queueing.
    TargetInImage,
    /// The command batch ended.
    Batch,
}

impl Flush {
    fn or(self, b: Flush) -> Flush {
        if self == Flush::None { b } else { self }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Batch {
    pos: u32,
    cap: u32,
    extended: bool,
    entries: u32,
    budget: u32,
    pending: u32,
    refs: u32,
    last: Option<Prim>,
    text_staged: bool,
    /// The context holds text rendering parameters (else its next glyph run makes and records the default ones).
    text_params: bool,
    /// The device's vertex buffer, a ring that outlives flushes and frames: its write cursor, and where the region locked
    /// since the last flush starts (`None`: unlocked).
    vb_cur: u32,
    vb_lock: Option<u32>,
}

const INLINE: u32 = 1536;
const EXTENDED: u32 = 0x8000;
const HEADROOM: u32 = 496;
/// The head of a fresh command batch.
const HEAD: u32 = 16;
/// A glyph run's command before its brush and glyph data.
const GLYPH_RUN: u32 = 56;
/// The most glyph data a glyph run keeps in the batch.
const GLYPH_DATA_IN_BATCH: u32 = 1024;
/// The device's general vertex buffer on WARP, the space a lock discards it below, and the headroom a tessellated fill
/// needs (216 twelve-byte vertices, the largest simple primitive).
const VB: u32 = 12024;
const VB_DISCARD: u32 = 5184;
const VB_HEADROOM: u32 = 12 * 216;

impl Default for Batch {
    fn default() -> Self {
        Batch { pos: HEAD, cap: INLINE, extended: false, entries: 0, budget: 0, pending: 0, refs: 0, last: None, text_staged: false, text_params: false, vb_cur: 0, vb_lock: None }
    }
}

impl Batch {
    /// A fresh command batch; rendering it starts with the target's own two entries (its target and clip).
    fn new_batch(&mut self) {
        *self = Batch { text_params: self.text_params, vb_cur: self.vb_cur, entries: 2, ..Batch::default() };
    }

    /// BeginDraw: a fresh batch, its first command resending the context's state (the target set again: one entry).
    pub(crate) fn begin_draw(&mut self) {
        self.new_batch();
        self.pos += 152;
        self.entries += 1;
    }

    /// The batch ended (EndDraw, Flush, SetTarget).
    pub(crate) fn end(&mut self) {
        self.new_batch();
    }

    /// A command of `bytes`, checking for `extra` bytes beyond the headroom.
    pub(crate) fn command(&mut self, bytes: u32, extra: u32) -> Flush {
        let mut flush = Flush::None;
        if extra + HEADROOM > self.cap - self.pos {
            if !self.extended {
                (self.extended, self.pos, self.cap) = (true, 0, EXTENDED);
            } else {
                self.new_batch();
                flush = Flush::Batch;
            }
        }
        self.pos += bytes;
        flush
    }

    /// SetTextRenderingParams (`set`: non-null), recording them when `drawing`.
    pub(crate) fn set_text_params(&mut self, set: bool, drawing: bool) -> Flush {
        self.text_params = set;
        if set && drawing { self.command(32, 0) } else { Flush::None }
    }

    /// A glyph run's command: the default text rendering parameters first if the context has none, then the run with
    /// its brush copy (`brush` bytes) and its glyph data: `glyphs` indices, advances, and offsets when any is nonzero.
    pub(crate) fn glyph_run(&mut self, brush: u32, glyphs: u32, offsets: bool) -> Flush {
        let mut flush = Flush::None;
        if !self.text_params {
            self.text_params = true;
            flush = self.command(32, 0);
        }
        let per = 2 + 4 + if offsets { 8 } else { 0 };
        let data = (64 + per * glyphs + (2 * glyphs).wrapping_neg() % 4 + 7) & !7;
        // ponytail: a run with more glyph data than the batch keeps holds it outside, checked as a plain command.
        let (bytes, extra) = if data <= GLYPH_DATA_IN_BATCH { (GLYPH_RUN + brush + data, data) } else { (GLYPH_RUN + brush, 0) };
        flush.or(self.command(bytes, extra))
    }

    fn reset_target(&mut self) {
        (self.entries, self.budget, self.pending, self.refs, self.last, self.text_staged) = (0, 0, 0, 0, None, false);
        self.vb_lock = None;
    }

    /// The vertex buffer locked for the queue's writes, discarded first when little of it is left.
    fn vb_locked(&mut self) {
        if self.vb_lock.is_none() {
            if VB - self.vb_cur <= VB_DISCARD {
                self.vb_cur = 0;
            }
            self.vb_lock = Some(self.vb_cur);
        }
    }

    /// The target rendered its queue on its own (a DrawImage finding the deferred image context's memory full).
    pub(crate) fn target_flushed(&mut self) {
        self.reset_target();
    }

    /// The limits checked before a new entry.
    fn check(&mut self, in_image: bool) -> Flush {
        if self.entries >= 200 || self.budget + self.pending > 235 || self.refs == 128 {
            self.reset_target();
            return if in_image { Flush::TargetInImage } else { Flush::Target };
        }
        Flush::None
    }

    fn entry(&mut self, units: u32) {
        self.entries += 1;
        self.budget += self.pending;
        self.pending = units;
    }

    /// A primitive queued on the target.
    pub(crate) fn prim(&mut self, p: Prim) -> Flush {
        let mut flush = Flush::None;
        if p == Prim::Text && !self.text_staged {
            flush = self.check(false);
            self.entry(0);
            self.text_staged = true;
            self.last = None;
        }
        // A tessellated fill locks the vertex buffer and, past the other limits, renders the queue when the buffer lacks
        // its headroom; then writes its vertices: none for a solid fill WARP draws itself, a placeholder triangle for a
        // brushed one or a layer's bitmap draw-back, a quad for an aliased clip strip or corner.
        let vb = match p {
            Prim::Fill | Prim::LayerFill(1) => Some(0),
            Prim::Brush(..) | Prim::LayerFill(_) => Some(36),
            Prim::Strip(_) | Prim::Corner => Some(72),
            _ => None,
        };
        if let Some(bytes) = vb {
            self.vb_locked();
            let f = self.check(false);
            flush = flush.or(f);
            if f == Flush::None && VB - self.vb_cur < VB_HEADROOM {
                self.reset_target();
                flush = flush.or(Flush::Target);
            }
            self.vb_locked();
            self.vb_cur += bytes;
        } else {
            flush = flush.or(self.check(p == Prim::Image));
        }
        let units = match p {
            Prim::Clear | Prim::Image | Prim::Mark => 0,
            Prim::Fill => 1,
            Prim::Brush(_, rows) | Prim::LayerFill(rows) => rows,
            Prim::Bitmap | Prim::Strip(_) | Prim::Corner => 2,
            Prim::Text => 4,
        };
        self.entry(units);
        if matches!(p, Prim::Fill | Prim::Brush(..) | Prim::Text | Prim::LayerFill(_)) {
            self.refs += 1;
        }
        let merges = match p {
            Prim::Clear | Prim::Mark | Prim::LayerFill(_) => false,
            Prim::Corner => matches!(self.last, Some(Prim::Strip(_) | Prim::Corner)),
            _ => self.last == Some(p),
        };
        if merges {
            self.entries -= 1;
        }
        self.last = Some(p);
        flush
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Native traces of the target's counters: an alternating DrawImage and fill frame renders its queue at the
    /// 98th fill (5 entries before the first DrawImage), and fills after a DrawImage render at the 129th.
    #[test]
    fn limits_match_native_traces() {
        let mut b = Batch::default();
        b.begin_draw();
        b.prim(Prim::Clear);
        b.prim(Prim::Clear);
        let mut at = None;
        for k in 1..=110 {
            assert_eq!(b.prim(Prim::Image), Flush::None);
            if b.prim(Prim::Fill) != Flush::None {
                at = Some(k);
                break;
            }
        }
        assert_eq!(at, Some(98));
        // Fills: the 129th after a DrawImage; bitmaps: the 119th; glyph runs: the 60th.
        for (p, n) in [(Prim::Fill, 129), (Prim::Bitmap, 119), (Prim::Text, 60)] {
            let mut b = Batch::default();
            b.prim(Prim::Image);
            assert_eq!((1..=300).find(|_| b.prim(p) != Flush::None), Some(n));
        }
    }

    /// Native counter traces of brushed fills, clips and layers between two DrawImages (entries, then the budget with the
    /// pending rows, as the closing DrawImage sees them).
    #[test]
    fn kinds_match_native_counter_traces() {
        let frame = |ops: &[Prim]| {
            let mut b = Batch::default();
            b.begin_draw();
            for &p in [Prim::Clear, Prim::Clear, Prim::Image].iter().chain(ops).chain(&[Prim::Image]) {
                b.prim(p);
            }
            (b.entries, b.budget + b.pending, b.refs)
        };
        // Three linear fills (opacity below 1), three bitmap-brush fills, three radial ones with a focal offset, two solid.
        let brushes = [[Prim::Brush(1, 3); 3], [Prim::Brush(2, 3); 3], [Prim::Brush(3, 5); 3]].concat();
        assert_eq!(frame(&[brushes, vec![Prim::Fill; 2]].concat()), (11, 35, 11));
        // An aliased clip around a fill; an antialiased one with four fractional edges around a fill; a fill.
        let (m, f) = (Prim::Mark, Prim::Fill);
        let strips = [Prim::Strip(true), Prim::Strip(true), Prim::Strip(false), Prim::Strip(false), Prim::Corner, Prim::Corner, Prim::Corner, Prim::Corner];
        assert_eq!(frame(&[&[m, f, m, m, m, m, m, m, f, m][..], &strips, &[f]].concat()), (20, 19, 3));
        // A layer around a fill (begin, copy, clip, clear; clip, draw-back, end, clip), then a fill.
        let (c1, c2) = (Prim::LayerFill(1), Prim::LayerFill(2));
        assert_eq!(frame(&[m, m, m, c1, f, m, c2, m, m, f]), (17, 5, 4));
        // From the background, then at opacity 0.5 (two draw-back fills), then a fill.
        assert_eq!(frame(&[m, m, m, f, m, c2, m, m, m, m, m, c1, f, m, c1, c2, m, m, f]), (26, 9, 7));
    }

    /// The vertex buffer of twenty antialiased clips with fractional edges, each around a fill, drawn twice on one device:
    /// the first frame (a fresh device) renders its queue for the constant rows at the 126th tessellated fill, the last
    /// corner of the fourteenth pop; the second, from where the first left the buffer, for its space at the 95th, the
    /// eleventh pop's fourth strip (VM traces of the first and the second frame).
    #[test]
    fn vertex_buffer_matches_native_traces() {
        let mut b = Batch::default();
        let mut frame = || {
            b.begin_draw();
            let mut flushes = Vec::new();
            let mut fills = 0;
            for p in [Prim::Clear, Prim::Clear, Prim::Image] {
                b.prim(p);
            }
            let strips = [Prim::Strip(true), Prim::Strip(true), Prim::Strip(false), Prim::Strip(false), Prim::Corner, Prim::Corner, Prim::Corner, Prim::Corner];
            for _ in 0..20 {
                let cycle = [&[Prim::Mark; 5][..], &[Prim::Fill, Prim::Mark], &strips].concat();
                for p in cycle {
                    fills += matches!(p, Prim::Fill | Prim::Strip(_) | Prim::Corner) as u32;
                    if b.prim(p) != Flush::None {
                        flushes.push(fills);
                    }
                }
            }
            b.prim(Prim::Image);
            b.end();
            flushes
        };
        assert_eq!(frame(), [126]);
        assert_eq!(frame(), [95]);
    }

    /// The command batch of an alternating DrawImage (64 bytes) and solid FillRectangle (72) frame, each op tagged
    /// (32), ends at op 334 (native: the batch position traced at each SetTags).
    #[test]
    fn command_batch_ends_where_native_does() {
        let mut b = Batch::default();
        b.begin_draw();
        b.command(32, 0); // the harness's clear
        let ops: Vec<u32> = [32, 0, 0, 0].into_iter().chain((0..240).flat_map(|_| [64, 72])).collect();
        let end = ops.iter().position(|&op| b.command(32, 0) == Flush::Batch || (op > 0 && b.command(op, 0) == Flush::Batch));
        assert_eq!(end, Some(334));
    }

    /// Glyph-run commands as traced natively (each op's text antialias mode 24 and, for DrawGlyphRun, its rendering
    /// params 32): DrawText of 1, 2, 3 and 5 glyphs records 176, 184, 192 and 200 bytes; DrawGlyphRun 208 to 232.
    #[test]
    fn glyph_run_sizes_match_native() {
        let mut b = Batch { text_params: true, extended: true, cap: EXTENDED, ..Batch::default() };
        for (n, text, run) in [(1, 176, 208), (2, 184, 216), (3, 192, 224), (5, 200, 232)] {
            let p = b.pos;
            b.command(24, 0);
            b.glyph_run(24, n, false);
            assert_eq!(b.pos - p, text);
            let p = b.pos;
            b.command(24, 0);
            b.set_text_params(true, true);
            b.glyph_run(24, n, false);
            assert_eq!(b.pos - p, run);
        }
    }
}
