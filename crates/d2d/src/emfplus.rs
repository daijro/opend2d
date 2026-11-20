//! EMF+ (GDI+ metafile) record parsing — the first layer of the GDI+ metafile playback port.
//!
//! A GDI+ recorded metafile stores its EMF+ records inside `EMR_COMMENT` records of the enclosing EMF
//! (an `EMR_COMMENT_EMFPLUS`: the comment data begins with the `EMF+` signature, then a run of EMF+
//! records). This module reassembles that embedded stream and enumerates its records; the per-record
//! interpreters (ported 1:1 from gdiplus `RecordPlayFuncs[]`) and the AA rasterizer build on top.
//!
//! Record layout ([MS-EMFPLUS] 2.3): `{ Type: u16, Flags: u16, Size: u32, DataSize: u32, Data }`, where
//! `Size` is the whole record (incl. the 12-byte header, padded to 4) and `DataSize ≤ Size-12`.

use crate::metafile::GdiMetafile;

const EMR_COMMENT: u32 = 70;
/// The `EMF+` signature that marks an `EMR_COMMENT` as carrying EMF+ records (little-endian `EMF+`).
const EMFPLUS_SIGNATURE: u32 = 0x2B46_4D45;

// EMF+ record types ([MS-EMFPLUS] 2.1.1.1). Only the ones the port references are named; the enumerator
// yields every type regardless.
pub mod rec {
    pub const HEADER: u16 = 0x4001;
    pub const END_OF_FILE: u16 = 0x4002;
    pub const COMMENT: u16 = 0x4003;
    pub const GET_DC: u16 = 0x4004;
    pub const OBJECT: u16 = 0x4008;
    pub const CLEAR: u16 = 0x4009;
    pub const FILL_RECTS: u16 = 0x400A;
    pub const DRAW_RECTS: u16 = 0x400B;
    pub const FILL_POLYGON: u16 = 0x400C;
    pub const DRAW_LINES: u16 = 0x400D;
    pub const FILL_ELLIPSE: u16 = 0x400E;
    pub const DRAW_ELLIPSE: u16 = 0x400F;
    pub const FILL_PATH: u16 = 0x4014;
    pub const DRAW_PATH: u16 = 0x4015;
    pub const DRAW_IMAGE: u16 = 0x401A;
    pub const DRAW_IMAGE_POINTS: u16 = 0x401B;
    pub const DRAW_STRING: u16 = 0x401C;
    pub const SET_ANTIALIAS_MODE: u16 = 0x401E;
    pub const SET_WORLD_TRANSFORM: u16 = 0x402A;
    pub const RESET_WORLD_TRANSFORM: u16 = 0x402B;
    pub const MULTIPLY_WORLD_TRANSFORM: u16 = 0x402C;
    pub const TRANSLATE_WORLD_TRANSFORM: u16 = 0x402D;
    pub const SCALE_WORLD_TRANSFORM: u16 = 0x402E;
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

/// One EMF+ record: its type, 16-bit flags, and the `DataSize` bytes of record data (the TLV header
/// stripped). The low byte of `flags` is the object-table index for records that reference one.
pub struct EmfPlusRecord<'a> {
    pub rtype: u16,
    pub flags: u16,
    pub data: &'a [u8],
}

impl GdiMetafile {
    /// Reassemble the embedded EMF+ record stream from the enclosing EMF's `EMR_COMMENT_EMFPLUS`
    /// records (each contributes its comment data after the 4-byte `EMF+` signature). Empty when the
    /// metafile carries no EMF+ content (a plain GDI EMF).
    pub fn emfplus_stream(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for (itype, rec) in self.records() {
            if itype != EMR_COMMENT || rec.len() < 16 {
                continue;
            }
            // EMR_COMMENT: {iType, nSize, cbData, Data...}; Data @12, cbData bytes. EMF+ comments begin
            // with the signature, then the EMF+ record bytes.
            let cb = u32_at(rec, 8) as usize;
            let end = (12 + cb).min(rec.len());
            if end < 16 || u32_at(rec, 12) != EMFPLUS_SIGNATURE {
                continue;
            }
            out.extend_from_slice(&rec[16..end]);
        }
        out
    }

    /// Enumerate the EMF+ records in `stream` (from [`Self::emfplus_stream`]). Stops at a truncated or
    /// malformed record (what a player would treat as the end).
    pub fn emfplus_records(stream: &[u8]) -> impl Iterator<Item = EmfPlusRecord<'_>> {
        let mut off = 0usize;
        std::iter::from_fn(move || {
            if off + 12 > stream.len() {
                return None;
            }
            let rtype = u16_at(stream, off);
            let flags = u16_at(stream, off + 2);
            let size = u32_at(stream, off + 4) as usize;
            let data_size = u32_at(stream, off + 8) as usize;
            // Size must cover the header + its data, and the data must fit.
            if size < 12 || size % 4 != 0 || off + size > stream.len() || 12 + data_size > size {
                return None;
            }
            let data = &stream[off + 12..off + 12 + data_size];
            off += size;
            Some(EmfPlusRecord { rtype, flags, data })
        })
    }
}

/// A solid fill/stroke colour, straight RGBA (from an EMF+ `ARGB` = `0xAARRGGBB`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Argb(pub [u8; 4]); // [R, G, B, A]

impl Argb {
    fn from_u32(c: u32) -> Self {
        Argb([(c >> 16) as u8, (c >> 8) as u8, c as u8, (c >> 24) as u8])
    }
}

/// An EMF+ pen: a solid brush colour, a width (in the pen's unit; world units by default), and the
/// stroke style fields that live in the `PenData` flag-gated optional block (`LineJoin`, start/end
/// `LineCap`, miter limit, dash style). Values match GDI+'s enums: join 0=miter/1=bevel/2=round,
/// cap 0=flat/1=square/2=round, dash 0=solid/1=dash/2=dot/3=dashdot/4=dashdotdot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pen {
    pub color: Argb,
    pub width: f32,
    pub join: u8,
    pub start_cap: u8,
    pub end_cap: u8,
    pub miter_limit: f32,
    pub dash: u8,
}

/// An object-table entry (EMF+ `Object` record). Only the kinds the port draws with are modelled.
#[derive(Clone, Debug)]
enum Object {
    Brush(Argb),
    Pen(Pen),
}

/// One drawing intent produced by the record interpreter, in the metafile's logical coordinates, with
/// the world transform (2×3, row-major `[m11,m12,m21,m22,dx,dy]`) in effect when it was recorded. The
/// rasterizer applies the transform chain (world → page → device) and fills/strokes it like GpGraphics.
#[derive(Clone, Debug, PartialEq)]
pub enum DrawCmd {
    Clear(Argb),
    FillRects { brush: Argb, rects: Vec<[f32; 4]>, xform: [f32; 6] },
    FillEllipse { brush: Argb, rect: [f32; 4], xform: [f32; 6] },
    FillPolygon { brush: Argb, points: Vec<[f32; 2]>, xform: [f32; 6] },
    DrawLines { pen: Pen, points: Vec<[f32; 2]>, closed: bool, xform: [f32; 6] },
}

const IDENTITY: [f32; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

fn f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn i16_at(b: &[u8], o: usize) -> i16 {
    i16::from_le_bytes([b[o], b[o + 1]])
}

/// Parse an `EmfPlusPen` object: width @16, then the `PenData` optional fields consumed in
/// PenDataFlags (@8) bit order, then the trailing solid-brush ARGB. Only the style fields the
/// widener needs are kept; the rest are stepped over so each field reads at the right offset.
fn parse_pen(d: &[u8]) -> Pen {
    let pflags = u32_at(d, 8);
    let (mut join, mut start_cap, mut end_cap, mut dash) = (0u8, 0u8, 0u8, 0u8);
    let mut miter_limit = 10.0f32; // GDI+ default
    let mut o = 20;
    let mut i32u8 = |o: &mut usize| -> u8 {
        let v = u32_at(d, *o) as u8;
        *o += 4;
        v
    };
    if pflags & 0x0001 != 0 {
        o += 24; // PenTransform: 6 floats
    }
    if pflags & 0x0002 != 0 {
        start_cap = i32u8(&mut o);
    }
    if pflags & 0x0004 != 0 {
        end_cap = i32u8(&mut o);
    }
    if pflags & 0x0008 != 0 {
        join = i32u8(&mut o);
    }
    if pflags & 0x0010 != 0 {
        miter_limit = f32_at(d, o);
        o += 4;
    }
    if pflags & 0x0020 != 0 {
        dash = i32u8(&mut o);
    }
    Pen { width: f32_at(d, 16), color: Argb::from_u32(u32_at(d, d.len() - 4)), join, start_cap, end_cap, miter_limit, dash }
}

/// The EMF+ record interpreter: a 1:1 port of gdiplus's `RecordPlayFuncs[]` play functions, as far as
/// the geometric draw intent goes (the pixels come from the rasterizer). Tracks the world transform and
/// a 256-slot object table, and turns each drawing record into a [`DrawCmd`] in logical space.
pub struct Player {
    world: [f32; 6],
    objects: Vec<Option<Object>>,
    pub cmds: Vec<DrawCmd>,
}

impl Player {
    fn new() -> Self {
        Player { world: IDENTITY, objects: vec![None; 256], cmds: Vec::new() }
    }

    /// Interpret the EMF+ records of `mf` into an ordered list of draw commands.
    pub fn interpret(mf: &GdiMetafile) -> Vec<DrawCmd> {
        let stream = mf.emfplus_stream();
        let mut p = Player::new();
        for r in GdiMetafile::emfplus_records(&stream) {
            p.play(&r);
        }
        p.cmds
    }

    /// Read `count` rectangles from `data` — int16 `RectI` (x,y,w,h) when `int16`, else float `RectF`.
    fn read_rects(data: &[u8], count: usize, int16: bool) -> Vec<[f32; 4]> {
        let stride = if int16 { 8 } else { 16 };
        (0..count)
            .filter(|i| (i + 1) * stride <= data.len())
            .map(|i| {
                let o = i * stride;
                if int16 {
                    [i16_at(data, o) as f32, i16_at(data, o + 2) as f32, i16_at(data, o + 4) as f32, i16_at(data, o + 6) as f32]
                } else {
                    [f32_at(data, o), f32_at(data, o + 4), f32_at(data, o + 8), f32_at(data, o + 12)]
                }
            })
            .collect()
    }

    /// Read `count` points — int16 (x,y) when `int16`, else float.
    fn read_points(data: &[u8], count: usize, int16: bool) -> Vec<[f32; 2]> {
        let stride = if int16 { 4 } else { 8 };
        (0..count)
            .filter(|i| (i + 1) * stride <= data.len())
            .map(|i| {
                let o = i * stride;
                if int16 {
                    [i16_at(data, o) as f32, i16_at(data, o + 2) as f32]
                } else {
                    [f32_at(data, o), f32_at(data, o + 4)]
                }
            })
            .collect()
    }

    /// The brush for a drawing record: inline `ARGB` when the `C` flag (0x8000) is set, else the
    /// solid colour of object `brush_id` (the first `u32` of the record data).
    fn brush_of(&self, flags: u16, data: &[u8]) -> Option<Argb> {
        if flags & 0x8000 != 0 {
            Some(Argb::from_u32(u32_at(data, 0)))
        } else {
            match self.objects.get(u32_at(data, 0) as usize & 0xFF)? {
                Some(Object::Brush(c)) => Some(*c),
                _ => None,
            }
        }
    }

    fn play(&mut self, r: &EmfPlusRecord) {
        let (d, flags) = (r.data, r.flags);
        match r.rtype {
            rec::OBJECT => {
                // flags: low byte = object index, high byte = object type (2 = Pen, 0/1 = Brush).
                let idx = (flags & 0xFF) as usize;
                let obj = match (flags >> 8) & 0xFF {
                    // Pen: {version, type, PenData{flags @8, unit, width @16, optional…}, Brush{…,ARGB}}.
                    // The optional block holds, in PenDataFlags bit order, the style fields GDI+ reads
                    // (StartCap 0x2, EndCap 0x4, Join 0x8, MiterLimit 0x10, LineStyle/dash 0x20, …);
                    // the solid brush's ARGB is the last u32. ([MS-EMFPLUS] 2.2.1.7 EmfPlusPenData.)
                    2 if d.len() >= 24 => Some(Object::Pen(parse_pen(d))),
                    // Brush: {version, type, ARGB} for a solid brush (type 0).
                    0 if d.len() >= 12 && u32_at(d, 4) == 0 => Some(Object::Brush(Argb::from_u32(u32_at(d, 8)))),
                    _ => None,
                };
                self.objects[idx] = obj;
            }
            rec::CLEAR if d.len() >= 4 => self.cmds.push(DrawCmd::Clear(Argb::from_u32(u32_at(d, 0)))),
            rec::FILL_RECTS if d.len() >= 8 => {
                if let Some(brush) = self.brush_of(flags, d) {
                    let count = u32_at(d, 4) as usize;
                    let rects = Self::read_rects(&d[8..], count, flags & 0x4000 != 0);
                    self.cmds.push(DrawCmd::FillRects { brush, rects, xform: self.world });
                }
            }
            rec::FILL_ELLIPSE if d.len() >= 4 => {
                if let Some(brush) = self.brush_of(flags, d) {
                    // Inline-colour: [ARGB, rect]; object brush: [brushId, rect]. Rect follows the u32.
                    let rects = Self::read_rects(&d[4..], 1, flags & 0x4000 != 0);
                    if let Some(&rect) = rects.first() {
                        self.cmds.push(DrawCmd::FillEllipse { brush, rect, xform: self.world });
                    }
                }
            }
            rec::FILL_POLYGON if d.len() >= 8 => {
                // Same layout as FillRects: [color|brushId](u32), count(u32), points[count].
                if let Some(brush) = self.brush_of(flags, d) {
                    let count = u32_at(d, 4) as usize;
                    let points = Self::read_points(&d[8..], count, flags & 0x4000 != 0);
                    self.cmds.push(DrawCmd::FillPolygon { brush, points, xform: self.world });
                }
            }
            rec::DRAW_LINES if d.len() >= 4 => {
                if let Some(Object::Pen(pen)) = self.objects.get((flags & 0xFF) as usize).and_then(|o| o.as_ref()) {
                    let pen = *pen;
                    let count = u32_at(d, 0) as usize;
                    let points = Self::read_points(&d[4..], count, flags & 0x4000 != 0);
                    self.cmds.push(DrawCmd::DrawLines { pen, points, closed: flags & 0x2000 != 0, xform: self.world });
                }
            }
            rec::SET_WORLD_TRANSFORM if d.len() >= 24 => {
                self.world = std::array::from_fn(|i| f32_at(d, i * 4));
            }
            rec::RESET_WORLD_TRANSFORM => self.world = IDENTITY,
            rec::MULTIPLY_WORLD_TRANSFORM if d.len() >= 24 => {
                let m: [f32; 6] = std::array::from_fn(|i| f32_at(d, i * 4));
                // flags bit 0x2000 = prepend (M·world) vs append (world·M); default append.
                self.world = if flags & 0x2000 != 0 { mul_xform(&m, &self.world) } else { mul_xform(&self.world, &m) };
            }
            rec::TRANSLATE_WORLD_TRANSFORM if d.len() >= 8 => {
                let t = [1.0, 0.0, 0.0, 1.0, f32_at(d, 0), f32_at(d, 4)];
                self.world = if flags & 0x2000 != 0 { mul_xform(&t, &self.world) } else { mul_xform(&self.world, &t) };
            }
            rec::SCALE_WORLD_TRANSFORM if d.len() >= 8 => {
                let s = [f32_at(d, 0), 0.0, 0.0, f32_at(d, 4), 0.0, 0.0];
                self.world = if flags & 0x2000 != 0 { mul_xform(&s, &self.world) } else { mul_xform(&self.world, &s) };
            }
            _ => {} // Header/EndOfFile/Comment/GetDC/SetAntiAlias/… — no geometric intent yet.
        }
    }
}

/// Compose two EMF+ 3×2 affines: `apply(mul(a,b), p) = apply(a, apply(b, p))` (b applied first).
fn mul_xform(a: &[f32; 6], b: &[f32; 6]) -> [f32; 6] {
    [
        a[0] * b[0] + a[2] * b[1],
        a[1] * b[0] + a[3] * b[1],
        a[0] * b[2] + a[2] * b[3],
        a[1] * b[2] + a[3] * b[3],
        a[0] * b[4] + a[2] * b[5] + a[4],
        a[1] * b[4] + a[3] * b[5] + a[5],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    // Append an EMF+ record {type, flags, size, datasize, data} to a stream.
    fn eprec(out: &mut Vec<u8>, rtype: u16, flags: u16, data: &[u8]) {
        let data_size = data.len() as u32;
        let size = 12 + data_size; // already 4-aligned in these tests
        out.extend_from_slice(&rtype.to_le_bytes());
        out.extend_from_slice(&flags.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(&data_size.to_le_bytes());
        out.extend_from_slice(data);
    }

    // Build a minimal EMF (88-byte header, one EMR_COMMENT_EMFPLUS carrying `plus`, EMR_EOF).
    fn emf_with_plus(plus: &[u8]) -> Vec<u8> {
        let mut b = vec![0u8; 88];
        let put = |b: &mut Vec<u8>, o: usize, v: u32| b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        put(&mut b, 0, 1); // EMR_HEADER
        put(&mut b, 4, 88);
        put(&mut b, 40, 0x464D_4520); // ' EMF'
        put(&mut b, 72, 960);
        put(&mut b, 76, 480);
        put(&mut b, 80, 254);
        put(&mut b, 84, 127);
        // EMR_COMMENT (type 70): iType, nSize, cbData, 'EMF+' signature, then the EMF+ records.
        let cb = 4 + plus.len();
        let nsize = 12 + cb;
        let pad = (4 - nsize % 4) % 4;
        let mut c = Vec::new();
        c.extend_from_slice(&70u32.to_le_bytes());
        c.extend_from_slice(&((nsize + pad) as u32).to_le_bytes());
        c.extend_from_slice(&(cb as u32).to_le_bytes());
        c.extend_from_slice(&0x2B46_4D45u32.to_le_bytes());
        c.extend_from_slice(plus);
        c.extend_from_slice(&vec![0u8; pad]);
        b.extend_from_slice(&c);
        // EMR_EOF (type 14).
        let mut eof = vec![0u8; 20];
        eof[0..4].copy_from_slice(&14u32.to_le_bytes());
        eof[4..8].copy_from_slice(&20u32.to_le_bytes());
        b.extend_from_slice(&eof);
        b
    }

    #[test]
    fn extracts_and_enumerates_emfplus() {
        // Header + FillRects(brushId=0, count=1, one RectF 10,10,30,20) + EndOfFile.
        let mut plus = Vec::new();
        eprec(&mut plus, rec::HEADER, 0, &[0u8; 16]);
        let mut fr = Vec::new();
        fr.extend_from_slice(&0u32.to_le_bytes()); // brushId
        fr.extend_from_slice(&1u32.to_le_bytes()); // count
        for v in [10.0f32, 10.0, 30.0, 20.0] {
            fr.extend_from_slice(&v.to_le_bytes());
        }
        eprec(&mut plus, rec::FILL_RECTS, 0, &fr);
        eprec(&mut plus, rec::END_OF_FILE, 0, &[]);

        let mf = GdiMetafile::parse(emf_with_plus(&plus)).expect("valid EMF");
        let stream = mf.emfplus_stream();
        assert!(!stream.is_empty(), "EMF+ stream should be extracted");
        let recs: Vec<_> = GdiMetafile::emfplus_records(&stream).map(|r| (r.rtype, r.data.len())).collect();
        assert_eq!(recs, vec![(rec::HEADER, 16), (rec::FILL_RECTS, 24), (rec::END_OF_FILE, 0)]);
    }

    #[test]
    fn interprets_fill_and_line_records() {
        // The exact EMF+ records a System.Drawing EmfPlusOnly metafile recorded for
        // FillRectangle(red,10,10,30,20) + FillEllipse(blue,20,15,25,25) + DrawLine(green/2,0,0,63,47).
        let mut plus = Vec::new();
        eprec(&mut plus, rec::HEADER, 0, &hex("0210c0db010000006000000060000000"));
        eprec(&mut plus, rec::FILL_RECTS, 0xC000, &hex("0000ffff010000000a000a001e001400"));
        eprec(&mut plus, rec::FILL_ELLIPSE, 0xC000, &hex("ff0000ff14000f0019001900"));
        eprec(&mut plus, rec::OBJECT, 0x0200, &hex("0210c0db00000000800000000000000000000040000000000210c0db00000000008000ff"));
        eprec(&mut plus, rec::DRAW_LINES, 0x4000, &hex("02000000000000003f002f00"));
        eprec(&mut plus, rec::END_OF_FILE, 0, &[]);

        let mf = GdiMetafile::parse(emf_with_plus(&plus)).expect("valid EMF");
        let cmds = Player::interpret(&mf);
        assert_eq!(
            cmds,
            vec![
                DrawCmd::FillRects { brush: Argb([255, 0, 0, 255]), rects: vec![[10.0, 10.0, 30.0, 20.0]], xform: IDENTITY },
                DrawCmd::FillEllipse { brush: Argb([0, 0, 255, 255]), rect: [20.0, 15.0, 25.0, 25.0], xform: IDENTITY },
                DrawCmd::DrawLines {
                    pen: Pen { color: Argb([0, 128, 0, 255]), width: 2.0, join: 0, start_cap: 0, end_cap: 0, miter_limit: 10.0, dash: 0 },
                    points: vec![[0.0, 0.0], [63.0, 47.0]],
                    closed: false,
                    xform: IDENTITY,
                },
            ]
        );
    }

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    #[test]
    fn plain_gdi_emf_has_no_emfplus() {
        // The sample from metafile.rs has no EMR_COMMENT_EMFPLUS.
        let mf = GdiMetafile::parse(emf_with_plus(&[])).expect("valid EMF");
        // A comment with an empty EMF+ payload still yields an (empty) stream; a real GDI EMF yields none.
        let plain = {
            let mut b = vec![0u8; 88];
            let put = |b: &mut Vec<u8>, o: usize, v: u32| b[o..o + 4].copy_from_slice(&v.to_le_bytes());
            put(&mut b, 0, 1);
            put(&mut b, 4, 88);
            put(&mut b, 40, 0x464D_4520);
            put(&mut b, 80, 254);
            put(&mut b, 84, 127);
            b
        };
        assert!(GdiMetafile::parse(plain).expect("valid").emfplus_stream().is_empty());
        let _ = mf; // the empty-payload comment case is exercised above
    }
}
