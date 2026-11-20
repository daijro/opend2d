//! Thai word breaking, as DirectWrite's line breaker finds Thai word boundaries.
//!
//! DirectWrite runs a Thai run (characters of line-break class SA in the Thai script, with the spaces among
//! them) through its own dictionary breaker: the text is mapped to TIS-620 bytes, each position is matched
//! against the word dictionary (the `SIAMMAIN` resource: a two-letter index into length/hash records) and the
//! special-word list (`SIAMSPL`), the candidate lengths are disambiguated by looking one word ahead, and a
//! syllable grammar (an LALR parser) measures what the dictionary does not know. Each word's end becomes a
//! line-break opportunity. This module reproduces that breaker and its tables (DirectWrite 10.0.26100); the
//! search state and parser control block are byte-addressed blocks laid out as native's, so every overlapping
//! access lands where native's does.

/// The dictionaries: dwrite.dll's `SIAMMAIN` (words) and `SIAMSPL` (special words) resources. They are
/// loaded from a dwrite.dll image ([`set_dictionary_from_dwrite`]) rather than shipped; without them Thai
/// text breaks only where the Unicode rules allow.
static DICT: std::sync::OnceLock<(Vec<u8>, Vec<u8>)> = std::sync::OnceLock::new();

/// Load the dictionaries from a dwrite.dll image (its `SIAMDB` resources); false when the image has none.
/// The first successful call wins.
pub fn set_dictionary_from_dwrite(dll: &[u8]) -> bool {
    let (Some(main), Some(spl)) = (pe_resource(dll, "SIAMDB", "SIAMMAIN"), pe_resource(dll, "SIAMDB", "SIAMSPL")) else { return false };
    // Both start with the resource magic and their version (0 for the words, 1 for the special words).
    let ok = |d: &[u8], v: u32| d.len() > 16 && d[..4] == 0x2E43_5269u32.to_le_bytes() && d[4..8] == v.to_le_bytes();
    if !ok(main, 0) || !ok(spl, 1) {
        return false;
    }
    let _ = DICT.set((main.to_vec(), spl.to_vec()));
    true
}

fn dict() -> Option<&'static (Vec<u8>, Vec<u8>)> {
    DICT.get()
}

/// A named resource `kind`/`name` (first language) of a PE image.
fn pe_resource<'a>(pe: &'a [u8], kind: &str, name: &str) -> Option<&'a [u8]> {
    let u16le = |o: usize| Some(u16::from_le_bytes([*pe.get(o)?, *pe.get(o + 1)?]));
    let u32le = |o: usize| Some(u32::from_le_bytes(pe.get(o..o + 4)?.try_into().ok()?));
    let nt = u32le(0x3C)? as usize;
    let nsec = u16le(nt + 6)? as usize;
    let opt = nt + 24;
    let dirs = opt + if u16le(opt)? == 0x20B { 112 } else { 96 };
    let rsrc_rva = u32le(dirs + 2 * 8)? as usize;
    let secs = opt + u16le(nt + 20)? as usize;
    let off = |rva: usize| {
        (0..nsec).find_map(|i| {
            let h = secs + 40 * i;
            let (va, raw_sz, raw) = (u32le(h + 12)? as usize, u32le(h + 16)? as usize, u32le(h + 20)? as usize);
            (rva >= va && rva < va + raw_sz).then(|| raw + rva - va)
        })
    };
    let base = off(rsrc_rva)?;
    let entry = |dir: usize, want: &str| -> Option<usize> {
        let n = u16le(dir + 12)? as usize + u16le(dir + 14)? as usize;
        (0..n).find_map(|k| {
            let e = dir + 16 + 8 * k;
            let id = u32le(e)?;
            if id & 0x8000_0000 == 0 {
                return None;
            }
            let s = base + (id & 0x7FFF_FFFF) as usize;
            let len = u16le(s)? as usize;
            let nm: String = char::decode_utf16((0..len).map(|i| u16le(s + 2 + 2 * i).unwrap_or(0))).map(|c| c.unwrap_or('?')).collect();
            (nm == want).then(|| u32le(e + 4)).flatten().map(|v| v as usize)
        })
    };
    let t = entry(base, kind)?;
    let n = entry(base + (t & 0x7FFF_FFFF), name)?;
    let lang_dir = base + (n & 0x7FFF_FFFF);
    let leaf = base + (u32le(lang_dir + 16 + 4)? & 0x7FFF_FFFF) as usize;
    let (rva, size) = (u32le(leaf)? as usize, u32le(leaf + 4)? as usize);
    let o = off(rva)?;
    pe.get(o..o + size)
}
/// Per TIS-620 byte: the character class (bit 0: a character that may end a cluster; 2 and over: a base).
static CLS: &[u8] = include_bytes!("thai/cls.bin");
/// The syllable grammar: rule lengths then rule tokens, per-state token list starts and default slots, the
/// action codes then their parameters, the byte → token map, the sorted token lists, the reduction kinds.
static RULE: &[u8] = include_bytes!("thai/rule.bin");
static SBT: &[u8] = include_bytes!("thai/sbt.bin");
static SBE: &[u8] = include_bytes!("thai/sbe.bin");
static ACT: &[u8] = include_bytes!("thai/act.bin");
static CHR: &[u8] = include_bytes!("thai/chr.bin");
static TOK: &[u8] = include_bytes!("thai/tok.bin");
static RP: &[u8] = include_bytes!("thai/rp.bin");

fn cls(b: u8) -> i32 {
    let o = 4 * b as usize;
    i32::from_le_bytes([CLS[o], CLS[o + 1], CLS[o + 2], CLS[o + 3]])
}
fn u16at(d: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([d[o], d[o + 1]])
}
fn sbt(s: i32) -> u32 {
    u16at(SBT, 2 * s as usize) as u32
}
fn sbe(s: i32) -> u32 {
    u16at(SBE, 2 * s as usize) as u32
}
fn rule_len(r: i32) -> i32 {
    RULE[r as usize] as i32
}
fn rule_tok(r: i32) -> i32 {
    RULE[r as usize + 48] as i32
}

/// A little-endian byte-addressed block: the native structure's memory.
struct Mem<const N: usize>([u8; N]);
impl<const N: usize> Mem<N> {
    fn i(&self, o: usize) -> i32 {
        i32::from_le_bytes(self.0[o..o + 4].try_into().unwrap())
    }
    fn si(&mut self, o: usize, v: i32) {
        self.0[o..o + 4].copy_from_slice(&v.to_le_bytes());
    }
    fn q(&self, o: usize) -> usize {
        u64::from_le_bytes(self.0[o..o + 8].try_into().unwrap()) as usize
    }
    fn sq(&mut self, o: usize, v: usize) {
        self.0[o..o + 8].copy_from_slice(&(v as u64).to_le_bytes());
    }
    fn zero(&mut self, o: usize, n: usize) {
        self.0[o..o + n].fill(0);
    }
}

// The search state: the text end, then answer slots of a position pointer and eight candidate lengths, the
// chosen lengths, the special-list lengths and the special-list "|" cut.
type Ac = Mem<368>;
const AC_END: usize = 0;
const RES: usize = 296;
const ADD: usize = 328;
const CUT: usize = 360;

/// Put a candidate length at the front of a slot's answers.
fn putans(ac: &mut Ac, slot: usize, mut v: i32) {
    for k in 0..8 {
        let o = slot + 8 + 4 * k;
        let old = ac.i(o);
        if old == 0 {
            ac.si(o, v);
            return;
        }
        ac.si(o, v);
        v = old;
    }
}

/// The dictionary lengths of the word starting at the slot's position.
fn search_max(ac: &mut Ac, t: &[u8], slot: usize, main: &[u8]) {
    let w = &main[16..];
    let num = i32::from_le_bytes(main[12..16].try_into().unwrap()) as u32;
    let p = ac.q(slot);
    let mut at = p + 1;
    let key = t[at] as u32 + ((t[p] as u32) << 8);
    let (mut lo, mut hi) = (1u32, num);
    let mut mid;
    if num == 0 {
        return;
    }
    loop {
        mid = (hi + lo) >> 1;
        let k = w[8 * mid as usize - 7] as u32 + ((w[8 * mid as usize - 8] as u32) << 8);
        if k < key {
            lo = mid + 1;
        } else {
            if k <= key {
                break;
            }
            hi = mid.wrapping_sub(1);
        }
        if hi < lo {
            break;
        }
    }
    if hi < lo {
        return;
    }
    let rd = |o: usize| i32::from_le_bytes(w[o..o + 4].try_into().unwrap()) as usize;
    let mut c = t[at];
    let mut rec = rd(8 * mid as usize - 4);
    let rec_end = rd(8 * mid as usize + 4);
    let mut hash: u16 = 0xFFFF;
    let mut len: u16 = 1;
    loop {
        if rec >= rec_end {
            break;
        }
        let end = ac.q(AC_END);
        if at >= end {
            break;
        }
        'scan: loop {
            loop {
                if c <= 0x7F {
                    break 'scan;
                }
                len = len.wrapping_add(1);
                at += 1;
                hash = (hash.wrapping_add(1) as u32).wrapping_mul((c & 0x7F) as u32) as u16;
                if cls(c) >= 2 {
                    break;
                }
                c = t[at];
            }
            c = t[at];
            if c == 0 || at >= end {
                break;
            }
            let odd = (cls(c) % 2).abs();
            if !(odd != 1 || (at + 2 <= end && (t[at + 1] == 0xEC || (at + 2 != end && t[at + 2] == 0xEC)))) {
                break;
            }
        }
        while rec < rec_end {
            if len == u16at(w, rec) {
                let body = rec + 2;
                if len == 2 {
                    putans(ac, slot, 2);
                } else {
                    let mut count = u16at(w, body);
                    rec = body + 2;
                    let mut hit = false;
                    while count != 0 {
                        let h = u16at(w, rec);
                        if hash == h {
                            putans(ac, slot, len as i32);
                            hit = true;
                            break;
                        }
                        if hash <= h {
                            hit = true;
                            break;
                        }
                        rec += 2;
                        count -= 1;
                    }
                    if hit {
                        rec += 2 * count as usize;
                        break;
                    }
                }
                break;
            }
            if len < u16at(w, rec) {
                break;
            }
            if w[rec] == 2 {
                rec += 2;
            } else {
                rec += 2 * u16at(w, rec + 2) as usize + 4;
            }
        }
        if !(c >= 0x80 && at <= ac.q(AC_END)) {
            break;
        }
    }
}

/// The special-word lengths at the slot's position, or the "|" cut.
fn search_add(ac: &mut Ac, t: &[u8], slot: usize, spl: &[u8]) {
    let start = ac.q(slot);
    let mut written = 0usize;
    ac.si(ADD, 0);
    ac.si(CUT, 0);
    let special = 16usize;
    let half = 16 + i32::from_le_bytes(spl[12..16].try_into().unwrap()) as usize;
    let mut at = if t[start] >= 0xCA { half } else { special };
    let mut count = 0usize;
    'outer: loop {
        let mut pos = start;
        let mut dst = ADD + 4 * count;
        let mut bar: Option<usize> = None;
        loop {
            let b = spl[at];
            if b > 0x80 {
                if b <= t[pos] {
                    if b == t[pos] {
                        pos += 1;
                        at += 1;
                        continue;
                    }
                } else if pos == start {
                    break 'outer;
                }
                loop {
                    at += 1;
                    if spl[at] <= 0x80 {
                        break;
                    }
                }
                if spl[at] != 3 {
                    at += 1;
                    continue 'outer;
                }
                break 'outer;
            }
            let len;
            match b {
                3 => break 'outer,
                0x7E => len = start as i32 - pos as i32,
                0x7C => {
                    bar = Some(pos);
                    at += 1;
                    continue;
                }
                _ if (b | 0x80) == t[pos] => {
                    if let Some(bar_pos) = bar {
                        ac.si(CUT, bar_pos as i32 - start as i32);
                        return;
                    }
                    len = pos as i32 - start as i32;
                }
                _ => {
                    at += 1;
                    continue;
                }
            }
            ac.si(dst, len);
            written += 1;
            dst += 4;
            count += 1;
            if count >= 4 {
                break 'outer;
            }
            at += 1;
        }
    }
    ac.si(ADD + 4 * written, 0);
}

/// All candidate lengths of the word at the slot's position.
fn search(ac: &mut Ac, t: &[u8], slot: usize) {
    let c = t[ac.q(slot)];
    if c <= 0x9F || c == 0xCF || c == 0xE6 {
        ac.si(slot + 8, 1);
        return;
    }
    let (main, spl) = dict().expect("Thai dictionary");
    search_max(ac, t, slot, main);
    search_add(ac, t, slot, spl);
    if ac.i(CUT) != 0 {
        ac.zero(slot + 8, 32);
        ac.si(slot + 8, ac.i(CUT));
        return;
    }
    let mut len = ac.i(ADD);
    let mut next = ADD + 4;
    while len != 0 {
        let mut k = 0usize;
        let mag = len.wrapping_abs();
        let mut done = false;
        loop {
            let o = slot + 8 + 4 * k;
            let have = ac.i(o).wrapping_abs();
            if mag == have {
                ac.si(o, len);
                done = true;
                break;
            }
            if mag > have {
                break;
            }
            k += 1;
        }
        if !done {
            let mut o = slot + 8 + 4 * k;
            loop {
                let old = ac.i(o);
                ac.si(o, len);
                len = old;
                o += 4;
                if old == 0 {
                    break;
                }
            }
        }
        len = ac.i(next);
        next += 4;
    }
}

/// Choose the word length at `start`, looking one word ahead; the number of lengths left in `RES`.
fn main_search(ac: &mut Ac, t: &[u8], start: usize) -> i32 {
    ac.sq(16, start);
    let mut len = ac.i(24);
    let mut chosen: i32 = 0;
    'keep_first: {
        if len.wrapping_abs() == 0 {
            ac.zero(24, 32);
            search(ac, t, 16);
            len = ac.i(24);
            if len.wrapping_abs() == 0 {
                break 'keep_first;
            }
        }
        let mut idx: i32 = -1;
        let mut len_off = 24usize;
        let (mut leading, mut count) = (0i32, 0i32);
        let mut slot = 16usize;
        let mut single = false;
        loop {
            let next_pos;
            if len >= 0 {
                next_pos = start + len as usize;
                if t[next_pos] == 0 || next_pos >= ac.q(AC_END) {
                    chosen = count;
                    single = true;
                    break;
                }
            } else {
                next_pos = start;
            }
            slot += 40;
            let next_idx = idx + 1;
            ac.zero(slot + 8, 32);
            ac.sq(slot, next_pos);
            search(ac, t, (40 * idx + 96) as usize);
            if ac.i(slot + 8) != 0 {
                chosen = count;
                leading += 1;
            }
            count += 1;
            len_off += 4;
            len = ac.i(len_off);
            if len.wrapping_abs() == 0 {
                break;
            }
            idx = next_idx;
        }
        if !single {
            if leading == 0 {
                break 'keep_first;
            }
            if leading == 1 {
                len = ac.i(24 + 4 * chosen as usize);
                single = true;
            }
        }
        if single {
            ac.si(RES, len);
            ac.si(RES + 4, 0);
        } else {
            // Several lengths lead on to a word: keep those whose continuation does not end where the first did.
            chosen = 0;
            let mut found = false;
            'outer: loop {
                let k = chosen as usize;
                let mut i = 0usize;
                loop {
                    let next_len = ac.i(64 + 40 * k + 4 * i).wrapping_abs();
                    if next_len == 0 {
                        break;
                    }
                    let after = ac.q(56 + 40 * k) + next_len as usize;
                    if after < ac.q(AC_END) {
                        ac.sq(256, after);
                        ac.zero(264, 32);
                        search(ac, t, 256);
                        if ac.i(264) == 0 {
                            i += 1;
                            continue;
                        }
                    }
                    let mut res_out = RES + 4;
                    ac.sq(RES, ac.i(24 + 4 * k) as u32 as usize);
                    let mut left = leading - 1;
                    if left != 0 {
                        let mut other = k + 1;
                        loop {
                            let other_off = 24 + 4 * other;
                            let mut j = 0usize;
                            loop {
                                let olen = ac.i(64 + 40 * other + 4 * j).wrapping_abs();
                                if olen == 0 {
                                    break;
                                }
                                let sum = olen.wrapping_add(ac.i(other_off).wrapping_abs());
                                if ac.i(RES).wrapping_abs() == sum {
                                    break;
                                }
                                let oafter = ac.q(56 + 40 * other) + olen as usize;
                                if oafter < ac.q(AC_END) {
                                    ac.sq(256, oafter);
                                    ac.zero(264, 32);
                                    search(ac, t, 256);
                                    let third = ac.i(264);
                                    if third != 0 {
                                        let other_len = ac.i(other_off);
                                        let total = third.wrapping_abs().wrapping_add(other_len.wrapping_abs().wrapping_add(ac.i(64 + 40 * other + 4 * j).wrapping_abs()));
                                        if ac.i(RES).wrapping_abs() != total {
                                            ac.si(res_out, other_len);
                                            res_out += 4;
                                            ac.si(res_out, 0);
                                        }
                                        break;
                                    }
                                }
                                j += 1;
                            }
                            left -= 1;
                            if left == 0 {
                                break;
                            }
                            other += 1;
                        }
                    }
                    found = true;
                    break 'outer;
                }
                if i == 0 || {
                    leading -= 1;
                    leading != 0
                } {
                    chosen += 1;
                    continue;
                }
                break;
            }
            if !found {
                break 'keep_first;
            }
        }
        // The chosen slot's answers become the next position's (`RES` keeps its non-negative lengths).
        if ac.i(RES) <= 0 {
            ac.si(24, 0);
        } else {
            let k = chosen as usize;
            ac.sq(16, ac.q(56 + 40 * k));
            for j in 0..8 {
                ac.si(24 + 4 * j, ac.i(64 + 40 * k + 4 * j));
            }
        }
        let mut r = ac.i(RES);
        let mut kept = 0usize;
        if r != 0 {
            let (mut all_i, mut kept_i) = (0usize, 0usize);
            let mut src = RES;
            let mut out = RES;
            loop {
                if r >= 0 {
                    if kept_i != all_i {
                        ac.si(out, r);
                    }
                    kept += 1;
                    kept_i += 1;
                    out += 4;
                }
                src += 4;
                all_i += 1;
                r = ac.i(src);
                if r == 0 {
                    break;
                }
            }
        }
        ac.si(RES + 4 * kept, 0);
        return kept as i32;
    }
    ac.si(RES, ac.i(24));
    ac.si(RES + 4, 0);
    ac.zero(24, 32);
    0
}

// The syllable parser's control block (an AnaGram LALR parser).
type Pcb = Mem<616>;
const TOKEN: usize = 0;
const RTOK: usize = 4;
const LINE: usize = 20;
const COL: usize = 24;
const SSX: usize = 28;
const SN: usize = 32;
const EF_TOK: usize = 40;
const EF_SSX: usize = 44;
const EF_SN: usize = 48;
const SS: usize = 52;
const VS: usize = 180;
const CS: usize = 312;
const BSX: usize = 568;
const PTR: usize = 576;
const LA: usize = 584;
const EXIT: usize = 601;
const VALUE: usize = 604;
const AP: usize = 608;

fn exit(p: &Pcb) -> u8 {
    p.0[EXIT]
}
fn set_exit(p: &mut Pcb, v: u8) {
    p.0[EXIT] = v;
}

fn ag_track(p: &mut Pcb, t: &[u8]) {
    let mut n = p.i(LA).wrapping_sub(p.i(PTR));
    if n == 0 {
        return;
    }
    let mut ptr = p.q(PTR);
    loop {
        let c = t[ptr];
        n -= 1;
        ptr += 1;
        p.sq(PTR, ptr);
        match c {
            9 => {
                let col = p.i(COL);
                p.si(COL, col - (col - 1) % 8 + 8);
            }
            10 => {
                p.si(LINE, p.i(LINE) + 1);
                p.si(COL, 1);
            }
            12 | 13 => {}
            _ => p.si(COL, p.i(COL) + 1),
        }
        if n == 0 {
            break;
        }
    }
}

fn ag_prot(p: &mut Pcb) {
    let bsx = p.i(BSX);
    let r = bsx + 1;
    let slot = 32 - r;
    p.si(BSX, r);
    let ssx = p.i(SSX);
    if slot < 0 || ssx < 0 || slot <= ssx {
        set_exit(p, 4);
        return;
    }
    let (slot, ssx) = (slot as usize, ssx as usize);
    p.si(444 + 4 * bsx as usize, p.i(SN));
    p.si(440 + 4 * slot, ssx as i32);
    p.si(VS + 4 * slot, p.i(VS + 4 * ssx));
    p.si(SS + 4 * slot, p.i(SS + 4 * ssx));
    p.si(CS + 4 * slot, p.i(CS + 4 * ssx));
}

fn ag_ra(p: &mut Pcb) {
    let k = RP[p.i(AP) as usize] as i32 - 1;
    let ssx = p.i(SSX) as usize;
    if k == 0 {
        p.si(VALUE, p.i(CS + 4 * (ssx + 1)));
    } else if k == 1 {
        p.si(VALUE, p.i(CS + 4 * (ssx + 1)).wrapping_neg());
    }
    p.sq(LA, p.q(PTR));
}

fn ag_undo(p: &mut Pcb) {
    if p.i(EF_TOK) == -1 {
        return;
    }
    let mut bsx = p.i(BSX);
    while bsx != 0 {
        p.si(SN, p.i(440 + 4 * bsx as usize));
        p.si(BSX, bsx - 1);
        let saved = (32 - bsx) as usize;
        let ssx = p.i(440 + 4 * saved);
        p.si(SSX, ssx);
        let ssx = ssx as usize;
        p.si(VS + 4 * ssx, p.i(VS + 4 * saved));
        p.si(SS + 4 * ssx, p.i(SS + 4 * saved));
        p.si(CS + 4 * ssx, p.i(CS + 4 * saved));
        bsx = p.i(BSX);
    }
    let tok = p.i(EF_TOK);
    p.si(EF_TOK, -1);
    p.si(TOKEN, tok);
    p.si(SSX, p.i(EF_SSX));
    p.si(SN, p.i(EF_SN));
}

/// Pop a reduction's right side (all but one stack entry): the shared tail of the reduce actions.
fn pop_rule(p: &mut Pcb) {
    let n = rule_len(p.i(AP)) - 1;
    if n != 0 {
        let ssx = p.i(SSX) - n;
        p.si(SSX, ssx);
        p.si(SN, p.i(SS + 4 * ssx as usize));
    } else {
        let ssx = p.i(SSX) as usize;
        p.si(CS + 4 * ssx, p.i(COL));
    }
    let ssx = p.i(SSX) as usize;
    p.si(SS + 4 * ssx, p.i(SN));
}

/// Follow the reduction token's goto actions until one stops; false when one returned false.
fn goto_loop(p: &mut Pcb) -> bool {
    loop {
        if exit(p) != 0 {
            return true;
        }
        let sn = p.i(SN);
        let mut lo = sbe(sn) + 1;
        let mut hi = sbt(sn + 1) - 1;
        let rt = p.i(RTOK) as u8;
        let mut next;
        loop {
            let mid = (hi + lo) >> 1;
            next = mid + 1;
            if TOK[mid as usize] >= rt {
                next = lo;
                hi = mid;
            }
            lo = next;
            if next >= hi {
                break;
            }
        }
        p.si(AP, ACT[next as usize + 512] as i32);
        if !goto_action(p, ACT[next as usize]) {
            return false;
        }
    }
}

/// The goto-phase actions (accept, goto, goto-reduce, reduce).
fn goto_action(p: &mut Pcb, code: u8) -> bool {
    match code {
        0 => {
            set_exit(p, 1);
            false
        }
        1 => {
            p.si(SSX, p.i(SSX) + 1);
            p.si(SN, p.i(AP));
            false
        }
        2 => {
            let r = p.i(AP);
            if rule_len(r) != 1 {
                let ssx = p.i(SSX) - (rule_len(r) - 1);
                p.si(SSX, ssx);
                p.si(SN, p.i(SS + 4 * ssx as usize));
            }
            p.si(EF_TOK, -1);
            p.si(RTOK, rule_tok(r));
            p.si(BSX, 0);
            ag_ra(p);
            exit(p) == 0
        }
        _ => {
            let r = p.i(AP);
            if rule_len(r) != 1 {
                let ssx = p.i(SSX) - (rule_len(r) - 1);
                p.si(SSX, ssx);
                p.si(SN, p.i(SS + 4 * ssx as usize));
            }
            p.si(RTOK, rule_tok(r));
            true
        }
    }
}

/// The error-recovery actions.
fn er_action(p: &mut Pcb, code: u8) {
    match code {
        0 => set_exit(p, 1),
        1 => {
            if p.i(SSX) as u32 > 31 {
                set_exit(p, 4);
                return;
            }
            let ssx = p.i(SSX) as usize;
            p.si(CS + 4 * ssx, p.i(COL));
            p.si(SS + 4 * ssx, p.i(SN));
            p.si(SSX, ssx as i32 + 1);
            p.si(SN, p.i(AP));
        }
        2 | 3 => {
            p.si(RTOK, rule_tok(p.i(AP)));
            pop_rule(p);
            if code == 2 {
                ag_ra(p);
            }
            goto_loop(p);
        }
        c => {
            goto_action(p, c - 4);
        }
    }
}

fn ag_error_resynch(p: &mut Pcb, t: &[u8]) {
    let ssx0 = p.i(SSX);
    if exit(p) != 0 {
        return;
    }
    let mut at;
    loop {
        at = sbt(p.i(SN)) as usize;
        let mut stop = at;
        if TOK[at] != 3 {
            let mut k = at;
            let mut tok = TOK[at];
            loop {
                if tok == 0 {
                    break;
                }
                k += 1;
                stop += 1;
                tok = TOK[k];
                if tok == 3 {
                    break;
                }
            }
        }
        if TOK[stop] != 0 {
            break;
        }
        let ssx = p.i(SSX);
        if ssx == 0 {
            p.si(SSX, ssx0);
            p.si(SN, p.i(SS + 4 * ssx0 as usize));
            set_exit(p, 2);
            return;
        }
        p.si(SSX, ssx - 1);
        p.si(SN, p.i(SS + 4 * (ssx - 1) as usize));
    }
    while TOK[at] != 3 && TOK[at] != 0 {
        at += 1;
    }
    p.si(AP, ACT[at + 512] as i32);
    er_action(p, ACT[at]);
    loop {
        let mut i = sbt(p.i(SN)) as usize;
        while TOK[i] != p.i(TOKEN) as u8 {
            if TOK[i] == 0 {
                break;
            }
            i += 1;
        }
        if TOK[i] != 0 && ACT[i] != 9 {
            break;
        }
        if p.i(TOKEN) == 39 {
            set_exit(p, 2);
            return;
        }
        p.sq(LA, p.q(PTR) + 1);
        ag_track(p, t);
        let la = p.q(LA);
        p.si(TOKEN, CHR[t[la] as usize] as i32);
        p.sq(LA, la + 1);
    }
    p.sq(LA, p.q(PTR));
}

/// The parse actions.
fn action(p: &mut Pcb, t: &[u8], code: u8) {
    match code {
        0 => {
            ag_track(p, t);
            set_exit(p, 1);
        }
        1 => {
            p.si(BSX, 0);
            p.si(EF_TOK, -1);
            if p.i(SSX) as u32 > 0x1F {
                set_exit(p, 4);
                return;
            }
            let ssx = p.i(SSX) as usize;
            p.si(VS + 4 * ssx, t[p.q(PTR)] as i32);
            p.si(CS + 4 * ssx, p.i(COL));
            p.si(SS + 4 * ssx, p.i(SN));
            p.si(SSX, ssx as i32 + 1);
            p.si(SN, p.i(AP));
            ag_track(p, t);
        }
        2 | 3 => {
            p.si(BSX, 0);
            p.si(EF_TOK, -1);
            if code == 3 {
                p.si(RTOK, rule_tok(p.i(AP)));
            }
            let ssx = p.i(SSX) as usize;
            p.si(VS + 4 * ssx, t[p.q(PTR)] as i32);
            pop_rule(p);
            ag_track(p, t);
            if code == 2 {
                p.si(RTOK, rule_tok(p.i(AP)));
                ag_ra(p);
            }
            goto_loop(p);
        }
        5 => {
            let r = p.i(AP);
            let n = rule_len(r);
            p.si(RTOK, rule_tok(r));
            if p.i(EF_TOK) == -1 {
                p.si(EF_TOK, p.i(TOKEN));
                p.si(EF_SSX, p.i(SSX));
                p.si(EF_SN, p.i(SN));
            }
            if n != 0 {
                let ssx = p.i(SSX) - n;
                p.si(SSX, ssx);
                p.si(SN, p.i(SS + 4 * ssx as usize));
            } else {
                ag_prot(p);
                let ssx = p.i(SSX) as usize;
                p.si(VS + 4 * ssx, 0);
                p.si(CS + 4 * ssx, p.i(COL));
                p.si(SS + 4 * ssx, p.i(SN));
            }
            p.sq(LA, p.q(PTR));
            goto_loop(p);
        }
        7 => {
            let mut i = sbt(p.i(SN)) as usize;
            while TOK[i] != 3 && TOK[i] != 0 {
                i += 1;
            }
            if TOK[i] == 0 {
                ag_undo(p);
            }
            p.sq(LA, p.q(PTR));
            ag_error_resynch(p, t);
        }
        8 => {
            if p.i(EF_TOK) == -1 {
                p.si(EF_TOK, p.i(TOKEN));
                p.si(EF_SSX, p.i(SSX));
                p.si(EF_SN, p.i(SN));
            }
            ag_prot(p);
            let ssx = p.i(SSX) as usize;
            p.si(VS + 4 * ssx, 0);
            p.si(CS + 4 * ssx, p.i(COL));
            p.si(SS + 4 * ssx, p.i(SN));
            p.si(SSX, ssx as i32 + 1);
            p.si(SN, p.i(AP));
            p.sq(LA, p.q(PTR));
        }
        9 => {
            p.si(BSX, 0);
            p.si(EF_TOK, -1);
            let tok = p.i(TOKEN);
            loop {
                ag_track(p, t);
                let la = p.q(LA);
                let c = CHR[t[la] as usize] as i32;
                p.sq(LA, la + 1);
                p.si(TOKEN, c);
                if c != tok {
                    break;
                }
            }
            p.sq(LA, p.q(PTR));
        }
        // Codes 4, 6 and 10 are not in the grammar's action table.
        _ => set_exit(p, 1),
    }
}

/// Run the syllable grammar from `start`; its reduction value.
fn parse(t: &[u8], start: usize) -> i32 {
    let mut p: Pcb = Mem([0; 616]);
    p.sq(PTR, start);
    p.si(SN, 0);
    p.sq(592, 0);
    p.si(SSX, 0);
    p.si(BSX, 0);
    p.si(EF_TOK, -1);
    p.si(308, 0);
    p.sq(LA, start);
    p.si(SS, 0);
    p.si(AP, 0);
    p.si(LINE, 1);
    p.si(COL, 1);
    set_exit(&mut p, 0);
    loop {
        let sn = p.i(SN);
        let mut at = sbt(sn);
        if TOK[at as usize] != 0 {
            let fallback = sbe(sn);
            let la = p.q(LA);
            let tok = CHR[t[la] as usize];
            p.si(TOKEN, tok as i32);
            p.sq(LA, la + 1);
            let mut hi = fallback - 1;
            let mut next;
            loop {
                let mid = (hi + at) >> 1;
                next = mid + 1;
                if TOK[mid as usize] <= tok {
                    next = at;
                    hi = mid;
                }
                at = next;
                if next >= hi {
                    break;
                }
            }
            if TOK[next as usize] != tok {
                at = fallback;
            }
        }
        p.si(AP, ACT[at as usize + 512] as i32);
        action(&mut p, t, ACT[at as usize]);
        if exit(&p) != 0 {
            break;
        }
    }
    p.i(VALUE)
}

/// The length of the syllable the grammar reads at `start` (end `end`): a word the dictionary does not know.
fn lalr(t: &[u8], start: usize, end: usize) -> i32 {
    if end as isize - start as isize > 13 {
        let mut karan = 10usize;
        while karan != 0 && t[start + karan] != 0xEC {
            karan -= 1;
        }
        if karan != 0 {
            let next = t[start + karan + 1];
            if next.wrapping_add(32) > 4 && next.wrapping_add(95) <= 0x2D && t[start + karan + 2].wrapping_add(32) <= 4 {
                return karan as i32 + 3;
            }
            return karan as i32 + 2;
        }
    }
    let value = parse(t, start);
    let mut len = value;
    if value < 0 {
        len = value.wrapping_neg();
        let n = value.wrapping_neg() as usize;
        let last = t[start + n - 1];
        if last.wrapping_add(32) > 4 {
            len = 0;
            if last.wrapping_add(95) <= 0x2D {
                let after = t[start + n];
                if cls(after) & 1 != 0 {
                    if after.wrapping_add(32) <= 4 {
                        return 1i32.wrapping_sub(value);
                    }
                } else {
                    return value.wrapping_neg();
                }
            }
        }
    }
    len
}

/// The word lengths (in bytes) of the TIS-620 text `t[..n]` (`t` NUL-terminated and padded), at most `max - 1`
/// words, as DirectWrite's Thai breaker returns them; a zero length ends the list.
fn find_word_breaks(t: &[u8], n: usize, max: usize) -> Vec<u8> {
    let mut out = vec![0u8; max + 2];
    let mut ac: Ac = Mem([0; 368]);
    let mut q = 0usize;
    out[0] = 0;
    let mut left = max as i64 - 1;
    if left == 0 {
        return out;
    }
    let end = n;
    ac.sq(AC_END, end);
    ac.sq(8, end.wrapping_sub(2));
    let mut a = 0usize;
    let mut unknown = false;
    loop {
        if a >= end || t[a] == 0 {
            out[q + 1] = 0;
            return out;
        }
        let new_word;
        if t[a] <= 0x9F {
            unknown = false;
            a += 1;
            if out[q] == 0xFF {
                q += 1;
                out[q] = 1;
                left -= 1;
                if left == 0 {
                    out[q + 1] = 0;
                    return out;
                }
                continue;
            }
            ac.si(24, 0);
            out[q] += 1;
            continue;
        }
        let next = t[a + 1];
        if t[a] == 0xCF {
            if next == 0xC5 && t[a + 2] == 0xCF {
                a += 3;
                out[q] = out[q].wrapping_add(3);
                new_word = true;
            } else if next == 0xBE && t[a + 2] == 0xB3 && t[a + 3] == 0xCF {
                a += 4;
                out[q] = out[q].wrapping_add(4);
                new_word = true;
            } else {
                a += 1;
                out[q] = out[q].wrapping_add(1);
                new_word = true;
            }
        } else if next <= 0xA0 {
            a += 1;
            out[q] = out[q].wrapping_add(1);
            new_word = true;
        } else {
            let found = main_search(&mut ac, t, a);
            let mut take: Option<i32> = None;
            if found != 0 {
                if unknown {
                    unknown = false;
                    if out[q] >= 3 {
                        q += 1;
                        out[q] = 0;
                        left -= 1;
                        if left == 0 {
                            out[q + 1] = 0;
                            return out;
                        }
                    }
                }
                let r0 = ac.i(RES);
                if found == 1 {
                    take = Some(r0);
                } else {
                    let (mut rank, mut prev) = (1, r0);
                    for i in 1..found as usize {
                        let ri = ac.i(RES + 4 * i);
                        if prev - ri >= 3 {
                            if rank == 1 {
                                take = Some(r0);
                                break;
                            }
                        } else {
                            prev = ri;
                        }
                        rank += 1;
                    }
                    if take.is_none() {
                        a += r0 as usize;
                        out[q] = out[q].wrapping_add(r0 as u8);
                        continue;
                    }
                }
            } else {
                let mut len = ac.i(RES);
                if len == 0 {
                    len = lalr(t, a, end) - 1;
                    ac.si(RES, len);
                    if len <= 1 {
                        // An unknown letter: it and the marks after it join the word.
                        let c = t[a];
                        unknown = true;
                        a += if c > 0xCE { 2 } else { 1 };
                        let mut k = out[q].wrapping_add((c > 0xCE) as u8).wrapping_add(1);
                        loop {
                            out[q] = k;
                            if a > end || cls(t[a]) & 1 != 0 {
                                break;
                            }
                            a += 1;
                            k = k.wrapping_add(1);
                        }
                        continue;
                    }
                }
                take = Some(len);
            }
            let len = take.unwrap();
            a = (a as isize + len as isize) as usize;
            out[q] = out[q].wrapping_add(len as u8);
            let mut follow;
            loop {
                follow = t[a];
                if follow != 0xE6 && follow != 0xCF {
                    break;
                }
                ac.si(24, 0);
                a += 1;
                out[q] = out[q].wrapping_add(1);
            }
            if a > end {
                out[q + 1] = 0;
                return out;
            }
            if follow <= 0x9F {
                continue;
            }
            new_word = true;
        }
        if new_word {
            q += 1;
            out[q] = 0;
            left -= 1;
            if left == 0 {
                out[q + 1] = 0;
                return out;
            }
        }
    }
}

/// Thai word ends in `text`: byte offsets after the last character of each word inside a Thai run (a run of
/// line-break class SA characters in the Thai script, U+0E01–U+0E5B, with the spaces and zero-width spaces
/// among them), where DirectWrite allows a line break.
pub fn word_breaks(text: &str) -> Vec<usize> {
    use unicode_linebreak::{break_property, BreakClass};
    if dict().is_none() {
        return Vec::new();
    }
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let sa_thai = |c: char| (0x0E01..=0x0E5B).contains(&(c as u32)) && break_property(c as u32) == BreakClass::ComplexContext;
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if !sa_thai(chars[i].1) {
            i += 1;
            continue;
        }
        // The run: Thai SA letters and spaces, mapped to TIS-620.
        let mut bytes = Vec::new();
        let mut j = i;
        while j < chars.len() {
            let c = chars[j].1;
            let b = if c == ' ' || c == '\u{200B}' {
                b' '
            } else if sa_thai(c) {
                (c as u32 - 96) as u8
            } else {
                break;
            };
            bytes.push(b);
            j += 1;
        }
        let n = bytes.len();
        bytes.resize(n + 64, 0);
        let words = find_word_breaks(&bytes, n, n);
        let mut pos = 0usize;
        for &w in words.iter().take_while(|&&w| w != 0) {
            pos += w as usize;
            if pos >= n {
                break;
            }
            out.push(chars.get(i + pos).map_or(text.len(), |c| c.0));
        }
        i = j.max(i + 1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The breaker against 349 phrases probed on the reference VM (Tahoma 12, th-th, in a 0.5-DIP box so every
    /// opportunity shows): `probes.txt` holds each phrase and the UTF-16 positions where native allows a line
    /// break; those after a space are the Unicode rules', not the breaker's. Opt-in, as the dictionaries are not
    /// shipped: runs only with `D2D_THAI_DICT` naming a dwrite.dll.
    #[test]
    fn breaks_match_native_probes() {
        let Some(dll) = std::env::var_os("D2D_THAI_DICT").and_then(|p| std::fs::read(p).ok()) else { return };
        assert!(set_dictionary_from_dwrite(&dll));
        let mut bad = Vec::new();
        for line in include_str!("thai/probes.txt").lines() {
            let (text, want) = line.split_once('\t').unwrap();
            let u: Vec<u16> = text.encode_utf16().collect();
            let want: Vec<usize> = want.split(',').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).filter(|&p: &usize| u[p - 1] != 0x20).collect();
            let got: Vec<usize> = word_breaks(text).iter().map(|&b| text[..b].encode_utf16().count()).collect();
            if got != want {
                bad.push(format!("{text}: ours {got:?} native {want:?}"));
            }
        }
        assert!(bad.is_empty(), "{} probes differ:\n{}", bad.len(), bad.join("\n"));
    }
}
