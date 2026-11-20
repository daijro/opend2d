//! The exact `deflate` WIC's PNG encoder produces. WIC bundles Chromium's zlib,
//! which is stock zlib with one deviation: the `deflate_slow` match-finder hashes a 4-byte window
//! (3-byte at level ≥ 6) through SSE4.2 `crc32c` instead of the multiplicative hash. Everything else
//! — `longest_match`, `fill_window`, the sliding window / hash chains and the whole `_tr_flush_block`
//! Huffman stage — is stock zlib (`deflate_slow`, `longest_match`, `fill_window`, `_tr_flush_block`,
//! `deflateInit2_` and the config table).
//!
//! A faithful port of stock zlib's `deflate.c`/`trees.c` (public, BSD-licensed) with the crc32c
//! hash, `memLevel = 9` (hash_bits 16, lit_bufsize 32768) and the levels/window WIC uses. It produces
//! byte-identical compressed streams, which is required for byte-identical PNGs.

const MIN_MATCH: u32 = 3;
const MAX_MATCH: u32 = 258;
const MIN_LOOKAHEAD: u32 = 262; // MAX_MATCH + MIN_MATCH + 1
const L_CODES: usize = 286;
const D_CODES: usize = 30;
const BL_CODES: usize = 19;
const HEAP_SIZE: usize = 2 * L_CODES + 1;
const MAX_BITS: usize = 15;
const MAX_BL_BITS: usize = 7;
const LITERALS: usize = 256;
const END_BLOCK: usize = 256;
const REP_3_6: usize = 16;
const REPZ_3_10: usize = 17;
const REPZ_11_138: usize = 18;
const TOO_FAR: u32 = 4096;

const EXTRA_LBITS: [u8; 29] =
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const EXTRA_DBITS: [u8; 30] =
    [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const EXTRA_BLBITS: [u8; 19] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7];
const BL_ORDER: [usize; 19] =
    [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/// CRC-32C (reflected poly 0x82F63B78) over a word's 4 bytes low-to-high: software `_mm_crc32_u32(0,v)`.
fn crc32c_u32(mut v: u32) -> u32 {
    let mut crc = 0u32;
    for _ in 0..4 {
        let mut c = (crc ^ (v & 0xff)) & 0xff;
        for _ in 0..8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0x82F6_3B78 } else { c >> 1 };
        }
        crc = (crc >> 8) ^ c;
        v >>= 8;
    }
    crc
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in data.chunks(5552) {
        for &x in chunk {
            a += x as u32;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

fn bi_reverse(mut code: u32, len: usize) -> u32 {
    let mut res = 0u32;
    for _ in 0..len {
        res = (res << 1) | (code & 1);
        code >>= 1;
    }
    res
}

/// Static Huffman tables, generated exactly as zlib's `tr_static_init`.
struct Static {
    length_code: [u8; 256],
    dist_code: [u8; 512],
    base_length: [u32; 29],
    base_dist: [u32; 30],
    ltree_code: [u16; L_CODES + 2],
    ltree_len: [u8; L_CODES + 2],
    dtree_code: [u16; D_CODES],
    dtree_len: [u8; D_CODES],
}

impl Static {
    fn new() -> Static {
        let mut s = Static {
            length_code: [0; 256],
            dist_code: [0; 512],
            base_length: [0; 29],
            base_dist: [0; 30],
            ltree_code: [0; L_CODES + 2],
            ltree_len: [0; L_CODES + 2],
            dtree_code: [0; D_CODES],
            dtree_len: [0; D_CODES],
        };
        let mut length = 0usize;
        for code in 0..28 {
            s.base_length[code] = length as u32;
            for _ in 0..(1u32 << EXTRA_LBITS[code]) {
                s.length_code[length] = code as u8;
                length += 1;
            }
        }
        s.length_code[255] = 28;
        let mut dist = 0usize;
        for code in 0..16 {
            s.base_dist[code] = dist as u32;
            for _ in 0..(1u32 << EXTRA_DBITS[code]) {
                s.dist_code[dist] = code as u8;
                dist += 1;
            }
        }
        dist >>= 7;
        for code in 16..D_CODES {
            s.base_dist[code] = (dist << 7) as u32;
            for _ in 0..(1u32 << (EXTRA_DBITS[code] - 7)) {
                s.dist_code[256 + dist] = code as u8;
                dist += 1;
            }
        }
        let len_of = |n: usize| -> u8 {
            if n <= 143 {
                8
            } else if n <= 255 {
                9
            } else if n <= 279 {
                7
            } else {
                8
            }
        };
        let mut bl_count = [0u16; MAX_BITS + 1];
        for n in 0..288 {
            bl_count[len_of(n) as usize] += 1;
        }
        let mut next = [0u16; MAX_BITS + 1];
        let mut code = 0u32;
        for bits in 1..=MAX_BITS {
            code = (code + bl_count[bits - 1] as u32) << 1;
            next[bits] = code as u16;
        }
        for n in 0..288 {
            let l = len_of(n) as usize;
            let c = next[l];
            next[l] += 1;
            if n < L_CODES + 2 {
                s.ltree_len[n] = l as u8;
                s.ltree_code[n] = bi_reverse(c as u32, l) as u16;
            }
        }
        for n in 0..D_CODES {
            s.dtree_len[n] = 5;
            s.dtree_code[n] = bi_reverse(n as u32, 5) as u16;
        }
        s
    }
    fn d_code(&self, dist: u32) -> usize {
        (if dist < 256 { self.dist_code[dist as usize] } else { self.dist_code[256 + (dist >> 7) as usize] }) as usize
    }
}

/// One Huffman tree's per-symbol arrays (zlib's `ct_data`: freq/code and dad/len).
#[derive(Clone)]
struct Tree {
    freq: Vec<u16>,
    code: Vec<u16>,
    dad: Vec<u16>,
    len: Vec<u8>,
}
impl Tree {
    fn new(n: usize) -> Tree {
        Tree { freq: vec![0; n], code: vec![0; n], dad: vec![0; n], len: vec![0; n] }
    }
}

struct Deflate<'a> {
    st: &'a Static,
    level: u32,
    strategy: u32,
    good_match: u32,
    max_lazy: u32,
    nice_match: u32,
    max_chain: u32,

    window: Vec<u8>,
    w_size: u32,
    w_mask: u32,
    window_size: usize,
    head: Vec<u16>,
    prev: Vec<u16>,
    hash_mask: u32,

    input: &'a [u8],
    next_in: usize,

    strstart: u32,
    block_start: i64,
    lookahead: u32,
    match_start: u32,
    prev_match: u32,
    prev_length: u32,
    match_length: u32,
    match_available: bool,
    insert: u32,

    l: Tree,
    d: Tree,
    bl: Tree,
    opt_len: usize,
    static_len: usize,

    sym: Vec<u8>,
    sym_next: usize,
    sym_end: usize,
    l_max_code: usize,
    d_max_code: usize,

    out: Vec<u8>,
    bi_buf: u32,
    bi_valid: u32,
}

const CONFIG: [(u32, u32, u32, u32); 10] = [
    (0, 0, 0, 0),
    (4, 4, 8, 4),
    (4, 5, 16, 8),
    (4, 6, 32, 32),
    (4, 4, 16, 16),
    (8, 16, 32, 32),
    (8, 16, 128, 128),
    (8, 32, 128, 256),
    (32, 128, 258, 1024),
    (32, 258, 258, 4096),
];

impl<'a> Deflate<'a> {
    fn new(st: &'a Static, input: &'a [u8], level: u32, window_bits: u32, strategy: u32) -> Deflate<'a> {
        let (good, lazy, nice, chain) = CONFIG[level as usize];
        let w_size = 1u32 << window_bits;
        // memLevel 9: hash_bits 16, lit_bufsize 32768.
        let hash_bits = 16usize;
        let lit_bufsize = 1usize << 15;
        Deflate {
            st,
            level,
            strategy,
            good_match: good,
            max_lazy: lazy,
            nice_match: nice,
            max_chain: chain,
            window: vec![0u8; 2 * w_size as usize + MAX_MATCH as usize + 8],
            w_size,
            w_mask: w_size - 1,
            window_size: 2 * w_size as usize,
            head: vec![0u16; 1 << hash_bits],
            prev: vec![0u16; w_size as usize],
            hash_mask: (1u32 << hash_bits) - 1,
            input,
            next_in: 0,
            strstart: 0,
            block_start: 0,
            lookahead: 0,
            match_start: 0,
            prev_match: 0,
            prev_length: MIN_MATCH - 1,
            match_length: MIN_MATCH - 1,
            match_available: false,
            insert: 0,
            l: Tree::new(HEAP_SIZE),
            d: Tree::new(2 * D_CODES + 1),
            bl: Tree::new(2 * BL_CODES + 1),
            opt_len: 0,
            static_len: 0,
            sym: vec![0u8; lit_bufsize * 3],
            sym_next: 0,
            sym_end: 3 * (lit_bufsize - 1),
            l_max_code: 0,
            d_max_code: 0,
            out: Vec::new(),
            bi_buf: 0,
            bi_valid: 0,
        }
    }

    // --- bit output ---
    fn send_bits(&mut self, value: u32, length: u32) {
        self.bi_buf |= value << self.bi_valid;
        self.bi_valid += length;
        while self.bi_valid >= 8 {
            self.out.push(self.bi_buf as u8);
            self.bi_buf >>= 8;
            self.bi_valid -= 8;
        }
    }
    fn bi_windup(&mut self) {
        if self.bi_valid > 0 {
            self.out.push(self.bi_buf as u8);
        }
        self.bi_buf = 0;
        self.bi_valid = 0;
    }

    // --- hashing (crc32c) ---
    fn hash(&self, pos: u32) -> usize {
        let p = pos as usize;
        let mut v = self.window[p] as u32
            | (self.window[p + 1] as u32) << 8
            | (self.window[p + 2] as u32) << 16
            | (self.window[p + 3] as u32) << 24;
        if self.level >= 6 {
            v &= 0x00FF_FFFF;
        }
        (crc32c_u32(v) & self.hash_mask) as usize
    }
    fn insert_string(&mut self, pos: u32) -> u32 {
        let h = self.hash(pos);
        let head = self.head[h];
        self.prev[(pos & self.w_mask) as usize] = head;
        self.head[h] = pos as u16;
        head as u32
    }

    fn slide_hash(&mut self) {
        let w = self.w_size as u16;
        for h in self.head.iter_mut() {
            *h = if *h >= w { *h - w } else { 0 };
        }
        for p in self.prev[..self.w_size as usize].iter_mut() {
            *p = if *p >= w { *p - w } else { 0 };
        }
    }

    fn read_buf(&mut self, start: usize, want: usize) -> usize {
        let avail = (self.input.len() - self.next_in).min(want);
        self.window[start..start + avail]
            .copy_from_slice(&self.input[self.next_in..self.next_in + avail]);
        self.next_in += avail;
        avail
    }

    fn fill_window(&mut self) {
        let wsize = self.w_size;
        loop {
            let mut more = self.window_size as u32 - self.lookahead - self.strstart;
            if self.strstart >= wsize + (wsize - MIN_LOOKAHEAD) {
                let n = (wsize - more) as usize;
                self.window.copy_within(wsize as usize..wsize as usize + n, 0);
                self.match_start = self.match_start.wrapping_sub(wsize);
                self.strstart -= wsize;
                self.block_start -= wsize as i64;
                if self.insert > self.strstart {
                    self.insert = self.strstart;
                }
                self.slide_hash();
                more += wsize;
            }
            if self.next_in >= self.input.len() {
                break;
            }
            let start = (self.strstart + self.lookahead) as usize;
            let got = self.read_buf(start, more as usize) as u32;
            self.lookahead += got;
            if self.lookahead + self.insert >= MIN_MATCH {
                let mut s = self.strstart - self.insert;
                while self.insert > 0 {
                    let h = self.hash(s);
                    self.prev[(s & self.w_mask) as usize] = self.head[h];
                    self.head[h] = s as u16;
                    s += 1;
                    self.insert -= 1;
                    if self.lookahead + self.insert < MIN_MATCH {
                        break;
                    }
                }
            }
            // Zero-fill MAX_MATCH bytes past the data (zlib's high-water region) so hash reads and
            // longest_match scans past the valid data see deterministic zeros, as zlib guarantees.
            let hi = (self.strstart + self.lookahead) as usize;
            let end = (hi + MAX_MATCH as usize).min(self.window.len());
            for b in self.window[hi..end].iter_mut() {
                *b = 0;
            }
            if self.lookahead >= MIN_LOOKAHEAD || self.next_in >= self.input.len() {
                break;
            }
        }
    }

    fn longest_match(&mut self, cur_in: u32) -> u32 {
        let mut cur = cur_in as usize;
        let mut chain = if self.prev_length >= self.good_match { self.max_chain >> 2 } else { self.max_chain };
        let strstart = self.strstart as usize;
        let w = &self.window;
        let mut best_len = self.prev_length as usize;
        let limit = if self.strstart > self.w_size - MIN_LOOKAHEAD {
            (self.strstart - (self.w_size - MIN_LOOKAHEAD)) as usize
        } else {
            0
        };
        let mut nice = self.nice_match as usize;
        if nice > self.lookahead as usize {
            nice = self.lookahead as usize;
        }
        let strend = strstart + MAX_MATCH as usize;
        let mut scan_end1 = w[strstart + best_len - 1];
        let mut scan_end = w[strstart + best_len];
        let mut best_match = self.match_start;
        loop {
            if w[cur + best_len] == scan_end
                && w[cur + best_len - 1] == scan_end1
                && w[cur] == w[strstart]
                && w[cur + 1] == w[strstart + 1]
            {
                let mut sp = strstart + 2;
                let mut mp = cur + 2;
                loop {
                    sp += 1;
                    mp += 1;
                    if w[sp] != w[mp] || sp >= strend {
                        break;
                    }
                }
                let len = MAX_MATCH as usize - (strend - sp);
                if len > best_len {
                    best_match = cur as u32;
                    best_len = len;
                    if len >= nice {
                        break;
                    }
                    scan_end1 = w[strstart + best_len - 1];
                    scan_end = w[strstart + best_len];
                }
            }
            cur = self.prev[(cur as u32 & self.w_mask) as usize] as usize;
            if cur <= limit {
                break;
            }
            chain -= 1;
            if chain == 0 {
                break;
            }
        }
        self.match_start = best_match;
        if (best_len as u32) <= self.lookahead {
            best_len as u32
        } else {
            self.lookahead
        }
    }

    // --- symbol buffer / block flush ---
    fn tally_lit(&mut self, c: u8) -> bool {
        self.sym[self.sym_next] = 0;
        self.sym[self.sym_next + 1] = 0;
        self.sym[self.sym_next + 2] = c;
        self.sym_next += 3;
        self.l.freq[c as usize] += 1;
        self.sym_next == self.sym_end
    }
    fn tally_dist(&mut self, dist: u32, lc: u32) -> bool {
        self.sym[self.sym_next] = dist as u8;
        self.sym[self.sym_next + 1] = (dist >> 8) as u8;
        self.sym[self.sym_next + 2] = lc as u8;
        self.sym_next += 3;
        self.l.freq[self.st.length_code[lc as usize] as usize + LITERALS + 1] += 1;
        self.d.freq[self.st.d_code(dist - 1)] += 1;
        self.sym_next == self.sym_end
    }

    fn init_block(&mut self) {
        for f in self.l.freq.iter_mut() {
            *f = 0;
        }
        for f in self.d.freq.iter_mut() {
            *f = 0;
        }
        for f in self.bl.freq.iter_mut() {
            *f = 0;
        }
        self.l.freq[END_BLOCK] = 1;
        self.opt_len = 0;
        self.static_len = 0;
        self.sym_next = 0;
    }

    fn flush_block(&mut self, last: bool) {
        let buf_start = self.block_start;
        let stored_len = (self.strstart as i64 - self.block_start) as u32;
        self.tr_flush_block(buf_start, stored_len, last);
        self.block_start = self.strstart as i64;
    }

    fn deflate_slow(&mut self) {
        loop {
            if self.lookahead < MIN_LOOKAHEAD {
                self.fill_window();
                if self.lookahead == 0 {
                    break;
                }
            }
            let mut hash_head = 0u32;
            if self.lookahead >= MIN_MATCH {
                hash_head = self.insert_string(self.strstart);
            }
            self.prev_length = self.match_length;
            self.prev_match = self.match_start;
            self.match_length = MIN_MATCH - 1;
            if hash_head != 0
                && self.prev_length < self.max_lazy
                && self.strstart - hash_head <= self.w_size - MIN_LOOKAHEAD
            {
                self.match_length = self.longest_match(hash_head);
                if self.match_length <= 5
                    && (self.strategy == 1
                        || (self.match_length == MIN_MATCH
                            && self.strstart - self.match_start > TOO_FAR))
                {
                    self.match_length = MIN_MATCH - 1;
                }
            }
            if self.prev_length >= MIN_MATCH && self.match_length <= self.prev_length {
                let max_insert = self.strstart + self.lookahead - MIN_MATCH;
                let dist = self.strstart - 1 - self.prev_match;
                let lc = self.prev_length - MIN_MATCH;
                let bflush = self.tally_dist(dist, lc);
                self.lookahead -= self.prev_length - 1;
                self.prev_length -= 2;
                loop {
                    self.strstart += 1;
                    if self.strstart <= max_insert {
                        self.insert_string(self.strstart);
                    }
                    self.prev_length -= 1;
                    if self.prev_length == 0 {
                        break;
                    }
                }
                self.match_available = false;
                self.match_length = MIN_MATCH - 1;
                self.strstart += 1;
                if bflush {
                    self.flush_block(false);
                }
            } else if self.match_available {
                let c = self.window[(self.strstart - 1) as usize];
                let bflush = self.tally_lit(c);
                if bflush {
                    self.flush_block(false);
                }
                self.strstart += 1;
                self.lookahead -= 1;
            } else {
                self.match_available = true;
                self.strstart += 1;
                self.lookahead -= 1;
            }
        }
        if self.match_available {
            let c = self.window[(self.strstart - 1) as usize];
            self.tally_lit(c);
            self.match_available = false;
        }
        self.insert = if self.strstart < MIN_MATCH - 1 { self.strstart } else { MIN_MATCH - 1 };
        self.flush_block(true);
    }

    // --- trees (stock zlib trees.c) ---
    fn pqdownheap(freq: &[u16], depth: &[u8], heap: &mut [usize], heap_len: usize, mut k: usize) {
        let v = heap[k];
        let mut j = k << 1;
        while j <= heap_len {
            if j < heap_len
                && (freq[heap[j + 1]] < freq[heap[j]]
                    || (freq[heap[j + 1]] == freq[heap[j]] && depth[heap[j + 1]] <= depth[heap[j]]))
            {
                j += 1;
            }
            if freq[v] < freq[heap[j]]
                || (freq[v] == freq[heap[j]] && depth[v] <= depth[heap[j]])
            {
                break;
            }
            heap[k] = heap[j];
            k = j;
            j <<= 1;
        }
        heap[k] = v;
    }

    /// Build one Huffman tree, returning (opt_len, static_len, max_code). `extra`/`extra_base`/`max_length`
    /// and the static tree come from the descriptor. Mirrors zlib `build_tree` + `gen_bitlen` + `gen_codes`.
    fn build_tree(
        freq: &mut [u16],
        dad: &mut [u16],
        len: &mut [u8],
        code: &mut [u16],
        elems: usize,
        extra: &[u8],
        extra_base: usize,
        max_length: usize,
        stree_len: Option<&[u8]>,
        bl_count: &mut [u16; MAX_BITS + 1],
    ) -> (usize, usize, usize) {
        let mut heap = [0usize; HEAP_SIZE];
        let mut depth = [0u8; HEAP_SIZE];
        let mut heap_len = 0usize;
        let mut heap_max = HEAP_SIZE;
        let mut max_code: i32 = -1; // -1 until a nonzero-freq symbol is seen (stock zlib)
        for n in 0..elems {
            if freq[n] != 0 {
                heap_len += 1;
                heap[heap_len] = n;
                max_code = n as i32;
                depth[n] = 0;
            } else {
                len[n] = 0;
            }
        }
        // zlib forces at least two codes: an all-zero tree gets symbols {0,1} (max_code -1 -> 1). Each
        // forced node is given Freq 1, so zlib pre-subtracts its contribution (`s->opt_len--`, and
        // `s->static_len -= stree[node].Len` when a static tree exists) to cancel what gen_bitlen will
        // add back for it. Accumulate that adjustment and apply it to the returned lengths — without it
        // an empty distance tree leaves static_len 10 bits high (2 forced codes × len 5), which flips
        // the stored-vs-fixed choice to `stored` for tiny blocks (`_tr_flush_block`).
        let mut forced_opt = 0usize;
        let mut forced_static = 0usize;
        while heap_len < 2 {
            let node = if max_code < 2 {
                max_code += 1;
                max_code as usize
            } else {
                0
            };
            heap_len += 1;
            heap[heap_len] = node;
            freq[node] = 1;
            depth[node] = 0;
            forced_opt += 1;
            if let Some(sl) = stree_len {
                forced_static += sl[node] as usize;
            }
        }
        for k in (1..=heap_len / 2).rev() {
            Deflate::pqdownheap(freq, &depth, &mut heap, heap_len, k);
        }
        let mut node = elems;
        let mut hl = heap_len;
        loop {
            let n = heap[1];
            heap[1] = heap[hl];
            hl -= 1;
            Deflate::pqdownheap(freq, &depth, &mut heap, hl, 1);
            let m = heap[1];
            heap_max -= 1;
            heap[heap_max] = n;
            heap_max -= 1;
            heap[heap_max] = m;
            freq[node] = freq[n] + freq[m];
            depth[node] = depth[n].max(depth[m]) + 1;
            dad[n] = node as u16;
            dad[m] = node as u16;
            heap[1] = node;
            node += 1;
            Deflate::pqdownheap(freq, &depth, &mut heap, hl, 1);
            if hl < 2 {
                break;
            }
        }
        heap_max -= 1;
        heap[heap_max] = heap[1];

        // gen_bitlen
        let mut bl = [0u16; MAX_BITS + 1];
        let root = heap[heap_max];
        len[root] = 0;
        let mut overflow = 0i32;
        let mut opt_len = 0usize;
        let mut static_len = 0usize;
        for h in (heap_max + 1)..HEAP_SIZE {
            let n = heap[h];
            let mut bits = len[dad[n] as usize] as usize + 1;
            if bits > max_length {
                bits = max_length;
                overflow += 1;
            }
            len[n] = bits as u8;
            if n as i32 > max_code {
                continue;
            }
            bl[bits] += 1;
            let xbits = if n >= extra_base { extra[n - extra_base] as usize } else { 0 };
            let f = freq[n] as usize;
            opt_len += f * (bits + xbits);
            if let Some(sl) = stree_len {
                static_len += f * (sl[n] as usize + xbits);
            }
        }
        if overflow > 0 {
            while overflow > 0 {
                let mut bits = max_length - 1;
                while bl[bits] == 0 {
                    bits -= 1;
                }
                bl[bits] -= 1;
                bl[bits + 1] += 2;
                bl[max_length] -= 1;
                overflow -= 2;
            }
            let mut h = HEAP_SIZE; // persists across bit lengths (stock zlib), do not reset per-bits
            for bits in (1..=max_length).rev() {
                let mut n = bl[bits];
                while n != 0 {
                    h -= 1;
                    let m = heap[h];
                    if m as i32 > max_code {
                        continue;
                    }
                    if len[m] as usize != bits {
                        opt_len += (bits as i64 - len[m] as i64) as usize * freq[m] as usize;
                        len[m] = bits as u8;
                    }
                    n -= 1;
                }
            }
        }
        // bl_count for gen_codes: count only real codes (len > 0); bl_count[0] must stay 0 so the
        // canonical-code base at bits=1 is 0 (stock zlib).
        for b in bl_count.iter_mut() {
            *b = 0;
        }
        let max_code = max_code as usize;
        for n in 0..=max_code {
            if len[n] != 0 {
                bl_count[len[n] as usize] += 1;
            }
        }
        // gen_codes
        let mut next = [0u16; MAX_BITS + 1];
        let mut c = 0u32;
        for bits in 1..=MAX_BITS {
            c = (c + bl_count[bits - 1] as u32) << 1;
            next[bits] = c as u16;
        }
        for n in 0..=max_code {
            let l = len[n] as usize;
            if l != 0 {
                let cc = next[l];
                next[l] += 1;
                code[n] = bi_reverse(cc as u32, l) as u16;
            }
        }
        (opt_len - forced_opt, static_len - forced_static, max_code)
    }

    fn scan_tree(bl_freq: &mut [u16], len: &[u8], max_code: usize) {
        let mut prevlen: i32 = -1;
        let mut nextlen = len[0];
        let mut count = 0;
        let (mut max_count, mut min_count) = (7, 4);
        if nextlen == 0 {
            max_count = 138;
            min_count = 3;
        }
        // len[max_code+1] sentinel
        let mut len_ext = len[..=max_code].to_vec();
        len_ext.push(0xff);
        for n in 0..=max_code {
            let curlen = nextlen;
            nextlen = len_ext[n + 1];
            count += 1;
            if count < max_count && curlen == nextlen {
                continue;
            } else if count < min_count {
                bl_freq[curlen as usize] += count;
            } else if curlen != 0 {
                if curlen as i32 != prevlen {
                    bl_freq[curlen as usize] += 1;
                }
                bl_freq[REP_3_6] += 1;
            } else if count <= 10 {
                bl_freq[REPZ_3_10] += 1;
            } else {
                bl_freq[REPZ_11_138] += 1;
            }
            count = 0;
            prevlen = curlen as i32;
            if nextlen == 0 {
                max_count = 138;
                min_count = 3;
            } else if curlen == nextlen {
                max_count = 6;
                min_count = 3;
            } else {
                max_count = 7;
                min_count = 4;
            }
        }
    }

    fn send_tree(&mut self, len: &[u8], max_code: usize) {
        let mut prevlen: i32 = -1;
        let mut nextlen = len[0];
        let mut count = 0;
        let (mut max_count, mut min_count) = (7, 4);
        if nextlen == 0 {
            max_count = 138;
            min_count = 3;
        }
        let mut len_ext = len[..=max_code].to_vec();
        len_ext.push(0xff);
        for n in 0..=max_code {
            let curlen = nextlen;
            nextlen = len_ext[n + 1];
            count += 1;
            if count < max_count && curlen == nextlen {
                continue;
            } else if count < min_count {
                for _ in 0..count {
                    self.send_bits(self.bl.code[curlen as usize] as u32, self.bl.len[curlen as usize] as u32);
                }
            } else if curlen != 0 {
                if curlen as i32 != prevlen {
                    self.send_bits(self.bl.code[curlen as usize] as u32, self.bl.len[curlen as usize] as u32);
                    count -= 1;
                }
                self.send_bits(self.bl.code[REP_3_6] as u32, self.bl.len[REP_3_6] as u32);
                self.send_bits((count - 3) as u32, 2);
            } else if count <= 10 {
                self.send_bits(self.bl.code[REPZ_3_10] as u32, self.bl.len[REPZ_3_10] as u32);
                self.send_bits((count - 3) as u32, 3);
            } else {
                self.send_bits(self.bl.code[REPZ_11_138] as u32, self.bl.len[REPZ_11_138] as u32);
                self.send_bits((count - 11) as u32, 7);
            }
            count = 0;
            prevlen = curlen as i32;
            if nextlen == 0 {
                max_count = 138;
                min_count = 3;
            } else if curlen == nextlen {
                max_count = 6;
                min_count = 3;
            } else {
                max_count = 7;
                min_count = 4;
            }
        }
    }

    fn tr_flush_block(&mut self, buf_start: i64, stored_len: u32, last: bool) {
        let mut opt_lenb;
        let static_lenb;
        let mut max_blindex = 0usize;
        if self.level > 0 {
            // build literal and distance trees
            let mut bl_count = [0u16; MAX_BITS + 1];
            let (lopt, lstat, lmax) = Deflate::build_tree(
                &mut self.l.freq,
                &mut self.l.dad,
                &mut self.l.len,
                &mut self.l.code,
                L_CODES,
                &EXTRA_LBITS,
                LITERALS + 1,
                MAX_BITS,
                Some(&self.st.ltree_len),
                &mut bl_count,
            );
            self.l_max_code = lmax;
            let (dopt, dstat, dmax) = Deflate::build_tree(
                &mut self.d.freq,
                &mut self.d.dad,
                &mut self.d.len,
                &mut self.d.code,
                D_CODES,
                &EXTRA_DBITS,
                0,
                MAX_BITS,
                Some(&self.st.dtree_len),
                &mut bl_count,
            );
            self.d_max_code = dmax;
            self.opt_len = lopt + dopt;
            self.static_len = lstat + dstat;

            // build_bl_tree
            for f in self.bl.freq.iter_mut() {
                *f = 0;
            }
            Deflate::scan_tree(&mut self.bl.freq, &self.l.len[..=self.l_max_code], self.l_max_code);
            Deflate::scan_tree(&mut self.bl.freq, &self.d.len[..=self.d_max_code], self.d_max_code);
            let mut bl_count2 = [0u16; MAX_BITS + 1];
            let (blopt, _, _) = Deflate::build_tree(
                &mut self.bl.freq,
                &mut self.bl.dad,
                &mut self.bl.len,
                &mut self.bl.code,
                BL_CODES,
                &EXTRA_BLBITS,
                0,
                MAX_BL_BITS,
                None,
                &mut bl_count2,
            );
            self.opt_len += blopt;
            max_blindex = BL_CODES - 1;
            while max_blindex >= 3 {
                if self.bl.len[BL_ORDER[max_blindex]] != 0 {
                    break;
                }
                max_blindex -= 1;
            }
            self.opt_len += 3 * (max_blindex + 1) + 5 + 5 + 4;

            opt_lenb = (self.opt_len + 3 + 7) >> 3;
            static_lenb = (self.static_len + 3 + 7) >> 3;
            if static_lenb <= opt_lenb {
                opt_lenb = static_lenb;
            }
        } else {
            opt_lenb = stored_len as usize + 5;
            static_lenb = opt_lenb;
        }

        if stored_len as usize + 4 <= opt_lenb && buf_start >= 0 {
            self.tr_stored_block(buf_start as usize, stored_len, last);
        } else if static_lenb == opt_lenb {
            self.send_bits(((1 << 1) | last as u32) as u32, 3);
            self.compress_block(true);
        } else {
            self.send_bits(((2 << 1) | last as u32) as u32, 3);
            self.send_all_trees(self.l_max_code + 1, self.d_max_code + 1, max_blindex + 1);
            self.compress_block(false);
        }
        self.init_block();
        if last {
            self.bi_windup();
        }
    }

    fn tr_stored_block(&mut self, buf_start: usize, stored_len: u32, last: bool) {
        self.send_bits(last as u32, 3); // BTYPE=00
        self.bi_windup();
        self.out.extend_from_slice(&(stored_len as u16).to_le_bytes());
        self.out.extend_from_slice(&(!stored_len as u16).to_le_bytes());
        self.out
            .extend_from_slice(&self.window[buf_start..buf_start + stored_len as usize]);
    }

    fn send_all_trees(&mut self, lcodes: usize, dcodes: usize, blcodes: usize) {
        self.send_bits((lcodes - 257) as u32, 5);
        self.send_bits((dcodes - 1) as u32, 5);
        self.send_bits((blcodes - 4) as u32, 4);
        for rank in 0..blcodes {
            let c = BL_ORDER[rank];
            self.send_bits(self.bl.len[c] as u32, 3);
        }
        let lmax = self.l_max_code;
        let dmax = self.d_max_code;
        let llen = self.l.len.clone();
        let dlen = self.d.len.clone();
        self.send_tree(&llen, lmax);
        self.send_tree(&dlen, dmax);
    }

    fn compress_block(&mut self, static_trees: bool) {
        let mut sx = 0usize;
        if self.sym_next != 0 {
            loop {
                let dist = self.sym[sx] as u32 | (self.sym[sx + 1] as u32) << 8;
                let lc = self.sym[sx + 2] as u32;
                sx += 3;
                if dist == 0 {
                    self.send_symbol(lc as usize, static_trees, false);
                } else {
                    let code = self.st.length_code[lc as usize] as usize;
                    self.send_symbol(code + LITERALS + 1, static_trees, false);
                    let extra = EXTRA_LBITS[code] as u32;
                    if extra != 0 {
                        self.send_bits(lc - self.st.base_length[code], extra);
                    }
                    let d = dist - 1;
                    let dcode = self.st.d_code(d);
                    self.send_symbol(dcode, static_trees, true);
                    let dextra = EXTRA_DBITS[dcode] as u32;
                    if dextra != 0 {
                        self.send_bits(d - self.st.base_dist[dcode], dextra);
                    }
                }
                if sx >= self.sym_next {
                    break;
                }
            }
        }
        self.send_symbol(END_BLOCK, static_trees, false);
    }

    fn send_symbol(&mut self, sym: usize, static_trees: bool, dist: bool) {
        let (code, len) = match (static_trees, dist) {
            (true, false) => (self.st.ltree_code[sym], self.st.ltree_len[sym]),
            (true, true) => (self.st.dtree_code[sym], self.st.dtree_len[sym]),
            (false, false) => (self.l.code[sym], self.l.len[sym]),
            (false, true) => (self.d.code[sym], self.d.len[sym]),
        };
        self.send_bits(code as u32, len as u32);
    }
}

/// Compress `data` into a full zlib stream (2-byte header + deflate + adler32), byte-identical to
/// WIC's bundled Chromium zlib at the given `level`/`window_bits`/`strategy` (1 = `Z_FILTERED`).
pub fn zlib_compress(data: &[u8], level: u32, window_bits: u32, strategy: u32) -> Vec<u8> {
    let st = Static::new();
    let mut d = Deflate::new(&st, data, level, window_bits, strategy);
    d.init_block();
    d.deflate_slow();
    // zlib header: CMF (method 8 | (windowBits-8)<<4), FLG (level flags + check).
    let cmf = 0x08 | ((window_bits - 8) << 4);
    let level_flags = if level < 2 {
        0
    } else if level < 6 {
        1
    } else if level == 6 {
        2
    } else {
        3
    };
    let mut header = (cmf << 8) | (level_flags << 6);
    header += 31 - (header % 31);
    let mut out = Vec::with_capacity(d.out.len() + 6);
    out.push((header >> 8) as u8);
    out.push(header as u8);
    out.extend_from_slice(&d.out);
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_2x2_matches_wic_idat() {
        // Filtered scanlines of WIC's 2x2 gray8 PNG (filter 0 both rows): recovered from the oracle.
        let data = [0x00u8, 0x00, 0x40, 0x00, 0x80, 0xff];
        let got = zlib_compress(&data, 4, 9, 1);
        let want =
            [0x18u8, 0x57, 0x63, 0x60, 0x70, 0x60, 0x68, 0xf8, 0x0f, 0x00, 0x03, 0x05, 0x01, 0xc0];
        assert_eq!(got, want, "got {got:02x?}");
    }

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    #[test]
    fn tiny_all_literal_block_is_fixed_not_stored() {
        // A tiny incompressible block: zlib picks a FIXED-Huffman block, not stored, because the
        // empty distance tree's 2 forced codes (`while heap_len < 2`) pre-subtract 10 bits from
        // static_len — without that, stored_len+4 <= opt_lenb flips true and we'd emit a stored block
        // 3 bytes longer. Filtered scanlines of native-r3 seed 100315 (Gray8 2x3); native IDAT exact.
        let data = [1u8, 168, 41, 1, 238, 221, 3, 247, 184];
        assert_eq!(zlib_compress(&data, 4, 9, 1), hex("1857635ca1c9f8ee2ef3f71d0011400451"));
    }

    #[test]
    fn bgr_2x2_matches_wic_idat() {
        // Sub + Paeth filtered rows of WIC's 2x2 RGB PNG (distance codes + extra bits).
        let data = [1u8, 30, 20, 10, 30, 30, 30, 4, 60, 60, 60, 30, 30, 30];
        assert_eq!(zlib_compress(&data, 4, 9, 1), hex("1857639413e192939363b1b1b191939303000a7e01aa"));
    }

    #[test]
    fn gray_8x8_matches_wic_idat() {
        // 8 rows (Sub then Up) with long runs: exercises match-finding and tree selection.
        let mut data = vec![1u8, 0, 8, 8, 8, 8, 8, 8, 8];
        for _ in 0..7 {
            data.extend_from_slice(&[2, 3, 3, 3, 3, 3, 3, 3, 3]);
        }
        assert_eq!(zlib_compress(&data, 4, 9, 1), hex("18576364e08000266628208f010025dc00f0"));
    }
}

