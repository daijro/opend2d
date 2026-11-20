//! The memory Direct2D's deferred image context spends on each DrawImage of an effect, and the flush it makes when
//! that memory passes its limit.
//!
//! The context keeps a batch's DrawImages in an arena of 1024-byte blocks: every allocation takes its size plus an
//! 8-byte header, and one that does not fit what is left of the current block starts a new block, the rest of the
//! old one wasted. A DrawImage first checks the arena: past 0x32000 bytes (counting whole blocks before the current
//! one) it flushes the target's queue, which renders the batch, frees every DrawImage's memory but the current
//! block's, and closes the atlas set. The arena empties when the target's batch ends (EndDraw, Flush, SetTarget).
//!
//! A DrawImage's own memory is its deferred operation (96 bytes), the render graph Direct2D builds for it, and,
//! when its intermediates are atlased, the atlas set's growth ([`AtlasState`]). The render graph comes from
//! resolving the transform nodes the effect graph becomes, root first: a root operation drawing into the target,
//! an operation for every input a node samples, an intermediate for a node an operation samples, and the arrays that
//! hold them ([`resolve`]).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;

use super::{ColorOp, Effect, Input, Kind, Rect};

/// Past this many bytes, a DrawImage flushes the batch before it queues.
pub(crate) const FLUSH_BYTES: u64 = 0x32000;

const OPERATION: u32 = 96 + 8;
const RENDER_OP: u32 = 245 + 8;
const INTERMEDIATE: u32 = 191 + 8;
const INPUT_RECORD: u32 = 12 + 8;
const BLEND_DESC: u32 = 40 + 8;

#[derive(Clone, Debug, Default)]
pub(crate) struct Arena {
    blocks: u64,
    used: u64,
    live: bool,
}

impl Arena {
    /// One allocation of `bytes`, its header included.
    pub(crate) fn alloc(&mut self, bytes: u32) {
        let bytes = bytes as u64;
        if !self.live || bytes > 1024 - self.used {
            self.blocks += 1;
            self.used = 0;
            self.live = true;
        }
        self.used += bytes;
    }

    /// What the DrawImage check reads: the blocks before the current one, whole, and the current one's use.
    pub(crate) fn measure(&self) -> u64 {
        if self.live && self.blocks > 0 {
            (self.blocks << 10) + self.used - 1024
        } else {
            self.used
        }
    }

    /// The target processed its queue: every DrawImage's memory is freed but the current block's.
    pub(crate) fn flush(&mut self) {
        if self.live {
            self.blocks = 1;
        }
    }

    /// The target's batch ended.
    pub(crate) fn reset(&mut self) {
        *self = Arena::default();
    }
}

/// The first growth of an intermediate's operation list.
const LIST_ADD: u32 = 8 + 8;
/// A shader-linking graph, one function node of it, and the pixel-shader record of the operation it draws.
const LINK_GRAPH: u32 = 208 + 8;
const LINK_NODE: u32 = 136 + 8;
const PS_DATA: u32 = 144 + 8;
/// The most shader functions one operation links.
const MAX_LINKED: u32 = 8;

/// Composite modes, as the blends operations draw into their targets with.
const SOURCE_OVER: u32 = 0;
const DEST_OVER: u32 = 1;
const PLUS: u32 = 9;
const COPY: u32 = 10;

/// Whether drawing with `mode` leaves the target alone where the source is transparent black.
fn keeps_under_transparent(mode: u32) -> bool {
    // SOURCE_OVER, DESTINATION_OVER, DESTINATION_OUT, SOURCE_ATOP, XOR, PLUS, MASK_INVERT.
    matches!(mode, 0 | 1 | 5 | 6 | 8 | 9 | 12)
}

/// Whether an operation drawing with `mode` keeps a blend description of its own: anything but SOURCE_OVER and a copy.
fn needs_desc(mode: u32) -> bool {
    !matches!(mode, SOURCE_OVER | COPY | 11)
}

/// Whether a draw folds into the border operation that samples it: always (an instruction-count ratio of at most a
/// quarter), never, or (a cost the planner does not know) when the border's rect is at most twice the area it asks
/// of the draw, the graph otherwise not modelled.
#[derive(Clone, Copy)]
enum Fold {
    Always,
    Never,
    Unknown,
}

/// A transform node of the render graph, as the resolver walks it.
#[derive(Clone)]
enum TKind {
    /// A pixel-shader draw. `copy`: Direct2D's copy shader without vertices of its own. `fold`: whether it folds into a
    /// border operation that samples it. `links`: its shader links into the operation that samples it.
    Draw { copy: bool, fold: Fold, links: bool },
    /// A blend of its inputs with a composite mode.
    Blend { mode: u32 },
    /// A border (extend modes) of its input.
    Border,
    /// A bitmap.
    Bitmap,
    /// A DrawImage's image brush, at a fractional position.
    Brush,
}

/// What an operation at a node asks of one of its inputs.
#[derive(Clone, Copy)]
enum Ask {
    /// The operation's own rect.
    Same,
    /// That rect (`None`: all of it).
    Rect(Option<Rect>),
    /// The operation's rect grown by left, top, right and bottom.
    Grow([i32; 4]),
}

#[derive(Clone)]
struct TNode {
    kind: TKind,
    inputs: Vec<usize>,
    /// Its output rect; `None` when unbounded.
    bounds: Option<Rect>,
    asks: Vec<Ask>,
    /// An effect's output node (not one inside an effect's own graph).
    output: bool,
}

fn within(r: Rect, b: Option<Rect>) -> bool {
    let empty = r.r <= r.l || r.b <= r.t;
    empty || b.is_none_or(|b| r.l >= b.l && r.t >= b.t && r.r <= b.r && r.b <= b.b)
}

/// `b`, or the whole plane when unbounded.
fn whole(b: Option<Rect>) -> Rect {
    b.unwrap_or(Rect { l: i32::MIN, t: i32::MIN, r: i32::MAX, b: i32::MAX })
}

fn area(r: Rect) -> i64 {
    (r.r as i64 - r.l as i64).max(0) * (r.b as i64 - r.t as i64).max(0)
}

/// The capacity an arena array grows to for `n` elements from `cap`: `max(n, 2·cap)`.
fn grown(n: u32, cap: u32) -> u32 {
    n.max(2 * cap)
}

/// The shader-linking graph of an operation: its function nodes, its unbound inputs (the operation's inputs), and
/// the capacities of the arrays holding them and its signature.
#[derive(Clone, Default)]
struct Link {
    nodes: u32,
    node_cap: u32,
    unbound: u32,
    unbound_cap: u32,
    sig_cap: u32,
}

impl Link {
    /// A function of `m` inputs added to the graph, in place of one unbound input unless it is the first; a later
    /// one re-signs the graph. Its allocations go to `out`.
    fn add(&mut self, m: u32, first: bool, out: &mut Vec<u32>) {
        out.push(LINK_NODE);
        if self.nodes == self.node_cap {
            self.node_cap = grown(self.nodes + 1, self.node_cap);
            out.push(8 * self.node_cap + 8);
        }
        self.nodes += 1;
        if m > 0 {
            // Its input-node and input-index arrays.
            out.extend([8 * m + 8, 4 * m + 8]);
        }
        if !first && self.unbound > 0 {
            self.unbound -= 1;
        }
        let need = self.unbound + m;
        if need > self.unbound_cap {
            // Unbound-input records of 40 bytes.
            self.unbound_cap = grown(need, self.unbound_cap);
            out.push(40 * self.unbound_cap + 8);
        }
        self.unbound = need;
        if !first {
            // A 16-byte record per node and unbound input.
            let s = self.nodes + self.unbound;
            if s > self.sig_cap {
                self.sig_cap = grown(s, self.sig_cap);
                out.push(16 * self.sig_cap + 8);
            }
        }
    }
}

/// An operation of the render graph being resolved.
#[derive(Clone)]
struct ROp {
    /// It draws into a target (the root's, an intermediate the resolver made for it, or a blend's).
    target: bool,
    /// The operation that samples what it renders, when it renders an input of another.
    parent: Option<usize>,
    rect: Option<Rect>,
    /// Its input-intermediate array.
    cap: u32,
    count: u32,
    /// A border operation.
    border: bool,
    /// It draws a pixel shader (its node's, or one folded into its border), and whether that shader links.
    shader: bool,
    linkable: bool,
    /// The blend it draws into its target with.
    blend: u32,
    /// Inputs given an intermediate.
    set: u32,
    link: Option<Link>,
    /// The atlas depth of its target (the draw's own is 0), and the intermediate it made for its node.
    depth: u32,
    own: Option<usize>,
}

impl ROp {
    fn new(target: bool, parent: Option<usize>, rect: Option<Rect>, blend: u32) -> ROp {
        ROp { target, parent, rect, cap: 0, count: 0, border: false, shader: false, linkable: false, blend, set: 0, link: None, depth: 0, own: None }
    }
}

/// An intermediate the resolver made: its rect, atlas depth, and whether it is a texture to atlas (not a bitmap's).
struct Inter {
    rect: Rect,
    depth: u32,
    drawn: bool,
}

struct Resolver {
    nodes: Vec<TNode>,
    ops: Vec<ROp>,
    pending: Vec<Vec<usize>>,
    out: Vec<u32>,
    inters: Vec<Inter>,
    /// The atlas depths an intermediate's operations read: (the reading target's depth, the read one's).
    reads: Vec<(u32, u32)>,
}

impl Resolver {
    /// Grow `op`'s input-intermediate array to hold `n`.
    fn grow_inputs(&mut self, op: usize, n: u32) {
        let o = &mut self.ops[op];
        if n > o.cap {
            o.cap = grown(n, o.cap);
            self.out.push(16 * o.cap + 8);
        }
        o.count = n;
    }

    /// One operation per input of `node`, each queued on that input, sampling it for `op`.
    fn queue_inputs(&mut self, op: usize, node: usize, n: usize) {
        let rect = self.ops[op].rect;
        for i in 0..n.min(self.nodes[node].inputs.len()) {
            self.out.push(RENDER_OP);
            let ask = match self.nodes[node].asks[i] {
                Ask::Same => rect,
                Ask::Rect(r) => r,
                Ask::Grow(g) => rect.map(|r| Rect { l: r.l - g[0], t: r.t - g[1], r: r.r + g[2], b: r.b + g[3] }),
            };
            self.ops.push(ROp::new(false, Some(op), ask, COPY));
            let input = self.nodes[node].inputs[i];
            let new = self.ops.len() - 1;
            self.pending[input].push(new);
        }
    }

    /// `op` renders `node` into its target.
    fn with_target(&mut self, op: usize, node: usize) -> Option<()> {
        let n = self.nodes[node].inputs.len();
        match self.nodes[node].kind.clone() {
            TKind::Draw { copy, links, .. } => {
                let rect = self.ops[op].rect.unwrap_or(Rect { l: 0, t: 0, r: 0, b: 0 });
                if !within(rect, self.nodes[node].bounds) && !keeps_under_transparent(self.ops[op].blend) {
                    // A clear before it: the operation is cloned (its inputs with it) and a clear made.
                    self.out.push(RENDER_OP);
                    let c = self.ops[op].count;
                    if c > 0 {
                        self.out.push(16 * c + 8);
                    }
                    self.out.push(RENDER_OP);
                }
                if copy && n > 0 {
                    let input = self.nodes[node].inputs[0];
                    self.pending[input].push(op);
                    return Some(());
                }
                let o = &mut self.ops[op];
                (o.shader, o.linkable) = (true, links);
                if n > 0 {
                    self.grow_inputs(op, n as u32);
                }
                self.queue_inputs(op, node, n);
            }
            TKind::Border => {
                self.ops[op].border = true;
                self.grow_inputs(op, 1);
                self.queue_inputs(op, node, 1);
            }
            TKind::Brush => {
                self.grow_inputs(op, 1);
                self.queue_inputs(op, node, 1);
            }
            TKind::Bitmap => {
                // Copied from the bitmap: a wrapper of it, then the operation's input.
                self.out.push(INTERMEDIATE);
                self.grow_inputs(op, 1);
            }
            TKind::Blend { mode } => {
                // An operation per input drawing it straight into the target. The composite's mode, reordered to
                // combine with the target's blend unless that is a copy or the same mode, draws every input but
                // one, which takes the target's blend. The inputs are made last first, each put ahead of the
                // others in the target's list, except under DESTINATION_OVER (first first, each put after) or a
                // reordered mode (the other way about); the target's blend goes to the last made, or under
                // DESTINATION_OVER the first.
                let parent = self.ops[op].blend;
                let (later, _) = super::plan::distribute(parent, mode)?;
                let reordered = !matches!(parent, COPY | 11) && !(matches!(parent, SOURCE_OVER | DEST_OVER | PLUS) && mode == parent);
                let dest_over = later == DEST_OVER;
                let ins = self.nodes[node].inputs.clone();
                let (rect, depth) = (self.ops[op].rect, self.ops[op].depth);
                for k in 0..n {
                    let i = if dest_over == reordered { ins[n - 1 - k] } else { ins[k] };
                    let takes_target = if dest_over { k == 0 } else { k == n - 1 };
                    let blend = if takes_target { parent } else { later };
                    let r = rect.map(|r| if takes_target && !keeps_under_transparent(blend) { r } else { r.intersect(whole(self.nodes[i].bounds)) });
                    if r.is_none_or(|r| area(r) == 0) {
                        continue;
                    }
                    self.out.push(RENDER_OP);
                    if needs_desc(blend) {
                        self.out.push(BLEND_DESC);
                    }
                    let mut o = ROp::new(true, None, r, blend);
                    o.depth = depth;
                    self.ops.push(o);
                    let new = self.ops.len() - 1;
                    self.pending[i].push(new);
                }
            }
        }
        Some(())
    }

    /// The draw `node` links into the operation `p` that samples it: its function joins `p`'s linking graph (made
    /// with `p`'s own function first), and its inputs become `p`'s.
    fn link(&mut self, p: usize, node: usize) {
        let m = self.nodes[node].inputs.len();
        let first = self.ops[p].link.is_none();
        let mut out = Vec::new();
        let mut l = self.ops[p].link.take().unwrap_or_default();
        if first {
            out.push(LINK_GRAPH);
            l.add(self.ops[p].count, true, &mut out);
        }
        l.add(m as u32, false, &mut out);
        if first {
            out.push(PS_DATA);
        }
        // Its constant buffers, one per function, gathered afresh.
        out.push(8 * l.nodes + 8);
        let o = &mut self.ops[p];
        o.count = l.unbound;
        o.link = Some(l);
        self.out.extend(out);
        self.queue_inputs(p, node, m);
    }

    fn resolve_node(&mut self, n: usize) -> Option<()> {
        let list = std::mem::take(&mut self.pending[n]);
        let bounds = self.nodes[n].bounds;
        let kind = self.nodes[n].kind.clone();
        let mut own: Option<usize> = None;
        let mut sampled: Vec<usize> = Vec::new();
        let mut direct: Vec<usize> = Vec::new();
        for &op in &list {
            let Some(rect) = self.ops[op].rect else { continue };
            if area(rect) == 0 {
                continue;
            }
            // Handed to the input that covers what it asks.
            let first = self.nodes[n].inputs.first().copied();
            match kind {
                // A copy passes it on when, on every side, it stays within the copy's bounds or those bounds are
                // the input's.
                TKind::Draw { copy: true, .. } => {
                    if let Some(i) = first {
                        let (b, ib) = (whole(bounds), whole(self.nodes[i].bounds));
                        if (rect.l >= b.l || b.l == ib.l) && (rect.t >= b.t || b.t == ib.t) && (rect.r <= b.r || b.r == ib.r) && (rect.b <= b.b || b.b == ib.b) {
                            self.pending[i].push(op);
                            continue;
                        }
                    }
                }
                // A border passes it on when it asks for no more than the input covers.
                TKind::Border => {
                    if let Some(i) = first {
                        if within(rect, self.nodes[i].bounds) {
                            self.pending[i].push(op);
                            continue;
                        }
                    }
                }
                // A blend passes it to the one input it overlaps.
                TKind::Blend { .. } => {
                    let over: Vec<usize> = self.nodes[n].inputs.iter().copied().filter(|&i| {
                        let c = whole(self.nodes[i].bounds).intersect(rect);
                        c.width() > 0 && c.height() > 0
                    }).collect();
                    if let [i] = over[..] {
                        self.pending[i].push(op);
                        continue;
                    }
                }
                _ => {}
            }
            if let Some(p) = self.ops[op].parent {
                let po = self.ops[p].clone();
                // A linkable shader links into the shader operation sampling it, while none of that operation's
                // inputs has an intermediate yet.
                if let TKind::Draw { copy: false, links: true, .. } = kind {
                    if po.shader && po.linkable && po.set == 0 && within(rect, bounds) && po.link.as_ref().map_or(1, |l| l.nodes) < MAX_LINKED {
                        self.link(p, n);
                        continue;
                    }
                }
                // A shader folds into the border operation that samples it: that operation draws it.
                if let TKind::Draw { fold, links, .. } = kind {
                    if po.border && !po.shader && within(rect, bounds) {
                        let ratio = po.rect.is_some_and(|pr| area(rect) > 0 && area(pr) / area(rect) <= 2);
                        let folds = match fold {
                            Fold::Always => true,
                            Fold::Never => false,
                            Fold::Unknown if ratio => true,
                            Fold::Unknown => return None,
                        };
                        if folds {
                            let ins = self.nodes[n].inputs.len();
                            self.grow_inputs(p, ins as u32);
                            let o = &mut self.ops[p];
                            (o.shader, o.linkable) = (true, links);
                            self.queue_inputs(p, n, ins);
                            continue;
                        }
                    }
                }
                // A bitmap the border operation samples becomes that operation's intermediate, by texture addressing,
                // when the operation made that intermediate and asks for the bitmap's rect.
                if let (TKind::Bitmap, Some(k)) = (&kind, po.own) {
                    if po.border && !po.shader && bounds.is_some_and(|b| (rect.l, rect.t, rect.r, rect.b) == (b.l, b.t, b.r, b.b)) {
                        self.inters[k].drawn = false;
                        continue;
                    }
                }
            }
            if self.ops[op].target {
                // A composite that does not combine with the target's blend is drawn into an intermediate of its own
                // first, which the operation then draws.
                if let TKind::Blend { mode } = kind {
                    if super::plan::distribute(self.ops[op].blend, mode).is_none() {
                        self.out.push(RENDER_OP);
                        let mut o = ROp::new(true, None, Some(rect), COPY);
                        o.depth = self.ops[op].depth + 1;
                        o.own = Some(self.inters.len());
                        self.inters.push(Inter { rect, depth: o.depth, drawn: true });
                        self.reads.push((self.ops[op].depth, o.depth));
                        self.out.push(INTERMEDIATE);
                        let c = self.ops[op].count;
                        self.grow_inputs(op, c + 1);
                        self.out.push(LIST_ADD);
                        self.ops.push(o);
                        direct.push(self.ops.len() - 1);
                        continue;
                    }
                }
                direct.push(op);
                continue;
            }
            match kind {
                // A bitmap sampled by an operation: a wrapper of it as its input.
                TKind::Bitmap => {
                    self.out.push(INTERMEDIATE);
                    if let Some(p) = self.ops[op].parent {
                        self.ops[p].set += 1;
                    }
                }
                _ => {
                    own.get_or_insert(op);
                    sampled.push(op);
                }
            }
        }
        for op in direct {
            self.with_target(op, n)?;
        }
        if let Some(op) = own {
            // The node's own intermediate, an input record for every operation that samples it, the intermediate's
            // operation list, and the node drawn there by the first of them. It sits one atlas below the deepest
            // target that samples it, over what they ask.
            self.out.push(INTERMEDIATE);
            let mut depth = 0;
            let mut rect = self.ops[op].rect?;
            for &s in &sampled {
                if let Some(p) = self.ops[s].parent {
                    self.out.push(INPUT_RECORD);
                    self.ops[p].set += 1;
                    depth = depth.max(self.ops[p].depth + 1);
                }
                if let Some(r) = self.ops[s].rect {
                    rect = Rect { l: rect.l.min(r.l), t: rect.t.min(r.t), r: rect.r.max(r.r), b: rect.b.max(r.b) };
                }
            }
            for &s in &sampled {
                if let Some(p) = self.ops[s].parent {
                    self.reads.push((self.ops[p].depth, depth));
                }
            }
            self.out.push(LIST_ADD);
            let k = self.inters.len();
            self.inters.push(Inter { rect, depth, drawn: true });
            let o = &mut self.ops[op];
            (o.target, o.blend, o.depth, o.own) = (true, COPY, depth, Some(k));
            self.with_target(op, n)?;
        }
        Some(())
    }
}

/// The order Direct2D resolves `nodes` in, from `root`: depth-first discovery, then every node once all its
/// consumers are resolved, a ready bitmap or effect output after every node waiting and a node inside an effect's
/// graph before them.
fn order(nodes: &[TNode], root: usize) -> Vec<usize> {
    let mut consumers = vec![0u32; nodes.len()];
    let mut seen = vec![false; nodes.len()];
    let mut stack = vec![root];
    seen[root] = true;
    while let Some(n) = stack.pop() {
        for &i in &nodes[n].inputs {
            consumers[i] += 1;
            if !seen[i] {
                seen[i] = true;
                stack.push(i);
            }
        }
    }
    let mut done = vec![0u32; nodes.len()];
    let mut queue = std::collections::VecDeque::from([root]);
    let mut out = Vec::new();
    while let Some(n) = queue.pop_front() {
        out.push(n);
        for &i in &nodes[n].inputs {
            done[i] += 1;
            if done[i] == consumers[i] {
                if nodes[i].output || matches!(nodes[i].kind, TKind::Bitmap) {
                    queue.push_back(i);
                } else {
                    queue.push_front(i);
                }
            }
        }
    }
    out
}

/// A ConvolveMatrix that passes its input through: an odd kernel of a 1 at its centre and 0s, divisor 1, no bias or
/// offset, alpha not preserved.
fn convolve_identity(e: &Effect) -> bool {
    let (w, h) = (e.get_u32(2, 3), e.get_u32(3, 3));
    let k = e.get_floats(4, &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    let n = (w * h) as usize;
    n % 2 == 1
        && k.len() == n
        && e.get_f32(5, 1.0) == 1.0
        && e.get_f32(6, 0.0) == 0.0
        && !e.get_bool(8, false)
        && e.get_v2(7, [0.0, 0.0]) == [0.0, 0.0]
        && k.iter().enumerate().all(|(i, &v)| v == if i == n / 2 { 1.0 } else { 0.0 })
}

struct Builder {
    nodes: Vec<TNode>,
    by_ptr: HashMap<usize, usize>,
    bitmaps: Vec<(Rc<super::Bitmap>, usize)>,
}

impl Builder {
    fn push(&mut self, node: TNode) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    fn input(&mut self, i: &Input, need: Rect) -> Option<usize> {
        match i {
            Input::Bitmap(b) => {
                // One node per image. Setting a bitmap as an input takes a copy of it, so the same bitmap reaching two
                // inputs is told by its pixels. ponytail: two distinct bitmaps of the same pixels share a node.
                let same = |c: &super::Bitmap| (c.width, c.height, c.format) == (b.width, b.height, b.format) && c.premul_bgra == b.premul_bgra;
                if let Some(&(_, n)) = self.bitmaps.iter().find(|(c, _)| Rc::ptr_eq(c, b) || same(c)) {
                    return Some(n);
                }
                let bounds = Rect { l: 0, t: 0, r: b.width as i32, b: b.height as i32 };
                let n = self.push(TNode { kind: TKind::Bitmap, inputs: Vec::new(), bounds: Some(bounds), asks: Vec::new(), output: false });
                self.bitmaps.push((b.clone(), n));
                Some(n)
            }
            Input::List(_) => None,
            Input::Effect(e) => self.effect(e, need),
        }
    }

    /// A one-node effect of `kind` over `inputs`, each asked for what the operation asks.
    fn draw(&mut self, kind: TKind, inputs: &[Input], need: Rect, bounds: Option<Rect>) -> Option<usize> {
        let ins = inputs.iter().map(|i| self.input(i, need)).collect::<Option<Vec<_>>>()?;
        let asks = vec![Ask::Same; ins.len()];
        Some(self.push(TNode { kind, inputs: ins, bounds, asks, output: false }))
    }

    /// The transform nodes of `e` asked for `need`, its output node returned.
    fn effect(&mut self, e: &Effect, need: Rect) -> Option<usize> {
        let key = Rc::as_ptr(&e.0) as usize;
        if let Some(&n) = self.by_ptr.get(&key) {
            return Some(n);
        }
        let (kind, inputs) = {
            let d = e.0.borrow();
            (d.kind, d.inputs.clone())
        };
        let b = e.bounds();
        let need = need.intersect(b);
        let cheap = TKind::Draw { copy: false, fold: Fold::Always, links: true };
        let n = match kind {
            Kind::ColorMatrix if inputs.len() == 1 => self.draw(cheap, &inputs, need, Some(b))?,
            Kind::Color(ColorOp::Premultiply | ColorOp::UnPremultiply | ColorOp::Grayscale | ColorOp::HueRotation | ColorOp::RgbToHue)
                if inputs.len() == 1 =>
            {
                self.draw(cheap, &inputs, need, Some(b))?
            }
            Kind::Arithmetic | Kind::CrossFade if inputs.len() == 2 => {
                self.draw(TKind::Draw { copy: false, fold: Fold::Unknown, links: true }, &inputs, need, Some(b))?
            }
            Kind::Crop if inputs.len() == 1 => self.draw(TKind::Draw { copy: true, fold: Fold::Always, links: false }, &inputs, need, Some(b))?,
            Kind::Composite if !inputs.is_empty() => self.draw(TKind::Blend { mode: e.get_u32(0, 0) }, &inputs, need, Some(b))?,
            // A Flood is a shader of no inputs, which links into the operation sampling it.
            Kind::Flood if inputs.is_empty() => self.push(TNode { kind: TKind::Draw { copy: false, fold: Fold::Unknown, links: true }, inputs: Vec::new(), bounds: Some(b), asks: Vec::new(), output: false }),
            // A Border reads its input wherever its extend modes map what it is asked for: within the input.
            Kind::Border if inputs.len() == 1 => {
                let i = self.input(&inputs[0], need)?;
                let ib = self.nodes[i].bounds;
                let ask = ib.map(|ib| need.intersect(ib));
                self.push(TNode { kind: TKind::Border, inputs: vec![i], bounds: None, asks: vec![Ask::Rect(ask)], output: false })
            }
            Kind::ConvolveMatrix if inputs.len() == 1 && convolve_identity(e) => self.input(&inputs[0], need)?,
            // A ConvolveMatrix of the default unit length, soft border and unclamped colour is one draw asking its
            // input for its rect grown by the kernel's reach, which a drawn input's intermediate clears around.
            // ponytail: the other forms (a pre-scale, unpremultiply or post-colour pass) and kernels past 3x3 are not
            // traced.
            Kind::ConvolveMatrix
                if inputs.len() == 1
                    && e.get_u32(2, 3) <= 3
                    && e.get_u32(3, 3) <= 3
                    && e.get_v2(0, [1.0, 1.0]) == [1.0, 1.0]
                    && e.get_u32(9, 0) == 0
                    && !e.get_bool(8, false)
                    && !e.get_bool(10, false) =>
            {
                let i = self.input(&inputs[0], need)?;
                let ib = whole(self.nodes[i].bounds);
                let grow = [ib.l - b.l, ib.t - b.t, b.r - ib.r, b.b - ib.b].map(|v| v.max(0));
                self.push(TNode { kind: TKind::Draw { copy: false, fold: Fold::Unknown, links: false }, inputs: vec![i], bounds: Some(b), asks: vec![Ask::Grow(grow)], output: false })
            }
            // A HARD blur of no radius passes its input through.
            Kind::GaussianBlur if inputs.len() == 1 && e.get_u32(2, 0) == 1 && e.sigma() <= 0.0 => {
                self.draw(TKind::Draw { copy: true, fold: Fold::Never, links: false }, &inputs, need, Some(b))?
            }
            Kind::GaussianBlur if inputs.len() == 1 && e.get_u32(2, 0) == 1 => self.hard_blur(e, &inputs[0], need)?,
            _ => return None,
        };
        self.by_ptr.insert(key, n);
        self.nodes[n].output = true;
        Some(n)
    }

    /// A HARD GaussianBlur as Direct2D builds it: a clip to its input's bounds over the post-upscale (when it
    /// downscales), the Y and X kernel passes (each a blend of one kernel), a passthrough, a mirror border, and, when
    /// it pre-downscales, the shrink (reading a clamp border of its input) under that border.
    fn hard_blur(&mut self, e: &Effect, input: &Input, need: Rect) -> Option<usize> {
        use super::builtin::filters_hp::{hard_rects, hard_shrunk};
        let (sigma, mode) = (e.sigma(), e.get_u32(1, 1));
        let ib = super::input_bounds(input);
        let arr = |r: Rect| [r.l, r.t, r.r, r.b];
        let rect = |a: [i32; 4]| Rect { l: a[0], t: a[1], r: a[2], b: a[3] };
        // The rects with the input realized, so the passes' reads are known whatever the input is.
        let r = hard_rects(arr(ib), arr(need), sigma, mode, false)?;
        let shrunk = hard_shrunk(arr(ib), sigma, mode).map(rect);
        let empty = Ask::Rect(Some(Rect { l: 0, t: 0, r: 0, b: 0 }));
        // What the X pass reads of the passthrough under it.
        let x_reads = rect(r.pre.or(r.input)?);
        let inner = match shrunk {
            Some(j) => {
                let i = self.input(input, rect(r.input?))?;
                let clamp = self.push(TNode { kind: TKind::Border, inputs: vec![i], bounds: None, asks: vec![Ask::Rect(Some(ib))], output: false });
                let fold = if r.shrink.is_some() { Fold::Never } else { Fold::Always };
                let shrink = self.push(TNode { kind: TKind::Draw { copy: false, fold, links: false }, inputs: vec![clamp, i], bounds: Some(j), asks: vec![Ask::Rect(Some(rect(r.input?))), empty], output: false });
                self.push(TNode { kind: TKind::Border, inputs: vec![shrink], bounds: None, asks: vec![Ask::Rect(Some(j))], output: false })
            }
            None => {
                let i = self.input(input, x_reads.intersect(ib))?;
                self.push(TNode { kind: TKind::Border, inputs: vec![i], bounds: None, asks: vec![Ask::Rect(Some(ib))], output: false })
            }
        };
        let pass = self.push(TNode { kind: TKind::Draw { copy: true, fold: Fold::Never, links: false }, inputs: vec![inner], bounds: None, asks: vec![Ask::Same], output: false });
        let kx = self.push(TNode { kind: TKind::Draw { copy: false, fold: Fold::Never, links: false }, inputs: vec![pass], bounds: None, asks: vec![Ask::Rect(Some(x_reads))], output: false });
        let bx = self.push(TNode { kind: TKind::Blend { mode: SOURCE_OVER }, inputs: vec![kx], bounds: None, asks: vec![Ask::Same], output: false });
        let ky = self.push(TNode { kind: TKind::Draw { copy: false, fold: Fold::Never, links: false }, inputs: vec![bx], bounds: None, asks: vec![Ask::Rect(Some(rect(r.x)))], output: false });
        let by = self.push(TNode { kind: TKind::Blend { mode: SOURCE_OVER }, inputs: vec![ky], bounds: None, asks: vec![Ask::Same], output: false });
        let top = match r.y {
            Some(y) => self.push(TNode { kind: TKind::Draw { copy: false, fold: Fold::Never, links: false }, inputs: vec![by], bounds: None, asks: vec![Ask::Rect(Some(rect(y)))], output: false }),
            None => by,
        };
        let i = self.input(input, need)?;
        Some(self.push(TNode { kind: TKind::Draw { copy: true, fold: Fold::Never, links: false }, inputs: vec![top, i], bounds: Some(ib), asks: vec![Ask::Same, empty], output: false }))
    }
}

/// A DrawImage's render graph: its allocations in the order Direct2D makes them (the deferred operation and the atlas
/// set's growth aside), and the atlases its intermediates go into, each with the depths of the atlases its
/// intermediates read (`None` when they are not atlased: there are none, or one is larger than an atlas).
pub(super) struct Graph {
    pub(super) allocs: Vec<u32>,
    pub(super) atlases: Option<BTreeMap<u32, BTreeSet<u32>>>,
}

/// The render graph of a DrawImage of `root` showing `need` (the effect's space) with composite mode `composite`,
/// drawn by an image brush asking for `brush` at a fractional position; `None` for a graph not modelled yet.
pub(super) fn graph(root: &Effect, need: Rect, composite: u32, brush: Option<Rect>) -> Option<Graph> {
    let mut b = Builder { nodes: Vec::new(), by_ptr: HashMap::new(), bitmaps: Vec::new() };
    let out = b.effect(root, need)?;
    let top = match brush {
        Some(r) => b.push(TNode { kind: TKind::Brush, inputs: vec![out], bounds: None, asks: vec![Ask::Rect(Some(r))], output: false }),
        None => out,
    };
    let nodes = b.nodes;
    let mut r = Resolver { pending: vec![Vec::new(); nodes.len()], nodes, ops: Vec::new(), out: Vec::new(), inters: Vec::new(), reads: Vec::new() };
    // The root intermediate (the target), its operation, the blend it draws with when that is neither SOURCE_OVER
    // nor a copy, and the target's operation list.
    r.out.push(INTERMEDIATE);
    r.out.push(RENDER_OP);
    if needs_desc(composite) {
        r.out.push(BLEND_DESC);
    }
    r.out.push(LIST_ADD);
    r.ops.push(ROp::new(true, None, Some(need), composite));
    r.pending[top].push(0);
    for n in order(&r.nodes, top) {
        r.resolve_node(n)?;
    }
    let drawn: Vec<&Inter> = r.inters.iter().filter(|i| i.drawn).collect();
    let largest = drawn.iter().map(|i| i.rect.width().max(i.rect.height())).max().unwrap_or(0);
    let atlases = (1..=ATLAS_SIZE as u32).contains(&largest).then(|| {
        let mut m: BTreeMap<u32, BTreeSet<u32>> = drawn.iter().map(|i| (i.depth, BTreeSet::new())).collect();
        for &(from, to) in &r.reads {
            if m.contains_key(&to) {
                if let Some(e) = m.get_mut(&from) {
                    e.insert(to);
                }
            }
        }
        m
    });
    Some(Graph { allocs: r.out, atlases })
}

/// The deferred operation of a DrawImage.
pub(crate) const DRAW_OPERATION: u32 = OPERATION;

const ATLAS_SET: u32 = 72 + 8;
/// An atlas's side.
const ATLAS_SIZE: i32 = 512;
const ATLAS: u32 = 88 + 8;

/// One atlas of a set: the depths of the atlases its intermediates read, that list's capacity, and the capacity of
/// its operation's input array (`None` before the operation is made).
#[derive(Clone, Debug, Default)]
struct AtlasRecord {
    reads: Vec<u32>,
    reads_cap: u32,
    op: Option<u32>,
}

/// The arena memory an atlas set holds: the set, its array of atlases (one per depth), and each atlas, its list of
/// the atlases it reads, its target and the operation that draws it.
#[derive(Clone, Debug, Default)]
pub(crate) struct AtlasMem {
    open: bool,
    cap: u32,
    atlases: std::collections::BTreeMap<u32, AtlasRecord>,
}

impl AtlasMem {
    /// A graph's intermediates go into the set's atlas at each depth of `reads` (with the depths that atlas's
    /// intermediates read): the allocations that makes, in order. The set first; the atlas array grown to the deepest
    /// depth; then, deepest first, each atlas new to the set and each read new to its list; then, deepest first,
    /// each new atlas's target and operation, the operation's inputs the atlases it reads.
    pub(crate) fn add(&mut self, reads: &BTreeMap<u32, BTreeSet<u32>>) -> Vec<u32> {
        let mut out = Vec::new();
        if !self.open {
            self.open = true;
            out.push(ATLAS_SET);
        }
        if let Some(&deepest) = reads.keys().next_back() {
            if deepest + 1 > self.cap {
                self.cap = (deepest + 1).max(2 * self.cap);
                out.push(8 * self.cap + 8);
            }
        }
        for (&d, r) in reads.iter().rev() {
            let a = self.atlases.entry(d).or_insert_with(|| {
                out.push(ATLAS);
                AtlasRecord::default()
            });
            for &e in r {
                if !a.reads.contains(&e) {
                    let n = a.reads.len() as u32;
                    if n == a.reads_cap {
                        // ponytail: an array's growth (max(n + 1, 2n)) is read off its other arrays; only one read per
                        // atlas has been traced.
                        a.reads_cap = (n + 1).max(2 * n);
                        out.push(4 * a.reads_cap + 8);
                    }
                    a.reads.push(e);
                }
            }
        }
        for &d in reads.keys().rev() {
            let a = self.atlases.get_mut(&d).unwrap();
            let n = a.reads.len() as u32;
            match a.op {
                None => {
                    out.extend([INTERMEDIATE, RENDER_OP]);
                    if n > 0 {
                        out.push(16 * n + 8);
                    }
                    a.op = Some(n);
                }
                Some(c) if n > c => {
                    let c = n.max(2 * c);
                    out.push(16 * c + 8);
                    a.op = Some(c);
                }
                Some(_) => {}
            }
        }
        out
    }
}

/// `D2D_ARENA_TRACE` set: each DrawImage of an effect prints the arena at its entry and what it took.
pub(crate) fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("D2D_ARENA_TRACE").is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The arena at each DrawImage entry for a sequence of graphs' allocations.
    fn measures(graphs: &[Vec<u32>]) -> Vec<u64> {
        let mut a = Arena::default();
        graphs.iter().map(|g| {
            let m = a.measure();
            g.iter().for_each(|&b| a.alloc(b));
            m
        }).collect()
    }

    #[test]
    fn block_packing_matches_native_traces() {
        // Four DrawImages of a ColorMatrix of a bitmap (native: 0, 1223, 2500, 3548).
        let cm = vec![OPERATION, INTERMEDIATE, RENDER_OP, LIST_ADD, 24, RENDER_OP, INTERMEDIATE];
        assert_eq!(measures(&[cm.clone(), cm.clone(), cm.clone(), cm]), [0, 1223, 2500, 3548]);
    }

    #[test]
    fn linking_allocations_match_native_traces() {
        // A ColorMatrix linked into the ColorMatrix drawing it, and into a two-input Arithmetic draw.
        for (n0, want) in [
            (1, vec![216, 144, 16, 16, 12, 48, 144, 24, 16, 12, 56, 152, 24]),
            (2, vec![216, 144, 16, 24, 16, 88, 144, 24, 16, 12, 72, 152, 24]),
        ] {
            let mut r = Resolver { nodes: vec![TNode { kind: TKind::Bitmap, inputs: vec![], bounds: None, asks: vec![], output: false }, TNode { kind: TKind::Draw { copy: false, fold: Fold::Always, links: true }, inputs: vec![0], bounds: None, asks: vec![Ask::Same], output: true }], ops: vec![], pending: vec![vec![], vec![]], out: vec![], inters: vec![], reads: vec![] };
            let mut p = ROp::new(true, None, None, SOURCE_OVER);
            (p.shader, p.linkable, p.count) = (true, true, n0);
            r.ops.push(p);
            r.link(0, 1);
            // The linked function's input is queued as one more operation.
            assert_eq!(r.out, [want, vec![RENDER_OP]].concat());
        }
    }
}
