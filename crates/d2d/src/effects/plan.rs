//! Direct2D's intermediate planning for one DrawImage: the render operations an effect graph becomes, the
//! intermediate each one writes, and where the atlas allocator puts each intermediate.
//!
//! The engine needs only the placements: a HARD blur's passes sample their intermediates through texcoords
//! that include where those sit in their atlas ([`BlurPlace`]). For the effects it models, the planner builds
//! the operations as Direct2D does:
//! - a pixel-shader draw is one operation;
//! - a linkable draw folds into a linkable consumer's operation (up to 8 functions);
//! - a cheap draw (or one whose border op is at most twice its area) folds into a blur's realized-input
//!   border operation;
//! - a Composite draws each input, in order, into one intermediate, or straight into its target when its mode
//!   combines with the target's blend ([`distribute`]);
//! - a blur is its chain of passes, with the last one drawing straight into the intermediate its consumer
//!   reads (or into the Composite it feeds, or inside the border of a blur that reads it), and a pre-downscale
//!   whose mirrored apron spans more than 25 copies of the shrunk image split into the shrink and the apron;
//! - a Flood feeding a draw is a constant, not an intermediate.
//!
//! Then it runs the allocator:
//! - an intermediate's depth is one more than its deepest consumer's;
//! - the operations are visited from the output with every intermediate's operations pushed once all its
//!   consumers are visited, pending intermediates first-in first-out, and with no memory budget, so the
//!   pending ones go first whenever an operation would open an intermediate not yet visited;
//! - each depth's 512×512 atlas is packed in reverse visit order, left to right on shelves that start a
//!   new row when the width does not fit.
//!
//! A graph with an effect the planner does not model, or whose intermediates do not fit (wider or taller than
//! the atlas, or overflowing it), gets no plan: every blur at the origin, as alone.

use std::collections::HashMap;
use std::rc::Rc;

use super::builtin::filters_hp::{hard_rects, BlurPlace};
use super::{ColorOp, Effect, Input, Kind, Rect};

const ATLAS: i32 = 512;
/// The largest intermediate Direct2D renders a DrawImage with in one go; a larger one tiles it ([`tiles`]).
const TILE: i32 = 1024;
/// A request for all of an input.
const WHOLE: Rect = Rect { l: i32::MIN, t: i32::MIN, r: i32::MAX, b: i32::MAX };
/// The most shader functions Direct2D links into one operation.
const MAX_LINKED: u32 = 8;

#[derive(Clone, Copy)]
enum Node {
    Leaf,
    Flood,
    /// Forwards every request, clipped to its bounds, to its one input (a single-input Composite, a zero-σ blur, a
    /// Crop).
    Pass,
    /// A pixel-shader draw. `cheap`: its instruction-count hint is at most a quarter, so it always folds into a
    /// border; `None` when unknown.
    Draw { linkable: bool, cheap: Option<bool> },
    /// A Border: drawn by its own quads into an intermediate of its own over what is asked of it, from its whole
    /// input; a blur's border copies it rather than taking it in.
    Border { modes: [u32; 2] },
    Blend { mode: u32 },
    /// A ConvolveMatrix. A 3×3 kernel has its own shader (and vertex shader) that links with nothing; a larger one
    /// folds into a blur's border as a draw does. Either reads a drawn input from a padded copy: an intermediate over
    /// what it reads, `grow` beyond the rect it draws, cleared and then drawn into.
    Convolve { folds: bool, grow: [i32; 4] },
    /// A HARD GaussianBlur, whose passes are sized from what its consumers read.
    Blur { sigma: f32, mode: u32, texture: bool },
}

struct NodeInfo {
    kind: Node,
    inputs: Vec<usize>,
    bounds: Rect,
    effect: Option<Effect>,
}

struct Tex {
    rect: Rect,
    ops: Vec<usize>,
    /// The node whose intermediate it is, and what it holds (`None`: a bitmap, or the draw's own target).
    by: Option<(usize, Role)>,
}

/// What an intermediate holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    /// The effect's output.
    Output,
    /// A ConvolveMatrix's input drawn into its padded copy.
    PaddedCopy,
    /// A HARD GaussianBlur's realized input, drawn by its border.
    Border,
    /// A GaussianBlur pass: the shrunk image, the pre-downscale, the X pass or the Y pass.
    Shrink,
    Pre,
    X,
    Y,
}

/// One intermediate Direct2D renders for a DrawImage ([`stages`]).
#[allow(dead_code)] // rect, atlas and consumers describe the plan; the store reads producer, role and precision
pub(crate) struct Stage {
    /// Its rect in the effect's space (device pixels, the draw's origin at its `at`).
    pub rect: [i32; 4],
    /// Its top-left atlas pixel; `None` when the draw has no atlas set.
    pub atlas: Option<[i32; 2]>,
    /// The effect whose intermediate it is, and what it holds.
    pub producer: Effect,
    pub role: Role,
    /// The effects whose draws read it (each operation's own effect: a shader linked into it is not listed).
    pub consumers: Vec<Effect>,
    /// The precision it is stored at (D2D1_BUFFER_PRECISION): the producer's output precision, a padded copy its
    /// input's.
    pub precision: u32,
}

struct Op {
    out: usize,
    rect: Rect,
    slots: Vec<usize>,
    linkable: bool,
    /// A blur's realized-input border operation with no shader folded in yet.
    border_open: bool,
    links: u32,
}

enum Slot {
    Pending,
    Tex(usize),
    Gone,
    Spliced(Vec<usize>),
}

enum Want {
    Sample { op: usize, slot: usize, need: Rect, border: bool },
    /// Draw into `op`'s intermediate with the composite mode `blend` ([`COPY`] for the first draw into a fresh one).
    Target { op: usize, need: Rect, blend: u32 },
}

const SOURCE_OVER: u32 = 0;
const DEST_OVER: u32 = 1;
const PLUS: u32 = 9;
const COPY: u32 = 10;

/// How a Composite of `mode` drawn with `blend` draws: `Some((later, reversed))` when it draws each input
/// straight into the target (the first input in draw order with `blend`, the others with `later`, the inputs
/// last-first when `reversed`), `None` when the two blends do not combine and it needs its own intermediate.
/// They combine when the target is fresh (a copy), when both are the same SOURCE_OVER, DEST_OVER or PLUS, and,
/// with the inputs' order swapped, for SOURCE_OVER over DEST_OVER and the reverse. The first input is the
/// Composite's destination: a copy lays it down and composites the source onto it with the Composite's own mode;
/// a DEST_OVER draw tucks each input under the target, so it takes the inputs topmost first (a SOURCE_OVER
/// Composite's source, a DEST_OVER one's destination). (VM, a DEST_OVER Composite under SOURCE_COPY and both
/// Composites under DEST_OVER: the swapped orders drew one input on the wrong side of the other, r27 32893.)
fn distribute(blend: u32, mode: u32) -> Option<(u32, bool)> {
    if matches!(blend, COPY | 11) {
        return Some((mode, false));
    }
    match (blend, mode) {
        (SOURCE_OVER, SOURCE_OVER) | (DEST_OVER, DEST_OVER) | (PLUS, PLUS) => Some((mode, false)),
        (SOURCE_OVER, DEST_OVER) | (DEST_OVER, SOURCE_OVER) => Some((blend, true)),
        _ => None,
    }
}

/// The intermediates of one blur, to read its placement back.
#[derive(Default)]
struct BlurTex {
    /// Its input's blur runs its last pass in this blur's border.
    fold: bool,
    input: Option<usize>,
    shrink: Option<usize>,
    pre: Option<usize>,
    x: Option<usize>,
    y: Option<usize>,
    out: Option<usize>,
    /// What its consumers read of its output.
    need: [i32; 4],
    /// The operation drawing its realized input (its border).
    border: Option<usize>,
}

#[derive(Default)]
struct Planner {
    nodes: Vec<NodeInfo>,
    by_ptr: HashMap<usize, usize>,
    texs: Vec<Tex>,
    ops: Vec<Op>,
    slots: Vec<Slot>,
    wants: Vec<Vec<Want>>,
    leaf_tex: HashMap<usize, usize>,
    blurs: HashMap<usize, BlurTex>,
    /// Each blur's border operation, by the blur it realizes the input of.
    border_of: HashMap<usize, usize>,
    /// How many inputs of other nodes each node is.
    users: Vec<u32>,
}

/// A rect's area, exact for any rect: a request for a whole input (an unbounded one included) spans up to 2³² − 1
/// pixels a side.
fn area(r: Rect) -> i128 {
    (r.r as i128 - r.l as i128).max(0) * (r.b as i128 - r.t as i128).max(0)
}

fn within(inner: Rect, outer: Rect) -> bool {
    area(inner) == 0 || (inner.l >= outer.l && inner.t >= outer.t && inner.r <= outer.r && inner.b <= outer.b)
}

/// What a Border reads of its input `[a, b)` along one axis for the output span `[lo, hi)`, by edge mode: clamp (0)
/// the clamped span, wrap (1) the whole input, mirror (2) the span and its reflections.
fn edge_read(lo: i32, hi: i32, a: i32, b: i32, mode: u32) -> (i32, i32) {
    if lo >= a && hi <= b || b <= a || hi <= lo {
        return (lo.max(a), hi.min(b));
    }
    match mode {
        0 => (lo.clamp(a, b - 1), (hi - 1).clamp(a, b - 1) + 1),
        2 if a.saturating_sub(lo) < b - a && hi.saturating_sub(b) < b - a => {
            let (mut l, mut r) = (lo.max(a), hi.min(b));
            if lo < a {
                r = r.max(a + (a - lo));
            }
            if hi > b {
                l = l.min(b - (hi - b));
            }
            (l.max(a), r.min(b))
        }
        _ => (a, b),
    }
}

impl Planner {
    /// The node for an input, or `None` when the graph holds something the planner does not model.
    fn input(&mut self, i: &Input) -> Option<usize> {
        match i {
            Input::Bitmap(b) => {
                let key = Rc::as_ptr(b) as usize;
                if let Some(&n) = self.by_ptr.get(&key) {
                    return Some(n);
                }
                let bounds = Rect { l: 0, t: 0, r: b.width as i32, b: b.height as i32 };
                Some(self.push(key, NodeInfo { kind: Node::Leaf, inputs: Vec::new(), bounds, effect: None }))
            }
            // A command list reaches the graph as its rendering for this draw: a texture, as a bitmap is.
            Input::List(l) => {
                let key = Rc::as_ptr(l) as usize;
                if let Some(&n) = self.by_ptr.get(&key) {
                    return Some(n);
                }
                let bounds = super::input_bounds(i);
                Some(self.push(key, NodeInfo { kind: Node::Leaf, inputs: Vec::new(), bounds, effect: None }))
            }
            Input::Effect(e) => self.effect(e),
        }
    }

    fn push(&mut self, key: usize, info: NodeInfo) -> usize {
        self.nodes.push(info);
        self.wants.push(Vec::new());
        self.by_ptr.insert(key, self.nodes.len() - 1);
        self.nodes.len() - 1
    }

    fn effect(&mut self, e: &Effect) -> Option<usize> {
        let key = Rc::as_ptr(&e.0) as usize;
        if let Some(&n) = self.by_ptr.get(&key) {
            return Some(n);
        }
        let (kind, inputs) = {
            let d = e.0.borrow();
            (d.kind, d.inputs.clone())
        };
        let ins = inputs.iter().map(|i| self.input(i)).collect::<Option<Vec<_>>>()?;
        let node = match kind {
            Kind::ColorMatrix
            | Kind::Color(ColorOp::Premultiply | ColorOp::UnPremultiply | ColorOp::Grayscale | ColorOp::HueRotation | ColorOp::RgbToHue) => {
                Node::Draw { linkable: true, cheap: Some(true) }
            }
            Kind::Custom => {
                let n = e.0.borrow().custom.as_ref().and_then(|c| c.single_shader())?;
                if n as usize != ins.len() {
                    return None;
                }
                Node::Draw { linkable: false, cheap: None }
            }
            // Two-input linkable draws; a blur feeding one keeps its last pass apart (each input's own intermediate).
            Kind::Arithmetic | Kind::CrossFade if ins.len() == 2 => Node::Draw { linkable: true, cheap: None },
            Kind::Flood if ins.is_empty() => Node::Flood,
            Kind::Composite if ins.len() == 1 => Node::Pass,
            Kind::Composite if ins.len() > 1 => Node::Blend { mode: e.get_u32(0, 0) },
            Kind::GaussianBlur if ins.len() == 1 && e.get_u32(2, 0) == 1 => {
                if e.sigma() > 0.0 {
                    Node::Blur { sigma: e.sigma(), mode: e.get_u32(1, 1), texture: matches!(self.nodes[ins[0]].kind, Node::Leaf) }
                } else {
                    Node::Pass
                }
            }
            Kind::Crop if ins.len() == 1 => Node::Pass,
            Kind::Border if ins.len() == 1 => Node::Border { modes: [e.get_u32(0, 0), e.get_u32(1, 0)] },
            Kind::ConvolveMatrix if ins.len() == 1 => {
                let (b, ib) = (e.bounds(), self.nodes[ins[0]].bounds);
                let folds = e.get_u32(2, 3) > 3 || e.get_u32(3, 3) > 3;
                Node::Convolve { folds, grow: [ib.l - b.l, ib.t - b.t, b.r - ib.r, b.b - ib.b] }
            }
            _ => return None,
        };
        let bounds = e.bounds();
        Some(self.push(key, NodeInfo { kind: node, inputs: ins, bounds, effect: Some(e.clone()) }))
    }

    fn tex(&mut self, rect: Rect, by: Option<(usize, Role)>) -> usize {
        self.texs.push(Tex { rect, ops: Vec::new(), by });
        self.texs.len() - 1
    }

    fn op(&mut self, out: usize, rect: Rect) -> usize {
        self.ops.push(Op { out, rect, slots: Vec::new(), linkable: false, border_open: false, links: 1 });
        self.texs[out].ops.push(self.ops.len() - 1);
        self.ops.len() - 1
    }

    fn slot(&mut self, op: usize) -> usize {
        self.slots.push(Slot::Pending);
        let s = self.slots.len() - 1;
        self.ops[op].slots.push(s);
        s
    }

    /// Request each of `n`'s inputs from `op` (its draw now runs there), into fresh slots. A shader the planner does
    /// not know the sampling of (not linkable) asks for its inputs whole.
    fn request_inputs(&mut self, n: usize, op: usize, need: Rect) -> Vec<usize> {
        let ins = self.nodes[n].inputs.clone();
        if let Node::Convolve { grow: [l, t, r, b], .. } = self.nodes[n].kind {
            let shown = need.intersect(self.nodes[n].bounds);
            let read = Rect { l: shown.l - l, t: shown.t - t, r: shown.r + r, b: shown.b + b };
            self.slots.push(Slot::Pending);
            let s = self.slots.len() - 1;
            if matches!(self.nodes[ins[0]].kind, Node::Leaf) {
                self.wants[ins[0]].push(Want::Sample { op, slot: s, need: read, border: false });
            } else {
                // The padded copy: cleared over the whole read rect, the input drawn into it.
                let t = self.tex(read, Some((n, Role::PaddedCopy)));
                let copy = self.op(t, read);
                self.slots[s] = Slot::Tex(t);
                self.wants[ins[0]].push(Want::Target { op: copy, need: read, blend: COPY });
            }
            return vec![s];
        }
        let need = if matches!(self.nodes[n].kind, Node::Draw { linkable: false, .. }) { WHOLE } else { need };
        let border = matches!(self.nodes[n].kind, Node::Border { .. });
        ins.iter()
            .map(|&i| {
                self.slots.push(Slot::Pending);
                let s = self.slots.len() - 1;
                // A Border reads its whole input, which is the input's bounds.
                let need = if border { self.nodes[i].bounds } else { need };
                self.wants[i].push(Want::Sample { op, slot: s, need, border: false });
                s
            })
            .collect()
    }

    /// Turn operation `op` into `n` consecutive operations writing the same intermediate (`op` first), in place.
    fn split_op(&mut self, op: usize, n: usize) -> Vec<usize> {
        let t = self.ops[op].out;
        let at = self.texs[t].ops.iter().position(|&k| k == op).unwrap();
        let mut list = vec![op];
        for k in 1..n {
            let rect = self.ops[op].rect;
            self.ops.push(Op { out: t, rect, slots: Vec::new(), linkable: false, border_open: false, links: 1 });
            let new = self.ops.len() - 1;
            self.texs[t].ops.insert(at + k, new);
            list.push(new);
        }
        list
    }

    /// A Composite's inputs drawn into the operations `list` (draw order): the first with `first`, the others
    /// with `later`, last input first when `reversed`.
    fn draw_inputs(&mut self, ins: &[usize], list: &[usize], need: Rect, first: u32, later: u32, reversed: bool) {
        for (k, &op) in list.iter().enumerate() {
            let i = if reversed { ins[ins.len() - 1 - k] } else { ins[k] };
            self.wants[i].push(Want::Target { op, need, blend: if k == 0 { first } else { later } });
        }
    }

    /// A Composite of `mode` drawn into its own intermediate `t`, which starts fresh.
    fn own_draws(&mut self, ins: &[usize], t: usize, mode: u32) {
        let rect = self.texs[t].rect;
        let list: Vec<usize> = (0..ins.len()).map(|_| self.op(t, rect)).collect();
        let (later, reversed) = distribute(COPY, mode).expect("a fresh target takes every mode");
        self.draw_inputs(ins, &list, rect, COPY, later, reversed);
    }

    /// `n`'s draw becomes operation `op`.
    fn fill(&mut self, n: usize, op: usize, need: Rect, linkable: bool) {
        let s = self.request_inputs(n, op, need);
        let o = &mut self.ops[op];
        o.linkable = linkable;
        o.border_open = false;
        o.slots.extend(s);
    }

    /// `n`'s draw folds into the operation behind `slot`: that slot becomes `n`'s inputs.
    fn fold(&mut self, n: usize, op: usize, slot: usize, need: Rect) {
        let s = self.request_inputs(n, op, need);
        self.slots[slot] = Slot::Spliced(s);
    }

    fn resolve(&mut self, n: usize) -> Option<()> {
        let wants = std::mem::take(&mut self.wants[n]);
        let bounds = self.nodes[n].bounds;
        // A linkable one-input draw, or a Border, over a HARD blur nothing else reads is linked into the blur's last
        // pass: that pass draws wherever the draw would (a Border's quads by border addressing, asking the blur for what
        // its edge modes read).
        if let [i] = self.nodes[n].inputs[..] {
            let over_blur = matches!(self.nodes[i].kind, Node::Blur { .. }) && self.users[i] == 1;
            let modes = match self.nodes[n].kind {
                Node::Draw { linkable: true, .. } => Some(None),
                Node::Border { modes } => Some(Some(modes)),
                _ => None,
            };
            if let (true, Some(modes)) = (over_blur, modes) {
                let ib = self.nodes[i].bounds;
                let map = |r: Rect| -> Rect {
                    let r = r.intersect(bounds);
                    let Some([mx, my]) = modes else { return r };
                    let (l, rr) = edge_read(r.l, r.r, ib.l, ib.r, mx);
                    let (t, b) = edge_read(r.t, r.b, ib.t, ib.b, my);
                    Rect { l, t, r: rr, b }
                };
                self.wants[i].extend(wants.into_iter().map(|w| match w {
                    Want::Sample { op, slot, need, border } => Want::Sample { op, slot, need: map(need), border },
                    Want::Target { op, need, blend } => Want::Target { op, need: map(need), blend },
                }));
                return Some(());
            }
        }
        match self.nodes[n].kind {
            Node::Pass => {
                let i = self.nodes[n].inputs[0];
                let clip = |r: Rect| r.intersect(bounds);
                self.wants[i].extend(wants.into_iter().map(|w| match w {
                    Want::Sample { op, slot, need, border } => Want::Sample { op, slot, need: clip(need), border },
                    Want::Target { op, need, blend } => Want::Target { op, need: clip(need), blend },
                }));
            }
            Node::Leaf => {
                let t = *self.leaf_tex.entry(n).or_insert_with(|| {
                    self.texs.push(Tex { rect: bounds, ops: Vec::new(), by: None });
                    self.texs.len() - 1
                });
                for w in wants {
                    match w {
                        Want::Sample { slot, .. } => self.slots[slot] = Slot::Tex(t),
                        Want::Target { op, .. } => {
                            let s = self.slot(op);
                            self.slots[s] = Slot::Tex(t);
                        }
                    }
                }
            }
            Node::Flood => {
                for w in wants {
                    if let Want::Sample { slot, .. } = w {
                        self.slots[slot] = Slot::Gone;
                    }
                }
            }
            Node::Draw { .. } | Node::Border { .. } | Node::Convolve { .. } => {
                let is_border = matches!(self.nodes[n].kind, Node::Border { .. });
                // A Border asked only for what lies inside its input draws nothing of its own: the input is read there.
                let wants: Vec<Want> = if is_border {
                    let (i, ib) = (self.nodes[n].inputs[0], self.nodes[self.nodes[n].inputs[0]].bounds);
                    let (inside, rest): (Vec<Want>, Vec<Want>) = wants.into_iter().partition(|w| match w {
                        Want::Sample { need, .. } | Want::Target { need, .. } => within(*need, ib),
                    });
                    self.wants[i].extend(inside);
                    rest
                } else {
                    wants
                };
                let (linkable, cheap, folds) = match self.nodes[n].kind {
                    Node::Draw { linkable, cheap } => (linkable, cheap, true),
                    Node::Convolve { folds, .. } => (false, None, folds),
                    _ => (false, Some(false), false),
                };
                let mut own: Vec<(usize, Rect)> = Vec::new();
                for w in wants {
                    match w {
                        Want::Target { op, need, .. } => self.fill(n, op, need, linkable),
                        Want::Sample { op, slot, need, border } => {
                            let (o_border, o_linkable, o_links, o_rect) = {
                                let o = &self.ops[op];
                                (o.border_open, o.linkable, o.links, o.rect)
                            };
                            let fits = within(need, bounds);
                            if border && o_border && fits && folds {
                                // A border op takes a draw whose cost is low, or whose area it at most doubles.
                                let ratio_ok = area(need) > 0 && area(o_rect) <= 2 * area(need);
                                if cheap != Some(true) && !ratio_ok && cheap.is_none() {
                                    return None;
                                }
                                if cheap == Some(true) || ratio_ok {
                                    self.ops[op].border_open = false;
                                    self.ops[op].linkable = linkable;
                                    self.fold(n, op, slot, need.intersect(bounds));
                                    continue;
                                }
                            }
                            if linkable && o_linkable && fits && o_links < MAX_LINKED {
                                self.ops[op].links += 1;
                                self.fold(n, op, slot, need.intersect(bounds));
                                continue;
                            }
                            own.push((slot, need));
                        }
                    }
                }
                if !own.is_empty() {
                    let rect = own.iter().map(|&(_, r)| r).reduce(|a, b| a.union(b)).unwrap().intersect(bounds);
                    let t = self.tex(rect, Some((n, Role::Output)));
                    let op = self.op(t, rect);
                    for (slot, _) in own {
                        self.slots[slot] = Slot::Tex(t);
                    }
                    self.fill(n, op, rect, linkable);
                    // A Border's mirror quads take a linkable input linked in.
                    if is_border {
                        self.ops[op].linkable = true;
                    }
                }
            }
            Node::Blend { mode } => {
                // Every consumer that samples the Composite shares one intermediate, which its inputs draw into
                // from fresh. Drawn into another target, it draws its inputs straight there when the two blends
                // combine ([`distribute`]); otherwise it renders on its own and is copied there.
                let ins = self.nodes[n].inputs.clone();
                let mut sampled: Option<(Rect, Vec<usize>)> = None;
                let mut own: Vec<(Rect, usize)> = Vec::new();
                for w in wants {
                    match w {
                        Want::Sample { slot, need, .. } => {
                            let g = sampled.get_or_insert((need, Vec::new()));
                            g.0 = g.0.union(need);
                            g.1.push(slot);
                        }
                        Want::Target { op, need, blend } => match distribute(blend, mode) {
                            Some((later, reversed)) => {
                                let list = self.split_op(op, ins.len());
                                self.draw_inputs(&ins, &list, need, blend, later, reversed);
                            }
                            None => own.push((need, op)),
                        },
                    }
                }
                if let Some((need, slots)) = sampled {
                    let t = self.tex(need.intersect(bounds), Some((n, Role::Output)));
                    for s in slots {
                        self.slots[s] = Slot::Tex(t);
                    }
                    self.own_draws(&ins, t, mode);
                }
                for (need, op) in own {
                    let t = self.tex(need.intersect(bounds), Some((n, Role::Output)));
                    let s = self.slot(op);
                    self.slots[s] = Slot::Tex(t);
                    self.own_draws(&ins, t, mode);
                }
            }
            Node::Blur { sigma, mode, texture } => {
                let input = self.nodes[n].inputs[0];
                let ib = self.nodes[input].bounds;
                // One output, over what every consumer reads of it (its bounds are its input's).
                let need = wants.iter().map(|w| match w {
                    Want::Sample { need, .. } | Want::Target { need, .. } => *need,
                }).reduce(Rect::union)?.intersect(ib);
                if area(need) == 0 {
                    return None;
                }
                let r = hard_rects([ib.l, ib.t, ib.r, ib.b], [need.l, need.t, need.r, need.b], sigma, mode, texture)?;
                let rect = |a: [i32; 4]| Rect { l: a[0], t: a[1], r: a[2], b: a[3] };
                // The last pass draws into each target it is composited into, and once into its own
                // intermediate for every consumer that samples it. Read through another blur's border, it folds into
                // that border, whatever the border's area, and runs there over the border's quads.
                let mut finals: Vec<usize> = Vec::new();
                let mut own: Option<usize> = None;
                let mut folds: Vec<(usize, usize)> = Vec::new();
                for w in wants {
                    match w {
                        Want::Target { op, .. } => finals.push(op),
                        Want::Sample { op, slot, border: true, .. } if self.ops[op].border_open => {
                            self.ops[op].border_open = false;
                            folds.push((op, slot));
                            let consumer = *self.border_of.get(&op)?;
                            self.blurs.get_mut(&consumer)?.fold = true;
                        }
                        Want::Sample { slot, .. } => {
                            let t = match own {
                                Some(t) => t,
                                None => {
                                    let t = self.tex(rect(r.out), Some((n, Role::Output)));
                                    finals.push(self.op(t, rect(r.out)));
                                    own = Some(t);
                                    t
                                }
                            };
                            self.slots[slot] = Slot::Tex(t);
                        }
                    }
                }
                // One output: every consumer reads the same pixels in the engine.
                if finals.len() > 1 || finals.is_empty() && folds.is_empty() {
                    return None;
                }
                let mut bt = BlurTex { out: finals.first().map(|&f| self.ops[f].out), need: [need.l, need.t, need.r, need.b], ..Default::default() };
                let stage = |p: &mut Planner, at: [i32; 4], into: usize, role: Role| -> usize {
                    let rect = rect(at);
                    let t = p.tex(rect, Some((n, role)));
                    let op = p.op(t, rect);
                    let s = p.slot(into);
                    p.slots[s] = Slot::Tex(t);
                    op
                };
                // The intermediate the last pass reads (the Y pass's, or the X pass's at full resolution), read by every
                // final draw and every border it folds into.
                let first = rect(r.y.unwrap_or(r.x));
                let t = self.tex(first, Some((n, if r.y.is_some() { Role::Y } else { Role::X })));
                let mut last = self.op(t, first);
                for &f in &finals {
                    let s = self.slot(f);
                    self.slots[s] = Slot::Tex(t);
                }
                for &(_, slot) in &folds {
                    self.slots[slot] = Slot::Tex(t);
                }
                if r.y.is_some() {
                    bt.y = Some(t);
                    last = stage(self, r.x, last, Role::X);
                }
                bt.x = Some(self.ops[last].out);
                if let Some(pre) = r.pre {
                    last = stage(self, pre, last, Role::Pre);
                    bt.pre = Some(self.ops[last].out);
                }
                if let Some(shrink) = r.shrink {
                    last = stage(self, shrink, last, Role::Shrink);
                    bt.shrink = Some(self.ops[last].out);
                }
                if let Some(real) = r.input {
                    last = stage(self, real, last, Role::Border);
                    bt.input = Some(self.ops[last].out);
                    self.ops[last].border_open = true;
                    bt.border = Some(last);
                    let key = self.nodes[n].effect.as_ref().map(|e| Rc::as_ptr(&e.0) as usize)?;
                    self.border_of.insert(last, key);
                    let s = self.slot(last);
                    // The border asks its input for the realized rect clipped to the input's bounds.
                    let need = rect(real).intersect(ib);
                    self.wants[input].push(Want::Sample { op: last, slot: s, need, border: true });
                } else {
                    // The first pass samples a bitmap itself; anything else between would be a mismatch.
                    if !matches!(self.nodes[input].kind, Node::Leaf) {
                        return None;
                    }
                    let s = self.slot(last);
                    self.wants[input].push(Want::Sample { op: last, slot: s, need: ib, border: false });
                }
                let key = self.nodes[n].effect.as_ref().map(|e| Rc::as_ptr(&e.0) as usize)?;
                self.blurs.insert(key, bt);
            }
        }
        Some(())
    }

    fn flatten(&self, s: usize, out: &mut Vec<usize>) -> Option<()> {
        match &self.slots[s] {
            Slot::Pending => return None,
            Slot::Tex(t) => out.push(*t),
            Slot::Gone => {}
            Slot::Spliced(v) => {
                for &x in v {
                    self.flatten(x, out)?;
                }
            }
        }
        Some(())
    }
}

/// Where Direct2D places every HARD blur of `root`'s graph for a DrawImage of its rect `need` whose output lands
/// at device pixel `at`, keyed by the blur effect, with the largest intermediate's size; `None` when the graph is not
/// one the planner models.
/// A DrawImage's plan: the planner, the draw's own target, each operation's input textures, the allocator's walk and
/// where it put each intermediate (empty when the draw has no atlas set).
struct Built {
    p: Planner,
    root: usize,
    root_tex: usize,
    inputs: Vec<Vec<usize>>,
    visited: Vec<usize>,
    place: HashMap<usize, [i32; 2]>,
    largest: [i32; 2],
}

fn build(root: &Effect, need: [i32; 4], blend: u32) -> Option<Built> {
    let mut p = Planner::default();
    let r = p.effect(root)?;
    p.users = vec![0; p.nodes.len()];
    for k in 0..p.nodes.len() {
        for i in p.nodes[k].inputs.clone() {
            p.users[i] += 1;
        }
    }
    // Consumers before producers: reverse post-order from the root.
    let mut order = Vec::new();
    let mut seen = vec![false; p.nodes.len()];
    fn post(p: &Planner, n: usize, seen: &mut [bool], order: &mut Vec<usize>) {
        if std::mem::replace(&mut seen[n], true) {
            return;
        }
        for &i in &p.nodes[n].inputs {
            post(p, i, seen, order);
        }
        order.push(n);
    }
    post(&p, r, &mut seen, &mut order);
    let root_rect = Rect { l: need[0], t: need[1], r: need[2], b: need[3] }.intersect(p.nodes[r].bounds);
    if area(root_rect) == 0 {
        return None;
    }
    let root_tex = p.tex(root_rect, None);
    let root_op = p.op(root_tex, root_rect);
    p.wants[r].push(Want::Target { op: root_op, need: root_rect, blend });
    for &n in order.iter().rev() {
        p.resolve(n)?;
    }
    if p.wants.iter().any(|w| !w.is_empty()) {
        return None;
    }

    // The allocator's walk.
    let inputs: Vec<Vec<usize>> = (0..p.ops.len())
        .map(|k| {
            let mut v = Vec::new();
            p.ops[k].slots.iter().try_for_each(|&s| p.flatten(s, &mut v)).map(|_| v)
        })
        .collect::<Option<_>>()?;
    let nt = p.texs.len();
    let mut consumers = vec![0u32; nt];
    for ins in &inputs {
        for &t in ins {
            consumers[t] += 1;
        }
    }
    let (mut depth, mut visits) = (vec![0u32; nt], vec![0u32; nt]);
    let mut stack: Vec<usize> = p.texs[root_tex].ops.clone();
    let mut pending: std::collections::VecDeque<usize> = Default::default();
    let mut visited = Vec::new();
    while let Some(&k) = stack.last() {
        let fresh = inputs[k].iter().any(|&t| !p.texs[t].ops.is_empty() && visits[t] == 0);
        if fresh && !pending.is_empty() {
            while let Some(t) = pending.pop_back() {
                stack.extend(&p.texs[t].ops);
            }
            if stack.last() != Some(&k) {
                continue;
            }
        }
        let out = p.ops[k].out;
        for &t in &inputs[k] {
            if p.texs[t].ops.is_empty() {
                continue;
            }
            depth[t] = depth[t].max(depth[out] + 1);
            visits[t] += 1;
            if visits[t] == consumers[t] {
                pending.push_front(t);
            }
        }
        stack.pop();
        if stack.last().is_none_or(|&n| p.ops[n].out != out) {
            while let Some(t) = pending.pop_back() {
                stack.extend(&p.texs[t].ops);
            }
        }
        visited.push(k);
    }

    // Shelf-pack each depth's atlas in reverse visit order. An intermediate wider or taller than the atlas, or one that
    // overflows it, leaves the graph unatlased: every intermediate in a texture of its own, at its origin.
    let mut shelves: HashMap<u32, (i32, i32, i32)> = HashMap::new();
    let mut place: HashMap<usize, [i32; 2]> = HashMap::new();
    let (mut atlased, mut largest) = (true, [0, 0]);
    for &k in visited.iter().rev() {
        let t = p.ops[k].out;
        if t == root_tex || place.contains_key(&t) {
            continue;
        }
        let rc = p.texs[t].rect;
        let (w, h) = (rc.r - rc.l, rc.b - rc.t);
        largest = [largest[0].max(w), largest[1].max(h)];
        if !atlased {
            continue;
        }
        if w > ATLAS || h > ATLAS {
            atlased = false;
            continue;
        }
        let (x, y, sh) = shelves.entry(depth[t]).or_insert((0, 0, 0));
        if w >= ATLAS - *x {
            (*x, *y, *sh) = (0, *y + *sh, 0);
        }
        if h > ATLAS - *y {
            atlased = false;
            continue;
        }
        place.insert(t, [*x, *y]);
        *x += w;
        *sh = (*sh).max(h);
    }
    if !atlased {
        place.clear();
    }
    Some(Built { p, root: r, root_tex, inputs, visited, place, largest })
}

fn plan(root: &Effect, need: [i32; 4], at: Option<[i32; 2]>, blend: u32) -> Option<(HashMap<usize, BlurPlace>, [i32; 2])> {
    let Built { p, root_tex, place, largest, .. } = build(root, need, blend)?;
    let at_of = |t: Option<usize>| match t {
        Some(t) if t == root_tex => at,
        Some(t) => Some(place.get(&t).copied().unwrap_or_default()),
        None => Some([0, 0]),
    };
    // Where a blur's output rect starts in the target its last pass draws: the target's place, moved by where the
    // rect sits in it (the device pixel of the effect's origin, for the draw's own target).
    let out_at = |t: Option<usize>, out: [i32; 4]| {
        let base = at_of(t)?;
        let org = match t {
            Some(t) if t == root_tex => [0, 0],
            Some(t) => [p.texs[t].rect.l, p.texs[t].rect.t],
            None => [out[0], out[1]],
        };
        Some([base[0] + out[0] - org[0], base[1] + out[1] - org[1]])
    };
    let places = p
        .blurs
        .iter()
        .map(|(&k, b)| {
            // A border reading one drawn intermediate (not a bitmap, not a blur's last pass it runs itself).
            let src = b.border.filter(|_| !b.fold).and_then(|op| {
                let mut ts = Vec::new();
                p.ops[op].slots.iter().try_for_each(|&s| p.flatten(s, &mut ts))?;
                let [t] = ts[..] else { return None };
                if p.texs[t].ops.is_empty() {
                    return None;
                }
                let r = p.texs[t].rect;
                Some((at_of(Some(t))?, [r.l, r.t, r.r, r.b]))
            });
            Some((k, BlurPlace { input: at_of(b.input)?, shrink: at_of(b.shrink)?, pre: at_of(b.pre)?, x: at_of(b.x)?, y: at_of(b.y)?, out: out_at(b.out, b.need)?, fold: b.fold, need: Some(b.need), src }))
        })
        .collect::<Option<_>>()?;
    Some((places, largest))
}

/// The tiles Direct2D renders a DrawImage of `root`'s rect `need` in: the rect itself while every intermediate fits a
/// [`TILE`], else its halves (the longer side split at the floored middle, the second half first), each tiled the same way. Each tile renders
/// the whole graph for that part of the output. A graph the planner does not model is one tile.
pub(super) fn tiles(root: &Effect, need: [i32; 4], at: Option<[i32; 2]>, blend: u32) -> Vec<[i32; 4]> {
    fn split(root: &Effect, need: [i32; 4], at: Option<[i32; 2]>, blend: u32, out: &mut Vec<[i32; 4]>) {
        let (w, h) = (need[2] - need[0], need[3] - need[1]);
        let big = plan(root, need, at, blend).is_some_and(|(_, l)| l[0] > TILE || l[1] > TILE);
        if !big || (w < 2 && h < 2) {
            out.push(need);
        } else if w >= h {
            let mid = need[0] + w / 2;
            split(root, [mid, need[1], need[2], need[3]], at, blend, out);
            split(root, [need[0], need[1], mid, need[3]], at, blend, out);
        } else {
            let mid = need[1] + h / 2;
            split(root, [need[0], mid, need[2], need[3]], at, blend, out);
            split(root, [need[0], need[1], need[2], mid], at, blend, out);
        }
    }
    let mut out = Vec::new();
    split(root, need, at, blend, &mut out);
    out
}

/// Plan `root`'s DrawImage of its rect `need` (output at device pixel `at`, composite mode `blend`) and store each HARD blur's placement on it. When a
/// placement changes, every effect in the graph drops its realized output, which was rendered with the old one.
/// The intermediates Direct2D renders for a DrawImage of `root`'s rect `need` (its origin at device pixel `at`, drawn
/// with composite mode `blend`; the arguments of [`apply`]), producers first: every one the planner allocates, none
/// for a draw onto the target or a bitmap. `None` for a graph the planner does not model. Read-only.
pub(crate) fn stages(root: &Effect, need: [i32; 4], at: Option<[i32; 2]>, blend: u32) -> Option<Vec<Stage>> {
    let _ = at; // the stages do not depend on where the output lands
    let Built { p, root, root_tex, inputs, visited, place, .. } = build(root, need, blend)?;
    let effect = |n: usize| p.nodes[n].effect.clone();
    // An operation's own effect: whose intermediate it draws, or the root's for the draw onto the target.
    let owner = |k: usize| match p.texs[p.ops[k].out].by {
        Some((n, _)) => effect(n),
        None => effect(root),
    };
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for &k in visited.iter().rev() {
        let t = p.ops[k].out;
        if t == root_tex || !seen.insert(t) {
            continue;
        }
        let (n, role) = p.texs[t].by?;
        let producer = effect(n)?;
        let precision = match role {
            Role::PaddedCopy => super::input_precision(&producer.0.borrow().inputs[0]),
            _ => producer.out_precision(),
        };
        let mut consumers: Vec<Effect> = Vec::new();
        for k in (0..p.ops.len()).filter(|&k| inputs[k].contains(&t)) {
            if let Some(e) = owner(k) && !consumers.iter().any(|c| Rc::ptr_eq(&c.0, &e.0)) {
                consumers.push(e);
            }
        }
        let r = p.texs[t].rect;
        out.push(Stage { rect: [r.l, r.t, r.r, r.b], atlas: place.get(&t).copied(), producer, role, consumers, precision });
    }
    Some(out)
}

pub(super) fn apply(root: &Effect, need: [i32; 4], at: Option<[i32; 2]>, blend: u32) {
    // A custom shader the plan gives an output intermediate stores its output there, at that intermediate's
    // precision: it has no linkable form, so its consumer reads it from there. A built-in effect's shader links into
    // a consumer that samples it 1:1 (a ConvolveMatrix into a ColorMatrix keeps its negative values), so its output
    // is not stored.
    let stores = stages(root, need, at, blend)
        .unwrap_or_default()
        .into_iter()
        .filter(|s| s.role == Role::Output && s.producer.0.borrow().kind == Kind::Custom)
        .map(|s| (Rc::as_ptr(&s.producer.0) as usize, s.precision))
        .collect();
    store(root, plan(root, need, at, blend).map(|(p, _)| p).unwrap_or_default(), stores);
}

/// The draws a DrawImage of `root` with composite mode `blend` makes straight onto the target when `root` is a
/// Composite whose mode combines with it ([`distribute`]): each input with its blend, in draw order, nested
/// Composites (and single-input ones) flattened the same way. `None` when `root` is drawn as one image.
pub(super) fn distributed(root: &Effect, blend: u32) -> Option<Vec<(Input, u32)>> {
    fn expand(i: Input, blend: u32, out: &mut Vec<(Input, u32)>) {
        if let Input::Effect(e) = &i {
            let d = e.0.borrow();
            if d.kind == Kind::Composite && d.inputs.len() == 1 {
                let only = d.inputs[0].clone();
                drop(d);
                return expand(only, blend, out);
            }
            if d.kind == Kind::Composite
                && d.inputs.len() > 1
                && let Some((later, reversed)) = distribute(blend, e.get_u32(0, 0))
            {
                let mut ins = d.inputs.clone();
                drop(d);
                if reversed {
                    ins.reverse();
                }
                for (k, i) in ins.into_iter().enumerate() {
                    expand(i, if k == 0 { blend } else { later }, out);
                }
                return;
            }
        }
        out.push((i, blend));
    }
    let d = root.0.borrow();
    if d.kind != Kind::Composite || d.inputs.len() < 2 {
        return None;
    }
    distribute(blend, root.get_u32(0, 0))?;
    drop(d);
    let mut out = Vec::new();
    expand(Input::Effect(root.clone()), blend, &mut out);
    Some(out)
}

/// Every blur of `root`'s graph back at the origin and every effect linked (a realize outside a DrawImage).
pub(super) fn clear(root: &Effect) {
    store(root, HashMap::new(), HashMap::new());
}

fn store(root: &Effect, places: HashMap<usize, BlurPlace>, stores: HashMap<usize, u32>) {
    let mut all = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![root.clone()];
    while let Some(e) = stack.pop() {
        if !seen.insert(Rc::as_ptr(&e.0) as usize) {
            continue;
        }
        let d = e.0.borrow();
        stack.extend(d.inputs.iter().filter_map(|i| match i {
            Input::Effect(e) => Some(e.clone()),
            Input::Bitmap(_) | Input::List(_) => None,
        }));
        drop(d);
        all.push(e);
    }
    let mut changed = false;
    for e in &all {
        let mut d = e.0.borrow_mut();
        let st = stores.get(&(Rc::as_ptr(&e.0) as usize)).copied();
        if d.store != st {
            d.store = st;
            changed = true;
        }
        if d.kind == Kind::GaussianBlur {
            let new = places.get(&(Rc::as_ptr(&e.0) as usize)).copied().unwrap_or_default();
            if d.place != new {
                d.place = new;
                changed = true;
            }
        }
    }
    if changed {
        for e in &all {
            let mut d = e.0.borrow_mut();
            d.realized = None;
            d.realized_f = None;
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use d2d_types::abi::{CLSID_D2D1ColorMatrix, CLSID_D2D1GaussianBlur};

    // A HARD halving blur over a ColorMatrix of a bitmap renders its border (the ColorMatrix linked in), its X pass and
    // its Y pass; its upscale draws onto the target. Each is the blur's, at the context-less 8bpc precision.
    #[test]
    fn stages_of_a_halving_hard_blur() {
        let dc = crate::context::DeviceContext::new(crate::resource::Domain { factory_id: 0, device_id: 0 });
        let bm = crate::bitmap::Bitmap::from_straight_rgba(48, 48, &vec![200; 48 * 48 * 4]);
        let cm = dc.create_effect(CLSID_D2D1ColorMatrix).unwrap();
        cm.set_input(0, Input::Bitmap(Rc::new(bm)));
        let blur = dc.create_effect(CLSID_D2D1GaussianBlur).unwrap();
        blur.set_input(0, Input::Effect(cm.clone()));
        blur.set_value(0, super::super::Prop::F32(3.0)).unwrap();
        blur.set_value(1, super::super::Prop::U32(2)).unwrap();
        blur.set_value(2, super::super::Prop::U32(1)).unwrap();
        let st = stages(&blur, [0, 0, 48, 48], Some([0, 0]), 0).expect("a modelled graph");
        assert_eq!(st.iter().map(|s| s.role).collect::<Vec<_>>(), [Role::Border, Role::X, Role::Y]);
        assert!(st.iter().all(|s| Rc::ptr_eq(&s.producer.0, &blur.0) && s.precision == 1));
        // Each pass reads the one before it; the last is read by the blur's own draw onto the target.
        assert!(st.iter().all(|s| s.consumers.len() == 1 && Rc::ptr_eq(&s.consumers[0].0, &blur.0)));
        assert!(st.iter().all(|s| s.atlas.is_some()));
    }
}
