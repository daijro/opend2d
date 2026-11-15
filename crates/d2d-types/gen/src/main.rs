//! Transform the pinned `windows 0.62.2` Win32 source into a portable (native + wasm32)
//! `d2d-types` by stripping COM/link machinery and flattening namespaces to one `abi` module,
//! plus an interface inventory (name/IID/base/method order) from `define_interface!`,
//! `interface_hierarchy!` and the `*_Vtbl` structs. Deterministic, no network.
//!
//! Usage: `cargo run` (defaults), or `cargo run -- <graphics_src_dir> <d2d_types_src_dir>`.

use proc_macro2::{Literal, TokenStream};
use quote::quote;
use std::collections::{BTreeMap, BTreeSet};
use syn::visit::Visit;
use syn::visit_mut::{self, VisitMut};
use syn::{Fields, Item, Type, Visibility};

/// Foreign Win32 types we hand-provide portably in `d2d_types::prelude` (kept small on purpose).
/// `ComPtr` stands in for every COM-interface-typed field (all pointer-sized).
// Only the foreign types actually referenced by emitted value structs (vtable-only refs don't
// count, since vtbls are not emitted). `ComPtr` replaces COM-interface fields; PCWSTR/PCSTR are
// the string pointers windows-core only re-exports on Windows.
const PRELUDE: &[&str] =
    &["ComPtr", "HANDLE", "HWND", "HMONITOR", "RECT", "POINT", "LUID", "PCWSTR", "PCSTR"];
/// Never counted as a needed type: the impl self-alias and std wrappers that are always in scope.
const IGNORE: &[&str] = &["Self", "Option", "ManuallyDrop", "Box", "Some", "None", "Default"];

/// COM bases of two WIC interfaces that live outside the four Win32 namespaces we process.
/// (name, IID, total slots, generated const ident). IUnknown is handled separately.
const EXTERN_BASES: &[(&str, u128, u32, &str)] = &[
    ("IStream", 0x0000000c_0000_0000_c000_000000000046, 14, "ISTREAM"),
    ("IPersistStream", 0x00000109_0000_0000_c000_000000000046, 8, "IPERSISTSTREAM"),
];
const PRIMS: &[&str] = &[
    "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize", "f32",
    "f64", "bool", "char",
];
const FILES: &[&str] = &[
    "Direct2D/mod.rs", "Direct2D/Common/mod.rs", "DirectWrite/mod.rs", "Imaging/mod.rs",
    "Imaging/D2D/mod.rs", "Dxgi/mod.rs", "Dxgi/Common/mod.rs",
];

fn main() {
    let mut args = std::env::args().skip(1);
    let src = args.next().unwrap_or_else(|| {
        format!(
            "{}/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/windows-0.62.2/src/Windows/Win32/Graphics",
            std::env::var("HOME").unwrap()
        )
    });
    let out = args.next().unwrap_or_else(|| {
        format!("{}/../src", env!("CARGO_MANIFEST_DIR"))
    });

    let files: Vec<syn::File> = FILES
        .iter()
        .map(|f| syn::parse_file(&std::fs::read_to_string(format!("{src}/{f}")).unwrap()).unwrap())
        .collect();
    let items: Vec<&Item> = files.iter().flat_map(|f| &f.items).collect();

    emit_abi(&items, &out);
    emit_inventory(&items, &out);
}

// ---- value types / enums / constants / GUIDs -> abi.rs ---------------------------------------

fn emit_abi(items: &[&Item], out: &str) {
    // Pass 1: names of every value type we could emit, plus the interface set (used to lower
    // COM-interface-typed fields to `ComPtr`).
    let mut defined = BTreeSet::new();
    let mut ifaces = BTreeSet::new();
    for it in items {
        match it {
            Item::Struct(s) => match s.ident.to_string().strip_suffix("_Vtbl") {
                Some(iface) => { ifaces.insert(iface.to_string()); }
                None => { defined.insert(s.ident.to_string()); }
            },
            Item::Union(u) => { defined.insert(u.ident.to_string()); }
            Item::Type(t) => { defined.insert(t.ident.to_string()); }
            _ => {}
        }
    }

    // Candidate items (flattened), with the external names each one needs. `ty` holds the type
    // name a struct/union/type-alias defines (None for consts/impls, which define no type).
    struct Cand { label: String, ty: Option<String>, ts: TokenStream, needs: BTreeSet<String> }
    let mut cands: Vec<Cand> = Vec::new();
    for it in items {
        // `extra_need` ties a `Default` impl to its self-type surviving the fixpoint.
        let (label, ty, extra_need, mut node): (String, Option<String>, Option<String>, Item) = match it {
            Item::Struct(s) if s.ident.to_string().ends_with("_Vtbl") => continue,
            Item::Struct(s) => (s.ident.to_string(), Some(s.ident.to_string()), None, Item::Struct(s.clone())),
            Item::Union(u) => (u.ident.to_string(), Some(u.ident.to_string()), None, Item::Union(u.clone())),
            Item::Type(t) => (t.ident.to_string(), Some(t.ident.to_string()), None, Item::Type(t.clone())),
            Item::Const(c) => (c.ident.to_string(), None, None, Item::Const(c.clone())),
            // `impl Default for <value type>` (all use `zeroed()`); drop method wrappers, bitops, etc.
            Item::Impl(i) if is_default_impl(i) => {
                let n = impl_self_name(i).unwrap();
                (n.clone(), None, Some(n), Item::Impl(i.clone()))
            }
            _ => continue,
        };
        strip_item_attrs(&mut node);
        Rewrite { ifaces: &ifaces }.visit_item_mut(&mut node);
        let mut needs = BTreeSet::new();
        Collect(&mut needs).visit_item(&node);
        if let Some(t) = &ty { needs.remove(t); } // a type may reference itself
        needs.extend(extra_need);
        cands.push(Cand { label, ty, ts: quote!(#node), needs });
    }

    // Fixpoint over type-defining items: a type survives only if every type it needs survives
    // (or is in the prelude). Dropping one cascades to its dependents.
    let mut keep: BTreeSet<String> = defined.clone();
    loop {
        let before = keep.len();
        keep = cands.iter()
            .filter(|c| c.ty.as_ref().is_some_and(|t| keep.contains(t) && c.needs.iter().all(|d| resolvable(d, &keep))))
            .map(|c| c.ty.clone().unwrap())
            .collect();
        if keep.len() == before { break; }
    }
    // Emit every candidate whose needs all resolve against the surviving type set.
    let mut skipped: Vec<String> = Vec::new();
    let body: TokenStream = cands
        .iter()
        .filter_map(|c| {
            if c.needs.iter().all(|d| resolvable(d, &keep)) {
                Some(c.ts.clone())
            } else {
                let missing: Vec<_> = c.needs.iter().filter(|d| !resolvable(d, &keep)).cloned().collect();
                skipped.push(format!("{} (needs {})", c.label, missing.join(", ")));
                None
            }
        })
        .collect();

    let header = "// @generated by crates/d2d-types/gen from windows 0.62.2 — DO NOT EDIT. See README.\n";
    let file: syn::File = syn::parse2(quote! {
        #![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code)]
        use crate::prelude::*;
        #body
    }).unwrap();
    std::fs::write(format!("{out}/abi.rs"), format!("{header}{}", prettyplease::unparse(&file))).unwrap();
    eprintln!("abi.rs: {} items kept; {} value types skipped:", keep.len(), skipped.len());
    for s in &skipped { eprintln!("  skip {s}"); }
}

fn resolvable(name: &str, keep: &BTreeSet<String>) -> bool {
    PRIMS.contains(&name) || PRELUDE.contains(&name) || keep.contains(name)
}

fn impl_self_name(i: &syn::ItemImpl) -> Option<String> {
    if let Type::Path(p) = &*i.self_ty { Some(p.path.segments.last()?.ident.to_string()) } else { None }
}

fn is_default_impl(i: &syn::ItemImpl) -> bool {
    matches!(&i.trait_, Some((_, p, _)) if p.segments.last().is_some_and(|s| s.ident == "Default"))
}

/// Flatten any multi-segment path (other than core/std/windows_core/windows_numerics) to its last
/// segment (collapsing the four Win32 namespaces into one flat module), and replace every
/// COM-interface-typed field with `ComPtr` (all pointer-sized, so layout is preserved).
struct Rewrite<'a> { ifaces: &'a BTreeSet<String> }
impl<'a> VisitMut for Rewrite<'a> {
    fn visit_type_mut(&mut self, t: &mut Type) {
        if self.is_com(t) {
            *t = syn::parse_quote!(ComPtr);
            return;
        }
        visit_mut::visit_type_mut(self, t);
    }
    fn visit_path_mut(&mut self, p: &mut syn::Path) {
        visit_mut::visit_path_mut(self, p);
        if p.segments.len() > 1 {
            let root = p.segments.first().unwrap().ident.to_string();
            let last = p.segments.last().unwrap().ident.to_string();
            // Keep core/std/windows_* paths, except the string pointers windows-core gates behind
            // `cfg(windows)` — collapse those to the portable prelude copies.
            let keep_root = matches!(root.as_str(), "windows_core" | "windows_numerics" | "core" | "std" | "alloc")
                && !(root == "windows_core" && matches!(last.as_str(), "PCWSTR" | "PCSTR"));
            if !keep_root {
                let seg = p.segments.last().unwrap().clone();
                p.segments = std::iter::once(seg).collect();
                p.leading_colon = None;
            }
        }
    }
    fn visit_field_mut(&mut self, f: &mut syn::Field) {
        f.attrs.retain(keep_attr);
        visit_mut::visit_field_mut(self, f);
    }
}
impl<'a> Rewrite<'a> {
    /// A COM reference: an interface path, `*T`/`Option<T>`/`ManuallyDrop<T>` wrapping one.
    fn is_com(&self, t: &Type) -> bool {
        match t {
            Type::Ptr(p) => self.is_com(&p.elem),
            Type::Path(tp) => {
                let seg = tp.path.segments.last().unwrap();
                let name = seg.ident.to_string();
                if self.ifaces.contains(&name) {
                    return true;
                }
                if matches!(name.as_str(), "Option" | "ManuallyDrop") {
                    if let syn::PathArguments::AngleBracketed(a) = &seg.arguments {
                        if let Some(syn::GenericArgument::Type(inner)) = a.args.first() {
                            return self.is_com(inner);
                        }
                    }
                }
                false
            }
            _ => false,
        }
    }
}

/// Collect single-segment type names referenced by an item (what it needs defined), skipping
/// std wrappers and lowercase idents (fields, locals, params).
struct Collect<'a>(&'a mut BTreeSet<String>);
impl<'a, 'ast> Visit<'ast> for Collect<'a> {
    fn visit_attribute(&mut self, _: &'ast syn::Attribute) {} // repr/derive are not type references
    fn visit_path(&mut self, p: &'ast syn::Path) {
        syn::visit::visit_path(self, p);
        if p.segments.len() == 1 {
            let id = p.segments[0].ident.to_string();
            if !IGNORE.contains(&id.as_str()) && id.starts_with(|c: char| c.is_uppercase()) {
                self.0.insert(id);
            }
        }
    }
}

fn keep_attr(a: &syn::Attribute) -> bool {
    a.path().is_ident("repr") || a.path().is_ident("derive")
}
fn strip_item_attrs(it: &mut Item) {
    let attrs = match it {
        Item::Struct(s) => &mut s.attrs,
        Item::Union(u) => &mut u.attrs,
        Item::Type(t) => &mut t.attrs,
        Item::Const(c) => &mut c.attrs,
        Item::Impl(i) => &mut i.attrs,
        _ => return,
    };
    attrs.retain(keep_attr);
}

// ---- interface inventory -> interfaces.rs ------------------------------------------------------

struct Iface { iid: u128, base: String, methods: Vec<String> }

fn entry<'a>(m: &'a mut BTreeMap<String, Iface>, name: &str) -> &'a mut Iface {
    m.entry(name.to_string()).or_insert_with(|| Iface { iid: 0, base: String::new(), methods: vec![] })
}

fn emit_inventory(items: &[&Item], out: &str) {
    let mut ifaces: BTreeMap<String, Iface> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();

    for it in items {
        if let Item::Macro(m) = it {
            let last = m.mac.path.segments.last().unwrap().ident.to_string();
            let toks = m.mac.tokens.to_string();
            let parts: Vec<String> = top_level_commas(&toks);
            if last == "define_interface" {
                let name = parts[0].replace(' ', "");
                let iid = u128::from_str_radix(&parts[2].replace(['_', ' ', 'x'], "")[1..], 16).unwrap();
                order.push(name.clone());
                entry(&mut ifaces, &name).iid = iid;
            } else if last == "interface_hierarchy" {
                let name = parts[0].replace(' ', "");
                let base = parts.last().unwrap().rsplit("::").next().unwrap().replace(' ', "");
                entry(&mut ifaces, &name).base = base;
            }
        } else if let Item::Struct(s) = it {
            let n = s.ident.to_string();
            if let Some(iface) = n.strip_suffix("_Vtbl") {
                if let Fields::Named(f) = &s.fields {
                    entry(&mut ifaces, iface).methods = f.named.iter()
                        .filter(|fd| matches!(fd.vis, Visibility::Public(_)) && fd.ident.as_ref().unwrap() != "base__")
                        .map(|fd| fd.ident.as_ref().unwrap().to_string())
                        .collect();
                }
            }
        }
    }

    // Resolve slot counts (IUnknown = 3, plus the two external COM bases) with memoisation.
    let mut slots: BTreeMap<String, u32> = BTreeMap::new();
    slots.insert("IUnknown".into(), 3);
    for (name, _, s, _) in EXTERN_BASES { slots.insert((*name).into(), *s); }
    fn resolve(n: &str, ifaces: &BTreeMap<String, Iface>, memo: &mut BTreeMap<String, u32>) -> u32 {
        if let Some(&s) = memo.get(n) { return s; }
        let f = &ifaces[n];
        let s = resolve(&f.base, ifaces, memo) + f.methods.len() as u32;
        memo.insert(n.into(), s);
        s
    }
    for n in &order { resolve(n, &ifaces, &mut slots); }

    assert_eq!(order.len(), 310, "expected 310 interfaces");
    assert_eq!(order.iter().map(|n| ifaces[n].methods.len()).sum::<usize>(), 1458, "expected 1458 methods");

    let mut defs: Vec<TokenStream> = Vec::new();
    for n in &order {
        let f = &ifaces[n];
        let ident = syn::Ident::new(n, proc_macro2::Span::call_site());
        let iid = Literal::u128_unsuffixed(f.iid);
        let total = *slots.get(n).unwrap();
        let base_slots = *slots.get(&f.base).unwrap();
        let base_const = if f.base == "IUnknown" {
            "IUNKNOWN".to_string()
        } else if let Some((_, _, _, c)) = EXTERN_BASES.iter().find(|(n, ..)| *n == f.base) {
            c.to_string()
        } else {
            f.base.clone()
        };
        let b = syn::Ident::new(&base_const, proc_macro2::Span::call_site());
        let base = quote!(Some(&#b));
        let methods = f.methods.iter().enumerate().map(|(i, m)| {
            let slot = base_slots + i as u32;
            quote!(Method { name: #m, slot: #slot, support: Support::Unsupported })
        });
        defs.push(quote! {
            pub static #ident: Interface = Interface {
                name: #n, iid: windows_core::GUID::from_u128(#iid), slots: #total,
                base: #base, methods: &[#(#methods),*],
            };
        });
    }
    let refs = order.iter().map(|n| syn::Ident::new(n, proc_macro2::Span::call_site()));
    let qiid = Literal::u128_unsuffixed(0x00000000_0000_0000_c000_000000000046u128);
    // External COM bases (not D2D/DWrite/WIC/DXGI, so not in INTERFACES) referenced as `base`.
    let externs = EXTERN_BASES.iter().map(|(name, iid, s, cident)| {
        let ident = syn::Ident::new(cident, proc_macro2::Span::call_site());
        let iid = Literal::u128_unsuffixed(*iid);
        quote! {
            pub static #ident: Interface = Interface {
                name: #name, iid: windows_core::GUID::from_u128(#iid), slots: #s,
                base: None, methods: &[],
            };
        }
    });
    let file: syn::File = syn::parse2(quote! {
        #![allow(non_upper_case_globals)]
        use crate::{Interface, Method, Support};
        pub static IUNKNOWN: Interface = Interface {
            name: "IUnknown", iid: windows_core::GUID::from_u128(#qiid), slots: 3, base: None,
            methods: &[
                Method { name: "QueryInterface", slot: 0, support: Support::Unsupported },
                Method { name: "AddRef", slot: 1, support: Support::Unsupported },
                Method { name: "Release", slot: 2, support: Support::Unsupported },
            ],
        };
        #(#externs)*
        #(#defs)*
        pub static INTERFACES: &[&Interface] = &[#(&#refs),*];
    }).unwrap();
    let header = "// @generated by crates/d2d-types/gen from windows 0.62.2 — DO NOT EDIT. See README.\n";
    std::fs::write(format!("{out}/interfaces.rs"), format!("{header}{}", prettyplease::unparse(&file))).unwrap();
    eprintln!("interfaces.rs: {} interfaces", order.len());
}

/// Split a macro token string on top-level commas (there are no nested delimiters in these macros).
fn top_level_commas(s: &str) -> Vec<String> {
    s.split(',').map(|p| p.trim().to_string()).collect()
}
