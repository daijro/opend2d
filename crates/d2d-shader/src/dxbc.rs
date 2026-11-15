//! DXBC container reader: the little-endian chunk table, the shader token chunk (`SHEX`/`SHDR`),
//! and the input/output signature chunks (`ISGN`/`OSGN` and their `ISG1`/`OSG1` variants).
//!
//! Only what the CPU executor needs is parsed. The constant buffer is bound as raw bytes by the
//! caller (D2D's `SetPixelShaderConstantBuffer`), so `RDEF` is not decoded here.

use crate::sm4;

/// One element of an input or output signature.
#[derive(Clone, PartialEq, Debug)]
pub struct SigElem {
    pub semantic: String,
    pub semantic_index: u32,
    /// System-value type (0 = none/arbitrary, 1 = SV_Position, …).
    pub system_value: u32,
    pub register: u32,
    /// Component mask (x=1 y=2 z=4 w=8).
    pub mask: u8,
}

/// A parsed shader: the decoded program plus its signatures.
#[derive(Clone, PartialEq, Debug)]
pub struct Shader {
    pub program: sm4::Program,
    pub inputs: Vec<SigElem>,
    pub outputs: Vec<SigElem>,
    /// Per input index (the `TEXCOORD`n register), whether Direct2D's shader linker baked it as a
    /// **simple** input — `__d2dinput_color`n, `D2D1_PIXEL_OPTIONS_TRIVIAL_SAMPLING`, pre-sampled 1:1
    /// at the output pixel's own scene position — rather than **complex** (`__d2dinput_uv`n, a texcoord
    /// the shader samples). Both carry the identical top-level `TEXCOORD`n signature, so only the
    /// linker-baked parameter name tells them apart. `false` for a classic (non-linked) shader with no
    /// such marker, which keeps the normalized sampler path. See `effects/shader.rs`.
    pub input_simple: Vec<bool>,
}

fn u32le(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// Return each chunk as `(tag, payload)` where payload excludes the 8-byte tag+size header.
fn chunks(bytes: &[u8]) -> Result<Vec<(&[u8], &[u8])>, &'static str> {
    if bytes.len() < 32 || &bytes[0..4] != b"DXBC" {
        return Err("not a DXBC container");
    }
    let count = u32le(bytes, 28) as usize;
    let table = 32;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let off = u32le(bytes, table + i * 4) as usize;
        if off + 8 > bytes.len() {
            return Err("chunk offset out of range");
        }
        let size = u32le(bytes, off + 4) as usize;
        let start = off + 8;
        if start + size > bytes.len() {
            return Err("chunk size out of range");
        }
        out.push((&bytes[off..off + 4], &bytes[start..start + size]));
    }
    Ok(out)
}

/// Parse a signature chunk payload. `wide` is true for the `ISG1`/`OSG1` variant, whose elements
/// carry a leading stream index (the name offset is then the second field).
fn parse_sig(p: &[u8], wide: bool) -> Vec<SigElem> {
    let count = u32le(p, 0) as usize;
    // ISGN/OSGN elements are 24 bytes; the ISG1/OSG1 variant is 32 (a leading u32 stream index and
    // a trailing u32 min-precision field), with the name offset moved to the second u32.
    let stride = if wide { 32 } else { 24 };
    let base = 8;
    let name_field = if wide { 4 } else { 0 };
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let e = base + i * stride;
        if e + stride > p.len() {
            break;
        }
        let name_off = u32le(p, e + name_field) as usize;
        if name_off >= p.len() {
            continue;
        }
        let mut end = name_off;
        while end < p.len() && p[end] != 0 {
            end += 1;
        }
        let semantic = String::from_utf8_lossy(&p[name_off..end]).into_owned();
        let f = e + name_field + 4;
        out.push(SigElem {
            semantic,
            semantic_index: u32le(p, f),
            system_value: u32le(p, f + 4),
            register: u32le(p, f + 12),
            mask: p[f + 16],
        });
    }
    out
}

impl Shader {
    /// Parse a DXBC pixel-shader container.
    pub fn parse(bytes: &[u8]) -> Result<Shader, &'static str> {
        let chunks = chunks(bytes)?;
        let find = |tags: &[&[u8]]| chunks.iter().find(|(t, _)| tags.contains(t)).map(|(_, p)| *p);
        let code = find(&[b"SHEX", b"SHDR"]).ok_or("no shader-code chunk")?;
        if code.len() % 4 != 0 {
            return Err("odd-length code chunk");
        }
        let tokens: Vec<u32> = code.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        if tokens.len() < 2 {
            return Err("truncated code chunk");
        }
        let program = sm4::decode(&tokens);
        let inputs = find(&[b"ISGN"]).map(|p| parse_sig(p, false))
            .or_else(|| find(&[b"ISG1"]).map(|p| parse_sig(p, true)))
            .unwrap_or_default();
        let outputs = find(&[b"OSGN"]).map(|p| parse_sig(p, false))
            .or_else(|| find(&[b"OSG1"]).map(|p| parse_sig(p, true)))
            .unwrap_or_default();
        // Per TEXCOORD input, read Direct2D's linker-baked parameter name: `__d2dinput_color`n (a
        // simple input, pre-sampled 1:1) vs `__d2dinput_uv`n (a complex input, a texcoord). These live
        // in the container as function-parameter names, so scan the raw bytes by input index.
        let mut input_simple: Vec<bool> = Vec::new();
        for e in inputs.iter().filter(|e| e.semantic.starts_with("TEXCOORD")) {
            let idx = e.semantic_index as usize;
            if input_simple.len() <= idx {
                input_simple.resize(idx + 1, false);
            }
            input_simple[idx] = has_token(bytes, format!("__d2dinput_color{idx}").as_bytes());
        }
        Ok(Shader { program, inputs, outputs, input_simple })
    }

    /// Whether input `i` is a **simple** (trivially-sampled, pre-sampled 1:1) linked input; see
    /// [`Shader::input_simple`]. `false` for an out-of-range index or an unlinked shader.
    pub fn is_input_simple(&self, i: u32) -> bool {
        self.input_simple.get(i as usize).copied().unwrap_or(false)
    }
}

/// Whether `hay` contains `needle` not immediately followed by an ASCII digit, so `__d2dinput_color1`
/// does not spuriously match inside `__d2dinput_color10`.
fn has_token(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len())
        .enumerate()
        .any(|(i, w)| w == needle && hay.get(i + needle.len()).map_or(true, |b| !b.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use super::has_token;

    #[test]
    fn token_scan_respects_digit_boundary() {
        // `color1` must not match inside `color10`, but must match at the string end or before a non-digit.
        assert!(has_token(b"...__d2dinput_color1\0...", b"__d2dinput_color1"));
        assert!(has_token(b"__d2dinput_color1", b"__d2dinput_color1")); // at end of buffer
        assert!(!has_token(b"__d2dinput_color10\0", b"__d2dinput_color1")); // trailing digit
        assert!(has_token(b"a __d2dinput_color10", b"__d2dinput_color10"));
        assert!(!has_token(b"__d2dinput_uv0", b"__d2dinput_color0"));
    }
}
