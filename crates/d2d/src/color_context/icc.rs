//! ICC profiles used by the ColorManagement effect; LUT/CLUT models live in `lut`.
//! NORMAL/FAST evaluate float curves into a 32³/16³ byte cube. WARP interpolates that cube in
//! red, green, blue order at Q8 coordinates and keeps value×128 until the alpha-mode store.
//! BEST samples 1024-entry UNORM16 curves at Q8 coordinates around a combined color matrix.
//! The com-g13 oracle scenes cover all three paths, both directions and alpha modes, and
//! perceptual versus absolute-colorimetric media-white scaling for four RGB gamuts.
use super::{ColorContext, Space, N, FORWARD};

struct Profile {
    matrix: [[f32; 3]; 3],
    white: [f32; 3],
    curves: [Curve; 3],
}
fn u32be(data: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(data.get(at..at.checked_add(4)?)?.try_into().ok()?))
}
fn fixed(data: &[u8], at: usize) -> Option<f32> { Some(u32be(data, at)? as i32 as f32 / 65536.0) }
fn tag<'a>(data: &'a [u8], name: &[u8; 4]) -> Option<&'a [u8]> {
    let count = u32be(data, 128)? as usize;
    let entries = data.get(132..132usize.checked_add(count.checked_mul(12)?)?)?;
    for e in entries.chunks_exact(12) {
        if &e[..4] == name {
            let off = u32be(e, 4)? as usize;
            let len = u32be(e, 8)? as usize;
            return data.get(off..off.checked_add(len)?);
        }
    }
    None
}
fn xyz(data: &[u8]) -> Option<[f32; 3]> {
    if data.get(..4)? != b"XYZ " { return None; }
    Some([fixed(data, 8)?, fixed(data, 12)?, fixed(data, 16)?])
}
enum Curve {
    Param([f32; 7]),
    Table(Vec<f32>),
    Gpu([u16;N]),
    GpuTable(Vec<u16>, u32),
}
impl Curve {
    fn apply(&self, x: f32, inverse: bool) -> f32 {
        match self {
            Self::Param([g,a,b,c,d,e,f]) => {
                let y = if inverse {
                    if x < c*d+f {
                        if *c == 0.0 { 0.0 } else { (x-f)/c }
                    } else { (crate::crt::system_crt::powf(x-e,1.0/g)-b)/a }
                } else if x < *d { c*x+f } else { crate::crt::system_crt::powf(a*x+b,*g)+e };
                y.clamp(0.0,1.0)
            }
            Self::GpuTable(t,bits) => {
                let n=t.len();
                let uv=x.clamp(0.0,1.0)*((n-1) as f32/n as f32)+0.5/n as f32;
                let q=((uv*n as f32-0.5)*256.0).floor() as i32;
                let i=(q>>8).clamp(0,n as i32-1) as usize;
                let j=(i+1).min(n-1);let w=q&255;
                if *bits==8 {
                    let v=((t[j] as i32-t[i] as i32)*w+((t[i] as i32)<<8))>>1;
                    crate::sampling::norm_v128(v)
                } else {
                    let f=w as f32/256.0;
                    {
                        let a=t[i] as f32*(1.0/65535.0);let b=t[j] as f32*(1.0/65535.0);
                        a+(b-a)*f
                    }
                }
            }
            Self::Gpu(t) => {
                let p = (x.clamp(0.0,1.0)*(N-1) as f32*256.0).floor()/256.0;
                let i = (p as usize).min(N-1);
                let f = p-i as f32;
                (t[i] as f32*(1.0-f)+t[(i+1).min(N-1)] as f32*f)/65535.0
            }
            Self::Table(t) => {
                let p = x.clamp(0.0,1.0)*(t.len()-1) as f32;
                let i = (p as usize).min(t.len()-1);
                t[i]+(t[(i+1).min(t.len()-1)]-t[i])*(p-i as f32)
            }
        }
    }
    fn gpu(self, inverse: bool) -> Self {
        let values: [u16; N] = std::array::from_fn(|i| (self.apply(i as f32/(N-1) as f32, inverse)*65535.0+0.5).clamp(0.0,65535.0) as u16);
        let values = if inverse && matches!(self, Self::Table(_)) { super::invert_curve(&values) } else { values };
        Self::Gpu(values)
    }
    fn sampled(self, inverse: bool) -> Self {
        if let Self::Param(_) = &self {
            Self::Table((0..N).map(|i|self.apply(i as f32*(1.0/(N-1) as f32),inverse)).collect())
        } else if inverse { self.inverted() } else { self }
    }
    fn inverted(self) -> Self {
        let Self::Table(t) = self else { return self; };
        let mut j = 0usize;
        Self::Table((0..N).map(|i| {
            let y = i as f32/(N-1) as f32;
            while j < t.len() && t[j] < y { j += 1; }
            if j == 0 { 0.0 } else if j == t.len() { 1.0 } else {
                ((y-t[j-1])/(t[j]-t[j-1])+j as f32-1.0)/(t.len()-1) as f32
            }
        }).collect())
    }
}
fn curve(data: &[u8]) -> Option<Curve> {
    let kind = data.get(..4)?;
    if kind == b"curv" {
        let n = u32be(data,8)? as usize;
        let bytes = data.get(12..12usize.checked_add(n.checked_mul(2)?)?)?;
        return Some(match n {
            0 => Curve::Param([1.0,1.0,0.0,0.0,0.0,0.0,0.0]),
            1 => Curve::Param([u16::from_be_bytes([bytes[0],bytes[1]]) as f32/256.0,1.0,0.0,0.0,0.0,0.0,0.0]),
            _ => Curve::Table(bytes.chunks_exact(2).map(|v|u16::from_be_bytes([v[0],v[1]]) as f32/65535.0).collect()),
        });
    }
    if kind != b"para" { return None; }
    let ty = u16::from_be_bytes(data.get(8..10)?.try_into().ok()?) as usize;
    let n = *[1,3,4,5,7].get(ty)?;
    let mut p = [0.0;7];
    for (i,v) in p.iter_mut().enumerate().take(n) { *v=fixed(data,12+i*4)?; }
    match ty {
        0 => p[1]=1.0,
        1 => p[4]=-p[2]/p[1],
        2 => { p[5]=p[3]; p[6]=p[3]; p[3]=0.0; p[4]=-p[2]/p[1]; }
        _ => (),
    }
    if p[0] <= 0.0 || p[1] == 0.0 { return None; }
    Some(Curve::Param(p))
}
impl Profile {
    fn new(ctx: &ColorContext) -> Option<Self> {
        if ctx.space==Space::Custom && [b"A2B0",b"A2B1",b"A2B2",b"B2A0",b"B2A1",b"B2A2"].iter().any(|key|tag(&ctx.profile,*key).is_some()) {return None;}
        Self::matrix(ctx)
    }
    fn matrix(ctx: &ColorContext) -> Option<Self> {
        if ctx.is_simple() { return None; }
        if ctx.space != Space::Custom {
            // Named sRGB/scRGB contexts expose these s15Fixed16 colorants and D65 media white.
            let matrix = [[28578.,25241.,9376.],[14581.,46981.,3972.],[912.,6362.,46799.]].map(|r| r.map(|x|x/65536.0));
            return Some(Self { matrix, white: [62289.0/65536.0,1.0,71372.0/65536.0], curves: std::array::from_fn(|_| if ctx.space == Space::Srgb { Curve::Table(FORWARD.iter().map(|v|*v as f32/65535.0).collect()) } else { Curve::Param([1.0,1.0,0.0,0.0,0.0,0.0,0.0]) }) });
        }
        let data = &ctx.profile;
        if data.get(36..40)? != b"acsp" || data.get(16..20)? != b"RGB " || data.get(20..24)? != b"XYZ " { return None; }
        let r = xyz(tag(data,b"rXYZ")?)?;
        let g = xyz(tag(data,b"gXYZ")?)?;
        let b = xyz(tag(data,b"bXYZ")?)?;
        Some(Self { matrix: std::array::from_fn(|i|[r[i],g[i],b[i]]), white: xyz(tag(data,b"wtpt")?)?, curves: [curve(tag(data,b"rTRC")?)?,curve(tag(data,b"gTRC")?)?,curve(tag(data,b"bTRC")?)?] })
    }
}
fn inverse(m: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let cof: [[f32; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| {
        let (a,b,c,d) = ((i+1)%3,(i+2)%3,(j+1)%3,(j+2)%3);
        m[a][c]*m[b][d]-m[a][d]*m[b][c]
    }));
    let det = (m[0][0]*cof[0][0]+m[0][1]*cof[0][1])+m[0][2]*cof[0][2];
    if det == 0.0 || !det.is_finite() { return None; }
    Some(std::array::from_fn(|i|std::array::from_fn(|j|cof[j][i]/det)))
}

mod lut;

pub(super) enum Transform {
    Matrix(MatrixTransform),
    Lut(lut::Transform),
}
impl Transform {
    pub(super) fn new(src: &ColorContext, dst: &ColorContext, intents: [u32; 2], quality: u32) -> Option<Self> {
        if let Some(t) = MatrixTransform::new(src, dst, intents, quality) { return Some(Self::Matrix(t)); }
        lut::Transform::new(src, dst, intents, quality).map(Self::Lut)
    }
    pub(super) fn apply(&self, p: [f32; 4], straight: bool) -> [f32; 4] {
        match self {
            Self::Lut(t) => t.apply(p, straight),
            Self::Matrix(t) => {
                let a = p[3];
                let rgb = if straight { [p[0],p[1],p[2]] } else if a != 0.0 { [p[0]/a,p[1]/a,p[2]/a] } else { [0.0;3] };
                let rgb = t.apply(rgb, straight);
                let k = if straight { 1.0 } else { a };
                [rgb[0]*k,rgb[1]*k,rgb[2]*k,a]
            }
        }
    }
}

pub(super) struct MatrixTransform {
    cube: Vec<[u8; 3]>,
    edge: usize,
    best: bool,
    source_matrix: [[f32;3];3],
    dest_matrix: [[f32;3];3],
    scale: [f32;3],
    source: [Curve; 3],
    matrix: [[f32; 3]; 3],
    dest: [Curve; 3],
}
impl MatrixTransform {
    pub(super) fn new(src: &ColorContext, dst: &ColorContext, intents: [u32; 2], quality: u32) -> Option<Self> {
        let (s,d) = (Profile::new(src)?,Profile::new(dst)?);
        let inv = inverse(d.matrix)?;
        let d50 = [0.9642,1.0,0.8249];
        let scale: [f32;3] = std::array::from_fn(|i| (if intents[0] == 3 { s.white[i]/d50[i] } else { 1.0 }) * (if intents[1] == 3 { d50[i]/d.white[i] } else { 1.0 }));
        let matrix = std::array::from_fn(|i|std::array::from_fn(|j| {
            (inv[i][0]*scale[0]*s.matrix[0][j]+inv[i][1]*scale[1]*s.matrix[1][j])+inv[i][2]*scale[2]*s.matrix[2][j]
        }));
        let edge = if quality == 0 { 16 } else { 32 };
        let mut result = Self { cube: Vec::new(), edge, best: quality == 2, source_matrix: s.matrix, dest_matrix: inv, scale,
            source: s.curves.map(|c|if quality == 2 { c.gpu(false) } else { c.sampled(false) }), matrix,
            dest: d.curves.map(|c|if quality == 2 { c.gpu(true) } else { c.sampled(true) }) };
        if quality == 2 { return Some(result); }
        let step = 1.0 / (edge-1) as f32;
        for r in 0..edge { for g in 0..edge { for b in 0..edge {
            result.cube.push(result.direct([r as f32 * step, g as f32 * step, b as f32 * step]).map(|v|(v*255.0+0.5).clamp(0.0,255.0) as u8));
        } } }
        Some(result)
    }
    pub(super) fn apply(&self, rgb: [f32; 3], straight: bool) -> [f32; 3] {
        if self.best { return self.direct(rgb); }
        let axes = rgb.map(|v| {
            let q = (v.clamp(0.0, 1.0) * (self.edge-1) as f32 * 256.0).floor() as i32;
            let lo = (q >> 8).min(self.edge as i32-2);
            (lo as usize, q - lo * 256)
        });
        std::array::from_fn(|c| {
            let at = |r: usize,g: usize,b: usize| self.cube[(axes[0].0+r)*self.edge*self.edge+(axes[1].0+g)*self.edge+axes[2].0+b][c] as i32;
            let x = |g,b| ((at(1,g,b)-at(0,g,b))*axes[0].1+(at(0,g,b)<<8))>>1;
            let y = |b| x(0,b)+((x(1,b)-x(0,b))*axes[1].1>>8);
            let v = y(0)+((y(1)-y(0))*axes[2].1>>8);
            if straight { (v>>7) as f32/255.0 } else { crate::sampling::norm_v128(v) }
        })
    }
    fn direct(&self, rgb: [f32; 3]) -> [f32; 3] {
        let v: [f32;3] = std::array::from_fn(|i|self.source[i].apply(rgb[i], false));
        let dot = |row: [f32;3], v: [f32;3]|(row[0]*v[0]+row[1]*v[1])+row[2]*v[2];
        let rgb = if self.best { self.matrix.map(|row|dot(row,v)) } else {
            let xyz = std::array::from_fn(|i|dot(self.source_matrix[i],v)*self.scale[i]);
            self.dest_matrix.map(|row|dot(row,xyz))
        };
        std::array::from_fn(|i| {
            let x = rgb[i];
            self.dest[i].apply(x, true)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_tag_ranges_are_rejected() {
        let suite = std::env::var_os("D2D_SUITE").map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance/suite"));
        let Ok(original) = std::fs::read(suite.join("fixtures/com-g13/p3.icc")) else { return };
        for len in 0..original.len() {
            // Every prefix must remain safe, including truncation inside a tag header or curve.
            let _ = Profile::new(&ColorContext::from_profile_bytes(&original[..len]));
        }
        let tag = original.windows(4).position(|v|v == b"rXYZ").unwrap();
        for offset in [128,tag+4,tag+8] {
            let mut bytes = original.to_vec();
            bytes[offset..offset+4].copy_from_slice(&u32::MAX.to_be_bytes());
            assert!(Profile::new(&ColorContext::from_profile_bytes(&bytes)).is_none());
        }
    }
}
