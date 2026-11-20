//! Native d2d1.dll 10.0.26100.8875 colour-profile parser, CPU colour translator and ND-LUT pipeline.
use super::{ColorContext, Curve, Profile, Space, curve, fixed, inverse, tag, u32be, xyz};

struct Clut {
    channels: usize,
    output: usize,
    grid: Vec<usize>,
    values: Vec<f32>,
    bits: u32,
}
impl Clut {
    // The ND-LUT uses a simplex in the cell: descending fractions select successive axis corners.
    fn apply(&self, v: [f32;4]) -> [f32;4] {
        let mut stride = [0usize;4];
        let mut step = self.output;
        for c in (0..self.channels).rev() { stride[c] = step; step *= self.grid[c]; }
        let mut base = 0;
        let mut frac = [0.0;4];
        for c in 0..self.channels {
            let p = v[c].clamp(0.0,1.0) * (self.grid[c]-1) as f32;
            let i = (p as usize).min(self.grid[c]-2);
            base += i * stride[c];
            frac[c] = p-i as f32;
        }
        let mut axes: Vec<_> = (0..self.channels).collect();
        axes.sort_by(|&a,&b| frac[b].total_cmp(&frac[a]).then(b.cmp(&a)));
        let mut out = [0.0;4];
        for c in 0..self.output {
            let mut i = base+c;
            out[c] = self.values[i];
            for &a in &axes {
                out[c] += (self.values[i+stride[a]]-self.values[i])*frac[a];
                i += stride[a];
            }
        }
        out
    }
    fn gpu(&self, v:[f32;4]) -> [f32;4] {
        if self.channels==3 {
            // Native BEST CLUT shader evaluates four tetrahedron corners in this weighted order.
            let stride=[self.grid[1]*self.grid[2]*self.output,self.grid[2]*self.output,self.output];
            let mut base=0;let mut f=[0.0f32;3];
            for c in 0..3 {let p=v[c].clamp(0.0,1.0)*(self.grid[c]-1) as f32;let i=(p.floor() as usize).min(self.grid[c]-2);base+=i*stride[c];f[c]=p-i as f32;}
            let mut axes=[0usize,1,2];axes.sort_by(|&a,&b|f[b].total_cmp(&f[a]).then(b.cmp(&a)));
            let hi=f[axes[0]];let lo=f[axes[2]];let mid=((f[2]+f[1])+f[0]-hi)-lo;
            let one=base+stride[axes[0]];let two=one+stride[axes[1]];let top=two+stride[axes[2]];
            let mut out=[0.0;4];
            for c in 0..self.output {out[c]=((self.values[one+c]*(hi-mid)+self.values[base+c]*(1.0-hi))+self.values[two+c]*(mid-lo))+self.values[top+c]*lo;}
            return out;
        }
        // Four-channel native LUTs pack Y and K along X, filter C/M/K in hardware,
        // then interpolate the two Y slices with the unquantized shader fraction.
        let stride=[self.grid[1]*self.grid[2]*self.grid[3]*self.output,self.grid[2]*self.grid[3]*self.output,self.grid[3]*self.output,self.output];
        let y=v[2].clamp(0.0,1.0)*(self.grid[2]-1) as f32;
        let yi=(y.floor() as usize).min(self.grid[2]-2);let yf=y-yi as f32;
        let mut out=[0.0;4];
        let sample=|ys:usize,ch:usize| {
            let mut axes=[(0usize,0i32);3];
            for (j,c) in [3usize,1,0].into_iter().enumerate() {
                let n=self.grid[c] as f32;
                let p=if c==3 {
                    let uv=((v[c].clamp(0.0,1.0)*(n-1.0)+(ys*self.grid[3]) as f32)+0.5)*(1.0/n)*(1.0/self.grid[2] as f32);
                    uv*(self.grid[2]*self.grid[3]) as f32-0.5-(ys*self.grid[3]) as f32
                }else{((v[c].clamp(0.0,1.0)*(n-1.0)+0.5)*(1.0/n))*n-0.5};
                // WARP rounds the RGBA16 3D sampler fraction, while its R16 curve sampler truncates.
                let q=(p*256.0+if self.bits==16{0.5}else{0.0}).floor().max(0.0) as i32;
                let i=(q>>8).min(self.grid[c] as i32-2);axes[j]=(i as usize,q-i*256);
            }
            let mut a=[0.0;4];
            for km in 0..4 {
                let idx=axes[2].0*stride[0]+(axes[1].0+(km&1))*stride[1]+ys*stride[2]+(axes[0].0+(km>>1))*stride[3]+ch;
                let (lo,hi)=(self.values[idx],self.values[idx+stride[0]]);let w=axes[2].1;
                a[km]=if self.bits==8 {((((hi*255.0).round() as i32-(lo*255.0).round() as i32)*w+(((lo*255.0).round() as i32)<<8))>>1) as f32}else{lo+(hi-lo)*(w as f32/256.0)};
            }
            let lerp=|a:f32,b:f32,w:i32|if self.bits==8 {a+(((b as i32-a as i32)*w)>>8) as f32}else{a+(b-a)*(w as f32/256.0)};
            let val=lerp(lerp(a[0],a[1],axes[1].1),lerp(a[2],a[3],axes[1].1),axes[0].1);
            if self.bits==8 {crate::sampling::norm_v128(val as i32)}else{val}
        };
        for c in 0..self.output {let a=sample(yi,c);out[c]=a+(sample(yi+1,c)-a)*yf;}
        out
    }

}

enum Stage {
    Curves(Vec<Curve>),
    Curves8(Vec<Curve>),
    GpuLut(Clut),
    Matrix([[f32;3];3]),
    Lut(Clut),
    Scale([f32;3]),
    Offset([f32;3]),
    GrayToXyz,
    XyzToGray,
    LabToXyz(bool, bool),
    XyzToLab(bool),
    Clamp,
}
struct Pipeline {
    channels: usize,
    legacy: bool,
    stages: Vec<Stage>,
}
impl Pipeline {
    fn apply(&self, mut v: [f32;4]) -> [f32;4] {
        for stage in &self.stages {
            v = match stage {
                Stage::Curves(curves) | Stage::Curves8(curves) => {
                    let mut out = v;
                    for (i,c) in curves.iter().enumerate() { out[i] = c.apply(v[i],false); }
                    out
                }
                Stage::Clamp => [v[0].max(0.0),v[1].max(0.0),v[2].max(0.0),v[3]],
                Stage::LabToXyz(mft2, gpu) => {
                    // ColorUtils::ConvertLabToXYZ / CIELabDecode, native 10.0.26100.8875.
                    let k=if *mft2 {1.00390625}else{1.0};
                    let l=if *gpu {v[0]*(k*100.0)}else{v[0]*k*100.0};
                    let a=if *gpu {v[1]*(k*255.0)-128.0}else{v[1]*k*255.0-128.0};
                    let b=if *gpu {v[2]*(k*255.0)-128.0}else{v[2]*k*255.0-128.0};
                    let fy=(l+16.0)*f32::from_bits(0x3c0d3dcb);
                    let decode=|x:f32|if x>f32::from_bits(if *gpu {0x3e53dcb1}else{0x3e53dcb0}){(x*x)*x}else{(x-f32::from_bits(0x3e0d3dcb))*f32::from_bits(0x3e038027)};
                    let mut out=[decode(a*0.002+fy)*0.9642,decode(fy),decode(fy-b*0.005)*0.8249,v[3]];
                    if !*gpu {for v in &mut out[..3] {*v=v.max(0.0);}}
                    out
                }
                Stage::XyzToLab(mft2) => {
                    let encode=|x:f32|if x>f32::from_bits(0x3c111aa7){crate::crt::system_crt::powf(x,f32::from_bits(0x3eaaaaab))}else{x*f32::from_bits(0x40f92f68)+f32::from_bits(0x3e0d3dcb)};
                    let x=encode(v[0].max(0.0)*(1.0/0.9642));let y=encode(v[1].max(0.0));let z=encode(v[2].max(0.0)*(1.0/0.8249));
                    let k=if *mft2 {256.0/257.0}else{1.0};
                    [((y*116.0-16.0).max(0.0)/100.0)*k,(((x-y)*500.0+128.0)/255.0)*k,(((y-z)*200.0+128.0)/255.0)*k,v[3]]
                }
                Stage::Matrix(m) => {
                    let r = m.map(|row|(row[0]*v[0]+row[1]*v[1])+row[2]*v[2]);
                    [r[0],r[1],r[2],v[3]]
                }
                Stage::Lut(lut) => lut.apply(v),
                Stage::GpuLut(lut) => lut.gpu(v),
                Stage::Scale(k) => [v[0]*k[0],v[1]*k[1],v[2]*k[2],v[3]],
                Stage::Offset(k) => [v[0]+k[0],v[1]+k[1],v[2]+k[2],v[3]],
                Stage::GrayToXyz => [v[0]*0.9642,v[0],v[0]*0.8249,v[3]],
                Stage::XyzToGray => [v[1],0.0,0.0,v[3]],
            };
        }
        v
    }
    fn gpu(mut self) -> Self {
        self.stages=self.stages.into_iter().map(|stage|match stage {
            Stage::Curves8(c) => Stage::Curves(c.into_iter().map(|c|match c {
                Curve::Table(t)=>Curve::GpuTable(t.into_iter().map(|v|(v*255.0+0.5).clamp(0.0,255.0) as u16).collect(),8),c=>c.gpu(false),
            }).collect()),
            Stage::Curves(c) => {
                // The original tag's precision is retained by the native resource texture.
                Stage::Curves(c.into_iter().map(|c|match c {
                    Curve::Table(t)=> {
                        let n=t.len().max(256);let c=Curve::Table(t);
                        Curve::GpuTable((0..n).map(|i|(c.apply(i as f32/(n-1) as f32,false)*65535.0+0.5).clamp(0.0,65535.0) as u16).collect(),16)
                    },
                    c@Curve::Gpu(_)=>c, c=>c.gpu(false),
                }).collect())
            }
            Stage::Lut(l)=>Stage::GpuLut(l),
            Stage::LabToXyz(mft2,_)=>Stage::LabToXyz(mft2,true),
            s=>s,
        }).collect();self
    }
    fn new(ctx: &ColorContext, reverse: bool, intent: u32, best: bool) -> Option<Self> {
        if ctx.space != Space::Custom {
            let profile = Profile::new(ctx)?;
            return Self::matrix(profile,reverse,intent,best);
        }
        let data = &ctx.profile;
        if data.get(36..40)? != b"acsp" || !matches!(data.get(20..24)?,b"XYZ "|b"Lab ") { return None; }
        let channels = match data.get(16..20)? { b"RGB " => 3, b"GRAY" => 1, b"CMYK" => 4, _ => return None };
        let mut key = if reverse { *b"B2A0" } else { *b"A2B0" };
        key[3] += if intent==3 {1}else{intent.min(2)} as u8;
        let selected = tag(data,&key).or_else(||tag(data,if reverse {b"B2A0"} else {b"A2B0"}));
        if let Some(lut) = selected {
            let expected=if reverse {[3,channels]}else{[channels,3]};
            if lut.get(8..10)?!=[expected[0] as u8,expected[1] as u8] {return None;}
            let mut stages = Vec::new();
            let white = xyz(tag(data,b"wtpt")?).unwrap_or([0.9642,1.0,0.8249]);
            let abs = std::array::from_fn(|i| if intent == 3 { if reverse {[0.9642,1.0,0.8249][i]/white[i]} else {white[i]/[0.9642,1.0,0.8249][i]} } else {1.0});
            let lab=data.get(20..24)?==b"Lab ";
            let mft2=lut.get(..4)?==b"mft2";
            if reverse {
                stages.push(Stage::Scale(abs.map(|v|if lab{v}else{v*0.50000763})));
                if lab {stages.push(Stage::XyzToLab(mft2));}
            }
            stages.extend(parse_lut(lut,reverse && !lab)?);
            if !reverse {
                if lab {stages.push(Stage::LabToXyz(mft2,false));}
                stages.push(Stage::Scale(abs.map(|v|if lab{v}else{v*1.9999695})));
            }
            return Some(Self { channels, stages, legacy: intent==0 && u32be(data,8)?<0x04000000 });
        }
        if channels == 1 {
            let curve = curve(tag(data,b"kTRC")?)?.sampled(reverse);
            let stages = if reverse {vec![Stage::XyzToGray,Stage::Curves(vec![curve])]} else {vec![Stage::Curves(vec![curve]),Stage::GrayToXyz]};
            return Some(Self {channels,stages,legacy:intent==0});
        }
        Self::matrix(Profile::matrix(ctx)?,reverse,intent,best)
    }
    fn matrix(p: Profile, reverse: bool, intent: u32, best: bool) -> Option<Self> {
        let scale = std::array::from_fn(|i| if intent == 3 {if reverse {[0.9642,1.0,0.8249][i]/p.white[i]}else{p.white[i]/[0.9642,1.0,0.8249][i]}}else{1.0});
        let m = if reverse {inverse(p.matrix)?} else {p.matrix};
        let c = Stage::Curves(p.curves.into_iter().map(|c|if best{c.gpu(reverse)}else{c.sampled(reverse)}).collect());
        let stages = if reverse {vec![Stage::Scale(scale),Stage::Matrix(m),c]} else {vec![c,Stage::Matrix(m),Stage::Scale(scale)]};
        Some(Self {channels:3,stages,legacy:intent==0})
    }
}

fn embedded_curves(data: &[u8], mut at: usize, count: usize) -> Option<Stage> {
    let mut curves=Vec::new();
    for _ in 0..count {
        let bytes=data.get(at..)?;
        curves.push(if bytes.get(..4)?==b"curv" && u32be(bytes,8)?==0 {Curve::Table(vec![0.0,1.0])}else{curve(bytes)?.sampled(false)});
        let size=match bytes.get(..4)? {
            b"curv"=>12usize.checked_add((u32be(bytes,8)? as usize).checked_mul(2)?)?,
            b"para"=>12+4*[1usize,3,4,5,7].get(*bytes.get(9)? as usize)?,
            _=>return None,
        };
        at=at.checked_add((size+3)&!3)?;
    }
    Some(Stage::Curves(curves))
}
fn parse_mab(data: &[u8]) -> Option<Vec<Stage>> {
    let input=*data.get(8)? as usize;let output=*data.get(9)? as usize;
    if !(1..=4).contains(&input)||!(1..=4).contains(&output){return None;}
    let reverse=data.get(..4)?==b"mBA ";
    let (a,b,m,clut,matrix)=(u32be(data,28)? as usize,u32be(data,12)? as usize,u32be(data,20)? as usize,u32be(data,24)? as usize,u32be(data,16)? as usize);
    let mut stages=Vec::new();
    if a!=0 {stages.push(embedded_curves(data,a,if reverse{output}else{input})?);}
    if clut!=0 {
        let p=data.get(clut..)?;let grid:Vec<_>=p.get(..input)?.iter().map(|v|*v as usize).collect();
        if grid.iter().any(|v|*v<2){return None;}
        let count=grid.iter().try_fold(output,|n,g|n.checked_mul(*g))?;
        let values=match *p.get(16)? {
            1=>p.get(20..20usize.checked_add(count)?)?.iter().map(|v|*v as f32/255.0).collect(),
            2=>p.get(20..20usize.checked_add(count.checked_mul(2)?)?)?.chunks_exact(2).map(|v|u16::from_be_bytes([v[0],v[1]]) as f32/65535.0).collect(),
            _=>return None,
        };
        stages.push(Stage::Lut(Clut{channels:input,output,grid,values,bits:if *p.get(16)?==1{8}else{16}}));
    }
    if m!=0 {stages.push(embedded_curves(data,m,if reverse{input}else{output})?);}
    if matrix!=0 {
        let mut mat=[[0.0;3];3];let mut off=[0.0;3];
        for i in 0..3 {for j in 0..3 {mat[i][j]=fixed(data,matrix+4*(i*3+j))?;}off[i]=fixed(data,matrix+36+i*4)?;}
        stages.push(Stage::Matrix(mat));stages.push(Stage::Offset(off));
    }
    if b!=0 {stages.push(embedded_curves(data,b,if reverse{input}else{output})?);}
    if reverse {
        stages.reverse();
        // Reversing the ICC stage order keeps the affine matrix's offset after its multiply.
        if let Some(i)=stages.windows(2).position(|s|matches!((&s[0],&s[1]),(Stage::Offset(_),Stage::Matrix(_)))) {stages.swap(i,i+1);}
    }
    Some(stages)
}
fn parse_lut(data: &[u8], reverse: bool) -> Option<Vec<Stage>> {
    if matches!(data.get(..4)?,b"mAB "|b"mBA "){return parse_mab(data);}
    let is16 = match data.get(..4)? {b"mft1"=>false,b"mft2"=>true,_=>return None};
    let input = *data.get(8)? as usize;
    let output = *data.get(9)? as usize;
    let grid = *data.get(10)? as usize;
    if !(1..=4).contains(&input) || !(1..=4).contains(&output) || grid<2 {return None;}
    let word = |at| Some(u16::from_be_bytes(data.get(at..at+2)?.try_into().ok()?) as usize);
    let ni = if is16 {word(48)?} else {256};
    let no = if is16 {word(50)?} else {256};
    if ni<2 || no<2 || ni>4096 || no>4096 {return None;}
    let mut at: usize = if is16 {52} else {48};
    let mut read = |n: usize| -> Option<Vec<f32>> {
        let bytes = data.get(at..at.checked_add(n.checked_mul(if is16 {2}else{1})?)?)?;
        at += bytes.len();
        Some(if is16 {bytes.chunks_exact(2).map(|v|u16::from_be_bytes([v[0],v[1]]) as f32/65535.0).collect()} else {bytes.iter().map(|v|*v as f32/255.0).collect()})
    };
    let mut curves = Vec::new();
    for _ in 0..input {curves.push(Curve::Table(read(ni)?));}
    let values = read(grid.checked_pow(input as u32)?.checked_mul(output)?)?;
    let mut out = Vec::new();
    for _ in 0..output {out.push(Curve::Table(read(no)?));}
    let mut stages=vec![if is16{Stage::Curves(curves)}else{Stage::Curves8(curves)},Stage::Lut(Clut {channels:input,output,grid:vec![grid;input],values,bits:if is16{16}else{8}}),if is16{Stage::Curves(out)}else{Stage::Curves8(out)}];
    if reverse {
        if input!=3{return None;}
        let mut m=[[0.0;3];3];for i in 0..3{for j in 0..3{m[i][j]=fixed(data,12+4*(i*3+j))?;}}
        stages.insert(0,Stage::Matrix(m));
    }
    Some(stages)
}

pub(in crate::color_context) struct Transform {
    source: Pipeline,
    dest: Pipeline,
    edge: usize,
    cube: Vec<[u8;4]>,
    best: bool,
    cube4: Option<Clut>,
}
impl Transform {
    pub(super) fn new(src: &ColorContext,dst: &ColorContext,intents:[u32;2],quality:u32) -> Option<Self> {
        let mut source = Pipeline::new(src,false,intents[0],quality==2)?;
        let mut dest = Pipeline::new(dst,true,intents[1],quality==2)?;
        if source.legacy==dest.legacy && !intents.contains(&3) {
            let from=source.stages.iter().find_map(|s|if let Stage::LabToXyz(mft2,_) = s {Some(*mft2)}else{None});
            let to=dest.stages.iter().find_map(|s|if let Stage::XyzToLab(mft2) = s {Some(*mft2)}else{None});
            if let (Some(from),Some(to))=(from,to) {
                // TranslatePCS skips a Lab→XYZ→Lab round trip when only the encoding changes.
                source.stages.retain(|s|!matches!(s,Stage::LabToXyz(..)|Stage::Scale(_)));
                dest.stages.retain(|s|!matches!(s,Stage::XyzToLab(_)|Stage::Scale(_)));
                if from!=to {source.stages.push(Stage::Scale([if from{1.00390625}else{256.0/257.0};3]));}
            }
        }
        if source.legacy != dest.legacy {
            // The XYZ PCS black-point compensation, native 10.0.26100.8875 constants.
            let black=[0x3b5c3372,0x3b639cf3,0x3b3c169c].map(f32::from_bits);
            let scale=if source.legacy {[0x3f7f1ba0,0x3f7f1c63,0x3f7f1bfd]}else{[0x3f807296,0x3f807234,0x3f807267]}.map(f32::from_bits);
            let offset=std::array::from_fn(|i|if source.legacy{black[i]}else{-black[i]*scale[i]});
            if let Some(Stage::Scale(k))=source.stages.last_mut() {for i in 0..3{k[i]*=scale[i];}}
            else {source.stages.push(Stage::Scale(scale));}
            source.stages.push(Stage::Offset(offset));
            source.stages.push(Stage::Clamp);
        }
        if intents.contains(&3) {source.stages.push(Stage::Clamp);}
        if quality==2 && source.stages.iter().chain(&dest.stages).any(|s| {
            matches!(s,Stage::Lut(lut) if !matches!(lut.channels,3|4) || !matches!(lut.output,1|3|4))
        }) {return None;}
        let edge = match source.channels {1=>if quality==0{256}else{1024},4=>16,_=>if quality==0{16}else{32}};
        let mut result = Self {source,dest,edge,cube:Vec::new(),best:quality==2,cube4:None};
        if result.best {result.source=result.source.gpu();result.dest=result.dest.gpu();return Some(result);}
        let count = edge.pow(result.source.channels as u32);
        let step = 1.0/(edge-1) as f32;
        for i in 0..count {
            let mut n=i;let mut v=[0.0;4];
            for c in (0..result.source.channels).rev() {v[c]=(n%edge) as f32*step;n/=edge;}
            let v=result.dest.apply(result.source.apply(v));
            result.cube.push(v.map(|v|(v*255.0+0.5).clamp(0.0,255.0) as u8));
        }
        if result.source.channels==4 {
            result.cube4=Some(Clut{channels:4,output:4,grid:vec![edge;4],bits:8,values:result.cube.iter().flatten().map(|v|*v as f32/255.0).collect()});
        }
        Some(result)
    }
    pub(super) fn apply(&self,p:[f32;4],straight:bool)->[f32;4] {
        let alpha = if self.source.channels==4 {1.0}else{p[3]};
        let input = if self.source.channels==4 || straight {p}else if alpha!=0.0 {p.map(|v|v/alpha)}else{[0.0;4]};
        if self.best || self.cube4.is_some() {
            let mut out=if let Some(cube)=&self.cube4 {cube.gpu(input)}else{self.dest.apply(self.source.apply(input))};
            if self.dest.channels==1{out[1]=out[0];out[2]=out[0];}
            if self.dest.channels!=4 {
                let k=if straight{1.0}else{alpha};out[0]*=k;out[1]*=k;out[2]*=k;out[3]=alpha;
            }
            return out;
        }
        let mut axes=[(0usize,0i32);4];let mut stride=[0usize;4];let mut step=1;
        for c in (0..self.source.channels).rev() {
            stride[c]=step;step*=self.edge;
            let q=(input[c].clamp(0.0,1.0)*(self.edge-1) as f32*256.0).floor() as i32;
            let i=(q>>8).min(self.edge as i32-2);axes[c]=(i as usize,q-i*256);
        }
        fn sample(t:&Transform,axes:&[(usize,i32);4],stride:&[usize;4],dim:usize,index:usize,ch:usize)->i32 {
            if dim==0 {
                let a=t.cube[index+axes[0].0*stride[0]][ch] as i32;
                let b=t.cube[index+(axes[0].0+1)*stride[0]][ch] as i32;
                return ((b-a)*axes[0].1+(a<<8))>>1;
            }
            let a=sample(t,axes,stride,dim-1,index+axes[dim].0*stride[dim],ch);
            let b=sample(t,axes,stride,dim-1,index+(axes[dim].0+1)*stride[dim],ch);
            a+((b-a)*axes[dim].1>>8)
        }
        let mut out=[0.0;4];
        for c in 0..self.dest.channels {
            let v=sample(self,&axes,&stride,self.source.channels-1,0,c);
            out[c]=if (straight && self.source.channels!=4) || self.dest.channels==4 {(v>>7) as f32/255.0}else{crate::sampling::norm_v128(v)};
        }
        if self.dest.channels==1 {out[1]=out[0];out[2]=out[0];}
        if self.dest.channels!=4 {
            let k=if straight{1.0}else{alpha};
            out[0]*=k;out[1]*=k;out[2]*=k;out[3]=alpha;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncated_and_oversized_lut_tags_are_rejected() {
        let mut tag = vec![0u8; 52];
        tag[..4].copy_from_slice(b"mft2");
        tag[8..11].copy_from_slice(&[3, 3, 2]);
        tag[48..52].copy_from_slice(&[0, 2, 0, 2]);
        tag.extend_from_slice(&[0, 0, 255, 255].repeat(3));
        tag.extend_from_slice(&[0, 0].repeat(8 * 3));
        tag.extend_from_slice(&[0, 0, 255, 255].repeat(3));
        assert!(parse_lut(&tag, false).is_some());
        for len in 0..tag.len() {
            assert!(parse_lut(&tag[..len], false).is_none(), "accepted prefix {len}");
        }
        for (at, value) in [(8, 0), (8, 5), (9, 0), (9, 5), (10, 0), (10, 1), (10, 255), (48, 255)] {
            let mut malformed = tag.clone();
            malformed[at] = value;
            assert!(parse_lut(&malformed, false).is_none(), "accepted field {at}={value}");
        }
    }

    #[test]
    fn mba_matrix_offset_follows_the_matrix() {
        let mut tag = vec![0u8; 80];
        tag[..4].copy_from_slice(b"mBA ");
        tag[8..10].copy_from_slice(&[3, 3]);
        tag[16..20].copy_from_slice(&32u32.to_be_bytes());
        for (i, value) in [2i32, 0, 0, 0, 3, 0, 0, 0, 4, 1, 2, 3].into_iter().enumerate() {
            tag[32+i*4..36+i*4].copy_from_slice(&(value*65536).to_be_bytes());
        }
        let p = Pipeline { channels: 3, legacy: false, stages: parse_mab(&tag).unwrap() };
        assert_eq!(p.apply([1.0, 2.0, 3.0, 1.0]), [3.0, 8.0, 15.0, 1.0]);
    }
}
