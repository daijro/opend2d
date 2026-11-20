//! Component information and enumeration: `IWICBitmapDecoderInfo` / `IWICBitmapEncoderInfo` (codec
//! identity — friendly name, author, versions, container, MIME/extensions, capability flags),
//! `IWICPixelFormatInfo2` (per-format channel/numeric-representation data), and the
//! `IWICComponentEnumerator`. The values are the exact ones native WIC returns,
//! read off the VM with `wic-dump compinfo/pfinfo/enumcomp` (verified byte-for-byte for every codec
//! and every pixel format our decoders emit).

use super::{e, guid_to_str};
use std::cell::Cell;
use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Imaging::*;
use windows::Win32::System::Com::*;
use windows_core::{BOOL, GUID, HRESULT, IUnknown, Interface, PCWSTR, PWSTR, Ref, Result as WResult, implement};

/// Every WIC component shares these (read off the oracle).
const AUTHOR: &str = "Microsoft";
const VERSION: &str = "1.0.0.0";
fn vendor() -> GUID {
    GUID::from_u128(0xf0e749ca_edef_4589_a73a_ee0e626a2a2b)
}
/// `WINCODEC_ERR_INSUFFICIENTBUFFER` (not in the `windows` metadata as a constant).
const INSUFFICIENT_BUFFER: u32 = 0x8898_2F8C;

/// Write a WIC string getter result: `*actual` = length incl. NUL; copy when the buffer fits, a
/// size query (`cch == 0`) just reports the length, a short buffer is `WINCODEC_ERR_INSUFFICIENTBUFFER`.
pub(crate) fn put_wstr(s: &str, cch: u32, wz: PWSTR, actual: *mut u32) -> WResult<()> {
    let u: Vec<u16> = s.encode_utf16().collect();
    let need = u.len() as u32 + 1;
    if !actual.is_null() {
        unsafe { *actual = need };
    }
    if cch == 0 {
        return Ok(());
    }
    if cch < need || wz.0.is_null() {
        return Err(e(HRESULT(INSUFFICIENT_BUFFER as i32)));
    }
    unsafe {
        for (i, &c) in u.iter().enumerate() {
            *wz.0.add(i) = c;
        }
        *wz.0.add(u.len()) = 0;
    }
    Ok(())
}

// --- decoder / encoder codec info --------------------------------------------------------------

struct Codec {
    container: u128,
    clsid: u128,
    friendly: &'static str,
    mimetypes: &'static str,
    fileexts: &'static str,
    animation: bool,
    chromakey: bool,
    lossless: bool,
    multiframe: bool,
    formats: &'static [u128],
    patterns: &'static [&'static [u8]],
}

// Windows 11 codec registry metadata and pattern tables.
const DECODERS: &[Codec] = &[
    Codec { container: 0x0af1d87e_fcfe_4188_bdeb_a7906471cbe3, clsid: 0x6b462062_7cbf_400d_9fdb_813dd10f2778, friendly: "BMP Decoder", mimetypes: "image/bmp", fileexts: ".bmp,.dib,.rle", animation: false, chromakey: false, lossless: true, multiframe: false, formats: &[0x6fddc324_4e03_4bfe_b185_3d77768dc901, 0x6fddc324_4e03_4bfe_b185_3d77768dc903, 0x6fddc324_4e03_4bfe_b185_3d77768dc904, 0x6fddc324_4e03_4bfe_b185_3d77768dc909, 0x6fddc324_4e03_4bfe_b185_3d77768dc90a, 0x6fddc324_4e03_4bfe_b185_3d77768dc90c, 0x6fddc324_4e03_4bfe_b185_3d77768dc90e, 0x6fddc324_4e03_4bfe_b185_3d77768dc90f, 0x6fddc324_4e03_4bfe_b185_3d77768dc91d], patterns: &[&[66, 77]] },
    Codec { container: 0x9967cb95_2e85_4ac8_8ca2_83d7ccd425c9, clsid: 0x9053699f_a341_429d_9e90_ee437cf80c73, friendly: "DDS Decoder", mimetypes: "image/vnd.ms-dds", fileexts: ".dds", animation: false, chromakey: false, lossless: false, multiframe: true, formats: &[0x6fddc324_4e03_4bfe_b185_3d77768dc90f], patterns: &[&[68, 68, 83, 32]] },
    Codec { container: 0x1f8a5601_7d4d_4cbd_9c82_1bc8d4eeb9a5, clsid: 0x381dda3c_9ce9_4834_a23e_1f98f8fc52be, friendly: "GIF Decoder", mimetypes: "image/gif", fileexts: ".gif", animation: true, chromakey: true, lossless: true, multiframe: true, formats: &[0x6fddc324_4e03_4bfe_b185_3d77768dc904], patterns: &[&[71, 73, 70, 56, 55, 97], &[71, 73, 70, 56, 57, 97]] },
    Codec { container: 0xa3a860c4_338f_4c17_919a_fba4b5628f21, clsid: 0xc61bfcdf_2e0f_4aad_a8d7_e06bafebcdfe, friendly: "ICO Decoder", mimetypes: "image/ico,image/x-icon", fileexts: ".ico,.icon", animation: false, chromakey: true, lossless: true, multiframe: true, formats: &[0x6fddc324_4e03_4bfe_b185_3d77768dc90f], patterns: &[&[0, 0, 1, 0]] },
    Codec { container: 0x19e4a5aa_5662_4fc5_a0c0_1758028e1057, clsid: 0x9456a480_e88b_43ea_9e73_0b2d9b71b1ca, friendly: "JPEG Decoder", mimetypes: "image/jpeg,image/jpe,image/jpg", fileexts: ".jpeg,.jpe,.jpg,.jfif,.exif", animation: false, chromakey: false, lossless: false, multiframe: false, formats: &[0x6fddc324_4e03_4bfe_b185_3d77768dc908, 0x6fddc324_4e03_4bfe_b185_3d77768dc90c, 0x6fddc324_4e03_4bfe_b185_3d77768dc91c], patterns: &[&[255, 216]] },
    Codec { container: 0x1b7cfaf4_713f_473c_bbcd_6137425faeaf, clsid: 0x389ea17b_5078_4cde_b6ef_25c15175c751, friendly: "PNG Decoder", mimetypes: "image/png", fileexts: ".png", animation: false, chromakey: true, lossless: true, multiframe: false, formats: &[0x6fddc324_4e03_4bfe_b185_3d77768dc901, 0x6fddc324_4e03_4bfe_b185_3d77768dc902, 0x6fddc324_4e03_4bfe_b185_3d77768dc903, 0x6fddc324_4e03_4bfe_b185_3d77768dc904, 0x6fddc324_4e03_4bfe_b185_3d77768dc905, 0x6fddc324_4e03_4bfe_b185_3d77768dc906, 0x6fddc324_4e03_4bfe_b185_3d77768dc907, 0x6fddc324_4e03_4bfe_b185_3d77768dc908, 0x6fddc324_4e03_4bfe_b185_3d77768dc90b, 0x6fddc324_4e03_4bfe_b185_3d77768dc90c, 0x6fddc324_4e03_4bfe_b185_3d77768dc90f, 0x6fddc324_4e03_4bfe_b185_3d77768dc915, 0x6fddc324_4e03_4bfe_b185_3d77768dc916], patterns: &[&[137, 80, 78, 71, 13, 10, 26, 10]] },
    Codec { container: 0x163bcc30_e2e9_4f0b_961d_a3e9fdb788a3, clsid: 0xb54e85d9_fe23_499f_8b88_6acea713752b, friendly: "TIFF Decoder", mimetypes: "image/tiff,image/tif", fileexts: ".tiff,.tif", animation: false, chromakey: true, lossless: true, multiframe: true, formats: &[0x6fddc324_4e03_4bfe_b185_3d77768dc901, 0x6fddc324_4e03_4bfe_b185_3d77768dc903, 0x6fddc324_4e03_4bfe_b185_3d77768dc904, 0x6fddc324_4e03_4bfe_b185_3d77768dc905, 0x6fddc324_4e03_4bfe_b185_3d77768dc907, 0x6fddc324_4e03_4bfe_b185_3d77768dc908, 0x6fddc324_4e03_4bfe_b185_3d77768dc90b, 0x6fddc324_4e03_4bfe_b185_3d77768dc911, 0x6fddc324_4e03_4bfe_b185_3d77768dc90c, 0x6fddc324_4e03_4bfe_b185_3d77768dc90f, 0x6fddc324_4e03_4bfe_b185_3d77768dc910, 0x6fddc324_4e03_4bfe_b185_3d77768dc915, 0x6fddc324_4e03_4bfe_b185_3d77768dc91c, 0x6fddc324_4e03_4bfe_b185_3d77768dc92c, 0x6fddc324_4e03_4bfe_b185_3d77768dc916, 0x6fddc324_4e03_4bfe_b185_3d77768dc917, 0x6fddc324_4e03_4bfe_b185_3d77768dc91f, 0x6fddc324_4e03_4bfe_b185_3d77768dc92d, 0xe3fed78f_e8db_4acf_84c1_e97f6136b327, 0x6fddc324_4e03_4bfe_b185_3d77768dc919, 0x6fddc324_4e03_4bfe_b185_3d77768dc91a], patterns: &[&[73, 73, 42, 0], &[77, 77, 0, 42]] },
    #[cfg(feature = "jxr")]
    Codec { container: 0x57a37caa_367a_4540_916b_f183c5093a4b, clsid: 0xa26cec36_234c_4950_ae16_e34aace71d0d, friendly: "WMPhoto Decoder", mimetypes: "image/vnd.ms-photo", fileexts: ".wdp,.jxr", animation: false, chromakey: false, lossless: true, multiframe: true, formats: &[0x6fddc324_4e03_4bfe_b185_3d77768dc905, 0x6fddc324_4e03_4bfe_b185_3d77768dc908, 0x6fddc324_4e03_4bfe_b185_3d77768dc909, 0x6fddc324_4e03_4bfe_b185_3d77768dc90b, 0x6fddc324_4e03_4bfe_b185_3d77768dc90c, 0x6fddc324_4e03_4bfe_b185_3d77768dc90d, 0x6fddc324_4e03_4bfe_b185_3d77768dc90e, 0x6fddc324_4e03_4bfe_b185_3d77768dc90f, 0x6fddc324_4e03_4bfe_b185_3d77768dc912, 0x6fddc324_4e03_4bfe_b185_3d77768dc913, 0x6fddc324_4e03_4bfe_b185_3d77768dc914, 0x6fddc324_4e03_4bfe_b185_3d77768dc915, 0x6fddc324_4e03_4bfe_b185_3d77768dc916, 0x6fddc324_4e03_4bfe_b185_3d77768dc918, 0x6fddc324_4e03_4bfe_b185_3d77768dc919, 0x6fddc324_4e03_4bfe_b185_3d77768dc91b, 0x6fddc324_4e03_4bfe_b185_3d77768dc91c, 0x6fddc324_4e03_4bfe_b185_3d77768dc91d, 0x6fddc324_4e03_4bfe_b185_3d77768dc91e, 0x6fddc324_4e03_4bfe_b185_3d77768dc91f, 0x6fddc324_4e03_4bfe_b185_3d77768dc920, 0x6fddc324_4e03_4bfe_b185_3d77768dc921, 0x6fddc324_4e03_4bfe_b185_3d77768dc922, 0x6fddc324_4e03_4bfe_b185_3d77768dc923, 0x6fddc324_4e03_4bfe_b185_3d77768dc924, 0x6fddc324_4e03_4bfe_b185_3d77768dc925, 0x6fddc324_4e03_4bfe_b185_3d77768dc926, 0x6fddc324_4e03_4bfe_b185_3d77768dc927, 0x6fddc324_4e03_4bfe_b185_3d77768dc928, 0x6fddc324_4e03_4bfe_b185_3d77768dc929, 0x6fddc324_4e03_4bfe_b185_3d77768dc92a, 0x6fddc324_4e03_4bfe_b185_3d77768dc92b, 0x6fddc324_4e03_4bfe_b185_3d77768dc92c, 0x6fddc324_4e03_4bfe_b185_3d77768dc92d, 0x6fddc324_4e03_4bfe_b185_3d77768dc92e, 0x6fddc324_4e03_4bfe_b185_3d77768dc92f, 0x6fddc324_4e03_4bfe_b185_3d77768dc930, 0x6fddc324_4e03_4bfe_b185_3d77768dc931, 0x6fddc324_4e03_4bfe_b185_3d77768dc932, 0x6fddc324_4e03_4bfe_b185_3d77768dc933, 0x6fddc324_4e03_4bfe_b185_3d77768dc934, 0x6fddc324_4e03_4bfe_b185_3d77768dc935, 0x6fddc324_4e03_4bfe_b185_3d77768dc936, 0x6fddc324_4e03_4bfe_b185_3d77768dc937, 0x6fddc324_4e03_4bfe_b185_3d77768dc938, 0x6fddc324_4e03_4bfe_b185_3d77768dc939, 0x6fddc324_4e03_4bfe_b185_3d77768dc93a, 0x6fddc324_4e03_4bfe_b185_3d77768dc93b, 0x6fddc324_4e03_4bfe_b185_3d77768dc93d, 0x6fddc324_4e03_4bfe_b185_3d77768dc93e, 0x6fddc324_4e03_4bfe_b185_3d77768dc93f, 0x6fddc324_4e03_4bfe_b185_3d77768dc940, 0x6fddc324_4e03_4bfe_b185_3d77768dc941, 0x6fddc324_4e03_4bfe_b185_3d77768dc942], patterns: &[&[73, 73, 188, 0], &[73, 73, 188, 1]] },
];

const ENCODERS: &[Codec] = &[
    Codec { container: 0x0af1d87e_fcfe_4188_bdeb_a7906471cbe3, clsid: 0x69be8bb4_d66d_47c8_865a_ed1589433782, friendly: "BMP Encoder", mimetypes: "image/bmp", fileexts: ".bmp,.dib,.rle", animation: false, chromakey: false, lossless: true, multiframe: false, formats: &[0x6fddc324_4e03_4bfe_b185_3d77768dc901, 0x6fddc324_4e03_4bfe_b185_3d77768dc903, 0x6fddc324_4e03_4bfe_b185_3d77768dc904, 0x6fddc324_4e03_4bfe_b185_3d77768dc909, 0x6fddc324_4e03_4bfe_b185_3d77768dc90a, 0x6fddc324_4e03_4bfe_b185_3d77768dc90c, 0x6fddc324_4e03_4bfe_b185_3d77768dc90e, 0x6fddc324_4e03_4bfe_b185_3d77768dc90f, 0x6fddc324_4e03_4bfe_b185_3d77768dc910, 0x6fddc324_4e03_4bfe_b185_3d77768dc91d, 0x356de33c_54d2_4a23_bb04_9b7bf9b1d42d], patterns: &[] },
    Codec { container: 0x1f8a5601_7d4d_4cbd_9c82_1bc8d4eeb9a5, clsid: 0x114f5598_0b22_40a0_86a1_c83ea495adbd, friendly: "GIF Encoder", mimetypes: "image/gif", fileexts: ".gif", animation: true, chromakey: true, lossless: true, multiframe: true, formats: &[0x6fddc324_4e03_4bfe_b185_3d77768dc904], patterns: &[] },
    Codec { container: 0x19e4a5aa_5662_4fc5_a0c0_1758028e1057, clsid: 0x1a34f5c1_4a5a_46dc_b644_1f4567e7a676, friendly: "JPEG Encoder", mimetypes: "image/jpeg,image/jpe,image/jpg", fileexts: ".jpeg,.jpe,.jpg,.jfif,.exif", animation: false, chromakey: false, lossless: false, multiframe: false, formats: &[0x6fddc324_4e03_4bfe_b185_3d77768dc908, 0x6fddc324_4e03_4bfe_b185_3d77768dc90c, 0x6fddc324_4e03_4bfe_b185_3d77768dc91c], patterns: &[] },
    Codec { container: 0x1b7cfaf4_713f_473c_bbcd_6137425faeaf, clsid: 0x27949969_876a_41d7_9447_568f6a35a4dc, friendly: "PNG Encoder", mimetypes: "image/png", fileexts: ".png", animation: false, chromakey: true, lossless: true, multiframe: false, formats: &[0x6fddc324_4e03_4bfe_b185_3d77768dc901, 0x6fddc324_4e03_4bfe_b185_3d77768dc902, 0x6fddc324_4e03_4bfe_b185_3d77768dc903, 0x6fddc324_4e03_4bfe_b185_3d77768dc904, 0x6fddc324_4e03_4bfe_b185_3d77768dc905, 0x6fddc324_4e03_4bfe_b185_3d77768dc906, 0x6fddc324_4e03_4bfe_b185_3d77768dc907, 0x6fddc324_4e03_4bfe_b185_3d77768dc908, 0x6fddc324_4e03_4bfe_b185_3d77768dc90b, 0x6fddc324_4e03_4bfe_b185_3d77768dc90c, 0x6fddc324_4e03_4bfe_b185_3d77768dc90f, 0x6fddc324_4e03_4bfe_b185_3d77768dc915, 0xe605a384_b468_46ce_bb2e_36f180e64313, 0x6fddc324_4e03_4bfe_b185_3d77768dc916, 0x1562ff7c_d352_46f9_979e_42976b792246], patterns: &[] },
    Codec { container: 0x163bcc30_e2e9_4f0b_961d_a3e9fdb788a3, clsid: 0x0131be10_2001_4c5f_a9b0_cc88fab64ce8, friendly: "TIFF Encoder", mimetypes: "image/tiff,image/tif", fileexts: ".tiff,.tif", animation: false, chromakey: true, lossless: true, multiframe: true, formats: &[0x6fddc324_4e03_4bfe_b185_3d77768dc901, 0x6fddc324_4e03_4bfe_b185_3d77768dc903, 0x6fddc324_4e03_4bfe_b185_3d77768dc904, 0x6fddc324_4e03_4bfe_b185_3d77768dc905, 0x6fddc324_4e03_4bfe_b185_3d77768dc907, 0x6fddc324_4e03_4bfe_b185_3d77768dc908, 0x6fddc324_4e03_4bfe_b185_3d77768dc90b, 0x6fddc324_4e03_4bfe_b185_3d77768dc90c, 0x6fddc324_4e03_4bfe_b185_3d77768dc90f, 0x6fddc324_4e03_4bfe_b185_3d77768dc91c, 0x6fddc324_4e03_4bfe_b185_3d77768dc915, 0x6fddc324_4e03_4bfe_b185_3d77768dc916], patterns: &[] },
    #[cfg(feature = "jxr")]
    Codec { container: 0x57a37caa_367a_4540_916b_f183c5093a4b, clsid: 0xac4ce3cb_e1c1_44cd_8215_5a1665509ec2, friendly: "WMPhoto Encoder", mimetypes: "image/vnd.ms-photo", fileexts: ".wdp,.jxr", animation: false, chromakey: false, lossless: true, multiframe: false, formats: &[0x6fddc324_4e03_4bfe_b185_3d77768dc905, 0x6fddc324_4e03_4bfe_b185_3d77768dc908, 0x6fddc324_4e03_4bfe_b185_3d77768dc909, 0x6fddc324_4e03_4bfe_b185_3d77768dc90b, 0x6fddc324_4e03_4bfe_b185_3d77768dc90c, 0x6fddc324_4e03_4bfe_b185_3d77768dc90d, 0x6fddc324_4e03_4bfe_b185_3d77768dc90e, 0x6fddc324_4e03_4bfe_b185_3d77768dc90f, 0x6fddc324_4e03_4bfe_b185_3d77768dc912, 0x6fddc324_4e03_4bfe_b185_3d77768dc913, 0x6fddc324_4e03_4bfe_b185_3d77768dc914, 0x6fddc324_4e03_4bfe_b185_3d77768dc915, 0x6fddc324_4e03_4bfe_b185_3d77768dc916, 0x6fddc324_4e03_4bfe_b185_3d77768dc918, 0x6fddc324_4e03_4bfe_b185_3d77768dc919, 0x6fddc324_4e03_4bfe_b185_3d77768dc91b, 0x6fddc324_4e03_4bfe_b185_3d77768dc91c, 0x6fddc324_4e03_4bfe_b185_3d77768dc91d, 0x6fddc324_4e03_4bfe_b185_3d77768dc91e, 0x6fddc324_4e03_4bfe_b185_3d77768dc91f, 0x6fddc324_4e03_4bfe_b185_3d77768dc920, 0x6fddc324_4e03_4bfe_b185_3d77768dc921, 0x6fddc324_4e03_4bfe_b185_3d77768dc922, 0x6fddc324_4e03_4bfe_b185_3d77768dc923, 0x6fddc324_4e03_4bfe_b185_3d77768dc924, 0x6fddc324_4e03_4bfe_b185_3d77768dc925, 0x6fddc324_4e03_4bfe_b185_3d77768dc926, 0x6fddc324_4e03_4bfe_b185_3d77768dc927, 0x6fddc324_4e03_4bfe_b185_3d77768dc928, 0x6fddc324_4e03_4bfe_b185_3d77768dc929, 0x6fddc324_4e03_4bfe_b185_3d77768dc92a, 0x6fddc324_4e03_4bfe_b185_3d77768dc92b, 0x6fddc324_4e03_4bfe_b185_3d77768dc92c, 0x6fddc324_4e03_4bfe_b185_3d77768dc92d, 0x6fddc324_4e03_4bfe_b185_3d77768dc92e, 0x6fddc324_4e03_4bfe_b185_3d77768dc92f, 0x6fddc324_4e03_4bfe_b185_3d77768dc930, 0x6fddc324_4e03_4bfe_b185_3d77768dc931, 0x6fddc324_4e03_4bfe_b185_3d77768dc932, 0x6fddc324_4e03_4bfe_b185_3d77768dc933, 0x6fddc324_4e03_4bfe_b185_3d77768dc934, 0x6fddc324_4e03_4bfe_b185_3d77768dc935, 0x6fddc324_4e03_4bfe_b185_3d77768dc936, 0x6fddc324_4e03_4bfe_b185_3d77768dc937, 0x6fddc324_4e03_4bfe_b185_3d77768dc938, 0x6fddc324_4e03_4bfe_b185_3d77768dc939, 0x6fddc324_4e03_4bfe_b185_3d77768dc93a, 0x6fddc324_4e03_4bfe_b185_3d77768dc93b, 0x6fddc324_4e03_4bfe_b185_3d77768dc93d, 0x6fddc324_4e03_4bfe_b185_3d77768dc93e, 0x6fddc324_4e03_4bfe_b185_3d77768dc93f, 0x6fddc324_4e03_4bfe_b185_3d77768dc940, 0x6fddc324_4e03_4bfe_b185_3d77768dc941, 0x6fddc324_4e03_4bfe_b185_3d77768dc942], patterns: &[] },
];

/// The decoder info for a container GUID, if we decode that container.
pub(crate) fn decoder_info(container: &GUID) -> WResult<IWICBitmapDecoderInfo> {
    let key = guid_to_str(container);
    DECODERS
        .iter()
        .find(|c| guid_to_str(&GUID::from_u128(c.container)) == key)
        .map(|c| DecoderInfoObj { c }.into())
        .ok_or(e(WINCODEC_ERR_COMPONENTNOTFOUND))
}

/// The encoder info for a container GUID, if we encode that container.
pub(crate) fn encoder_info(container: &GUID) -> WResult<IWICBitmapEncoderInfo> {
    let key = guid_to_str(container);
    ENCODERS
        .iter()
        .find(|c| guid_to_str(&GUID::from_u128(c.container)) == key)
        .map(|c| EncoderInfoObj { c }.into())
        .ok_or(e(WINCODEC_ERR_COMPONENTNOTFOUND))
}

/// Factory codec lookup; the vendor is a preference, so built-in codecs remain the fallback.
pub(crate) fn create_decoder(container: *const GUID, _vendor: *const GUID) -> WResult<IWICBitmapDecoder> {
    let container = unsafe { container.as_ref() }.ok_or(e(E_INVALIDARG))?;
    if !DECODERS.iter().any(|c| c.container == container.to_u128()) { return Err(e(WINCODEC_ERR_COMPONENTNOTFOUND)); }
    Ok(super::decode::DecoderObj::for_container(*container).into())
}

pub(crate) fn create_component_info(clsid: *const GUID) -> WResult<IWICComponentInfo> {
    let clsid = unsafe { clsid.as_ref() }.ok_or(e(E_INVALIDARG))?;
    if let Some(c) = DECODERS.iter().find(|c| c.clsid == clsid.to_u128()) {
        let info: IWICBitmapDecoderInfo = DecoderInfoObj { c }.into();
        return info.cast();
    }
    if let Some(c) = ENCODERS.iter().find(|c| c.clsid == clsid.to_u128()) {
        let info: IWICBitmapEncoderInfo = EncoderInfoObj { c }.into();
        return info.cast();
    }
    if DECODERS.iter().chain(ENCODERS).any(|c| c.formats.contains(&clsid.to_u128())) || pf_spec(clsid).is_some() {
        let info: IWICPixelFormatInfo2 = PfInfoObj { format: *clsid }.into();
        return info.cast();
    }
    Err(e(WINCODEC_ERR_COMPONENTNOTFOUND))
}

pub(crate) fn is_codec_class(clsid: &GUID) -> bool {
    DECODERS.iter().chain(ENCODERS).any(|c| c.clsid == clsid.to_u128())
}

pub(crate) fn create_codec_instance(clsid: &GUID) -> WResult<IUnknown> {
    if let Some(c) = DECODERS.iter().find(|c| c.clsid == clsid.to_u128()) {
        let decoder: IWICBitmapDecoder = super::decode::DecoderObj::for_container(GUID::from_u128(c.container)).into();
        return decoder.cast();
    }
    if let Some(c) = ENCODERS.iter().find(|c| c.clsid == clsid.to_u128()) {
        return super::encode::EncoderObj::for_container(GUID::from_u128(c.container))?.cast();
    }
    Err(e(CLASS_E_CLASSNOTAVAILABLE))
}

/// `CreateComponentEnumerator`: our implemented decoders (`types & WICDecoder`) and/or encoders
/// (`types & WICEncoder`). Other component kinds are not enumerated.
pub(crate) fn create_enumerator(types: u32) -> IEnumUnknown {
    let mut items: Vec<IUnknown> = Vec::new();
    if types & WICDecoder.0 as u32 != 0 {
        items.extend(DECODERS.iter().map(|c| DecoderInfoObj { c }.into()));
    }
    if types & WICEncoder.0 as u32 != 0 {
        items.extend(ENCODERS.iter().map(|c| EncoderInfoObj { c }.into()));
    }
    EnumUnknownObj { items, pos: Cell::new(0) }.into()
}

#[implement(IWICBitmapDecoderInfo)]
struct DecoderInfoObj { c: &'static Codec }
#[implement(IWICBitmapEncoderInfo)]
struct EncoderInfoObj { c: &'static Codec }

macro_rules! codec_info_impl {
    ($ty:ty, $decoder:expr) => {
impl IWICComponentInfo_Impl for $ty {
    fn GetComponentType(&self) -> WResult<WICComponentType> {
        Ok(if $decoder { WICDecoder } else { WICEncoder })
    }
    fn GetCLSID(&self) -> WResult<GUID> {
        Ok(GUID::from_u128(self.c.clsid))
    }
    fn GetSigningStatus(&self) -> WResult<u32> {
        Ok(1) // WICComponentSigned
    }
    fn GetAuthor(&self, cch: u32, wz: PWSTR, a: *mut u32) -> WResult<()> {
        put_wstr(AUTHOR, cch, wz, a)
    }
    fn GetVendorGUID(&self) -> WResult<GUID> {
        Ok(vendor())
    }
    fn GetVersion(&self, cch: u32, wz: PWSTR, a: *mut u32) -> WResult<()> {
        put_wstr(VERSION, cch, wz, a)
    }
    fn GetSpecVersion(&self, cch: u32, wz: PWSTR, a: *mut u32) -> WResult<()> {
        put_wstr(VERSION, cch, wz, a)
    }
    fn GetFriendlyName(&self, cch: u32, wz: PWSTR, a: *mut u32) -> WResult<()> {
        put_wstr(self.c.friendly, cch, wz, a)
    }
}

impl IWICBitmapCodecInfo_Impl for $ty {
    fn GetContainerFormat(&self) -> WResult<GUID> {
        Ok(GUID::from_u128(self.c.container))
    }
    fn GetPixelFormats(&self, count: u32, formats: *mut GUID, actual: *mut u32) -> WResult<()> {
        if actual.is_null() { return Err(e(E_INVALIDARG)); }
        unsafe { *actual = self.c.formats.len() as u32; }
        if formats.is_null() { return Ok(()); }
        if (count as usize) < self.c.formats.len() { return Err(e(HRESULT(INSUFFICIENT_BUFFER as i32))); }
        for (i, &format) in self.c.formats.iter().enumerate() { unsafe { *formats.add(i) = GUID::from_u128(format); } }
        Ok(())
    }
    fn GetColorManagementVersion(&self, cch: u32, wz: PWSTR, a: *mut u32) -> WResult<()> {
        put_wstr(VERSION, cch, wz, a)
    }
    fn GetDeviceManufacturer(&self, cch: u32, wz: PWSTR, a: *mut u32) -> WResult<()> {
        put_wstr("", cch, wz, a)
    }
    fn GetDeviceModels(&self, cch: u32, wz: PWSTR, a: *mut u32) -> WResult<()> {
        put_wstr("", cch, wz, a)
    }
    fn GetMimeTypes(&self, cch: u32, wz: PWSTR, a: *mut u32) -> WResult<()> {
        put_wstr(self.c.mimetypes, cch, wz, a)
    }
    fn GetFileExtensions(&self, cch: u32, wz: PWSTR, a: *mut u32) -> WResult<()> {
        put_wstr(self.c.fileexts, cch, wz, a)
    }
    fn DoesSupportAnimation(&self) -> WResult<BOOL> {
        Ok(self.c.animation.into())
    }
    fn DoesSupportChromakey(&self) -> WResult<BOOL> {
        Ok(self.c.chromakey.into())
    }
    fn DoesSupportLossless(&self) -> WResult<BOOL> {
        Ok(self.c.lossless.into())
    }
    fn DoesSupportMultiframe(&self) -> WResult<BOOL> {
        Ok(self.c.multiframe.into())
    }
    fn MatchesMimeType(&self, mime: &PCWSTR) -> WResult<BOOL> {
        let m = unsafe { mime.to_string() }.unwrap_or_default();
        Ok(self.c.mimetypes.split(',').any(|t| t == m).into())
    }
}

    };
}
codec_info_impl!(DecoderInfoObj_Impl, true);
codec_info_impl!(EncoderInfoObj_Impl, false);

impl IWICBitmapDecoderInfo_Impl for DecoderInfoObj_Impl {
    fn GetPatterns(&self, cb: u32, p: *mut WICBitmapPattern, pc: *mut u32, pcb: *mut u32) -> WResult<()> {
        // WIC's GetPatterns: inline pointer-relocated payload.
        if pcb.is_null() { return Err(e(E_INVALIDARG)); }
        let count = self.c.patterns.len();
        let headers = count * core::mem::size_of::<WICBitmapPattern>();
        let size = headers + self.c.patterns.iter().map(|p| p.len() * 2).sum::<usize>();
        unsafe { *pcb = size as u32; if !pc.is_null() { *pc = count as u32; } }
        if p.is_null() { return Ok(()); }
        if (cb as usize) < size { return Err(e(HRESULT(INSUFFICIENT_BUFFER as i32))); }
        let mut offset = headers;
        for (i, pattern) in self.c.patterns.iter().enumerate() {
            unsafe {
                let data = p.cast::<u8>().add(offset);
                let mask = data.add(pattern.len());
                core::ptr::copy_nonoverlapping(pattern.as_ptr(), data, pattern.len());
                core::ptr::write_bytes(mask, 0xff, pattern.len());
                p.add(i).write(WICBitmapPattern { Position: 0, Length: pattern.len() as u32, Pattern: data, Mask: mask, EndOfStream: false.into() });
            }
            offset += pattern.len() * 2;
        }
        Ok(())
    }
    fn MatchesPattern(&self, s: Ref<IStream>) -> WResult<BOOL> {
        // WIC's MatchesPattern tests from file origin, then restores position.
        let stream = s.as_ref().ok_or(e(E_INVALIDARG))?;
        let mut pos = 0;
        unsafe { stream.Seek(0, STREAM_SEEK_CUR, Some(&mut pos))?; }
        let mut matched = false;
        for pattern in self.c.patterns {
            let mut buf = vec![0; pattern.len()];
            let mut read = 0;
            if unsafe { stream.Seek(0, STREAM_SEEK_SET, None) }.is_ok() {
                let hr = unsafe { stream.Read(buf.as_mut_ptr().cast(), buf.len() as u32, Some(&mut read)) };
                matched = hr.is_ok() && read as usize == pattern.len() && buf == *pattern;
            }
            unsafe { stream.Seek(pos as i64, STREAM_SEEK_SET, None)?; }
            if matched { break; }
        }
        Ok(matched.into())
    }
    fn CreateInstance(&self) -> WResult<IWICBitmapDecoder> {
        Ok(super::decode::DecoderObj::for_container(GUID::from_u128(self.c.container)).into())
    }
}

impl IWICBitmapEncoderInfo_Impl for EncoderInfoObj_Impl {
    fn CreateInstance(&self) -> WResult<IWICBitmapEncoder> {
        super::encode::EncoderObj::for_container(GUID::from_u128(self.c.container))
    }
}

/// Enumerator over a fixed component-info list (`IWICComponentEnumerator`).
#[implement(IEnumUnknown)]
struct EnumUnknownObj {
    items: Vec<IUnknown>,
    pos: Cell<usize>,
}

impl IEnumUnknown_Impl for EnumUnknownObj_Impl {
    fn Next(&self, celt: u32, rgelt: *mut Option<IUnknown>, fetched: *mut u32) -> HRESULT {
        let mut n = 0usize;
        let out = unsafe { core::slice::from_raw_parts_mut(rgelt, celt as usize) };
        while n < celt as usize && self.pos.get() < self.items.len() {
            out[n] = Some(self.items[self.pos.get()].clone());
            self.pos.set(self.pos.get() + 1);
            n += 1;
        }
        if !fetched.is_null() {
            unsafe { *fetched = n as u32 };
        }
        if n == celt as usize { S_OK } else { S_FALSE }
    }
    fn Skip(&self, celt: u32) -> WResult<()> {
        self.pos.set((self.pos.get() + celt as usize).min(self.items.len()));
        Ok(())
    }
    fn Reset(&self) -> WResult<()> {
        self.pos.set(0);
        Ok(())
    }
    fn Clone(&self) -> WResult<IEnumUnknown> {
        Ok(EnumUnknownObj { items: self.items.clone(), pos: Cell::new(self.pos.get()) }.into())
    }
}

// --- pixel format info -------------------------------------------------------------------------

struct PfSpec {
    bpp: u32,
    channels: u32,
    numrep: u32,
    transp: bool,
    name: &'static str,
}

/// The pixel-format spec for a `GUID_WICPixelFormat*` GUID (the formats our decoders / converter
/// emit), by the family's final byte or the one non-family format. Verified against `wic-dump pfinfo`.
fn pf_spec(g: &GUID) -> Option<PfSpec> {
    // 32bppPRGBA = {3cc4a650-…}
    if g.data1 == 0x3cc4a650 {
        return Some(PfSpec { bpp: 32, channels: 4, numrep: 2, transp: true, name: "32bpp pRGBA" });
    }
    // 32bppRGBA (straight) = {f5c7ad2d-…}
    if g.data1 == 0xf5c7ad2d {
        return Some(PfSpec { bpp: 32, channels: 4, numrep: 2, transp: true, name: "32bpp RGBA" });
    }
    if g.data1 != 0x6fddc324 {
        return None;
    }
    Some(match g.data4[7] {
        0x01 => PfSpec { bpp: 1, channels: 1, numrep: 1, transp: false, name: "1bpp Indexed" },
        0x02 => PfSpec { bpp: 2, channels: 1, numrep: 1, transp: false, name: "2bpp Indexed" },
        0x03 => PfSpec { bpp: 4, channels: 1, numrep: 1, transp: false, name: "4bpp Indexed" },
        0x04 => PfSpec { bpp: 8, channels: 1, numrep: 1, transp: false, name: "8bpp Indexed" },
        0x05 => PfSpec { bpp: 1, channels: 1, numrep: 2, transp: false, name: "Black and White" },
        0x06 => PfSpec { bpp: 2, channels: 1, numrep: 2, transp: false, name: "2bpp Gray" },
        0x07 => PfSpec { bpp: 4, channels: 1, numrep: 2, transp: false, name: "4bpp Gray" },
        0x08 => PfSpec { bpp: 8, channels: 1, numrep: 2, transp: false, name: "8bpp Gray" },
        0x09 => PfSpec { bpp: 16, channels: 3, numrep: 2, transp: false, name: "16bpp BGR555" },
        0x0a => PfSpec { bpp: 16, channels: 3, numrep: 2, transp: false, name: "16bpp BGR565" },
        0x0b => PfSpec { bpp: 16, channels: 1, numrep: 2, transp: false, name: "16bpp Gray" },
        0x0c => PfSpec { bpp: 24, channels: 3, numrep: 2, transp: false, name: "24bpp BGR" },
        0x0d => PfSpec { bpp: 24, channels: 3, numrep: 2, transp: false, name: "24bpp RGB" },
        0x0e => PfSpec { bpp: 32, channels: 3, numrep: 2, transp: false, name: "32bpp BGR" },
        0x0f => PfSpec { bpp: 32, channels: 4, numrep: 2, transp: true, name: "32bpp BGRA" },
        0x10 => PfSpec { bpp: 32, channels: 4, numrep: 2, transp: true, name: "32bpp pBGRA" },
        0x15 => PfSpec { bpp: 48, channels: 3, numrep: 2, transp: false, name: "48bpp RGB" },
        0x16 => PfSpec { bpp: 64, channels: 4, numrep: 2, transp: true, name: "64bpp RGBA" },
        0x1c => PfSpec { bpp: 32, channels: 4, numrep: 2, transp: false, name: "32bpp CMYK" },
        _ => return None,
    })
}

/// Channel `idx`'s mask for a format: sub-byte single-channel formats mask the used low bits in one
/// byte; otherwise each channel owns an equal run of the `bpp/8` bytes (byte `idx*per..`).
fn channel_mask(bpp: u32, channels: u32, idx: u32) -> Vec<u8> {
    if bpp < 8 {
        return vec![((1u16 << bpp) - 1) as u8];
    }
    let total = (bpp / 8) as usize;
    let per = total / channels as usize;
    let mut m = vec![0u8; total];
    for b in 0..per {
        m[idx as usize * per + b] = 0xff;
    }
    m
}

#[implement(IWICPixelFormatInfo2)]
pub(crate) struct PfInfoObj {
    pub(crate) format: GUID,
}

impl IWICComponentInfo_Impl for PfInfoObj_Impl {
    fn GetComponentType(&self) -> WResult<WICComponentType> {
        Ok(WICPixelFormat)
    }
    fn GetCLSID(&self) -> WResult<GUID> {
        Ok(self.format)
    }
    fn GetSigningStatus(&self) -> WResult<u32> {
        Ok(1)
    }
    fn GetAuthor(&self, cch: u32, wz: PWSTR, a: *mut u32) -> WResult<()> {
        put_wstr(AUTHOR, cch, wz, a)
    }
    fn GetVendorGUID(&self) -> WResult<GUID> {
        Ok(vendor())
    }
    fn GetVersion(&self, cch: u32, wz: PWSTR, a: *mut u32) -> WResult<()> {
        put_wstr(VERSION, cch, wz, a)
    }
    fn GetSpecVersion(&self, cch: u32, wz: PWSTR, a: *mut u32) -> WResult<()> {
        put_wstr(VERSION, cch, wz, a)
    }
    fn GetFriendlyName(&self, cch: u32, wz: PWSTR, a: *mut u32) -> WResult<()> {
        put_wstr(pf_spec(&self.format).map(|s| s.name).unwrap_or(""), cch, wz, a)
    }
}

impl IWICPixelFormatInfo_Impl for PfInfoObj_Impl {
    fn GetFormatGUID(&self) -> WResult<GUID> {
        Ok(self.format)
    }
    fn GetColorContext(&self) -> WResult<IWICColorContext> {
        Err(e(E_NOTIMPL))
    }
    fn GetBitsPerPixel(&self) -> WResult<u32> {
        // Verified pixel formats carry full info; any other format our decoders emit (e.g. BMP's
        // packed 16bppBGR555/565, 32bppBGR) still needs a correct bpp for CopyPixels sizing.
        Ok(pf_spec(&self.format).map(|s| s.bpp).unwrap_or_else(|| super::bpp_of(&self.format)))
    }
    fn GetChannelCount(&self) -> WResult<u32> {
        pf_spec(&self.format).map(|s| s.channels).ok_or(e(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT))
    }
    fn GetChannelMask(&self, idx: u32, cb: u32, pb: *mut u8, actual: *mut u32) -> WResult<()> {
        let s = pf_spec(&self.format).ok_or(e(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT))?;
        if idx >= s.channels {
            return Err(e(E_INVALIDARG));
        }
        let m = channel_mask(s.bpp, s.channels, idx);
        if !actual.is_null() {
            unsafe { *actual = m.len() as u32 };
        }
        if cb == 0 {
            return Ok(());
        }
        if (cb as usize) < m.len() || pb.is_null() {
            return Err(e(HRESULT(INSUFFICIENT_BUFFER as i32)));
        }
        unsafe { core::ptr::copy_nonoverlapping(m.as_ptr(), pb, m.len()) };
        Ok(())
    }
}

impl IWICPixelFormatInfo2_Impl for PfInfoObj_Impl {
    fn GetNumericRepresentation(&self) -> WResult<WICPixelFormatNumericRepresentation> {
        Ok(WICPixelFormatNumericRepresentation(pf_spec(&self.format).map(|s| s.numrep).unwrap_or(0) as i32))
    }
    fn SupportsTransparency(&self) -> WResult<BOOL> {
        Ok(pf_spec(&self.format).map(|s| s.transp).unwrap_or(false).into())
    }
}
