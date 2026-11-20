//! Native + wasm tests: the generated inventory is internally consistent and matches the native D2D
//! API facts (verified against native); representative value types have the expected portable layout. The `windows`
//! module additionally proves every sampled layout/value/GUID matches the `windows` crate — it only
//! compiles on a Windows target (the crate is `#![cfg(windows)]` elsewhere).

use d2d_types::{abi::*, interfaces::*, Interface, Method, Support};

fn method<'a>(iface: &'a Interface, name: &str) -> &'a Method {
    iface.methods.iter().find(|m| m.name == name).unwrap_or_else(|| panic!("{name} not in {}", iface.name))
}

#[test]
fn inventory_matches_api_note_and_preserves_inherited_slots() {
    assert_eq!(INTERFACES.len(), 310);
    assert_eq!(INTERFACES.iter().map(|i| i.methods.len()).sum::<usize>(), 1458);
    for interface in INTERFACES {
        let inherited = interface.base.map_or(0, |base| base.slots);
        assert_eq!(interface.slots, inherited + interface.methods.len() as u32, "{}", interface.name);
        for (index, m) in interface.methods.iter().enumerate() {
            assert_eq!(m.slot, inherited + index as u32);
            assert_eq!(m.support, Support::Unsupported);
        }
    }
    // Anchors from the native API and the windows vtables (RenderTarget 53/57, DeviceContext7 3/123).
    assert_eq!(ID2D1RenderTarget.slots, 57);
    assert_eq!(ID2D1RenderTarget.methods.len(), 53);
    assert_eq!(method(&ID2D1RenderTarget, "EndDraw").slot, 49);
    assert_eq!(ID2D1DeviceContext7.slots, 123);
    assert_eq!(ID2D1DeviceContext7.methods[0].slot, 120);
    assert_eq!(ID2D1Factory.slots, 17);
}

#[test]
fn representative_values_and_layouts_are_portable() {
    assert_eq!(core::mem::size_of::<D2D1_COLOR_F>(), 16);
    assert_eq!(core::mem::align_of::<D2D1_COLOR_F>(), 4);
    assert_eq!(core::mem::size_of::<D2D1_PIXEL_FORMAT>(), 8);
    assert_eq!(core::mem::size_of::<DWRITE_MATRIX>(), 24);
    assert_eq!(core::mem::size_of::<WICRect>(), 16);
    assert_eq!(core::mem::size_of::<DXGI_SAMPLE_DESC>(), 8);
    assert_eq!(DXGI_FORMAT_B8G8R8A8_UNORM.0, 87);
    assert_eq!(D2D1_ALPHA_MODE_PREMULTIPLIED.0, 1);
    assert_eq!(DWRITE_FONT_WEIGHT_NORMAL.0, 400);
    assert_eq!(CLSID_D2D1GaussianBlur.to_u128(), 0x1feb6d69_2fe6_4ac9_8c58_1d7f93e7a6a5);
    assert_eq!(ID2D1Factory.iid.to_u128(), 0x06152247_6f50_465a_9245_118bfd3b6007);
}

#[cfg(windows)]
mod windows_parity {
    use core::mem::{align_of, size_of};
    use windows::Win32::Graphics::{Direct2D, Direct2D::Common as D2DC, DirectWrite, Dxgi, Dxgi::Common as DxgiC, Imaging};
    use windows_core::Interface;

    /// abi type has the same size and alignment as the windows-crate type.
    macro_rules! same_layout {
        ($($abi:ident == $win:path),* $(,)?) => {$({
            assert_eq!(size_of::<d2d_types::abi::$abi>(), size_of::<$win>(), concat!(stringify!($abi), " size"));
            assert_eq!(align_of::<d2d_types::abi::$abi>(), align_of::<$win>(), concat!(stringify!($abi), " align"));
        })*};
    }

    #[test]
    fn layouts_match_windows_crate() {
        same_layout! {
            D2D1_COLOR_F == D2DC::D2D1_COLOR_F,
            D2D1_PIXEL_FORMAT == D2DC::D2D1_PIXEL_FORMAT,
            D2D_RECT_F == D2DC::D2D_RECT_F,
            D2D1_BITMAP_PROPERTIES1 == Direct2D::D2D1_BITMAP_PROPERTIES1,
            D2D1_LAYER_PARAMETERS1 == Direct2D::D2D1_LAYER_PARAMETERS1,
            D2D1_ARC_SEGMENT == Direct2D::D2D1_ARC_SEGMENT,
            DWRITE_MATRIX == DirectWrite::DWRITE_MATRIX,
            DWRITE_GLYPH_RUN == DirectWrite::DWRITE_GLYPH_RUN,
            DWRITE_PANOSE == DirectWrite::DWRITE_PANOSE,
            WICRect == Imaging::WICRect,
            DXGI_SAMPLE_DESC == DxgiC::DXGI_SAMPLE_DESC,
            DXGI_SWAP_CHAIN_DESC1 == Dxgi::DXGI_SWAP_CHAIN_DESC1,
        }
    }

    #[test]
    fn enum_values_match_windows_crate() {
        assert_eq!(d2d_types::abi::DXGI_FORMAT_B8G8R8A8_UNORM.0, DxgiC::DXGI_FORMAT_B8G8R8A8_UNORM.0);
        assert_eq!(d2d_types::abi::D2D1_ALPHA_MODE_PREMULTIPLIED.0, D2DC::D2D1_ALPHA_MODE_PREMULTIPLIED.0);
        assert_eq!(d2d_types::abi::DWRITE_FONT_WEIGHT_NORMAL.0, DirectWrite::DWRITE_FONT_WEIGHT_NORMAL.0);
        assert_eq!(d2d_types::abi::D2D1_FIGURE_BEGIN_HOLLOW.0, D2DC::D2D1_FIGURE_BEGIN_HOLLOW.0);
        assert_eq!(d2d_types::abi::DXGI_FORMAT_R32G32B32A32_FLOAT.0, DxgiC::DXGI_FORMAT_R32G32B32A32_FLOAT.0);
    }

    #[test]
    fn guids_and_iids_match_windows_crate() {
        assert_eq!(d2d_types::abi::CLSID_D2D1GaussianBlur, Direct2D::CLSID_D2D1GaussianBlur);
        assert_eq!(d2d_types::abi::CLSID_D2D1ColorMatrix, Direct2D::CLSID_D2D1ColorMatrix);
        // Inventory IIDs and computed vtable slot counts match the windows interfaces.
        assert_eq!(d2d_types::interfaces::ID2D1Factory.iid, Direct2D::ID2D1Factory::IID);
        assert_eq!(d2d_types::interfaces::ID2D1DeviceContext.iid, Direct2D::ID2D1DeviceContext::IID);
        assert_eq!(d2d_types::interfaces::ID2D1RenderTarget.iid, Direct2D::ID2D1RenderTarget::IID);
        let ptr = size_of::<*const core::ffi::c_void>();
        assert_eq!(d2d_types::interfaces::ID2D1Factory.slots as usize, size_of::<Direct2D::ID2D1Factory_Vtbl>() / ptr);
        assert_eq!(d2d_types::interfaces::ID2D1RenderTarget.slots as usize, size_of::<Direct2D::ID2D1RenderTarget_Vtbl>() / ptr);
        assert_eq!(d2d_types::interfaces::IDWriteFactory.iid, DirectWrite::IDWriteFactory::IID);
    }
}
