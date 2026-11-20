//! `ID2D1DrawingStateBlock1` and `ID2D1GradientStopCollection1` round-trip their `...1` state through
//! the real COM surface (item 4: the `CreateDrawingStateBlock1` / `CreateGradientStopCollection1`
//! objects). The base-overload creators map to the `...1` getters exactly as native Direct2D does.
use windows::Win32::Graphics::Direct2D::Common::*;
use windows::Win32::Graphics::Direct2D::*;
use windows_core::Interface;
use windows_numerics::Matrix3x2;

fn factory() -> ID2D1Factory1 {
    unsafe {
        let mut raw = std::ptr::null_mut();
        d2d1::win::D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &ID2D1Factory1::IID, std::ptr::null(), &mut raw).ok().unwrap();
        ID2D1Factory1::from_raw(raw)
    }
}

#[test]
fn drawing_state_block1_roundtrips_blend_and_unit_mode() {
    unsafe {
        let f = factory();
        let desc = D2D1_DRAWING_STATE_DESCRIPTION1 {
            antialiasMode: D2D1_ANTIALIAS_MODE_ALIASED,
            textAntialiasMode: D2D1_TEXT_ANTIALIAS_MODE_GRAYSCALE,
            tag1: 7,
            tag2: 9,
            transform: Matrix3x2 { M11: 2.0, M12: 0.0, M21: 0.0, M22: 3.0, M31: 4.0, M32: 5.0 },
            primitiveBlend: D2D1_PRIMITIVE_BLEND_COPY,
            unitMode: D2D1_UNIT_MODE_PIXELS,
        };
        let block = f.CreateDrawingStateBlock(Some(&desc), None).unwrap();
        let mut out = D2D1_DRAWING_STATE_DESCRIPTION1::default();
        block.GetDescription(&mut out);
        assert_eq!(out.primitiveBlend, D2D1_PRIMITIVE_BLEND_COPY);
        assert_eq!(out.unitMode, D2D1_UNIT_MODE_PIXELS);
        assert_eq!(out.antialiasMode, D2D1_ANTIALIAS_MODE_ALIASED);
        assert_eq!(out.tag1, 7);
        assert_eq!(out.transform.M22, 3.0);

        // The base ID2D1Factory::CreateDrawingStateBlock makes the same object type (QIs to ...1).
        let base = f.cast::<ID2D1Factory>().unwrap();
        let b2 = base.CreateDrawingStateBlock(None, None).unwrap();
        assert!(b2.cast::<ID2D1DrawingStateBlock1>().is_ok());
    }
}

#[test]
fn gradient_stop_collection1_roundtrips_and_base_overload_maps_to_native_defaults() {
    unsafe {
        let f = factory();
        let dc = f.CreateDevice(None).unwrap().CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE).unwrap();
        let stops = [
            D2D1_GRADIENT_STOP { position: 0.0, color: D2D1_COLOR_F { r: 1.0, g: 0.0, b: 0.0, a: 1.0 } },
            D2D1_GRADIENT_STOP { position: 1.0, color: D2D1_COLOR_F { r: 0.0, g: 1.0, b: 0.0, a: 1.0 } },
        ];
        // The ...1 overload stores every field verbatim.
        let c1 = dc
            .CreateGradientStopCollection(&stops, D2D1_COLOR_SPACE_SCRGB, D2D1_COLOR_SPACE_SRGB, D2D1_BUFFER_PRECISION_16BPC_FLOAT, D2D1_EXTEND_MODE_MIRROR, D2D1_COLOR_INTERPOLATION_MODE_PREMULTIPLIED)
            .unwrap();
        assert_eq!(c1.GetPreInterpolationSpace(), D2D1_COLOR_SPACE_SCRGB);
        assert_eq!(c1.GetPostInterpolationSpace(), D2D1_COLOR_SPACE_SRGB);
        assert_eq!(c1.GetBufferPrecision(), D2D1_BUFFER_PRECISION_16BPC_FLOAT);
        assert_eq!(c1.GetExtendMode(), D2D1_EXTEND_MODE_MIRROR);
        assert_eq!(c1.GetColorInterpolationMode(), D2D1_COLOR_INTERPOLATION_MODE_PREMULTIPLIED);

        // The base ID2D1RenderTarget overload maps (as Direct2D's CreateGradientStopCollection base
        // overload does) to pre = post = SRGB, 8BPC_UNORM, STRAIGHT, regardless of gamma.
        let rt = dc.cast::<ID2D1RenderTarget>().unwrap();
        let base = rt.CreateGradientStopCollection(&stops, D2D1_GAMMA_1_0, D2D1_EXTEND_MODE_CLAMP).unwrap();
        let b1 = base.cast::<ID2D1GradientStopCollection1>().unwrap();
        assert_eq!(b1.GetPreInterpolationSpace(), D2D1_COLOR_SPACE_SRGB);
        assert_eq!(b1.GetPostInterpolationSpace(), D2D1_COLOR_SPACE_SRGB);
        assert_eq!(b1.GetBufferPrecision(), D2D1_BUFFER_PRECISION_8BPC_UNORM);
        assert_eq!(b1.GetColorInterpolationMode(), D2D1_COLOR_INTERPOLATION_MODE_STRAIGHT);
        assert_eq!(base.GetExtendMode(), D2D1_EXTEND_MODE_CLAMP);
    }
}
