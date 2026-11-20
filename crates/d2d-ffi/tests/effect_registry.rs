//! `ID2D1Factory1::GetRegisteredEffects` and `GetEffectProperties` (item 4), matching native's
//! count/buffer contract (Direct2D's registered-effects enumeration).
use windows::Win32::Graphics::Direct2D::*;
use windows_core::{Interface, GUID};

fn factory() -> ID2D1Factory1 {
    unsafe {
        let mut raw = std::ptr::null_mut();
        d2d1::win::D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &ID2D1Factory1::IID, std::ptr::null(), &mut raw).ok().unwrap();
        ID2D1Factory1::from_raw(raw)
    }
}

#[test]
fn get_registered_effects_count_buffer_contract() {
    unsafe {
        let f = factory();
        // Query the count (no buffer): returns S_OK, *registered = total (the modelled system effects).
        let (mut returned, mut registered) = (123u32, 0u32);
        f.GetRegisteredEffects(None, Some(&mut returned), Some(&mut registered)).unwrap();
        assert_eq!(returned, 0, "no buffer ⇒ nothing written");
        assert!(registered >= 50, "the system effect set is large (got {registered})");

        // An exact-size buffer: every CLSID written, *returned = *registered, all non-NULL and unique.
        let total = registered as usize;
        let mut buf = vec![GUID::from_u128(0); total];
        let (mut r2, mut reg2) = (0u32, 0u32);
        f.GetRegisteredEffects(Some(&mut buf), Some(&mut r2), Some(&mut reg2)).unwrap();
        assert_eq!((r2 as usize, reg2 as usize), (total, total));
        assert!(buf.iter().all(|g| *g != GUID::from_u128(0)), "no GUID_NULL entries");

        // An undersized buffer: D2DERR_INSUFFICIENT_BUFFER, partial write, *returned = buffer size.
        let mut small = [GUID::from_u128(0); 2];
        let (mut r3, mut reg3) = (0u32, 0u32);
        let hr = f.GetRegisteredEffects(Some(&mut small), Some(&mut r3), Some(&mut reg3));
        assert_eq!(hr.err().map(|e| e.code().0 as u32), Some(0x8899_003A), "insufficient buffer");
        assert_eq!((r3, reg3 as usize), (2, total));

        // A write request with no `effectsReturned` pointer is E_INVALIDARG (native writes GUID_NULL).
        let mut one = [GUID::from_u128(1)];
        let hr = f.GetRegisteredEffects(Some(&mut one), None, Some(&mut reg3));
        assert_eq!(hr.err().map(|e| e.code().0 as u32), Some(0x8007_0057));
        assert_eq!(one[0], GUID::from_u128(0));
    }
}

#[test]
fn get_effect_properties_reports_the_effect_schema() {
    unsafe {
        let f = factory();
        // A built-in effect's properties object reflects its schema (Flood exposes its Color property).
        let props = f.GetEffectProperties(&CLSID_D2D1Flood).unwrap();
        assert_eq!(props.GetPropertyCount(), 1);
        // A properties object is returned for any modelled effect, not E_NOTIMPL.
        assert!(f.GetEffectProperties(&CLSID_D2D1GaussianBlur).is_ok());
        // An unknown CLSID is an error, not a properties object.
        assert!(f.GetEffectProperties(&GUID::from_u128(0xdead_beef)).is_err());
    }
}
