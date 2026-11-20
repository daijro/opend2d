//! De-risk loader: calls the exported `D2D1CreateFactory` (resolved from our own `d2d1.dll`) and a
//! COM method on the returned object. Run under Wine with `d2d1.dll` beside the exe.
#[cfg(windows)]
fn main() {
    use windows::Win32::Graphics::Direct2D::{D2D1CreateFactory, ID2D1Multithread, D2D1_FACTORY_TYPE_SINGLE_THREADED};
    unsafe {
        let mt: ID2D1Multithread = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, None).unwrap();
        println!("SMOKE OK protected={}", windows_core::Interface::as_raw(&mt) as usize != 0 && mt.GetMultithreadProtected().as_bool());
    }
}
#[cfg(not(windows))]
fn main() {}
