#[cfg(windows)]
#[path = "../render.rs"]
mod render;

fn main() -> anyhow::Result<()> {
    #[cfg(windows)]
    {
        render::run()
    }
    #[cfg(not(windows))]
    anyhow::bail!("d2d-ref requires Windows; use the Windows executable under Wine for smoke tests")
}
