#[cfg(windows)]
#[path = "../render.rs"]
mod render;

fn main() -> anyhow::Result<()> {
    #[cfg(windows)]
    {
        render::present_run()
    }
    #[cfg(not(windows))]
    anyhow::bail!("present-ref requires Windows; run the Windows executable under Wine or on the VM")
}
