//! blut-core binary — registers the core + standard cookbooks and hands
//! off to the BLUT CLI.

/// Stack for the thread that drives the CLI. A debug build needs more than
/// the 1 MiB Windows gives the main thread: `recipe run` overflowed it there
/// (exit 0xC00000FD) before any stage started, while Linux's 8 MiB main
/// thread never noticed. Running the CLI on a thread with this stack gives
/// every platform the same.
const CLI_STACK_BYTES: usize = 8 * 1024 * 1024;

fn main() -> anyhow::Result<()> {
    let cli = std::thread::Builder::new()
        .name("blut-cli".into())
        .stack_size(CLI_STACK_BYTES)
        .spawn(|| -> anyhow::Result<()> {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            runtime.block_on(async {
                let reg = blut_cookbook_core::registry();
                // Return the CLI's result rather than discarding it. This used
                // to be `let _ = blut::cli::run(reg).await;`, so every failure
                // — a failed stage, a bad argument, a missing recipe — exited
                // 0, and no script or CI job calling this binary could tell a
                // failed run from a good one.
                blut::cli::run(reg).await
            })
        })?;
    // A panic on the CLI thread has already been reported by the panic hook;
    // re-raise it so the process still exits as a panic does.
    cli.join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
}
