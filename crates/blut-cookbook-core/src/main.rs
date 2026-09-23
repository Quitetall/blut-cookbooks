//! blut-core binary — registers the core + standard cookbooks and hands
//! off to the BLUT CLI.

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let reg = blut_cookbook_core::registry();
    // Return the CLI's result rather than discarding it. This used to be
    // `let _ = blut::cli::run(reg).await;`, so every failure — a failed
    // stage, a bad argument, a missing recipe — exited 0, and no script or
    // CI job calling this binary could tell a failed run from a good one.
    blut::cli::run(reg).await
}
