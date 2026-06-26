// SPDX-License-Identifier: MIT
// Binary entry point for the standard ML cookbook.
// Usage: blut-standard recipe list / blut-standard tui / etc.
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let reg = blut_cookbook_standard::registry();
    blut::cli::run(reg).await
}
