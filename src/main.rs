// SPDX-License-Identifier: AGPL-3.0-or-later
// Binary entry point for the standard ML cookbook.
// Usage: blut-standard recipe list / blut-standard tui / etc.
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let reg = blut_backends::registry();
    blut::cli::run(reg).await
}
