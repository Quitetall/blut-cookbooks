//! blut-core binary — registers the core + standard cookbooks and hands
//! off to the BLUT CLI.

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let reg = blut_cookbook_core::registry();
    blut::cli::run(reg).await
}
