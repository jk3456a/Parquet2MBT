mod cli;
mod config;
mod scanner;
mod reader;
mod preprocessor;
mod tokenizer;
mod writer;
mod index;
mod pipeline;
mod metrics;
use anyhow::Result;
use tracing_subscriber::EnvFilter;

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_level(true)
        .with_ansi(false) // 禁用ANSI颜色，避免日志中出现控制字符
        .compact()
        .init();
}

fn main() -> Result<()> {
    init_tracing();
    let args = cli::parse();
    let cfg = config::Config::from_args(&args)?;
    tracing::info!(?cfg, "starting parquet2mbt");

    let files = scanner::scan_inputs(&cfg)?;
    tracing::info!(count=%files.len(), "files matched");

    pipeline::run(cfg, files)?;
    Ok(())
}


