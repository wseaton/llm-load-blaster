use anyhow::Result;
use clap::Parser;
use dotenv::dotenv;
use std::env;
use tracing::info;

use load_blaster::{benchmark::Benchmark, data::download_dataset};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Base URL for the API
    #[arg(long)]
    base_url: Option<String>,

    /// API key to use
    #[arg(long)]
    api_key: Option<String>,

    /// Model to use for chat completion
    #[arg(long)]
    model: Option<String>,

    /// Number of requests per second
    #[arg(long, default_value_t = 20)]
    rps: u64,

    /// Maximum concurrent requests
    #[arg(long, default_value_t = 50)]
    max_concurrent: usize,

    /// Total number of requests to make (0 for infinite)
    #[arg(long, default_value_t = 0)]
    total_requests: u64,

    /// Prompt to use
    #[arg(long)]
    prompt: Option<String>,

    /// Dataset to use for prompts
    #[arg(long)]
    dataset: Option<String>,

    /// Number of prompts to download from the dataset
    #[arg(long, default_value_t = 100)]
    dataset_samples: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    tracing_subscriber::fmt::init();
    info!("Starting load-blaster");
    let args = Args::parse();

    let api_key = args.api_key.or_else(|| env::var("OPENAI_API_KEY").ok());

    let base_url = args.base_url.or_else(|| env::var("OPENAI_BASE_URL").ok());

    let model = args.model.or_else(|| env::var("MODEL_NAME").ok());

    let prompts = if let Some(dataset) = args.dataset {
        download_dataset(&dataset, args.dataset_samples).await?
    } else if let Some(prompt) = args.prompt {
        vec![prompt]
    } else {
        // Default prompt
        vec!["Hello!".to_string()]
    };

    let benchmark = Benchmark::builder()
        .rps(args.rps)
        .max_concurrent(args.max_concurrent)
        .total_requests(args.total_requests)
        .prompts(prompts);

    let benchmark = if let Some(api_key) = api_key {
        benchmark.api_key(api_key)
    } else {
        benchmark
    };

    let benchmark = if let Some(base_url) = base_url {
        benchmark.base_url(base_url)
    } else {
        benchmark
    };

    let benchmark = if let Some(model) = model {
        benchmark.model(model)
    } else {
        benchmark
    };

    let benchmark = benchmark.build()?;
    benchmark.run().await?;

    Ok(())
}
