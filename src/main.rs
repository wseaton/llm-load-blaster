use anyhow::{Context, Result};
use async_openai::{
    config::OpenAIConfig,
    types::{ChatCompletionRequestMessage, CreateChatCompletionRequest, Role},
    Client,
};
use clap::Parser;
use dotenv::dotenv;
use futures::StreamExt as FuturesStreamExt;
use rand::seq::SliceRandom;
use std::{env, sync::Arc, time::Duration};
use tokio::sync::Semaphore;
use tokio_stream::{self as stream, StreamExt as TokioStreamExt};
use tracing::{error, info, warn};
use std::sync::atomic::{AtomicU64, Ordering};

use hf_hub::api::tokio::Api;

use load_blaster::data::{download_parquet_dataset, UltraFeedbackItem};

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

async fn download_dataset(dataset_name: &str, num_samples: usize) -> Result<Vec<String>> {
    info!("Downloading dataset: {}", dataset_name);
    let api = Api::new()?;
    match dataset_name {
        "openai/summarize_from_feedback" => {
            todo!("Implement download for openai/summarize_from_feedback");
        }
        "HuggingFaceH4/ultrafeedback_binarized" => {
            let chunks =
                download_parquet_dataset::<UltraFeedbackItem>(dataset_name, &api, num_samples)
                    .await?;
            dbg!(chunks.len());
            let res: Vec<String> = chunks
                .into_iter()
                .take(num_samples)
                .flat_map(|item| item.prompt)
                .filter(|prompt| !prompt.is_empty())
                .collect();
            Ok(res)
        }
        _ => {
            info!("Could not parse prompts from dataset, using fallback prompts");
            let fallbacks = vec![
                "Write a short story about a robot learning to paint".to_string(),
                "Explain quantum computing to a 10-year-old".to_string(),
                "What are three ways to improve productivity?".to_string(),
                "Create a recipe for a dish using only ingredients that start with 'B'".to_string(),
                "Describe the process of photosynthesis".to_string(),
            ];
            Ok(fallbacks)
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables from .env file
    dotenv().ok();

    // Initialize logging
    tracing_subscriber::fmt::init();
    info!("Starting load-blaster");

    // Parse command-line arguments
    let args = Args::parse();

    // Get configuration values with CLI args taking precedence over env vars
    let api_key = args
        .api_key
        .or_else(|| env::var("OPENAI_API_KEY").ok())
        .context("API key not found. Set it in .env file or provide --api-key")?;

    let base_url = args
        .base_url
        .or_else(|| env::var("OPENAI_BASE_URL").ok())
        .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

    let model = args
        .model
        .or_else(|| env::var("MODEL_NAME").ok())
        .unwrap_or_else(|| "gpt-3.5-turbo".to_string());

    info!("Using API base URL: {}", base_url);
    info!("Using model: {}", model);

    // Get prompts to use
    let prompts = if let Some(dataset) = args.dataset {
        download_dataset(&dataset, args.dataset_samples).await?
    } else if let Some(prompt) = args.prompt {
        vec![prompt]
    } else {
        // Default prompt
        vec!["Hello!".to_string()]
    };

    info!("Loaded {} prompts", prompts.len());

    // Configuration for the OpenAI client
    let config = OpenAIConfig::new()
        .with_api_base(&base_url)
        .with_api_key(&api_key);

    let client = Client::with_config(config);

    // Create a semaphore to limit concurrent requests
    let semaphore = Arc::new(Semaphore::new(args.max_concurrent));

    // Track statistics
    let total_sent = Arc::new(AtomicU64::new(0));
    let success_count = Arc::new(AtomicU64::new(0));
    let error_count = Arc::new(AtomicU64::new(0));

    // Create a stream that emits at the desired rate
    let interval = Duration::from_secs_f64(1.0 / args.rps as f64);
    let request_stream = stream::iter(0..args.total_requests.max(1));
    let throttled = TokioStreamExt::throttle(request_stream, interval);
    let total_sent_clone = Arc::clone(&total_sent);
    let throttled_stream = TokioStreamExt::take_while(throttled, move |_| {
        // Continue until the total requests are sent
        args.total_requests == 0 || total_sent_clone.load(Ordering::Relaxed) < args.total_requests
    });

    info!("Sending {} requests per second to {}", args.rps, base_url);
    info!("Max concurrent requests: {}", args.max_concurrent);

    // Set up shared prompts for concurrent access
    let prompts = Arc::new(prompts);

    // Process the stream
    let mut pinned_stream = Box::pin(throttled_stream);
    FuturesStreamExt::for_each_concurrent(&mut pinned_stream, None, |req_num| {
        let sem_clone = Arc::clone(&semaphore);
        let client_clone = client.clone();
        let model_clone = model.clone();
        let prompts_clone = Arc::clone(&prompts);
        let total_sent_clone = Arc::clone(&total_sent);
        let success_count_clone = Arc::clone(&success_count);
        let error_count_clone = Arc::clone(&error_count);

        async move {
            // Use semaphore to limit concurrent requests
            let permit = match sem_clone.acquire().await {
                Ok(permit) => permit,
                Err(e) => {
                    error!("Failed to acquire semaphore: {}", e);
                    return;
                }
            };

            let start = std::time::Instant::now();

            let current_total = total_sent_clone.fetch_add(1, Ordering::Relaxed) + 1;
            if current_total % 100 == 0 {
                info!(
                    "Sent {} requests (success: {}, error: {})",
                    current_total, 
                    success_count_clone.load(Ordering::Relaxed), 
                    error_count_clone.load(Ordering::Relaxed)
                );
            }

            // Choose a random prompt
            let prompt = prompts_clone
                .choose(&mut rand::thread_rng())
                .unwrap_or(&"Hello!".to_string())
                .clone();

            // Create the chat completion request
            let request = CreateChatCompletionRequest {
                model: model_clone.to_string(),
                messages: vec![ChatCompletionRequestMessage {
                    role: Role::User,
                    content: Some(prompt.clone()),
                    name: None,
                    function_call: None,
                }],
                ..Default::default()
            };

            // Send the request
            match client_clone.chat().create(request).await {
                Ok(response) => {
                    let duration = start.elapsed();
                    if let Some(choice) = response.choices.first() {
                        info!("Request {} completed in {:?}", req_num, duration);

                        let prompt_snippet = if prompt.len() > 50 {
                            format!("{}...", &prompt[..47])
                        } else {
                            prompt.clone()
                        };

                        let response_snippet = if let Some(content) = &choice.message.content {
                            if content.len() > 50 {
                                format!("{}...", &content[..47])
                            } else {
                                content.clone()
                            }
                        } else {
                            "No content".to_string()
                        };

                        info!(
                            "Prompt: \"{}\", Response: \"{}\"",
                            prompt_snippet, response_snippet
                        );
                        success_count_clone.fetch_add(1, Ordering::Relaxed);
                    } else {
                        warn!("Request {} returned no choices", req_num);
                        error_count_clone.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Err(e) => {
                    error!("Request {} failed: {}", req_num, e);
                    error_count_clone.fetch_add(1, Ordering::Relaxed);
                }
            }

            drop(permit);
        }
    })
    .await;

    info!("Load test completed");
    info!("Total requests sent: {}", total_sent.load(Ordering::Relaxed));
    info!("Successful requests: {}", success_count.load(Ordering::Relaxed));
    info!("Failed requests: {}", error_count.load(Ordering::Relaxed));

    Ok(())
}
