pub mod data {
    use std::fs::File;

    use anyhow::Result;
    use hf_hub::{
        api::tokio::{Api, ApiRepo},
        Repo,
    };
    use parquet::{
        file::reader::{FileReader, SerializedFileReader},
        record::RecordReader,
    };

    use tracing::{error, info};

    pub const READ_SIZE: usize = 100;

    pub async fn download_dataset(dataset_name: &str, num_samples: usize) -> Result<Vec<String>> {
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
                    "Create a recipe for a dish using only ingredients that start with 'B'"
                        .to_string(),
                    "Describe the process of photosynthesis".to_string(),
                ];
                Ok(fallbacks)
            }
        }
    }

    #[derive(Debug)]
    pub struct UltraFeedbackItem {
        pub prompt: Option<String>,
    }

    impl ::parquet::record::RecordReader<UltraFeedbackItem> for Vec<UltraFeedbackItem> {
        fn read_from_row_group(
            &mut self,
            row_group_reader: &mut dyn ::parquet::file::reader::RowGroupReader,
            num_records: usize,
        ) -> Result<(), ::parquet::errors::ParquetError> {
            use ::parquet::column::reader::ColumnReader;
            // using existing param
            let mut name_to_index = std::collections::HashMap::new();
            for (idx, col) in row_group_reader
                .metadata()
                .schema_descr()
                .columns()
                .iter()
                .enumerate()
            {
                name_to_index.insert(col.name().to_string(), idx);
            }
            for _ in 0..num_records {
                self.push(UltraFeedbackItem {
                    prompt: Default::default(),
                })
            }
            let records = self;
            {
                let idx: usize = match name_to_index.get("prompt") {
                    Some(&col_idx) => col_idx,
                    None => {
                        return Err(::parquet::errors::ParquetError::General(
                            "Column 'prompt' not found".into(),
                        ));
                    }
                };
                if let Ok(column_reader) = row_group_reader.get_column_reader(idx) {
                    {
                        let mut vals = Vec::new();
                        if let ColumnReader::ByteArrayColumnReader(mut typed) = column_reader {
                            let mut definition_levels = Vec::new();
                            let (_total_num, valid_num, decoded_num) = typed.read_records(
                                num_records,
                                Some(&mut definition_levels),
                                None,
                                &mut vals,
                            )?;
                            if valid_num != decoded_num {
                                {
                                    return Err(::parquet::errors::ParquetError::General(
                                        "Invalid number of records".into(),
                                    ));
                                };
                            }
                        } else {
                            {
                                return Err(::parquet::errors::ParquetError::General(
                                    "Invalid column reader type".into(),
                                ));
                            };
                        }
                        for (i, r) in &mut records[..num_records].iter_mut().enumerate() {
                            r.prompt = Some(String::from(
                                std::str::from_utf8(vals[i].data())
                                    .expect("invalid UTF-8 sequence"),
                            ));
                        }
                    }
                } else {
                    return Err(::parquet::errors::ParquetError::General(
                        "Failed to get next column".into(),
                    ));
                }
            }
            Ok(())
        }
    }

    async fn sibling_to_parquet(
        rfilename: &str,
        repo: &ApiRepo,
    ) -> anyhow::Result<SerializedFileReader<File>> {
        let local = repo.get(rfilename).await?;
        let file = File::open(local)?;
        let reader = SerializedFileReader::new(file)?;
        Ok(reader)
    }

    async fn download_parquet_dataset<T>(
        dataset_name: &str,
        api: &Api,
        num_samples: usize,
    ) -> anyhow::Result<Vec<T>>
    where
        Vec<T>: RecordReader<T>,
        T: Send + 'static,
    {
        let repo = Repo::with_revision(
            dataset_name.to_string(),
            hf_hub::RepoType::Dataset,
            "refs/convert/parquet".to_string(),
        );
        let repo = api.repo(repo);

        let read_size = if num_samples < READ_SIZE {
            num_samples
        } else {
            READ_SIZE
        };

        let info = match repo.info().await {
            Ok(info) => info,
            Err(e) => {
                error!("Failed to get dataset info: {}", e);
                return Err(anyhow::anyhow!("Failed to get dataset info"));
            }
        };

        info!("Dataset info: {:?}", info);

        let parquet_filenames = info
            .siblings
            .into_iter()
            .filter_map(|s| {
                if s.rfilename.ends_with(".parquet") {
                    Some(s.rfilename)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let mut files = Vec::new();
        for filename in parquet_filenames {
            let reader_result = sibling_to_parquet(&filename, &repo).await;
            files.push(reader_result);
        }

        let files: Result<Vec<_>, _> = files.into_iter().collect();
        let files = files?;
        let mut chunks: Vec<T> = Vec::new();
        for file in files {
            let mut row_group = file.get_row_group(0)?;
            // Call the trait method through Vec's implementation of RecordReader trait
            chunks.read_from_row_group(&mut *row_group, read_size)?;
        }

        Ok(chunks)
    }
}

pub mod benchmark {
    use anyhow::{Context, Result};
    use async_openai::{
        config::OpenAIConfig,
        types::{ChatCompletionRequestMessage, CreateChatCompletionRequest, Role},
        Client,
    };
    use futures::StreamExt as FuturesStreamExt;
    use rand::seq::SliceRandom;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::{sync::Arc, time::Duration};
    use tokio::sync::{Mutex, Semaphore};
    use tokio_stream::{self as stream, StreamExt as TokioStreamExt};
    use tracing::{error, info};

    #[derive(Debug, Default, Clone)]
    pub struct MetricsSnapshot {
        pub total_tokens: usize,
        pub first_token_latency_ms: Vec<u64>,
        pub inter_token_latency_ms: Vec<u64>,
        pub tokens_per_second: Vec<f64>,
        pub total_time_ms: Vec<u64>,
    }

    impl MetricsSnapshot {
        pub fn average_first_token_latency_ms(&self) -> f64 {
            if self.first_token_latency_ms.is_empty() {
                return 0.0;
            }
            self.first_token_latency_ms.iter().sum::<u64>() as f64
                / self.first_token_latency_ms.len() as f64
        }

        pub fn average_inter_token_latency_ms(&self) -> f64 {
            if self.inter_token_latency_ms.is_empty() {
                return 0.0;
            }
            self.inter_token_latency_ms.iter().sum::<u64>() as f64
                / self.inter_token_latency_ms.len() as f64
        }

        pub fn average_tokens_per_second(&self) -> f64 {
            if self.tokens_per_second.is_empty() {
                return 0.0;
            }
            self.tokens_per_second.iter().sum::<f64>() / self.tokens_per_second.len() as f64
        }

        pub fn average_completion_time_ms(&self) -> f64 {
            if self.total_time_ms.is_empty() {
                return 0.0;
            }
            self.total_time_ms.iter().sum::<u64>() as f64 / self.total_time_ms.len() as f64
        }
    }

    pub struct Benchmark {
        base_url: String,
        api_key: String,
        model: String,
        rps: u64,
        max_concurrent: usize,
        total_requests: u64,
        prompts: Arc<Vec<String>>,
        total_sent: Arc<AtomicU64>,
        success_count: Arc<AtomicU64>,
        error_count: Arc<AtomicU64>,
        metrics: Arc<Mutex<MetricsSnapshot>>,
    }

    pub struct BenchmarkBuilder {
        base_url: Option<String>,
        api_key: Option<String>,
        model: Option<String>,
        rps: u64,
        max_concurrent: usize,
        total_requests: u64,
        prompts: Vec<String>,
    }

    impl BenchmarkBuilder {
        pub fn new() -> Self {
            Self {
                base_url: None,
                api_key: None,
                model: None,
                rps: 20,
                max_concurrent: 50,
                total_requests: 0,
                prompts: vec!["Hello!".to_string()],
            }
        }

        pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
            self.base_url = Some(base_url.into());
            self
        }

        pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
            self.api_key = Some(api_key.into());
            self
        }

        pub fn model(mut self, model: impl Into<String>) -> Self {
            self.model = Some(model.into());
            self
        }

        pub fn rps(mut self, rps: u64) -> Self {
            self.rps = rps;
            self
        }

        pub fn max_concurrent(mut self, max_concurrent: usize) -> Self {
            self.max_concurrent = max_concurrent;
            self
        }

        pub fn total_requests(mut self, total_requests: u64) -> Self {
            self.total_requests = total_requests;
            self
        }

        pub fn prompts(mut self, prompts: Vec<String>) -> Self {
            self.prompts = prompts;
            self
        }

        pub fn build(self) -> Result<Benchmark> {
            let base_url = self
                .base_url
                .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

            let api_key = self
                .api_key
                .context("API key not found. Set it in .env file or provide --api-key")?;

            let model = self.model.unwrap_or_else(|| "gpt-3.5-turbo".to_string());

            Ok(Benchmark {
                base_url,
                api_key,
                model,
                rps: self.rps,
                max_concurrent: self.max_concurrent,
                total_requests: self.total_requests,
                prompts: Arc::new(self.prompts),
                total_sent: Arc::new(AtomicU64::new(0)),
                success_count: Arc::new(AtomicU64::new(0)),
                error_count: Arc::new(AtomicU64::new(0)),
                metrics: Arc::new(Mutex::new(MetricsSnapshot::default())),
            })
        }
    }

    impl Default for BenchmarkBuilder {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Benchmark {
        pub fn builder() -> BenchmarkBuilder {
            BenchmarkBuilder::new()
        }

        pub async fn run(&self) -> Result<()> {
            info!("Starting load test");
            info!("Using API base URL: {}", self.base_url);
            info!("Using model: {}", self.model);
            info!("Loaded {} prompts", self.prompts.len());

            let config = OpenAIConfig::new()
                .with_api_base(&self.base_url)
                .with_api_key(&self.api_key);

            let client = Client::with_config(config);

            // limit the number of concurrent requests
            let semaphore = Arc::new(Semaphore::new(self.max_concurrent));

            let interval = Duration::from_secs_f64(1.0 / self.rps as f64);
            let request_stream = stream::iter(0..self.total_requests.max(1));
            let throttled = TokioStreamExt::throttle(request_stream, interval);
            let total_sent_clone = Arc::clone(&self.total_sent);
            let throttled_stream = TokioStreamExt::take_while(throttled, move |_| {
                self.total_requests == 0
                    || total_sent_clone.load(Ordering::Relaxed) < self.total_requests
            });

            info!(
                "Sending {} requests per second to {}",
                self.rps, self.base_url
            );
            info!("Max concurrent requests: {}", self.max_concurrent);

            // Process the stream
            let mut pinned_stream = Box::pin(throttled_stream);
            FuturesStreamExt::for_each_concurrent(&mut pinned_stream, None, |req_num| {
                self.process_streaming_request(req_num, Arc::clone(&semaphore), client.clone())
            })
            .await;

            // Report metrics summary
            let metrics = self.metrics.lock().await;
            info!("Load test completed");
            info!(
                "Total requests sent: {}",
                self.total_sent.load(Ordering::Relaxed)
            );
            info!(
                "Successful requests: {}",
                self.success_count.load(Ordering::Relaxed)
            );
            info!(
                "Failed requests: {}",
                self.error_count.load(Ordering::Relaxed)
            );
            info!("Total tokens: {}", metrics.total_tokens);
            info!(
                "Average TTFT (time to first token): {:.2} ms",
                metrics.average_first_token_latency_ms()
            );
            info!(
                "Average inter-token latency: {:.2} ms",
                metrics.average_inter_token_latency_ms()
            );
            info!(
                "Average tokens per second: {:.2}",
                metrics.average_tokens_per_second()
            );
            info!(
                "Average completion time: {:.2} ms",
                metrics.average_completion_time_ms()
            );

            Ok(())
        }

        async fn process_streaming_request(
            &self,
            req_num: u64,
            semaphore: Arc<Semaphore>,
            client: Client<OpenAIConfig>,
        ) {
            let permit = match semaphore.acquire().await {
                Ok(permit) => permit,
                Err(e) => {
                    error!("Failed to acquire semaphore: {}", e);
                    return;
                }
            };

            let current_total = self.total_sent.fetch_add(1, Ordering::Relaxed) + 1;
            if current_total % 100 == 0 {
                info!(
                    "Sent {} requests (success: {}, error: {})",
                    current_total,
                    self.success_count.load(Ordering::Relaxed),
                    self.error_count.load(Ordering::Relaxed)
                );
            }

            let prompt = self
                .prompts
                .choose(&mut rand::thread_rng())
                .unwrap_or(&"Hello!".to_string())
                .clone();

            let prompt_snippet = if prompt.len() > 50 {
                format!("{}...", &prompt[..47])
            } else {
                prompt.clone()
            };

            let start_time = std::time::Instant::now();
            let request = CreateChatCompletionRequest {
                model: self.model.to_string(),
                messages: vec![ChatCompletionRequestMessage {
                    role: Role::User,
                    content: Some(prompt.clone()),
                    name: None,
                    function_call: None,
                }],
                stream: Some(true),
                ..Default::default()
            };

            // Track streaming metrics
            let mut token_count = 0;
            let mut first_token_received = false;
            let mut first_token_time = None;
            let mut last_token_time = start_time;
            let mut last_content = String::new();
            let mut inter_token_latencies = Vec::new();

            // Create the streaming request
            match client.chat().create_stream(request).await {
                Ok(mut stream) => {
                    while let Some(response) = FuturesStreamExt::next(&mut stream).await {
                        match response {
                            Ok(response) => {
                                for choice in &response.choices {
                                    if let Some(ref content_delta) = choice.delta.content {
                                        if !content_delta.is_empty()
                                            && !content_delta.trim().is_empty()
                                        {
                                            let now = std::time::Instant::now();

                                            // Track token timing
                                            token_count += 1;

                                            if !first_token_received {
                                                first_token_received = true;
                                                first_token_time = Some(now);

                                                // Time to first token
                                                let ttft =
                                                    now.duration_since(start_time).as_millis()
                                                        as u64;
                                                info!(
                                                    "Request {} - First token received in {} ms",
                                                    req_num, ttft
                                                );
                                            } else {
                                                // Inter-token latency
                                                let inter_token_latency =
                                                    now.duration_since(last_token_time).as_millis()
                                                        as u64;
                                                inter_token_latencies.push(inter_token_latency);
                                            }

                                            last_token_time = now;
                                            last_content.push_str(content_delta);
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                error!("Request {} streaming error: {}", req_num, e);
                                self.error_count.fetch_add(1, Ordering::Relaxed);
                                break;
                            }
                        }
                    }

                    // Calculate final metrics
                    let total_time = start_time.elapsed();
                    let total_time_ms = total_time.as_millis() as u64;
                    let tokens_per_second = if total_time.as_secs_f64() > 0.0 {
                        token_count as f64 / total_time.as_secs_f64()
                    } else {
                        0.0
                    };

                    let response_snippet = if last_content.len() > 50 {
                        format!("{}...", &last_content[..47])
                    } else {
                        last_content.clone()
                    };

                    info!(
                        "Request {} completed in {:?} - {} tokens, {:.2} tokens/sec",
                        req_num, total_time, token_count, tokens_per_second
                    );
                    info!(
                        "Prompt: \"{}\", Response: \"{}\"",
                        prompt_snippet, response_snippet
                    );

                    // Update global metrics
                    if first_token_time.is_some() {
                        let mut metrics = self.metrics.lock().await;
                        metrics.total_tokens += token_count;
                        metrics.first_token_latency_ms.push(
                            first_token_time
                                .unwrap()
                                .duration_since(start_time)
                                .as_millis() as u64,
                        );
                        metrics.inter_token_latency_ms.extend(inter_token_latencies);
                        metrics.tokens_per_second.push(tokens_per_second);
                        metrics.total_time_ms.push(total_time_ms);
                    }

                    self.success_count.fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => {
                    error!("Request {} failed to start streaming: {}", req_num, e);
                    self.error_count.fetch_add(1, Ordering::Relaxed);
                }
            }

            drop(permit);
        }
    }
}
