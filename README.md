# Load Blaster

A simple load testing tool for LLM API endpoints.

## Setup

1. Clone the repository
2. Copy `.env.example` to `.env`
3. Update the values in `.env` with your API details
4. Build the project with `cargo build --release`

## Usage

Run with default settings (20 requests per second):

```
cargo run
```

Customize your load test:

```
cargo run -- --rps 50 --prompt "Tell me a joke" --max-concurrent 100
```

### Available Options

- `--rps`: Requests per second (default: 20)
- `--max-concurrent`: Maximum number of concurrent requests (default: 50)
- `--total-requests`: Total number of requests to send before stopping (default: 0 = infinite)
- `--prompt`: The prompt to send with each request (default: "Hello!")
- `--dataset`: Name of a Hugging Face dataset to use for prompts (e.g., "openai/summarize_from_feedback")
- `--dataset-samples`: Number of prompts to download from the dataset (default: 100)
- `--api-key`: Override the API key from .env
- `--base-url`: Override the base URL from .env
- `--model`: Override the model name from .env

## Environment Variables

You can set the following environment variables in your `.env` file:

- `OPENAI_API_KEY`: Your API key
- `OPENAI_BASE_URL`: The base URL for the API 
- `MODEL_NAME`: The model to use for completions
- `HF_TOKEN`: Your Hugging Face API token (required for dataset download)

## Using Hugging Face Datasets

You can use prompts from Hugging Face datasets:

```
cargo run -- --dataset "HuggingFaceH4/ultrafeedback_binarized" --dataset-samples 50
```

This will download prompts from the specified dataset and randomly select them for your load test.
Supported datasets include:
- `openai/summarize_from_feedback`
- `HuggingFaceH4/ultrafeedback_binarized`
- And many others with a `prompt` field