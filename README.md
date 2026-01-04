# GGUF Model API Server

OpenAI-compatible chat completions API server for GGUF models using llama-cpp-python.

## Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn llama-cpp-python huggingface_hub pydantic

# Run server (downloads model automatically)
python scripts/gguf_api_server.py
```

## Usage

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` | HuggingFace repo or local path |
| `MODEL_FILE` | (auto-detect) | Specific GGUF file path |
| `PORT` | `8000` | Server port |
| `CONTEXT_SIZE` | `8192` | Context window size |

## API Endpoints

- `POST /v1/chat/completions` - Chat completions (OpenAI-compatible)
- `GET /health` - Health check

## Docker

```bash
docker build -f Dockerfile.gguf -t gguf-api .
docker run -p 8000:8000 gguf-api
```

## Tinfoil Deployment

This project includes a `tinfoil-config.yml` for confidential computing deployment via [Tinfoil](https://tinfoil.sh). The configuration runs the Qwen 0.5B model in a secure enclave with CPU-only inference.
