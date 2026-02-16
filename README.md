# qwen2-5-05b

Minimal OpenAI-compatible chat completions API for a GGUF model using `llama-cpp-python`.

## Run

```bash
docker build -t qwen2-5-05b .
docker run --rm -p 8000:8000 \
  -e MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  -e PORT=8000 \
  qwen2-5-05b
```

## Endpoints

- `POST /v1/chat/completions`
- `GET /health`

## Note

- By default, startup downloads a `.gguf` from Hugging Face.
