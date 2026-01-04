#!/usr/bin/env python3
"""
GGUF model API server using llama-cpp-python.
OpenAI-compatible /v1/chat/completions endpoint.
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

# ----------------------------
# Environment configuration
# ----------------------------

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
MODEL_FILE = os.getenv("MODEL_FILE", "")  # optional explicit gguf path
PORT = int(os.getenv("PORT", "8000"))
CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", "8192"))

llm: Optional[Llama] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    load_model()
    yield


app = FastAPI(title="GGUF Chat Completions API", lifespan=lifespan)


# ----------------------------
# Request / response models
# ----------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 256
    stop: Optional[List[str]] = None


# ----------------------------
# Model loading helpers
# ----------------------------

def find_gguf_file(path_or_repo: str) -> str:
    """
    Resolve a GGUF file from either:
    - a local file path
    - a local directory containing *.gguf
    - a Hugging Face repo ID containing *.gguf
    """
    p = Path(path_or_repo)

    # Local path handling
    if p.exists():
        if p.is_file() and p.suffix == ".gguf":
            return str(p)
        if p.is_dir():
            ggufs = sorted(p.glob("*.gguf"))
            if not ggufs:
                raise ValueError(f"No GGUF files found in directory: {path_or_repo}")
            return str(ggufs[0])

    # Hugging Face repo ID
    if "/" in path_or_repo:
        try:
            from huggingface_hub import list_repo_files, hf_hub_download
        except ImportError as e:
            raise ValueError(
                "huggingface_hub not installed. Install it or set MODEL_FILE to a local GGUF path."
            ) from e

        files = list_repo_files(path_or_repo)
        gguf_files = sorted(f for f in files if f.endswith(".gguf"))
        if not gguf_files:
            raise ValueError(f"No GGUF files found in HF repo: {path_or_repo}")
        filename = gguf_files[0]
        print(f"Downloading GGUF file '{filename}' from repo '{path_or_repo}'")
        return hf_hub_download(repo_id=path_or_repo, filename=filename)

    raise ValueError(f"Invalid GGUF path or repo: {path_or_repo}")


def load_model() -> None:
    """Load the GGUF model into llama-cpp (CPU)."""
    global llm

    print(f"[load_model] Loading model: {MODEL_NAME}")

    model_path = MODEL_FILE or find_gguf_file(MODEL_NAME)
    if not Path(model_path).exists():
        raise ValueError(f"Resolved model file does not exist: {model_path}")

    print(f"[load_model] Using GGUF file: {model_path}")

    llm = Llama(
        model_path=model_path,
        n_ctx=CONTEXT_SIZE,
        n_threads=None,   # auto: use all cores
        verbose=False,
    )

    print(f"[load_model] Model loaded successfully: {MODEL_NAME}")


# ----------------------------
# API endpoints
# ----------------------------

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": "cpu",
        "format": "gguf",
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "llama-cpp",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Chat completions endpoint.
    """
    global llm

    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    max_tokens = request.max_tokens if request.max_tokens is not None else 256
    temperature = request.temperature if request.temperature is not None else 0.7

    try:
        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=request.stop,
        )
        raw_text = result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    usage = result.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

    response = {
        "id": f"chatcmpl-{os.urandom(8).hex()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": raw_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }

    return response


# ----------------------------
# Entrypoint
# ----------------------------

if __name__ == "__main__":
    print(f"Starting GGUF API server for {MODEL_NAME} on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
