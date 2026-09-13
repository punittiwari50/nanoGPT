"""
nanoGPT Phase 3: Production Streaming HTTP Daemon
Implements OpenAI-compatible /v1/chat/completions with Server-Sent Events (SSE).
Adheres strictly to core-code-standard.md (STD-COD-001 through STD-COD-009).
"""

import json
import os
import time
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import tiktoken
import torch

from model import TransformerConfig, ModernTransformer

from contextlib import asynccontextmanager

MODEL_HOLDER = {}
ENCODER_HOLDER = {}


def load_runtime_artifacts():
    """Initializes model and tokenizer at server startup (STD-COD-002)."""
    ckpt_path = os.getenv("NANOGPT_CKPT", "dist/model/ckpt.pt")
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        config = TransformerConfig(**checkpoint["model_args"])
        model = ModernTransformer(config)
        model.load_state_dict(checkpoint["model"])
    else:
        config = TransformerConfig(n_layer=2, n_head=2, n_embd=64)
        model = ModernTransformer(config)

    model.eval()
    MODEL_HOLDER["model"] = model
    ENCODER_HOLDER["enc"] = tiktoken.get_encoding("gpt2")


@asynccontextmanager
async def lifespan(application: FastAPI):
    load_runtime_artifacts()
    yield


app = FastAPI(title="nanoGPT Production Inference API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health_probe():
    """Healthcheck endpoint for container orchestration probes (STD-BLD-022)."""
    return {"status": "healthy", "timestamp": time.time(), "model_ready": "model" in MODEL_HOLDER}


@app.get("/v1/models")
def list_models():
    """Returns available models conforming to OpenAI specification."""
    return {"object": "list", "data": [{"id": "nanogpt-phase3", "object": "model", "owned_by": "organization"}]}


async def token_stream_generator(model: ModernTransformer, enc: tiktoken.Encoding, prompt: str, max_tokens: int, temp: float, top_k: int) -> AsyncGenerator[str, None]:
    """Yields Server-Sent Events (SSE) data chunks for each generated token (STD-COD-007.2)."""
    input_ids = enc.encode_ordinary(prompt)
    x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

    for _ in range(max_tokens):
        logits, _ = model(x[:, -64:])
        logits = logits[:, -1, :] / max(temp, 1e-5)
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < values[:, [-1]]] = -float("Inf")
        next_tok = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        x = torch.cat([x, next_tok], dim=1)

        token_text = enc.decode([next_tok.item()])
        chunk = {
            "choices": [{"delta": {"content": token_text}, "index": 0, "finish_reason": None}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the author: system, user, or assistant")
    content: str = Field(..., description="Contents of the message")


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="nanogpt-phase3", description="ID of the model to use")
    messages: list[ChatMessage]
    max_tokens: int = Field(default=30, ge=1, le=256)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=40, ge=1)
    stream: bool = Field(default=False, description="Whether to stream back partial progress via SSE")


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """Processes OpenAI-compatible chat completion requests (STD-COD-007.3)."""
    model = MODEL_HOLDER.get("model")
    enc = ENCODER_HOLDER.get("enc")
    if model is None or enc is None:
        raise HTTPException(status_code=503, detail="Model runtime not initialized")

    prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages) + "\nassistant: "

    if req.stream:
        return StreamingResponse(
            token_stream_generator(model, enc, prompt, req.max_tokens, req.temperature, req.top_k),
            media_type="text/event-stream"
        )

    input_ids = enc.encode_ordinary(prompt)
    x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    out_tokens = model.generate_cached(x, max_new_tokens=req.max_tokens, temperature=req.temperature, top_k=req.top_k)
    completion_text = enc.decode(out_tokens[0][len(input_ids):].tolist())

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "choices": [{"message": {"role": "assistant", "content": completion_text}, "finish_reason": "stop", "index": 0}],
    }
