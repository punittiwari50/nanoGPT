# nanoGPT Phase 3: Request & Response Specification

## 1. Phase Metadata & Runtime Context
- **Project**: `nanogpt-phase3`
- **Architecture**: Production Transformer with stateful KV-Cache (`LayerKVCache` / `ModelKVCache`) for $O(1)$ step generation and INT8 dynamic quantization.
- **Serving Protocol**: Production FastAPI HTTP & Server-Sent Events (SSE) streaming daemon on port 8003.
- **API Standard**: OpenAI-compatible `/v1/chat/completions` specification.

---

## 2. OpenAI API Requests & Responses

### Request 1: Non-Streaming Chat Completion
- **HTTP Method**: `POST`
- **URL**: `http://localhost:8003/v1/chat/completions`
- **Headers**: `Content-Type: application/json`
- **Payload**:
  ```json
  {
    "model": "nanogpt-phase3",
    "messages": [
      {
        "role": "user",
        "content": "Explain artificial intelligence in simple terms."
      }
    ],
    "max_tokens": 30,
    "temperature": 0.7,
    "top_k": 40,
    "stream": false
  }
  ```

### Exact Response 1:
```json
{
  "id": "chatcmpl-1789150618",
  "object": "chat.completion",
  "created": 1789150618,
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": " they have at wasAnd for over's are's we the into, in theThe! is the \"T a it for that\n\n\n"
      },
      "finish_reason": "stop",
      "index": 0
    }
  ]
}
```

---

### Request 2: Streaming Chat Completion (Server-Sent Events)
- **HTTP Method**: `POST`
- **URL**: `http://localhost:8003/v1/chat/completions`
- **Headers**: `Content-Type: application/json`
- **Payload**:
  ```json
  {
    "model": "nanogpt-phase3",
    "messages": [
      {
        "role": "user",
        "content": "Explain artificial intelligence in simple terms."
      }
    ],
    "max_tokens": 30,
    "temperature": 0.7,
    "top_k": 40,
    "stream": true
  }
  ```

### Exact Response 2 (Raw SSE Stream):
```text
data: {"choices": [{"delta": {"content": "."}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": "T"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": " is"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": " are"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": " he"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": " \ufffd"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": " and"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": " a"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": "<|endoftext|>"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": "<|endoftext|>"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": " of"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": " into"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": " we"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": "<|endoftext|>"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": "\n"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": "The"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": " to"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": " and"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": ","}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": " are"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": ",\""}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": "'s"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": "\n"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": " a"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": "."}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": "I"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": "."}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": " he"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": "\n"}, "index": 0, "finish_reason": null}]}

data: {"choices": [{"delta": {"content": "."}, "index": 0, "finish_reason": null}]}

data: [DONE]
```

---

## 3. High-Performance KV-Cache Benchmark Execution

### Request:
- **Prompt**: `"Explain artificial intelligence in simple terms."`
- **Generated Length**: `30` tokens
- **Inference Mode**: Stateful layer-wise key/value cache

### Exact Response:
```text
============================================================
KV-CACHE BENCHMARK (30 tokens):
  KV-Cache Time: 45.08 ms (665.6 tokens/sec)
  Generated Sample: <|endoftext|> heMy he. and one and that me notAnd over! hisT...
============================================================
```

---

## 4. Health & Model Discovery Endpoints
- **Healthcheck Probe**:
  ```bash
  curl -s http://localhost:8003/health
  ```
  ```json
  {"status":"healthy","timestamp":1789150125.34543,"model_ready":true}
  ```
- **OpenAI Model Listing**:
  ```bash
  curl -s http://localhost:8003/v1/models
  ```
  ```json
  {"object":"list","data":[{"id":"nanogpt-phase3","object":"model","owned_by":"organization"}]}
  ```
