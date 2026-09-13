# nanoGPT Phase 1: Request & Response Specification

## 1. Phase Metadata & Runtime Context
- **Project**: `nanogpt-phase1`
- **Architecture**: Modern Transformer with Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU feed-forward, and PyTorch SDPA.
- **Serving Protocol**: Static Model Artifact Server (`python -m http.server`) on port 8001 + CLI text generation (`sample.py`).
- **Input Preprocessing**: OpenAI `gpt2` BPE tokenizer (`tiktoken`), `vocab_size=50304`.

---

## 2. Standard Benchmark Request & Response

### Request 1: Universal Benchmark Prompt
- **Command**:
  ```bash
  python3 source/sample.py \
      --ckpt_path=dist/model/ckpt.pt \
      --start="Explain artificial intelligence in simple terms." \
      --max_new_tokens=30 \
      --temperature=0.7 \
      --top_k=40
  ```
- **Input Parameters**:
  | Parameter | Value | Description |
  |---|---|---|
  | `start` | `"Explain artificial intelligence in simple terms."` | Input text prompt |
  | `max_new_tokens` | `30` | Number of autoregressively predicted tokens |
  | `temperature` | `0.7` | Logit scaling factor |
  | `top_k` | `40` | Top-K vocabulary truncation |

### Exact Response 1:
```text
============================================================
INPUT PROMPT: Explain artificial intelligence in simple terms.
============================================================

--- [SAMPLE 1/1] ---
>>> GENERATED OUTPUT:
And a the the and," to of of one me:<|endoftext|> is as, he's to
 andT a then to over they
 the,"

>>> FULL SEQUENCE (PROMPT + GENERATION):
Explain artificial intelligence in simple terms.And a the the and," to of of one me:<|endoftext|> is as, he's to
 andT a then to over they
 the,"
------------------------------------------------------------
```

---

### Request 2: Future Outlook Prompt
- **Command**:
  ```bash
  python3 source/sample.py \
      --ckpt_path=dist/model/ckpt.pt \
      --start="The future of intelligence" \
      --max_new_tokens=20 \
      --temperature=0.7 \
      --top_k=40
  ```

### Exact Response 2:
```text
============================================================
INPUT PROMPT: The future of intelligence
============================================================

--- [SAMPLE 1/1] ---
>>> GENERATED OUTPUT:
 have<|endoftext|>- of of  are-. ".The!<|endoftext|> a
 to in  

>>> FULL SEQUENCE (PROMPT + GENERATION):
The future of intelligence have<|endoftext|>- of of  are-. ".The!<|endoftext|> a
 to in  
------------------------------------------------------------
```

---

## 3. HTTP Delivery Endpoint Request & Response
- **Endpoint**: `GET http://localhost:8001/`
- **Request**:
  ```bash
  curl -s -I http://localhost:8001/
  ```
- **Response**:
  ```http
  HTTP/1.0 200 OK
  Server: SimpleHTTP/0.6 Python/3.12.14
  Date: Fri, 11 Sep 2026 17:56:54 GMT
  Content-type: text/html; charset=utf-8
  Content-Length: 411
  ```
