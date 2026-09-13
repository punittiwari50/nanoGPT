# nanoGPT Baseline: Request & Response Specification

## 1. Baseline Metadata & Runtime Context
- **Project**: `repo_nanogpt` / `build_nanogpt`
- **Source**: Canonical Karpathy nanoGPT (`andrej-karpathy/nanoGPT`)
- **Architecture**: Classic GPT-2 style Transformer with Learned Absolute Positional Embeddings (`wpe`), Layer Normalization (`ln_1`, `ln_2`, `ln_f`), GELU activations, Standard Multi-Head Self-Attention ($O(N^2)$ per-token generation, no KV-Cache).
- **Serving Protocol**: Static Model Artifact Server (`python -m http.server`) on port 8000 + CLI text generation (`sample.py`).
- **Input Preprocessing**: OpenAI `gpt2` BPE tokenizer (`tiktoken`), `vocab_size=50304` (padded from 50257).

---

## 2. Standard Benchmark Request & Response

### Request 1: Universal Benchmark Prompt
- **Command**:
  ```bash
  python3 source/sample.py \
      --out_dir=dist/model \
      --device=cpu \
      --dtype=float32 \
      --start="Explain artificial intelligence in simple terms." \
      --num_samples=1 \
      --max_new_tokens=30 \
      --temperature=0.7 \
      --top_k=40 \
      --seed=1337
  ```
- **Input Parameters**:
  | Parameter | Value | Description |
  |---|---|---|
  | `out_dir` | `dist/model` | Checkpoint directory containing `ckpt.pt` |
  | `start` | `"Explain artificial intelligence in simple terms."` | Input prompt string (or `FILE:path.txt`) |
  | `max_new_tokens` | `30` | Number of autoregressively predicted tokens |
  | `temperature` | `0.7` | Softmax logit temperature scaling |
  | `top_k` | `40` | Top-K vocabulary cutoff |
  | `seed` | `1337` | Deterministic random number seed |

### Exact Response 1:
```text
Overriding: out_dir = /workspace/dist/model
Overriding: device = cpu
Overriding: dtype = float32
Overriding: start = Explain artificial intelligence in simple terms.
Overriding: num_samples = 1
Overriding: max_new_tokens = 30
Overriding: temperature = 0.7
Overriding: top_k = 40
Overriding: seed = 1337
number of parameters: 3.32M
No meta.pkl found, assuming GPT-2 encodings...

============================================================
INPUT PROMPT: Explain artificial intelligence in simple terms.
============================================================

--- [SAMPLE 1/1] ---
>>> GENERATED OUTPUT:
oho thoroughly Marathonemaculus slasheddede Contemporary sent baff baff Ferr maternity portrait portrait050 stew Catholic Gold Fun 357 ' comparatively regularly Advancedorders task__ Between

>>> FULL SEQUENCE (PROMPT + GENERATION):
Explain artificial intelligence in simple terms.oho thoroughly Marathonemaculus slasheddede Contemporary sent baff baff Ferr maternity portrait portrait050 stew Catholic Gold Fun 357 ' comparatively regularly Advancedorders task__ Between
------------------------------------------------------------
```

---

## 3. HTTP Delivery Endpoint Request & Response
- **Endpoint**: `GET http://localhost:8000/`
- **Request**:
  ```bash
  curl -s -I http://localhost:8000/
  ```
- **Response**:
  ```http
  HTTP/1.0 200 OK
  Server: SimpleHTTP/0.6 Python/3.10.12
  Content-type: text/html; charset=utf-8
  Content-Length: 792
  ```
  *(Serves static build deliverables, wheel binaries, and model artifacts `model.bin`, `model.onnx`, `train.bin`, `val.bin`)*
