# nanoGPT Phase 2: Request & Response Specification

## 1. Phase Metadata & Runtime Context
- **Project**: `nanogpt-phase2`
- **Architecture**: Modern Transformer with LoRA adapters (rank 4, alpha 8.0) and Instruction Supervised Fine-Tuning (SFT) with prompt loss masking.
- **Serving Protocol**: Static Artifact Daemon on port 8002 + Instruction generation CLI (`sample.py`).
- **Prompt Format**: Alpaca-style instruction template:
  ```text
  ### Instruction:
  {instruction}

  ### Response:
  ```

---

## 2. Standard Benchmark Request & Response

### Request 1: Universal Benchmark Prompt (Direct)
- **Command**:
  ```bash
  python3 source/sample.py \
      --ckpt_path=dist/model/ckpt.pt \
      --start="Explain artificial intelligence in simple terms." \
      --max_new_tokens=30 \
      --temperature=0.7 \
      --top_k=40
  ```

### Exact Response 1:
```text
============================================================
INPUT INSTRUCTION PROMPT:
Explain artificial intelligence in simple terms.
============================================================
>>> GENERATED INSTRUCTION RESPONSE:
 toA . and of!,"The:T the, he one one. to
 The- into the to-: we was:

>>> FULL SEQUENCE:
Explain artificial intelligence in simple terms. toA . and of!,"The:T the, he one one. to
 The- into the to-: we was:
------------------------------------------------------------
```

---

### Request 2: Formatted Instruction Prompt
- **Command**:
  ```bash
  python3 source/sample.py \
      --ckpt_path=dist/model/ckpt.pt \
      --start="### Instruction:\nExplain artificial intelligence in simple terms.\n\n### Response:\n" \
      --max_new_tokens=30 \
      --temperature=0.7 \
      --top_k=40
  ```

### Exact Response 2:
```text
============================================================
INPUT INSTRUCTION PROMPT:
### Instruction:\nExplain artificial intelligence in simple terms.\n\n### Response:\n
============================================================
>>> GENERATED INSTRUCTION RESPONSE:
  at into  have world<|endoftext|> for the,T for ofI<|endoftext|> to and is to and
 into.
 of he 
 " at

>>> FULL SEQUENCE:
### Instruction:\nExplain artificial intelligence in simple terms.\n\n### Response:\n  at into  have world<|endoftext|> for the,T for ofI<|endoftext|> to and is to and
 into.
 of he 
 " at
------------------------------------------------------------
```

---

## 3. HTTP Delivery Endpoint Request & Response
- **Endpoint**: `GET http://localhost:8002/`
- **Request**:
  ```bash
  curl -s -I http://localhost:8002/
  ```
- **Response**:
  ```http
  HTTP/1.0 200 OK
  Server: SimpleHTTP/0.6 Python/3.12.14
  Date: Fri, 11 Sep 2026 18:02:26 GMT
  Content-type: text/html; charset=utf-8
  Content-Length: 366
  ```
