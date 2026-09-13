# nanoGPT Hermetic Build & Operational Runbook

> **Standard Compliance:** `STD-BLD-001` through `STD-BLD-021`  
> **Lifecycle Pattern:** Canonical 5-Phase Sequential Lifecycle (`Clean` $\rightarrow$ `Build` $\rightarrow$ `Train` $\rightarrow$ `Run` $\rightarrow$ `Verify`)  
> **Platform Parity:** Windows (PowerShell / CMD), Linux (GNU Bash), macOS (Zsh / Homebrew)  
> **Hardware Support:** CPU (default fallback) and NVIDIA CUDA (auto-detected `sm_120`, `sm_89`, etc.)

---

## 1. Quick-Start Execution Matrix

Execute the complete 5-phase lifecycle directly from the `build_nanogpt` directory using the platform-specific commands below:

| Phase | Windows (PowerShell 7+ / 5.1) | Linux (Bash) | macOS (Zsh) |
| :--- | :--- | :--- | :--- |
| **1. Clean** | `docker compose -f docker/docker-compose.yaml down -v --remove-orphans; if (Test-Path dist) { Remove-Item -Recurse -Force dist }` | `docker compose -f docker/docker-compose.yaml down -v --remove-orphans && rm -rf dist/` | `docker compose -f docker/docker-compose.yaml down -v --remove-orphans && rm -rf dist/` |
| **2. Build** | `docker compose -f docker/docker-compose.yaml build` | `docker compose -f docker/docker-compose.yaml build` | `docker compose -f docker/docker-compose.yaml build` |
| **3. Train** | `docker compose -f docker/docker-compose.yaml run --rm nanogpt-build` | `docker compose -f docker/docker-compose.yaml run --rm nanogpt-build` | `docker compose -f docker/docker-compose.yaml run --rm nanogpt-build` |
| **4. Run** | `docker compose -f docker/docker-compose.yaml up -d nanogpt-app` | `docker compose -f docker/docker-compose.yaml up -d nanogpt-app` | `docker compose -f docker/docker-compose.yaml up -d nanogpt-app` |
| **5. Verify** | `docker compose -f docker/docker-compose.yaml run --rm nanogpt-verify` | `docker compose -f docker/docker-compose.yaml run --rm nanogpt-verify` | `docker compose -f docker/docker-compose.yaml run --rm nanogpt-verify` |

> **Zero-Friction Compose Execution:** Thanks to `.env` (`COMPOSE_FILE=docker/docker-compose.yaml`), you can run `docker compose <cmd>` directly (e.g. `docker compose build`, `docker compose run --rm nanogpt-build`) from `build_nanogpt` without needing the `-f docker/docker-compose.yaml` flag!

---

## 2. Phase-by-Phase Operational Guide

### Phase 1: CLEAN (Workspace & Container Sanitization)

#### Objective
Reset the build environment by terminating active containers, removing orphaned networks, purging stale distribution deliverables (`dist/`), and clearing temporary cache files. This guarantees 100% reproducible bit-for-bit compilation.

#### Windows (PowerShell)
```powershell
# Navigate to build directory
cd build_nanogpt

# Teardown existing containers, networks, and anonymous volumes
docker compose -f docker/docker-compose.yaml down -v --remove-orphans

# Clean host dist directory and stale model weights
if (Test-Path dist) {
    Write-Host "Removing dist directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force dist
}

# Optional: Prune dangling images to conserve disk space
docker image prune -f
```

#### Linux (Bash)
```bash
# Navigate to build directory
cd build_nanogpt

# Teardown existing containers, networks, and anonymous volumes
docker compose -f docker/docker-compose.yaml down -v --remove-orphans

# Clean host dist directory and stale model weights
rm -rf dist/

# Optional: Prune dangling images to conserve disk space
docker image prune -f
```

#### macOS (Zsh)
```zsh
# Navigate to build directory
cd build_nanogpt

# Teardown existing containers, networks, and anonymous volumes
docker compose -f docker/docker-compose.yaml down -v --remove-orphans

# Clean host dist directory and stale model weights
rm -rf dist/

# Optional: Prune dangling images to conserve disk space
docker image prune -f
```

---

### Phase 2: BUILD (Hermetic Toolchains & Container Image Compilation)

#### Objective
Build the multi-stage Docker image using BuildKit. The container encapsulates Python 3.10+, PyTorch, `uv`, `onnx`, and C++ compilation tools into an isolated virtual environment (`/opt/venv_nanogpt`). Zero dependencies are installed onto your host machine.

#### Windows (PowerShell)
```powershell
# Ensure BuildKit is enabled
$env:DOCKER_BUILDKIT = 1

# Build the hermetic container image
docker compose -f docker/docker-compose.yaml build

# Inspect the compiled builder image
docker images | Select-String "build_nanogpt"
```

#### Linux (Bash)
```bash
# Ensure BuildKit is enabled
export DOCKER_BUILDKIT=1

# Build the hermetic container image
docker compose -f docker/docker-compose.yaml build

# Inspect the compiled builder image
docker images | grep "build_nanogpt"
```

#### macOS (Zsh)
```zsh
# Ensure BuildKit is enabled
export DOCKER_BUILDKIT=1

# Build the hermetic container image
docker compose -f docker/docker-compose.yaml build

# Inspect the compiled builder image
docker images | grep "build_nanogpt"
```

---

### Phase 3: TRAIN (Data Ingestion, Tokenization, Training & Dual Export)

#### Objective
Inside the isolated container, the build runner:
1. Ingests raw datasets across multiple vendors:
   - **GitHub Raw:** TinyShakespeare
   - **Hugging Face:** TinyStories (`roneneldan/TinyStories`)
   - **Hugging Face:** OpenWebText (`openwebtext`)
2. Tokenizes and unifies all corpora using GPT-2 BPE tokenizer into:
   - `/workspace/dist/train.bin` (~1,018,745 tokens)
   - `/workspace/dist/val.bin` (~115,702 tokens)
   - `/workspace/dist/meta.pkl`
3. Packages the project source into a standalone Python wheel:
   - `/workspace/dist/nanogpt-0.1.0-py3-none-any.whl`
4. Executes model training loop using `train.py --dataset=combined`.
5. Exports dual model deliverables:
   - **PyTorch Checkpoint:** `/workspace/dist/model.bin` and `/workspace/dist/model/ckpt.pt`
   - **ONNX Model:** `/workspace/dist/model.onnx` (exported via `torch.onnx.export(dynamo=False, opset_version=17)`)

#### Windows (PowerShell)
```powershell
# Execute the complete build, dataset preparation, training, and export pipeline
docker compose -f docker/docker-compose.yaml run --rm nanogpt-build

# Verify generated deliverables in host dist directory
Get-ChildItem -Path dist
```

#### Linux (Bash)
```bash
# Execute the complete build, dataset preparation, training, and export pipeline
docker compose -f docker/docker-compose.yaml run --rm nanogpt-build

# Verify generated deliverables in host dist directory
ls -lh dist/
```

#### macOS (Zsh)
```zsh
# Execute the complete build, dataset preparation, training, and export pipeline
docker compose -f docker/docker-compose.yaml run --rm nanogpt-build

# Verify generated deliverables in host dist directory
ls -lh dist/
```

---

### Phase 4: RUN (Inference Daemon Startup & Readiness Probing)

#### Objective
Validate host port 8000 availability, launch the hardened background runtime container in detached mode (`-d`), and poll HTTP readiness probe until healthy (`200 OK`).

#### Windows (PowerShell)
```powershell
# Step 1: Pre-flight check for port 8000
$portActive = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($portActive) {
    Write-Warning "Port 8000 is occupied. Terminating existing process or choosing another port..."
}

# Step 2: Start container in background
docker compose -f docker/docker-compose.yaml up -d nanogpt-app

# Step 3: Poll health endpoint until healthy
Write-Host "Waiting for service to become healthy on port 8000..." -ForegroundColor Cyan
do {
    Start-Sleep -Seconds 1
    $statusCode = (curl.exe -s -o /dev/null -w "%{http_code}" http://localhost:8000/)
    Write-Host "Probe response: $statusCode"
} until ($statusCode -eq "200")

Write-Host "nanogpt-app is ONLINE and READY!" -ForegroundColor Green

# Step 4: Perform a sample text generation request
curl.exe -s -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{\"prompt\": \"Once upon a time\", \"max_new_tokens\": 30}'
```

#### Linux (Bash)
```bash
# Step 1: Pre-flight check for port 8000
if ss -tuln | grep -q ':8000 '; then
    echo "WARNING: Port 8000 is occupied!" >&2
fi

# Step 2: Start container in background
docker compose -f docker/docker-compose.yaml up -d nanogpt-app

# Step 3: Poll health endpoint until healthy
echo "Waiting for service to become healthy on port 8000..."
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/)" -eq 200 ]; do
    sleep 1
done
echo "nanogpt-app is ONLINE and READY!"

# Step 4: Perform a sample text generation request
curl -s -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Once upon a time", "max_new_tokens": 30}'
```

#### macOS (Zsh)
```zsh
# Step 1: Pre-flight check for port 8000
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "WARNING: Port 8000 is occupied!" >&2
fi

# Step 2: Start container in background
docker compose -f docker/docker-compose.yaml up -d nanogpt-app

# Step 3: Poll health endpoint until healthy
echo "Waiting for service to become healthy on port 8000..."
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/)" -eq 200 ]; do
    sleep 1
done
echo "nanogpt-app is ONLINE and READY!"

# Step 4: Perform a sample text generation request
curl -s -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Once upon a time", "max_new_tokens": 30}'
```

---

### Phase 5: VERIFY (Automated 7-Gate Validation Suite)

#### Objective
Run the non-interactive automated verification container (`nanogpt-verify`). It rigorously executes 7 verification gates and exits with code `0` only if all gates pass.

#### The 7 Verification Gates
1. **Gate 1:** In-container virtual environment auto-switch (`/opt/venv_nanogpt`).
2. **Gate 2:** Zero `.cache` in dataset environment variables (`NANOGPT_DATASET_DIR`).
3. **Gate 3:** Structured host dataset repositories and manifests (`metadata.json`).
4. **Gate 4:** Binary deliverables integrity (`train.bin`, `val.bin`, `model.bin`, `model.onnx`).
5. **Gate 5:** Standalone Python wheel package verification (`dist/*.whl`).
6. **Gate 6:** PyTorch model inference & sample text generation (`sample.py`).
7. **Gate 7:** ONNX model structural integrity validation (`onnx.checker.check_model`).

#### Windows (PowerShell)
```powershell
# Execute the automated verification suite
docker compose -f docker/docker-compose.yaml run --rm nanogpt-verify

# Check exit code (0 = success)
if ($LASTEXITCODE -eq 0) {
    Write-Host "ALL 7 GATES PASSED! Build is 100% verified." -ForegroundColor Green
} else {
    Write-Error "Verification failed with exit code $LASTEXITCODE"
}
```

#### Linux (Bash)
```bash
# Execute the automated verification suite
docker compose -f docker/docker-compose.yaml run --rm nanogpt-verify

# Check exit code
if [ $? -eq 0 ]; then
    echo "ALL 7 GATES PASSED! Build is 100% verified."
else
    echo "Verification failed!" >&2
    exit 1
fi
```

#### macOS (Zsh)
```zsh
# Execute the automated verification suite
docker compose -f docker/docker-compose.yaml run --rm nanogpt-verify

# Check exit code
if [ $? -eq 0 ]; then
    echo "ALL 7 GATES PASSED! Build is 100% verified."
else
    echo "Verification failed!" >&2
    exit 1
fi
```

---

## 3. Interactive Shell & Developer Debugging

If you need to enter the container for interactive debugging, inspection, or manual script execution:

#### Windows / Linux / macOS
```bash
# Launch interactive bash session in the builder environment
docker compose -f docker/docker-compose.yaml run --rm nanogpt-build bash

# Inside the container:
# Virtual environment /opt/venv_nanogpt is activated automatically!
which python3
# Output: /opt/venv_nanogpt/bin/python3

# Run sample text generation directly:
python3 /workspace/source/sample.py --out_dir=/workspace/dist/model --dataset=combined --max_new_tokens=50 --device=cpu

# Validate ONNX model directly:
python3 -c "import onnx; model = onnx.load('/workspace/dist/model.onnx'); onnx.checker.check_model(model); print('ONNX Validated successfully!')"
```

---

## 4. Teardown & Maintenance

When testing is complete:

```bash
# Stop and remove background runtime containers
docker compose -f docker/docker-compose.yaml down -v --remove-orphans
```
