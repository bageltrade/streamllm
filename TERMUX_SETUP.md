# streamllm — Termux Setup Guide

Disk-streaming LLM engine. Runs models larger than RAM by streaming weights from storage.
Tested: 8B GGUF on 1.2 GB RAM host with ~35 MB process RSS.

## 1. Install Termux packages

```bash
pkg update -y
pkg install -y clang make git curl wget tar
```

Optional (faster downloads):

```bash
pkg install -y aria2
```

## 2. Get the source

### Option A — from the tarball you downloaded

```bash
# put streamllm_termux.tar.gz in your home (via Share, scp, or termux-setup-storage)
cd ~
tar -xzf streamllm_termux.tar.gz
# if the archive extracted files flat into ~, make a folder:
mkdir -p ~/streamllm
# if it extracted as ARCHITECTURE.md etc in current dir:
mv ARCHITECTURE.md LICENSE README.md TERMUX_SETUP.md include src validation ~/streamllm/ 2>/dev/null || true
cd ~/streamllm
```

### Option B — you already have the folder

```bash
cd ~/streamllm
```

## 3. Build

```bash
cd ~/streamllm
# Preferred:
make termux
# Or manually (important: -lc++ on Termux):
clang++ -O3 -std=c++17 -Wall -Iinclude -o engine src/engine.cpp -lc++ -lpthread
chmod +x engine
./engine --version
./engine --help
```

If `clang++` is missing: `pkg install clang`.

**Termux link error `undefined symbol: std::__ndk1::__hash_memory`:**
use the `make termux` recipe or add `-lc++` as above. Also `pkg upgrade clang libc++` if the package is old.

On some devices `-march=native` can miscompile; recipes omit it on purpose for Termux.

## 4. Download a GGUF model

Pick a size that fits your **storage** (not RAM). Examples:

**~2 GB — Qwen2.5-3B Q4_K_M (good start)**

```bash
mkdir -p ~/models && cd ~/models
wget -O Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
```

**~3.6 GB — Llama-3.1-8B IQ3_M (what we validated at ~35 MB RSS)**

```bash
mkdir -p ~/models && cd ~/models
wget -O Meta-Llama-3.1-8B-Instruct-IQ3_M.gguf \
  "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-IQ3_M.gguf"
```

Smaller / faster download options: search Hugging Face for `Q4_0` or `IQ2_M` of the same model.

## 5. Run

### Inspect (metadata only — should show low RSS)

```bash
cd ~/streamllm
./engine --model ~/models/Meta-Llama-3.1-8B-Instruct-IQ3_M.gguf --inspect
```

### Interactive chat (recommended flags for low RAM)

```bash
./engine --model ~/models/Meta-Llama-3.1-8B-Instruct-IQ3_M.gguf \
  --interactive \
  --io-mode direct \
  --memory-cap-mb 48 \
  --kv-disk /data/data/com.termux/files/home/kv.bin \
  --telemetry
```

Commands inside chat:
- type normally and press Enter
- `/reset` — clear conversation + KV window
- `/exit` or `/quit` — leave

### One-shot prompt

```bash
./engine --model ~/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  --prompt "What is the capital of France?" \
  --io-mode direct --memory-cap-mb 48 \
  --kv-disk ~/kv.bin
```

### Long context (disk-backed KV, RSS stays flat)

```bash
./engine --model ~/models/Meta-Llama-3.1-8B-Instruct-IQ3_M.gguf \
  --interactive --io-mode direct --memory-cap-mb 48 \
  --kv-disk ~/kv.bin \
  --long-context 8192 --ctx 16384
```

## 6. Flags that matter on phone

| Flag | Why |
|------|-----|
| `--io-mode direct` | pread into 4 MB staging pool — avoids page-cache blowup |
| `--memory-cap-mb 48` | soft cap on tracked pages (use 32–64 on 3–4 GB phones) |
| `--kv-disk PATH` | put KV on internal storage; use a path under `$HOME` |
| `--long-context N` | simulate large context without growing RSS |
| `--telemetry` | print RSS / bytes-read after turns |

## 7. Storage tips (Termux)

```bash
# Allow access to shared storage
termux-setup-storage

# Optional: keep models on shared storage
mkdir -p ~/storage/shared/models
# then --model ~/storage/shared/models/your.gguf
```

KV and model files can be large; prefer internal storage for speed.

## 8. Expected memory

On the validation host (1.2 GB RAM):

| Model | File size | Peak RSS during chat |
|-------|-----------|----------------------|
| Llama-3.1-8B IQ3_M | 3.6 GB | **~30–35 MB** |
| Qwen2.5-3B Q4_K_M | 1.8 GB | **~40 MB** |

Your phone will differ slightly, but RSS should stay far below model size.

## 9. Troubleshooting

| Problem | Fix |
|---------|-----|
| `clang++: not found` | `pkg install clang` |
| `error: failed to open/parse GGUF` | re-download model; check free space |
| exit code 2 | bad/missing model path |
| exit code 5 | invalid `--template` name |
| OOM killed | add `--io-mode direct --memory-cap-mb 32`, close other apps |
| Very slow first token | normal on cold storage; later turns reuse less I/O |

## 10. Project layout

```
streamllm/
  include/     gguf.hpp streamer.hpp kv_disk.hpp tokenizer.hpp telemetry.hpp
  src/         engine.cpp
  validation/  REPORT.md chat_transcript.txt bench.sh
  ARCHITECTURE.md README.md LICENSE TERMUX_SETUP.md
```

Build is a single `clang++` line — no CMake required for the core engine.
