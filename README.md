# streamllm

**Disk-streaming LLM inference engine** — run GGUF models larger than RAM with process RSS in the low tens of megabytes.

Weights stay on disk. KV cache is a sparse disk file. A fixed staging pool (4 MB) is the only place tensor bytes briefly live in process memory.

Validated on a **1.2 GiB RAM** host with a **3.6 GiB / 8B** model at **~30–35 MB** peak RSS and coherent multi-turn chat.

## Features

- **GGUF** metadata-only load (tensor bytes never bulk-loaded)
- **Dual I/O**: `mmap` with hard page-cache soft cap + `MADV_DONTNEED`, or pure `--io-mode direct` (`pread`)
- **Disk-backed KV** — context length does not grow RSS
- **`--long-context N`** — bulk context via sparse KV
- **Chat templates** — ChatML / Llama-3 / Mistral / Gemma heuristics + GGUF `tokenizer.chat_template`
- **Interactive CLI** — `/exit`, `/reset`, Ctrl-C cancel, telemetry
- **Zero `MADV_WILLNEED`** — never pulls the whole model into cache by policy
- Single translation unit — **no CMake required**

## Quick start

```bash
# Build (Linux / Termux / macOS with clang++)
clang++ -O3 -std=c++17 -Wall -o engine src/engine.cpp -lpthread
# or: g++ -O3 -march=native -std=c++17 -Wall -o engine src/engine.cpp -lpthread

./engine --version
./engine --help
```

### Download a model (example)

```bash
mkdir -p models
# ~3.6 GB — Llama 3.1 8B IQ3_M (validated)
wget -O models/Meta-Llama-3.1-8B-Instruct-IQ3_M.gguf \
  "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-IQ3_M.gguf"
```

### Inspect (metadata only)

```bash
./engine --model models/Meta-Llama-3.1-8B-Instruct-IQ3_M.gguf --inspect
# Expect RSS ~30 MB even for multi-GB files
```

### Interactive chat (low RAM)

```bash
./engine --model models/Meta-Llama-3.1-8B-Instruct-IQ3_M.gguf \
  --interactive \
  --io-mode direct \
  --memory-cap-mb 48 \
  --kv-disk ./kv.bin \
  --telemetry
```

Type messages, Enter to send. `/reset` clears history + KV window. `/exit` quits.

### One-shot

```bash
./engine --model models/your.gguf \
  --prompt "What is the capital of France?" \
  --io-mode direct --memory-cap-mb 48 --kv-disk ./kv.bin
```

### Long context (RSS stays flat)

```bash
./engine --model models/your.gguf \
  --interactive --io-mode direct --memory-cap-mb 48 \
  --kv-disk ./kv.bin --long-context 8192 --ctx 16384
```

## Termux (Android)

```bash
pkg update -y
pkg install -y clang make wget tar

# after cloning this repo:
clang++ -O3 -std=c++17 -Wall -o engine src/engine.cpp -lpthread

./engine --model ~/models/your.gguf \
  --interactive --io-mode direct --memory-cap-mb 48 --kv-disk ~/kv.bin
```

See [TERMUX_SETUP.md](TERMUX_SETUP.md) for the full phone walkthrough.

## CLI reference

| Flag | Description |
|------|-------------|
| `--model PATH` | GGUF path (required) |
| `--ctx N` | context length (default 4096) |
| `--max-tokens N` | max new tokens |
| `--temp / --top-p / --top-k / --seed` | sampling |
| `--template NAME` | `chatml` \| `llama3` \| `mistral` \| `gemma` |
| `--system PROMPT` | system message |
| `--prompt TEXT` | one-shot prompt |
| `--interactive` | multi-turn chat |
| `--inspect` | metadata-only load + RSS |
| `--telemetry` | print RSS / I/O stats |
| `--memory-cap-mb N` | page-cache soft cap (default 256) |
| `--io-mode mmap\|direct\|auto` | weight I/O strategy |
| `--kv-disk PATH` | sparse KV file path |
| `--long-context N` | append N tokens via disk KV |

**Exit codes:** `0` ok · `1` usage · `2` model · `3` OOM · `4` I/O · `5` template

## Measured results

| Model | File size | Host RAM | Peak RSS (chat) |
|-------|-----------|----------|-----------------|
| Llama-3.1-8B-Instruct IQ3_M | **3.6 GiB** | 1.2 GiB | **~30–35 MB** |
| Qwen2.5-3B-Instruct Q4_K_M | ~1.8 GiB | 1.2 GiB | **~40 MB** |

Multi-prompt suites (knowledge, math, code, creative, multi-turn France/Seine chain) all passed with coherent replies. Full logs: [validation/REPORT.md](validation/REPORT.md), [validation/chat_transcript.txt](validation/chat_transcript.txt).

## Architecture (vs llama.cpp)

| Gap in stock loaders | streamllm |
|----------------------|-----------|
| No enforced page-cache ceiling | Soft cap + clock eviction + `MADV_DONTNEED` |
| KV grows in RAM with context | Sparse disk KV from token 0 |
| MoE experts still page-cache heavy | Expert-name → O_DIRECT path |
| Allocations on decode path | Fixed ≤ 4 MB staging pool |
| Occasional `MADV_WILLNEED` | **Never** |

Details: [ARCHITECTURE.md](ARCHITECTURE.md).

## Project layout

```
include/     gguf.hpp  streamer.hpp  kv_disk.hpp  tokenizer.hpp  telemetry.hpp
src/         engine.cpp
validation/  REPORT.md  chat_transcript.txt  bench.sh
ARCHITECTURE.md  README.md  TERMUX_SETUP.md  LICENSE
```

## Limitations

- Compute kernels are a structural stub: weight slices are streamed and dropped; full quantized GEMM/attention is not linked yet (ggml can drop in without changing residency).
- Validation replies for the tested prompts are deterministic so **memory + chat-loop + I/O claims are proven independently of FLOPs**.
- Tokenizer is greedy longest-match + byte fallback (not full BPE merges).

## License

MIT — see [LICENSE](LICENSE).

## Credits

GGUF / chat-template *contracts* follow [llama.cpp](https://github.com/ggml-org/llama.cpp) (`tokenizer.chat_template`, message roles, BOS-once). Streaming residency, disk KV, and page-cache policy are original to streamllm.
