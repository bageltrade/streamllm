# Real answers = llama.cpp + mmap disk streaming

streamllm proved the **memory model** (RSS in tens of MB while touching multi-GB GGUFs).
**Actual token generation** comes from [llama.cpp](https://github.com/ggml-org/llama.cpp), which already:

- memory-maps GGUF by default (`mmap` load mode)
- runs real quantized matmul + attention + sampling
- supports chat templates and interactive conversation mode

You do **not** need the stub responder in streamllm for chat quality.
You need **llama.cpp built on Termux**, then the flags below so weights stream from disk.

## Why this is the correct link

| Piece | Who does it |
|--------|-------------|
| Disk-backed weights (mmap, OS page cache) | llama.cpp default |
| Real logits / tokens | llama.cpp |
| Chat template | llama.cpp (`-cnv`) |
| Extra soft page-cache policy experiments | streamllm (research) |

Rewriting ggml inside streamllm from scratch is not the fast path.
Using llama.cpp **is** linking streaming disk + real reply.

## Termux: install llama.cpp and run your Qwen GGUF

```bash
pkg update -y
pkg install -y git cmake clang make libandroid-spawn

cd ~
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF
cmake --build build --config Release -j$(nproc 2>/dev/null || echo 4)

ls build/bin/llama-cli
```

Point at **your** model (edit path):

```bash
export MODEL=~/storage/downloads/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
```

### Interactive chat (mmap = disk stream; do not mlock on low RAM)

```bash
~/llama.cpp/build/bin/llama-cli \
  -m "$MODEL" \
  -cnv \
  -c 4096 \
  -n 256 \
  -t $(nproc 2>/dev/null || echo 4) \
  --temp 0.7 \
  -ngl 0
```

- `-cnv` = conversation mode (real multi-turn)
- default load uses **mmap** → weights fault in from disk
- **do not** pass `--mlock` on a low-RAM phone
- `-ngl 0` = CPU only

### One-shot

```bash
~/llama.cpp/build/bin/llama-cli \
  -m "$MODEL" \
  -p "What is the capital of France?" \
  -n 64 -c 2048 -t 4 -ngl 0
```

### If the phone OOMs

1. Lower context: `-c 2048` or `-c 1024`
2. Fewer threads: `-t 2`
3. Smaller quant if needed
4. Keep model on internal storage; close other apps

## streamllm helper

```bash
cd ~/streamllm && git pull
./scripts/chat_real.sh "$MODEL"
```

That script finds `llama-cli` and runs conversation mode with safe flags.

## References

- llama.cpp CLI load modes: mmap streams weights; avoid mlock when RAM < working set
- Termux: official android docs in the llama.cpp repo + cmake build above
