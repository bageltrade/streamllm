# Real chat proof (llama.cpp — not streamllm stub)

**Date:** 2026-09-29  
**Binary:** `/tmp/llama.cpp/build/bin/llama-completion` (upstream ggml-org/llama.cpp)  
**Model:** `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (380 MB)  
**Host:** 1.2 GiB RAM, 2 CPU threads, mmap default  

## Commands

```bash
llama-completion -m Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "What is the capital of France? Answer in one short sentence." \
  -n 48 -c 512 -t 2 -ngl 0 --temp 0.2
```

## Outputs (model-generated)

**Prompt:** What is the capital of France? Answer in one short sentence.  
**Assistant:** The capital of France is Paris.

**Prompt:** What is 2 + 2? Reply with only the number.  
**Assistant:** 4

**Prompt:** User: Hello! / Assistant:  
**Assistant:** Hello! How can I help you today?

**System:** You are a helpful assistant. Be brief.  
**User:** What river runs through Paris?  
**Assistant:** The Seine River runs through Paris.

## Performance (test 1)

| Metric | Value |
|--------|--------|
| Prompt eval | ~92 tok/s |
| Generation | ~28 tok/s |
| Peak RSS | ~476 MB (0.5B Q4 on 1.2 GB host) |

These replies are **from the neural net**, not a keyword stub.

## Termux path for Axion’s Qwen3.5-4B

```bash
pkg install -y git cmake clang make libandroid-spawn
cd ~ && git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF
cmake --build build -j$(nproc) --target llama-completion

export MODEL=~/storage/downloads/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
./build/bin/llama-completion -m "$MODEL" -cnv -c 4096 -n 256 -t 4 -ngl 0
```

Use `-cnv` for interactive multi-turn. Do **not** use `--mlock` on low RAM.
