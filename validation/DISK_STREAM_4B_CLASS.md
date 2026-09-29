# Disk-stream class test (llama.cpp mmap + real tokens)

**Host:** 1.2 GiB RAM · no swap · 2 CPUs  
**Engine:** upstream `llama-completion` (`--load-mode mmap --fit off`)

## Model A — Qwen2.5-3B-Instruct-Q4_K_M (1.84 GiB file)

| | |
|--|--|
| File size | **1.84 GiB > 1.2 GiB RAM** |
| Result | **SIGKILL / OOM** during load (`common_fit_params` / peak RSS ~1.01 GiB then killed) |
| Conclusion | On this host, **3B Q4 does not fit** even with mmap + fit off. Need more RAM or smaller quant / smaller model. |

## Model B — Qwen2.5-1.5B-Instruct-Q4_K_M (941 MiB file)

| | |
|--|--|
| File size | 941 MiB |
| Peak process RSS | **~995 MiB** (`Maximum resident set size: 1018496 kB`) |
| Load mode | `mmap` |
| Real answers | **Yes** |

### Replies (neural net, not stub)

- *What is the capital of France?* → **The capital of France is Paris.**
- *What is 2 + 2?* → **4**
- *What river runs through Paris?* → **The Seine river runs through Paris.**

### Throughput (1.5B, cold-ish)

- Prompt eval ~28 tok/s  
- Generation ~3.2 tok/s on 2 threads (I/O + CPU limited)

## What “disk streaming” means here

llama.cpp **mmap** maps the GGUF; the OS pages weights in on demand.  
Process RSS still grows toward the working set (weights touched + KV + activations).  
It is **not** the ultra-low tens-of-MB RSS of the streamllm research stub; it **is** real inference that can run when the working set fits in RAM+page-cache pressure allows.

## Termux note for Axion’s ~4B Q4_K_M

Your `Qwen3.5-4B-...-Q4_K_M.gguf` is typically **~2.5–2.8 GiB**.  
You need roughly **that much free RAM + headroom** for KV, or a smaller quant (Q3/IQ3), or expect heavy swap/OOM on a 4–6 GB phone under load.

```bash
~/llama.cpp/build/bin/llama-completion \
  -m "$MODEL" --load-mode mmap --fit off \
  -cnv -c 2048 -n 128 -t 4 -ngl 0
```

If OOM: lower `-c`, use fewer threads, or a smaller quant.
