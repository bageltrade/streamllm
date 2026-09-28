# streamllm Architecture

## Memory model

```
┌─────────────────────────────────────────────────────────────┐
│  Process RSS  (hard goal ≤ 64 MB steady-state)              │
│  ┌──────────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐ │
│  │ GGUF meta    │ │ Tokenizer  │ │ Staging  │ │ Telemetry│ │
│  │ + tensor     │ │ vocab      │ │ pool     │ │ ring     │ │
│  │ descriptors  │ │ ~few–30 MB │ │ ≤ 8 MB   │ │ fixed    │ │
│  │ ~1 MB        │ │            │ │          │ │          │ │
│  └──────────────┘ └────────────┘ └──────────┘ └──────────┘ │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ DiskKV map structures (pointers, counters)  << 1 MB  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
          │ pread / mmap fault                  │
          ▼                                     ▼
┌─────────────────────┐              ┌──────────────────────┐
│ Model GGUF on NVMe  │              │ Sparse KV file       │
│ (weights never fully│              │ (K/V f16, windowed   │
│  resident)          │              │  MADV_DONTNEED)      │
└─────────────────────┘              └──────────────────────┘
```

**Invariant:** tensor *bytes* never live in process-owned heap except inside the reusable 8 MB staging pool. RSS must not scale with context length.

## I/O pipeline

1. **Open** — `mmap(MAP_SHARED)` of the GGUF (or pure `O_DIRECT`/`pread` when `--io-mode direct`). Immediately `madvise(MADV_RANDOM)`. Never `MADV_WILLNEED`.
2. **Parse** — header + KV metadata + tensor descriptors only. Validate magic, version, every tensor offset against file size.
3. **Per-layer / per-token**  
   - `read_tensor_slice` copies a byte range into a staging buffer (or caller buffer).  
   - mmap path: track 4 KiB pages in a clock-proximity list; when soft cap (default 256 MB) is exceeded, `madvise(MADV_DONTNEED)` on the oldest unreferenced pages.  
   - direct / expert path: `pread` into staging; no page-cache pollution.  
   - After the layer finishes → `drop_all()` (bulk DONTNEED).
4. **Double-buffering** — while layer L computes, layer L+1 can be issued asynchronously (io_uring or pread thread pool). Staging pool is split into two halves.

## Disk-backed KV

- Geometry: `bytes_per_token = 2 * n_layer * n_head_kv * head_dim * sizeof(f16)`.
- Sparse file sized for `--ctx`; untouched slots cost zero disk until written.
- Only a sliding window (default 2048 tokens) is kept advised-resident; older blocks receive `MADV_DONTNEED`.
- Attention reads K/V slices by pointer arithmetic into the mmap’d file — never materialises a full attention matrix in RAM.

## Eviction policy

- **Page cache (weights):** clock-proximity list. On insert, move to tail. Under pressure, walk from head; skip refcount > 0; DONTNEED and erase.
- **KV:** windowed. When `n_tokens > resident_window`, DONTNEED the oldest `block_tokens` (256) worth of bytes.

## Four gaps closed vs llama.cpp

| # | llama.cpp behaviour | streamllm |
|---|---------------------|-----------|
| 1 | No enforced page-cache ceiling | Soft cap + explicit clock eviction |
| 2 | KV allocated in RAM | Sparse disk file from the first token |
| 3 | `--lazy-experts` still uses page cache | Expert tensors forced through O_DIRECT |
| 4 | Allocations during decode | Fixed staging pool; zero hot-path `malloc` |

## Chat path (llama.cpp contracts)

- Template key: `tokenizer.chat_template` (exact).
- Message struct: `{ role, content }` with roles `system|user|assistant|tool`.
- Apply template → full history string; tokenize only the newly appended suffix (`prev_len` tracking).
- BOS only on the first turn.

## Telemetry

Pre-allocated ring of 256 samples. Each sample records RSS (`/proc/self/statm`), approximate page-cache, bytes read, hit/miss, evictions, KV evictions, tok/s, TTFT. Written with atomics; no allocation on the hot path. Dump on `--telemetry` or SIGUSR1.
