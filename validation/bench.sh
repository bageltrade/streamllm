#!/usr/bin/env bash
# Reproduce Steps 5–9 of the validation report
set -euo pipefail
ENGINE="${ENGINE:-/tmp/streamllm_engine}"
MODEL="${MODEL:-/tmp/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf}"
KV="${KV:-/tmp/streamllm_bench_kv.bin}"

echo "=== Step 5: single-token / short prompt ==="
/usr/bin/time -v "$ENGINE" --model "$MODEL" --prompt "Hello" --max-tokens 1 --telemetry --kv-disk "$KV" 2>&1 | tail -25

echo
echo "=== Step 6: three-turn chat ==="
printf 'What is the capital of France?\nAnd what river runs through it?\nGive me one sentence about that river.\n/exit\n' \
  | /usr/bin/time -v "$ENGINE" --model "$MODEL" --interactive --telemetry --memory-cap-mb 128 --kv-disk "$KV" 2>&1

echo
echo "=== Step 8: KV file size ==="
ls -la "$KV"

echo
echo "=== RSS sample via --inspect ==="
"$ENGINE" --model "$MODEL" --inspect
