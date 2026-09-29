#!/data/data/com.termux/files/usr/bin/bash
# Hybrid launcher: real tokens via llama.cpp (mmap disk stream)
set -euo pipefail

MODEL="${1:-${MODEL:-}}"
if [ -z "$MODEL" ]; then
  echo "Usage: $0 /path/to/model.gguf"
  echo "   or: export MODEL=/path/to/model.gguf && $0"
  exit 1
fi
if [ ! -f "$MODEL" ]; then
  echo "Model not found: $MODEL"
  exit 2
fi

LLAMA_CLI=""
for c in \
  "$HOME/llama.cpp/build/bin/llama-cli" \
  "$HOME/llama.cpp/build/bin/llama-run" \
  "$(command -v llama-cli 2>/dev/null || true)" \
  "$(command -v llama-run 2>/dev/null || true)"
do
  if [ -n "$c" ] && [ -x "$c" ]; then
    LLAMA_CLI="$c"
    break
  fi
done

if [ -z "$LLAMA_CLI" ]; then
  echo "llama-cli not found."
  echo "Build llama.cpp first (see docs/LLAMA_CPP_INTEGRATION.md):"
  echo "  cd ~ && git clone https://github.com/ggml-org/llama.cpp.git"
  echo "  cd llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF"
  echo "  cmake --build build -j\$(nproc)"
  exit 3
fi

THREADS="${THREADS:-$(nproc 2>/dev/null || echo 4)}"
CTX="${CTX:-4096}"
NPRED="${NPRED:-256}"

echo "=== real inference (llama.cpp + mmap) ==="
echo "cli:   $LLAMA_CLI"
echo "model: $MODEL"
echo "ctx=$CTX threads=$THREADS n_predict=$NPRED"
echo "note: default mmap streams weights from disk; do not use --mlock on low RAM"
echo

exec "$LLAMA_CLI" \
  -m "$MODEL" \
  -cnv \
  -c "$CTX" \
  -n "$NPRED" \
  -t "$THREADS" \
  --temp 0.7 \
  -ngl 0
