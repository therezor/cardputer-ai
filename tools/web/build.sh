#!/bin/bash
# Build the browser demo: main/llm.cpp compiled to WebAssembly.
#
# The engine is portable as-is — the Xtensa SIMD kernel is guarded by
# `#if defined(__XTENSA__)` and tools/host/ stubs the ESP-IDF headers — so this
# runs the identical Q4 weights, BPE and sampler the Cardputer runs.
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/tools/web/dist"
mkdir -p "$OUT"

source ~/emsdk/emsdk_env.sh > /dev/null 2>&1

em++ -std=c++17 -O3 \
  -I "$ROOT/tools/host" -I "$ROOT/main" \
  "$ROOT/tools/web/wasm_chat.cpp" "$ROOT/main/llm.cpp" \
  -s WASM=1 \
  -s MODULARIZE=1 \
  -s EXPORT_NAME=createLLM \
  -s EXPORTED_FUNCTIONS='["_chat_init","_chat_reply","_chat_info","_malloc","_free"]' \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPU8","UTF8ToString","stringToUTF8","lengthBytesUTF8"]' \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=64MB \
  -s SINGLE_FILE=1 \
  -o "$OUT/llm.js"

echo "[+] built $OUT/llm.js ($(du -h "$OUT/llm.js" | cut -f1))"
