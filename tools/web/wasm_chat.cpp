// WebAssembly wrapper around the cardputer_ai inference engine.
//
// This is the *same* main/llm.cpp the firmware runs — the Xtensa SIMD kernel is
// behind `#if defined(__XTENSA__)`, so a WASM build silently takes the scalar
// path, and tools/host/ stubs the ESP-IDF headers. So the browser demo is not a
// reimplementation: identical Q4 weights, identical byte-level BPE, identical
// sampler. What you get in a tab is what the device produces.
//
// Build: see tools/web/build.sh

#include "llm.h"

#include <emscripten/emscripten.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

namespace {
Transformer T;
Tokenizer   K;
Sampler     S;
bool        ready = false;
int         kv_len = 72;
std::string out_buf;
std::string info_buf;
}  // namespace

extern "C" {

// Takes ownership of nothing: the JS side keeps the buffers alive for the
// lifetime of the page (llm_init_embedded reads weights in place, exactly as
// the firmware reads them straight out of memory-mapped flash).
EMSCRIPTEN_KEEPALIVE
int chat_init(const uint8_t* model, int model_len,
              const uint8_t* tok, int tok_len, int kv) {
  kv_len = kv;
  if (!llm_init_embedded(&T, model, (size_t)model_len, kv)) return 0;
  if (!llm_tokenizer_from_memory(&K, tok, (size_t)tok_len, T.config.vocab_size)) return 0;
  kv_len = T.kv_seq_len;
  char b[256];
  snprintf(b, sizeof(b),
           "{\"arch\":%d,\"dim\":%d,\"hidden\":%d,\"layers\":%d,\"heads\":%d,"
           "\"vocab\":%d,\"seq\":%d,\"kv\":%d}",
           T.config.arch, T.config.dim, T.config.hidden_dim, T.config.n_layers,
           T.config.n_heads, T.config.vocab_size, T.config.seq_len, T.kv_seq_len);
  info_buf = b;
  ready = true;
  return 1;
}

EMSCRIPTEN_KEEPALIVE
const char* chat_info() { return info_buf.c_str(); }

// `prompt` uses "<|eos|>" to separate past exchanges, matching how main.cpp
// rebuilds multi-turn history with the EOS token between turns.
EMSCRIPTEN_KEEPALIVE
const char* chat_reply(const char* prompt, int max_new, float temp,
                       float top_p, unsigned seed) {
  out_buf.clear();
  if (!ready) return out_buf.c_str();

  llm_build_sampler(&S, T.config.vocab_size, temp, top_p, seed);

  std::string p(prompt);
  std::vector<int> toks(p.size() + 16);
  int n_prompt = 0;
  size_t start = 0;
  while (start <= p.size()) {
    size_t mark = p.find("<|eos|>", start);
    std::string seg = p.substr(start, mark == std::string::npos
                                          ? std::string::npos : mark - start);
    int m = 0;
    llm_encode(&K, seg.c_str(), 1, 0, toks.data() + n_prompt, &m);
    n_prompt += m;
    if (mark == std::string::npos) break;
    toks[n_prompt++] = K.eos_id;
    start = mark + 7;
  }
  if (n_prompt < 1) return out_buf.c_str();

  int token = toks[0];
  char scratch[64];
  int pos = 0, abspos = 0, tokens_out = 0;

  // Mirrors main.cpp stepGeneration(), including the sliding KV window: once
  // the cache fills, a quarter-window prefix stays pinned as an attention sink
  // and the oldest slots after it are evicted, so replies are not cut short.
  while (tokens_out < max_new) {
    if (abspos >= T.config.seq_len - 1) break;
    if (pos >= kv_len - 1) {
      int keep_head = kv_len / 4;
      if (n_prompt < keep_head) keep_head = n_prompt;
      int evict = kv_len / 4;
      if (evict < 1) evict = 1;
      evict = llm_kv_slide(&T, keep_head, evict);
      if (evict < 1) break;
      pos -= evict;
    }
    float* logits = llm_forward_at(&T, token, pos, abspos);
    int next;
    if (abspos < n_prompt - 1) {
      next = toks[abspos + 1];          // still replaying the prompt
    } else {
      next = llm_sample(&S, logits);
      if (next == K.eos_id) break;
      out_buf += llm_decode(&K, token, next, scratch, sizeof(scratch));
      tokens_out++;
    }
    token = next;
    pos++;
    abspos++;
  }
  return out_buf.c_str();
}

}  // extern "C"
