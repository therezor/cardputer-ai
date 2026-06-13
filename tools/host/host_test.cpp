// Host-side test driver for llm.cpp. Loads the converter's .bin files and runs
// encode → forward → sample exactly like the firmware does, printing to stdout.
//
// Build (from the sketch root):
//   clang++ -std=c++17 -O2 -I tools/host tools/host/host_test.cpp main/llm.cpp -o /tmp/llm_host
// Run:
//   /tmp/llm_host embed/model_neo_q4.bin embed/tok_neo.bin "Summary: ...\nStory:" [opts]
// Options: --max N   --temp F   --kv N   --top10 (print top-10 logits per step)
#include "../../main/llm.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

static std::vector<uint8_t> slurp(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> v(n);
  if (fread(v.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(1); }
  fclose(f);
  return v;
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s model.bin tok.bin prompt [--max N] [--temp F] [--kv N] [--top10]\n", argv[0]); return 1; }
  std::string prompt = argv[3];
  int   max_new = 80, kv = 96;
  float temp = 0.0f;
  bool  top10 = false;
  for (int i = 4; i < argc; i++) {
    if (!strcmp(argv[i], "--max"))  max_new = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--temp")) temp = atof(argv[++i]);
    else if (!strcmp(argv[i], "--kv"))   kv = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--top10")) top10 = true;
  }
  // Allow literal "\n" in the prompt argument.
  for (size_t p; (p = prompt.find("\\n")) != std::string::npos;)
    prompt.replace(p, 2, "\n");

  static std::vector<uint8_t> model, tok;
  model = slurp(argv[1]);
  tok   = slurp(argv[2]);

  static Transformer T;
  static Tokenizer   K;
  static Sampler     S;
  if (!llm_init_embedded(&T, model.data(), model.size(), kv)) {
    fprintf(stderr, "model init failed\n"); return 1;
  }
  if (!llm_tokenizer_from_memory(&K, tok.data(), tok.size(), T.config.vocab_size)) {
    fprintf(stderr, "tokenizer init failed\n"); return 1;
  }
  llm_build_sampler(&S, T.config.vocab_size, temp, 1234);
  fprintf(stderr, "arch=%d dim=%d hidden=%d layers=%d heads=%d vocab=%d seq=%d kv=%d\n",
          T.config.arch, T.config.dim, T.config.hidden_dim, T.config.n_layers,
          T.config.n_heads, T.config.vocab_size, T.config.seq_len, T.kv_seq_len);

  // "<|eos|>" markers split the prompt into segments joined by the EOS token
  // id — mirrors how the firmware's chat mode rebuilds multi-turn history.
  std::vector<int> toks(prompt.size() + 16);
  int n_prompt = 0;
  size_t start = 0;
  while (start <= prompt.size()) {
    size_t mark = prompt.find("<|eos|>", start);
    std::string seg = prompt.substr(start, mark == std::string::npos ? std::string::npos
                                                                     : mark - start);
    int m = 0;
    llm_encode(&K, seg.c_str(), 1, 0, toks.data() + n_prompt, &m);
    n_prompt += m;
    if (mark == std::string::npos) break;
    toks[n_prompt++] = K.eos_id;
    start = mark + 7;
  }
  fprintf(stderr, "prompt tokens (%d):", n_prompt);
  for (int i = 0; i < n_prompt; i++) fprintf(stderr, " %d", toks[i]);
  fprintf(stderr, "\n");

  int token = toks[0];
  char scratch[64];
  for (int pos = 0; pos < kv - 1 && pos < n_prompt - 1 + max_new; pos++) {
    float* logits = llm_forward(&T, token, pos);
    int next;
    if (pos < n_prompt - 1) {
      next = toks[pos + 1];
    } else {
      if (top10) {
        std::vector<int> idx(T.config.vocab_size);
        for (int i = 0; i < T.config.vocab_size; i++) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        fprintf(stderr, "pos %d top10:", pos);
        for (int i = 0; i < 10; i++) fprintf(stderr, " %d:%.3f", idx[i], logits[idx[i]]);
        fprintf(stderr, "\n");
      }
      next = llm_sample(&S, logits);
      if (next == K.eos_id || (K.style == ARCH_LLAMA && (next == 1 || next == 2))) break;
      printf("%s", llm_decode(&K, token, next, scratch, sizeof(scratch)));
      fflush(stdout);
    }
    token = next;
  }
  printf("\n");
  return 0;
}
