# Cardputer AI — a tiny local chatbot for the ESP32 Cardputer ADV

A fully local, offline chatbot running on the **M5Stack Cardputer ADV**
(ESP32-S3FN8, 512 KB SRAM, 8 MB flash, no PSRAM). It also runs on the
**original M5Stack Cardputer** — the firmware ships both keyboard drivers
(the ADV's TCA8418 and the original's IO matrix) and picks the right one at
boot. No Wi-Fi, no API, no SD card — the LLM lives in the firmware and runs
on the microcontroller itself. It makes small talk with multi-turn memory
and writes stories on request, at ~7 tok/s.

The model is [roneneldan/TinyStories-Instruct-3M][hf-neo] (GPT-Neo) fine-tuned
on simple-English dialogues (filtered [allenai/SODA][soda] +
[DailyDialog][dd], formatted `User: ...\nBot: ...<|endoftext|>`, loss masked
to bot replies) mixed with 30% story data so the story skill survives, plus
two small hand-templated skills: kindergarten Q&A it *can* answer (colors,
animal sounds, opposites — `tools/qa_facts.py`) and graceful "I don't know"
replies for questions beyond a 3M-param model (paired with [SciQ][sciq]
questions). The fine-tune runs in about an hour on an Apple-Silicon Mac via
`tools/finetune_chat.py`.

[soda]: https://huggingface.co/datasets/allenai/soda
[dd]: https://huggingface.co/datasets/li2017dailydialog/daily_dialog
[sciq]: https://huggingface.co/datasets/allenai/sciq

The engine also still runs the original [Maykeye/TinyLLama-v0][hf-llama]
completion model — the embedded model's header selects the architecture
(LLaMA vs GPT-Neo) at boot. Swap models by re-running the matching converter.

Weights are quantized to **Q4_0** and embedded into the firmware binary, so
flashing through [bmorcelli/Launcher][lc] installs everything in one step —
no SD card, no model partition flashing, no setup.

[hf-neo]: https://huggingface.co/roneneldan/TinyStories-Instruct-3M
[hf-llama]: https://huggingface.co/Maykeye/TinyLLama-v0
[lc]: https://github.com/bmorcelli/Launcher

## Modes

- **chat** (default): turn-taking small talk. The firmware rebuilds the
  training format every turn — recent exchanges joined by EOS tokens — and
  trims the oldest exchanges to fit the prompt budget. `/new` resets the
  conversation. Replies end when the model emits EOS (it learned to stop).
- **story**: wraps your text as `Summary: <text>\nStory:` — type what the
  story should be about, get that story.
- **raw**: plain completion, no wrapper (works with the old LLaMA model too).

Honest limits: kindergarten English and an 80-token context (~3 short
exchanges of memory). It can answer kindergarten facts it was trained on
(colors, animal sounds, opposites) and is trained to say "I don't know" to
questions beyond it, instead of confabulating — mostly. It's still a
3M-param model; expect charming nonsense at the edges.

## Ukrainian model (branch `ukr`)

The `ukr` branch talks Ukrainian instead of English. This is **not** a
fine-tune of the English model, and it cannot be: byte-level BPE over the
pruned GPT-2 vocabulary spells Cyrillic out at roughly two tokens per
*character*, so a Ukrainian tokenizer is mandatory — and a new tokenizer means
a new embedding table, which is a third of a 10M-param model. So the model is
retrained on Ukrainian and then fine-tuned on dialogue. The *embedding table*
starts random — nothing of the English vocabulary survives — but the
transformer body is warm-started from TinyStories-Instruct-8M (107 tensors:
all 8 blocks, `ln_f`, and the first 256 learned positions), which converges
measurably faster than a cold start at this budget.

What did **not** have to change is the engine. `main/llm.cpp` does exact BPE by
merging the adjacent pair whose result has the lowest id, which is correct as
long as token-id order equals merge-rank order — a property a freshly trained
`ByteLevelBPETokenizer` has by construction. So the Q4 model blob and the
tokenizer blob swap in with no change to the inference code.

```sh
# 1. corpus: OPUS OpenSubtitles (uk) + Tatoeba (uk)
mkdir -p data/ukr/raw && cd data/ukr/raw
curl -L -o opensubtitles_uk.txt.gz \
    https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono/uk.txt.gz
curl -L -o tatoeba_uk.txt.gz \
    https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/mono/uk.txt.gz
gzcat opensubtitles_uk.txt.gz > opensubtitles_uk.txt
gzcat tatoeba_uk.txt.gz > tatoeba_uk.txt && cd -

../cardputer_ai_venv/bin/python tools/prepare_ukr_data.py       # ~8 min
../cardputer_ai_venv/bin/python tools/train_ukr_tokenizer.py    # ~3 min
../cardputer_ai_venv/bin/python tools/pretrain_ukr.py \
    --init scratch --minutes 180 --out-dir data/ukr/pretrained
../cardputer_ai_venv/bin/python tools/finetune_chat.py \
    --base-dir data/ukr/pretrained --data-dir data/ukr \
    --epochs 1 --out-dir data/ukr/chat_model
../cardputer_ai_venv/bin/python tools/convert_tinystories_instruct.py \
    --model-dir data/ukr/chat_model --corpus data/ukr/chat_train.txt \
    --min-count 1 --keep-bin
```

Subtitles are the right register almost for free: 15M lines averaging 4.3
words each, spoken and simple, and two consecutive lines are usually an
exchange — which is where the `User:`/`Bot:` pairs come from. The filter drops
Russian lines (ы/э/ъ/ё never appear in Ukrainian), sound effects, credits and
anything with digits or Latin letters. The train/val split is taken as whole
contiguous chunks of the source file, because pairs are built from *adjacent*
lines and a random split would leak one half of a training pair into val.

### Why there is a prose tier

Subtitles alone will not teach Ukrainian morphology. At 4.3 words per line,
most of the corpus is fragments — `Добре.`, `Звичайно.`, `Так.` — which carry
no agreement relation at all. But Ukrainian morphology *is* agreement:
adjective-noun concord in case, gender and number; past-tense verbs agreeing
with the subject's gender; prepositions governing case. Those relations live in
complete clauses, so the pretraining tier mixes in ~30% edited prose from
[lang-uk/malyuk][malyuk] — full sentences with subordinate clauses. Register is
recovered by the second stage: pretrain on the mix, fine-tune on dialogue only.

(Note that malyuk documents are hard-wrapped mid-sentence, so single newlines
are line breaks and not sentence boundaries — the prep joins them before
splitting, and requires a capital start, or every wrap leaks in as a
mid-sentence fragment and teaches broken syntax.)

[malyuk]: https://huggingface.co/datasets/lang-uk/malyuk

### Measuring morphology

Loss will not tell you whether the output is grammatical — a small model can be
fluent and still wrong in exactly the places Ukrainian marks. `tools/eval_morphology.py`
scores two things against [VESUM][vesum], the reference Ukrainian morphological
dictionary (3.9M word forms):

- **wordform validity** — what fraction of generated words actually exist.
  Catches invented morphology, where a plausible stem gets a plausible ending
  and the result is not a word.
- **adj-noun agreement** — of adjective+noun bigrams where both forms are
  known, how many share a case, a number and (in singular) a gender.

Proper nouns are excluded from validity: VESUM has no transliterated foreign
names, and 70% of out-of-dictionary words in the subtitle corpus are exactly
those. For the same reason VESUM is used only as a *metric* — filtering the
corpus to in-dictionary lines would systematically delete dialogue containing
names.

```sh
../cardputer_ai_venv/bin/python tools/eval_morphology.py \
    --model embed/model_neo_q4.bin --tok embed/tok_neo.bin
```

[vesum]: https://github.com/brown-uk/dict_uk

Scoring uses the same harness as the English model, with a Ukrainian battery:

```sh
../cardputer_ai_venv/bin/python tools/eval_battery.py --lang ukr \
    --prompts tools/eval_prompts_ukr.txt \
    --model ukr=embed/model_neo_q4.bin,embed/tok_neo.bin
```

### Typing Ukrainian on a Latin keyboard

The Cardputer's keycaps are Latin, so `main/translit.cpp` transliterates
phonetically as you type — the raw keystrokes stay the source of truth and the
whole buffer is re-transliterated on every key, which is what makes digraphs
work (`s` shows `с`, then `h` turns the pair into `ш`).

| type | get | type | get | type | get |
|------|-----|------|-----|------|-----|
| `zh` | ж | `ya` | я | `q`  | ь |
| `ch` | ч | `yu` | ю | `w`  | ш |
| `sh` | ш | `ye` | є | `x`  | х |
| `shch` | щ | `yi`/`ji` | ї | `c` | ц |
| `kh` | х | `yo` | йо | `gg` | ґ |

`h` and `g` both give **г** (what people actually type); the rare **ґ** is
`gg`. **ї** is `yi`, so Київ is typed `Kyjiv` — plain `Kyiv` reads its `yi` as
the digraph and comes out `Кїв`. Lines starting with `/` stay Latin so `/new`
still works.

Display needed a new font too: m5gfx's bundled efont has the Russian Cyrillic
letters but omits **Ґ Є І Ї ґ є і ї**, which puts holes in ordinary Ukrainian
words. `main/ukr_font.c` vendors GNU Unifont's Cyrillic subset via u8g2 (6.6 KB,
full U+0400–U+052F). Since Cyrillic is two bytes per character in UTF-8, the UI
paths that used to step by byte — backspace, input-box scrolling, long-word
wrapping — now step by character.

## Fine-tune pipeline (rebuild the chat model from scratch)

```sh
../cardputer_ai_venv/bin/python tools/prepare_chat_data.py       # ~15 min
../cardputer_ai_venv/bin/python tools/finetune_chat.py --epochs 2 \
    --warmup 400 --out-dir data/chat_model_v2                    # ~1 h MPS
../cardputer_ai_venv/bin/python tools/convert_tinystories_instruct.py \
    --model-dir data/chat_model_v2 --corpus data/chat_train.txt \
    --min-count 8 --keep-bin
```

The data prep keeps the longest *window* of simple turns per SODA dialogue
(not just the prefix) and renames speakers to TinyStories-frequent names —
together those roughly double the usable-dialogue yield vs the old prefix
filter. Loss is masked to bot replies and story bodies (the model never
trains on producing user turns), and chat samples are tokenized
segment-by-segment in exactly the shapes the firmware feeds at inference.

To compare a new fine-tune against the previous one before flashing:

```sh
# masked val loss on the frozen val set (predates the current corpus)
../cardputer_ai_venv/bin/python tools/eval_chat.py \
    --model-dir data/chat_model_masked --model-dir data/chat_model_v2
# fixed prompt battery through the host harness (facts / IDK / over-refusal)
../cardputer_ai_venv/bin/python tools/eval_battery.py \
    --model old=path/to/old/model_neo_q4.bin,path/to/old/tok_neo.bin \
    --model new=embed/model_neo_q4.bin,embed/tok_neo.bin
```

NOTE: keep the venv (and anything containing PyTorch) outside the repo so
build tooling never walks into torch's headers.

## What's in the box

```
cardputer_ai/
├── platformio.ini             PlatformIO build (espidf framework, IDF 5.5)
├── CMakeLists.txt             plain ESP-IDF build (`idf.py build`) works too
├── sdkconfig.defaults         flash/CPU/WDT config for the Cardputer ADV
├── partitions.csv             6 MB factory app slot for app + embedded model
├── main/                      the ESP-IDF "main" component
│   ├── main.cpp               boot + chat loop + story-mode prompt wrapper
│   ├── llm.{h,cpp}            Q4_0 engine: LLaMA + GPT-Neo forward paths,
│   │                          dual-core matmul, exact byte-level BPE
│   ├── ui.{h,cpp}             chat UI
│   ├── keyboard/              Cardputer keyboard driver, ported to ESP-IDF
│   │                          from m5stack/M5Cardputer v1.1.1 (MIT)
│   ├── model_data.cpp         generated — Q4 model bytes (~1.9 MB for 3M)
│   └── tok_data.cpp           generated — pruned GPT-2 tokenizer (~190 KB)
└── tools/
    ├── prepare_chat_data.py              SODA/DailyDialog/QA/IDK → corpus
    ├── qa_facts.py                       hand-written kindergarten Q&A + deflections
    ├── finetune_chat.py                  masked-loss fine-tune (MPS/CPU)
    ├── eval_chat.py                      masked val loss, old vs new checkpoint
    ├── eval_battery.py / eval_prompts.txt  fixed prompt battery, scored
    ├── convert_tinystories_instruct.py   HF GPT-Neo → Q4_0 + pruned vocab
    ├── convert_tinyllama_v0.py           the old TinyLLama-v0 converter
    └── host/                             macOS/Linux test harness for llm.cpp
```

The display (and board autodetect, power, etc.) comes from the
`m5stack/m5unified` + `m5stack/m5gfx` managed components, resolved from the
ESP Component Registry on first build (`main/idf_component.yml`).

## One-time setup

```sh
python3 -m venv ../cardputer_ai_venv   # keep the venv outside the repo
../cardputer_ai_venv/bin/pip install huggingface_hub tokenizers torch numpy datasets transformers
../cardputer_ai_venv/bin/python tools/convert_tinystories_instruct.py            # base 3M model
../cardputer_ai_venv/bin/python tools/convert_tinystories_instruct.py --model 8M # bigger, ~5MB
```

The converter downloads the model, prunes the 50,257-token GPT-2 vocab to the
~12K tokens the TinyStories dataset actually uses (closed under BPE merge
derivation, so encoding stays exact), quantizes to Q4_0, and writes
**`main/model_data.cpp`** / **`main/tok_data.cpp`**.

Why prune? The full GPT-2 embedding table would be 13M params — bigger than
the 3M transformer itself — and a 50K-logit buffer (196 KB) doesn't fit our
heap. Pruned: ~1.9 MB total model, ~50 KB logits.

## Build & flash

PlatformIO (the `espidf` framework via the [pioarduino] platform, which ships
ESP-IDF v5.5 — the official `espressif32` platform stopped at 5.4):

```sh
pio run                # build → .pio/build/cardputer/firmware.bin
pio run -t upload      # flash over USB
pio device monitor     # serial logs
```

One image covers both the original Cardputer and the ADV: they share the
M5Stamp-S3 module, the keyboard driver is chosen at runtime from
`M5.getBoard()`, and M5Unified autodetects the display. Flash the same
`firmware.bin` to either board.

The same tree is a standard ESP-IDF project, so this works too:

```sh
idf.py build flash monitor
```

Settings that used to be Arduino IDE menu choices (8 MB flash, DIO/80 MHz, no
PSRAM, custom partition table, 240 MHz CPU) live in `sdkconfig.defaults`.

For [bmorcelli/Launcher][lc] installs, ship
`.pio/build/cardputer/firmware.bin` — the partition table in this repo
keeps the factory app at offset 0x10000, which is where Launcher writes it.
(`firmware.factory.bin` in the same directory is the full-flash image:
bootloader + partition table + app, for esptool at offset 0x0.)

**Upgrading from a pre-8M build**: the partition table changed (the unused
SPIFFS partition was removed to make room for the 8M model) and the flash
mode changed DIO→QIO. The flash mode lives in the *bootloader* header, which
Launcher does not rewrite — so for full speed, flash the complete image once
over USB:

```sh
pio run -t upload   # or: esptool.py write_flash 0x0 firmware.factory.bin
```

If boot logs show ~13 MB/s flash bandwidth instead of ~25, the bootloader is
still DIO.

[pioarduino]: https://github.com/pioarduino/platform-espressif32

## Host testing (no hardware needed)

`tools/host/` stubs the ESP32 APIs so the exact engine code runs on your Mac:

```sh
../cardputer_ai_venv/bin/python tools/convert_tinystories_instruct.py --keep-bin --no-cpp
clang++ -std=c++17 -O2 -I tools/host tools/host/host_test.cpp main/llm.cpp -o /tmp/llm_host
/tmp/llm_host embed/model_neo_q4.bin embed/tok_neo.bin \
  "Summary: a girl finds a lost cat.\nStory:" --temp 0.8 --top-p 0.9 --kv 80
```

## Memory budget (Cardputer ADV, ~280 KB free heap)

| Buffer (8M model, dim=256)           | Bytes    |
|--------------------------------------|----------|
| KV cache, ctx=72, int4 + group scales| ~166 KB  |
| Logits (vocab=12929)                 | ~50 KB   |
| Activations + attention scores       | ~19 KB   |
| FreeRTOS + matmul worker stack       | 5 KB     |

(The 3M model at dim=128 uses an 80-token window and only ~90 KB of KV.)
The KV window is picked from the model header at boot (`kvLenForModel` in
`main/main.cpp`). The converter enforces `--max-vocab` (default 15,500) so
a corpus change can't silently blow the logits budget.

The GPT-Neo KV cache is stored as **int8 with one fp32 scale per row** (the
LLaMA path keeps bf16) — at dim=128 that's 2 KB/position, which is what makes
an 80-token window fit. `KV_SEQ_LEN` lives in `main/main.cpp`; the model
ships 256 position embeddings, so RAM is the binding constraint, not flash.

## Engine notes

- GPT-Neo forward path: LayerNorm (+bias), learned position embeddings,
  GELU MLP, and — faithful to the original — **no 1/sqrt(d) attention
  scaling**. The alternating "local attention" layers have a 256-token
  window ≥ our context, so they degenerate to global causal attention.
- **ESP32-S3 PIE SIMD matmul** (`main/dot_q4_pie.S`): activations are
  quantized to int8 once per matmul (llama.cpp's Q4_0×Q8_0 scheme), then
  each 32-weight block is 2× `EE.VMULAS.S8.ACCX` — 16 int8 MACs per
  instruction. The CRDP v3 blob stores weights row-planar
  ([bf16 scales | pad | 16-byte nibble groups]) so every vector load is
  16-byte aligned. A boot-time selftest compares the SIMD kernel against
  the scalar reference on real weight rows and refuses to run on mismatch.
  Host builds (and `-DLLM_FORCE_SCALAR`) use the scalar path.
- **KV cache is int4** (nibbles + one bf16 scale per 32-element group,
  Q4_0-style asymmetric). Each head lies inside one scale group, so
  attention applies a single scale per (head, position). Measured on the
  eval battery this matches int8-KV quality — and it's what fits the 8M
  model's 72-token window in SRAM.
- Weights stream from MMU-mapped flash every token, so **flash bandwidth is
  the throughput ceiling** for the 8M model — the build uses QIO + 64-byte
  cache lines, and boot logs the measured flash bandwidth over serial.
- Tokenizer is **exact** GPT-2 byte-level BPE: the blob embeds the merge
  pair table (binary-searched from flash); verified 0 mismatches vs
  HuggingFace on 500 dataset lines.
- Q4 logits match the fp32 HF reference closely (top-3 identical on test
  prompts); divergence in long greedy decodes is normal quantization noise.
- Sampler: greedy at T=0; otherwise softmax + **top-p (nucleus)** sampling
  over the 64 most likely tokens (single pass, no full-vocab sort — a
  min-tracked candidate array costs ~1 compare per vocab entry). Top-p 1.0
  falls back to the exact full-vocab multinomial. Temperature and top-p are
  live in the settings screen.

## What's not great yet

- 80 tokens of context ≈ 3 short exchanges of conversational memory; older
  turns silently fall out of the prompt. Within a single reply the window
  now *slides* rather than stopping, so replies can run past it — but
  sliding evicts old context to make room, so a long reply trades memory
  for length.
- Reply length tops out around 250 tokens of *useful* output. That is the
  model, not the firmware: TinyStories-Instruct-8M was fine-tuned at
  `--seq-len 256`, and generating past that (even in PyTorch with full
  attention and no sliding window) degenerates into looping `Summary:`
  fragments. The converter's 256 position embeddings sit right at that
  limit deliberately — exporting more buys tokens the model can't use.
- Only kindergarten facts. Everything else gets a (trained) "I don't know" —
  for real factual Q&A you'd need Wi-Fi + an API, or different hardware.
- ~5 tok/s measured on device for the 8M model (196 ms/token; PIE SIMD +
  QIO + 64-byte cache lines). The ceiling is flash streaming (~30 MB/s,
  ~5.6 MB read per token); the next levers are a harder vocab prune, 120 MHz
  flash (experimental HPM), or batched/speculative decoding.
- Chat quality is bounded by 8M params — noticeably better grammar and
  context-tracking than 3M, still no real-world knowledge.

## Get the model (without a Cardputer)

The chat fine-tune is published as **TinyTalk 2**:

- [TheREZOR/TinyTalk-2](https://huggingface.co/TheREZOR/TinyTalk-2) —
  safetensors (transformers, GPT-Neo), with embedded chat template
- [TheREZOR/TinyTalk-2-GGUF](https://huggingface.co/TheREZOR/TinyTalk-2-GGUF) —
  GGUF for llama.cpp / Ollama: `ollama run hf.co/TheREZOR/TinyTalk-2-GGUF`
  (GGUF uses a mathematically-exact GPT-2 conversion, `tools/export_gpt2.py`,
  since llama.cpp doesn't support plain GPT-Neo)
- [TheREZOR/TinyTalk](https://huggingface.co/TheREZOR/TinyTalk) — v1 (3M)

The Ukrainian model is **TinyTalk UA** — try it in a browser at
[tinytalk-ua.rezor.me](https://tinytalk-ua.rezor.me/):

- [TheREZOR/TinyTalk-UA](https://huggingface.co/TheREZOR/TinyTalk-UA) —
  safetensors, plus the firmware's Q4 blobs under `esp32/`
- [TheREZOR/TinyTalk-UA-GGUF](https://huggingface.co/TheREZOR/TinyTalk-UA-GGUF) —
  `ollama run hf.co/TheREZOR/TinyTalk-UA-GGUF`

## Changelog

- **v3.0-ukr** — the bot speaks Ukrainian (branch `ukr`)
  - **Model retrained on Ukrainian**, not fine-tuned: a Ukrainian tokenizer
    replaces the embedding table, so nothing of the English vocabulary
    survives (the transformer body is warm-started from TinyStories-Instruct-8M).
    14K byte-level BPE, 9.96M params, 700M tokens over filtered
    OpenSubtitles-uk + ~30% edited prose from lang-uk/malyuk, then a masked-loss
    dialogue fine-tune (3 epochs). Chat val 3.454.
  - **No engine change was needed.** `ctk2_encode` only requires token-id order
    to equal merge-rank order, which a freshly trained ByteLevelBPE satisfies by
    construction — verified 40/40 against HF on Ukrainian text.
  - **Scores:** facts 8/8, "не знаю" deflection 8/8, no-over-refusal 8/8;
    wordform validity **99.5%** and adjective-noun agreement **90.6%** against
    VESUM (`tools/eval_morphology.py`).
  - **Cyrillic font + phonetic input.** m5gfx's efont covers Russian but omits
    Ґ Є І Ї ґ є і ї, so GNU Unifont's Cyrillic subset is vendored (6.6 KB).
    `main/translit.cpp` transliterates Latin keystrokes as you type, and the UI
    paths that stepped byte-by-byte now step by UTF-8 character.
  - Firmware 6.66 MB (79.4% of flash).

- **v2.1** — sliding context window
  - **Replies no longer stop at the KV window.** When the cache fills mid-reply
    the window slides: a quarter-window prefix stays pinned as an attention
    sink, the oldest slots after it are evicted, and generation continues. The
    physical KV slot and the absolute sequence position are now tracked
    separately (`llm_forward_at` / `llm_kv_slide`), since GPT-Neo bakes its
    learned position into the cached keys and they can't be re-rotated.
  - Fixes a bug where a prompt of ≥55 tokens (routine in chat mode, which packs
    history to a 64-token budget) would rewind into prompt replay after a slide
    and silently re-inject its own tail mid-reply, forever.
  - Where the eviction band lands matters: pinning half the window pinned the
    *oldest* history and cut straight through the question being answered. The
    quarter-window pin drops stale history instead, so the live turn survives
    the first slide — a reply long enough to slide twice still loses it.
  - Generation now stops cleanly when the model's 256 position embeddings run
    out, instead of reusing the last row and degenerating.
  - Reply length is decoupled from the KV window: the setting goes up to 256
    (was 64) with coarse steps, plus **until eos (slides)** below 4 — which is
    now the default, so replies run until the model stops rather than being
    cut at 44 tokens. The old **unsafe** mode is gone; sliding replaces it and
    doesn't corrupt the cache.
  - `tools/host/host_test.cpp` mirrors the firmware's generation loop
    (`--slide`, `--sink N`, `--old-policy`, `--replay-by-pos`) and reports
    which prompt tokens each slide drops, so the policy is testable on a host.
- **v2.0** — the **TinyTalk 2** release
  ([TinyTalk 1 on HuggingFace](https://huggingface.co/TheREZOR/TinyTalk))
  - **8M model** (dim=256): same chat fine-tune recipe on
    TinyStories-Instruct-8M — noticeably better language quality than 3M
    (frozen-val loss 1.49 vs 1.80). **~5 tok/s measured on device**
    (196 ms/token), with boot-time serial benchmarks for flash bandwidth
    and ms/token.
  - **ESP32-S3 PIE SIMD** Q4×Q8 matmul kernel + QIO flash + 64-byte cache
    lines to keep it fast; CRDP v3 row-planar blob format (16B-aligned).
  - **int4 KV cache** (group-32 bf16 scales) — halves KV memory; the 8M
    model runs a 72-token window, the 3M keeps 80.
  - Stale v2 `model_data.cpp` blobs are rejected at boot instead of
    producing gibberish; SIMD kernel self-tests against the scalar path.
- **v1.2**
  - Smarter model, same speed: retrained on a ~2x larger corpus — SODA
    window filtering (85% yield vs 47%) with TinyStories speaker renaming,
    plus DailyDialog. Frozen-val loss 1.84 → 1.80; kindergarten-fact
    battery 1/8 → 6/8; "I don't know" on impossible questions 6/8 → 8/8.
  - New skills: answers simple kindergarten Q&A (colors, animal sounds,
    opposites, baby animals); deflects questions it can't know instead of
    confabulating (trained on SciQ questions + hand-written deflections).
  - **Top-p (nucleus) sampling**, default 0.9, adjustable in settings
    (1.00 = off). Cheap single-pass implementation, no full-vocab sort.
  - Eval tooling: `tools/eval_chat.py` (masked val loss) and
    `tools/eval_battery.py` (scored prompt battery via the host harness).
- **v1.1**
  - Press the backtick (`` ` ``) key to stop a reply while it's being typed out.
  - Two new reply-length options below the normal range: **unlimited** (keeps
    going until the model decides to stop) and **unsafe** (lets longer replies
    keep going by reusing memory, clearing the chat when it runs out).
- **v1.0** — initial release.

## License

Code: MIT (see LICENSE). The embedded model derives from
TinyStories-Instruct-3M and the SODA (CC BY 4.0), DailyDialog
(CC BY-NC-SA 4.0) and SciQ (CC BY-NC 3.0) datasets — the latter two are
**non-commercial**; see NOTICE.md for full third-party attributions.
