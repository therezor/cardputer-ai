# Cardputer AI — a tiny local chatbot for the ESP32 Cardputer ADV

A fully local, offline chatbot running on the **M5Stack Cardputer ADV**
(ESP32-S3FN8, 512 KB SRAM, 8 MB flash, no PSRAM). It also runs on the
**original M5Stack Cardputer** — the firmware ships both keyboard drivers
(the ADV's TCA8418 and the original's IO matrix) and picks the right one at
boot. No Wi-Fi, no API, no SD card — the LLM lives in the firmware and runs
on the microcontroller itself. It makes small talk with multi-turn memory
and writes stories on request, at ~7 tok/s.

The model is [roneneldan/TinyStories-Instruct-3M][hf-neo] (GPT-Neo) fine-tuned
on ~70K simple-English dialogues (filtered [allenai/SODA][soda], formatted
`User: ...\nBot: ...<|endoftext|>`, loss masked to bot replies) mixed with 30%
story data so the story skill survives. The fine-tune runs in ~30 min on an
Apple-Silicon Mac via `tools/finetune_chat.py`.

[soda]: https://huggingface.co/datasets/allenai/soda

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

Honest limits: kindergarten English, no world knowledge (factual questions
get friendly confabulation), and an 80-token context (~3 short exchanges of
memory).

## Fine-tune pipeline (rebuild the chat model from scratch)

```sh
../cardputer_ai_venv/bin/python tools/prepare_chat_data.py --dialogues 150000   # ~5 min
../cardputer_ai_venv/bin/python tools/finetune_chat.py --epochs 3 \
    --out-dir data/chat_model_masked                             # ~30 min MPS
../cardputer_ai_venv/bin/python tools/convert_tinystories_instruct.py \
    --model-dir data/chat_model_masked --corpus data/chat_train.txt
```

Loss is masked to bot replies and story bodies (the model never trains on
producing user turns), and chat samples are tokenized segment-by-segment in
exactly the shapes the firmware feeds at inference.

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
heap. Pruned: 1.84 MB total model, 48 KB logits.

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

[pioarduino]: https://github.com/pioarduino/platform-espressif32

## Host testing (no hardware needed)

`tools/host/` stubs the ESP32 APIs so the exact engine code runs on your Mac:

```sh
../cardputer_ai_venv/bin/python tools/convert_tinystories_instruct.py --keep-bin --no-cpp
clang++ -std=c++17 -O2 -I tools/host tools/host/host_test.cpp main/llm.cpp -o /tmp/llm_host
/tmp/llm_host embed/model_neo_q4.bin embed/tok_neo.bin \
  "Summary: a girl finds a lost cat.\nStory:" --temp 0.8 --kv 80
```

## Memory budget (Cardputer ADV, ~280 KB free heap)

| Buffer                              | Bytes    |
|-------------------------------------|----------|
| KV cache, ctx=80, int8 + row scales | ~170 KB  |
| Logits (vocab=12060)                | 48 KB    |
| Activations + attention scores      | ~12 KB   |
| FreeRTOS + matmul worker stack      | 4 KB     |

The GPT-Neo KV cache is stored as **int8 with one fp32 scale per row** (the
LLaMA path keeps bf16) — at dim=128 that's 2 KB/position, which is what makes
an 80-token window fit. `KV_SEQ_LEN` lives in `main/main.cpp`; the model
ships 256 position embeddings, so RAM is the binding constraint, not flash.

## Engine notes

- GPT-Neo forward path: LayerNorm (+bias), learned position embeddings,
  GELU MLP, and — faithful to the original — **no 1/sqrt(d) attention
  scaling**. The alternating "local attention" layers have a 256-token
  window ≥ our context, so they degenerate to global causal attention.
- Tokenizer is **exact** GPT-2 byte-level BPE: the blob embeds the merge
  pair table (binary-searched from flash); verified 0 mismatches vs
  HuggingFace on 500 dataset lines.
- Q4 logits match the fp32 HF reference closely (top-3 identical on test
  prompts); divergence in long greedy decodes is normal quantization noise.

## What's not great yet

- 80 tokens of context ≈ 3 short exchanges of conversational memory; older
  turns silently fall out of the prompt.
- No facts, only vibes — for factual Q&A you'd need Wi-Fi + an API, or
  different hardware.
- ~7 tok/s measured on device. ESP32-S3 PIE SIMD for the Q4 dot product
  would roughly double it but isn't implemented.
- Chat quality is bounded by 3M params and the 47%-yield SODA filter; the
  next quality lever is more/better dialogue data, not more epochs.

## Changelog

- **v1.1**
  - Press the backtick (`` ` ``) key to stop a reply while it's being typed out.
  - Two new reply-length options below the normal range: **unlimited** (keeps
    going until the model decides to stop) and **unsafe** (lets longer replies
    keep going by reusing memory, clearing the chat when it runs out).
- **v1.0** — initial release.

## License

Code: MIT (see LICENSE). The embedded model derives from
TinyStories-Instruct-3M and the SODA dialogue dataset (CC BY 4.0) — see
NOTICE.md for full third-party attributions.
