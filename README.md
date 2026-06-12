# Cardputer AI — a tiny local chatbot for the ESP32 Cardputer ADV

A fully local, offline chatbot running on the **M5Stack Cardputer ADV**
(ESP32-S3FN8, 512 KB SRAM, 8 MB flash, no PSRAM). No Wi-Fi, no API, no SD
card — the LLM lives in the firmware and runs on the microcontroller itself.
It makes small talk with multi-turn memory and writes stories on request,
at ~7 tok/s.

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

NOTE: keep the venv (and anything containing PyTorch) outside the sketch
directory — the Arduino IDE's sketch scanner walks the folder and errors on
torch's header filenames.

## What's in the box

```
cardputer_ai/
├── cardputer_ai.ino           boot + chat loop + story-mode prompt wrapper
├── llm.{h,cpp}                Q4_0 engine: LLaMA + GPT-Neo forward paths,
│                              dual-core matmul, exact byte-level BPE
├── ui.{h,cpp}                 chat UI
├── partitions.csv             6 MB factory app slot for app + embedded model
├── model_data.cpp             generated — Q4 model bytes (~1.9 MB for 3M)
├── tok_data.cpp               generated — pruned GPT-2 tokenizer (~190 KB)
└── tools/
    ├── convert_tinystories_instruct.py   HF GPT-Neo → Q4_0 + pruned vocab
    ├── convert_tinyllama_v0.py           the old TinyLLama-v0 converter
    └── host/                             macOS/Linux test harness for llm.cpp
```

## One-time setup

```sh
python3 -m venv ../cardputer_ai_venv  # OUTSIDE the sketch dir - the Arduino IDE
                                       # sketch scanner chokes on torch headers
../cardputer_ai_venv/bin/pip install huggingface_hub tokenizers torch numpy datasets transformers
../cardputer_ai_venv/bin/python tools/convert_tinystories_instruct.py            # base 3M model
../cardputer_ai_venv/bin/python tools/convert_tinystories_instruct.py --model 8M # bigger, ~5MB
```

The converter downloads the model, prunes the 50,257-token GPT-2 vocab to the
~12K tokens the TinyStories dataset actually uses (closed under BPE merge
derivation, so encoding stays exact), quantizes to Q4_0, and writes
**`model_data.cpp`** / **`tok_data.cpp`** next to the sketch.

Why prune? The full GPT-2 embedding table would be 13M params — bigger than
the 3M transformer itself — and a 50K-logit buffer (196 KB) doesn't fit our
heap. Pruned: 1.84 MB total model, 48 KB logits.

## Build & flash

Arduino IDE 2.x with the M5Stack board package (or `arduino-cli`):

- Board: **M5Cardputer** / **M5Stack-CardputerADV**
- PSRAM: **Disabled**
- Flash Size: **8 MB (64 Mb)**
- Partition Scheme: **Custom** (picks up `partitions.csv` automatically)

```sh
arduino-cli compile -b m5stack:esp32:m5stack_cardputer \
  --board-options PartitionScheme=custom,FlashSize=8M .
```

Flash via USB from the IDE, or export the compiled binary and install with
bmorcelli/Launcher as before.

## Host testing (no hardware needed)

`tools/host/` stubs the ESP32 APIs so the exact engine code runs on your Mac:

```sh
../cardputer_ai_venv/bin/python tools/convert_tinystories_instruct.py --keep-bin --no-cpp
clang++ -std=c++17 -O2 -I tools/host tools/host/host_test.cpp llm.cpp -o /tmp/llm_host
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
an 80-token window fit. `KV_SEQ_LEN` lives in `cardputer_ai.ino`; the model
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

## License

Code: MIT (see LICENSE). The embedded model derives from
TinyStories-Instruct-3M and the SODA dialogue dataset (CC BY 4.0) — see
NOTICE.md for full third-party attributions.
