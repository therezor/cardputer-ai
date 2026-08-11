# Third-party components and attributions

The source code in this repository is MIT-licensed (see LICENSE). The
embedded model and tokenizer artifacts derive from the following works:

## Base model — TinyStories-Instruct-3M

- Ronen Eldan & Yuanzhi Li, *TinyStories: How Small Can Language Models Be
  and Still Speak Coherent English?* (arXiv:2305.07759).
- Weights: https://huggingface.co/roneneldan/TinyStories-Instruct-3M —
  published without an explicit license tag. The companion TinyStories
  dataset is CDLA-Sharing-1.0, which places no restrictions on results
  (e.g. trained models). If the licensing of the base weights matters for
  your use case, contact the model author.

## Fine-tuning data

- **SODA** (https://huggingface.co/datasets/allenai/soda), CC BY 4.0.
  Kim et al., *SODA: Million-scale Dialogue Distillation with Social
  Commonsense Contextualization* (arXiv:2212.10465). The chat fine-tune
  embedded in `model_data.cpp` was trained on a filtered subset.
- **TinyStoriesInstruct**
  (https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct),
  CDLA-Sharing-1.0 (per the TinyStories dataset family).
- **DailyDialog**
  (https://huggingface.co/datasets/li2017dailydialog/daily_dialog),
  **CC BY-NC-SA 4.0 (non-commercial)**. Li et al., *DailyDialog: A Manually
  Labelled Multi-turn Dialogue Dataset* (arXiv:1710.03957). Filtered subset
  used in the chat fine-tune.
- **SciQ** (https://huggingface.co/datasets/allenai/sciq),
  **CC BY-NC 3.0 (non-commercial)**. Welbl et al., *Crowdsourcing Multiple
  Choice Science Questions* (arXiv:1707.06209). Only question texts are
  used (paired with hand-written deflection replies) to teach graceful
  "I don't know" behavior.

Note: the DailyDialog and SciQ licenses are non-commercial; a model
fine-tuned on them should not be distributed commercially.

## Tokenizer

- GPT-2 byte-level BPE vocabulary and merges (`vocab.json` / `merges.txt`),
  from OpenAI's GPT-2 release (https://github.com/openai/gpt-2), MIT
  (Modified MIT License, Copyright (c) 2019 OpenAI). Embedded here in
  pruned, re-encoded form inside `tok_data.cpp`.

## Ukrainian model (branch `ukr`)

The Ukrainian build replaces the base model entirely — a Ukrainian tokenizer
means a new embedding table, so nothing of TinyStories-Instruct is reused.

- **OPUS OpenSubtitles v2024**, Ukrainian monolingual
  (https://opus.nlpl.eu/OpenSubtitles/), derived from OpenSubtitles.org.
  Lison & Tiedemann, *OpenSubtitles2016: Extracting Large Parallel Corpora
  from Movie and TV Subtitles* (LREC 2016). Used, filtered, as both the
  pretraining text and the source of dialogue pairs. OPUS distributes these
  corpora for research use; check the source terms before commercial use.
- **OPUS Tatoeba** (https://opus.nlpl.eu/Tatoeba/), from Tatoeba.org,
  **CC BY 2.0 FR**. Ukrainian sentences mixed into the pretraining tier.
- **lang-uk/malyuk** (https://huggingface.co/datasets/lang-uk/malyuk),
  **no license tag**. A community compilation of UberText 2.0, OSCAR
  (unshuffled_deduplicated_uk) and Ukrainian news; its author states it is not
  an official release. ~30% of the pretraining tier — the complete-clause
  prose that carries Ukrainian agreement, which subtitles do not. Consult the
  constituent corpora before any commercial use.
- `tools/ukr_qa_facts.py` — hand-written for this repository, MIT with the
  rest of the source.

## Vendored font (`main/ukr_font.c`)

- **GNU Unifont** Cyrillic subset, via u8g2
  (https://github.com/olikraus/u8g2), font
  `u8g2_font_unifont_t_cyrillic`. **SIL Open Font License 1.1**,
  Copyright (C) 1998-2024 Roman Czyborra, Paul Hardy, Qianqian Fang,
  Andrew Miller, Johnnie Weaver, David Corbett, Nils Moskopp,
  Rebecca Bettencourt, Ho-Seok Ee, et al. Needed because m5gfx's bundled
  efont covers Russian Cyrillic but omits Ukrainian Ґ Є І Ї ґ є і ї.

## Alternative model (not embedded by default)

- **Maykeye/TinyLLama-v0** (https://huggingface.co/Maykeye/TinyLLama-v0),
  Apache-2.0. Supported by the engine via
  `tools/convert_tinyllama_v0.py`.

## Vendored keyboard driver (`main/keyboard/`)

- Ported from **M5Cardputer** v1.1.1
  (https://github.com/m5stack/M5Cardputer), MIT,
  Copyright (c) 2025 M5Stack Technology CO LTD. Arduino GPIO/interrupt
  calls were replaced with ESP-IDF `driver/gpio` equivalents.
- Includes M5Stack's adaptation of the **Adafruit TCA8418** keypad driver
  (https://github.com/adafruit/Adafruit_TCA8418), BSD, Copyright (c)
  Limor Fried (Adafruit Industries).

## Acknowledgements

- The inference engine follows the structure of Andrej Karpathy's
  **llama2.c** (https://github.com/karpathy/llama2.c), MIT. The code here
  is an independent implementation extended with Q4_0 quantization, a
  GPT-Neo forward path, int8 KV cache, and a flash-walking tokenizer.
