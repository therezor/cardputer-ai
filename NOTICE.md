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

## Tokenizer

- GPT-2 byte-level BPE vocabulary and merges (`vocab.json` / `merges.txt`),
  from OpenAI's GPT-2 release (https://github.com/openai/gpt-2), MIT
  (Modified MIT License, Copyright (c) 2019 OpenAI). Embedded here in
  pruned, re-encoded form inside `tok_data.cpp`.

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
