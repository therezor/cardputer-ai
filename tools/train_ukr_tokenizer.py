#!/usr/bin/env python3
"""
Train a Ukrainian byte-level BPE tokenizer for the cardputer_ai engine.

The device does exact BPE by repeatedly merging the adjacent pair whose merge
*result* has the lowest id (main/llm.cpp:ctk2_encode). That is only correct
when token-id order equals merge-rank order — which a freshly trained
ByteLevelBPE satisfies by construction, since ids are handed out as
[specials][256 byte alphabet][merge results in rank order]. So no firmware
change is needed to swap in a Ukrainian vocabulary; only the blob changes.

Two properties are asserted here rather than discovered on the device:

  1. id order == merge-rank order (what ctk2_encode assumes)
  2. encode(A + B) == encode(A) + encode(B) at the boundaries main.cpp splits
     on ("User: u\\nBot:" | " reply" | eos | "\\n"). The ByteLevel pre-tokenizer
     never merges across a pre-token boundary, so this holds — but a mismatch
     here is exactly the failure that produced silent gibberish on device last
     time while host tests passed, so it is checked explicitly.

Usage:
  ../cardputer_ai_venv/bin/python tools/train_ukr_tokenizer.py
"""

import argparse
import json
from pathlib import Path

EOS = "<|endoftext|>"


def train(args):
    from tokenizers import ByteLevelBPETokenizer

    data = Path(args.data_dir)
    srcs = [data / f for f in ("chat_train.txt", "pretrain_train.txt")
            if (data / f).exists()]
    if not srcs:
        raise SystemExit(f"no corpus in {data} — run tools/prepare_ukr_data.py first")

    # BPE merge statistics converge long before the full 800 MB corpus; training
    # on a head slice of each source keeps this to a couple of minutes and a
    # bounded memory footprint. The chat file comes first so "User:"/"Bot:" are
    # well represented in the sample.
    sample = Path(args.out_dir) / "_tokenizer_sample.txt"
    sample.parent.mkdir(parents=True, exist_ok=True)
    budget = args.sample_mb * 1_000_000
    with sample.open("wb") as w:
        for src in srcs:
            take = budget // len(srcs)
            with src.open("rb") as r:
                chunk = r.read(take)
            # Do not cut mid-UTF-8 sequence.
            while chunk and (chunk[-1] & 0xC0) == 0x80:
                chunk = chunk[:-1]
            if chunk and chunk[-1] >= 0x80:
                chunk = chunk[:-1]
            w.write(chunk)
            w.write(b"\n")
    files = [str(sample)]
    print(f"[+] training on {args.sample_mb} MB sampled from "
          f"{', '.join(s.name for s in srcs)}")

    tok = ByteLevelBPETokenizer()
    # The chat corpus is included so "User:" / "Bot:" earn merges. Without them
    # byte-level BPE spells the labels out one byte per token, and at two
    # labels per exchange that is real context burned inside a 72-token window.
    tok.train(files=files, vocab_size=args.vocab_size, min_frequency=args.min_freq,
              special_tokens=[EOS])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok.save_model(str(out))

    # A GPT2TokenizerFast-compatible directory so the training scripts can load
    # it with from_pretrained().
    (out / "tokenizer_config.json").write_text(json.dumps({
        "model_max_length": 2048,
        "tokenizer_class": "GPT2Tokenizer",
        "bos_token": EOS, "eos_token": EOS, "unk_token": EOS,
    }, indent=2), encoding="utf-8")
    (out / "special_tokens_map.json").write_text(json.dumps({
        "bos_token": EOS, "eos_token": EOS, "unk_token": EOS,
    }, indent=2), encoding="utf-8")

    verify(out, data, args)


def verify(out: Path, data: Path, args):
    from tokenizers import ByteLevelBPETokenizer

    vocab = json.loads((out / "vocab.json").read_text(encoding="utf-8"))
    merge_lines = (out / "merges.txt").read_text(encoding="utf-8").splitlines()
    if merge_lines and merge_lines[0].startswith("#"):
        merge_lines = merge_lines[1:]

    # --- property 1: id order == merge-rank order -----------------------------
    prev = -1
    for rank, line in enumerate(merge_lines):
        a, b = line.split(" ")
        rid = vocab[a + b]
        if rid <= prev:
            raise SystemExit(f"FAIL: id order != rank order at rank {rank}")
        prev = rid
    print(f"[ok] id order == merge-rank order ({len(merge_lines):,} merges)")

    tok = ByteLevelBPETokenizer(str(out / "vocab.json"), str(out / "merges.txt"))
    enc = lambda s: tok.encode(s).ids

    # --- property 2: segment concatenation is byte-exact ----------------------
    val = (data / "chat_val.txt").read_text(encoding="utf-8")
    samples = [s for s in val.split("\n\n") if s.strip()][:400]
    checked = 0
    for s in samples:
        lines = s.split("\n")
        for i in range(0, len(lines) - 1, 2):
            u, b = lines[i], lines[i + 1]
            if not b.startswith("Bot: "):
                continue
            prefix = u + "\nBot:"
            reply = " " + b[len("Bot: "):].replace(EOS, "")
            if enc(prefix + reply) != enc(prefix) + enc(reply):
                raise SystemExit(
                    "FAIL: encode(prefix+reply) != encode(prefix)+encode(reply)\n"
                    f"  prefix={prefix!r}\n  reply={reply!r}")
            checked += 1
    print(f"[ok] segment concatenation is byte-exact ({checked:,} boundaries)")

    # --- compression / label cost --------------------------------------------
    for probe in ("User:", "Bot:", "\n", " Привіт! Як твої справи?"):
        print(f"[i] encode({probe!r}) -> {len(enc(probe))} tokens")

    body = "\n".join(l for l in val.split("\n")
                     if l and not l.startswith(("User:", "Bot:")))[:200_000]
    txt = val[:200_000]
    n_tok = len(enc(txt))
    print(f"[i] compression: {len(txt) / n_tok:.2f} chars/token on chat_val "
          f"({n_tok:,} tokens)")
    print(f"[i] vocab: {len(vocab):,} tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/ukr")
    ap.add_argument("--out-dir", default="data/ukr/tokenizer")
    ap.add_argument("--vocab-size", type=int, default=14000,
                    help="Ukrainian morphology fragments badly at English "
                         "vocab sizes; stay under the converter's 15,500 cap")
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--sample-mb", type=int, default=200,
                    help="MB of corpus to train the merge table on")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
