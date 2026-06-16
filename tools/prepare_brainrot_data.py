#!/usr/bin/env python3
"""
Build a brainrot-translator fine-tuning corpus for TinyStories-Instruct-3M.

Source: shvn22k/brainrot-dataset (parallel corpus, ~6.7K pairs) — each example
pairs plain English (`source`) with an internet-slang "brainrot" rewrite
(`target`). We frame it as the firmware's native chat format so the on-device
chat mode becomes a translator with NO firmware change: you type normal English
(the "User" turn) and the model replies with the brainrot version (the "Bot"
turn).

Output samples (plain text, one exchange per block, blank-line separated):
  User: <plain english>\nBot: <brainrot><|endoftext|>
The literal "<|endoftext|>" is the GPT2 EOS token, so finetune_chat.py
tokenizes samples as-is and the device reproduces this exact shape at runtime.

Usage:
  venv/bin/python tools/prepare_brainrot_data.py
"""

import argparse
import json
import re
from pathlib import Path

EOS = "<|endoftext|>"

# Smart punctuation the dataset uses -> ASCII the Cardputer font can render.
_PUNCT = {
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "—": "-", "–": "-", "…": "...", " ": " ",
}


def clean(s: str, ascii_only: bool = True) -> str:
    """Collapse whitespace/newlines so each turn stays on one line (the
    finetune encoder walks samples two lines at a time: User then Bot).

    The Cardputer display font has no emoji/pictograph glyphs, so by default
    we normalize smart punctuation to ASCII and drop every remaining
    non-ASCII codepoint (emoji, variation selectors, ZWJ, etc). The model
    then never learns to emit anything the device can't draw."""
    for a, b in _PUNCT.items():
        s = s.replace(a, b)
    if ascii_only:
        s = "".join(ch for ch in s if ch == "\n" or 0x20 <= ord(ch) <= 0x7E)
    return re.sub(r"\s+", " ", s).strip()


def format_pair(source: str, target: str, ascii_only: bool):
    src, tgt = clean(source, ascii_only), clean(target, ascii_only)
    if tgt.endswith(EOS):
        tgt = tgt[: -len(EOS)].rstrip()
    if not src or not tgt:          # e.g. an emoji-only turn cleaned to empty
        return None
    return f"User: {src}\nBot: {tgt}{EOS}"


def build_split(ds_split, src_col, tgt_col, ascii_only):
    out, dropped = [], 0
    for ex in ds_split:
        src, tgt = ex.get(src_col), ex.get(tgt_col)
        if not src or not tgt:
            continue
        s = format_pair(src, tgt, ascii_only)
        # a valid block is exactly two lines (User / Bot); skip anything that
        # collapsed wrong (empty turn after cleaning)
        if s is None or s.count("\n") != 1:
            dropped += 1
            continue
        out.append(s)
    if dropped:
        print(f"    dropped {dropped} rows (empty after cleaning)")
    return out


def pick_cols(features):
    names = list(features)
    src = next((c for c in ("source", "english", "input", "formal", "text") if c in names), None)
    tgt = next((c for c in ("target", "brainrot", "output", "slang", "translation") if c in names), None)
    if not src or not tgt:
        raise SystemExit(f"could not identify source/target columns in {names}")
    return src, tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="shvn22k/brainrot-dataset")
    ap.add_argument("--out-dir", default="data/brainrot")
    ap.add_argument("--fold-test", action="store_true",
                    help="fold the test split into training (more data)")
    ap.add_argument("--keep-emoji", action="store_true",
                    help="keep emoji/non-ASCII (default: strip — Cardputer font "
                         "has no emoji glyphs)")
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "cardputer_ai"))
    args = ap.parse_args()
    ascii_only = not args.keep_emoji

    from datasets import load_dataset
    ds = load_dataset(args.repo, cache_dir=args.cache)
    print(f"[+] splits: { {k: len(v) for k, v in ds.items()} }")

    src_col, tgt_col = pick_cols(ds["train"].features)
    print(f"[+] columns: source='{src_col}'  target='{tgt_col}'")

    print(f"[+] ascii_only={ascii_only} (strip emoji/non-ASCII)")
    train = build_split(ds["train"], src_col, tgt_col, ascii_only)
    val_key = "validation" if "validation" in ds else ("valid" if "valid" in ds else None)
    val = build_split(ds[val_key], src_col, tgt_col, ascii_only) if val_key else []
    if args.fold_test and "test" in ds:
        extra = build_split(ds["test"], src_col, tgt_col, ascii_only)
        train += extra
        print(f"[+] folded {len(extra):,} test rows into training")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "chat_train.txt").write_text("\n\n".join(train) + "\n")
    (out / "chat_val.txt").write_text("\n\n".join(val) + "\n")
    meta = {"repo": args.repo, "train_samples": len(train), "val_samples": len(val),
            "train_chars": sum(len(s) for s in train)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[+] wrote {out/'chat_train.txt'} ({len(train):,}) and "
          f"{out/'chat_val.txt'} ({len(val):,})")
    print("\n[+] sample blocks:")
    for s in train[:3]:
        print("    " + s.replace("\n", "  ||  "))


if __name__ == "__main__":
    main()
