#!/usr/bin/env python3
"""
Build a chat fine-tuning corpus for TinyStories-Instruct-3M.

Sources:
  - allenai/soda (CC BY 4.0): everyday two-person dialogues. Filtered hard for
    simple English (the 3M model only knows TinyStories-level vocabulary):
    we keep the longest prefix of each dialogue whose turns are short and
    nearly free of tokens outside the TinyStories token set.
  - TinyStories-Instruct validation text: mixed in so the fine-tune doesn't
    forget how to write stories (story mode keeps working).

Output samples (plain text, one per line-block, separated by blank lines):
  User: <turn>\nBot: <turn><|endoftext|>\nUser: ...
The literal "<|endoftext|>" is understood by GPT2Tokenizer as the EOS token,
so the training script can tokenize samples as-is. The on-device chat mode
reproduces exactly this format, inserting EOS between exchanges.

Usage:
  venv/bin/python tools/prepare_chat_data.py --dialogues 60000
"""

import argparse
import collections
import json
import random
import re
from pathlib import Path

EOS = "<|endoftext|>"


def tinystories_token_set(snap: Path, valid_txt: Path):
    from tokenizers import ByteLevelBPETokenizer
    tok = ByteLevelBPETokenizer(str(snap / "vocab.json"), str(snap / "merges.txt"))
    text = valid_txt.read_text()
    seen = set()
    CH = 1_000_000
    for i in range(0, len(text), CH):
        seen.update(tok.encode(text[i:i + CH]).ids)
    return tok, seen


def simple_prefix(turns, tok, known, max_words, max_oov_frac):
    """Longest prefix of turns that stays simple. Returns list of turns."""
    kept = []
    for t in turns:
        t = re.sub(r"\s+", " ", t).strip()
        words = t.split()
        if not words or len(words) > max_words:
            break
        ids = tok.encode(t).ids
        oov = sum(1 for i in ids if i not in known)
        if oov / len(ids) > max_oov_frac:
            break
        kept.append(t)
    return kept


def format_dialogue(turns):
    """Alternate User/Bot; every Bot turn ends with EOS (the device's stop)."""
    out = []
    for i, t in enumerate(turns):
        if i % 2 == 0:
            out.append(f"User: {t}")
        else:
            out.append(f"Bot: {t}{EOS}")
    return "\n".join(out)


def story_samples(valid_txt: Path, n_chars: int, rng):
    """Random complete records (instruction header + story) from the
    TinyStories-Instruct text, which uses <|endoftext|> as separator."""
    text = valid_txt.read_text()
    records = [r.strip() for r in text.split(EOS) if len(r.strip()) > 100]
    rng.shuffle(records)
    out, total = [], 0
    for r in records:
        if total >= n_chars:
            break
        out.append(r + EOS)
        total += len(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", type=int, default=60000,
                    help="SODA dialogues to scan (yield after filtering is lower)")
    ap.add_argument("--max-words", type=int, default=22, help="per turn")
    ap.add_argument("--max-oov", type=float, default=0.02,
                    help="max fraction of tokens outside the TinyStories set")
    ap.add_argument("--min-turns", type=int, default=2)
    ap.add_argument("--story-frac", type=float, default=0.3,
                    help="fraction of corpus chars that are story records")
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "cardputer_ai"))
    args = ap.parse_args()

    from huggingface_hub import snapshot_download, hf_hub_download
    snap = Path(snapshot_download(repo_id="roneneldan/TinyStories-Instruct-3M",
                                  cache_dir=args.cache,
                                  allow_patterns=["*.json", "merges.txt"]))
    valid = Path(hf_hub_download(repo_id="roneneldan/TinyStoriesInstruct",
                                 repo_type="dataset",
                                 filename="TinyStories-Instruct-valid.txt",
                                 cache_dir=args.cache))

    print("[+] building TinyStories token set...")
    tok, known = tinystories_token_set(snap, valid)
    print(f"[+] {len(known):,} known tokens")

    from datasets import load_dataset
    ds = load_dataset("allenai/soda", split="train", streaming=True)

    rng = random.Random(1234)
    chats, chat_chars = [], 0
    scanned = 0
    for ex in ds:
        scanned += 1
        if scanned > args.dialogues:
            break
        turns = simple_prefix(ex["dialogue"], tok, known, args.max_words, args.max_oov)
        if len(turns) < args.min_turns:
            continue
        if len(turns) % 2 == 1:          # end on a Bot turn
            turns = turns[:-1]
        if len(turns) < args.min_turns:
            continue
        s = format_dialogue(turns)
        chats.append(s)
        chat_chars += len(s)
        if scanned % 10000 == 0:
            print(f"    scanned {scanned:,}: kept {len(chats):,} ({chat_chars/1e6:.1f} M chars)")

    print(f"[+] dialogues kept: {len(chats):,} / {scanned:,} ({chat_chars/1e6:.1f} M chars)")

    n_story_chars = int(chat_chars * args.story_frac / (1 - args.story_frac))
    stories = story_samples(valid, n_story_chars, rng)
    print(f"[+] story records mixed in: {len(stories):,} ({n_story_chars/1e6:.1f} M chars)")

    samples = chats + stories
    rng.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_frac))

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    (out / "chat_val.txt").write_text("\n\n".join(samples[:n_val]) + "\n")
    (out / "chat_train.txt").write_text("\n\n".join(samples[n_val:]) + "\n")
    meta = {"train_samples": len(samples) - n_val, "val_samples": n_val,
            "chat_chars": chat_chars, "story_chars": n_story_chars}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[+] wrote {out/'chat_train.txt'} ({len(samples)-n_val:,} samples) "
          f"and {out/'chat_val.txt'} ({n_val:,})")


if __name__ == "__main__":
    main()
