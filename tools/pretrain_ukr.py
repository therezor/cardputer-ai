#!/usr/bin/env python3
"""
Pretrain a Ukrainian GPT-Neo at the cardputer_ai 8M architecture.

The English pipeline fine-tunes roneneldan/TinyStories-Instruct-8M. That is not
available to us: a Ukrainian tokenizer replaces the embedding table, so the
English input/output embeddings (which are tied, and are a third of the model)
are meaningless. What remains transferable is the transformer body, so this
script supports both inits and lets a short pilot decide between them:

  --init scratch   random init at the target architecture
  --init warm      body + ln_f from TinyStories-Instruct-8M, fresh embeddings

Architecture is pinned to what main/llm.cpp already understands (GPT-Neo,
dim 256, 8 layers, 16 heads, alternating global/local attention, window 256).
max_position_embeddings is 256 because the converter keeps exactly 256 wpe rows
and llm_forward_at saturates abspos there — training longer contexts would be
compute spent on a regime the device never reaches.

Usage:
  ../cardputer_ai_venv/bin/python tools/pretrain_ukr.py --init scratch \\
      --minutes 30 --out-dir data/ukr/pilot_scratch
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

SEQ_LEN = 256
BASE_8M = "roneneldan/TinyStories-Instruct-8M"


def pick_device(name: str = "auto"):
    """cuda covers ROCm too — PyTorch's ROCm builds expose AMD GPUs (including
    Strix Halo / gfx1151) through the torch.cuda API, so no separate branch."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def autocast_ctx(device: str, precision: str):
    """bf16 autocast where the backend supports it. At ~10M params the model is
    launch-overhead bound rather than FLOP bound, so this matters less than
    batch size — but it stacks with it."""
    import contextlib
    if precision == "fp32":
        return contextlib.nullcontext()
    dev = "cuda" if device.startswith("cuda") else device
    return torch.autocast(device_type=dev, dtype=torch.bfloat16)


def build_config(vocab_size: int, intermediate: int = 1024):
    from transformers import GPTNeoConfig
    return GPTNeoConfig(
        vocab_size=vocab_size,
        intermediate_size=intermediate,
        hidden_size=256,
        num_layers=8,
        num_heads=16,
        max_position_embeddings=SEQ_LEN,
        window_size=256,
        attention_types=[[["global", "local"], 4]],
        activation_function="gelu_new",
        resid_dropout=0.0, embed_dropout=0.0, attention_dropout=0.0,
        bos_token_id=0, eos_token_id=0,
    )


def tokenize_corpus(paths, tok_dir: Path, cache: Path, max_tokens: int):
    """Tokenize the corpus once into a flat uint16 array, cached on disk."""
    if cache.exists():
        arr = np.load(cache, mmap_mode="r")
        print(f"[+] cached tokens: {len(arr):,} from {cache}")
        return arr

    from tokenizers import ByteLevelBPETokenizer
    tok = ByteLevelBPETokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))
    # Never hardcode ids: the GPT-2 values (50256 for EOS, 198 for newline) are
    # meaningless in a freshly trained vocabulary, and a mismatch here shows up
    # only as gibberish on device, after training.
    eos_id, nl_id = eos_newline_ids(tok_dir)

    out = np.empty(max_tokens, dtype=np.uint16)
    n = 0
    t0 = time.time()
    for path in paths:
        if n >= max_tokens:
            break
        print(f"[+] tokenizing {path}")
        with open(path, encoding="utf-8") as f:
            batch, chars = [], 0
            for line in f:
                # Blank line = end of a block/sample: emit EOS so the model
                # learns document boundaries the same way inference feeds them.
                if not line.strip():
                    batch.append(None)
                    continue
                batch.append(line.rstrip("\n"))
                chars += len(line)
                if chars < 4_000_000:
                    continue
                n = _flush(tok, batch, out, n, eos_id, nl_id, max_tokens)
                batch, chars = [], 0
                if n >= max_tokens:
                    break
                print(f"    {n:,} tokens ({time.time()-t0:.0f}s)", flush=True)
            if batch and n < max_tokens:
                n = _flush(tok, batch, out, n, eos_id, nl_id, max_tokens)

    arr = out[:n]
    np.save(cache, arr)
    print(f"[+] tokenized {n:,} tokens in {time.time()-t0:.0f}s -> {cache}")
    return arr


def eos_newline_ids(tok_dir: Path):
    """Resolve <|endoftext|> and "\\n" to real ids in this tokenizer."""
    import json
    from tokenizers import ByteLevelBPETokenizer
    vocab = json.loads((tok_dir / "vocab.json").read_text(encoding="utf-8"))
    eos = vocab.get("<|endoftext|>")
    if eos is None:
        raise SystemExit("tokenizer has no <|endoftext|>")
    t = ByteLevelBPETokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))
    nl = t.encode("\n").ids
    if len(nl) != 1:
        raise SystemExit(f'"\\n" encodes to {len(nl)} tokens, expected 1')
    return eos, nl[0]


def _flush(tok, batch, out, n, eos_id, nl_id, max_tokens):
    texts = [b for b in batch if b is not None]
    encs = tok.encode_batch(texts) if texts else []
    it = iter(encs)
    for b in batch:
        if n >= max_tokens:
            break
        if b is None:
            out[n] = eos_id
            n += 1
            continue
        ids = next(it).ids
        take = min(len(ids), max_tokens - n)
        out[n:n + take] = ids[:take]
        n += take
        if n < max_tokens:
            out[n] = nl_id   # keep line structure; "\n" is a real token
            n += 1
    return n


def make_model(args, vocab_size, device):
    from transformers import GPTNeoForCausalLM
    cfg = build_config(vocab_size, args.hidden)
    model = GPTNeoForCausalLM(cfg)

    if args.init == "warm":
        from huggingface_hub import snapshot_download
        snap = Path(snapshot_download(repo_id=BASE_8M, cache_dir=args.cache,
                                      allow_patterns=["*.bin", "*.json", "*.txt",
                                                      "*.safetensors"]))
        base = GPTNeoForCausalLM.from_pretrained(snap)
        bsd = base.state_dict()
        tsd = model.state_dict()
        loaded, skipped = 0, []
        for k, v in bsd.items():
            if k not in tsd:
                skipped.append(k)
                continue
            if k.endswith("wte.weight"):
                skipped.append(k)          # vocabulary differs entirely
                continue
            if k.endswith("wpe.weight"):
                tsd[k].copy_(v[:SEQ_LEN])  # keep the first 256 learned positions
                loaded += 1
                continue
            if tsd[k].shape != v.shape:
                skipped.append(k)
                continue
            tsd[k].copy_(v)
            loaded += 1
        model.load_state_dict(tsd)
        print(f"[+] warm start from {BASE_8M}: {loaded} tensors copied, "
              f"{len(skipped)} fresh ({', '.join(skipped[:4])}...)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[+] model: {n_params/1e6:.2f}M params, init={args.init}")
    return model.to(device)


def batches(arr, batch_size, seq_len, rng, device):
    """Random contiguous windows. The corpus is far larger than one epoch of
    training, so sampling windows beats materialising a shuffled block index."""
    hi = len(arr) - seq_len - 1
    while True:
        idx = rng.integers(0, hi, size=batch_size)
        x = np.stack([arr[i:i + seq_len] for i in idx]).astype(np.int64)
        y = np.stack([arr[i + 1:i + seq_len + 1] for i in idx]).astype(np.int64)
        yield (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))


@torch.no_grad()
def evaluate(model, arr, args, device, n_batches=40):
    model.eval()
    rng = np.random.default_rng(1234)          # fixed windows => comparable runs
    tot, cnt = 0.0, 0
    for _, (x, y) in zip(range(n_batches), batches(arr, args.batch_size, SEQ_LEN, rng, device)):
        with autocast_ctx(device, args.precision):
            loss = model(input_ids=x, labels=y).loss
        tot += loss.item()
        cnt += 1
    model.train()
    return tot / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/ukr")
    ap.add_argument("--tok-dir", default="data/ukr/tokenizer")
    ap.add_argument("--out-dir", default="data/ukr/pretrained")
    ap.add_argument("--cache", default=".cache")
    ap.add_argument("--init", choices=["scratch", "warm"], default="scratch")
    ap.add_argument("--hidden", type=int, default=1024,
                    help="MLP intermediate size. The device reads this from "
                         "the model header, and the KV cache is sized by "
                         "dim/layers only, so widening the MLP buys capacity "
                         "at zero context cost.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--steps", type=int, default=0, help="0 = run until --minutes")
    ap.add_argument("--minutes", type=float, default=0, help="wall-clock budget")
    ap.add_argument("--max-tokens", type=int, default=240_000_000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--device", default="auto",
                    help="auto | cuda (also ROCm) | mps | cpu")
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.steps and not args.minutes:
        ap.error("give --steps or --minutes")

    device = pick_device(args.device)
    print(f"[+] device: {device}  precision: {args.precision}")
    torch.manual_seed(args.seed)
    data = Path(args.data_dir)
    tok_dir = Path(args.tok_dir)
    vocab_size = len(__import__("json").loads(
        (tok_dir / "vocab.json").read_text(encoding="utf-8")))

    # Pretraining uses the prose/dialogue text tier only. The chat corpus is a
    # separate stage (tools/finetune_chat.py) with masked loss; appending it
    # here would also be silently truncated away by --max-tokens.
    train_arr = tokenize_corpus(
        [data / "pretrain_train.txt"], tok_dir,
        data / f"tokens_train_{args.max_tokens//1_000_000}M.npy", args.max_tokens)
    val_arr = tokenize_corpus(
        [data / "pretrain_val.txt"], tok_dir,
        data / "tokens_val.npy", 2_000_000)

    model = make_model(args, vocab_size, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)

    rng = np.random.default_rng(args.seed)
    gen = batches(train_arr, args.batch_size, SEQ_LEN, rng, device)
    total_steps = args.steps if args.steps else 10**9
    deadline = time.time() + args.minutes * 60 if args.minutes else None

    t0 = time.time()
    tokens_done = 0
    step = 0
    best = float("inf")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    while step < total_steps:
        if deadline and time.time() > deadline:
            break
        # Cosine schedule over the *planned* horizon; with a wall-clock budget
        # estimate the horizon from measured throughput after warmup.
        if args.steps:
            horizon = args.steps
        elif step > args.warmup:
            rate = step / (time.time() - t0)
            horizon = max(step + 1, int(rate * args.minutes * 60))
        else:
            horizon = args.warmup * 10
        lr = (args.lr * step / max(args.warmup, 1) if step < args.warmup
              else args.lr * (0.1 + 0.9 * 0.5 *
                              (1 + math.cos(math.pi * min(1.0, (step - args.warmup) /
                                                          max(1, horizon - args.warmup))))))
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = next(gen)
        with autocast_ctx(device, args.precision):
            loss = model(input_ids=x, labels=y).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        tokens_done += x.numel()
        step += 1

        if step % 50 == 0:
            el = time.time() - t0
            print(f"step {step:6d}  loss {loss.item():.4f}  lr {lr:.2e}  "
                  f"{tokens_done/el:,.0f} tok/s  {el/60:.1f} min", flush=True)
        if step % args.eval_every == 0:
            vl = evaluate(model, val_arr, args, device)
            print(f"  [eval] step {step}  val_loss {vl:.4f}", flush=True)
            if vl < best:
                best = vl
                model.save_pretrained(out, safe_serialization=False)

    vl = evaluate(model, val_arr, args, device)
    print(f"[final] step {step}  val_loss {vl:.4f}  best {min(best, vl):.4f}  "
          f"tokens {tokens_done:,}  {(time.time()-t0)/60:.1f} min")
    if vl <= best:
        model.save_pretrained(out, safe_serialization=False)
    for f in ("vocab.json", "merges.txt", "tokenizer_config.json",
              "special_tokens_map.json"):
        if (tok_dir / f).exists():
            (out / f).write_bytes((tok_dir / f).read_bytes())
    print(f"[+] saved {out}")


if __name__ == "__main__":
    main()
