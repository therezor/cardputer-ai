#!/usr/bin/env python3
"""
Fine-tune roneneldan/TinyStories-Instruct-3M on the chat corpus produced by
tools/prepare_chat_data.py. Runs on Apple Silicon (MPS) or CPU.

  venv/bin/python tools/finetune_chat.py --epochs 2

Writes the checkpoint to data/chat_model/ (HF format, pytorch_model.bin) —
feed that to convert_tinystories_instruct.py --model-dir data/chat_model.
"""

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


EOS_LIT = "<|endoftext|>"


def encode_sample(s: str, tokenizer):
    """Tokenize one sample into (ids, labels). Labels are -100 (no loss) on
    everything the model never has to produce — user turns, "Bot:" triggers,
    instruction headers — and real ids on bot replies / story bodies + EOS.

    Chat samples are encoded segment-by-segment in exactly the shapes the
    firmware feeds at inference ("User: u\\nBot:" then " reply" then eos),
    so train- and run-time tokenizations match byte for byte."""
    eos = tokenizer.eos_token_id
    enc = lambda t: tokenizer(t, add_special_tokens=False).input_ids
    ids, labels = [], []

    def put(toks, train):
        ids.extend(toks)
        labels.extend(toks if train else [-100] * len(toks))

    if s.startswith("User: "):
        lines = s.split("\n")
        for i in range(0, len(lines) - 1, 2):
            u = lines[i]
            b = lines[i + 1]
            if not b.startswith("Bot: "):
                break
            reply = " " + b[len("Bot: "):]
            reply = reply[:-len(EOS_LIT)] if reply.endswith(EOS_LIT) else reply
            if i > 0:
                put([198], False)                      # "\n" joiner
            put(enc(u + "\nBot:"), False)
            put(enc(reply), True)
            put([eos], True)                           # learn to stop
    else:
        # Story record: mask the instruction header through "Story:", train
        # on the story body. Falls back to training on everything.
        body_at = s.find("\nStory:")
        s = s[:-len(EOS_LIT)] if s.endswith(EOS_LIT) else s
        if body_at >= 0:
            cut = body_at + len("\nStory:")
            put(enc(s[:cut]), False)
            put(enc(s[cut:]), True)
        else:
            put(enc(s), True)
        put([eos], True)
    return ids, labels


def load_blocks(path: Path, tokenizer, seq_len: int):
    """Tokenize blank-line-separated samples and pack into seq_len blocks of
    (input ids, masked labels)."""
    text = path.read_text()
    samples = [s for s in text.split("\n\n") if s.strip()]
    ids, labels = [], []
    for s in samples:
        i, l = encode_sample(s, tokenizer)
        ids.extend(i)
        labels.extend(l)
    n_blocks = len(ids) // seq_len
    x = torch.tensor(ids[:n_blocks * seq_len], dtype=torch.long).view(n_blocks, seq_len)
    y = torch.tensor(labels[:n_blocks * seq_len], dtype=torch.long).view(n_blocks, seq_len)
    trained = (y != -100).float().mean().item()
    print(f"    {path.name}: {n_blocks:,} blocks, {trained*100:.0f}% of tokens in loss")
    return x, y


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x, labels=y)
        total += out.loss.item() * x.size(0)
        n += x.size(0)
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def sample(model, tokenizer, device, prompt, max_new=60):
    model.eval()
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=True,
                         temperature=0.8, top_k=50,
                         pad_token_id=tokenizer.eos_token_id)
    model.train()
    return tokenizer.decode(out[0][ids.shape[1]:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="data/chat_model")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "cardputer_ai"))
    args = ap.parse_args()

    from transformers import GPTNeoForCausalLM, GPT2TokenizerFast
    from huggingface_hub import snapshot_download

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[+] device: {device}")

    snap = snapshot_download(repo_id="roneneldan/TinyStories-Instruct-3M",
                             cache_dir=args.cache,
                             allow_patterns=["*.bin", "*.json", "merges.txt", "vocab.json"])
    tokenizer = GPT2TokenizerFast.from_pretrained(snap)
    assert tokenizer("<|endoftext|>").input_ids == [tokenizer.eos_token_id]
    model = GPTNeoForCausalLM.from_pretrained(snap).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[+] params: {n_params/1e6:.1f}M")

    data_dir = Path(args.data_dir)
    train_x, train_y = load_blocks(data_dir / "chat_train.txt", tokenizer, args.seq_len)
    val_x, val_y = load_blocks(data_dir / "chat_val.txt", tokenizer, args.seq_len)
    print(f"[+] train blocks: {len(train_x):,} ({len(train_x)*args.seq_len/1e6:.1f}M tokens), "
          f"val blocks: {len(val_x):,}")

    train_loader = DataLoader(TensorDataset(train_x, train_y),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=args.batch_size)

    steps_total = len(train_loader) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        p = (step - args.warmup) / max(1, steps_total - args.warmup)
        return 1e-5 + 0.5 * (args.lr - 1e-5) * (1 + math.cos(math.pi * p))

    print(f"[+] {steps_total} steps total")
    print(f"[+] initial val loss: {evaluate(model, val_loader, device):.4f}")

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for x, y in train_loader:
            lr = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = x.to(device), y.to(device)
            loss = model(x, labels=y).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                tok_s = step * args.batch_size * args.seq_len / (time.time() - t0)
                print(f"    step {step}/{steps_total} loss {loss.item():.4f} "
                      f"lr {lr:.2e} ({tok_s/1e3:.0f}K tok/s)", flush=True)
        vl = evaluate(model, val_loader, device)
        print(f"[+] epoch {epoch+1}: val loss {vl:.4f}")
        print("    sample:", sample(model, tokenizer, device,
                                    "User: hi! how are you today?\nBot:").replace("\n", " | "))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.cpu().save_pretrained(out, safe_serialization=False)  # pytorch_model.bin
    tokenizer.save_pretrained(out)
    print(f"[+] saved to {out}")


if __name__ == "__main__":
    main()
