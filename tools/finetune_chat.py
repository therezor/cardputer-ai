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
_PRECISION = "fp32"


def _autocast(device: str, precision: str):
    """bf16 autocast where supported. The AMD iGPU runs bf16 ~3.5x faster than
    fp32; on MPS this is a no-op unless asked for."""
    import contextlib
    if precision == "fp32":
        return contextlib.nullcontext()
    dev = "cuda" if device.startswith("cuda") else device
    return torch.autocast(device_type=dev, dtype=torch.bfloat16)


def encode_sample(s: str, tokenizer):
    """Tokenize one sample into (ids, labels). Labels are -100 (no loss) on
    everything the model never has to produce — user turns, "Bot:" triggers,
    instruction headers — and real ids on bot replies / story bodies + EOS.

    Chat samples are encoded segment-by-segment in exactly the shapes the
    firmware feeds at inference ("User: u\\nBot:" then " reply" then eos),
    so train- and run-time tokenizations match byte for byte."""
    eos = tokenizer.eos_token_id
    enc = lambda t: tokenizer(t, add_special_tokens=False).input_ids
    # The "\n" that joins exchanges must come from the tokenizer, not a literal
    # id: it is 198 in GPT-2 but something else entirely in the Ukrainian
    # vocabulary, and a wrong joiner is invisible until the device talks gibberish.
    nl = enc("\n")
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
                put(nl, False)                         # "\n" joiner
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


def load_blocks(path: Path, tokenizer, seq_len: int, max_samples: int = 0):
    """Tokenize blank-line-separated samples and pack into seq_len blocks of
    (input ids, masked labels).

    encode_sample tokenizes segment-by-segment (several calls per sample), so
    this is linear in samples with a large constant. The Ukrainian corpus has
    ~2M samples, far more than a fine-tune needs — max_samples caps it."""
    text = path.read_text()
    samples = [s for s in text.split("\n\n") if s.strip()]
    if max_samples and len(samples) > max_samples:
        # Stride rather than truncate, so the cap keeps the corpus's mix of
        # subtitle dialogue and the repeated hand-written QA/deflections.
        step = len(samples) / max_samples
        samples = [samples[int(i * step)] for i in range(max_samples)]
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
        with _autocast(device, _PRECISION):
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
    ap.add_argument("--base", default="3M", choices=["1M", "3M", "8M"],
                    help="TinyStories-Instruct base model size")
    ap.add_argument("--base-dir", default=None,
                    help="local checkpoint to fine-tune instead of a "
                         "TinyStories base (must contain vocab.json/merges.txt "
                         "for its own tokenizer) — this is how the Ukrainian "
                         "model, which has its own vocabulary, is fine-tuned")
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32",
                    help="bf16 is ~3.5x faster on RDNA3.5 (AMD iGPU)")
    ap.add_argument("--device", default="auto",
                    help="auto | cuda (also ROCm) | mps | cpu")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="cap on training samples (0 = all); the Ukrainian "
                         "corpus has ~2M, more than a fine-tune needs")
    ap.add_argument("--max-val-samples", type=int, default=4000)
    ap.add_argument("--sample-prompt",
                    default="User: hi! how are you today?\nBot:",
                    help="prompt shown after each epoch")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "cardputer_ai"))
    args = ap.parse_args()

    global _PRECISION
    _PRECISION = args.precision

    from transformers import GPTNeoForCausalLM, GPT2TokenizerFast
    from huggingface_hub import snapshot_download

    # cuda covers ROCm: PyTorch's ROCm builds expose AMD GPUs through the
    # torch.cuda API. Without this the script silently falls back to CPU on an
    # AMD box, since MPS is a Mac-only backend.
    if args.device != "auto":
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[+] device: {device}")

    if args.base_dir:
        snap = Path(args.base_dir)
        print(f"[+] fine-tuning local checkpoint {snap}")
    else:
        snap = Path(snapshot_download(
            repo_id=f"roneneldan/TinyStories-Instruct-{args.base}",
            cache_dir=args.cache,
            allow_patterns=["*.bin", "*.json", "merges.txt", "vocab.json"]))
    tokenizer = GPT2TokenizerFast.from_pretrained(snap)
    assert tokenizer("<|endoftext|>").input_ids == [tokenizer.eos_token_id]
    model = GPTNeoForCausalLM.from_pretrained(snap).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[+] params: {n_params/1e6:.1f}M")

    data_dir = Path(args.data_dir)
    train_x, train_y = load_blocks(data_dir / "chat_train.txt", tokenizer,
                                   args.seq_len, args.max_samples)
    val_x, val_y = load_blocks(data_dir / "chat_val.txt", tokenizer, args.seq_len,
                               args.max_val_samples)
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
            with _autocast(device, _PRECISION):
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
                                    args.sample_prompt).replace("\n", " | "))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.cpu().save_pretrained(out, safe_serialization=False)  # pytorch_model.bin
    tokenizer.save_pretrained(out)
    print(f"[+] saved to {out}")


if __name__ == "__main__":
    main()
