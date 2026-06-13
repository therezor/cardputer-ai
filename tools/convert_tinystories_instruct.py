#!/usr/bin/env python3
"""
Convert roneneldan/TinyStories-Instruct-{1M,3M,8M} (GPT-Neo) into:
  - model_neo_q4.bin : CRDP v2 format (header + fp32 norms/biases/wpe + Q4_0)
  - tok_neo.bin      : CTK2 byte-level-BPE tokenizer blob (pruned vocab)

The GPT-2 vocab (50257) is pruned to the tokens actually used by the
TinyStories-Instruct dataset (~10K), which shrinks the embedding table ~5x
and the on-device logits buffer from 196 KB to ~40 KB. All 256 single-byte
tokens are always kept, so any input remains encodable.

Requires: huggingface_hub, tokenizers, torch, numpy
    pip install huggingface_hub tokenizers torch numpy
"""

import argparse
import struct
from pathlib import Path

import numpy as np

MAGIC = b"CRDP"
VERSION = 2
ARCH_GPTNEO = 2
QUANT_Q4_0 = 4
BLOCK_SIZE = 32
BYTES_PER_BLOCK = 18

TOK_MAGIC = 0x324B5443  # "CTK2" little-endian

GPT2_EOS = 50256

MODELS = {
    "1M": "roneneldan/TinyStories-Instruct-1M",
    "3M": "roneneldan/TinyStories-Instruct-3M",
    "8M": "roneneldan/TinyStories-Instruct-8M",
}

DATASET = ("roneneldan/TinyStoriesInstruct", "TinyStories-Instruct-valid.txt")


# ---------- Q4_0 quantization (same layout as convert_tinyllama_v0.py) ----------

def fp32_to_bf16_u16(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def quantize_q4_0(weights: np.ndarray) -> bytes:
    """Vectorized Q4_0: per 32-weight block, bf16 scale + 16 packed-nibble bytes."""
    flat = np.ascontiguousarray(weights, dtype=np.float32).reshape(-1)
    n = flat.size
    assert n % BLOCK_SIZE == 0, f"length {n} not divisible by {BLOCK_SIZE}"
    blk = flat.reshape(-1, BLOCK_SIZE)

    imax = np.abs(blk).argmax(axis=1)
    maxv = blk[np.arange(blk.shape[0]), imax]
    d = maxv / -8.0
    inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0)
    q = np.clip(np.floor(blk * inv[:, None] + 8.5).astype(np.int32), 0, 15).astype(np.uint8)

    scale = fp32_to_bf16_u16(d)
    packed = (q[:, :16] & 0x0F) | ((q[:, 16:] & 0x0F) << 4)

    out = np.zeros((blk.shape[0], BYTES_PER_BLOCK), dtype=np.uint8)
    out[:, 0] = scale & 0xFF
    out[:, 1] = scale >> 8
    out[:, 2:] = packed
    return out.tobytes()


# ---------- GPT-2 byte-level decoder (piece string -> raw bytes) ----------

def bytes_to_unicode():
    """The GPT-2 byte<->unicode bijection (from openai/gpt-2 encoder.py)."""
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("\xa1"), ord("\xac") + 1)) + \
         list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def piece_to_bytes(piece: str, uni2byte: dict) -> bytes:
    return bytes(uni2byte[ch] for ch in piece)


# ---------- Vocab pruning ----------

def build_kept_vocab(snap: Path, valid_txt: Path, min_count: int, extra_corpus=()):
    """Prune the GPT-2 vocab to tokens used by the dataset, closed under BPE
    derivation (every kept token's merge ancestors are kept too, otherwise the
    device couldn't reach it). Returns
    (old2new dict, pieces_bytes list, merges [(a,b,c) new ids], eos_new)."""
    import collections
    import json
    from tokenizers import ByteLevelBPETokenizer

    tok = ByteLevelBPETokenizer(str(snap / "vocab.json"), str(snap / "merges.txt"))
    text = valid_txt.read_text()
    for p in extra_corpus:
        # The raw BPE tokenizer has no special tokens; strip EOS literals so
        # they don't skew counts (EOS itself is always kept anyway).
        text += "\n" + Path(p).read_text().replace("<|endoftext|>", "\n")
    counts = collections.Counter()
    CH = 1_000_000
    for i in range(0, len(text), CH):
        counts.update(tok.encode(text[i:i + CH]).ids)
    print(f"[+] dataset sample: {len(text):,} chars, {len(counts):,} distinct tokens")

    vocab = json.loads((snap / "vocab.json").read_text())  # piece -> id
    id2piece = {v: k for k, v in vocab.items()}
    u2b = {v: k for k, v in bytes_to_unicode().items()}

    # merges.txt, rank order. In GPT-2 the resulting token id is 256 + rank,
    # so id order == merge-rank order; the device exploits this (it always
    # applies the adjacent pair with the lowest resulting id — exact BPE).
    merge_lines = (snap / "merges.txt").read_text().splitlines()
    if merge_lines and merge_lines[0].startswith("#"):
        merge_lines = merge_lines[1:]
    created_by = {}  # result_id -> (a_id, b_id)
    for rank, line in enumerate(merge_lines):
        a, b = line.split(" ")
        rid = vocab[a + b]
        assert rid == 256 + rank, f"id/rank mismatch at rank {rank}"
        created_by[rid] = (vocab[a], vocab[b])

    keep = {t for t, c in counts.items() if c >= min_count}
    for piece, tid in vocab.items():
        if len(piece) == 1:
            keep.add(tid)
    keep.add(GPT2_EOS)

    # Transitive closure over merge ancestors.
    stack = list(keep)
    while stack:
        t = stack.pop()
        if t in created_by:
            for p in created_by[t]:
                if p not in keep:
                    keep.add(p)
                    stack.append(p)

    kept = sorted(keep)  # preserves GPT-2 id order == rank order
    old2new = {old: new for new, old in enumerate(kept)}
    pieces = []
    for old in kept:
        if old == GPT2_EOS:
            pieces.append(b"<|endoftext|>")
        else:
            pieces.append(piece_to_bytes(id2piece[old], u2b))

    merges = sorted((old2new[a], old2new[b], old2new[r])
                    for r, (a, b) in created_by.items() if r in keep)
    print(f"[+] pruned vocab: {len(kept):,} tokens, {len(merges):,} merges "
          f"(min_count={min_count})")
    return old2new, pieces, merges, old2new[GPT2_EOS]


def write_tokenizer_blob(out: Path, pieces: list, merges: list, eos_new: int):
    """CTK2 blob:
       [u32 magic][i32 vocab][i32 max_len][i32 eos_id][i32 n_merges]
       [u16 byte_ids[256]]
       [n_merges x (u16 a, u16 b, u16 c)]   sorted by (a,b) for binary search
       then per token: [i32 len][bytes]"""
    byte_ids = [0xFFFF] * 256
    for i, p in enumerate(pieces):
        if len(p) == 1:
            byte_ids[p[0]] = i
    assert all(b != 0xFFFF for b in byte_ids), "missing single-byte token"
    assert len(pieces) < 0xFFFF and all(c < len(pieces) for _, _, c in merges)

    merges_ab = sorted(merges, key=lambda m: (m[0], m[1]))
    max_len = max(len(p) for p in pieces)
    with open(out, "wb") as f:
        f.write(struct.pack("<Iiiii", TOK_MAGIC, len(pieces), max_len,
                            eos_new, len(merges_ab)))
        f.write(struct.pack("<256H", *byte_ids))
        for a, b, c in merges_ab:
            f.write(struct.pack("<HHH", a, b, c))
        for p in pieces:
            f.write(struct.pack("<i", len(p)))
            f.write(p)
        sz = f.tell()
    print(f"[+] {out.name} = {sz:,} bytes ({sz / 1024:.1f} KB)")


# ---------- Model conversion ----------

def convert_model(snap: Path, out: Path, kept_old_ids, max_pos: int):
    import torch
    if (snap / "pytorch_model.bin").exists():
        sd = torch.load(snap / "pytorch_model.bin", map_location="cpu", weights_only=True)
    else:
        from safetensors.torch import load_file
        sd = load_file(snap / "model.safetensors")
    W = {k: v.float().numpy() for k, v in sd.items()}

    wte = W["transformer.wte.weight"]
    n_layers = max(int(k.split(".")[2]) for k in W if k.startswith("transformer.h.")) + 1
    dim = wte.shape[1]
    hidden = W["transformer.h.0.mlp.c_fc.weight"].shape[0]
    import json
    cfg = json.loads((snap / "config.json").read_text())
    n_heads = cfg["num_heads"]
    print(f"[+] GPT-Neo: dim={dim} hidden={hidden} layers={n_layers} heads={n_heads}")

    idx = np.asarray(kept_old_ids)
    wte_pruned = wte[idx]                       # [V', dim]
    wpe = W["transformer.wpe.weight"][:max_pos]  # [max_pos, dim]

    def L(l, name):
        return W[f"transformer.h.{l}.{name}"]

    def stack(name):
        return np.stack([L(l, name) for l in range(n_layers)])

    with open(out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<iiiiiii", dim, hidden, n_layers, n_heads,
                            n_heads, len(kept_old_ids), max_pos))
        f.write(struct.pack("<BBB", 1, QUANT_Q4_0, ARCH_GPTNEO))  # shared, quant, arch
        f.write(b"\x00" * (64 - f.tell()))

        # fp32 sections, order matched by llm.cpp::llm_init_embedded (v2)
        for t in (stack("ln_1.weight"), stack("ln_1.bias"),
                  stack("ln_2.weight"), stack("ln_2.bias"),
                  W["transformer.ln_f.weight"], W["transformer.ln_f.bias"],
                  stack("attn.attention.out_proj.bias"),
                  stack("mlp.c_fc.bias"), stack("mlp.c_proj.bias"),
                  wpe):
            f.write(np.ascontiguousarray(t, dtype=np.float32).tobytes())

        # Q4_0 sections
        f.write(quantize_q4_0(wte_pruned))  # also the classifier (tied)
        for name in ("attn.attention.q_proj.weight", "attn.attention.k_proj.weight",
                     "attn.attention.v_proj.weight", "attn.attention.out_proj.weight",
                     "mlp.c_fc.weight", "mlp.c_proj.weight"):
            f.write(quantize_q4_0(stack(name)))
        sz = f.tell()
    print(f"[+] {out.name} = {sz:,} bytes ({sz / 1024 / 1024:.2f} MB)")


# ---------- Encoding sanity check (device algorithm vs real BPE) ----------

def check_encoding(snap: Path, pieces: list, merges: list, old2new, samples: list):
    """Simulate the device's pair-table BPE and compare with HF BPE."""
    from tokenizers import ByteLevelBPETokenizer
    tok = ByteLevelBPETokenizer(str(snap / "vocab.json"), str(snap / "merges.txt"))
    byte_id = {p[0]: i for i, p in enumerate(pieces) if len(p) == 1}
    pair = {(a, b): c for a, b, c in merges}

    def device_encode(text: str):
        toks = [byte_id[b] for b in text.encode("utf-8")]
        while len(toks) > 1:
            best, best_idx = None, -1
            for i in range(len(toks) - 1):
                c = pair.get((toks[i], toks[i + 1]))
                if c is not None and (best is None or c < best):
                    best, best_idx = c, i
            if best is None:
                break
            toks[best_idx:best_idx + 2] = [best]
        return toks

    mismatches = 0
    for s in samples:
        ref = [old2new.get(t, -1) for t in tok.encode(s).ids]
        dev = device_encode(s)
        if ref != dev:
            mismatches += 1
            print(f"    encode mismatch on: {s!r}")
            print(f"      ref: {ref}\n      dev: {dev}")
    print(f"[+] encoding check: {len(samples) - mismatches}/{len(samples)} match")


# ---------- C++ embed emit (same as v1 converter) ----------

def emit_cpp_array(bin_path: Path, cpp_path: Path, sym: str):
    raw = bin_path.read_bytes()
    with open(cpp_path, "w") as f:
        f.write("// AUTO-GENERATED by tools/convert_tinystories_instruct.py. Do not edit.\n")
        f.write("#include <stdint.h>\n#include <stddef.h>\n\n")
        f.write(f'extern "C" const uint8_t {sym}[] __attribute__((aligned(4))) = {{\n')
        for i in range(0, len(raw), 16):
            f.write("  " + ",".join(f"0x{b:02x}" for b in raw[i:i + 16]) + ",\n")
        f.write("};\n\n")
        f.write(f'extern "C" const size_t {sym}_LEN = {len(raw)};\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="3M")
    ap.add_argument("--model-dir", default=None,
                    help="local HF checkpoint dir (e.g. data/chat_model) "
                         "instead of downloading --model")
    ap.add_argument("--corpus", action="append", default=[],
                    help="extra text file(s) for vocab-pruning counts "
                         "(e.g. data/chat_train.txt); repeatable")
    ap.add_argument("--min-count", type=int, default=3,
                    help="keep tokens appearing >= N times in the dataset sample")
    ap.add_argument("--max-pos", type=int, default=256,
                    help="position embeddings to keep (= max context)")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "cardputer_ai"))
    ap.add_argument("--keep-bin", action="store_true",
                    help="keep raw .bin files in embed/ (needed for tools/host test)")
    ap.add_argument("--no-cpp", action="store_true",
                    help="skip emitting model_data.cpp/tok_data.cpp")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download, hf_hub_download
    if args.model_dir:
        snap = Path(args.model_dir)
        print(f"[+] using local checkpoint {snap}")
    else:
        print(f"[+] fetching {MODELS[args.model]}")
        snap = Path(snapshot_download(repo_id=MODELS[args.model], cache_dir=args.cache,
                                      allow_patterns=["*.bin", "*.json", "merges.txt"]))
    valid = Path(hf_hub_download(repo_id=DATASET[0], repo_type="dataset",
                                 filename=DATASET[1], cache_dir=args.cache))

    out_dir = Path(args.out_dir)
    tmp = out_dir / "embed"
    tmp.mkdir(parents=True, exist_ok=True)

    old2new, pieces, merges, eos_new = build_kept_vocab(snap, valid, args.min_count,
                                                        args.corpus)
    kept = sorted(old2new, key=old2new.get)
    print(f"[+] eos id (pruned): {eos_new}")

    model_bin = tmp / "model_neo_q4.bin"
    tok_bin = tmp / "tok_neo.bin"
    convert_model(snap, model_bin, kept, args.max_pos)
    write_tokenizer_blob(tok_bin, pieces, merges, eos_new)

    check_encoding(snap, pieces, merges, old2new, [
        "Summary: a girl finds a lost cat and helps it find its way home.\nStory:",
        "Words: garden, rain, brave\nSummary: Tom plays outside.\nStory:",
        "Once upon a time there was a little dog named Spot.",
        'He said, "Yes, please. Read, please."',
    ])

    if not args.no_cpp:
        print("[+] emitting C++ embed sources")
        emit_cpp_array(model_bin, out_dir / "main" / "model_data.cpp", "MODEL_DATA")
        emit_cpp_array(tok_bin, out_dir / "main" / "tok_data.cpp", "TOKENIZER_DATA")

    if not args.keep_bin and not args.no_cpp:
        model_bin.unlink(); tok_bin.unlink()
        try: tmp.rmdir()
        except OSError: pass

    print("\nDone.")


if __name__ == "__main__":
    main()
