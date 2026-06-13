#!/usr/bin/env python3
"""
Convert Maykeye/TinyLLama-v0 from HuggingFace into:
  - model_q4.bin   : our custom Q4_0 format (header + fp32 norms + Q4_0 weights)
  - tok32000.bin   : llama2.c-style binary tokenizer

The output files are intended to be embedded into the Arduino firmware via
.incbin (see model_data.S) so the launcher flashes app+model as one image.

Requires: huggingface_hub, safetensors, sentencepiece, numpy
    pip install huggingface_hub safetensors sentencepiece numpy
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"CRDP"
VERSION = 1
QUANT_Q4_0 = 4
BLOCK_SIZE = 32          # weights per Q4_0 block
BYTES_PER_BLOCK = 18     # 2B bf16 scale + 16B packed nibbles


# ---------- Q4_0 quantization (llama.cpp-compatible nibble layout) ----------

def fp32_to_bf16_bytes(x: float) -> bytes:
    """Round-to-nearest-even bf16 (top 16 bits of fp32, with sticky rounding)."""
    u = np.float32(x).view(np.uint32).item()
    # round-to-nearest-even
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return struct.pack("<H", (rounded >> 16) & 0xFFFF)


def quantize_q4_0(weights: np.ndarray) -> bytes:
    """
    Quantize a flat fp32 array (length must be multiple of 32) to Q4_0.
    Layout per block: [bf16 scale][16 bytes nibbles] where byte k has
    low_nibble = q[k], high_nibble = q[k+16].
    """
    assert weights.dtype == np.float32
    n = weights.size
    assert n % BLOCK_SIZE == 0, f"length {n} not divisible by {BLOCK_SIZE}"

    out = bytearray()
    nblocks = n // BLOCK_SIZE
    flat = weights.reshape(nblocks, BLOCK_SIZE)

    for blk in flat:
        amax = np.abs(blk).max()
        if amax == 0.0:
            d = 0.0
            qs = np.zeros(BLOCK_SIZE, dtype=np.uint8)
        else:
            # llama.cpp Q4_0: d = max / -8, where max is the value with largest |x|
            imax = np.argmax(np.abs(blk))
            d = float(blk[imax]) / -8.0
            inv = 1.0 / d if d != 0 else 0.0
            q = np.clip(np.floor(blk * inv + 8.5).astype(np.int32), 0, 15).astype(np.uint8)
            qs = q

        out += fp32_to_bf16_bytes(d)
        packed = bytearray(16)
        for k in range(16):
            packed[k] = (qs[k] & 0x0F) | ((qs[k + 16] & 0x0F) << 4)
        out += bytes(packed)

    return bytes(out)


def write_q4_tensor(fh, t: np.ndarray):
    """Quantize tensor (any shape) row-major and append to file."""
    flat = np.ascontiguousarray(t, dtype=np.float32).reshape(-1)
    fh.write(quantize_q4_0(flat))


def write_f32_tensor(fh, t: np.ndarray):
    fh.write(np.ascontiguousarray(t, dtype=np.float32).tobytes())


# ---------- Model conversion ----------

_SAFETENSORS_DTYPES = {
    "F64": (np.float64, 8), "F32": (np.float32, 4), "F16": (np.float16, 2),
    "BF16": (np.uint16, 2),  # no native numpy bf16; widen to fp32 below
    "I64": (np.int64, 8), "I32": (np.int32, 4), "I16": (np.int16, 2),
    "I8": (np.int8, 1), "U8": (np.uint8, 1),
}


def read_safetensors(path: Path):
    """Parse a .safetensors file without torch (handles BF16, which the
    numpy backend of `safetensors` cannot). Format: <u64 header_len><JSON
    header>{raw tensor data}; header maps name -> dtype/shape/data_offsets."""
    import json
    with open(path, "rb") as fh:
        (hdr_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hdr_len))
        data = fh.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype, _ = _SAFETENSORS_DTYPES[info["dtype"]]
        beg, end = info["data_offsets"]
        t = np.frombuffer(data[beg:end], dtype=dtype).reshape(info["shape"])
        if info["dtype"] == "BF16":
            t = (t.astype(np.uint32) << 16).view(np.float32)
        out[name] = t.astype(np.float32)
    return out


def load_hf_weights(model_dir: Path):
    """Return dict[str, np.ndarray(fp32)] from HF safetensors checkpoint."""
    weights = {}
    found = False
    for f in sorted(model_dir.glob("*.safetensors")):
        found = True
        weights.update(read_safetensors(f))
    if not found:
        # fall back to pytorch_model.bin
        import torch
        for f in sorted(model_dir.glob("*.bin")):
            d = torch.load(f, map_location="cpu")
            for k, v in d.items():
                weights[k] = v.float().numpy()
            found = True
            break
    if not found:
        raise SystemExit(f"no weight files found in {model_dir}")
    return weights


def convert_model(model_dir: Path, out_bin: Path):
    print(f"[+] loading weights from {model_dir}")
    W = load_hf_weights(model_dir)

    # Architecture constants for Maykeye/TinyLLama-v0
    dim = 64
    hidden_dim = 256
    n_layers = 8
    n_heads = 16
    n_kv_heads = 16
    vocab_size = 32000
    seq_len = 2048
    shared = 0  # tie_word_embeddings = false in config.json

    head_size = dim // n_heads
    assert head_size == 4

    def grab(key):
        if key not in W:
            raise KeyError(f"missing weight {key}; have e.g. {list(W)[:5]}")
        return W[key]

    # Layer naming used by HF LLaMA architecture:
    # model.embed_tokens.weight
    # model.layers.{l}.input_layernorm.weight       (rms_att)
    # model.layers.{l}.self_attn.{q,k,v,o}_proj.weight
    # model.layers.{l}.post_attention_layernorm.weight  (rms_ffn)
    # model.layers.{l}.mlp.{gate,up,down}_proj.weight   (w1, w3, w2)
    # model.norm.weight
    # lm_head.weight

    embed = grab("model.embed_tokens.weight")          # [V, dim]
    final_norm = grab("model.norm.weight")             # [dim]
    lm_head = grab("lm_head.weight")                   # [V, dim]

    rms_att = np.stack([grab(f"model.layers.{l}.input_layernorm.weight") for l in range(n_layers)])
    rms_ffn = np.stack([grab(f"model.layers.{l}.post_attention_layernorm.weight") for l in range(n_layers)])

    wq = np.stack([grab(f"model.layers.{l}.self_attn.q_proj.weight") for l in range(n_layers)])
    wk = np.stack([grab(f"model.layers.{l}.self_attn.k_proj.weight") for l in range(n_layers)])
    wv = np.stack([grab(f"model.layers.{l}.self_attn.v_proj.weight") for l in range(n_layers)])
    wo = np.stack([grab(f"model.layers.{l}.self_attn.o_proj.weight") for l in range(n_layers)])
    w1 = np.stack([grab(f"model.layers.{l}.mlp.gate_proj.weight") for l in range(n_layers)])  # [hidden, dim]
    w3 = np.stack([grab(f"model.layers.{l}.mlp.up_proj.weight")   for l in range(n_layers)])
    w2 = np.stack([grab(f"model.layers.{l}.mlp.down_proj.weight") for l in range(n_layers)])  # [dim, hidden]

    print(f"[+] writing {out_bin}")
    with open(out_bin, "wb") as f:
        # ---- header (64 bytes) ----
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<iiiiiii", dim, hidden_dim, n_layers,
                            n_heads, n_kv_heads, vocab_size, seq_len))
        f.write(struct.pack("<BB", shared, QUANT_Q4_0))
        # pad to 64 bytes
        pad = 64 - f.tell()
        assert pad >= 0
        f.write(b"\x00" * pad)

        # ---- fp32 norms ----
        write_f32_tensor(f, rms_att)
        write_f32_tensor(f, rms_ffn)
        write_f32_tensor(f, final_norm)

        # ---- Q4_0 weight tensors (in fixed order matched by llm.cpp) ----
        write_q4_tensor(f, embed)   # token_embedding
        write_q4_tensor(f, wq)
        write_q4_tensor(f, wk)
        write_q4_tensor(f, wv)
        write_q4_tensor(f, wo)
        write_q4_tensor(f, w1)
        write_q4_tensor(f, w2)
        write_q4_tensor(f, w3)
        write_q4_tensor(f, lm_head) # wcls

        sz = f.tell()

    print(f"[+] model_q4.bin = {sz:,} bytes ({sz / 1024 / 1024:.2f} MB)")


# ---------- Tokenizer conversion (SentencePiece → llama2.c format) ----------

def convert_tokenizer(model_dir: Path, out_bin: Path):
    """Export SentencePiece tokenizer to the binary format llm.cpp reads."""
    import sentencepiece as spm

    sp_path = None
    for cand in ["tokenizer.model", "spm.model"]:
        p = model_dir / cand
        if p.exists():
            sp_path = p
            break
    if sp_path is None:
        raise SystemExit(f"no tokenizer.model in {model_dir}")

    sp = spm.SentencePieceProcessor(model_file=str(sp_path))
    vocab_size = sp.get_piece_size()
    print(f"[+] tokenizer vocab_size = {vocab_size}")

    tokens = []
    scores = []
    for i in range(vocab_size):
        t = sp.id_to_piece(i)
        s = sp.get_score(i)
        # SentencePiece uses U+2581 ('▁') for word-leading space; convert to ' '.
        t = t.replace("▁", " ")
        # Byte-fallback tokens look like "<0x41>" — keep them as-is; the runtime
        # decoder already handles that pattern.
        tokens.append(t.encode("utf-8"))
        scores.append(float(s))

    max_len = max(len(t) for t in tokens)

    with open(out_bin, "wb") as f:
        f.write(struct.pack("<I", max_len))
        for t, s in zip(tokens, scores):
            f.write(struct.pack("<f", s))
            f.write(struct.pack("<i", len(t)))
            f.write(t)
        sz = f.tell()

    print(f"[+] tok32000.bin = {sz:,} bytes ({sz / 1024:.1f} KB)")


# ---------- HF download ----------

def fetch_model(cache: Path) -> Path:
    """Download Maykeye/TinyLLama-v0 to `cache/`. Returns local dir."""
    from huggingface_hub import snapshot_download
    print("[+] downloading Maykeye/TinyLLama-v0 from HuggingFace")
    local = snapshot_download(
        repo_id="Maykeye/TinyLLama-v0",
        cache_dir=str(cache),
        allow_patterns=["*.safetensors", "*.bin", "*.json", "tokenizer.model", "*.model"],
    )
    return Path(local)


def emit_cpp_array(bin_path: Path, cpp_path: Path, sym: str):
    """Write a C++ source file with `const uint8_t SYM[] = { ... };` and
    `const size_t SYM_LEN = ...;` compiled into the firmware as .rodata.
    Uses 12 bytes per line — good readability without bloating compile time
    relative to longer lines (GCC parses fine either way).
    """
    raw = bin_path.read_bytes()
    with open(cpp_path, "w") as f:
        f.write("// AUTO-GENERATED by tools/convert_tinyllama_v0.py. Do not edit.\n")
        f.write("// Regenerate by running:  python tools/convert_tinyllama_v0.py\n")
        f.write("#include <stdint.h>\n#include <stddef.h>\n\n")
        f.write(f'extern "C" const uint8_t {sym}[] __attribute__((aligned(4))) = {{\n')
        per_line = 16
        for i in range(0, len(raw), per_line):
            chunk = raw[i:i+per_line]
            f.write("  " + ",".join(f"0x{b:02x}" for b in chunk) + ",\n")
        f.write("};\n\n")
        f.write(f'extern "C" const size_t {sym}_LEN = {len(raw)};\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent),
                    help="repo root — model_data.cpp + tok_data.cpp drop into <out-dir>/main/")
    ap.add_argument("--cache",   default=str(Path.home() / ".cache" / "cardputer_ai"),
                    help="HuggingFace download cache dir")
    ap.add_argument("--model-dir", default=None,
                    help="skip download; use this local HF snapshot directory")
    ap.add_argument("--keep-bin", action="store_true",
                    help="also keep the raw .bin files next to the .cpp output")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "embed"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = fetch_model(Path(args.cache))

    model_bin = tmp_dir / "model_q4.bin"
    tok_bin   = tmp_dir / "tok32000.bin"
    convert_model(model_dir, model_bin)
    convert_tokenizer(model_dir, tok_bin)

    print("[+] emitting C++ embed sources")
    emit_cpp_array(model_bin, out_dir / "main" / "model_data.cpp", "MODEL_DATA")
    emit_cpp_array(tok_bin,   out_dir / "main" / "tok_data.cpp",   "TOKENIZER_DATA")

    if not args.keep_bin:
        model_bin.unlink()
        tok_bin.unlink()
        try: tmp_dir.rmdir()
        except OSError: pass

    print()
    print("Files written into the IDF main component:")
    print(f"  {out_dir / 'main' / 'model_data.cpp'}")
    print(f"  {out_dir / 'main' / 'tok_data.cpp'}")
    print("Rebuild with `pio run` (or `idf.py build`).")


if __name__ == "__main__":
    main()
