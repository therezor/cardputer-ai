#!/usr/bin/env python3
"""Run the fixed prompt battery (tools/eval_prompts.txt) through the host
harness for one or more model/tokenizer pairs and score the results.

Scores:
  fact  — expected keyword present in the reply
  idk   — reply contains a deflection phrase ("I don't know" behavior)
  chat  — reply does NOT deflect (over-refusal check; should stay ~100%)

Each prompt runs twice: greedy (--temp 0, deterministic diff between models)
and sampled (--temp 0.8 --top-p 0.9 --seed 42).

Usage (compare shipped blobs against freshly converted ones):
  python tools/eval_battery.py \
      --model old=embed_old/model_neo_q4.bin,embed_old/tok_neo.bin \
      --model new=embed/model_neo_q4.bin,embed/tok_neo.bin
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFLECT_MARKERS = {
    "en": [
        "don't know", "do not know", "not sure", "too hard", "too tricky",
        "tricky for me", "little bot", "tiny bot", "small bot", "can't answer",
        "cannot answer", "wish i knew", "too little to know", "big question",
    ],
    # Ukrainian phrasings from tools/ukr_qa_facts.py DEFLECTIONS. Matched on the
    # lowercased reply, and kept to stems so inflected forms still hit.
    "ukr": [
        "не знаю", "не знав", "не вивчив", "не впевнен", "не можу сказати",
        "занадто складно", "дуже маленький бот", "маленький бот",
        "тільки прості речі", "вибач",
    ],
}


def build_host(out: Path):
    cmd = ["clang++", "-std=c++17", "-O2", "-I", str(ROOT / "tools/host"),
           str(ROOT / "tools/host/host_test.cpp"), str(ROOT / "main/llm.cpp"),
           "-o", str(out)]
    print("[+] building host harness:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_host(host, model, tok, prompt, temp, top_p, seed, max_new=48):
    cmd = [str(host), str(model), str(tok), prompt,
           "--max", str(max_new), "--temp", str(temp),
           "--top-p", str(top_p), "--seed", str(seed), "--kv", "80"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"host run failed for prompt: {prompt[:40]}")
    return r.stdout.strip()


def load_prompts(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        typ, text = parts[0], parts[1]
        expect = [e.strip().lower() for e in parts[2].split(",")] if len(parts) > 2 else []
        rows.append((typ, text, expect))
    return rows


def deflects(reply, lang):
    r = reply.lower()
    return any(m in r for m in DEFLECT_MARKERS[lang])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    metavar="NAME=model.bin,tok.bin",
                    help="repeatable; each is run over the whole battery")
    ap.add_argument("--prompts", default=str(ROOT / "tools/eval_prompts.txt"))
    ap.add_argument("--lang", choices=sorted(DEFLECT_MARKERS), default="en",
                    help="which deflection vocabulary scores idk/chat")
    ap.add_argument("--host-bin", default="/tmp/llm_host_eval")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    host = Path(args.host_bin)
    build_host(host)

    models = []
    for spec in args.model:
        name, files = spec.split("=", 1)
        mbin, tbin = files.split(",")
        models.append((name, Path(mbin), Path(tbin)))

    rows = load_prompts(Path(args.prompts))
    scores = {name: {"fact": [0, 0], "idk": [0, 0], "chat": [0, 0]}
              for name, _, _ in models}

    for typ, text, expect in rows:
        if typ in ("chat", "fact", "idk"):
            prompt = f"User: {text}\\nBot:"
        else:
            prompt = text
        print(f"\n=== [{typ}] {text}" + (f"  (expect: {', '.join(expect)})" if expect else ""))
        for name, mbin, tbin in models:
            greedy = run_host(host, mbin, tbin, prompt, 0.0, 1.0, args.seed)
            sampled = run_host(host, mbin, tbin, prompt, 0.8, 0.9, args.seed)
            verdict = ""
            if typ == "fact":
                ok = any(e in greedy.lower() for e in expect)
                scores[name]["fact"][0] += ok
                scores[name]["fact"][1] += 1
                verdict = "PASS" if ok else "FAIL"
            elif typ == "idk":
                ok = deflects(greedy, args.lang)
                scores[name]["idk"][0] += ok
                scores[name]["idk"][1] += 1
                verdict = "PASS" if ok else "FAIL"
            elif typ == "chat":
                ok = not deflects(greedy, args.lang)
                scores[name]["chat"][0] += ok
                scores[name]["chat"][1] += 1
                verdict = "PASS" if ok else "OVER-REFUSAL"
            print(f"  {name:>8} greedy : {greedy}" + (f"   [{verdict}]" if verdict else ""))
            print(f"  {name:>8} sampled: {sampled}")

    print("\n===== summary (scored on greedy runs) =====")
    print(f"{'model':>10}  {'facts':>10}  {'idk-hard':>10}  {'no-over-refusal':>16}")
    for name, _, _ in models:
        s = scores[name]
        def pct(k):
            got, tot = s[k]
            return f"{got}/{tot}" if tot else "-"
        print(f"{name:>10}  {pct('fact'):>10}  {pct('idk'):>10}  {pct('chat'):>16}")


if __name__ == "__main__":
    main()
