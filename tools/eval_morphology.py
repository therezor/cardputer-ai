#!/usr/bin/env python3
"""
Score the Ukrainian model's *morphology* — not just whether it says something,
but whether the word forms are real and whether they agree.

Ukrainian is morphologically rich: nouns inflect for seven cases, adjectives
agree with their noun in case/gender/number, and past-tense verbs agree with
the subject in gender. A small model can produce fluent-looking output that is
grammatically wrong in exactly these places, and loss alone will not tell you.

Two scores, both computed against VESUM (brown-uk/dict_uk), the reference
Ukrainian morphological dictionary:

  wordform validity  fraction of generated word forms that exist in VESUM.
                     Catches invented morphology — the "Яатиму" failure mode,
                     where the model splices a plausible stem onto a plausible
                     ending and produces a word that does not exist.

  adj-noun agreement of adjective+noun bigrams where both forms are known,
                     the fraction whose tag sets share a case, a number and
                     (in singular) a gender. This is the actual agreement
                     question, and the one a fluent-but-wrong model fails.

Proper nouns are excluded from the validity score: VESUM does not contain
transliterated foreign names, and 70% of out-of-dictionary words in the
subtitle corpus are exactly those.

Usage:
  ../cardputer_ai_venv/bin/python tools/eval_morphology.py \\
      --model embed/model_neo_q4.bin --tok embed/tok_neo.bin
"""

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WORD = re.compile(r"[а-щьюяїієґА-ЩЬЮЯЇІЄҐ'\-]+")
CASES = ["v_naz", "v_rod", "v_dav", "v_zna", "v_oru", "v_mis", "v_kly"]
GENDERS = ["m", "f", "n"]


def load_vesum(path: Path):
    """word form (lower) -> set of tag strings."""
    forms = defaultdict(set)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            forms[parts[0].lower()].add(parts[1].strip())
    return forms


def build_host(out: Path):
    cmd = ["clang++", "-std=c++17", "-O2", "-I", str(ROOT / "tools/host"),
           str(ROOT / "tools/host/host_test.cpp"), str(ROOT / "main/llm.cpp"),
           "-o", str(out)]
    subprocess.run(cmd, check=True)


def generate(host, model, tok, prompt, seed, max_new, kv):
    r = subprocess.run(
        [str(host), str(model), str(tok), prompt, "--max", str(max_new),
         "--temp", "0.8", "--top-p", "0.9", "--seed", str(seed), "--kv", str(kv)],
        capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        raise RuntimeError("host run failed")
    out = r.stdout.strip()
    # The harness prints a header line and a [done] trailer; keep the middle.
    out = re.sub(r"^arch=.*$", "", out, flags=re.M)
    out = re.sub(r"\[done\].*$", "", out, flags=re.S)
    return out.replace("<||>", " ").strip()


def tag_feats(tags):
    """(cases, numbers, genders) present across a form's tag set."""
    cases, nums, gens = set(), set(), set()
    for t in tags:
        for c in CASES:
            if c in t:
                cases.add(c)
        if ":p:" in t or t.endswith(":p"):
            nums.add("p")
        else:
            nums.add("s")
            for g in GENDERS:
                if f":{g}:" in t:
                    gens.add(g)
    return cases, nums, gens


def score(text, vesum):
    words = [w for w in WORD.findall(text) if len(w) > 1]
    known = invalid = 0
    proper_skipped = 0
    for w in words:
        lw = w.lower()
        if lw in vesum:
            known += 1
        elif w[:1].isupper():
            proper_skipped += 1        # almost certainly a name; not scored
        else:
            invalid += 1

    # Adjective-noun agreement over adjacent pairs.
    agree = total_pairs = 0
    for a, b in zip(words, words[1:]):
        ta, tb = vesum.get(a.lower()), vesum.get(b.lower())
        if not ta or not tb:
            continue
        if not any(t.startswith("adj") for t in ta):
            continue
        if not any(t.startswith("noun") for t in tb):
            continue
        ca, na, ga = tag_feats([t for t in ta if t.startswith("adj")])
        cb, nb, gb = tag_feats([t for t in tb if t.startswith("noun")])
        total_pairs += 1
        shared_num = na & nb
        ok = bool(ca & cb) and bool(shared_num)
        if ok and shared_num == {"s"}:
            ok = bool(ga & gb) if (ga and gb) else True
        agree += ok
    return known, invalid, proper_skipped, agree, total_pairs


# Weighted towards prompts that invite *description*, since adjective-noun
# pairs are the scarce measurement — a corpus of "Так." / "Добре." replies
# yields almost no scorable agreement.
PROMPTS = [
    "User: Привіт! Як твої справи?\\nBot:",
    "User: Розкажи про свій день.\\nBot:",
    "User: Що ти любиш робити?\\nBot:",
    "User: Мені сьогодні сумно.\\nBot:",
    "User: Яка сьогодні погода?\\nBot:",
    "User: Розкажи щось цікаве.\\nBot:",
    "User: Хто ти такий?\\nBot:",
    "User: Що ти думаєш про це?\\nBot:",
    "User: Опиши свою улюблену тварину.\\nBot:",
    "User: Який твій улюблений колір і чому?\\nBot:",
    "User: Розкажи про велике зелене дерево.\\nBot:",
    "User: Яка твоя найкраща подруга?\\nBot:",
    "User: Опиши гарний літній день.\\nBot:",
    "User: Розкажи казку про маленьку дівчинку.\\nBot:",
    "User: Що ти бачиш навколо себе?\\nBot:",
    "User: Розкажи про свою нову книгу.\\nBot:",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--vesum", default="data/ukr/raw/vesum.txt")
    ap.add_argument("--host-bin", default="/tmp/llm_host_morph")
    ap.add_argument("--samples", type=int, default=6, help="seeds per prompt")
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--kv", type=int, default=72)
    ap.add_argument("--show", type=int, default=8, help="replies to print")
    args = ap.parse_args()

    vesum = load_vesum(Path(args.vesum))
    print(f"[+] VESUM: {len(vesum):,} word forms")

    host = Path(args.host_bin)
    build_host(host)

    K = I = P = A = T = 0
    shown = 0
    for prompt in PROMPTS:
        for seed in range(args.samples):
            text = generate(host, args.model, args.tok, prompt, seed,
                            args.max_new, args.kv)
            if shown < args.show:
                print(f"  {prompt.split(chr(92)+'nBot:')[0][6:]:<28} -> {text}")
                shown += 1
            k, i, p, a, t = score(text, vesum)
            K += k; I += i; P += p; A += a; T += t

    total = K + I
    print()
    print(f"wordform validity : {K/total*100:.1f}%  ({K:,} valid / {I:,} invented"
          f", {P:,} proper nouns skipped)" if total else "wordform validity : n/a")
    print(f"adj-noun agreement: {A/T*100:.1f}%  ({A}/{T} scorable pairs)"
          if T else "adj-noun agreement: n/a (no scorable pairs)")


if __name__ == "__main__":
    main()
