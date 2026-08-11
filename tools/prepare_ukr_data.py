#!/usr/bin/env python3
"""
Build a Ukrainian pretraining + chat corpus for the cardputer_ai engine.

Unlike the English pipeline (tools/prepare_chat_data.py), which fine-tunes an
existing TinyStories model, the Ukrainian model has to be *pretrained*: the
pruned GPT-2 vocab is English-only and byte-level BPE spells Cyrillic out at
~2 tokens per character, so the tokenizer — and therefore the whole embedding
table — is replaced. That means we need two tiers of data:

  pretrain_{train,val}.txt   plain Ukrainian text, one utterance per line,
                             blank-line separated blocks — teaches the language
  chat_{train,val}.txt       "User: ...\\nBot: ...<|endoftext|>" samples in the
                             exact format main.cpp assembles — teaches the turn

Sources:
  - OPUS OpenSubtitles v2024, Ukrainian monolingual (15M lines, mean 4.3 words
    per line). Subtitles are the register we want: short, spoken, simple. Two
    consecutive lines are usually an exchange, which is where the dialogue
    pairs come from.
  - OPUS Tatoeba, Ukrainian monolingual: 186K hand-written simple sentences,
    mixed into the pretraining tier for clean grammar.
  - tools/ukr_qa_facts.py: hand-written kindergarten Q&A plus honest "не знаю"
    deflections. Subtitles never teach that a question has an answer.

The held-out split is taken as whole contiguous chunks of the source file, not
random lines: dialogue pairs are built from *adjacent* lines, so a random split
would leak the other half of a training pair into validation.

Usage:
  ../cardputer_ai_venv/bin/python tools/prepare_ukr_data.py
"""

import argparse
import random
import re
import unicodedata
from pathlib import Path

from ukr_qa_facts import FACTS, DEFLECTIONS, RECOVERIES, HARD_QUESTIONS

EOS = "<|endoftext|>"

# Ukrainian uses і ї є ґ and has no ы э ъ ё — those letters are a reliable
# marker of Russian text, of which OpenSubtitles has a fair amount.
RU_ONLY = set("ыэъёЫЭЪЁ")
CYR = re.compile(r"[Ѐ-ӿ]")
LAT = re.compile(r"[A-Za-z]")
DIGIT = re.compile(r"\d")

# Subtitle furniture: credits, sound effects, song markers, timing junk.
JUNK = re.compile(
    r"(субтитр|переклад|перекла[дв]|редагу|синхрон|opensubtitles|www\.|http|"
    r"\.com|\.net|\.org|title|sync by|corrected by|@)", re.I)

# Leading speaker dashes of every flavour subtitles use.
LEAD_DASH = re.compile(r"^[\-‐-―−]+\s*")
# Bracketed stage directions: [ПОСТРІЛИ], (сміх), ♪ lyrics ♪
BRACKET = re.compile(r"[\[\](){}<>♪♫#*_]")
WS = re.compile(r"\s+")

# The bundled u8g2 Cyrillic font has no U+2019/U+02BC, and the device keyboard
# can only produce ASCII punctuation, so fold every apostrophe to ASCII "'".
APOSTROPHES = {"’": "'", "ʼ": "'", "‘": "'", "´": "'",
               "`": "'", "ʹ": "'", "′": "'"}
DASHES = {"–": "-", "—": "-", "―": "-", "−": "-"}
QUOTES = {"“": '"', "”": '"', "„": '"', "«": '"',
          "»": '"', "″": '"'}
ELLIPSIS = {"…": "..."}
FOLD = {**APOSTROPHES, **DASHES, **QUOTES, **ELLIPSIS}
FOLD_TAB = str.maketrans(FOLD)

# Subtitle sources are inconsistent about spacing around punctuation
# ("Чому ми не їдемо ?"). Left alone it becomes a separate BPE token and the
# model learns to emit it, which reads as broken Ukrainian on the device.
SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:])")
SPACE_IN_ELLIPSIS = re.compile(r"\.\s+\.")


def clean_line(raw: str):
    """Normalise one subtitle line.

    Returns (text, had_speaker_dash) or None if the line should be dropped.
    The dash flag has to come out of here rather than be recovered later: it is
    the subtitle convention for "a different person is speaking now", and it is
    the strongest dialogue-coherence signal in the corpus (16% of pairs). It is
    stripped from the text, so once this function returns it is unrecoverable.

    Rejections are ordered cheapest-first — this runs 15M times."""
    s = raw.strip()
    if not s:
        return None
    had_dash = bool(LEAD_DASH.match(s))
    s = LEAD_DASH.sub("", s)
    if not s:
        return None
    if BRACKET.search(s) or JUNK.search(s):
        return None
    s = unicodedata.normalize("NFC", s).translate(FOLD_TAB)
    s = WS.sub(" ", s).strip()
    s = SPACE_IN_ELLIPSIS.sub("..", SPACE_IN_ELLIPSIS.sub("..", s))
    s = SPACE_BEFORE_PUNCT.sub(r"\1", s)
    if not s:
        return None

    n_words = s.count(" ") + 1
    if n_words < 2 or n_words > 12 or len(s) > 90:
        return None
    if any(c in RU_ONLY for c in s):
        return None
    if DIGIT.search(s) or LAT.search(s):
        # Digits and Latin letters are almost always transliterated names,
        # timestamps or credits, and they eat vocabulary a tiny model needs.
        return None

    letters = CYR.findall(s)
    if len(letters) < 0.6 * len(s.replace(" ", "")):
        return None
    # All-caps lines are shouted signs and titles, not speech.
    if s.upper() == s and len(letters) > 3:
        return None
    # Keep only lines that end like an utterance.
    if s[-1] not in ".!?,:;'\"" and not s[-1].isalpha():
        return None
    return s, had_dash


# Sentence boundary: terminator, space, capital letter. Crude but the prose
# tier only needs well-formed spans, not a perfect segmentation.
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[А-ЩЬЮЯЇІЄҐ\"'])")


def clean_prose_sentence(s: str, max_words: int):
    """Filter for the prose tier.

    Subtitle lines average 4.3 words, and a four-word fragment ("Добре.",
    "Звичайно.") carries no agreement relation at all. Ukrainian morphology
    lives in adjective-noun concord, past-tense subject-verb gender agreement
    and preposition-case governance — relations that need complete clauses. So
    prose is admitted with a much longer word limit than dialogue."""
    s = s.strip()
    if not s:
        return None
    if BRACKET.search(s) or JUNK.search(s):
        return None
    s = unicodedata.normalize("NFC", s).translate(FOLD_TAB)
    s = WS.sub(" ", s).strip()
    s = SPACE_BEFORE_PUNCT.sub(r"\1", s)
    n_words = s.count(" ") + 1
    if n_words < 5 or n_words > max_words:
        return None
    if any(c in RU_ONLY for c in s):
        return None
    if DIGIT.search(s) or LAT.search(s):
        return None
    letters = CYR.findall(s)
    if len(letters) < 0.65 * len(s.replace(" ", "")):
        return None
    # A complete clause: starts with a capital, ends with a terminator. Without
    # the capital check, hard-wrapped source lines leak in as mid-sentence
    # fragments ("народження або в дуже ранньому віці.") — which teach broken
    # syntax, the opposite of why the prose tier exists.
    if s[-1] not in ".!?" or not s[0].isupper():
        return None
    # All-caps news headlines are grammatical but teach the model to shout.
    # Only ~0.3% of the prose tier, so this costs nothing.
    if s.upper() == s:
        return None
    return s


def read_prose(shards, max_sentences: int, max_words: int, val_every: int):
    """Yield (sentence, is_val) from malyuk parquet shards."""
    import pyarrow.parquet as pq

    n = 0
    for si, shard in enumerate(shards):
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=512, columns=["text"]):
            for doc_i, doc in enumerate(batch.column("text").to_pylist()):
                # Hold out whole documents, not sentences.
                is_val = (n // 5000) % val_every == 0
                # malyuk documents are hard-wrapped mid-sentence, so single
                # newlines are line breaks, not boundaries — join them back
                # before splitting, or every wrap becomes a false sentence end.
                for para in re.split(r"\n\s*\n", doc):
                    para = WS.sub(" ", para)
                    for sent in SENT_SPLIT.split(para):
                        s = clean_prose_sentence(sent, max_words)
                        if s is None:
                            continue
                        yield s, is_val
                        n += 1
                        if n >= max_sentences:
                            return


def read_source(path: Path, val_every: int, chunk: int):
    """Yield (line_idx, text, had_dash, is_val) for cleaned lines.

    `is_val` is assigned per contiguous chunk of `chunk` source lines so that
    adjacent-line dialogue pairs never straddle the train/val boundary."""
    with path.open(encoding="utf-8", errors="replace") as f:
        for i, raw in enumerate(f):
            is_val = (i // chunk) % val_every == 0
            got = clean_line(raw)
            if got is not None:
                yield i, got[0], got[1], is_val


def build(args):
    rng = random.Random(args.seed)
    raw_dir = Path(args.raw_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    subs = raw_dir / "opensubtitles_uk.txt"
    if not subs.exists():
        raise SystemExit(
            f"missing {subs}\n"
            "  curl -L -o data/ukr/raw/opensubtitles_uk.txt.gz \\\n"
            "    https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono/uk.txt.gz")

    # ---- pass over the subtitles: pretrain blocks + dialogue pairs ----
    # The pretraining tier is written incrementally rather than accumulated:
    # at 24 prose shards it is ~3 GB of text, which as Python strings would be
    # several times that in RAM. The chat tier still buffers (it is ~150 MB and
    # has to be shuffled).
    out.mkdir(parents=True, exist_ok=True)
    pre_files = {sp: (out / f"pretrain_{sp}.txt").open("w", encoding="utf-8")
                 for sp in ("train", "val")}
    pre_counts = {"train": 0, "val": 0}

    class _PreTier:
        """Minimal shim so existing call sites keep reading as appends."""
        def __init__(self, split): self.split = split
        def append(self, text):
            pre_files[self.split].write(text + "\n\n")
            pre_counts[self.split] += 1
        def __len__(self): return pre_counts[self.split]

    pre = {"train": _PreTier("train"), "val": _PreTier("val")}
    chats = {"train": [], "val": []}
    block = []            # current run of consecutive kept lines
    block_val = False
    prev_idx = -10
    kept = seen = 0

    def flush_block():
        """A run of adjacent clean lines becomes one pretraining block and a
        few chat samples of 1-3 exchanges."""
        nonlocal block
        if len(block) >= 2:
            split = "val" if block_val else "train"
            pre[split].append("\n".join(t for t, _ in block))
            # Dialogue samples: walk the run in strides of 2 lines per
            # exchange, grouping 1-3 exchanges into one sample.
            i = 0
            while i + 1 < len(block):
                n_ex = rng.choice((1, 1, 1, 2, 2, 3))
                turns = []
                while n_ex > 0 and i + 1 < len(block):
                    u_text, _ = block[i]
                    b_text, b_dash = block[i + 1]
                    # Only keep exchanges with evidence that the second line is
                    # a *reply*: either the first line asks a question, or the
                    # subtitle marked a speaker change on the second. Adjacent
                    # lines without either are usually one person continuing,
                    # and training on them teaches non-sequiturs.
                    if b_dash or u_text.endswith("?"):
                        turns.append((u_text, b_text))
                    i += 2
                    n_ex -= 1
                if turns and (len(chats[split]) < args.max_chat or split == "val"):
                    chats[split].append(turns)
        block = []

    for idx, text, had_dash, is_val in read_source(subs, args.val_every, args.val_chunk):
        seen += 1
        if idx != prev_idx + 1 or is_val != block_val or len(block) >= args.block_lines:
            flush_block()
            block_val = is_val
        block.append((text, had_dash))
        prev_idx = idx
        kept += 1
        if args.limit and seen >= args.limit:
            break
    flush_block()

    print(f"[+] subtitles: kept {kept:,} lines")
    print(f"[+] pretrain blocks: {len(pre['train']):,} train / {len(pre['val']):,} val")
    print(f"[+] subtitle chats:  {len(chats['train']):,} train / {len(chats['val']):,} val")

    # ---- Tatoeba: clean hand-written sentences, pretraining tier only ----
    tat = raw_dir / "tatoeba_uk.txt"
    if tat.exists():
        buf = []
        n_tat = 0
        for _, text, _dash, is_val in read_source(tat, args.val_every, args.val_chunk):
            buf.append(text)
            n_tat += 1
            if len(buf) >= args.block_lines:
                pre["val" if is_val else "train"].append("\n".join(buf))
                buf = []
        if buf:
            pre["train"].append("\n".join(buf))
        print(f"[+] tatoeba: {n_tat:,} sentences")

    # ---- prose tier: complete clauses, pretraining only ----
    shards = sorted(raw_dir.glob("malyuk_*.parquet"))
    if shards and args.prose_sentences:
        buf, n_prose = [], 0
        for sent, is_val in read_prose(shards, args.prose_sentences,
                                       args.prose_max_words, args.val_every):
            buf.append(sent)
            n_prose += 1
            if len(buf) >= args.block_lines:
                pre["val" if is_val else "train"].append("\n".join(buf))
                buf = []
        if buf:
            pre["train"].append("\n".join(buf))
        print(f"[+] prose: {n_prose:,} sentences from {len(shards)} shard(s)")
    elif args.prose_sentences:
        print("[!] no malyuk_*.parquet in raw dir — prose tier skipped")

    # ---- hand-written Q&A and deflections ----
    def user_variant(q: str, rng: random.Random) -> str:
        """Case/punctuation variant of a hand-written *user* turn.

        The model binds to exact surface forms: with only the capitalised
        "Яка столиця України?" in the corpus, typing "яка столиця україни"
        produced gibberish, and "скільки днів у тижні" produced nothing at all.
        People type lowercase and drop the question mark, so every hand-written
        question is taught in those shapes too. The *reply* is never altered —
        the bot should always answer in well-formed Ukrainian."""
        r = rng.random()
        if r < 0.40:
            return q
        if r < 0.75:
            return q.lower()
        if r < 0.90:
            return q.lower().rstrip("?!.")
        return q.rstrip("?!.")

    qa = []
    for q, a in FACTS:
        qa.append([(q, a)])
    # Repeat the facts so they survive against millions of subtitle lines, and
    # pair some of them into two-turn samples for context robustness.
    facts_rep = []
    for _ in range(args.qa_repeat):
        for q, a in FACTS:
            if rng.random() < 0.25:
                q2, a2 = rng.choice(FACTS)
                facts_rep.append([(user_variant(q, rng), a),
                                  (user_variant(q2, rng), a2)])
            else:
                facts_rep.append([(user_variant(q, rng), a)])

    idk = []
    for _ in range(args.idk_repeat):
        for hq in HARD_QUESTIONS:
            reply = rng.choice(DEFLECTIONS)
            if rng.random() < 0.4:
                reply += " " + rng.choice(RECOVERIES)
                q2, a2 = rng.choice(FACTS)
                idk.append([(user_variant(hq, rng), reply),
                            (user_variant(q2, rng), a2)])
            else:
                idk.append([(user_variant(hq, rng), reply)])

    hand = facts_rep + idk
    rng.shuffle(hand)
    n_hand_val = max(1, len(hand) // 100)
    chats["val"].extend(hand[:n_hand_val])
    chats["train"].extend(hand[n_hand_val:])
    print(f"[+] hand-written: {len(facts_rep):,} fact samples, {len(idk):,} IDK samples")

    # ---- write ----
    def sample_text(turns):
        out_lines = []
        for u, b in turns:
            out_lines.append("User: " + u)
            out_lines.append("Bot: " + b + EOS)
        return "\n".join(out_lines)

    for split in ("train", "val"):
        rng.shuffle(chats[split])
        p = out / f"chat_{split}.txt"
        p.write_text("\n\n".join(sample_text(t) for t in chats[split]) + "\n",
                     encoding="utf-8")
        print(f"[+] {p} = {p.stat().st_size / 1e6:.1f} MB, {len(chats[split]):,} samples")

        pre_files[split].close()
        p = out / f"pretrain_{split}.txt"
        print(f"[+] {p} = {p.stat().st_size / 1e6:.1f} MB, {len(pre[split]):,} blocks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/ukr/raw")
    ap.add_argument("--out-dir", default="data/ukr")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N *kept* subtitle lines (smoke tests)")
    ap.add_argument("--block-lines", type=int, default=24,
                    help="max consecutive lines per pretraining block")
    ap.add_argument("--val-every", type=int, default=60,
                    help="hold out one chunk in N")
    ap.add_argument("--val-chunk", type=int, default=20000,
                    help="source lines per contiguous train/val chunk")
    ap.add_argument("--max-chat", type=int, default=2_000_000,
                    help="cap on subtitle-derived chat samples")
    ap.add_argument("--prose-sentences", type=int, default=4_000_000,
                    help="complete-clause sentences from lang-uk/malyuk to mix "
                         "into the pretraining tier (0 disables). Subtitle "
                         "fragments barely exercise agreement; prose does")
    ap.add_argument("--prose-max-words", type=int, default=25)
    ap.add_argument("--qa-repeat", type=int, default=400,
                    help="times to repeat the hand-written fact set")
    ap.add_argument("--idk-repeat", type=int, default=300,
                    help="times to repeat the hard-question set")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
