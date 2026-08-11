#!/usr/bin/env python3
"""Assemble the browser demo, in two shapes from one template.

  dist/index.html        everything inlined (WASM as JS, weights as base64).
                         This is the claude.ai artifact build: a strict CSP
                         blocks every external request, so nothing may be
                         fetched — and the artifact host supplies
                         <!doctype>/<head>/<body>, so this file has none.

  dist/site/             an ordinary website: a real HTML document plus
                         llm.js, model_neo_q4.bin and tok_neo.bin as separate
                         files, fetched at boot. Upload the directory as-is.

Keeping the artifact build at dist/index.html matters: republishing an
artifact from a different path mints a new URL.

Usage: python3 tools/web/make_page.py
"""

import argparse
import base64
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# The site build is a complete document. `lang="uk"` so spellcheck and
# hyphenation behave; the charset declaration must land inside the first 1024
# bytes or the browser ignores it and renders the Cyrillic as mojibake — which
# is exactly what happens when a host serves `text/html` with no charset.
SHELL = """<!doctype html>
<html lang="uk">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
</head>
<body>
__BODY__
</body>
</html>
"""

# Fetched in parallel; the model is ~20x the tokenizer, so it owns the progress bar.
FETCH_ASSETS = """(async (onProgress) => {
    const [model, tok] = await Promise.all([
      fetchBytes("model_neo_q4.bin", onProgress),
      fetchBytes("tok_neo.bin", () => {}),
    ]);
    return { model, tok };
  })"""

INLINE_ASSETS = """(async () => ({
    model: b64ToBytes("__MODEL_B64__"),
    tok: b64ToBytes("__TOK_B64__"),
  }))"""


def page_title(html: str) -> str:
    m = re.search(r"<title>(.*?)</title>", html, re.S)
    return m.group(1).strip() if m else "TinyTalk UA"


def build(tpl: str, engine: str, assets: str) -> str:
    # Order matters: substitute the (huge) base64 last so it is never rescanned
    # for placeholders, and never use str.format — the page is full of CSS braces.
    return tpl.replace("__ENGINE__", engine).replace("__ASSETS__", assets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "embed/model_neo_q4.bin"))
    ap.add_argument("--tok", default=str(ROOT / "embed/tok_neo.bin"))
    ap.add_argument("--js", default=str(ROOT / "tools/web/dist/llm.js"))
    ap.add_argument("--template", default=str(ROOT / "tools/web/page.html"))
    ap.add_argument("--out", default=str(ROOT / "tools/web/dist/index.html"))
    ap.add_argument("--site", default=str(ROOT / "tools/web/dist/site"))
    args = ap.parse_args()

    tpl = Path(args.template).read_text(encoding="utf-8")
    js = Path(args.js).read_text(encoding="utf-8")
    model = Path(args.model).read_bytes()
    tok = Path(args.tok).read_bytes()

    # --- artifact build: one file, zero requests -----------------------------
    inline = (INLINE_ASSETS
              .replace("__MODEL_B64__", base64.b64encode(model).decode())
              .replace("__TOK_B64__", base64.b64encode(tok).decode()))
    art = build(tpl, f"<script>{js}</script>", inline)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(art, encoding="utf-8")
    mb = out.stat().st_size / 1e6
    print(f"[+] {out} = {mb:.1f} MB  (artifact: inlined, no <head>)")
    if mb > 16:
        print("[!] over the 16 MB artifact limit")

    # --- site build: a real document + separate assets -----------------------
    body = build(tpl, '<script src="llm.js"></script>', FETCH_ASSETS)
    body = re.sub(r"<title>.*?</title>\s*", "", body, count=1, flags=re.S)
    site = Path(args.site)
    site.mkdir(parents=True, exist_ok=True)
    (site / "index.html").write_text(
        SHELL.replace("__TITLE__", page_title(tpl)).replace("__BODY__", body),
        encoding="utf-8")
    shutil.copyfile(args.js, site / "llm.js")
    (site / "model_neo_q4.bin").write_bytes(model)
    (site / "tok_neo.bin").write_bytes(tok)
    print(f"[+] {site}/")
    for f in sorted(site.iterdir()):
        print(f"      {f.name:22} {f.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
