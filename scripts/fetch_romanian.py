#!/usr/bin/env python3
"""Fetch Romanian pretraining sources into data/raw/{romanian, web_global}.

Usage:
    pip install -r requirements.txt
    python scripts/fetch_romanian.py --sample        # small sample first (default)
    python scripts/fetch_romanian.py --full          # everything (big disk!)
    python scripts/fetch_romanian.py --skip-wiki     # skip the ~780MB Wikipedia dump

Every file written is logged to data/manifests/provenance.json with source + license.

Verified configs (2026-10-02, HF tree API):
  - FineWeb2 Romanian: config = "ron_Latn"  (NOT "ro")
  - mC4 Romanian:      config = "ro"        (bare ISO code, NOT "mc4_ro")
  - Wikipedia RO dump: rowiki-latest-pages-articles.xml.bz2 (~780 MB)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_RO = ROOT / "data" / "raw" / "romanian"
RAW_WEB = ROOT / "data" / "raw" / "web_global"
MANIFEST = ROOT / "data" / "manifests" / "provenance.json"

SAMPLE_ROWS = 20_000  # per-source sample size for --sample


def log_provenance(entry: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    if MANIFEST.exists():
        try:
            entries = json.loads(MANIFEST.read_text(encoding="utf-8")).get("entries", [])
        except Exception:
            entries = []
    entries.append(entry)
    MANIFEST.write_text(
        json.dumps({"entries": entries}, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def fetch_hf_dataset(hf_id: str, config: str | None, out_dir: Path, sample: bool, text_key: str = "text") -> None:
    """Stream a HF dataset (parquet-backed) and write JSONL."""
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[..] streaming {hf_id} ({config or 'default'}) ...")
    ds = load_dataset(hf_id, config, split="train", streaming=True)
    n = 0
    fname = f"{hf_id.replace('/', '__')}_{config or 'default'}.jsonl"
    with open(out_dir / fname, "w", encoding="utf-8") as f:
        for row in ds:
            text = (row.get(text_key) or "").strip()
            if not text:
                continue
            f.write(json.dumps({"source": hf_id, "config": config, "text": text}, ensure_ascii=False) + "\n")
            n += 1
            if sample and n >= SAMPLE_ROWS:
                break
    print(f"[ok] {hf_id} ({config}) -> {out_dir.name}/{fname}: {n} docs")
    log_provenance({
        "source": hf_id, "config": config, "docs_written": n,
        "license": "see docs/02_training_data_catalog.md",
        "path": str(out_dir / fname),
    })


def fetch_wikipedia_ro(out_dir: Path) -> None:
    """Download latest Romanian Wikipedia articles XML (CC-BY-SA 4.0).

    Verified file name pattern: rowiki-latest-pages-articles.xml.bz2 (~780 MB)
    Falls back to multistream parts if single file is missing.
    """
    import requests

    out_dir.mkdir(parents=True, exist_ok=True)
    base = "https://dumps.wikimedia.org/rowiki/latest/"
    r = requests.get(base, timeout=30)
    r.raise_for_status()

    # Prefer single-file dump (simplest to parse with mwparserfromhell / wikiextractor later)
    m = re.findall(r'href="([^"]+pages-articles\.xml\.bz2)"', r.text)
    if not m:
        # fallback: multistream parts
        m = sorted(re.findall(r'href="(rowiki-latest-pages-articles\d*\.xml-p[^\"]+\.bz2)"', r.text))
    if not m:
        print("[WARN] Could not find Wikipedia RO articles dump link; skipping.")
        return

    fname = m[0].split("/")[-1]
    url = base + fname
    dest = out_dir / fname
    print(f"[..] downloading {url} (~780 MB)")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print(f"[ok] wikipedia-RO -> {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    log_provenance({
        "source": url, "docs_written": None,
        "license": "cc-by-sa-4.0", "path": str(dest),
    })


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Romanian pretraining data")
    ap.add_argument("--full", action="store_true", help="download full corpora (default: small sample)")
    ap.add_argument("--sample", action="store_true", help="explicit sample mode (= default; kept for README compatibility)")
    ap.add_argument("--skip-wiki", action="store_true", help="skip the ~780MB Wikipedia dump")
    args = ap.parse_args()
    if args.full and args.sample:
        ap.error("--full and --sample are mutually exclusive")
    sample = args.sample or not args.full

    # 1) FineWeb2 Romanian — THE main commercial-safe RO web corpus (ODC-By)
    fetch_hf_dataset("HuggingFaceFW/fineweb-2", "ron_Latn", RAW_WEB / "fineweb2_ro", sample)

    # 2) mC4 Romanian slice (older pipeline; re-filter in P2-preprocess)
    fetch_hf_dataset("allenai/c4", "ro", RAW_RO / "mc4_ro", sample)

    # 3) Wikipedia RO dump (~780 MB, CC-BY-SA 4.0)
    if not args.skip_wiki:
        fetch_wikipedia_ro(RAW_RO / "wikipedia_ro")
    else:
        print("[skip] wikipedia-RO (--skip-wiki)")

    # 4) BAC math (small markdown files — via HF file download, no datasets lib needed)
    _fetch_bac_math(RAW_RO / "bac_math", sample)

    print("\nNext: python scripts/fetch_global.py" + ("" if args.full else " --sample"))
    return 0


def _fetch_bac_math(out_dir: Path, sample: bool) -> None:
    """Download BAC math markdown files from HF repo (Apache-2.0).

    Repo: asandeistefan/romanian-baccalaureate-mathematics
    Structure: markdown/*.md + metadata.csv
    Not a standard datasets builder → download files directly via huggingface_hub.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("[skip] BAC math — install huggingface_hub first (pip install -r requirements.txt)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    repo = "asandeistefan/romanian-baccalaureate-mathematics"
    api = HfApi()
    files = [f for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith("markdown/") and f.endswith(".md")]
    # Also grab metadata.csv
    try:
        meta_path = hf_hub_download(repo_id=repo, filename="metadata.csv", repo_type="dataset", local_dir=out_dir)
        print(f"[ok] BAC math metadata.csv -> {meta_path}")
    except Exception as e:
        print(f"[warn] metadata.csv failed: {e}")

    n = 0
    for f in files:
        if sample and n >= 50:  # cap at 50 files for sample mode
            break
        try:
            hf_hub_download(repo_id=repo, filename=f, repo_type="dataset", local_dir=out_dir)
            n += 1
        except Exception as e:
            print(f"[warn] {f}: {e}")
    print(f"[ok] BAC math -> {out_dir.name}/markdown/: {n} .md files")
    log_provenance({
        "source": repo, "config": None, "docs_written": n,
        "license": "apache-2.0", "path": str(out_dir),
    })


if __name__ == "__main__":
    sys.exit(main())
