#!/usr/bin/env python3
"""Fetch global (English + code) ballast corpora into data/raw/{web_global, code}.

Usage:
    pip install -r requirements.txt
    python scripts/fetch_global.py --sample   # default: small samples for pipeline testing
    python scripts/fetch_global.py --full     # full subsets (big disk!)
    python scripts/fetch_global.py --skip-dolma  # Dolma needs full datasets lib + is slow

All sources are commercial-safe per docs/02_training_data_catalog.md.

Verified configs (2026-10-02):
  - FineWeb-Edu score-2: default config, parquet-backed ✅
  - SmolLM-Corpus cosmopedia-v2: config "cosmopedia-v2", parquet-backed ✅
  - Dolma: builder-based (NO parquet expansion) → needs full `datasets` lib; slow
  - The Stack v2: GATED (email + agreement required) → use the-stack-smol as non-gated fallback
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_WEB = ROOT / "data" / "raw" / "web_global"
RAW_CODE = ROOT / "data" / "raw" / "code"
MANIFEST = ROOT / "data" / "manifests" / "provenance.json"

SAMPLE_ROWS = 20_000


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


def fetch_dolma_sample(out_dir: Path, sample: bool) -> None:
    """Dolma is builder-based (no parquet expansion). Load a small subset.

    Uses the `dolma_1.7` config which mixes books + wiki + arXiv.
    This is SLOW even for samples because it processes on-the-fly.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[skip] Dolma — install datasets first")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[..] loading allenai/dolma (dolma_1.7) ... SLOW for large samples")
    try:
        ds = load_dataset("allenai/dolma", "dolma_1.7", split="train", streaming=True)
        n = 0
        fname = out_dir / "allenai__dolma_dolma_1.7.jsonl"
        with open(fname, "w", encoding="utf-8") as f:
            for row in ds:
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                f.write(json.dumps({"source": "allenai/dolma", "config": "dolma_1.7", "text": text}, ensure_ascii=False) + "\n")
                n += 1
                if sample and n >= SAMPLE_ROWS:
                    break
        print(f"[ok] dolma -> {fname.name}: {n} docs")
        log_provenance({"source": "allenai/dolma", "config": "dolma_1.7", "docs_written": n,
                        "license": "odc-by-1.0", "path": str(fname)})
    except Exception as e:
        print(f"[FAIL] dolma: {type(e).__name__}: {e}")


def fetch_code_smol(out_dir: Path, sample: bool) -> None:
    """The Stack smol (non-gated fallback for The Stack v2).

    Repo: bigcode/the-stack-smol — 10K samples per language, JSON format.
    License: same BigCode terms as The Stack (permissive core languages OK).
    Python file: data/python/data.json (~90 MB)
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[skip] code — install huggingface_hub first")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    repo = "bigcode/the-stack-smol"
    # Download python subset (single JSON file with 10K entries)
    try:
        path = hf_hub_download(repo_id=repo, filename="data/python/data.json", repo_type="dataset", local_dir=out_dir / "the_stack_smol")
        print(f"[ok] the-stack-smol python -> {path}")
# Count ALL samples for provenance (the file is fully downloaded regardless of --sample)
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        print(f"     ({n} code samples)")
        log_provenance({"source": repo, "config": "python", "docs_written": n,
                        "license": "bigcode-open-datasets-v1.0 (see terms)", "path": str(path)})
    except Exception as e:
        print(f"[FAIL] the-stack-smol python: {type(e).__name__}: {e}")

    # Note: The Stack v2 (full, 653 GB) requires gated access:
    #   - email + agreement at https://huggingface.co/datasets/bigcode/the-stack-v2
    #   - then: huggingface-cli download bigcode/the-stack-v2 --include "data/Python/*" \
    #           --local-dir data/raw/code/the_stack_v2 --token $HF_TOKEN
    print("     NOTE: For full The Stack v2 (653 GB), request gated access at HF + use huggingface-cli with token.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch global ballast corpora")
    ap.add_argument("--full", action="store_true", help="download full subsets (default: sample)")
    ap.add_argument("--sample", action="store_true", help="explicit sample mode (= default; kept for README compatibility)")
    ap.add_argument("--skip-dolma", action="store_true", help="skip Dolma (slow, builder-based)")
    args = ap.parse_args()
    if args.full and args.sample:
        ap.error("--full and --sample are mutually exclusive")
    sample = args.sample or not args.full

    # 1) English knowledge ballast — FineWeb-Edu score-2 (ODC-By, parquet-backed, fast)
    fetch_hf_dataset("HuggingFaceFW/fineweb-edu-score-2", None, RAW_WEB / "fineweb_edu_en", sample)

    # 2) Synthetic quality ballast — Cosmopedia v2 via SmolLM-Corpus (ODC-By, parquet-backed)
    fetch_hf_dataset("HuggingFaceTB/SmolLM-Corpus", "cosmopedia-v2", RAW_WEB / "cosmopedia", sample, text_key="text")

    # 3) Long-form EN ballast — Dolma (books + wiki; SLOW, builder-based)
    if not args.skip_dolma:
        fetch_dolma_sample(RAW_WEB / "dolma", sample)
    else:
        print("[skip] dolma (--skip-dolma)")

    # 4) Code — The Stack smol (non-gated, 10K python samples ~90 MB)
    fetch_code_smol(RAW_CODE, sample)

    print("\nGlobal ballast ready. Next: build mixture with src/preprocess (datatrove).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
