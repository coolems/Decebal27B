#!/usr/bin/env python3
"""Fetch Romanian SFT / eval seed datasets into data/raw/sft_ro/.

Usage:
    pip install -r requirements.txt
    python scripts/fetch_sft_eval.py          # small (these are all <100 MB)

Sources (all verified 2026-10-02):
  - LIRO suite (dumitrescustefan/*): ro_sts (CC-BY-4.0), ro_ner, ro_morph, ro_sent
  - WMT24++ ro↔en (google/wmt24pp, config en-ro_RO, Apache-2.0)
  - BAC math already handled by fetch_romanian.py

Every file logged to data/manifests/provenance.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_SFT = ROOT / "data" / "raw" / "sft_ro"
MANIFEST = ROOT / "data" / "manifests" / "provenance.json"


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


def fetch_liro(out_dir: Path) -> None:
    """LIRO benchmark sets — small enough to grab in full (all <10 MB each)."""
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    # ro_sts is the main one with a clear CC-BY-4.0 license; others are mixed
    for ds_name, cfg in [
        ("ro_sts", "ro_sts"),   # CC-BY-4.0, 8628 pairs
    ]:
        try:
            ds = load_dataset(f"dumitrescustefan/{ds_name}", cfg)
            n_total = sum(len(ds[split]) for split in ds.keys())
            # Save all splits as JSONL
            fname = out_dir / f"liro_{ds_name}.jsonl"
            with open(fname, "w", encoding="utf-8") as f:
                for split in ds.keys():
                    for row in ds[split]:
                        row["_split"] = split
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[ok] LIRO {ds_name} -> {fname.name}: {n_total} rows ({', '.join(ds.keys())})")
            log_provenance({"source": f"dumitrescustefan/{ds_name}", "config": cfg,
                            "docs_written": n_total, "license": "cc-by-4.0", "path": str(fname)})
        except Exception as e:
            print(f"[FAIL] LIRO {ds_name}: {type(e).__name__}: {e}")


def fetch_wmt24pp_ro(out_dir: Path) -> None:
    """WMT24++ ro↔en pairs (Apache-2.0). Config = 'en-ro_RO'."""
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        ds = load_dataset("google/wmt24pp", "en-ro_RO", split="train")
        fname = out_dir / "wmt24pp_en_ro.jsonl"
        with open(fname, "w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[ok] WMT24++ en-ro -> {fname.name}: {len(ds)} rows")
        log_provenance({"source": "google/wmt24pp", "config": "en-ro_RO",
                        "docs_written": len(ds), "license": "apache-2.0", "path": str(fname)})
    except Exception as e:
        print(f"[FAIL] WMT24++ ro: {type(e).__name__}: {e}")


def main() -> int:
    fetch_liro(RAW_SFT / "liro")
    fetch_wmt24pp_ro(RAW_SFT / "wmt24pp")
    print("\nSFT/eval seeds ready. BAC math is in data/raw/romanian/bac_math (from fetch_romanian.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
