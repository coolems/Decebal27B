# Decebal-27B — Romanian LLM from scratch (open-weights skeleton)

Goal: build a **Romanian-first large language model**. Strategy = take the open-weights skeleton
**`Qwen/Qwen3.8-27B`** (Apache 2.0, verified on HF), continue-pretrain it on a curated
**free, commercially-safe** data mixture with heavy Romanian weighting, then SFT + evaluate.
Everything in this repo is organized to make that pipeline reproducible.

## Repository layout

```
Decebal-27B/
├── README.md                  ← you are here
├── docs/
│   ├── 01_model_skeletons.md  ← Qwen/Qwen3.8-27B: verified specs, license landscape, download commands
│   ├── 02_training_data_catalog.md ← global + Romanian datasets, sizes, LICENSES (configs verified live)
│   ├── ROADMAP_MASTER.md      ← ★ THE live point-by-point roadmap P0–P7: statuses ✅/🔄/⬜, what-we-did notes, plan links
│   └── 03_roadmap.md          ← original phased plan (superseded by ROADMAP_MASTER for tracking)
├── config/
│   ├── model_decebal27b.yaml  ← skeleton (Qwen/Qwen3.8-27B) + continued-pretraining hyperparameters
│   └── data_mixture.json      ← mixture recipe (source, weight, license)
├── scripts/
│   ├── fetch_romanian.py      ← FineWeb2-ro (ron_Latn), mC4-ro, Wikipedia-RO dump, BAC math → data/raw
│   ├── fetch_global.py        ← FineWeb-Edu score-2, Cosmopedia-v2, Dolma, code sample → data/raw/{web_global,code}
│   └── fetch_sft_eval.py      ← LIRO ro_sts + WMT24++ en-ro_RO seeds → data/raw/sft_ro
├── requirements.txt           ← pip install -r requirements.txt (datasets, huggingface_hub[cli], pyarrow…)
├── src/
│   ├── preprocess/            ← cleaning, dedup, language filtering (datatrove-based)
│   ├── tokenizer/             ← Romanian-augmented BPE training / vocab extension
│   ├── model/                 ← Qwen3.8-27B skeleton loading, CPT surgery for continued pretraining
│   ├── train/                 ← pretrain + SFT entrypoints (FSDP / DeepSpeed)
│   └── eval/                  ← RO benchmarks: perplexity, LIRO tasks, MT, math
├── data/                      ← ALL training data lives here (see data/README.md)
│   ├── raw/romanian/          ← mC4-ro, Wikipedia-RO dump, BAC math (markdown); CulturaX-ro only in research track
│   ├── raw/web_global/        ← FineWeb / Dolma samples (English+multilingual ballast)
│   ├── raw/code/              ← the-stack-smol python sample now; The Stack v2 after gated access is granted
│   ├── raw/sft_ro/            ← Romanian instruction data for SFT stage
│   ├── processed/             ← cleaned + deduplicated, mixture-ready jsonl
│   ├── tokenized/             ← binary token shards for the trainer
│   ├── eval/                  ← held-out RO test sets (NEVER in training)
│   └── manifests/             ← provenance: source, license, sha256 per file
├── checkpoints/               ← model weights + optimizer state (git-ignored)
└── .temp/                     ← scratch only; never commit
```

## Quick start

1. Read `docs/01_model_skeletons.md`, then pull the official skeleton (**Qwen/Qwen3.8-27B**, Apache 2.0, ~55 GB):
   ```bash
   huggingface-cli download Qwen/Qwen3.8-27B --local-dir checkpoints/skeleton/qwen3.8-27b
   ```
2. Read `docs/02_training_data_catalog.md`, then pull the Romanian corpus:
   ```bash
   python scripts/fetch_romanian.py --sample    # → data/raw/{romanian,web_global} (wiki dump is always full)
   python scripts/fetch_global.py --sample     # small sample first, full later
   ```
3. Pull the small SFT/eval seeds:
   ```bash
   python scripts/fetch_sft_eval.py        # LIRO ro_sts + WMT24++ en-ro_RO (all <100 MB)
   ```
4. Follow **`docs/ROADMAP_MASTER.md`** point by point — it is the live tracker with ✅/🔄/⬜ statuses,
   what-we-did notes and links to every plan file (data pipeline → tokenizer check → LoRA smoke CPT on 1
   node → full-param CPT on ~2 nodes → SFT → eval → release).

## Ground rules

- **License hygiene**: every file that lands in `data/raw/**` must have its license recorded in
  `data/manifests/provenance.json`. No license → no training. Same for the skeleton: exact revision
  SHA of `Qwen/Qwen3.8-27B` is pinned there at download time (Apache 2.0 only — the Qwen3.8 Max-class
  custom-license checkpoints are excluded). See catalog for the approved data list.
- **One external action needed from you**: request gated access to `bigcode/the-stack-v2` on HuggingFace
  (email + terms agreement; bulk download also needs a SoftwareHeritage contact) so the full code corpus
  is ready by the pretraining phase — until then we run on the non-gated `the-stack-smol` python sample.
- **Romanian-first mixture**: target ~60–75% Romanian tokens in continued pretraining, rest high-quality
  global (FineWeb-Edu / Dolma) + code, so the model keeps world knowledge while becoming RO-native.
- Scratch/diagnostic work goes in `.temp/` only and gets cleaned up.
