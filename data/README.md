# Decebal-27B training data — layout & rules

```
data/
├── raw/
│   ├── romanian/        # mC4-ro, Wikipedia-RO dump, BAC math (markdown), CulturaX-ro (research only)
│   ├── web_global/      # FineWeb2-ro, FineWeb-Edu(-score-2), Dolma, Cosmopedia samples
│   ├── code/            # the-stack-smol python (phase-1 sample); The Stack v2 once gated access granted
│   └── sft_ro/          # Romanian instruction/eval seeds: LIRO suite, WMT24++ en-ro_RO, synthetic later
├── processed/           # cleaned + deduped jsonl, mixture-ready (one file per source, then mixture_v1.jsonl)
├── tokenized/           # binary token shards consumed by src/train
├── eval/                # HELD-OUT Romanian test sets — NEVER fed to training
└── manifests/
    └── provenance.json  # every ingested file: source, config, license, docs_written, path
```

## Rules
1. **Provenance first**: nothing enters `processed/` without a `manifests/provenance.json` entry.
2. **License gate** (see docs/02_training_data_catalog.md):
   - commercial track: only Apache-2.0 / MIT / CC0 / ODC-By / CC-BY-SA (with attribution) sources;
   - research track may add CC-BY-NC (CulturaX-ro, OSCAR-ro) but those checkpoints are labeled non-commercial.
3. `eval/` is frozen after P1 — any file added later must also be excluded from training shards.
4. Disk: expect ~2–5 TB for a full commercial mixture; start with `--sample` (a few GB).
