# 03 — Roadmap (phased)

Compute assumption: **full-param CPT of the 27B skeleton ≈ 2×8-GPU nodes (A100/H100 80GB)**;
a single 8x80GB node runs the **LoRA smoke tests** on the same model. Everything is sized so that one
node can at least validate the entire pipeline before scaling out.

## P0 — Skeleton in hand (day 1)
- [ ] `huggingface-cli download Qwen/Qwen3.8-27B` → `checkpoints/skeleton/qwen3.8-27b` (Apache 2.0, ~55 GB)
- [ ] Load with transformers (`Qwen3_5ForConditionalGeneration`, text path), generate a Romanian prompt,
      sanity-check diacritics output; confirm vision tower + MTP stay frozen for CPT
- [ ] Record exact commit/revision SHA in `data/manifests/provenance.json`

## P1 — Data pipeline (week 1–2)
- [ ] Run `scripts/fetch_romanian.py --sample` → FineWeb2-ro (`ron_Latn`), mC4-ro slice, Wikipedia-RO dump, BAC math (CulturaX-ro = research track only, not fetched by default)
- [ ] Run `scripts/fetch_global.py --sample` → FineWeb-Edu sample, Dolma sample (skippable via `--skip-dolma`), the-stack-smol python (The Stack v2 pending gated access)
- [ ] datatrove pipeline in `src/preprocess/`: URL-strip → language filter (fasttext lid.176 / langdetect) →
      quality heuristics (line ratio, stopword density, diacritics sanity for RO) → exact + fuzzy dedup
      (MinHash) → length filter 50–8k chars
- [ ] Emit `data/processed/mixture_v1.jsonl` respecting weights in `config/data_mixture.json`; write
      per-file sha256 into `data/manifests/provenance.json`

## P2 — Tokenizer check (week 2)
- [ ] Measure RO tokenization efficiency of the Qwen3.8 tokenizer (vocab 248,320) on our mixture
      (tokens/kilochar vs. baseline BPE)
- [ ] If < target: train a small **vocab extension** (add ~8–16k frequent RO morphemes/diacritic sequences,
      init embeddings from mean, keep 95%+ old-vocab stability test). See `src/tokenizer/`.

## P3 — Continued pretraining (week 2–4)
- [ ] Smoke: **LoRA r=64 on the full Qwen/Qwen3.8-27B** (single node), ~10M RO tokens, verify loss curve +
      no English regression on held-out EN set
- [ ] Main: full-param CPT of `Qwen/Qwen3.8-27B`, ~50–100B RO-heavy mixture tokens, FSDP/ZeRO-3 on ~2 nodes,
      LR warmup → cosine 2e-5→2e-6, batch 2M tokens/step target (vision tower + MTP frozen)
- [ ] Checkpoints every 2.5B tokens → `checkpoints/cpt_decebal27b_ro/`

## P4 — Post-training (week 4–5)
- [ ] SFT on Romanian instruction mix (S1–S6 from catalog, §2b): chat template = the skeleton's, thinking-mode off for v1
- [ ] Optional: DPO on RO preference pairs (synthetic + human spot-checks)

## P4-option-B — literal from-scratch (parallel research track, optional)
- [ ] Plain LLaMA-style 350M–1B config in `src/model/`, same data pipeline, ~20–50B tokens.
      Expect a small but *fully ours* RO model; useful for tokenizer/vocab experiments and as an ablation.

## P5 — Evaluation (continuous)
- [ ] `src/eval/run_all.py`: LIRO suite, RO-Sentiment, BAC math, WMT24++ ro↔en COMET, MMLU-EN regression
- [ ] Publish a short report per checkpoint in `docs/reports/`

## P6 — Release
- [ ] Model card with full provenance (skeleton license + data licenses), weights on HF under our org,
      Apache-2.0 for the commercial track / CC-BY-NC clearly labeled for research-track checkpoints.
