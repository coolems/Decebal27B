# ROADMAP MASTER — Decebal-27B (status: live document)

Official model name: **Decebal-27B** · Skeleton: **`Qwen/Qwen3.8-27B`** (Apache 2.0, verified on HF 2026-10-02).
Rule of this repo: *no license → no training.* Every file that enters `data/` gets a provenance entry.

Legend: ✅ done · 🔄 in progress · ⬜ not started

---

## P0 — Skeleton acquisition & verification  (⬜)
*Plan record: the migration session ran before the working-root fix — its plan file is outside this root; content preserved in [docs/01_model_skeletons.md](01_model_skeletons.md) + decision log below.*

- ⬜ P0.1 Download `Qwen/Qwen3.8-27B` → `checkpoints/skeleton/qwen3.8-27b/` (~55 GB, Apache 2.0)
      `huggingface-cli download Qwen/Qwen3.8-27B --local-dir checkpoints/skeleton/qwen3.8-27b`
- ⬜ P0.2 Load with transformers (`Qwen3_5ForConditionalGeneration`, text path only); vision tower + MTP frozen for CPT
- ⬜ P0.3 Romanian sanity generation (diacritics check: ș/ț/ă/î/â), record sample outputs in `docs/reports/p0_sanity.md`
- ⬜ P0.4 Record exact revision SHA + license string into `data/manifests/provenance.json`

## P1 — Data gathering (Phase 1)  (🔄 IN PROGRESS)
*Plan links: [plan_20261002_1015.md](../plan_20261002_1015.md) (catalog + scripts); the data-gathering kickoff plan ran pre-root-fix — content preserved in the "What we did" list below; code audit 2026-10-02: [plan_20261002_1117.md](../plan_20261002_1117.md)*

**What we did in the 2026-10-02 kickoff:**
- ✅ Verified every dataset ID, config name and license LIVE via HF API/tree/parquet endpoints (details below + catalog §G/R/S)
- ✅ Rewrote `scripts/fetch_romanian.py` (correct configs: FineWeb2 **`ron_Latn`**, c4 **`ro`**; wiki dump targets the verified 780 MB `rowiki-latest-pages-articles.xml.bz2`; BAC math added via huggingface_hub file download)
- ✅ Rewrote `scripts/fetch_global.py` (The Stack v2 is **GATED** → phase-1 uses non-gated `bigcode/the-stack-smol` python ≈90 MB; Dolma flagged slow/builder-based with `--skip-dolma`)
- ✅ New `scripts/fetch_sft_eval.py` (LIRO ro_sts CC-BY-4.0 + WMT24++ config **`en-ro_RO`** Apache-2.0)
- ✅ New root `requirements.txt`; catalog §G5 split into G5a/G5b/G5c with the gating reality documented
    - ✅ Code audit (plan_20261002_1117.md): fixed data_mixture.json configs (`ron_Latn` / c4 `ro` / stack-v2 id), added explicit `--sample` flags to both fetch scripts, fixed Wikipedia multistream fallback regex + code provenance undercount, aligned catalog §3 with the mixture weights

### P1-A Romanian sources
- 🔄 P1.1 FineWeb2 **Romanian** (`HuggingFaceFW/fineweb-2` config **`ron_Latn`**, ODC-By) → sample first, then full RO shards  *(config verified via HF tree: `data/ron_Latn/{train,test}`)* → `data/raw/web_global/fineweb2_ro/`
- ⬜ P1.2 mC4 Romanian slice (`allenai/c4` config **`ro`** — bare ISO code, ODC-By)  *(verified via HF parquet API)* → `data/raw/romanian/mc4_ro/` (re-filter later in P2-preprocess)
- ⬜ P1.3 Wikipedia RO dump (`rowiki-latest-pages-articles.xml.bz2`, ≈780 MB, CC-BY-SA 4.0)  *(file name + size verified from dumps index)* → `data/raw/romanian/wikipedia_ro/`
- ⬜ P1.4 (research track, separate mixture) CulturaX-ro + OSCAR-ro — only after license verification with authors

### P1-B Global ballast (EN + code)
- 🔄 P1.5 FineWeb-Edu score-2 sample (`HuggingFaceFW/fineweb-edu-score-2`, ODC-By, parquet-backed ✅) → `data/raw/web_global/fineweb_edu_en/`
- ⬜ P1.6 Dolma subset (books/wiki, ODC-By; builder-based = slow → use `--skip-dolma` for smoke runs) → `data/raw/web_global/dolma/`
- ⬜ P1.7 Cosmopedia v2 via SmolLM-Corpus config **`cosmopedia-v2`** (ODC-By, parquet-backed ✅) → `data/raw/web_global/cosmopedia/`
- 🔄 P1.8 Code: phase 1 = `bigcode/the-stack-smol` python (non-gated, ≈90 MB); **ACTION ITEM: request gated access to `bigcode/the-stack-v2` now** (email + agreement; bulk download needs SoftwareHeritage contact) → full run uses v2 Python subset
- ⬜ P1.8b CodeContests (`deepmind/code_contests`, CC-BY-4.0, ≈36 GB parquet) — optional code-reasoning top-up

### P1-C SFT / eval seeds (small, high value)
- 🔄 P1.9 BAC math 2019→ (`asandeistefan/romanian-baccalaureate-mathematics`, Apache-2.0; file-based `markdown/*.md` + `metadata.csv`) → `data/raw/romanian/bac_math/`; newest year held out for P6 eval
- 🔄 P1.10 LIRO suite (`dumitrescustefan/*`: ro_sts **CC-BY-4.0 verified**; others mixed) → `data/raw/sft_ro/liro/` (eval-first)
- 🔄 P1.11 WMT24++ config **`en-ro_RO`** (`google/wmt24pp`, Apache-2.0 verified on card) → `data/raw/sft_ro/wmt24pp/`

### P1-D Verification & bookkeeping
- ✅ P1.12 Every fetch logs source/config/docs/license into `data/manifests/provenance.json` (implemented in all three scripts)
- ⬜ P1.13 Spot-check each file: line counts, diacritics intact (UTF-8), no HTML garbage; write `docs/reports/p1_data_audit.md`

## P2 — Data pipeline (preprocess)  (⬜)
*Plan link: [plan_20261002_1015.md](../plan_20261002_1015.md) step "datatrove pipeline"*

- ⬜ P2.1 Implement `src/preprocess/`: URL-strip → language filter (fasttext lid.176) → quality heuristics
      (line ratio, stopword density, RO diacritics sanity) → exact + fuzzy dedup (MinHash) → length 50–8k chars
- ⬜ P2.2 Emit `data/processed/mixture_v1.jsonl` respecting weights in `config/data_mixture.json`
      (target: ~70% RO tokens / 30% EN+code ballast — see catalog §3)
- ⬜ P2.3 Per-file sha256 into provenance manifest; keep research-track mixture as a SEPARATE file

## P3 — Tokenizer check & optional vocab extension  (⬜)
*Plan record: pre-root-fix session plan (outside this working root); specs preserved in [docs/01_model_skeletons.md](01_model_skeletons.md) §1.*

- ⬜ P3.1 Measure RO tokenization efficiency of the Qwen3.8 tokenizer on our mixture (tokens/kilochar vs baseline BPE)
- ⬜ P3.2 If below target: train small vocab extension (~8–16k frequent RO morphemes/diacritic sequences),
      init embeddings from mean, keep ≥95% old-vocab stability test → `src/tokenizer/`

## P4 — Continued pretraining  (⬜)
*Plan record: pre-root-fix session plan (outside this working root); compute sizing preserved in [docs/03_roadmap.md](03_roadmap.md) + [docs/01_model_skeletons.md](01_model_skeletons.md) §5.*

- ⬜ P4.1 Smoke: **LoRA r=64 on the full 27B** (single node), ~10M RO tokens; verify loss curve + no EN regression
      (held-out EN set from FineWeb-Edu sample)
- ⬜ P4.2 Main: full-param CPT, ~50–100B tokens of the RO-heavy mixture, FSDP/ZeRO-3 on ≈2×8x80GB nodes,
      LR warmup → cosine 2e-5→2e-6, batch target 2M tokens/step (vision tower + MTP frozen)
- ⬜ P4.3 Checkpoints every 2.5B tokens → `checkpoints/cpt_decebal27b_ro/`

## P5 — Post-training  (⬜)
*Plan link: [plan_20261002_1015.md](../plan_20261002_1015.md) (SFT catalog S1–S6)*

- ⬜ P5.1 SFT on Romanian instruction mix (BAC math, LIRO seeds, WMT24++, synthetic RO via our own generator);
      chat template = the skeleton's; thinking-mode OFF for v1
- ⬜ P5.2 Optional: DPO on RO preference pairs (synthetic + human spot-checks)

## P6 — Evaluation (continuous from P4.3 onward)  (⬜)
*Plan link: [plan_20261002_1015.md](../plan_20261002_1015.md) eval sets §2c*

- ⬜ P6.1 `src/eval/run_all.py`: LIRO suite, RO-Sentiment, BAC math (held-out years), WMT24++ ro↔en COMET
- ⬜ P6.2 MMLU-EN regression gate: CPT must not drop English below skeleton baseline −2%
- ⬜ P6.3 Short report per checkpoint in `docs/reports/`

## P7 — Release  (⬜)
*Plan record: pre-root-fix session plan (outside this working root); license ground rules preserved in [README.md](../README.md) + catalog §4.*

- ⬜ P7.1 Model card with full provenance (skeleton Apache 2.0 + every data license from the manifest)
- ⬜ P7.2 Weights on HF under our org: commercial track = Apache 2.0; research-track checkpoints labeled CC-BY-NC

---

## Optional parallel track — literal from-scratch (P4-option-B, ⬜, not scheduled)
Plain LLaMA-style 350M–1B in `src/model/`, same pipeline, ~20–50B tokens. Fully-ours small RO model;
useful for tokenizer experiments and as an ablation. See [docs/03_roadmap.md](03_roadmap.md).

## Decision log
| Date | Decision | Why |
|---|---|---|
| 2026-10-02 | Skeleton = `Qwen/Qwen3.8-27B` only (no old Qwen3 4B/1.7B) | User directive; Apache 2.0 verified on HF card + raw config.json |
| 2026-10-02 | Smoke test = LoRA r=64 on the full 27B, not a smaller model | Qwen3.8 has no small dense checkpoint — pre-root-fix migration session, see docs/01_model_skeletons.md |
| 2026-10-02 | CulturaX-ro / OSCAR-ro = research track only | Licenses unverified → honesty rule "no license, no training" for the commercial track |
| 2026-10-02 | Re-audit after fixes (plan_20261002_1332.md): all 7 prior defects verified fixed; also fixed 5 broken plan links in this file, removed stale .backup + __pycache__ artifacts |
- **2026-10-02** — Project published to GitHub: `https://github.com/coolems/Decebal27B` (commit df52941, 23 files). Heavy data/checkpoints stay git-ignored; placeholder dirs tracked via .gitkeep.
