# 02 — Training data catalog (global + Romanian), with LICENSES

Verified against HuggingFace API / papers on **2026-10-02**. Rule of this repo: *no license → no training.*

## 0. The honest answer about "the exact Qwen/DeepSeek data"

Neither Alibaba nor DeepSeek released their actual pretraining corpora (Qwen3 = ~36T tokens,
DeepSeek-V3 = 14.8T tokens — both confirmed in the tech reports). What IS available for free:

- **The same upstream sources** they built from: Common Crawl (public web crawl), Wikipedia dumps,
  public-domain books, and open code repositories.
- **Open re-processings of exactly those sources**, with permissive licenses — these are what
  the community uses to replicate Qwen/DeepSeek-style pretraining:

| What they used | Free equivalent we use | License | Size |
|---|---|---|---|
| Cleaned Common Crawl web text (Qwen3 ~36T tok; DeepSeek-V3 ~50% of 14.8T) | **FineWeb** `HuggingFaceFW/fineweb` / **FineWeb2** `HuggingFaceFW/fineweb-2` (1000+ langs incl. RO) / **Dolma** `allenai/dolma` | **ODC-By 1.0** (free, commercial OK; cite source) | 15T tok EN / FineWeb2 multilingual / 3.4T tok Dolma |
| Educational web subset (Qwen3 stage-2 "knowledge-intensive") | **FineWeb-Edu** `HuggingFaceFW/fineweb-edu` (+score-2 = 5.4T) | ODC-By 1.0 | 1.3T / 5.4T tok EN |
| Code (DeepSeek-V3: code is a *majority* of its tokens; Qwen3 uses Coder-synthetic + repos) | **The Stack v2** `bigcode/the-stack-v2` (per-language subsets, e.g. python, javascript, …) | **BigCode Open Datasets LICENSE v1.0** — free incl. commercial for *research/derivative models*; attribution required; some languages have extra caveats → use the permissive core languages only | ~653 GB / 2.3T tok (v2.2) |
| Math/science + synthetic data (Qwen3 uses Qwen2.5-Math/Coder synthesis) | **Cosmopedia v2** `HuggingFaceTB/cosmopedia` (synthetic, Llama-generated) inside SmolLM-Corpus; OpenMathInstruct etc. | ODC-By 1.0 / MIT-ish per dataset card | ~212 GB Cosmo v2 |
| Books/articles ballast | **Nemotron-CC** `nvidia/Nemotron-CC` (web + books, deduped) — check current card before use; fallback = Dolma `dolma_1.7` subset | ODC-By / per card | ~9T tok (books+web) |
| Romanian web text specifically | **FineWeb2-ro** (`HuggingFaceFW/fineweb-2`, config `ro`) + community mirror `rotarue/fineweb2-romanian-shards` | ODC-By 1.0 / Apache-2.0 (mirror) | see §2 |

> ⚠️ **ODC-By 1.0 nuance**: free to use commercially, but you must attribute the dataset and
> you may not present it as "your own" data. Fine for training a model we ship under our name —
> just keep `data/manifests/provenance.json` accurate.

## 1. Global corpora (approved list)

| # | Dataset (HF id) | License | What it is | Use in our mixture |
|---|---|---|---|---|
| G1 | `HuggingFaceFW/fineweb-2` (config **`ron_Latn`**) ⚠️ config name verified via HF tree API 2026-10-02 — it is the ISO+script code, NOT bare `ro` | ODC-By 1.0 | Cleaned Common Crawl, 1000+ languages, datatrove pipeline — **includes Romanian** | PRIMARY RO web source |
| G2 | `rotarue/fineweb2-romanian-shards` | Apache-2.0 | Community mirror of FineWeb2 Romanian shards (parquet) | Convenience mirror of G1 |
| G3 | `HuggingFaceFW/fineweb-edu` (+ `-score-2`) | ODC-By 1.0 | Quality-scored educational web (EN) | EN ballast, knowledge density |
| G4 | `allenai/dolma` (`dolma_1.7`, `wikipedia_20220301.en`, `pile_books3_2025`) | ODC-By 1.0 | 3.4T curated EN mix incl. books, wiki, arXiv | EN ballast + long-form text |
| G5a | `bigcode/the-stack-smol` (config **`python`**, 10K samples ≈90 MB, **non-gated**) | BigCode terms v1.0 | 10K random The-Stack samples per language | Phase-1 code sample — runs today without any access request |
| G5b | `bigcode/the-stack-v2` (python/javascript/… subsets) ⚠️ **GATED: email + agreement required** (`gated:auto`, verified 2026-10-02; bulk download needs SoftwareHeritage contact per its ToU) | BigCode license v1.0 | Decontaminated code repos, ~653 GB / 2.3T tok (v2.2) | Main code source for the full run — request access now so it's ready by P4; per-language license check before adding each subset |
| G5c | `deepmind/code_contests` (CodeContests, AlphaCode problems) | **CC-BY-4.0** (verified on card 2026-10-02) | ~38K competitive-programming problem+solution pairs | Code reasoning top-up; small but high signal |
| G6 | `HuggingFaceTB/SmolLM-Corpus` (`cosmopedia-v2`) | ODC-By 1.0 | Synthetic educational text (Llama-generated) | Quality synthetic ballast, cheap to mix in |
| G7 | `commoncrawl/common-crawl` (raw WARC, optional) | CC0-1.0 (the crawl itself) | Raw web — only if we run our own datatrove pipeline for extra RO volume | DIY fallback when G1 runs out of RO tokens |

## 2. Romanian datasets (approved list)

### 2a. Pretraining text (bulk)

| # | Dataset (HF id / source) | License | Size (RO) | Notes |
|---|---|---|---|---|
| R1 | `uonlp/CulturaX` config **ro** | ⚠️ **no license tag on HF card → treat as non-commercial / research-only until verified with authors** (paper arXiv:2309.09400; 6.3T tokens / 167 langs, cleaned+deduped) | ~1–2B tokens RO est. | Great quality; **research track only**, separate mixture file |
| R2 | `HuggingFaceFW/fineweb-2` config **`ron_Latn`** (or G2 mirror) | ODC-By 1.0 / Apache-2.0 | largest clean RO web corpus on HF | **main commercial-safe bulk source** |
| R3 | `allenai/c4` config **`ro`** (bare ISO code — verified via HF parquet API 2026-10-02; the old `mc4_ro` name does NOT exist on current HF) | ODC-By | ~hundreds of GB raw, ~5–10B tokens before filtering | Older pipeline; re-filter with our datatrove steps before use |
| R4 | Wikipedia RO dump (`https://dumps.wikimedia.org/rowiki/latest/`, file `rowiki-latest-pages-articles.xml.bz2` ≈ **780 MB**, verified 2026-10-02) | **CC-BY-SA 4.0** (text) / GFDL | ~1–2B tokens est. | CC-BY-SA is commercial-OK with attribution + share-alike on *derivatives of the data* — fine for model training; keep attribution in provenance |
| R5 | `oscar-corpus/OSCAR-2301` config **ro** (gated, free access) | ⚠️ OSCAR research license — **non-commercial** | ~1B tokens RO | Research track only |
| R6 | Romanian government open data: **dataguvernanta.ro** (dataset descriptions, statistics), **parlamentul.ro**, **judele/curtea de casatie** judgments | Public domain / open-data licenses per portal | tens of GB | High-value civic/legal RO text; scrape with robots.txt respect + record source URL in provenance |
| R7 | `google/wmt24pp` config **`en-ro_RO`** (Apache-2.0 verified on card 2026-10-02; ~43 language pairs total) | Apache-2.0 | 10K–100K rows per pair | Small, but perfect for cross-lingual SFT/eval |
| R8 | News archives: **Mediafax/Ziare.ro RSS archives** (where terms allow), **Agerpres** (verify license; some content is free with attribution) | mixed — verify per source | variable | Optional enrichment; never auto-scrape paywalled material |

### 2b. Romanian SFT / instruction data (for the post-training stage)

| # | Dataset (HF id) | License | Notes |
|---|---|---|---|
| S1 | `dumitrescustefan/ro_sent` (RO sentiment, from "Birth of Romanian BERT") | unknown → treat as research-only | eval + small SFT seed |
| S2 | LIRO benchmark sets (`dumitrescustefan/*`: ro_sts cc-by-4.0, ro_ner, ro_morph, ro_sent) | CC-BY-4.0 (ro_sts) / mixed | **eval-first**; small enough to also seed SFT for research runs |
| S3 | `asandeistefan/romanian-baccalaureate-mathematics` (2019→ BAC exams as `markdown/*.md` + `metadata.csv`; **file-based repo, not a datasets builder** → fetch via `huggingface_hub`, verified 2026-10-02) | Apache-2.0 | Excellent RO math reasoning data — SFT + eval (hold out newest year for P6) |
| S4 | Multilingual instruction sets with RO coverage: `HuggingFaceXLabs/xlsum`, Tulu 3 / OpenHermes multilingual splits (check per-card license, most ODC/CC-BY) | mixed, per card | Top-up for general instruction following in RO |
| S5 | **Synthetic RO data**: generate Romanian instructions by translating top EN instruction sets with `Qwen/Qwen3.8-27B` or DeepSeek-R1 (both permissively licensed as *tools*; output is ours to own) | ours | Standard practice; log generator + prompt template in provenance |
| S6 | `datadriven-company/TTS-Romanian` transcripts (720h audiobook text, CC-BY-4.0) — *text only* | CC-BY-4.0 | Long-form RO reading material if we ever do speech or long-text CPT |

### 2c. Romanian EVAL sets (held-out, never trained on)

| Set | Source | Task |
|---|---|---|
| LIRO suite | `dumitrescustefan/*` (ro_sts, ro_ner, ro_morph, ro_sent, ro_coref…) | NLU benchmark battery |
| RO-Sentiment / RO-IMDB | `dumitrescustefan/ro_sent`, RO-IMDB mirrors | sentiment |
| WMT24++ ro↔en | `google/wmt24pp` (BLEU/COMET) | translation sanity check |
| BAC math 2025–26 (new years, not in S3) | same HF repo when updated | RO math reasoning |
| MMLU-RO / XLM-style multilingual eval | build from `cais/mmlu` EN + verified translations | knowledge retention check (did CPT hurt English?) |

## 3. Target mixture for continued pretraining (v1, see config/data_mixture.json)

```
```
45%  Romanian web     → FineWeb2-ro (config ron_Latn)
10%  Romanian curated → Wikipedia-RO dump
15%  Romanian top-up  → mC4-ro re-filtered (datatrove, P2)
 5%  Romanian civic   → gov/legal open data (dataguvernanta.ro, parlamentul.ro)
12%  English ballast  → FineWeb-Edu-score-2 (sampled)
 5%  English longform → Dolma books/wiki sample
 6%  Code             → The Stack v2 permissive-language subsets (gated access pending)
 2%  Synthetic        → Cosmopedia-v2 sample (+ our own RO synthetic later)
```

(Weights mirror `config/data_mixture.json` exactly — sum = 1.0, Romanian share = 75%. BAC math enters at the SFT stage (P5), not as a CPT mixture weight.)

(Research track adds CulturaX-ro + OSCAR-ro at the cost of commercial use — separate config.)

## 4. License cheat-sheet (what's safe for a commercial model)

- ✅ **Apache-2.0 / MIT / CC0 / ODC-By** → free, commercial OK (ODC-By: keep attribution).
- ⚠️ **CC-BY-SA** (Wikipedia) → commercial OK; attribute; don't republish the corpus itself as your own dataset.
- ⚠️/❌ **unlicensed or CC-BY-NC / research-only** (CulturaX — unverified; OSCAR — research license) → research track only.
- ❌ Anything "unknown"/"other" without a readable license → excluded until verified.
