# 01 — Model skeleton: Qwen/Qwen3.8-27B (verified 2026-10-02)

Question answered here: *is there a free skeleton from Qwen or DeepSeek we can grab and work with?*
**Yes.** Neither lab releases their *training data*, but both release the full model + architecture
under permissive licenses — that is exactly what "skeleton" means in practice. For Decebal-27B the
official skeleton is **`Qwen/Qwen3.8-27B`**, verified directly on Hugging Face (API + raw `config.json`).

## 1. The chosen skeleton — specs from the official repo config

| Field | Value |
|---|---|
| HF id | `Qwen/Qwen3.8-27B` (public, ungated; ~6.9M downloads) |
| **License** | **Apache 2.0** — confirmed in model-card frontmatter (`license: apache-2.0`) and repo tag |
| Architecture class | `Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5` (transformers-native; vLLM/SGLang compatible) |
| Modality | Vision-language (image/video in, text out). For Decebal we do **text CPT**: freeze the vision tower + MTP module, train the LM path (`freeze.vision_tower: frozen` in config) |
| Dense size | 27B — hidden 5120 × 64 layers, intermediate 17408, head_dim 256 (24 Q heads / 4 KV heads) |
| Attention layout | Hybrid: `full_attention_interval = 4` → 16 blocks of (3× linear "Gated DeltaNet"-style attn + 1× gated softmax full-attn), partial rotary factor 0.25, rope_theta 1e7 |
| Context | **262,144 native** (extensible toward ~1M in a dedicated stage) |
| Vocab | 248,320 (padded); strong multilingual coverage incl. Romanian diacritics |
| MTP module | 1 hidden layer (`mtp_num_hidden_layers: 1`) — keep frozen for text CPT |
| Inference variant | `Qwen/Qwen3.8-27B-FP8` (Apache 2.0) — serving/eval only, never train from it |

## 2. License landscape (why we pin exactly this repo)

- ✅ **`Qwen/Qwen3.8-27B` = Apache 2.0** → unrestricted commercial use + derivatives; no revenue
  thresholds. This is the clean story Decebal needs, and it's what the name "Decebal-**27B**" refers to.
- ⚠️ **Qwen3.8 Max-class checkpoints are NOT Apache**: `Qwen/Qwen3.8-2.4T-A95B` (MoE) and
  `Qwen/Qwen3.8-Flash-Next` carry `license: other` — a custom Qwen Community License with revenue-based
  obligations. **Excluded from this project entirely.** Same for their FP8 variants.
- ✅ MIT track (optional A/B only): DeepSeek V3/R1 and its distills are MIT — maximally permissive.
  `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` stays in config as an *alternative_mit_track* for a possible
  comparison run; it is **not** the default path.

> Rule: any skeleton revision we download gets its exact commit SHA recorded in
> `data/manifests/provenance.json`. We pin to Apache-2.0 checkpoints only.

## 3. What "from scratch" means here (be honest about scope)

- **True from-scratch pretraining** of a competitive LLM ≈ 10^24 FLOPs scale → infeasible for us.
- The pragmatic, industry-standard interpretation: *open skeleton weights + our own tokenizer tweaks +
  our own data pipeline + continued pretraining on Romanian-heavy mixture + SFT* = "our" model.
  That is how most national LLMs are actually built (continued-pretrained open bases).
- If you later want a **literally from-scratch** small RO model: the repo supports that too —
  `src/model/` can be pointed at a plain LLaMA-style config and trained on the same data pipeline
  (see roadmap P4 option B, e.g. a 350M–1B model fully ours).

## 4. Skeleton download commands

```bash
# Official skeleton (Apache 2.0) — ~55 GB in bf16 safetensors
huggingface-cli download Qwen/Qwen3.8-27B --local-dir checkpoints/skeleton/qwen3.8-27b

# Optional: FP8 variant for fast serving/eval only (never a training starting point)
huggingface-cli download Qwen/Qwen3.8-27B-FP8 --local-dir checkpoints/serving/qwen3.8-27b-fp8

# Optional A/B track (MIT, DeepSeek distill) — only if we decide to compare
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --local-dir checkpoints/skeleton/r1-distill-qwen-7b
```

## 5. Compute reality for a 27B CPT (sized in docs/03_roadmap.md)

| Run | Hardware | Notes |
|---|---|---|
| Smoke test | **1×8x80GB node** | LoRA r=64 on the full 27B (`config/model_decebal27b.yaml → skeleton.smoke_test`); validates data pipeline + loss curve cheaply |
| Full-param CPT | **~2×8x80GB nodes** (FSDP/ZeRO-3, bf16) | ~55 GB weights + optimizer state in bf16; single node is not enough for full params |
