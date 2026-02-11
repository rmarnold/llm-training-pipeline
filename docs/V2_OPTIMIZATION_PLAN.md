# V2 Optimization Plan: GPT-OSS 20B Rust Coding Agent

**Date:** 2026-02-10
**Status:** Planning
**Scope:** Incorporate latest Unsloth features and GPT-OSS best practices into the training pipeline.

---

## Research Sources

- [Unsloth: Faster MoE Training (Split LoRA)](https://unsloth.ai/docs/new/faster-moe) — Feb 2026
- [Unsloth: GRPO Long Context](https://unsloth.ai/docs/new/grpo-long-context) — Jan 2026
- [Unsloth: FP8 Reinforcement Learning](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning) — Nov 2025
- [Unsloth: 3x Faster Training with Packing](https://docs.unsloth.ai/new/3x-faster-training-packing) — Dec 2025
- [Unsloth: GPT-OSS Fine-tuning](https://unsloth.ai/blog/gpt-oss) — Nov 2025
- [Unsloth: GPT-OSS Long Context (Flex Attention)](https://unsloth.ai/blog/gpt-oss-context) — Dec 2025
- [OpenAI Cookbook: Fine-tuning GPT-OSS](https://developers.openai.com/cookbook/articles/gpt-oss/fine-tune-transfomers)
- [NVIDIA: SFT + QAT for GPT-OSS](https://developer.nvidia.com/blog/fine-tuning-gpt-oss-for-accuracy-and-performance-with-quantization-aware-training/)
- [LMSYS: GPT-OSS QAT](https://lmsys.org/blog/2025-08-28-gpt-oss-qat/)
- [HuggingFace: MoE Load Balancing Review](https://huggingface.co/blog/NormalUhr/moe-balance)

---

## Priority Matrix

| Priority | Feature | Effort | Impact | Target GPU |
|----------|---------|--------|--------|------------|
| **P0** | Split LoRA for MoE | Medium | 7-12x faster MoE training | A100 + H100 |
| **P0** | FP8 RL (GRPO/IPO) | Low | 1.6x throughput, 60% less VRAM | H100 only |
| **P0** | Auto packing + Triton kernels | None (auto) | 3x faster, 30-90% less VRAM | All |
| **P1** | GRPO long context (chunked batching) | Low | 7x longer context for RL | All |
| **P1** | Flex Attention for GPT-OSS | Low | 8x longer sequences | All |
| **P1** | Extended context curriculum | Low | 32K → 64K+ on H100 | 80GB+ |
| **P1** | Harmony format compliance reward | Low | Prevents infinite reasoning loops | All |
| **P1** | Expert utilization monitoring | Medium | Detects routing collapse | All |
| **P2** | SFT + QAT export pipeline | Medium | 97-100% MXFP4 quality retention | All |
| **P2** | Reduced router_aux_loss_coef | Low | Prevents expert collapse in FT | All |
| **P3** | NVFP4 deployment (future) | TBD | 2-3% better than MXFP4 | TRT-LLM |

---

## P0: Split LoRA for MoE

**What:** Reorders LoRA computation from `(loraB @ loraA.t) @ X` to `loraB @ (loraA @ X)`, avoiding materializing the full adapter delta per expert.

**Benchmarks:**
- GPT-OSS: up to 12x faster MoE training vs Transformers v4
- 35%+ VRAM reduction
- GPT-OSS at 16K context: 55.13 GB vs OOM on Transformers v5

**Implementation:**
```python
import os
# Auto-select backend based on GPU
os.environ["UNSLOTH_MOE_BACKEND"] = "grouped_mm"   # H100/B200
os.environ["UNSLOTH_MOE_BACKEND"] = "unsloth_triton"  # A100
```

No model code changes needed — Unsloth applies this automatically when the env var is set.

---

## P0: FP8 Reinforcement Learning

**What:** FP8 precision for RL workflows. Frozen LoRA weights in FP8, optimizer states in higher precision. vLLM-backed inference for generation.

**Benchmarks:**
- 60% less VRAM
- 1.4x faster RL inference via vLLM
- 1.6x higher throughput on H100
- Loss curves match BF16 — no accuracy degradation

**Implementation:**
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="openai/gpt-oss-20b",
    load_in_fp8=True,       # NEW: FP8 weights (H100 only)
    fast_inference=True,     # NEW: vLLM-backed RL inference
    max_seq_length=32768,
)
```

**GPU requirement:** H100+ only. A100 falls back to 4-bit QLoRA.

---

## P0: Auto Packing + Triton Kernels

**What:** Automatically enabled in latest Unsloth. Fused varlen RoPE + int64 Triton kernels, padding-free training, uncontaminated packing.

**Impact:** 3x faster, 30-90% less VRAM. No code changes required.

---

## P1: GRPO Long Context (Chunked Batching)

**What:** Flattened sequence chunking, hidden states chunking, activation offloading for GRPO.

**Benchmarks:**
- GPT-OSS QLoRA: 380K context on 192GB B200
- Qwen3-8B GRPO: 110K context on 80GB H100
- GPT-OSS on 24GB: 20K context

**Implementation:**
```python
from trl import GRPOConfig
training_args = GRPOConfig(
    unsloth_grpo_mini_batch=3,
    unsloth_logit_chunk_multiplier=2,
    # Both auto-tune by default
)
```

---

## P1: Extended Context Curriculum

**Current (v1):**
```yaml
schedule:
  - seq_length: 4096    # steps: 1000
  - seq_length: 8192    # steps: 2000
  - seq_length: 16384   # steps: 3500
  - seq_length: 32768   # steps: 5000
```

**Proposed (v2 — H100 80GB):**
```yaml
schedule:
  - seq_length: 4096    # steps: 1000  — Single-file fixes
  - seq_length: 8192    # steps: 2000  — Multi-file (2-3 files)
  - seq_length: 16384   # steps: 3500  — Full module navigation
  - seq_length: 32768   # steps: 5000  — Large repo exploration
  - seq_length: 65536   # steps: 7000  — Full codebase reasoning (NEW)
```

---

## P1: Harmony Format Compliance Reward

**Problem:** Model can enter infinite reasoning loops after fine-tuning (outputs `thinking` channel indefinitely without producing `content`).

**Proposed reward addition:**
```yaml
rewards:
  # ... existing rewards ...
  harmony_format_compliant: 0.2      # Correct Harmony structure
  infinite_reasoning_loop: -0.8      # >10 thinking tokens without content
```

---

## P2: SFT + QAT Export Pipeline

**Problem:** PTQ after SFT loses 10-40% accuracy when quantizing back to MXFP4.

**Solution:** NVIDIA TensorRT Model Optimizer QAT pipeline:
1. Fine-tune in BF16 (existing pipeline)
2. QAT pass at reduced LR (1e-5) using `mtq.quantize()`
3. Export back to MXFP4

**Benchmark:** SFT + QAT achieves 97-100% quality retention vs 59-89% for PTQ.

---

## Known Issues to Address

| Issue | Impact | Fix |
|-------|--------|-----|
| FA3 incompatible with GPT-OSS backward pass | Incorrect training loss | Use Flex Attention or `eager` |
| FP16 diverges at ~step 50 | Training blows up on T4 | Always use BF16 |
| MXFP4 weights are `nn.Parameter` not `nn.Linear` | Breaks standard quant | Unsloth handles; verify version |
| Infinite reasoning loops | Bad inference output | Harmony compliance reward |
| `router_aux_loss_coef: 0.02` too high for FT | Expert collapse risk | Reduce to 0.005-0.01 |
| Potential config discrepancy | Wrong attention head count | Verify 24 vs 64 heads against model card |

---

## GPU Tier Configs (v2)

```python
GPU_CONFIGS_V2 = {
    "a100_40gb": {
        "moe_backend": "unsloth_triton",
        "load_mode": "4bit",            # QLoRA
        "fast_inference": False,         # No vLLM on A100
        "lang_rust": {"batch": 1, "grad_accum": 8, "seq_len": 8192, "max_steps": 3000},
        "core_agent": {"batch": 1, "grad_accum": 4, "seq_len": 12288, "max_steps": 2000},
        "ipo": {"batch": 1, "grad_accum": 8, "seq_len": 12288, "max_steps": 1000},
        "grpo": {"batch": 1, "grad_accum": 4, "seq_len": 16384, "max_steps": 2000, "num_gen": 2},
    },
    "a100_80gb": {
        "moe_backend": "unsloth_triton",
        "load_mode": "4bit",
        "fast_inference": False,
        "lang_rust": {"batch": 1, "grad_accum": 8, "seq_len": 8192, "max_steps": 5000},
        "core_agent": {"batch": 1, "grad_accum": 4, "seq_len": 16384, "max_steps": 3000},
        "ipo": {"batch": 1, "grad_accum": 16, "seq_len": 16384, "max_steps": 2000},
        "grpo": {"batch": 1, "grad_accum": 8, "seq_len": 32768, "max_steps": 5000, "num_gen": 4},
    },
    "h100_80gb": {
        "moe_backend": "grouped_mm",
        "load_mode": "fp8",             # FP8 RL
        "fast_inference": True,          # vLLM-backed RL inference
        "lang_rust": {"batch": 2, "grad_accum": 4, "seq_len": 8192, "max_steps": 5000},
        "core_agent": {"batch": 1, "grad_accum": 4, "seq_len": 16384, "max_steps": 3000},
        "ipo": {"batch": 1, "grad_accum": 16, "seq_len": 16384, "max_steps": 2000},
        "grpo": {"batch": 1, "grad_accum": 8, "seq_len": 65536, "max_steps": 7000, "num_gen": 4},
    },
}
```

---

## Key Numbers

| Metric | v1 | v2 (projected) |
|--------|-----|-----------------|
| MoE training speed | 1x (baseline) | 7-12x (Split LoRA) |
| GRPO throughput (H100) | 1x | 1.6x (FP8 RL) |
| SFT speed | 1x | 3x (auto packing) |
| Max GRPO context (H100 80GB) | 32K | 65K+ |
| VRAM usage (GRPO, H100) | ~64 GB | ~40 GB (FP8) |
| MXFP4 export quality | 59-89% (PTQ) | 97-100% (QAT) |
