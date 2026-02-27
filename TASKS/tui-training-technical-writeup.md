# TUI Training Pipeline: Technical Writeup

**Date**: 2026-02-26
**Author**: ML Infrastructure Team
**Status**: Validated (quick test pass on RunPod H100 NVL 96GB)

---

## 1. Overview

The TUI (Tool Use Interface) training pipeline implements sequential fine-tuning with a curriculum for the GPT-OSS 20B Mixture-of-Experts model. The goal is to progressively shape a general-purpose language model into a specialized coding agent that understands tool calling, multi-turn debugging, and code patching.

The pipeline runs as a single orchestrated process on a dedicated GPU node (H100 80GB+), executing six phases in sequence:

```
Data Prep -> Tool Calling SFT -> Merge -> Agent SFT -> IPO -> GRPO -> Export
```

Each phase builds on the learned representations of the previous phase. Quality gates between phases ensure training is converging before committing compute to the next stage. The entire pipeline supports auto-resume: if the process is interrupted at any point, restarting it will skip completed phases and resume the current one from the latest checkpoint.

The TUI checkpoint produced by this pipeline serves as the foundation for downstream language-specific code training (Rust, Python, TypeScript, Go). Those language pipelines load the TUI merged model as their base and train language-specific coding capabilities on top of the general agent skills established here.

### Key Files

| File | Purpose |
|------|---------|
| `scripts/run_tui_pipeline.sh` | Main orchestrator -- runs all 6 phases sequentially |
| `scripts/start_tui_training.sh` | Screen wrapper for persistent sessions |
| `scripts/tui_data_prep.py` | Downloads and formats training data from HuggingFace |
| `scripts/13_train_lang_adapter.py` | Phase 1: Tool Calling SFT training |
| `scripts/14_train_core_agent.py` | Phase 2: Agent SFT training |
| `scripts/17_ipo_preference.py` | Phase 3: IPO preference optimization |
| `scripts/18_grpo_rl.py` | Phase 4: GRPO reinforcement learning |
| `scripts/19_merge_adapter.py` | Phases 1.5 and 5: Adapter merge and export |

---

## 2. Architecture

### Two-Script Design

The pipeline uses a two-script architecture to support persistent, resumable training on cloud GPU instances:

**`start_tui_training.sh`** is a thin wrapper that:
- Checks if a `screen` session named `tui_training` already exists
- Installs `screen` if missing (for bare RunPod containers)
- Launches `run_tui_pipeline.sh` inside a detached screen session with full logging
- Auto-attaches to the session in interactive terminals

```bash
# Start training
bash scripts/start_tui_training.sh --quick-test

# Reconnect after SSH disconnect
screen -r tui_training

# Detach (training continues)
Ctrl+A, D
```

**`run_tui_pipeline.sh`** is the main orchestrator that:
- Parses CLI arguments (`--quick-test`, `--skip-data`, `--batch-profile`)
- Configures environment variables (WANDB, CUDA, HuggingFace)
- Redirects all caches to the `/workspace` volume
- Runs each phase function in sequence
- Checks quality gates between phases
- Produces a summary report with gate results

### Auto-Resume Logic

Each phase function checks three conditions before running:

1. **Phase complete**: If `checkpoints/<phase>/final/adapter_config.json` exists, skip the phase entirely.
2. **Partial progress**: If `checkpoints/<phase>/checkpoint-*` directories exist, pass `--resume_from_checkpoint <latest>` to the training script.
3. **Fresh start**: If neither exists, run from scratch.

This means the pipeline is fully idempotent. Running `bash scripts/run_tui_pipeline.sh` multiple times will never repeat completed work. If the SSH session drops, the screen session keeps running. If the GPU node crashes, restarting the script picks up where it left off.

### Phase Dependency Chain

```
Phase 0 (Data Prep)
    |
    v
Phase 1 (Tool Calling SFT) -- uses 13_train_lang_adapter.py
    |  Quality gate: avg loss <= 2.0 [HARD]
    v
Phase 1.5 (Merge) -- uses 19_merge_adapter.py
    |  Quality gate: model size >= 5GB + smoke test [HARD]
    v
Phase 2 (Agent SFT) -- uses 14_train_core_agent.py
    |  Quality gate: avg loss <= 1.8 [HARD]
    v
Phase 3 (IPO) -- uses 17_ipo_preference.py
    |  Quality gate: loss logged [SOFT -- always passes]
    v
Phase 4 (GRPO) -- uses 18_grpo_rl.py
    |  Quality gate: reward accuracy > 0.55 [SOFT]
    |  NOTE: Skipped in TUI-only pipeline (no language-specific task data)
    v
Phase 5 (Export) -- uses 19_merge_adapter.py
    |  Outputs: HuggingFace format + GGUF Q4
    v
Summary Report
```

---

## 3. Model Architecture

### GPT-OSS 20B MoE

The base model is OpenAI's GPT-OSS 20B (Apache 2.0), a Mixture-of-Experts architecture:

| Property | Value |
|----------|-------|
| Total parameters | ~20.9B |
| Active parameters per token | ~3.6B (top-4 of 32 experts) |
| Hidden size | 2,880 |
| Layers | 24 |
| Attention heads | 24 (3 KV heads, group size 8) |
| Experts per layer | 32 |
| Expert routing | Top-4 |
| Expert FFN size | 7,680 |
| Max context | 128K tokens |
| Positional encoding | RoPE (theta=500,000) |
| Tokenizer | o200k_harmony (201K vocab) |
| Weight format | MXFP4 (MoE layers) |
| Chat format | Harmony (mandatory) |

The MoE architecture means that while the model has 20.9B total parameters, only ~3.6B are active per forward pass. The router selects the top-4 experts per token, so each token uses 4 of the 32 expert FFN layers plus the shared attention layers.

### Unsloth QLoRA

All training phases use Unsloth's optimized QLoRA implementation, which provides:

- **4-bit quantization**: Base weights stay in 4-bit, LoRA adapters train in BF16
- **Flex Attention**: O(N) memory attention during training (FA2/FA3 are incompatible with GPT-OSS due to attention sinks)
- **Tiled MLP**: Chunks MLP operations along sequence dimension for ~40% VRAM savings
- **Embedding offload**: Moves embedding/LM-head to CPU during training, saving ~1GB VRAM
- **Gradient checkpointing**: CPU offload variant ("unsloth" mode)

A critical implementation detail is the MoE expert LoRA targeting. Unsloth Bug #3405 causes default target modules (`gate_proj`, `up_proj`, `down_proj`) to silently miss expert FFN layers. The `apply_lora_config()` function in `pipeline_lib/unsloth_utils.py` auto-detects MoE models and corrects the target modules to use singular names (`gate_up_proj`, `down_proj`) that Unsloth maps to fused expert layers. This increases trainable parameters from ~31.8M (attention-only) to ~200M+ (attention + experts).

### LoRA Configuration Per Phase

| Phase | LoRA Rank | Trainable Params | Target Modules |
|-------|-----------|------------------|----------------|
| Tool Calling SFT | 64 | ~739M | Attention + Expert FFN |
| Agent SFT | 128 | ~1.48B | Attention + Expert FFN |
| IPO | Inherited from Agent SFT | ~1.48B | Attention + Expert FFN |

The rank doubles from Phase 1 to Phase 2 because agent trajectories require modeling more complex multi-turn reasoning patterns than single-turn tool calling.

---

## 4. Training Phases

### Phase 0: Data Preparation

**Script**: `scripts/tui_data_prep.py`

Downloads and formats three categories of training data from HuggingFace Hub. All data is converted to Harmony format (GPT-OSS's mandatory chat format) before saving as HuggingFace Datasets on disk.

The data prep script is idempotent -- if the output directory already contains data, it skips the download. In `--quick-test` mode, it downloads small subsets (hundreds of examples instead of tens of thousands).

Data categories:
- **Tool calling** (Phase 1): Function calling examples formatted with tool schemas and invocations
- **Agent trajectories** (Phase 2): Multi-turn coding agent conversations with tool use
- **Preferences** (Phase 3): Chosen/rejected pairs for IPO training

See Section 5 for detailed data pipeline documentation.

### Phase 1: Tool Calling SFT

**Script**: `scripts/13_train_lang_adapter.py`
**Checkpoint**: `checkpoints/tool_calling_sft/final/`
**Quality Gate**: Average loss (last 10 steps) <= 2.0 [HARD]

Trains the model's foundational tool-calling capability. Uses a LoRA rank of 64 and trains on formatted function-calling examples from Glaive, xLAM, and Hermes datasets.

The training uses `13_train_lang_adapter.py` with the data path overridden via CLI:

```bash
python3 scripts/13_train_lang_adapter.py \
    --train_data_path data/coding_tui/tool_calling/train \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 6 \
    --max_steps -1 \
    --num_train_epochs 1 \
    --output_dir checkpoints/tool_calling_sft \
    --save_steps 250
```

This phase teaches the model to:
- Parse tool/function schemas from developer prompts
- Generate structured tool calls with correct JSON arguments
- Handle tool results and continue the conversation

### Phase 1.5: Merge Tool Calling Adapter

**Script**: `scripts/19_merge_adapter.py`
**Checkpoint**: `checkpoints/gpt-oss-20b-coding-tui-merged/hf/`
**Quality Gate**: Model directory >= 5.0 GB + smoke test [HARD]

Merges the tool-calling LoRA adapter into the base model weights. This is necessary because Phase 2 (Agent SFT) applies a new LoRA on top -- you cannot stack two LoRA adapters, so the first must be merged into the base.

The merge produces a full-weight model (~39 GB for GPT-OSS 20B). The quality gate validates:
1. The output directory exceeds 5 GB (catches truncated/corrupt merges)
2. `config.json` is valid and parseable
3. SafeTensor files exist and are loadable

A smoke test validates that the merged model produces coherent output by checking for unique character diversity in a test generation.

### Phase 2: Agent SFT

**Script**: `scripts/14_train_core_agent.py`
**Checkpoint**: `checkpoints/agent_sft/final/`
**Quality Gate**: Average loss (last 10 steps) <= 1.8 [HARD]

Trains multi-turn agent behavior on top of the merged tool-calling model. Uses a LoRA rank of 128 (doubled from Phase 1 due to the complexity of multi-turn reasoning). The base model is the merged output from Phase 1.5, not the raw GPT-OSS 20B.

```bash
python3 scripts/14_train_core_agent.py \
    --base_model checkpoints/gpt-oss-20b-coding-tui-merged/hf \
    --no-merge-lang-adapter \
    --train_data_path data/coding_tui/agent_traj/train \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 24 \
    --max_steps -1 \
    --num_train_epochs 3 \
    --output_dir checkpoints/agent_sft \
    --eval_strategy no \
    --max_seq_length 32768
```

Key flags:
- `--no-merge-lang-adapter`: Prevents the script from trying to merge a lang_rust adapter (not relevant for TUI)
- `--eval_strategy no`: Disabled because Unsloth Bug #3363 causes Flex Attention to produce gibberish with left padding during evaluation. Loss is validated from `trainer_state.json` instead.
- `--max_seq_length 32768`: Agent trajectories are long; 32K tokens accommodates most multi-turn conversations.

This phase supports optional sequence packing via `pipeline_lib/pack_sft.py`. If a pre-packed dataset exists at `data/coding_tui/agent_traj/train_packed_32768/`, it will be used automatically. Otherwise, the pipeline attempts to pack on the fly.

This phase teaches the model to:
- Engage in multi-turn debugging conversations
- Read files, apply patches, and run tests
- Diagnose failures and retry with corrected approaches
- Produce working code (not just analysis)

### Phase 3: IPO Preference Optimization

**Script**: `scripts/17_ipo_preference.py`
**Checkpoint**: `checkpoints/agent_sft_ipo/final/`
**Quality Gate**: Loss logged [SOFT -- always passes]

Applies Identity Preference Optimization (IPO) using preference pairs to align the model's outputs toward higher-quality responses. IPO was chosen over DPO for this pipeline because:

- More stable with noisy preference labels from automated ranking
- Does not require a separate reference model (saves VRAM)
- Less prone to reward hacking

```bash
python3 scripts/17_ipo_preference.py \
    --checkpoint checkpoints/agent_sft/final \
    --train_data_path data/coding_tui/preference/train \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --output_dir checkpoints/agent_sft_ipo \
    --beta 0.1
```

The IPO beta parameter (0.1) controls the strength of the preference signal. Lower values produce more conservative updates. The preference data includes both real human preference data (Anthropic hh-rlhf) and synthetic code quality preferences (CodeFeedback with truncated rejected responses).

IPO uses a soft quality gate because preference training loss trajectories are less predictable than SFT loss -- the absolute loss value is less meaningful. The pipeline logs it for monitoring but always proceeds to the next phase.

### Phase 4: GRPO RL Training

**Script**: `scripts/18_grpo_rl.py`
**Checkpoint**: `checkpoints/agent_sft_grpo/final/`
**Quality Gate**: Reward accuracy > 0.55 [SOFT]

Group Relative Policy Optimization (GRPO) uses execution-based rewards to further improve the model. For each prompt, the model generates N completions, each is evaluated against execution criteria (compilation, test passing, lint cleanliness), and the model is optimized using group-relative advantages.

**In the TUI-only pipeline, GRPO is skipped.** GRPO requires language-specific task data (e.g., `data/rust/grpo/tasks.jsonl` containing coding tasks with test suites for reward computation). Since the TUI pipeline trains general agent skills without a target language, there is no task data to drive GRPO rewards.

GRPO is activated when the TUI checkpoint is used as the base for a language-specific training session (e.g., Rust, Python). Those downstream pipelines generate mutation-based task data that provides concrete execution rewards.

```bash
# In the TUI pipeline, this check causes GRPO to skip:
if [ ! -f "$grpo_task_source" ]; then
    log "Skipping GRPO -- run language-specific training to generate task data."
fi
```

### Phase 5: Export

**Script**: `scripts/19_merge_adapter.py`
**Output**: `checkpoints/gpt-oss-20b-coding-tui-export/`

Merges the best available adapter (priority: GRPO > IPO > Agent SFT > Tool Calling) into the merged base model and exports in two formats:

- **HuggingFace** (`hf/`): Full-precision SafeTensor weights for inference servers
- **GGUF Q4** (`gguf_q4/`): 4-bit quantized format for llama.cpp and compatible runtimes

The export phase detects the base model automatically. If the adapter comes from Phase 2+ (Agent SFT or later), it uses the merged tool-calling model as the base. If only Phase 1 completed, it merges the tool-calling adapter directly from the raw GPT-OSS 20B.

GGUF export requires `cmake` and `libcurl`, which are installed automatically if missing:

```bash
if ! command -v cmake &>/dev/null; then
    apt-get update -qq && apt-get install -y -qq cmake libcurl4-openssl-dev
fi
```

If GGUF conversion fails but HuggingFace export succeeds, the pipeline considers the export successful and logs a warning. GGUF can be regenerated separately.

---

## 5. Data Pipeline

### Harmony Format

All GPT-OSS training data must use the Harmony format -- the model degrades without it. Harmony is implemented in `scripts/dataset_formatters/harmony.py` and uses special tokens to denote roles and structured content:

```
<|developer|>
You are a coding agent. Use tools to read files...
<|user|>
Fix the failing test in src/parser.rs
<|thinking|>
The test name suggests a parsing issue...
<|/thinking|>
<|tool_call|>
{"id": "1", "name": "run_tests", "arguments": {"cmd": "cargo test test_parse"}}
<|tool_result|>
<|tool_call_id|>1
FAILED: thread 'test_parse' panicked at 'assertion failed'
<|assistant|>
Let me look at the failing test source...
<|endoftext|>
```

The Harmony encoder (`encode_harmony_messages()`) first attempts to use the official `openai_harmony` package, falling back to a manual encoding that produces compatible output.

### Tool Calling Data (Phase 1)

Three HuggingFace datasets are downloaded and formatted:

| Dataset | Source | Examples (full) | Examples (quick test) |
|---------|--------|----------------|-----------------------|
| Glaive Function Calling v2 | `glaiveai/glaive-function-calling-v2` | 113K | 500 |
| xLAM-60K | `Salesforce/xlam-function-calling-60k` | 60K | 200 |
| Hermes Function Calling v1 | `NousResearch/hermes-function-calling-v1` | Full | 300 |

Each dataset has a dedicated formatter:
- **Glaive**: Uses `format_glaive_function_calling()` from `dataset_formatters/function_calling`
- **xLAM**: Uses `_format_xlam()` -- parses JSON tool schemas and answers into Harmony tool_call format
- **Hermes**: Uses `format_hermes_function_calling()` from `dataset_formatters/function_calling`

The xLAM dataset is gated on HuggingFace and requires an `HF_TOKEN` with access granted. The download is wrapped in a try/except for resilience -- if the token is missing or access is denied, the pipeline continues with Glaive + Hermes data.

Output: `data/coding_tui/tool_calling/train/` (HuggingFace Dataset on disk)

### Agent Trajectory Data (Phase 2)

Three datasets covering multi-turn coding interactions:

| Dataset | Source | Examples (full) | Examples (quick test) |
|---------|--------|----------------|-----------------------|
| code-act | `xingyaoww/code-act` (splits: codeact, general) | Full | 100/split |
| commitpackft | `bigcode/commitpackft` (6 languages) | 5K/lang | 50/lang |
| EditPackFT | `nuprl/EditPackFT` | 10K | 300 |

Formatters:
- **code-act**: `_format_code_act()` -- multi-turn conversations with tool use, prepends a developer prompt instructing agent behavior. Uses `reasoning_effort="high"` for Harmony encoding.
- **commitpackft**: `_format_commitpackft()` -- commit-level code diffs formatted as "apply this change" tasks. Filters out diffs larger than 8000 characters. Uses `reasoning_effort="medium"`.
- **EditPackFT**: `_format_editpackft()` -- instruction-based code edits. Same size filter and reasoning effort as commitpackft.

The coding agent developer prompt is consistent across all agent trajectory data:

```python
CODING_AGENT_DEV_PROMPT = (
    "You are a coding agent. Use tools to read files, write code, run tests, and "
    "complete programming tasks. Do not just analyze - always take action and produce "
    "working code. After making changes, verify they work by running the relevant tests. "
    "If a tool call fails, diagnose and retry with corrected parameters."
)
```

Output: `data/coding_tui/agent_traj/train/` (HuggingFace Dataset on disk)

### Preference Data (Phase 3)

Two datasets providing chosen/rejected pairs:

| Dataset | Source | Examples (full) | Examples (quick test) |
|---------|--------|----------------|-----------------------|
| hh-rlhf | `Anthropic/hh-rlhf` | 20K | 300 |
| CodeFeedback | `m-a-p/CodeFeedback-Filtered-Instruction` | 10K | 200 |

Formatters:
- **hh-rlhf**: `_format_hh_rlhf()` -- parses Anthropic's `\n\nHuman: / \n\nAssistant:` format into Harmony messages. The last assistant turn is split into chosen/rejected variants.
- **CodeFeedback**: `_format_code_feedback()` -- creates synthetic rejected responses by truncating the good answer at 30% and appending a stalling phrase ("I would need to analyze this further before proceeding."). This teaches the model to prefer complete, actionable responses over hedging.

The dataset is saved with columns: `text`, `pref_prompt`, `pref_chosen`, `pref_rejected`. The IPO trainer renames `pref_*` columns to match DPOTrainer's expected `prompt`/`chosen`/`rejected` format.

Output: `data/coding_tui/preference/train/` (HuggingFace Dataset on disk)

---

## 6. H100 Configuration

### Batch Profiles

The pipeline offers three batch profiles for different VRAM/speed trade-offs:

**Aggressive** (maximizes throughput, higher OOM risk):

| Phase | Batch Size | Grad Accum | Effective Batch | Seq Length |
|-------|-----------|------------|-----------------|------------|
| Tool Calling SFT | 10 | 5 | 50 | 8,192 |
| Agent SFT | 4 | 6 | 24 | 32,768 |
| IPO | 4 | 8 | 32 | 8,192 |
| GRPO | 2 | 8 | 16 | 65,536 |

**Balanced** (default -- recommended for H100 80/96GB):

| Phase | Batch Size | Grad Accum | Effective Batch | Seq Length |
|-------|-----------|------------|-----------------|------------|
| Tool Calling SFT | 8 | 6 | 48 | 8,192 |
| Agent SFT | 1 | 24 | 24 | 32,768 |
| IPO | 2 | 16 | 32 | 8,192 |
| GRPO | 1 | 16 | 16 | 65,536 |

**Conservative** (lowest VRAM usage, for constrained environments):

| Phase | Batch Size | Grad Accum | Effective Batch | Seq Length |
|-------|-----------|------------|-----------------|------------|
| Tool Calling SFT | 4 | 12 | 48 | 8,192 |
| Agent SFT | 1 | 24 | 24 | 32,768 |
| IPO | 1 | 32 | 32 | 8,192 |
| GRPO | 1 | 16 | 16 | 65,536 |

**Quick Test** overrides (for validation runs):

| Phase | Batch Size | Grad Accum | Seq Length | Max Steps |
|-------|-----------|------------|------------|-----------|
| Tool Calling SFT | 2 | 2 | 4,096 | 50 |
| Agent SFT | 1 | 2 | 4,096 | 50 |
| IPO | 1 | 2 | 4,096 | 50 |
| GRPO | 1 | 2 | 4,096 | 50 |

### VRAM Management

The pipeline employs several strategies to fit within H100 80/96GB VRAM:

1. **4-bit quantization** (QLoRA): Base model weights stay in 4-bit, only LoRA adapters train in BF16
2. **Gradient accumulation**: Small micro-batches (1-8) with high accumulation steps (6-32) achieve large effective batch sizes without VRAM pressure
3. **Expandable CUDA segments**: `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` reduces memory fragmentation
4. **Tiled MLP**: Chunks MLP operations along sequence dimension for ~40% VRAM savings (enables 290K+ context with QLoRA on H100)
5. **Embedding offload**: Moves embedding/LM-head to CPU during training, saving ~1GB VRAM
6. **Flex Attention**: O(N) memory attention pattern during training (vs. O(N^2) for standard attention)
7. **Gradient checkpointing**: Unsloth's CPU-offload variant reduces activation memory

---

## 7. RunPod Deployment

### Container Setup

The pipeline targets RunPod GPU pods with the following configuration:

- **GPU**: NVIDIA H100 NVL 96GB (or H100 80GB SXM)
- **Container disk**: 20GB (OS + packages only)
- **Volume**: 200GB mounted at `/workspace` (all data, checkpoints, and caches)
- **Image**: Standard PyTorch RunPod template (or custom with Unsloth pre-installed)

### SSH and Environment Variables

RunPod's SSH sessions do not inherit environment variables set on the container's PID 1 process. This causes `HF_TOKEN` (set via RunPod Secrets) to be invisible in SSH sessions. The pipeline handles this with a fallback:

```bash
# RunPod sets env vars on PID 1 but SSH sessions don't inherit them.
# Source from /proc/1/environ as fallback.
if [ -z "${HF_TOKEN:-}" ] && [ -f /proc/1/environ ]; then
    _hf_from_proc=$(tr '\0' '\n' < /proc/1/environ 2>/dev/null \
        | grep '^HF_TOKEN=' | cut -d= -f2-)
    if [ -n "$_hf_from_proc" ]; then
        export HF_TOKEN="$_hf_from_proc"
    fi
fi
```

Additionally, it checks for the `RUNPOD_SECRET_HF_TOKEN` naming convention used by newer RunPod secret injection.

### Cache and Disk Management

RunPod container disks are typically 20GB, which is insufficient for HuggingFace model downloads, torch caches, and temporary files. The pipeline redirects all caches to the persistent `/workspace` volume:

```bash
if [ -d "/workspace" ]; then
    export HF_HOME="/workspace/.cache/huggingface"
    export TORCH_HOME="/workspace/.cache/torch"
    export XDG_CACHE_HOME="/workspace/.cache"
    export TMPDIR="/workspace/tmp"
    export TEMP="/workspace/tmp"
    export TMP="/workspace/tmp"
    mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR"
fi
```

Without this redirection, the 20GB container disk fills up during the first model download (~10GB for GPT-OSS 20B quantized weights), causing the pipeline to fail with disk I/O errors.

### Screen Sessions

The `start_tui_training.sh` wrapper ensures training survives SSH disconnects:

```bash
# Launch in screen with logging
screen -dmS tui_training bash -c \
    "cd $PROJECT_DIR && bash scripts/run_tui_pipeline.sh $* 2>&1 | tee $LOG_FILE; \
     echo 'Pipeline finished. Press Enter to exit.'; read"
```

The screen session persists even if the SSH connection drops. After reconnecting:

```bash
screen -r tui_training   # Reattach to running session
```

---

## 8. Quality Gates

### Gate Types

The pipeline uses two types of quality gates:

**Hard gates** block progression to the next phase. If a hard gate fails, the pipeline exits with a non-zero return code. The operator must diagnose the issue and restart.

**Soft gates** log a warning but allow the pipeline to continue. These are used for phases where the quality metric is less predictable or where skipping would waste more compute than proceeding.

### Gate Thresholds

| Phase | Metric | Threshold | Type | Implementation |
|-------|--------|-----------|------|----------------|
| Tool Calling SFT | Avg loss (last 10 steps) | <= 2.0 | HARD | Read from `trainer_state.json` |
| Merge | Model directory size | >= 5.0 GB | HARD | `os.walk()` size calculation |
| Merge | Smoke test | config.json valid + safetensors exist | HARD | Python validation |
| Agent SFT | Avg loss (last 10 steps) | <= 1.8 | HARD | Read from `trainer_state.json` |
| IPO | Loss trajectory | Logged only | SOFT | Always passes |
| GRPO | Reward accuracy | > 0.55 | SOFT | Read from `trainer_state.json` |

### Gate State Tracking

All gate results are persisted to `logs/quality_gates.json`:

```json
{
  "tool_calling": {"loss": "1.458", "passed": "true"},
  "merge": {"size_gb": "38.98", "passed": "true", "smoke_test": "true"},
  "agent_sft": {"loss": "0.798", "passed": "true"},
  "ipo": {"loss": "0.542", "passed": "true"},
  "grpo": {"status": "skipped_no_tasks"}
}
```

The `get_avg_loss()` helper reads the last N entries from `trainer_state.json` (which Hugging Face Trainer writes at each logging step) and computes the average:

```bash
get_avg_loss() {
    local ckpt_dir="$1"
    local n="${2:-10}"
    python3 -c "
import json, glob, os
state_files = sorted(glob.glob('${ckpt_dir}/*/trainer_state.json'))
# ... finds latest trainer_state.json and averages last N losses
"
}
```

### Why eval_strategy is Disabled

Agent SFT uses `--eval_strategy no` because of Unsloth Bug #3363: Flex Attention produces incorrect output with left-padded sequences. Since evaluation batches use left padding for uniform sequence lengths, running eval during training would produce misleading loss values. Instead, the pipeline validates training loss from the trainer state log after training completes.

---

## 9. Quick Test Results

The pipeline was validated with a `--quick-test` run on an H100 NVL 96GB RunPod pod. Quick test mode reduces dataset sizes, sequence lengths, and caps training at 50 steps per phase.

| Phase | Status | Key Metrics |
|-------|--------|-------------|
| Phase 0: Data Prep | PASS | 800 tool calling + 500 agent traj + 497 preference examples from HuggingFace |
| Phase 1: Tool Calling SFT | PASS | Loss: 3.95 -> 0.66 (50 steps, 5.5 min), gate: 1.458 <= 2.0, LoRA rank 64, 739M trainable params |
| Phase 1.5: Merge | PASS | 38.98 GB merged model, smoke test passed (coherent output, unique_chars=43) |
| Phase 2: Agent SFT | PASS | Loss: 0.798 (50 steps, 7.4 min), gate: 0.798 <= 1.8, LoRA rank 128, 1.48B trainable params |
| Phase 3: IPO | PASS | 497 preference pairs, ref log prob precomputation (~10 min), 50 IPO steps |
| Phase 4: GRPO | SKIPPED | TUI-only pipeline -- no language-specific task data |
| Phase 5: Export | PASS | HF + GGUF formats, ~39 GB each |

### Analysis

**Tool Calling SFT** converged rapidly (3.95 -> 0.66 in 50 steps), suggesting the base model already has latent function-calling knowledge that LoRA fine-tuning surfaces quickly. The gate threshold of 2.0 was passed with significant margin (1.458 average over last 10 steps).

**Merge** produced a 38.98 GB model, consistent with GPT-OSS 20B's expected BF16 checkpoint size. The smoke test confirmed coherent output generation, verifying no weight corruption during the merge.

**Agent SFT** started from a lower loss (0.798 at gate check) than Tool Calling SFT because the merged base model already has tool-calling capability. The higher LoRA rank (128 vs 64) and longer sequences (4096 in quick test) resulted in slightly longer step times (7.4 min total vs 5.5 min).

**IPO** spent significant time on reference log probability precomputation (~10 min for 497 pairs). This is a one-time cost at the start of IPO training where the model computes log probabilities for all chosen/rejected pairs under the reference (frozen) model. In production with larger datasets, this step will take proportionally longer.

**GRPO was correctly skipped** because the TUI-only pipeline has no language-specific task data to drive execution rewards. This is by design -- GRPO is activated during downstream language-specific training.

**Export** successfully produced both HuggingFace SafeTensor format (~39 GB) and GGUF Q4 quantized format. The GGUF conversion required installing `cmake` and `libcurl` in the container.

---

## 10. Known Issues and Solutions

### Issue 1: Container Disk Overflow (20 GB)

**Symptom**: Pipeline crashes during HuggingFace model download with `OSError: No space left on device`. The RunPod container disk (20 GB) fills up because HuggingFace, PyTorch, and pip caches all default to `~/.cache` on the container filesystem.

**Solution**: Redirect all caches and temp directories to `/workspace` (200 GB persistent volume):

```bash
export HF_HOME="/workspace/.cache/huggingface"
export TORCH_HOME="/workspace/.cache/torch"
export XDG_CACHE_HOME="/workspace/.cache"
export TMPDIR="/workspace/tmp"
```

This is configured automatically by `run_tui_pipeline.sh` when `/workspace` exists.

**Prevention**: When creating RunPod pods, always attach a volume (>= 200 GB) and mount it at `/workspace`. The container disk should only hold the OS and installed packages.

### Issue 2: RunPod SSH Sessions Don't Inherit Container Env Vars

**Symptom**: `HF_TOKEN` is set as a RunPod secret but `echo $HF_TOKEN` returns empty in SSH sessions. Gated HuggingFace datasets (e.g., xLAM-60K) fail to download.

**Root Cause**: RunPod injects secrets as environment variables into the container's PID 1 process. SSH sessions spawn new shells that do not inherit PID 1's environment.

**Solution**: The pipeline reads `/proc/1/environ` (the environment of PID 1) as a fallback:

```bash
if [ -z "${HF_TOKEN:-}" ] && [ -f /proc/1/environ ]; then
    _hf_from_proc=$(tr '\0' '\n' < /proc/1/environ | grep '^HF_TOKEN=' | cut -d= -f2-)
    export HF_TOKEN="$_hf_from_proc"
fi
```

**Alternative**: Add `export HF_TOKEN=hf_xxxxx` to `~/.bashrc` on the pod, or use `huggingface-cli login` before starting the pipeline.

### Issue 3: GRPO Script Unsupported CLI Arguments

**Symptom**: GRPO phase fails immediately with argument parsing errors when passed `--num_generations`, `--reward_mode`, or `--developer_prompt`.

**Root Cause**: The GRPO script's CLI argument parser (`18_grpo_rl.py`) does not accept these arguments. They were present in an earlier version of the design document but were removed or renamed during implementation.

**Solution**: The pipeline script was updated to use only supported arguments:

```bash
python3 scripts/18_grpo_rl.py \
    --checkpoint "$grpo_base" \
    --per_device_train_batch_size "$GRPO_BATCH" \
    --gradient_accumulation_steps "$GRPO_GRAD" \
    --max_steps "$GRPO_MAX_STEPS" \
    --task_source "$grpo_task_source" \
    --output_dir checkpoints/agent_sft_grpo
```

`--num_generations` is now configured in the YAML config file. `--reward_mode` was replaced by the `--language` flag, which dispatches to the correct evaluator. `--developer_prompt` is set in the config.

### Issue 4: GRPO Needs Language-Specific Task Data

**Symptom**: GRPO fails or produces meaningless training because there is no task data with test suites to compute execution rewards.

**Root Cause**: GRPO relies on execution-based rewards (compilation success, test passing, lint cleanliness). These rewards require language-specific task data with test suites. The TUI pipeline teaches general agent skills and does not have a target language.

**Solution**: The pipeline now checks for task data existence and gracefully skips GRPO in TUI-only mode:

```bash
local grpo_task_source="data/rust/grpo/tasks.jsonl"
if [ ! -f "$grpo_task_source" ]; then
    log "Skipping GRPO -- run language-specific training to generate task data."
    update_gate "grpo" "status" '"skipped_no_tasks"'
    return 0
fi
```

GRPO is activated when the TUI checkpoint is used as the base model for a language-specific training pipeline (e.g., `scripts/18_grpo_rl.py --language rust`).

### Issue 5: GGUF Export Needs cmake and libcurl

**Symptom**: Export phase fails during GGUF conversion with `cmake: command not found` or linker errors related to `libcurl`.

**Root Cause**: GGUF export requires building `llama.cpp` from source, which needs `cmake` and `libcurl` development headers. Bare RunPod containers typically do not include these.

**Solution**: The pipeline installs dependencies non-interactively before export:

```bash
if ! command -v cmake &>/dev/null; then
    log "Installing cmake and libcurl for GGUF export..."
    apt-get update -qq && apt-get install -y -qq cmake libcurl4-openssl-dev 2>&1 | tail -1
fi
```

The `-qq` flags suppress verbose output, and `2>&1 | tail -1` reduces log noise to just the final status line. If GGUF export still fails, the pipeline considers HuggingFace format export alone as success.

### Issue 6: Gated HuggingFace Datasets (xLAM-60K)

**Symptom**: `Salesforce/xlam-function-calling-60k` download fails with `403 Forbidden` or `GatedRepoError`.

**Root Cause**: The xLAM-60K dataset requires accepting terms on the HuggingFace Hub before access is granted. Even with a valid `HF_TOKEN`, access must be explicitly approved for this dataset.

**Solution**: The data prep script wraps the download in a try/except:

```python
try:
    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    # ... format and add to examples
except Exception as e:
    print(f"  WARNING: xLAM download failed (gated dataset, needs HF_TOKEN): {e}")
```

The pipeline continues with Glaive + Hermes data if xLAM is unavailable. For production runs, operators should:
1. Visit https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
2. Accept the dataset terms while logged in
3. Ensure `HF_TOKEN` is set and has read access

---

## 11. Production Deployment Guide

### Prerequisites

1. **RunPod account** with H100 GPU access
2. **HuggingFace token** with access to gated datasets (xLAM-60K)
3. **RunPod secret**: Set `HF_TOKEN` in RunPod Secrets panel

### Step 1: Create Pod

Create a RunPod GPU pod:
- **GPU**: NVIDIA H100 NVL 96GB (or H100 80GB SXM)
- **Container disk**: 20 GB (minimum)
- **Volume**: 200+ GB, mounted at `/workspace`
- **Image**: `runpod/pytorch:2.5.1-py3.11-cuda12.4.1-devel-ubuntu22.04`

### Step 2: Clone and Install

```bash
cd /workspace
git clone <repository-url> llm-training-pipeline
cd llm-training-pipeline
pip install -e ".[gpt_oss]"
```

### Step 3: Launch Training

For production (full dataset, all epochs):

```bash
bash scripts/start_tui_training.sh --batch-profile balanced
```

For validation (small dataset, 50 steps per phase):

```bash
bash scripts/start_tui_training.sh --quick-test
```

If data is already prepared from a previous run:

```bash
bash scripts/start_tui_training.sh --skip-data --batch-profile aggressive
```

### Step 4: Monitor

```bash
# Attach to training session
screen -r tui_training

# Check quality gates
cat logs/quality_gates.json | python3 -m json.tool

# Check latest log
tail -f logs/tui_pipeline_*.log

# GPU utilization
nvidia-smi -l 5
```

### Step 5: Resume After Interruption

Simply restart the same command. The pipeline auto-detects completed phases and resumes from the latest checkpoint:

```bash
bash scripts/start_tui_training.sh --batch-profile balanced
```

### Step 6: Verify Output

After the pipeline completes, verify the exported model:

```bash
# Check export directory
ls -la checkpoints/gpt-oss-20b-coding-tui-export/

# Verify HuggingFace format
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('checkpoints/gpt-oss-20b-coding-tui-export/hf')
tokenizer = AutoTokenizer.from_pretrained('checkpoints/gpt-oss-20b-coding-tui-export/hf')
print(f'Model loaded: {model.config.model_type}')
print(f'Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B')
"

# Review gate results
cat logs/quality_gates.json
```

### Step 7: Use as Base for Language Training

The TUI checkpoint serves as the base model for language-specific code training:

```bash
# Example: Train Rust coding capabilities on top of TUI
python scripts/16_generate_mutations.py --language rust
python scripts/15_generate_trajectories.py --language rust
python scripts/14_train_core_agent.py --config configs/core_agent.yaml
python scripts/17_ipo_preference.py --config configs/ipo.yaml
python scripts/18_grpo_rl.py --config configs/grpo.yaml --language rust
python scripts/eval_coding_agent.py --config configs/rust_eval.yaml
```

The model already knows tool calling, multi-turn conversation, and code patching from TUI training. The language pipeline teaches language-specific debugging skills using mutation-derived trajectory data.

### Estimated Training Times (H100 80GB, balanced profile)

| Phase | Quick Test | Production (estimated) |
|-------|-----------|----------------------|
| Data Prep | ~5 min | ~30 min (full downloads) |
| Tool Calling SFT | ~5.5 min (50 steps) | ~2-4 hours (1 epoch over ~170K examples) |
| Merge | ~10 min | ~15 min |
| Agent SFT | ~7.4 min (50 steps) | ~6-12 hours (3 epochs over ~30K examples) |
| IPO | ~15 min (50 steps + ref precompute) | ~3-6 hours (full preference dataset) |
| GRPO | Skipped (TUI-only) | Skipped (TUI-only) |
| Export | ~20 min | ~25 min |
| **Total** | **~1 hour** | **~12-24 hours** |

### Checkpoint Paths Summary

| Phase | Path | Contents |
|-------|------|----------|
| Tool Calling SFT | `checkpoints/tool_calling_sft/final/` | LoRA adapter (rank 64) |
| Merged Model | `checkpoints/gpt-oss-20b-coding-tui-merged/hf/` | Full-weight merged model (~39 GB) |
| Agent SFT | `checkpoints/agent_sft/final/` | LoRA adapter (rank 128) |
| IPO | `checkpoints/agent_sft_ipo/final/` | LoRA adapter |
| GRPO | `checkpoints/agent_sft_grpo/final/` | LoRA adapter (skipped in TUI) |
| Export (HF) | `checkpoints/gpt-oss-20b-coding-tui-export/hf/` | Final merged SafeTensors |
| Export (GGUF) | `checkpoints/gpt-oss-20b-coding-tui-export/gguf_q4/` | Quantized GGUF for llama.cpp |

---

## Appendix: Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `HF_TOKEN` | (none) | HuggingFace access token for gated datasets |
| `RUNPOD_SECRET_HF_TOKEN` | (none) | RunPod-specific HF token injection |
| `WANDB_MODE` | `offline` | Weights & Biases logging mode |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduces CUDA memory fragmentation |
| `HF_HOME` | `/workspace/.cache/huggingface` | HuggingFace cache directory |
| `TORCH_HOME` | `/workspace/.cache/torch` | PyTorch cache directory |
| `XDG_CACHE_HOME` | `/workspace/.cache` | General cache directory |
| `TMPDIR` | `/workspace/tmp` | Temporary file directory |
| `TOKENIZERS_PARALLELISM` | `false` | Prevents HF tokenizer fork warnings |
