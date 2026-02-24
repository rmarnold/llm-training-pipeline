# Master Training Roadmap: GPT-OSS 20B Multi-Language Coding Agent

**Mission: Sequential fine-tuning with a curriculum** — each training stage builds on the previous session's learned representations, progressively shaping the model from general language understanding to specialized coding agent behavior.

**Date**: 2026-02-23
**Base Model**: GPT-OSS 20B MoE (~3.6B active params, 32 experts/layer, top-4 routing)
**Starting Checkpoint**: `checkpoints/coding_tui/final_merged` (TUI session complete)
**Strategy**: Mixture of Adapters (MoA) — separate LoRA adapters per language, hot-swappable at inference

---

## Architecture: Mixture of Adapters (MoA)

```
GPT-OSS 20B Base (frozen weights)
    |
    v
TUI Checkpoint (checkpoints/coding_tui/final_merged)
    |  Already trained: tool-call JSON format, multi-turn debugging,
    |  apply_patch, run_tests, IPO preferences, GRPO rewards
    |
    |-- LoRA Adapter: Rust       --> checkpoints/adapters/rust_coding
    |-- LoRA Adapter: Python     --> checkpoints/adapters/python_coding
    |-- LoRA Adapter: TypeScript --> checkpoints/adapters/typescript_coding
    '-- LoRA Adapter: Go         --> checkpoints/adapters/go_coding
```

### Why MoA

- **No catastrophic forgetting** — each language adapter trains independently from the same TUI base. A bad Go training run cannot corrupt the Rust adapter.
- **Hot-swappable** — PEFT's `set_adapter()` switches adapters in <1s. Base model stays in memory, only adapter weights change.
- **Incremental rollout** — ship Rust adapter while Python is still training. Add new languages without retraining existing ones.
- **Isolated evaluation** — each adapter is evaluated against its own language-specific benchmarks independently.

### Adapter Specifications

All adapters use QLoRA via Unsloth with identical hyperparameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank | 128 | High capacity for behavioral SFT (agent traces are complex) |
| LoRA alpha | 256 | 2x rank for stable training |
| Target modules | q/k/v/o_proj + gate_up_proj + down_proj | MoE expert targeting (Unsloth auto-detects) |
| Quantization | 4-bit (QLoRA) | Fits 20B MoE on single A100/H100 80GB |
| Max seq length | 32768 | Multi-turn agent sessions need long context |

---

## Phased Roadmap

```
Phase 1: Data Generation (parallelizable per language)
  |-- 1a. Rust mutations + trajectories        [Wave 1]
  |-- 1b. Python mutations + trajectories       [Wave 1]
  |-- 1c. TypeScript mutations + trajectories   [Wave 2]
  '-- 1d. Go mutations + trajectories           [Wave 2]

Phase 2: Adapter Training — Wave 1 (Rust, Python)
  |-- 2a. Rust:   CoreAgent(14) --> IPO(17) --> GRPO(18) --> Eval
  '-- 2b. Python: CoreAgent(14) --> IPO(17) --> GRPO(18) --> Eval

Phase 3: Adapter Training — Wave 2 (TypeScript, Go)
  |-- 3a. TypeScript: CoreAgent(14) --> IPO(17) --> GRPO(18) --> Eval
  '-- 3b. Go:         CoreAgent(14) --> IPO(17) --> GRPO(18) --> Eval

Phase 4: Cross-Language Validation
  '-- Eval all 4 adapters, regression checks, adapter serving test

Phase 5: Production Packaging
  '-- Adapter registry, inference router, deployment config
```

---

## Phase 1: Data Generation

Each language needs 4 datasets before training can start. Data generation is parallelizable across languages — all languages can generate simultaneously if toolchains are available.

### Dataset Requirements

| Dataset | Source | Script | Target Size | Format |
|---------|--------|--------|-------------|--------|
| Mutation debug pairs | Real repos via mutation tools | `16_generate_mutations.py --language X` | 5-10K pairs | harmony_debug |
| Agent trajectories | Claude API + mutations | `15_generate_trajectories.py --language X` | 3-5K trajectories | harmony_agent |
| IPO preference pairs | Ranked N completions from Core Agent | Generated during Phase 2 | 5-10K pairs | harmony_preference |
| GRPO tasks | Extracted from mutation repos | Subset of debug pairs reformatted | 1K tasks | harmony_task |

### Wave 1: Rust + Python

Rust and Python run first because their backends are the most mature.

**Rust data generation**:
```bash
# Generate mutations from curated repos (data_sources_rust.yaml)
python scripts/16_generate_mutations.py \
  --config configs/data_sources_rust.yaml \
  --output data/rust/mutations \
  --max_mutations 200

# Generate agent trajectories from mutations
python scripts/15_generate_trajectories.py \
  --language rust \
  --mutations data/rust/mutations \
  --output data/rust/core_agent/train
```

Curated Rust repos (from `data_sources_rust.yaml`):
- Data structures: bstr, linked-hash-map
- Error handling: thiserror, anyhow, eyre
- Utilities: once_cell, bitflags, log, walkdir, byteorder
- Parsing: nom
- Serialization: serde
- Async: tokio, bytes
- Web: http, axum-core, hyper
- Concurrency: crossbeam-channel, rayon

**Python data generation**:
```bash
python scripts/16_generate_mutations.py \
  --language python \
  --config configs/data_sources_python.yaml \
  --output data/python/mutations

python scripts/15_generate_trajectories.py \
  --language python \
  --mutations data/python/mutations \
  --output data/python/core_agent/train
```

### Wave 2: TypeScript + Go

Starts after Wave 1 adapter training validates the pipeline end-to-end.

**Required toolchains**:
- TypeScript: Node.js, npx, StrykerJS (`npx stryker`), tsc, jest, eslint
- Go: go, go-mutesting, golangci-lint (optional, degrades gracefully)

**TypeScript data generation**:
```bash
python scripts/16_generate_mutations.py \
  --language typescript \
  --config configs/data_sources_typescript.yaml \
  --output data/typescript/mutations

python scripts/15_generate_trajectories.py \
  --language typescript \
  --mutations data/typescript/mutations \
  --output data/typescript/core_agent/train
```

**Go data generation**:
```bash
python scripts/16_generate_mutations.py \
  --language go \
  --config configs/data_sources_go.yaml \
  --output data/go/mutations

python scripts/15_generate_trajectories.py \
  --language go \
  --mutations data/go/mutations \
  --output data/go/core_agent/train
```

### Data Quality Gates

Before proceeding to adapter training, each language must pass:

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| Caught mutations | >= 3,000 | Minimum for Core Agent SFT diversity |
| Trajectories with 2+ tool calls | >= 80% | Ensures multi-turn agent behavior |
| No duplicate (buggy, fixed) pairs | 0 duplicates | Prevents memorization |
| Trajectory avg turns | >= 3 | Agent should read, patch, and verify |
| Valid Harmony format | 100% | All data must parse as valid Harmony |

---

## Phase 2: Adapter Training — Wave 1 (Rust, Python)

Each language adapter trains in 3 sequential stages from the TUI checkpoint. All stages use QLoRA (rank 128) via Unsloth. The stages must run sequentially within a language (each loads the previous stage's checkpoint), but Rust and Python can train in parallel on separate GPUs.

### Stage 1: Core Agent SFT

Teaches the model to apply TUI-learned tool skills to language-specific debugging.

```bash
python scripts/14_train_core_agent.py \
  --base_model checkpoints/coding_tui/final_merged \
  --config configs/core_agent_{lang}.yaml \
  --train_data_path data/{lang}/core_agent/train
```

| Parameter | Value |
|-----------|-------|
| Epochs | 2 |
| Learning rate | 3e-5 |
| Batch size | 1 (grad accum 4) |
| Packing | enabled (Unsloth padding-free) |
| Optimizer | adamw_8bit |
| Output | `checkpoints/adapters/{lang}_core_agent/` |

**What the model learns**: "When pytest fails with TypeError, read the file, find the type mismatch, patch it, re-run tests." The model already knows the tool-call mechanics from TUI — this stage teaches it language-specific debugging patterns.

**Promotion gate**:

| Metric | Threshold |
|--------|-----------|
| Training loss | < 0.5 |
| Tool call format accuracy | > 95% |
| Eval loss trend | Decreasing over last 20% of training |

### Stage 2: IPO Preference Optimization

Teaches the model to prefer concise, correct, idiomatic solutions over verbose or fragile ones.

```bash
python scripts/17_ipo_preference.py \
  --config configs/ipo.yaml \
  --checkpoint checkpoints/adapters/{lang}_core_agent \
  --train_data data/{lang}/ipo/train
```

| Parameter | Value |
|-----------|-------|
| Epochs | 1 (single epoch — IPO is prone to collapse) |
| Learning rate | 5e-7 (extremely conservative) |
| Beta (KL penalty) | 0.1 |
| Batch size | 1 (grad accum 16) |
| Output | `checkpoints/adapters/{lang}_ipo/` |

**Preference ranking criteria** (language-specific):

| Criterion | Rust | Python | TypeScript | Go |
|-----------|------|--------|------------|-----|
| Hard gate 1 | patch applies | patch applies | patch applies | patch applies |
| Hard gate 2 | cargo check | syntax valid | tsc passes | go build |
| Hard gate 3 | cargo test | pytest passes | jest passes | go test |
| Hard gate 4 | clippy clean | ruff clean | eslint clean | go vet clean |
| Soft: smaller diff | yes | yes | yes | yes |
| Soft: fewer files | yes | yes | yes | yes |
| Soft: idiomatic | Result<> over unwrap | type hints | strict types | error wrapping |

**Promotion gate**:

| Metric | Threshold |
|--------|-----------|
| KL divergence | < 0.3 (warn), < 0.5 (abort) |
| Chosen > rejected reward | > 80% of pairs |
| Training loss | Decreasing, no collapse |

### Stage 3: GRPO Reinforcement Learning

Execution-based rewards from actually running the code. The evaluator dispatch routes to the correct language toolchain.

```bash
python scripts/18_grpo_rl.py \
  --config configs/grpo_{lang}.yaml \
  --language {lang} \
  --checkpoint checkpoints/adapters/{lang}_ipo
```

| Parameter | Value |
|-----------|-------|
| Max steps | 5,000 |
| Learning rate | 1e-6 (very conservative for RL) |
| Generations per prompt | 4 |
| Temperature | 0.7 |
| Batch size | 1 (grad accum 8) |
| Output | `checkpoints/adapters/{lang}_grpo/` |

**Reward function** (dispatched via `evaluator_dispatch.py`):

| Outcome | Rust Reward | Python Reward | TypeScript Reward | Go Reward |
|---------|------------|---------------|-------------------|-----------|
| Tests pass + lint clean | +1.0 | +1.0 | +1.0 | +1.0 |
| Tests pass, lint warnings | +0.7 | +0.7 | +0.7 | +0.7 |
| Compiles, tests fail | +0.1 | +0.1 | +0.1 | +0.1 |
| Compilation/syntax failure | -0.3 | -0.3 | -0.3 | -0.3 |
| Invalid tool call format | -1.0 | -1.0 | -1.0 | -1.0 |

**Long-context curriculum** (same for all languages):

| Steps | Seq Length | Task Complexity |
|-------|-----------|-----------------|
| 0-1000 | 4096 | Single-file fixes, simple bugs |
| 1000-2000 | 8192 | Multi-file with 2-3 files |
| 2000-3500 | 16384 | Full module navigation |
| 3500-5000 | 32768 | Large repo exploration |

**Promotion gate**:

| Metric | Threshold |
|--------|-----------|
| Mean reward (last 500 steps) | > 0.3 |
| Reward trend | No collapse (std > 0.01) |
| Max reward achieved | > 0.7 at least once |

### Final Adapter Merge

After GRPO, merge the adapter for cleaner serving:

```bash
python scripts/19_merge_adapter.py \
  --base_model checkpoints/coding_tui/final_merged \
  --adapter checkpoints/adapters/{lang}_grpo/final \
  --output checkpoints/adapters/{lang}_coding
```

This produces the final per-language adapter at `checkpoints/adapters/{lang}_coding/`.

---

## Phase 3: Adapter Training — Wave 2 (TypeScript, Go)

Identical process to Phase 2, using TypeScript and Go configs. Starts after:
1. Wave 1 (Rust + Python) has validated the full pipeline end-to-end
2. TypeScript and Go mutation data has been generated (Phase 1c/1d)

Any pipeline bugs discovered during Wave 1 are fixed before Wave 2 begins.

---

## Phase 4: Cross-Language Validation

After all 4 adapters are trained, run comprehensive evaluation.

### Per-Language Evaluation

```bash
python scripts/eval_coding_agent.py --config configs/rust_eval.yaml
python scripts/eval_coding_agent.py --config configs/python_eval.yaml
python scripts/eval_coding_agent.py --config configs/typescript_eval.yaml
python scripts/eval_coding_agent.py --config configs/go_eval.yaml
```

### Target Metrics

| Metric | Rust | Python | TypeScript | Go |
|--------|------|--------|------------|-----|
| Compilation/syntax pass rate | 85% | 90% | 85% | 85% |
| Test pass rate | 70% | 70% | 70% | 70% |
| Lint clean rate | 80% | 70% | 80% | 75% |
| Tool call format accuracy | 99% | 99% | 99% | 99% |
| Hallucinated API rate | < 5% | < 5% | < 5% | < 5% |
| Median iterations to green | <= 3 | <= 3 | <= 3 | <= 3 |
| Median diff size (lines) | <= 50 | <= 50 | <= 50 | <= 50 |

### Regression Checks

Every adapter must not regress from the TUI baseline:

| Benchmark | Max Regression |
|-----------|---------------|
| HumanEval Python pass@1 | 5% |
| MMLU subset accuracy | 5% |
| Tool call format accuracy | 0% (must match or exceed TUI) |

### Adapter Comparison

Compare all 4 adapters on shared metrics to identify outliers:
- If any adapter is > 15% below the best adapter on test pass rate, investigate data quality
- If any adapter has tool call accuracy < 95%, the TUI base may be getting overwritten — reduce learning rate

---

## Phase 5: Production Packaging

### Adapter Registry

```yaml
# configs/adapter_registry.yaml
base_model: "checkpoints/coding_tui/final_merged"
default_adapter: "rust"

adapters:
  rust:
    path: "checkpoints/adapters/rust_coding"
    toolchain: ["cargo", "cargo-mutants"]
    eval_metrics:
      cargo_test_pass_rate: 0.00   # filled after eval
      clippy_clean_rate: 0.00
    status: "trained"  # trained | evaluating | production | retired

  python:
    path: "checkpoints/adapters/python_coding"
    toolchain: ["python", "pytest", "mypy", "ruff"]
    eval_metrics:
      pytest_pass_rate: 0.00
      ruff_clean_rate: 0.00
    status: "trained"

  typescript:
    path: "checkpoints/adapters/typescript_coding"
    toolchain: ["node", "tsc", "jest", "eslint"]
    eval_metrics:
      jest_pass_rate: 0.00
      eslint_clean_rate: 0.00
    status: "trained"

  go:
    path: "checkpoints/adapters/go_coding"
    toolchain: ["go", "golangci-lint"]
    eval_metrics:
      go_test_pass_rate: 0.00
      golangci_lint_clean_rate: 0.00
    status: "trained"
```

### Inference Router

At inference time, detect the language and load the appropriate adapter:

1. Explicit: user passes `--language rust`
2. Auto-detect: scan prompt for language markers (file extensions, imports, syntax)
3. Fallback: use `default_adapter` from registry

PEFT adapter swapping:
```python
model.load_adapter("checkpoints/adapters/rust_coding", adapter_name="rust")
model.load_adapter("checkpoints/adapters/python_coding", adapter_name="python")
model.set_adapter("rust")   # switch in <1s
model.set_adapter("python") # switch in <1s
```

### Adding New Languages

To add a new language (e.g., Java, C++):

1. Create mutation runner: `pipeline_lib/{tool}_runner.py`
2. Create evaluator: `pipeline_lib/{lang}_evaluators.py` with `@register_evaluator("{lang}")`
3. Create configs: `data_sources_{lang}.yaml`, `core_agent_{lang}.yaml`, `grpo_{lang}.yaml`, `{lang}_eval.yaml`
4. Generate data: `16_generate_mutations.py --language {lang}`
5. Train adapter: CoreAgent → IPO → GRPO (3 stages)
6. Evaluate: `eval_coding_agent.py --config {lang}_eval.yaml`
7. Register: add entry to `adapter_registry.yaml`

No changes to existing adapters or base model required.

---

## Timeline Summary

| Phase | Work | Depends On | Parallel? | Estimated Steps |
|-------|------|-----------|-----------|-----------------|
| 1a | Rust data gen | cargo-mutants, curated repos | Yes (with 1b) | ~200 mutations x 20 repos |
| 1b | Python data gen | mutmut, Python repos | Yes (with 1a) | ~200 mutations x 15 repos |
| 2a | Rust adapter training | 1a complete | Sequential (3 stages) | SFT + IPO + GRPO |
| 2b | Python adapter training | 1b complete | Can overlap 2a on 2nd GPU | SFT + IPO + GRPO |
| 1c | TypeScript data gen | Node.js toolchain | Yes (with 1d) | After Wave 1 validates |
| 1d | Go data gen | Go toolchain | Yes (with 1c) | After Wave 1 validates |
| 3a | TypeScript adapter | 1c complete | Can overlap 3b | SFT + IPO + GRPO |
| 3b | Go adapter | 1d complete | Can overlap 3a | SFT + IPO + GRPO |
| 4 | Cross-language eval | All adapters trained | Single pass | 4 eval runs |
| 5 | Production packaging | Phase 4 gates pass | Final step | Registry + router |

---

## Checkpoint Directory Structure

```
checkpoints/
  coding_tui/
    final_merged/          <-- TUI base (already exists)
  adapters/
    rust_core_agent/       <-- Phase 2a, Stage 1 output
    rust_ipo/              <-- Phase 2a, Stage 2 output
    rust_grpo/             <-- Phase 2a, Stage 3 output
    rust_coding/           <-- Phase 2a, final merged adapter
    python_core_agent/
    python_ipo/
    python_grpo/
    python_coding/         <-- Phase 2b, final merged adapter
    typescript_core_agent/
    typescript_ipo/
    typescript_grpo/
    typescript_coding/     <-- Phase 3a, final merged adapter
    go_core_agent/
    go_ipo/
    go_grpo/
    go_coding/             <-- Phase 3b, final merged adapter
```

---

## Harmony Format Strategy

All GPT-OSS training data uses the Harmony format (`dataset_formatters/harmony.py`). Harmony provides structured special tokens that enable multi-channel training — the model doesn't just learn to generate code, it learns to *think*, *plan*, *use tools*, and *respond* through dedicated channels.

### Harmony Special Tokens

| Token | Purpose | Training Signal |
|-------|---------|----------------|
| `<\|developer\|>` | System-level instructions | Teaches constraint following |
| `<\|thinking\|>...<\|/thinking\|>` | Internal reasoning | Teaches structured problem-solving |
| `<\|tool_call\|>` | Structured function calls | Teaches tool selection and argument formatting |
| `<\|tool_result\|>` | Tool output ingestion | Teaches parsing execution results |
| `<\|tool_call_id\|>` | Call-result correlation | Teaches multi-tool coordination |
| `<\|assistant\|>` | Final response | Teaches concise user-facing output |

### Thinking Channel Optimization

The `<|thinking|>` block is the model's scratch space. Training data should use structured 4-phase reasoning:

```
<|thinking|>
**Observe**: The test fails with "TypeError: expected str, got int" on line 42.
**Hypothesize**: The `format_output()` function receives an int where it expects str.
**Plan**: Read the caller to find where the int originates, then add type conversion.
**Execute**: apply_patch to add str() conversion at the call site.
<|/thinking|>
```

Map `reasoning_effort` to task complexity in the curriculum:

| Curriculum Stage | Seq Length | reasoning_effort | Thinking Pattern |
|-----------------|-----------|-----------------|-----------------|
| Single-file fixes | 4096 | low | Observe → Execute (skip hypothesis) |
| Multi-file (2-3) | 8192 | medium | Full 4-phase |
| Module navigation | 16384 | high | Full 4-phase + alternative consideration |
| Large repo | 32768 | high | Full 4-phase + multi-step planning |

**Elastic reasoning** — train the model to produce shorter thinking for easy problems and longer thinking for hard problems, rather than fixed-length reasoning. Include both in training data.

### Tool Call Curriculum

Progressive tool complexity maps to GRPO curriculum stages:

| Stage | Steps | Tool Pattern | Example |
|-------|-------|-------------|---------|
| 1. Single tool | 0-1000 | One tool call per turn | `read_file("src/main.rs")` |
| 2. Chains | 1000-2000 | Sequential tool calls | `read_file → apply_patch → run_tests` |
| 3. Conditional | 2000-3500 | Branch on tool result | If test fails → read error → re-patch |
| 4. Planning | 3500-5000 | Think before multi-tool | Plan approach in `<\|thinking\|>`, then execute |

### New Harmony Formatters (Future Work)

Expand beyond the current 6 formatters to cover more developer workflows:

| Formatter | Priority | Purpose | Channels Used |
|-----------|----------|---------|--------------|
| `harmony_diff` | P0 | Unified diff generation/application | thinking + tool_call + assistant |
| `harmony_plan` | P1 | Multi-step plan-then-execute | thinking (plan) + tool_call (execute) |
| `harmony_review` | P1 | Code review with analysis | thinking (analysis) + assistant (feedback) |
| `harmony_refactor` | P2 | Before/after refactoring pairs | thinking (rationale) + tool_call + assistant |
| `harmony_explain` | P2 | Code explanation traces | thinking + assistant |
| `harmony_multifile` | P2 | Cross-file navigation patterns | tool_call chains across files |

`harmony_diff` is highest priority — the model should learn to generate and apply minimal diffs rather than rewriting entire files. This directly improves the "diff minimality" metric.

### Developer Prompt Diversity

Rotate `<|developer|>` prompts to prevent overfitting to a single persona:

**Per-stage rotation**:
- Core Agent SFT: "You are a {language} coding agent. Use tools to read, modify, and test code."
- IPO: "You are a senior {language} engineer. Write minimal, correct patches."
- GRPO: Randomize from a pool per task:
  - "Fix the bug without changing the public API."
  - "Debug this production issue. Minimize your changes."
  - "Review this code and fix any issues you find."
  - "Implement the function according to the test specification."

### Composite Reward Shaping (GRPO Enhancement)

Extend beyond binary pass/fail rewards with multi-signal composition:

| Signal | Weight | Description |
|--------|--------|-------------|
| Execution correctness | 0.5 | Tests pass + lint clean (existing) |
| Diff minimality | 0.2 | Bonus for smaller, targeted patches |
| Tool efficiency | 0.15 | Fewer tool calls to reach correct solution |
| Thinking quality | 0.15 | Structured thinking traces (gated: only counted when code is correct) |

**P-GRPO gating** — only reward thinking quality and tool efficiency when the code is correct. Rewarding "good reasoning + broken code" teaches the model to rationalize failures.

### Trajectory Quality Requirements

| Metric | Requirement | Rationale |
|--------|------------|-----------|
| Turn count | 3-7 per trajectory | < 3 = memorization, > 7 = noise |
| Thinking blocks | >= 1 per trajectory | Must demonstrate reasoning |
| Tool calls | >= 2 per trajectory | Must demonstrate read-patch-test cycle |
| Data mix | 10-20% gold, 40-50% self-gen, 30-40% template | Balances quality and diversity |

**Rejection sampling**: Generate N trajectories per task, keep only those that reach the correct solution. Discard trajectories where the model gets lucky without demonstrating reasoning.

### Self-Play Expansion (Phase 6 — Future)

After initial adapter training succeeds, use self-play to generate new training data:

1. **Bug injection**: Model injects mutations into working code, then must repair them. Creates infinite training data from any repo.
2. **Rejection sampling at scale**: Generate many solutions per task, keep best N, retrain on winners. Each iteration improves.
3. **Task self-generation**: Model reads a repo and creates its own coding challenges, then solves them. Filters by execution correctness.

Self-play requires a working adapter (Phase 2-3 complete) before it can begin.

---

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| Catastrophic forgetting of TUI skills | MoA architecture — TUI base is frozen, adapters are independent |
| IPO preference collapse | Single epoch, ultra-low LR (5e-7), KL monitoring with abort threshold |
| GRPO reward hacking | Long-context curriculum prevents shortcut solutions; reward caps at 1.0 |
| Insufficient mutation data | Quality gate requires 3K+ caught mutations before training starts |
| Language toolchain unavailable | go_evaluators degrades gracefully without golangci-lint; all evaluators have timeout guards |
| Bad adapter corrupts production | Adapter registry tracks status; only "production" adapters served |
| Cross-language interference | MoA by design — adapters never share weights |

---

## Success Criteria

The pipeline is complete when:

1. All 4 language adapters pass their evaluation targets (Phase 4 metrics table)
2. No adapter regresses TUI baseline by more than 5% on general benchmarks
3. Adapter hot-swapping works in <1s with PEFT
4. The adapter registry is populated with eval scores
5. A new language can be added by following the "Adding New Languages" checklist without modifying any existing code
6. Thinking traces follow the Observe → Hypothesize → Plan → Execute structure in > 80% of multi-file tasks
7. Model uses appropriate reasoning_effort scaling (short thinking for easy tasks, long for hard)
8. Tool call efficiency improves over training (fewer calls to reach correct solution)
