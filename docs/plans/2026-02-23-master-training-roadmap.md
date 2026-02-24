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

Phase 6: Advanced Agent Capabilities (~31-55 GPU-days)
  |-- 6a. Codebase Navigation          [P0] — repo maps, symbol search
  |-- 6b. Planning Before Coding       [P0] — structured plan-then-execute
  |-- 6c. Error Recovery & Backtracking [P0] — Verifier/Judger/Reflector
  |-- 6d. Web Research & Self-Learning  [P0] — search docs, learn libraries
  |-- 6e. Multi-Agent Coordination      [P1] — delegate_subtask, role switching
  |-- 6f. Testing & Test Generation     [P1] — TDD workflows, coverage gaps
  '-- 6g. Git Operations               [P1] — diffs, commits, PR review

Phase 7: Self-Play & Continuous Improvement (Future)
  '-- Bug injection, rejection sampling, task self-generation, multi-agent self-play
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

## Phase 6: Advanced Agent Capabilities

After language-specific adapters are trained and validated (Phases 1-5), the model needs higher-order skills to function as a truly self-sufficient coding agent. These capabilities are trained as extensions to the existing per-language adapters — not separate adapters.

### Phase 6a: Codebase Navigation (P0)

**Why**: Agents fail at multi-file tasks (21% vs 65% single-file). Navigation is the bottleneck, not code generation. Hybrid-Gym (Feb 2026) shows 25.4% SWE-bench improvement from navigation training alone.

**New tools**: `list_directory`, `find_references`, `get_call_graph`, `search_symbols`

**Training data**:
- 5-10K navigation trajectories via Hybrid-Gym synthetic tasks (0.07c/example)
- Repo exploration tasks: "Find where `validate_token` is called and what error handling exists"
- Teach the thinking channel to maintain a mental model of repo structure

**New formatter**: `harmony_navigation` — repo exploration traces with file tree context

**GRPO reward signals**:
- `files_found_relevant`: +0.2 — did the agent find the right files?
- `search_efficiency`: +0.1 — fewer hops to reach target code
- `unnecessary_reads`: -0.1 — penalty for reading irrelevant files

### Phase 6b: Planning Before Coding (P0)

**Why**: Planning doubles SWE-bench scores. Blueprint2Code validates this on small models. CodePlan (Microsoft) demonstrates repo-level planning.

**Training data**:
- 3-5K plan-then-execute trajectories
- Plans decompose tasks into ordered steps with file dependencies
- Include plans that get revised mid-execution (initial plan was wrong)

**New formatter**: `harmony_plan` — structured plan in thinking channel, then execution

**Thinking channel pattern**:
```
<|thinking|>
## Plan
1. Read src/auth.rs to understand current token validation
2. Read src/main.rs to find all callers of validate_token()
3. Apply patch to make validate_token async
4. Update callers to .await the result
5. Run cargo test to verify
6. Run cargo clippy for style

Dependencies: Step 4 depends on Step 3. Step 5 depends on Step 4.
<|/thinking|>
```

**GRPO reward**: `plan_completeness` (+0.15 if plan covers all modified files)

### Phase 6c: Error Recovery & Backtracking (P0)

**Why**: 30% of trajectories require course correction. BacktrackAgent (EMNLP 2025) proves error detection is trainable via SFT+RL with Verifier/Judger/Reflector pattern.

**Training data**:
- 3-5K recovery trajectories where initial approach fails
- Model detects failure, reasons about why, tries alternative
- Include "echo trap" detection — recognizing when retrying the same approach won't work

**Thinking channel pattern** (Verifier → Judger → Reflector):
```
<|thinking|>
**Verify**: The patch compiled but cargo test still fails with the same error.
**Judge**: My fix addressed the wrong call site — line 42 is a different validate_token.
**Reflect**: I need to find the CORRECT call site. Let me search for the one in the auth middleware, not the test helper.
<|/thinking|>
```

**GRPO reward signals**:
- `recovered_from_failure`: +0.3 — successfully fixed after initial failure
- `repeated_same_approach`: -0.3 — tried identical fix twice
- `backtracked_appropriately`: +0.2 — recognized failure and changed strategy

### Phase 6d: Web Research & Self-Learning (P0)

**Why**: DocPrompting shows 52% pass@1 improvement. Search-R1 shows 41% over RAG baselines. A self-sufficient agent must learn new libraries without human help.

**New tools**: `web_search(query)`, `fetch_url(url)`, `read_docs(library, item)`

**Training data**:
- 5-8K research trajectories: "I don't know this library → search → read docs → write code → test"
- 2-3K preference pairs: good research (targeted search, extract key info) vs. bad (blind guessing)
- 1K GRPO tasks requiring unfamiliar libraries

**Key skills to train**:
1. **Knowledge gap recognition** (R-Tuning) — "I should look this up" vs. "I know this"
2. **Query formulation** — specific queries (`"reqwest bytes_stream async Rust"`) beat vague ones
3. **Documentation extraction** — summarize key API details in thinking channel
4. **Version awareness** — check installed version before searching docs
5. **Conflict resolution** — when StackOverflow contradicts official docs, trust docs for installed version

**Thinking channel as research notebook**:
```
<|thinking|>
## Research Notes: tantivy crate (v0.22)
- Index::create_in_dir(path, schema) → creates index
- SchemaBuilder: add_text_field(), add_u64_field()
- IMPORTANT: writer.commit() required after adding documents
- IMPORTANT: Requires Rust 1.70+
I have enough to implement. The key gotcha is commit() after writes.
<|/thinking|>
```

**GRPO reward signals**:
- `searched_and_needed`: +0.1 — search was necessary and used
- `searched_unnecessarily`: -0.05 — trivially known, didn't need to search
- `didnt_search_and_failed`: -0.3 — hallucinated an API that doesn't exist
- `novel_library_task_pass_rate`: target 50%

**Search-R1's retrieved-token-masking**: During GRPO training, mask out search result tokens from loss computation. Only train on the model's own generated tokens.

### Phase 6e: Multi-Agent Coordination (P1)

**Why**: AgentCoder's 3-role decomposition (planner/coder/tester) achieves 96.3% on HumanEval. SERA validates that 8K trajectories suffice for specialization. Complex multi-file tasks benefit from delegation.

**Implementation**: Delegation as tool call — no new special tokens needed.

**New tool**: `delegate_subtask(role, task, context)`

**Available roles** (all played by the same model with different developer prompts):
- **Planner**: Decomposes complex tasks into ordered subtasks
- **Reviewer**: Checks code for correctness, style, safety
- **Tester**: Writes and runs comprehensive tests
- **Debugger**: Diagnoses and fixes specific errors

**Training data**:
- 5K role-specific trajectories (model operating in each role)
- 5K orchestrator trajectories (model delegating and synthesizing)
- 2-3K preference pairs: good delegation vs. monolithic single-pass
- Include tasks where delegation is unnecessary (simple fixes) — model must learn NOT to delegate trivially

**Harmony encoding** (uses existing tokens):
```
<|thinking|>
This task modifies 3 files across 2 modules. I should implement the core change
myself and delegate review and testing.
<|/thinking|>
<|tool_call|>
{"id": "del_01", "name": "delegate_subtask", "arguments": {"role": "planner", "task": "Decompose: add connection pooling to src/db/", "context": {"files": ["src/db/mod.rs", "src/db/pool.rs"]}}}
<|tool_result|>
<|tool_call_id|>del_01
{"subtasks": [{"id": 1, "description": "Add pool struct", "files": ["src/db/pool.rs"]}, ...]}
```

**GRPO reward signals**:
- `delegation_efficiency`: +0.2 — appropriate delegation improved outcome
- `unnecessary_delegation`: -0.3 — delegated a trivial task
- `conflict_resolved`: +0.3 — resolved disagreement between sub-agents
- `role_adherence`: target 90% — sub-agent stays in assigned role

### Phase 6f: Testing & Test Generation (P1)

**Why**: TDFlow shows 27.8% improvement. Test generation is "the final frontier" for autonomous repair.

**Training data**: 2-3K TDD trajectories (write test → implement → verify → iterate)

**GRPO reward**: `test_coverage_delta` (+0.15 for adding tests that cover new paths)

### Phase 6g: Git Operations (P1)

**Why**: All production workflows require git. AIDev-pop (Feb 2026) provides 33,596 free agentic PR examples.

**New tools**: `git_diff`, `git_log`, `git_blame`, `create_branch`, `commit_changes`

**Training data**: 2-3K git operation trajectories from AIDev-pop dataset

### Phase 6 Timeline

| Sub-phase | Depends On | Training Data | Compute |
|-----------|-----------|---------------|---------|
| 6a. Navigation | Phases 2-3 complete | 5-10K trajectories | 3-5 GPU-days |
| 6b. Planning | 6a | 3-5K trajectories | 2-3 GPU-days |
| 6c. Error Recovery | 6a, 6b | 3-5K trajectories | 3-5 GPU-days |
| 6d. Web Research | 6a | 5-8K trajectories + tool infra | 5-7 GPU-days |
| 6e. Multi-Agent | 6a, 6b, 6c | 10K trajectories | 5-7 GPU-days |
| 6f. Test Generation | 6a | 2-3K trajectories | 2-3 GPU-days |
| 6g. Git Operations | 6a | 2-3K trajectories | 1-2 GPU-days |

**Total Phase 6**: ~31-55 GPU-days, ~35-50K training examples

### Phase 6 Promotion Gates

```yaml
phase_6_gates:
  # Navigation (6a)
  multi_file_task_pass_rate: 0.50       # Up from baseline ~21%
  file_localization_accuracy: 0.70

  # Planning (6b)
  plan_completeness: 0.75
  plan_execution_success: 0.60

  # Error Recovery (6c)
  recovery_rate: 0.50                   # Recovers from 50%+ of failures
  echo_trap_detection: 0.70            # Recognizes repeated failed approach

  # Web Research (6d)
  novel_library_pass_rate: 0.50
  search_efficiency: 0.70              # 70%+ of searches are useful
  hallucinated_api_rate: 0.03          # <3% nonexistent APIs

  # Multi-Agent (6e)
  delegation_precision: 0.75
  delegation_recall: 0.80
  conflict_resolution_rate: 0.60
  unnecessary_delegation_rate: 0.10    # <10%

  # Test Generation (6f)
  generated_test_validity: 0.80        # 80%+ generated tests compile and are meaningful

  # Git (6g)
  valid_diff_generation: 0.85
  commit_message_quality: 0.70         # Scored by LLM judge
```

---

## Phase 7: Self-Play & Continuous Improvement (Future)

After Phase 6 produces a capable agent, use self-play to generate infinite training data:

1. **Bug Injection + Repair** (SWE-RL): Model injects mutations into working code, then must repair them. Creates training data from any repo without human labels.
2. **Rejection Sampling at Scale**: Generate N solutions per task, keep best, retrain on winners. Each iteration improves the model.
3. **Task Self-Generation**: Model reads repos and creates its own coding challenges, solves them, filters by execution correctness.
4. **Multi-Agent Self-Play** (MARS, MAE): Multiple model instances play planner/coder/reviewer/tester roles, debate and improve each other's output, trained via GRPO.

Requires a fully functional agent from Phases 1-6 before starting.

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
| Model always delegates (lazy agent) | Include simple tasks where delegation is penalized; 50/50 mix of single-agent and multi-agent data |
| Search API costs during training | Cache all web responses; local mirror of docs.rs for Rust; estimated $50-100 for 10K trajectories |
| Hallucinated tool calls to nonexistent APIs | Existing hallucinated_api_rate metric + GRPO penalty for didnt_search_and_failed |
| Context window overflow from search results | Teach distillation in thinking channel; default max_chars=8000 for fetch_url; Search-R1 retrieved-token masking |
| Delegation loops (A → B → A) | Explicit no-re-delegation constraint in sub-agent developer prompts; max depth=2 |
| Echo trap (retrying same failed approach) | BacktrackAgent-style Verifier/Judger/Reflector pattern in thinking channel; GRPO penalty for repeated_same_approach |

---

## Success Criteria

The pipeline is complete when:

**Language Adapters (Phases 1-5)**:
1. All 4 language adapters pass their evaluation targets (Phase 4 metrics table)
2. No adapter regresses TUI baseline by more than 5% on general benchmarks
3. Adapter hot-swapping works in <1s with PEFT
4. The adapter registry is populated with eval scores
5. A new language can be added by following the "Adding New Languages" checklist without modifying any existing code

**Harmony Format Quality**:
6. Thinking traces follow the Observe → Hypothesize → Plan → Execute structure in > 80% of multi-file tasks
7. Model uses appropriate reasoning_effort scaling (short thinking for easy tasks, long for hard)
8. Tool call efficiency improves over training (fewer calls to reach correct solution)

**Advanced Capabilities (Phase 6)**:
9. Multi-file task pass rate > 50% (up from ~21% baseline)
10. Model plans before coding on complex tasks (plan_completeness > 75%)
11. Model recovers from failures > 50% of the time (no echo traps)
12. Model can learn and use unfamiliar libraries via web research (novel_library_pass_rate > 50%)
13. Model delegates appropriately on complex tasks (delegation_precision > 75%, unnecessary_delegation < 10%)
14. Model generates valid tests for untested code (test_validity > 80%)
