---
name: init-data-pipeline
description: "Phase 0: Initialize SOTA data pipeline for a coding agent language. Creates capability taxonomy, governance config, eval benchmarks, and provenance schema."
disable-model-invocation: true
argument-hint: "[language: rust|python|typescript|go]"
---

# /init-data-pipeline

Initialize the SOTA data pipeline (Phase 0) for a target language. This creates the config files and directory structure that all downstream pipeline skills depend on.

## Instructions

### Step 1: Parse Language Argument

Parse `$ARGUMENTS` for the target language. Valid values: `rust`, `python`, `typescript`, `go`.

- If `$ARGUMENTS` is empty or not one of the valid values, ask the user which language to initialize using AskUserQuestion with the 4 options.
- Normalize to lowercase and trim whitespace.
- Store as `LANGUAGE`.

### Step 2: Read Existing Configs for Reference

Read these files to extract metric keys and thresholds — the generated configs MUST reference the same keys:

1. `configs/{LANGUAGE}_eval.yaml` — metric names for this language
2. `configs/promotion_gates.yaml` — gate threshold structure and patterns
3. `.claude/skills/init-data-pipeline/language_profiles.md` — canonical metric mappings per language

### Step 3: Create Shared Configs (idempotent)

Check if `configs/sota/governance.yaml` exists. If it does NOT exist, create both shared config files. If it already exists, skip this step and print "Shared configs already exist, skipping."

First, create the directory:
```
configs/sota/
```

#### File: `configs/sota/governance.yaml`

Write this file:

```yaml
# SOTA Data Pipeline — Governance Configuration
# Shared across all languages. Changes require MAJOR version bump.
version: "1.0.0"
changelog_path: "data/sota/CHANGELOG.md"

license_policy:
  permissive:
    - MIT
    - Apache-2.0
    - BSD-2-Clause
    - BSD-3-Clause
    - ISC
    - Unlicense
    - CC0-1.0
  attribution_required:
    - CC-BY-4.0
    - CC-BY-SA-4.0
  blocked:
    - GPL-2.0
    - GPL-3.0
    - AGPL-3.0
    - SSPL-1.0
    - BUSL-1.1
    - CC-BY-NC-4.0

pii_scrubbing:
  enabled: true
  code_aware: true  # avoid scrubbing variable names that look like PII patterns
  patterns:
    email: '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone: '\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    ssn: '\d{3}-\d{2}-\d{4}'
    credit_card: '\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
    api_key: '(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*["\']?[a-zA-Z0-9_\-]{16,}'
    aws_key: 'AKIA[0-9A-Z]{16}'
    private_key: '-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----'

oracle:
  default_model: "claude-sonnet-4-20250514"
  task_overrides:
    trajectory_generation: "claude-sonnet-4-20250514"
    preference_judgment: "claude-sonnet-4-20250514"
    capability_extraction: "claude-sonnet-4-20250514"
    taxonomy_review: "claude-sonnet-4-20250514"
  temperature:
    generation: 0.7
    judgment: 0.1
    extraction: 0.0
  max_retries: 3
  timeout_seconds: 120

versioning:
  scheme: "semver"
  changelog_path: "data/sota/CHANGELOG.md"
  major_bump_triggers:
    - "eval benchmark change"
    - "governance policy change"
    - "provenance schema change"
  minor_bump_triggers:
    - "new capability added to taxonomy"
    - "threshold adjustment"
  patch_bump_triggers:
    - "data regeneration with same config"
    - "bug fix in pipeline"

contamination_checking:
  enabled: true
  eval_benchmarks:
    - "HumanEval"
    - "MBPP"
    - "SWE-bench"
    - "SWE-bench-lite"
    - "LiveCodeBench"
    - "BigCodeBench"
    - "CrossCodeEval"
    - "DS-1000"
  dedup_method: "minhash"
  minhash_threshold: 0.8
  cross_split_check: true
  ngram_size: 5
```

#### File: `configs/sota/provenance_schema.yaml`

Write this file:

```yaml
# SOTA Data Pipeline — Provenance Schema
# Every generated training example must carry these metadata fields.
# Changes require a MAJOR version bump.
version: "1.0.0"

required_fields:
  # Identity
  example_id:
    type: "string"
    format: "uuid4"
    description: "Unique identifier for this training example"

  pipeline_version:
    type: "string"
    format: "semver"
    description: "Version of the SOTA pipeline that generated this example"

  phase:
    type: "string"
    enum:
      - "phase1_collection"
      - "phase2_extraction"
      - "phase3a_sft"
      - "phase3b_preference"
      - "phase3c_reward"
      - "phase3d_replay"
    description: "Pipeline phase that produced this example"

  taxonomy_capability:
    type: "string"
    description: "Capability from the taxonomy this example trains"

  taxonomy_category:
    type: "string"
    enum: ["behavioral", "skills", "knowledge", "procedural"]
    description: "Top-level taxonomy category"

  # Oracle fields
  oracle_model:
    type: "string"
    description: "Model ID used to generate or judge this example"

  oracle_temperature:
    type: "number"
    description: "Temperature setting used for oracle generation"

  oracle_system_prompt_hash:
    type: "string"
    format: "sha256"
    description: "SHA-256 hash of the system prompt sent to the oracle"

  # Source fields
  source_url:
    type: "string"
    format: "url"
    nullable: true
    description: "URL of the source material (null for synthetic)"

  source_license:
    type: "string"
    description: "SPDX license identifier of the source"

  source_repo:
    type: "string"
    nullable: true
    description: "Repository name (owner/repo) if from GitHub/GitLab"

  source_commit:
    type: "string"
    nullable: true
    description: "Commit SHA of the source snapshot"

  # Quality fields
  pii_scrubbed:
    type: "boolean"
    description: "Whether PII scrubbing was applied"

  contamination_checked:
    type: "boolean"
    description: "Whether eval contamination check was run"

  dedup_hash:
    type: "string"
    format: "minhash"
    description: "MinHash signature for deduplication"

optional_fields:
  harmony_format:
    type: "string"
    enum:
      - "harmony_code"
      - "harmony_completion"
      - "harmony_agent"
      - "harmony_preference"
      - "harmony_task"
      - "harmony_debug"
    description: "Harmony formatter used to produce the final training example"

  language:
    type: "string"
    enum: ["rust", "python", "typescript", "go"]
    description: "Target programming language"

  quality_score:
    type: "number"
    minimum: 0.0
    maximum: 1.0
    description: "Aggregate quality score from QA checks"

  generation_timestamp:
    type: "string"
    format: "iso8601"
    description: "When this example was generated"

batch_log_format:
  description: "Each pipeline run produces a batch log in data/sota/{language}/provenance/"
  filename_pattern: "batch_{phase}_{timestamp}.jsonl"
  fields:
    batch_id:
      type: "string"
      format: "uuid4"
    phase:
      type: "string"
    started_at:
      type: "string"
      format: "iso8601"
    completed_at:
      type: "string"
      format: "iso8601"
    example_count:
      type: "integer"
    oracle_model:
      type: "string"
    oracle_total_tokens:
      type: "integer"
    config_hash:
      type: "string"
      format: "sha256"
      description: "Hash of the full config snapshot used for this batch"
```

### Step 4: Create Per-Language Taxonomy

#### File: `configs/sota/{LANGUAGE}_taxonomy.yaml`

Generate this file using the language-specific metrics from the language_profiles.md reference. The structure must follow this template, with metric keys adapted per language.

**For Rust** (`configs/sota/rust_taxonomy.yaml`):

```yaml
# SOTA Capability Taxonomy — Rust
# Defines what the target model should learn and how to generate training data.
version: "1.0.0"
language: "rust"

categories:
  behavioral:
    description: "How the model behaves as an assistant"
    capabilities:
      helpfulness:
        description: "Provides useful, actionable responses to coding questions"
        harmony_format: "harmony_agent"
        eval_signal: "sota_judged"
        weight: 0.15
        data_generation_strategy: "sota_trajectory"
      safety:
        description: "Refuses harmful requests, avoids generating dangerous code"
        harmony_format: "harmony_agent"
        eval_signal: "safety_refusal_rate"
        weight: 0.10
        data_generation_strategy: "sota_preference"
      uncertainty_calibration:
        description: "Expresses appropriate uncertainty, asks for clarification"
        harmony_format: "harmony_agent"
        eval_signal: "sota_judged"
        weight: 0.05
        data_generation_strategy: "sota_preference"

  skills:
    description: "Core coding capabilities"
    capabilities:
      code_generation:
        description: "Generates correct, idiomatic Rust code from specifications"
        harmony_format: "harmony_task"
        eval_signal: "cargo_check_pass_rate"
        weight: 0.15
        data_generation_strategy: "sota_trajectory"
      debugging:
        description: "Diagnoses and fixes compilation errors and test failures"
        harmony_format: "harmony_debug"
        eval_signal: "cargo_test_pass_rate"
        weight: 0.15
        data_generation_strategy: "mutation_plus_sota"
      planning:
        description: "Breaks complex tasks into steps, reasons about approach"
        harmony_format: "harmony_agent"
        eval_signal: "iterations_to_green_median"
        weight: 0.05
        data_generation_strategy: "sota_trajectory"
      test_writing:
        description: "Writes comprehensive test cases for Rust code"
        harmony_format: "harmony_task"
        eval_signal: "cargo_test_pass_rate"
        weight: 0.05
        data_generation_strategy: "sota_trajectory"

  knowledge:
    description: "Language and ecosystem knowledge"
    capabilities:
      language_semantics:
        description: "Understands ownership, borrowing, lifetimes, type system"
        harmony_format: "harmony_code"
        eval_signal: "cargo_check_pass_rate"
        weight: 0.05
        data_generation_strategy: "web_extraction"
      std_library:
        description: "Knows standard library APIs and correct usage patterns"
        harmony_format: "harmony_completion"
        eval_signal: "hallucinated_api_rate"
        weight: 0.05
        data_generation_strategy: "web_extraction"
      ecosystem_crates:
        description: "Knows popular crates (serde, tokio, clap, etc.) and their APIs"
        harmony_format: "harmony_code"
        eval_signal: "hallucinated_api_rate"
        weight: 0.05
        data_generation_strategy: "web_extraction"
      idiomatic_patterns:
        description: "Uses Rust idioms (Result chains, iterators, pattern matching)"
        harmony_format: "harmony_code"
        eval_signal: "clippy_clean_rate"
        weight: 0.05
        data_generation_strategy: "web_extraction"

  procedural:
    description: "Multi-step agent workflows"
    capabilities:
      multi_step_debugging:
        description: "Iteratively reads errors, hypothesizes, patches, and verifies"
        harmony_format: "harmony_agent"
        eval_signal: "iterations_to_green_median"
        weight: 0.15
        data_generation_strategy: "mutation_plus_sota"
      tool_use:
        description: "Correctly formats and sequences tool calls"
        harmony_format: "harmony_agent"
        eval_signal: "tool_call_format_accuracy"
        weight: 0.10
        data_generation_strategy: "sota_trajectory"
      state_tracking:
        description: "Maintains context across multi-turn debugging sessions"
        harmony_format: "harmony_agent"
        eval_signal: "iterations_to_green_median"
        weight: 0.05
        data_generation_strategy: "sota_trajectory"
```

**For Python** (`configs/sota/python_taxonomy.yaml`):

Use the same structure but adapt:
- `language: "python"`
- Knowledge capabilities: `language_semantics` → "Understands dynamic typing, duck typing, decorators, context managers"; `std_library` → standard library; `ecosystem_packages` → "Knows popular packages (requests, pandas, fastapi, etc.)"; `idiomatic_patterns` → "Uses Python idioms (list comprehensions, generators, context managers)"
- Eval signals: `cargo_check_pass_rate` → `syntax_check_pass_rate`; `cargo_test_pass_rate` → `pytest_pass_rate`; `clippy_clean_rate` → `ruff_clean_rate`; keep shared signals unchanged
- Add `type_checking` knowledge capability with eval_signal `mypy_clean_rate`

**For TypeScript** (`configs/sota/typescript_taxonomy.yaml`):

Use the same structure but adapt:
- `language: "typescript"`
- Knowledge capabilities: `language_semantics` → "Understands TypeScript type system, generics, conditional types, mapped types"; `std_library` → "Knows built-in JS/TS APIs"; `ecosystem_packages` → "Knows popular packages (express, react, zod, prisma, etc.)"; `idiomatic_patterns` → "Uses TypeScript idioms (discriminated unions, type guards, utility types)"
- Eval signals: `cargo_check_pass_rate` → `tsc_pass_rate`; `cargo_test_pass_rate` → `jest_pass_rate`; `clippy_clean_rate` → `eslint_clean_rate`; keep shared signals unchanged

**For Go** (`configs/sota/go_taxonomy.yaml`):

Use the same structure but adapt:
- `language: "go"`
- Knowledge capabilities: `language_semantics` → "Understands goroutines, channels, interfaces, embedding, error handling"; `std_library` → standard library; `ecosystem_packages` → "Knows popular packages (gin, cobra, viper, etc.)"; `idiomatic_patterns` → "Uses Go idioms (error wrapping, table-driven tests, functional options)"
- Eval signals: `cargo_check_pass_rate` → `go_build_pass_rate`; `cargo_test_pass_rate` → `go_test_pass_rate`; `clippy_clean_rate` → `go_vet_clean_rate`; keep shared signals unchanged
- Add `linting` knowledge capability with eval_signal `golangci_lint_clean_rate`

### Step 5: Create Per-Language Eval Benchmarks

#### File: `configs/sota/{LANGUAGE}_eval_benchmarks.yaml`

Generate this file. The metric names MUST match the keys from `configs/{LANGUAGE}_eval.yaml` exactly. Threshold structure follows `configs/promotion_gates.yaml` patterns.

**For Rust** (`configs/sota/rust_eval_benchmarks.yaml`):

```yaml
# SOTA Eval Benchmarks — Rust
# Frozen evaluation suite. Changes require MAJOR version bump.
version: "1.0.0"
language: "rust"
frozen_at: "2026-02-24"

benchmarks:
  compilation_benchmark:
    description: "Can the model generate Rust code that compiles?"
    metric: "cargo_check_pass_rate"
    data_source: "data/sota/rust/eval/compilation.jsonl"
    targets:
      after_core_agent_sft: 0.70
      after_ipo: 0.80
      after_grpo: 0.85
      production: 0.90

  test_pass_benchmark:
    description: "Can the model fix bugs so tests pass?"
    metric: "cargo_test_pass_rate"
    data_source: "data/sota/rust/eval/test_pass.jsonl"
    targets:
      after_core_agent_sft: 0.50
      after_ipo: 0.60
      after_grpo: 0.70
      production: 0.75

  lint_benchmark:
    description: "Does the model produce clippy-clean code?"
    metric: "clippy_clean_rate"
    data_source: "data/sota/rust/eval/lint.jsonl"
    targets:
      after_core_agent_sft: 0.60
      after_ipo: 0.70
      after_grpo: 0.80
      production: 0.85

  efficiency_benchmark:
    description: "How many iterations does the model need to fix a bug?"
    metric: "iterations_to_green_median"
    data_source: "data/sota/rust/eval/efficiency.jsonl"
    targets:
      after_core_agent_sft: 5
      after_ipo: 4
      after_grpo: 3
      production: 3

  patch_size_benchmark:
    description: "Are patches minimal and focused?"
    metric: "diff_size_median"
    data_source: "data/sota/rust/eval/patch_size.jsonl"
    targets:
      after_core_agent_sft: 80
      after_ipo: 60
      after_grpo: 50
      production: 40

  tool_format_benchmark:
    description: "Does the model format tool calls correctly?"
    metric: "tool_call_format_accuracy"
    data_source: "data/sota/rust/eval/tool_format.jsonl"
    targets:
      after_core_agent_sft: 0.95
      after_ipo: 0.97
      after_grpo: 0.99
      production: 0.99

  hallucination_benchmark:
    description: "Does the model avoid hallucinating APIs?"
    metric: "hallucinated_api_rate"
    data_source: "data/sota/rust/eval/hallucination.jsonl"
    targets:
      after_core_agent_sft: 0.10
      after_ipo: 0.08
      after_grpo: 0.05
      production: 0.03

  safety_benchmark:
    description: "Does the model refuse harmful requests?"
    metric: "safety_refusal_rate"
    data_source: "data/sota/rust/eval/safety.jsonl"
    targets:
      after_core_agent_sft: 0.80
      after_ipo: 0.80
      after_grpo: 0.80
      production: 0.85

regression_checks:
  humaneval_python:
    description: "General coding ability regression check"
    metric: "humaneval_python"
    max_regression: 0.05
  mmlu_subset:
    description: "General knowledge regression check"
    metric: "mmlu_subset"
    max_regression: 0.05

capability_benchmark_map:
  helpfulness: ["safety_benchmark"]
  safety: ["safety_benchmark"]
  uncertainty_calibration: ["safety_benchmark"]
  code_generation: ["compilation_benchmark", "lint_benchmark"]
  debugging: ["test_pass_benchmark", "efficiency_benchmark"]
  planning: ["efficiency_benchmark"]
  test_writing: ["test_pass_benchmark"]
  language_semantics: ["compilation_benchmark"]
  std_library: ["hallucination_benchmark"]
  ecosystem_crates: ["hallucination_benchmark"]
  idiomatic_patterns: ["lint_benchmark"]
  multi_step_debugging: ["test_pass_benchmark", "efficiency_benchmark", "patch_size_benchmark"]
  tool_use: ["tool_format_benchmark"]
  state_tracking: ["efficiency_benchmark"]
```

**For Python** — same structure, adapt:
- `language: "python"`, metrics from `configs/python_eval.yaml`
- Replace `compilation_benchmark` metric with `syntax_check_pass_rate`, `test_pass_benchmark` with `pytest_pass_rate`, `lint_benchmark` with `ruff_clean_rate`
- Add `type_check_benchmark` with metric `mypy_clean_rate`
- Data source paths under `data/sota/python/eval/`
- `capability_benchmark_map`: add `type_checking: ["type_check_benchmark"]`

**For TypeScript** — same structure, adapt:
- `language: "typescript"`, metrics from `configs/typescript_eval.yaml`
- Replace `compilation_benchmark` metric with `tsc_pass_rate`, `test_pass_benchmark` with `jest_pass_rate`, `lint_benchmark` with `eslint_clean_rate`
- Data source paths under `data/sota/typescript/eval/`

**For Go** — same structure, adapt:
- `language: "go"`, metrics from `configs/go_eval.yaml`
- Replace `compilation_benchmark` metric with `go_build_pass_rate`, `test_pass_benchmark` with `go_test_pass_rate`, `lint_benchmark` metric with `go_vet_clean_rate`
- Add `golangci_lint_benchmark` with metric `golangci_lint_clean_rate`
- Data source paths under `data/sota/go/eval/`
- `capability_benchmark_map`: add `linting: ["golangci_lint_benchmark"]`

### Step 6: Create Directory Tree

Create the following directory structure (use `mkdir -p` for each):

```
data/sota/{LANGUAGE}/raw/
data/sota/{LANGUAGE}/extracted/
data/sota/{LANGUAGE}/sft/train/
data/sota/{LANGUAGE}/sft/val/
data/sota/{LANGUAGE}/preference/train/
data/sota/{LANGUAGE}/preference/val/
data/sota/{LANGUAGE}/reward/
data/sota/{LANGUAGE}/replay/
data/sota/{LANGUAGE}/eval/
data/sota/{LANGUAGE}/provenance/
data/sota/{LANGUAGE}/qa/
```

Add a `.gitkeep` file in each leaf directory so Git tracks the empty directories.

### Step 7: Initialize Changelog (idempotent)

If `data/sota/CHANGELOG.md` does NOT exist, create it with:

```markdown
# SOTA Data Pipeline Changelog

All notable changes to the SOTA data pipeline configs and data will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-24

### Added
- Initial capability taxonomy
- Governance configuration (license policy, PII scrubbing, oracle settings)
- Provenance schema for training example metadata
- Eval benchmark definitions
```

### Step 8: Print Summary

After all files are created, print a summary like this:

```
## SOTA Data Pipeline Initialized for {LANGUAGE}

### Config files created:
- `configs/sota/governance.yaml` — license policy, PII scrubbing, oracle settings
- `configs/sota/provenance_schema.yaml` — metadata schema for training examples
- `configs/sota/{LANGUAGE}_taxonomy.yaml` — capability taxonomy (4 categories, N capabilities)
- `configs/sota/{LANGUAGE}_eval_benchmarks.yaml` — frozen eval suite with per-stage targets

### Directory tree created:
- `data/sota/{LANGUAGE}/` — 11 subdirectories for pipeline stages

### Next steps:
1. Review the taxonomy weights in `configs/sota/{LANGUAGE}_taxonomy.yaml`
2. Adjust eval benchmark targets in `configs/sota/{LANGUAGE}_eval_benchmarks.yaml` if needed
3. Run `/collect-sources {LANGUAGE}` to begin Phase 1 (source collection)
```
