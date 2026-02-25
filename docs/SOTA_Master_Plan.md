# Master Plan: Leveraging SOTA Models to Build Curated Datasets & Specialized Training

## Core Premise

Use frontier SOTA models (Claude, GPT-4, etc.) as oracles to generate, curate, and validate high-quality training data for our GPT-OSS 20B pipeline. The SOTA model is a tool — not the thing we're training.

---

## Phase 0 --- Program Setup & Guardrails

### Objectives

-   Define capability taxonomy for the target model (GPT-OSS 20B):
    -   Behavioral (helpfulness, safety, tone, calibration)
    -   Skills (reasoning, coding, math, planning)
    -   Knowledge (facts, explanations, grounding)
    -   Procedural (multi-step workflows, state tracking)
-   Establish governance:
    -   Licensing filters for web-sourced material
    -   PII removal
    -   Provenance tracking (which SOTA model generated what, with which prompt)
    -   Dataset versioning
    -   Reproducibility logs (model version, temperature, system prompt hashes)
-   Design evaluation benchmarks before data collection — these measure the *target model*, not the oracle.

---

## Phase 1 --- Web Source Collection

### 1A. Query Planning

Use SOTA model to generate targeted search queries per category:
-   Knowledge: definitions, references, specs
-   Skills: tutorials, worked examples, code repos
-   Behavioral: policy docs, dialogue examples
-   Procedural: SOPs, runbooks, agent traces

SOTA model helps ensure domain, difficulty, and style diversity by reviewing query coverage gaps.

### 1B. Source Filtering

-   License allowlist enforcement
-   PII scrubbing
-   Deduplication (URL + semantic clustering)
-   Metadata capture (URL, timestamp, license, topic)

---

## Phase 2 --- SOTA-Assisted Extraction & Normalization

### Normalize Content

-   HTML to clean structured text
-   Preserve code, tables, headings

### Semantic Extraction (SOTA model as annotator)

Feed raw web content to SOTA model to extract typed records:
-   Knowledge atoms (claims, definitions)
-   Skill atoms (tasks, steps, examples)
-   Behavioral atoms (scenario, correct vs incorrect responses)
-   Procedural atoms (workflow steps, branches)

The SOTA model classifies, structures, and enriches raw text — turning messy web pages into training-ready records.

---

## Phase 3 --- SOTA-Generated Dataset Construction

### 3A. SFT Dataset

Use SOTA model to generate gold-standard completions from extracted prompts:
-   Ideal responses grounded in source material
-   Worked solutions with step-by-step reasoning
-   Structured explanations
-   Multi-turn agent traces (tool calls, debugging loops)

Format output as Harmony-compatible messages for the GPT-OSS pipeline.

### 3B. Preference (DPO / IPO)

Use SOTA model to produce and rank response pairs:
-   Generate multiple candidate responses at varying quality levels
-   SOTA model scores/ranks them, producing chosen vs rejected pairs
-   Tag failure modes: hallucination, verbosity, logic errors, safety violations

### 3C. Reward Signal Generation (RL)

-   SOTA model scores multiple target-model responses per prompt
-   Rubric: accuracy, safety, helpfulness, calibration
-   Scores feed into GRPO reward functions alongside execution-based signals

### 3D. Replay Buffer

-   Collect target model failures from evaluation runs
-   Feed failures to SOTA model to generate corrected targets
-   Tag failure types for targeted curriculum construction

---

## Phase 4 --- Quality Assurance

Multi-layer QC (SOTA model participates in layers 2-5):

1. Schema validation (automated)
2. Semantic consistency checks (SOTA model cross-references claims)
3. Fact cross-verification (SOTA model + web search)
4. Executable tests for code/skills (automated — cargo test, pytest, etc.)
5. Policy & safety audits (SOTA model flags violations)
6. Human sampling (spot-check SOTA model judgments)

---

## Phase 5 --- Specialized Training Strategy

### Training Sequence (applied to GPT-OSS 20B)

1.  SFT with curriculum + interleaving (using Phase 3A data)
2.  Preference optimization — IPO/DPO (using Phase 3B data)
3.  GRPO RL with hybrid rewards: execution-based + SOTA-model-scored (using Phase 3C data)
4.  Consolidation passes using replay buffer (Phase 3D)

### Category-Specific Focus

-   Behavioral: preference/RL emphasis, SOTA model as judge
-   Skills: executable verification primary, SOTA model for edge cases
-   Knowledge: SOTA model validates grounding against sources
-   Procedural: SOTA model generates multi-step trajectories, execution verifies them

---

## Phase 6 --- Evaluation & Closed-Loop Iteration

-   Frozen evaluation suites per category (target model evaluated, not oracle)
-   Regression dashboards
-   Closed-loop: target model failures → SOTA model generates fixes → new training data → retrain
-   Track diminishing returns — stop when SOTA-generated data no longer improves target metrics

---

## Deliverables

-   Capability taxonomy (for GPT-OSS 20B target)
-   Web acquisition specification + query templates
-   Normalized corpus with provenance (source URL + SOTA model generation metadata)
-   SFT dataset (SOTA-generated, Harmony-formatted)
-   Preference dataset (SOTA-ranked pairs)
-   Reward dataset (SOTA-scored + execution-scored)
-   Replay dataset (failure corrections)
-   QC pipeline (automated + SOTA-assisted)
-   Training recipe (staged curriculum for GPT-OSS 20B)
-   Evaluation suite (measuring target model, not oracle)
-   Iteration protocol (closed-loop data regeneration triggers)
