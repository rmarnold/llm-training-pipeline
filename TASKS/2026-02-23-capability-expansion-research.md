# Capability Expansion Research: GPT-OSS 20B Coding Agent

**Date**: 2026-02-23
**Status**: Research Complete -- Pending Implementation Planning
**Base**: GPT-OSS 20B MoE, TUI checkpoint + multi-language adapters (in progress)
**Last Updated**: 2026-02-23 (deep research pass with latest 2025-2026 papers)

---

## Executive Summary

After analyzing the state of the art across SWE-agent, DeepSWE, SERA, OpenHands, RAGEN, CodePlan, MapCoder, Blueprint2Code, Aider, Hybrid-Gym, SWE-Playground, Self-Play SWE-RL, and academic literature from 2024-2026, this report identifies **3 P0 (critical)**, **4 P1 (important)**, and **3 P2 (valuable)** capability areas. The top recommendation is to prioritize **Codebase Navigation** (P0), **Planning Before Coding** (P0), and **Error Recovery / Backtracking** (P0) as the next training phases, as these are the strongest differentiators between agents that solve 30% vs 60%+ of SWE-bench tasks.

The existing pipeline (Harmony format, GRPO rewards, IPO preferences, evaluator dispatch) can accommodate all 10 areas with minimal architectural changes. The primary work is **training data generation** and **new Harmony formatters**.

**Key new finding (Feb 2026)**: Hybrid-Gym (arXiv 2602.16819, Feb 2026) demonstrates that synthetic auxiliary tasks teaching general agent skills (reasoning, repo-exploration, tool use) generalize remarkably well to real-world tasks, achieving a 25.4% absolute gain on SWE-bench Verified. This validates our approach of training capabilities individually then composing them. The cost is 0.07 cents per example vs 2.32 cents for SWE-smith, making it highly compatible with our pipeline's data generation budget.

---

## Priority Rankings

| Priority | Capability | Impact on Autonomous Coding | Effort |
|----------|-----------|---------------------------|--------|
| **P0** | 1. Codebase Navigation & Understanding | Critical -- agents fail when they cannot find relevant code | Medium |
| **P0** | 2. Planning Before Coding | Critical -- planning doubles SWE-bench scores | Medium |
| **P0** | 5. Error Recovery & Backtracking | Critical -- 30% of trajectories require course correction | High |
| **P1** | 4. Testing & Test Generation | High -- TDD agents score 27.8% better on SWE-bench Lite | Medium |
| **P1** | 3. Git & Version Control | High -- all production workflows require git | Low |
| **P1** | 8. Context Window Management | High -- long sessions degrade without it | Medium |
| **P1** | 7. Refactoring & Code Quality | High -- common real-world task type | Low |
| **P2** | 6. Documentation & Communication | Medium -- improves human-agent collaboration | Low |
| **P2** | 10. Build System & Dependency Mgmt | Medium -- blocks resolution of 15-20% of issues | Medium |
| **P2** | 9. Security Awareness | Medium -- specialized, lower frequency | Low |

---

## P0: CRITICAL CAPABILITIES

### 1. Codebase Navigation & Understanding

**Why P0**: SWE-EVO shows agents fail at multi-file tasks (21% vs 65% on single-file). The primary failure mode is NOT code generation quality -- it is the inability to find the right files to edit. DeepSWE, SERA-32B, and SWE-agent all identify navigation as the core bottleneck. Hybrid-Gym (Feb 2026) confirms that repo-exploration is a "transferable skill" that generalizes across task types.

#### State of the Art

**SWE-agent (NeurIPS 2024)**: Designs an Agent-Computer Interface (ACI) with specialized navigation tools:
- `find_file` -- locate files by name pattern
- `search_dir` -- regex search within directory
- `search_file` -- regex search within a specific file
- Repository tree view via git-based file listing
- FQDN-based code maps for structured understanding

**SERA-32B (AI2, Jan 2026)**: Solves 54.2% of SWE-bench Verified. Uses Soft Verified Generation (SVG) where a teacher model navigates repos to generate synthetic (PR description -> patch) pairs. The navigation pattern is embedded in the training data implicitly. Requires only 40 GPU days to train on 2 NVIDIA Hopper GPUs. 26x cheaper than RL, 57x cheaper than prior synthetic data methods.

**DeepSWE (Together AI, Jul 2025)**: 59% on SWE-bench Verified (Pass@1: 42.2%, Pass@16: 71.0%). Trained via RL with rLLM framework on R2EGym dataset over 4,500 real-world SWE tasks across six days on 64 H100 GPUs. Emergent behaviors include: anticipating edge cases, conducting thorough regression tests, and systematic file exploration patterns. Navigation skills emerged from RL training on real repo environments.

**gskill / GEPA (Feb 2026)**: Automatically generates repository-specific skill files (.claude/skills/SKILL.md) that teach agents repo structure, key patterns, and navigation hints. Improves resolve rates from 24% to 93% on specific repos. Skills transfer across models. Uses SWE-smith to generate tasks, then GEPA optimization loop iteratively evolves skills through agent evaluation and reflective proposal. Claude Code with GEPA skills becomes 47% faster while achieving near-perfect resolve rates.

**Hybrid-Gym (Feb 2026, arXiv 2602.16819)**: Decomposes trajectories into fine-grained components and identifies repo-exploration as a key transferable skill. Agents trained on synthetic exploration tasks improve 25.4% on SWE-bench Verified. Only requires 2 Docker images at 0.07 cents per example (vs SWE-smith's 128 images at 2.32 cents per example).

**RepoMaster (2025)**: Uses FQDN-based code maps and systematic exploration strategies.

**Composio (2025)**: Demonstrates that tool design is the primary differentiator for SOTA SWE agents. Better navigation tools = better agents, independent of underlying model quality.

#### What the Model Needs to Learn

| Skill | Tool | Training Signal |
|-------|------|----------------|
| File tree exploration | `list_directory`, `repo_tree` | Trajectory: given issue -> explore tree -> find relevant files |
| Targeted search (grep) | `repo_search` (exists) | When to search for symbols vs. file names vs. error strings |
| Call graph following | `repo_search` + reasoning | "This function is called by X" -> navigate to caller |
| Dependency understanding | `read_file` on Cargo.toml/pyproject.toml | Read manifest -> understand module structure |
| "Where to look" heuristics | Thinking channel | Given error type, predict which files/modules are relevant |
| Repository map generation | New tool: `generate_repo_map` | Produce condensed view of repo structure on demand |
| Skill-file awareness | Thinking channel | Read and apply repo-specific navigation hints (.claude/skills/) |

#### Training Data Requirements

**Format**: `harmony_agent` trajectories with explicit navigation turns

**Volume**: 3,000-5,000 navigation-heavy trajectories per language

**Data sources**:
1. **Hybrid-Gym synthetic tasks** (Feb 2026): Function localization and dependency search tasks. Cheapest and most scalable option at 0.07 cents per example. Teaches transferable repo-exploration skills.
2. **SWE-smith synthetic tasks** (NeurIPS 2025): 50K instances from 128 repos. Tasks require multi-file navigation. Adapt their pipeline to generate GPT-OSS trajectories.
3. **R2EGym environments** (COLM 2025): 8.1K problems across 13 repos with execution environments. Use as evaluation and trajectory generation source.
4. **SERA SVG pipeline**: Generate (PR description -> navigation trajectory -> patch) triples from real commits. 26x cheaper than RL.
5. **Self-generated**: Use existing TUI checkpoint to navigate repos, filter successful trajectories.
6. **gskill/GEPA skill generation**: Generate repo-specific skill files that encode navigation patterns, use these as supplementary training data.

**Trajectory template** (new):
```
User: "Fix the failing test test_parse_nested in src/parser.rs"
Assistant:
  <|thinking|>
  **Observe**: Test failure in parser module. Need to find the test and the implementation.
  **Plan**: 1) List directory structure to understand layout, 2) Search for test definition,
  3) Read the test to understand expected behavior, 4) Find the implementation being tested,
  5) Identify and fix the bug.
  <|/thinking|>
  <|tool_call|> repo_search({"query": "test_parse_nested", "type": "symbol"})
  <|tool_result|> Found in tests/parser_tests.rs:42
  <|tool_call|> read_file({"path": "tests/parser_tests.rs", "start_line": 35, "end_line": 60})
  <|tool_result|> [test code showing expected nested parsing behavior]
  <|thinking|>
  The test expects `parse("(a (b c))")` to return `Node::Group(...)`. Let me find the parse function.
  <|/thinking|>
  <|tool_call|> repo_search({"query": "fn parse", "file_pattern": "src/**/*.rs"})
  ...
```

#### New Tools Needed

| Tool | Arguments | Purpose |
|------|-----------|---------|
| `list_directory` | path, depth, pattern | List files/dirs with optional glob filtering |
| `repo_tree` | depth, include_hidden | Generate condensed repo structure |
| `find_symbol` | symbol_name, kind (function/struct/type) | AST-aware symbol search |

Note: `repo_search` already exists. `list_directory` and `repo_tree` are the main additions.

#### Integration with Pipeline

- **New formatter**: `harmony_navigation` in `dataset_formatters/harmony.py` -- formats multi-step navigation trajectories
- **GRPO reward extension**: Add navigation efficiency signal (fewer search calls to find relevant code = higher reward)
- **IPO pairs**: (verbose navigation with 10 searches, concise navigation with 3 searches) -> prefer concise
- **Curriculum**: Start with "file is specified in prompt" (no search needed), progress to "find the file from error message only"

#### GRPO Reward Signals (Navigation)

| Signal | Reward | Condition |
|--------|--------|-----------|
| Found relevant file in <= 3 searches | +0.1 | Navigation efficiency bonus |
| Used targeted line-range read | +0.05 | vs full-file read |
| Used `repo_tree` before random searches | +0.05 | Strategic exploration |
| Redundant search (same query repeated) | -0.1 | Penalize waste |

#### Estimated Difficulty: Medium
- New tools: 1-2 days per tool
- Training data generation: 3-5 days (adapt Hybrid-Gym or SWE-smith pipeline)
- Harmony formatter: 1 day
- GRPO reward update: 1 day
- Total: ~2 weeks including evaluation

---

### 2. Planning Before Coding

**Why P0**: Agents that plan before editing achieve 2x higher resolve rates. CodePlan (Microsoft, FSE 2024) showed that without planning, NO multi-file repos passed validity checks. MapCoder's plan-then-code pipeline achieves 93.9% on HumanEval. SWE-EVO (Jan 2026) shows current agents fail at multi-step tasks precisely because they lack sustained planning. Blueprint2Code (2025) demonstrates that even small models achieve excellent results when given structured planning pipelines.

#### State of the Art

**CodePlan (Microsoft, FSE 2024)**: Repository-level coding as planning. Uses incremental dependency analysis + change may-impact analysis + adaptive planning. Got 5/7 repos to build without errors; baselines got 0/7. Code and eval scripts are publicly available.

**MapCoder (ACL 2024)**: 4-agent pipeline: Retrieval -> Planning -> Coding -> Debugging. The planning agent produces step-wise plans with confidence scores. Top plan is passed to coding agent. If debugging fails, backtrack to alternative plans. Results: HumanEval 93.9%, MBPP 83.1%, APPS 22.0%, CodeContests 28.5%.

**Blueprint2Code (Frontiers AI, 2025)**: 4-stage pipeline inspired by human workflow: Preview Agent (learns relevant algorithms), Blueprint Agent (generates ranked step-by-step plans), Coding Agent (implements best plan), Debugging Agent (iterative repair, max 5 rounds). If debugging fails, control returns to Blueprint Agent for re-planning. Significantly outperforms CoT, Reflexion, and MapCoder. Maintains performance in resource-constrained small model environments -- critical validation for our 20B model.

**CodePlan (Tsinghua, ICLR 2025)**: Code-form plans (pseudocode as planning representation). Enables automatic extraction of plans from massive corpora without curated task-specific datasets. Scalable learning from existing text, not just annotated data.

**Open SWE (LangChain, 2025)**: Multi-agent with dedicated Planner and Reviewer. Planner researches codebase to form a robust strategy first, then code is written.

**Cline/Roo Code (2025)**: Dual "Plan" and "Act" modes. Agent first devises a plan, then executes steps one by one.

**Visual Studio Copilot Planning (Oct 2025)**: Microsoft added explicit planning mode to VS Code Copilot. Agent creates implementation plan -> user reviews -> agent executes.

#### What the Model Needs to Learn

| Skill | Harmony Channel | Training Signal |
|-------|----------------|----------------|
| Decompose task into steps | `<\|thinking\|>` block | List numbered implementation steps before any tool calls |
| Estimate scope (files, changes) | `<\|thinking\|>` block | "This requires changes to 3 files: X, Y, Z" |
| Dependency ordering | `<\|thinking\|>` block | "Must change interface first, then update callers" |
| Plan verification | `<\|thinking\|>` after tool results | "Step 1 complete. Moving to step 2." |
| Plan revision | `<\|thinking\|>` after failure | "Step 3 failed. Revising: instead of X, try Y" |
| Scope awareness | `<\|assistant\|>` | "This task is too large for a single change. Breaking into 3 PRs." |
| Blueprint ranking | `<\|thinking\|>` | Generate multiple plans, rank by confidence, try best first |
| Re-planning after failure | `<\|thinking\|>` | Return to planning phase when implementation fails after 2 attempts |

#### Training Data Requirements

**Format**: New `harmony_plan` formatter

**Volume**: 2,000-4,000 plan-execute trajectories per language

**Data sources**:
1. **CodePlan dataset**: Microsoft released data + eval scripts for package migration and temporal edits
2. **Blueprint2Code trajectories**: Adapt their 4-stage pipeline output into Harmony format with plan -> code -> debug arc
3. **SWE-smith multi-file tasks**: Filter for tasks requiring 3+ file changes, generate plan-first trajectories
4. **SWE-Playground synthetic projects**: Full project generation with diverse task types, including planning
5. **Claude API generation**: Give Claude a multi-file task, ask for explicit plan then execution trace
6. **Self-play**: TUI checkpoint generates plans, executes them, filter by success
7. **Tsinghua CodePlan corpus**: Extract code-form plans from existing code repositories automatically

**Plan format in Harmony**:
```
<|thinking|>
**Plan**:
1. Read `src/config.rs` to understand the current config structure (read_file)
2. Add new `timeout` field to ConfigStruct (apply_patch to src/config.rs)
3. Update `ConfigStruct::default()` to include timeout default (apply_patch)
4. Update CLI parser in `src/main.rs` to accept --timeout flag (apply_patch)
5. Add test for timeout parsing in `tests/config_test.rs` (apply_patch)
6. Run tests to verify (run_tests)

**Estimated scope**: 3 files, ~40 lines changed
**Risk**: Changing ConfigStruct is a breaking change if other modules destructure it
**Confidence**: 0.85
**Alternative plan**: If ConfigStruct change breaks callers, use builder pattern instead
<|/thinking|>
```

#### GRPO Reward Extension

| Signal | Reward | Condition |
|--------|--------|-----------|
| Plan present before first edit | +0.1 | Has numbered plan in thinking block |
| Plan steps match actual actions | +0.1 | >70% of planned steps were executed |
| No unnecessary changes | +0.05 | Files touched <= files mentioned in plan |
| Task completed in fewer iterations | +0.1 | Below median iteration count |
| Plan includes confidence score | +0.02 | Explicit confidence in thinking |
| Re-plan after 2 failed attempts | +0.1 | Returns to planning rather than brute-forcing |

**Important**: Only reward plan quality when the code is correct (P-GRPO gating from roadmap).

#### Estimated Difficulty: Medium
- Harmony formatter: 1 day
- Training data generation: 5-7 days (requires Claude API calls for gold trajectories)
- GRPO reward signals: 2 days
- IPO preference pairs: 2 days (plan-then-execute vs jump-to-code)
- Total: ~2.5 weeks

---

### 5. Error Recovery & Backtracking

**Why P0**: RAGEN research shows RL-trained agents fall into "echo traps" -- repetitive failing patterns. In our current training data, `generate_multi_step_trajectory()` already teaches basic retry behavior, but it is synthetic and simplistic. Real-world recovery requires: recognizing a dead end, undoing a change, trying a fundamentally different approach, and knowing when to stop. BacktrackAgent (EMNLP 2025) proves error detection is explicitly teachable via SFT+RL.

#### State of the Art

**RAGEN / StarPO (Apr 2025)**: Identifies the "Echo Trap" -- agents overfit to locally rewarded reasoning patterns. Marked by reward variance collapse, entropy drop, gradient spikes. Solution: StarPO-S with trajectory filtering, critic incorporation, gradient stabilization.

**DeepSWE (2025)**: Through RL, emergent behaviors appeared: self-correction, alternative approach testing, knowing when to step back and re-analyze. These were NOT explicitly taught -- they emerged from execution rewards over many training steps.

**BacktrackAgent (EMNLP 2025)**: Explicitly trains agents to detect errors and backtrack in GUI tasks. Uses three learned components: **Verifier** (detects if current state is correct), **Judger** (decides whether to continue or backtrack), **Reflector** (generates corrective plan after backtracking). All three are trainable via SFT then RL. Achieved 7.59% increase in Task Success Rate on Mobile3M. Key insight: error detection is a teachable skill, not just an emergent property.

**Multi-Turn RL for SWE Agents (Nebius, Aug 2025, arXiv 2508.03501)**: Uses modified DAPO algorithm for long-horizon, multi-turn SWE scenarios. Key finding: "RL naturally supports error recovery: if taking a corrective action after a mistake leads to higher final reward, the policy can learn to do that." Technical innovations for DAPO in SWE context:
- **Asymmetric Clipping**: Prevents collapse in policy entropy
- **Dynamic Sample Filtering**: Focuses on trajectories with learning signal
- **Length Penalties**: Discourages excessive episode length
- **Token-Level Averaging**: Every token contributes equally to gradient
RL-trained Qwen2.5-72B achieves ~39% on SWE-bench Verified, doubling the baseline.

**Self-Play SWE-RL / SSR (Meta, Dec 2025)**: Single LLM trains via RL in self-play: iteratively inject and repair bugs of increasing complexity. Bug specification via test patches, not natural language. No human-labeled data needed. Provides a curriculum of increasing difficulty that naturally teaches recovery -- harder bugs require more iterations and backtracking.

**AgentGym-RL (2025)**: Supports agent behaviors ranging from direct action selection to long-horizon deliberation and recovery from failures.

#### What the Model Needs to Learn

| Skill | Current State | Target State |
|-------|--------------|-------------|
| Retry same approach | Yes (multi-step trajectories) | Keep (baseline) |
| Try different approach | Not trained | Teach: if 2 retries fail, change strategy |
| Undo/revert changes | Not trained | Teach: revert patch, try alternative |
| Detect infinite loops | GRPO penalizes (infinite_retry_loop: -0.5) | Also teach proactive detection in thinking |
| Know when to ask for help | Not trained | Teach: "I cannot resolve this without X" |
| Partial progress reporting | Not trained | Teach: "I fixed issue A but issue B remains" |
| Verifier behavior | Not trained | Teach: check if current state is as expected before proceeding |
| Judge behavior | Not trained | Teach: decide continue vs. backtrack based on evidence |
| Reflector behavior | Not trained | Teach: after backtracking, generate new strategy based on what failed |

#### Training Data Requirements

**Format**: Extended `harmony_agent` trajectories with explicit failure-and-recovery arcs

**Volume**: 2,000-3,000 recovery trajectories per language (mix with existing data)

**Data generation strategy**:
1. **BacktrackAgent-style SFT data**: Create (state, is_error, should_backtrack, recovery_plan) training triples. Train verifier/judger/reflector skills in the thinking channel.
2. **Intentional failure injection**: Generate trajectories where step 2 is intentionally wrong, then model must detect and recover
3. **RL self-play with failure paths**: During GRPO, save trajectories where the model failed then recovered -- these are the most valuable training data
4. **Self-Play SWE-RL (SSR) pipeline**: Adapt Meta's dual-role self-play (bug injection + repair) for our Rust/Python/TS/Go tasks. No human labels needed.
5. **Timeout recovery**: Trajectories where a command times out and the model adapts
6. **Environment error handling**: Trajectories where `cargo test` produces unexpected output, model adapts

**Trajectory template** (failure-recovery with BacktrackAgent-style reasoning):
```
<|thinking|>
[VERIFY] I patched the type annotation. Expected: tests pass. Actual: tests still fail.
The error is different now -- this is a logic error, not a type error.
[JUDGE] My initial hypothesis was wrong. Continuing down this path will not help.
Decision: BACKTRACK.
[REFLECT] My first attempt assumed the problem was a type mismatch, but the real issue
is in the comparison logic. I should revert my patch and re-analyze the function body
instead of the type signatures.
<|/thinking|>
<|tool_call|> apply_patch({"diff": "revert previous change"})
<|tool_result|> Patch applied successfully.
<|thinking|>
[REFLECT continued] Starting fresh. The real issue is in the comparison function.
Let me re-read with focus on the equality check logic.
<|/thinking|>
```

#### GRPO Reward Extension

| Signal | Reward | Description |
|--------|--------|-------------|
| Recovery from failure | +0.2 bonus | Final success after intermediate failure |
| Clean revert before retry | +0.05 | Undo bad change before trying alternative |
| Escalation signal | 0.0 (neutral) | "Cannot fix this" when truly stuck -- no penalty |
| Infinite retry (>5 same error) | -0.5 (existing) | Already implemented |
| **New: Echo detection** | -0.3 | Same tool call with same args repeated 3+ times |
| **New: Strategy diversity** | +0.1 | Uses different tool sequence on retry |
| **New: Verify-before-proceed** | +0.05 | Checks outcome after each patch before moving on |
| **New: Correct backtrack decision** | +0.1 | Backtracks when approach is failing (judged by outcome) |

#### Anti-Echo-Trap Measures (from RAGEN + DAPO)

For GRPO training, implement StarPO-S stabilization plus DAPO modifications:
1. **Trajectory filtering**: Discard trajectories with reward variance < threshold
2. **Diverse initial states**: Start each GRPO episode from a random point in the task space (not always from the beginning)
3. **Entropy bonus**: Add KL penalty if model's action distribution becomes too peaked
4. **Gradient clipping**: Already implemented (`max_grad_norm: 0.5`), but monitor for spikes
5. **Asymmetric clipping** (from DAPO): Prevent entropy collapse during long-horizon training
6. **Dynamic sample filtering** (from DAPO): Focus compute on trajectories with learning signal
7. **Length penalties** (from DAPO): Discourage excessive episode length
8. **Self-play curriculum** (from SSR): Progressively harder bug injection -> repair cycles

#### Estimated Difficulty: High
- BacktrackAgent-style SFT data: 3-5 days
- Failure trajectory generation: 5-7 days (requires running model, collecting failures)
- GRPO reward updates: 2 days
- StarPO-S + DAPO stabilization: 3-5 days
- Evaluation framework for recovery: 3 days
- Total: ~3-4 weeks

---

## P1: IMPORTANT CAPABILITIES

### 4. Testing & Test Generation

**Why P1**: TDFlow (2025) shows TDD-first agents score 27.8% better on SWE-bench Lite, reaching 88.8% with human-written tests and 94.3% on SWE-bench Verified. The model currently runs tests but does not write them. Test generation is the "final frontier" for fully autonomous repository repair (TDFlow's own conclusion). SWE-Playground confirms that test creation is a distinct skill that transfers across tasks.

#### State of the Art

- **TDFlow (2025, arXiv 2510.23761)**: Agentic TDD workflow -- write tests first, then implement. 27.8% improvement over non-TDD agents. Uses four sub-agents with tightly constrained tools, reducing long-context burden. Works with Claude 4 Sonnet, Kimi K2, Qwen3-Coder, Gemini 2.5 Pro. Key finding: "The final frontier for fully autonomous repository repair is the accurate generation of valid reproduction tests."
- **SWE-Playground (Dec 2025, arXiv 2512.12216)**: Generates synthetic projects with diverse tasks including test creation. Agents trained on SWE-Play-mix achieve competitive results across SWE-bench (issue resolution), SWT-Bench (test generation), and Commit-0 (library construction) using only 704 trajectories. Test creation is treated as a first-class training objective, not an afterthought.
- **Hybrid-Gym (Feb 2026)**: Identifies test generation as a task-specific behavior (not a transferable skill), meaning it needs dedicated training data rather than emerging from general agent training.
- **Tests as Instructions (2025)**: TDD benchmark where tests ARE the instructions. Identifies instruction following and in-context learning as critical abilities.
- **AgentCoder (2024)**: Multi-agent code generation with iterative testing and optimization.
- **Tweag Agentic Coding Handbook (2025)**: Practical TDD patterns for AI agents: write failing test -> implement minimal code -> refactor. Tested with Claude Code and Copilot.

#### Training Data Requirements

| Data Type | Volume | Source | Difficulty |
|-----------|--------|--------|------------|
| Test generation trajectories | 2,000 per language | Given function -> write tests covering edge cases | Medium |
| TDD trajectories | 1,000 per language | Write failing test -> implement -> pass | Medium |
| Coverage gap analysis | 500 per language | Identify untested paths, write targeted tests | High |
| Property-based testing | 500 per language | Generate proptest/hypothesis specs from function signatures | High |
| Reproduction test generation | 1,000 per language | Given bug report -> write test that reproduces bug | High (highest value) |

**Critical data source**: SWT-Bench provides test generation evaluation tasks. SWE-Playground provides synthetic test-creation training data.

#### New Tools

| Tool | Purpose |
|------|---------|
| `run_tests_with_coverage` | Run tests and return coverage report |
| `list_untested_functions` | Identify functions without test coverage |
| `run_single_test` | Run a specific test to check if it passes/fails |

#### GRPO Reward Signals

| Signal | Reward |
|--------|--------|
| Written test catches a real bug | +0.3 |
| Written test passes after implementation | +0.2 |
| Generated test has high branch coverage | +0.1 |
| Generated test is flaky (non-deterministic) | -0.3 |
| Reproduction test fails before fix, passes after | +0.3 |
| Test is redundant (same assertion as existing test) | -0.1 |

#### Harmony Format

New `harmony_test_gen` formatter:
```
<|thinking|>
I need to write tests for the `parse_config` function in `src/config.rs`.
Let me identify the key behaviors to test:
1. Valid config with all fields
2. Missing required field
3. Invalid type for a field
4. Empty config
5. Config with unknown fields (should be ignored or error?)
<|/thinking|>
<|tool_call|> read_file({"path": "src/config.rs", "start_line": 10, "end_line": 45})
<|tool_result|> [function implementation]
<|thinking|>
The function returns Result<Config, ConfigError>. I see 3 error paths and 2 success paths.
Let me write tests covering all 5 paths plus edge cases.
<|/thinking|>
<|tool_call|> apply_patch({"diff": "add test module with 5 test cases..."})
<|tool_result|> Patch applied successfully.
<|tool_call|> run_tests({"cmd": "cargo test test_parse_config"})
<|tool_result|> 5 passed; 0 failed
```

#### Estimated Difficulty: Medium
- 2-3 weeks for data generation + formatters + evaluation

---

### 3. Git & Version Control

**Why P1**: Every production coding workflow involves git. The model currently uses `apply_patch` which generates diffs, but it does not understand branches, commits, blame, log, or merge conflicts. Aider demonstrates that git-native workflows significantly improve the development experience. The AIDev-pop dataset (33,596 agentic PRs) reveals that each AI agent has distinctive git patterns -- we should train ours to have good patterns.

#### State of the Art

- **Aider (2025-2026)**: Git-native, human-in-the-loop workflow with auto-commit, conventional commit messages, and unified diff format. Uses weak model for commit message generation. Every AI-suggested code change gets automatic commit with clear message. Adapts to branch changes mid-session.
- **OpenHands (2025)**: Auto-generates PRs from issue descriptions. Git operations are first-class tools.
- **GitHub Copilot Workspace (2025)**: Creates branches, commits, and PRs as part of the agent workflow.
- **GitHub Agentic Workflows (Feb 2026)**: Technical preview of fully autonomous GitHub workflows. Agents can create branches, make changes, commit, and open PRs.
- **Claude Code git integration**: Automatically generates conversational commits, creates PRs in one command, assists with conflict resolution.
- **AIDev-pop dataset (Feb 2026)**: 33,596 agentic PRs from 5 coding agents (OpenAI Codex 64.9%, GitHub Copilot 14.8%, Devin 14.4%, Cursor 4.6%, Claude Code 1.4%). Includes metadata, commit messages, code changes, and reviews. A multi-class classifier achieves 97.2% F1-score identifying which agent wrote a PR, revealing distinctive agent-specific signatures.

#### Training Data Requirements

| Data Type | Volume | Source |
|-----------|--------|--------|
| Commit message generation | 5,000 examples | AIDev-pop dataset (33,596 agentic PRs), Conventional Commits spec |
| PR description generation | 2,000 examples | AIDev-pop PRs with descriptions + reviews |
| Branch management | 500 trajectories | Create branch -> make changes -> commit -> PR |
| Merge conflict resolution | 1,000 examples | Synthetic conflicts from real repos |
| Git blame for context | 500 trajectories | Use blame to find relevant commit -> understand change |
| Diff interpretation | 1,000 examples | Given diff, explain what changed and why |

#### New Tools

| Tool | Purpose |
|------|---------|
| `git_status` | Show working tree status |
| `git_diff` | Show changes (staged/unstaged) |
| `git_log` | Show recent commits |
| `git_blame` | Show last modification per line |
| `git_commit` | Create commit with message |
| `git_branch` | Create/switch branches |
| `create_pr` | Create pull request with title and description |

#### GRPO Reward Signals

| Signal | Reward |
|--------|--------|
| Commit message follows Conventional Commits format | +0.05 |
| PR description mentions what/why/how | +0.05 |
| Clean diff (no unrelated changes) | +0.05 |
| Merge conflict resolved correctly (tests pass) | +0.2 |
| Commit includes only related files | +0.05 |

#### Estimated Difficulty: Low
- Tools are thin wrappers around git commands
- Training data is abundant (AIDev-pop alone provides 33K examples)
- 1-2 weeks

---

### 8. Context Window Management

**Why P1**: Agents operating on large repos hit context limits. JetBrains research (Dec 2025) found observation masking outperforms LLM summarization while being cheaper. Anthropic's context engineering guidance emphasizes selective file reading over full-file loading. This is especially critical for GPT-OSS 20B which uses 32K context in GRPO curriculum.

#### State of the Art

- **Observation Masking (arXiv 2508.21433, 2025)**: Targets only environment observation (tool results), preserving full action/reasoning history. Replaces older tool outputs with placeholder (e.g., "Previous 8 lines omitted for brevity"). Lowest cost per instance in 4/5 experimental setups. Achieves on-par results with LLM summarization at fraction of the cost. Agent context is heavily skewed toward environment observations (tool outputs), making this the highest-leverage compression target.
- **Hierarchical Summarization**: Compress older conversation segments; keep recent verbatim.
- **Sub-Agent Architecture**: Specialized sub-agents with clean context windows, returning condensed summaries.
- **Claude Code Pattern**: Hybrid model with upfront CLAUDE.md + just-in-time file retrieval via glob/grep. Auto-compact at 95% capacity: summarizes entire trajectory preserving architectural decisions, unresolved bugs, and implementation details while discarding redundant tool outputs. Continues with compressed context plus 5 most recently accessed files.
- **gskill (Feb 2026)**: Auto-generates repo-specific skill files that provide compressed context about repo structure, reducing the need for broad exploration.
- **Anthropic context engineering guide (2025)**: "Effective context engineering for AI agents" -- framework for deciding what goes in context and when.

#### What the Model Needs to Learn

| Skill | Training Signal |
|-------|----------------|
| Read sections, not full files | `read_file(path, start_line=X, end_line=Y)` -- reward for smaller reads |
| Summarize before discarding | Thinking block: "Key takeaway from this file: X. I no longer need the full content." |
| Prioritize recent context | When replanning, reference recent observations, not stale ones |
| Request repo overview first | Start with `repo_tree` or manifest file, not random file reads |
| Know when to re-read | "This was 10 turns ago. Let me re-read to confirm." |
| Context budget awareness | "I have ~20K tokens remaining. I should be selective about what I read." |

#### GRPO Reward Signals

| Signal | Reward |
|--------|--------|
| Solution found with fewer total tokens read | +0.1 |
| Targeted read (line range) vs full file | +0.05 |
| Successful completion without context overflow | +0.1 |
| Read same file twice unnecessarily | -0.05 |

#### Implementation Note

The observation masking technique can be applied as a **data preprocessing step** during trajectory generation, not just at inference time. When generating training trajectories, apply masking to older tool results before formatting as Harmony. This teaches the model to work with partially masked context.

#### Estimated Difficulty: Medium
- Requires modifying trajectory generation to include context-efficient patterns
- Observation masking integration into data pipeline: 2 days
- Training data with masked observations: 3 days
- 2-3 weeks total

---

### 7. Refactoring & Code Quality

**Why P1**: 52.5% of AI agent refactorings target maintainability vs 11.7% for humans. Agents are naturally suited to mechanical refactoring tasks. This capability also improves the model's general code understanding. The Anthropic 2026 Agentic Coding Trends Report notes that AI demonstrates "repository intelligence" -- understanding relationships and intent, not just individual lines.

#### State of the Art

- Agents perform medium-level refactorings (extract method, change parameter types) at rates similar to humans (~21%)
- AI refactoring reliability is highest for: variable renaming, function extraction, dead code elimination
- Lower reliability for: architectural decisions, design pattern changes
- Anthropic 2026 report: Rakuten tested Claude Code on implementing an activation vector extraction method in vLLM (12.5M-line codebase), completing in 7 hours with 99.9% numerical accuracy -- demonstrating that agents can handle large-scale code modifications

#### Training Data Requirements

| Data Type | Volume | Source |
|-----------|--------|--------|
| Extract method | 1,000 per language | Before/after with extracted function |
| Dead code removal | 500 per language | Code with unused functions -> removed |
| Type narrowing | 500 per language | Broad types -> specific types |
| Performance optimization | 500 per language | Inefficient -> optimized (with benchmarks) |
| Rename/move | 500 per language | Consistent renaming across all references |
| Reduce duplication | 500 per language | DRY violations -> shared utility |

#### New Harmony Formatter

`harmony_refactor`:
```
<|thinking|>
**Before**: The `process_data` function is 150 lines with 3 distinct responsibilities.
**Rationale**: Extract the validation, transformation, and persistence into separate functions.
**Plan**: 1) Extract validate_input(), 2) Extract transform_data(), 3) Extract persist_result()
**Risk**: Must maintain the same error handling behavior at each step.
**Verification**: Run existing tests after each extraction to ensure no regressions.
<|/thinking|>
```

#### GRPO Reward Signals

| Signal | Reward |
|--------|--------|
| All tests pass after refactoring | +0.5 (primary gate) |
| Reduced cyclomatic complexity | +0.1 |
| Reduced function length | +0.05 |
| No behavior change (tests still pass) | Required (zero reward otherwise) |
| clippy/ruff/eslint warnings reduced | +0.1 |

#### Estimated Difficulty: Low
- 1-2 weeks (data generation from real refactoring PRs is straightforward)

---

## P2: VALUABLE CAPABILITIES

### 6. Documentation & Communication

**Why P2**: Important for human-agent collaboration but not for autonomous task completion. The model already generates `<|assistant|>` responses; this refines their quality. The AIDev-pop study shows that different AI agents have distinctive communication patterns -- we should train ours to produce high-quality, human-friendly output.

#### Key Skills

| Skill | Data Source | Volume |
|-------|-----------|--------|
| Commit message generation | AIDev-pop dataset, Conventional Commits spec | 5,000 examples |
| PR description writing | AIDev-pop PRs with reviews + human reactions | 2,000 examples |
| Code comment generation | Doc-comment extraction from real code | 3,000 examples |
| Asking clarifying questions | Ambiguous task -> "I need clarification on X" | 500 examples |
| Change explanation | (diff, human-readable explanation) pairs | 1,000 examples |
| Status updates | Progress reporting during long tasks | 500 examples |

#### AIDev-pop Study Insights

The AIDev-pop study (Feb 2026) reveals that AI agent PRs can be distinguished from each other with 97.2% accuracy based on their communication patterns. This means the current generation of agents have not been explicitly trained on communication quality -- they each have idiosyncratic patterns. Training on high-quality PR descriptions and commit messages from the best-reviewed PRs in the dataset could significantly improve our agent's communication.

#### Estimated Difficulty: Low (1 week)

---

### 10. Build System & Dependency Management

**Why P2**: Blocks 15-20% of SWE-bench issues. The model currently relies on language evaluators for build execution but does not understand build system concepts.

#### Key Skills

| Skill | Example |
|-------|---------|
| Read and modify Cargo.toml / pyproject.toml / package.json / go.mod | Add dependency, update version |
| Resolve dependency conflicts | "Version X requires Y >= 2.0 but Z requires Y < 2.0" |
| Interpret build errors | "linking failed" -> missing system library |
| CI/CD pipeline understanding | Read .github/workflows/*.yaml, understand failure context |
| Feature flag management | Enable/disable features in Cargo.toml/package.json |

#### Training Data Sources

- Real Cargo.toml/pyproject.toml/package.json modification commits from GitHub
- Dependency update PRs (Dependabot, Renovate bot)
- CI failure -> fix pairs from GitHub Actions logs
- SWE-bench tasks that require dependency changes (filter from SWE-smith)

#### Estimated Difficulty: Medium (2-3 weeks)

---

### 9. Security Awareness

**Why P2**: Specialized capability. Important for production code but not the primary use case for bug fixing / feature implementation. The OWASP Top 10 for LLM Applications (2025) includes "Excessive Agency" as a risk, which is directly relevant to coding agents.

#### Key Skills

| Skill | Example |
|-------|---------|
| Recognize injection vulnerabilities | SQL injection, command injection, path traversal |
| Avoid hardcoded secrets | Detect API keys, passwords in code |
| Secure defaults | Prefer parameterized queries, validate inputs |
| Dependency vulnerability awareness | Flag known CVEs in dependencies |
| Excessive agency prevention | Refuse to execute dangerous system commands |
| MCP security | Secure Model Context Protocol interactions |

#### Data Source
- OWASP Top 10 2025 examples
- CWE database
- Security-focused code review datasets
- Secure coding patterns per language (Rust has memory safety by default, but unsafe blocks need scrutiny)
- OWASP LLM Application Security resources

#### GRPO Reward Signals

| Signal | Reward |
|--------|--------|
| Code avoids known vulnerability patterns | +0.1 |
| Refuses to embed secrets in code | +0.2 |
| Suggests input validation where missing | +0.05 |
| Uses `unsafe` without justification (Rust) | -0.3 (existing: unnecessary_unsafe) |

#### Estimated Difficulty: Low (1-2 weeks for SFT data, no new tools needed)

---

## Implementation Roadmap

### Phase 6: Capability Expansion (post-multi-language adapters)

```
Phase 6a: P0 Capabilities (critical -- do first)
  |-- Navigation tools + data gen     [2 weeks]
  |-- Planning formatter + data gen   [2.5 weeks]  (can overlap with 6a navigation)
  '-- Error recovery trajectories     [3-4 weeks]  (requires running model)

Phase 6b: P1 Capabilities (important -- do next)
  |-- Test generation data + tools    [2-3 weeks]
  |-- Git tools + data gen            [1-2 weeks]
  |-- Context management training     [2-3 weeks]
  '-- Refactoring data gen            [1-2 weeks]

Phase 6c: P2 Capabilities (valuable -- do last)
  |-- Documentation/communication     [1 week]
  |-- Build system understanding      [2-3 weeks]
  '-- Security awareness SFT          [1-2 weeks]
```

### Data Generation Strategy (recommended approach)

Based on the research, the most cost-effective data generation pipeline combines multiple approaches:

| Approach | Cost per Example | Examples Available | Best For |
|----------|-----------------|-------------------|----------|
| **Hybrid-Gym synthetic tasks** | $0.0007 | Unlimited | Navigation, general skills |
| **SWE-Playground** | ~$0.02 | Unlimited | Test gen, library construction |
| **SERA SVG pipeline** | ~$0.01 | ~10K per repo set | Navigation + patching |
| **SWE-smith** | $0.0232 | 50K+ | Multi-file tasks |
| **Self-Play SWE-RL (SSR)** | GPU time only | Unlimited | Error recovery |
| **AIDev-pop** | Free (existing) | 33,596 PRs | Git, documentation |
| **Claude API gold trajectories** | ~$0.05 | Budget-limited | Planning, recovery (highest quality) |
| **R2EGym environments** | ~$0.03 | 8.1K | Evaluation + RL training |

**Recommended priority**:
1. Start with Hybrid-Gym for navigation (cheapest, proven transferability)
2. Use SWE-Playground for test generation tasks
3. Use AIDev-pop (free) for git and documentation
4. Use SSR self-play for error recovery (no data cost, just GPU time)
5. Reserve Claude API budget for gold plan-execute trajectories

### New Scripts Needed

| Script | Purpose | Phase |
|--------|---------|-------|
| `20_generate_navigation_data.py` | Generate navigation-heavy trajectories from repos | 6a |
| `21_generate_planning_data.py` | Generate plan-then-execute trajectories | 6a |
| `22_generate_recovery_data.py` | Generate failure-recovery trajectories (requires model inference) | 6a |
| `23_generate_test_writing_data.py` | Generate test-generation trajectories | 6b |
| `24_generate_git_data.py` | Generate git workflow trajectories from real PRs | 6b |
| `25_generate_refactoring_data.py` | Generate refactoring before/after pairs | 6b |

### New Harmony Formatters Needed

| Formatter | Priority | Location |
|-----------|----------|----------|
| `harmony_navigation` | P0 | `dataset_formatters/harmony.py` |
| `harmony_plan` | P0 | `dataset_formatters/harmony.py` |
| `harmony_recovery` | P0 | `dataset_formatters/harmony.py` |
| `harmony_test_gen` | P1 | `dataset_formatters/harmony.py` |
| `harmony_git` | P1 | `dataset_formatters/harmony.py` |
| `harmony_refactor` | P1 | `dataset_formatters/harmony.py` |

### New Tools for the Agent

| Tool | Arguments | Priority | Notes |
|------|-----------|----------|-------|
| `list_directory` | path, depth, pattern | P0 | Structured directory listing |
| `repo_tree` | depth, include_hidden | P0 | Condensed repo overview |
| `find_symbol` | name, kind | P0 | AST-aware symbol search |
| `git_status` | -- | P1 | Working tree status |
| `git_diff` | path, staged | P1 | Show changes |
| `git_log` | n, path | P1 | Recent commits |
| `git_blame` | path, line_range | P1 | Per-line attribution |
| `git_commit` | message, files | P1 | Create commit |
| `create_pr` | title, body, branch | P1 | Create pull request |
| `run_tests_with_coverage` | cmd, report_format | P1 | Tests + coverage |
| `list_untested_functions` | path | P1 | Functions without coverage |
| `run_single_test` | test_name | P1 | Run specific test |

### GRPO Reward Function Expansion

Update `configs/grpo.yaml` and `pipeline_lib/evaluator_dispatch.py`:

```yaml
# Extended rewards (Phase 6)
rewards:
  # Existing
  all_tests_pass_clippy_clean: 1.0
  all_tests_pass_clippy_warnings: 0.7
  compilation_success_some_tests_fail: 0.1
  compilation_failure: -0.3
  invalid_tool_call_format: -1.0
  unnecessary_unsafe: -0.3
  infinite_retry_loop: -0.5

  # New: Navigation efficiency
  navigation_efficiency_bonus: 0.1      # Found relevant code in <= 3 searches
  targeted_file_read_bonus: 0.05        # Used line ranges instead of full file
  strategic_exploration_bonus: 0.05     # Used repo_tree before random searches

  # New: Planning quality (P-GRPO gated -- only when code is correct)
  plan_present_bonus: 0.1               # Has numbered plan before first edit
  plan_accuracy_bonus: 0.1              # >70% of plan steps executed
  scope_accuracy_bonus: 0.05            # Files touched <= files in plan
  replan_after_failure_bonus: 0.1       # Returns to planning after 2 failed attempts

  # New: Recovery
  recovery_from_failure_bonus: 0.2      # Final success after intermediate failure
  clean_revert_bonus: 0.05              # Undo bad change before retry
  echo_detection_penalty: -0.3          # Same tool call repeated 3+ times
  strategy_diversity_bonus: 0.1         # Different approach on retry
  verify_before_proceed_bonus: 0.05     # Checks outcome after patch

  # New: Test quality
  test_catches_bug_bonus: 0.3           # Written test catches actual bug
  reproduction_test_bonus: 0.3          # Fails before fix, passes after
  test_is_flaky_penalty: -0.3           # Non-deterministic test
  redundant_test_penalty: -0.1          # Same assertion as existing test

  # New: Git quality
  clean_commit_message_bonus: 0.05      # Conventional Commits format
  clean_diff_bonus: 0.05                # No unrelated changes

  # New: Context efficiency
  context_efficiency_bonus: 0.1         # Fewer tokens read for same outcome
  unnecessary_reread_penalty: -0.05     # Read same file twice
```

### IPO Preference Expansion

New preference dimensions for Phase 6:

| Dimension | Chosen (preferred) | Rejected |
|-----------|-------------------|----------|
| Navigation | 3 targeted searches find code | 10 broad searches find code |
| Planning | Explicit plan then execute | Jump to editing immediately |
| Planning (advanced) | Plan with confidence + alternative | Single plan, no fallback |
| Recovery | Revert + try different approach | Keep retrying same approach |
| Recovery (advanced) | Verify/Judge/Reflect pattern | Blind retry without analysis |
| Tests | Writes regression test after fix | Fixes bug without test |
| Tests (TDD) | Write failing test -> implement | Implement -> write passing test |
| Git | Clean commit with good message | No commit / vague message |
| Context | Reads relevant sections only | Reads entire large files |
| Communication | Explains reasoning and uncertainty | Silent or cryptic output |

---

## Key Research References

### Papers (sorted by impact)

**P0 - Critical References**:
- **Hybrid-Gym** (Feb 2026, arXiv 2602.16819): Training agents on synthetic tasks for transferable skills. 25.4% gain on SWE-bench Verified. Cheapest data generation.
- **DeepSWE** (Together AI, Jul 2025): 59% SWE-bench via RL on Qwen3-32B. 64 H100s x 6 days. Fully open-sourced.
- **SERA** (AI2, Jan 2026): Soft Verified Generation. 54.2% SWE-bench. 40 GPU days on 2 Hopper GPUs. 26x cheaper than RL.
- **RAGEN / StarPO** (Apr 2025, arXiv 2504.20073): Echo trap detection and multi-turn RL stabilization.
- **Multi-Turn RL for SWE** (Nebius, Aug 2025, arXiv 2508.03501): Modified DAPO for long-horizon SWE. Doubles baseline performance.
- **Self-Play SWE-RL / SSR** (Meta, Dec 2025, arXiv 2512.18552): Self-play bug injection + repair. No human labels needed.
- **BacktrackAgent** (EMNLP 2025, arXiv 2505.20660): Explicit error detection and backtracking via SFT+RL. Verifier/Judger/Reflector architecture.
- **CodePlan** (Microsoft, FSE 2024): Repository-level coding as planning. Incremental dependency analysis.
- **MapCoder** (ACL 2024): Multi-agent plan-code-debug. 93.9% HumanEval.
- **Blueprint2Code** (Frontiers AI, 2025): 4-stage pipeline with re-planning. Works on small models.
- **SWE-agent** (NeurIPS 2024): Agent-Computer Interface design for navigation.

**P1 - Important References**:
- **TDFlow** (2025, arXiv 2510.23761): Agentic TDD workflow. 88.8% SWE-bench Lite with human tests.
- **SWE-Playground** (Dec 2025, arXiv 2512.12216): Synthetic project generation. 704 trajectories achieve competitive results.
- **SWE-smith** (NeurIPS 2025 D&B): Scaling synthetic task generation to 50K instances from 128 repos.
- **R2EGym** (COLM 2025): 8.1K problem gym environment for SWE agents.
- **SWE-EVO** (Jan 2026, arXiv 2512.18470): Multi-file, multi-step benchmark. GPT-5 gets 21%.
- **gskill / GEPA** (Feb 2026): Auto-generated repo-specific skill files. 24% -> 93% resolve rate. 47% faster.
- **Observation Masking** (2025, arXiv 2508.21433): Simple masking matches LLM summarization at lower cost.
- **CodePlan** (Tsinghua, ICLR 2025): Code-form plans as scalable planning representation.
- **SWE-rebench** (2025): 21K+ interactive Python SWE tasks for RL.

**P2 - Valuable References**:
- **AIDev-pop** (Feb 2026, arXiv 2602.09185): 33,596 agentic PRs. 97.2% F1-score agent fingerprinting.
- **AI Coding Agent PR Study** (Feb 2026, arXiv 2602.17084): PR description characteristics and human review responses.
- **OWASP Top 10 for LLMs** (2025): Security framework including Excessive Agency risk.
- **Anthropic 2026 Agentic Coding Trends Report**: 8 trends reshaping software engineering.

### Systems (with URLs)

- SWE-agent: https://github.com/SWE-agent/SWE-agent
- OpenHands: https://github.com/OpenHands/OpenHands
- Aider: https://aider.chat/ (git-native coding, edit formats)
- SERA: https://github.com/allenai/SERA / https://huggingface.co/allenai/SERA-32B
- DeepSWE: https://github.com/agentica-project/rllm / https://huggingface.co/agentica-org/DeepSWE-Preview
- R2EGym: https://github.com/R2E-Gym/R2E-Gym
- SWE-smith: https://github.com/SWE-bench/SWE-smith
- SWE-Playground: https://github.com/neulab/SWE-Playground
- Hybrid-Gym: https://arxiv.org/abs/2602.16819
- gskill: https://github.com/itsmostafa/gskill
- GEPA: https://github.com/gepa-ai/gepa
- Blueprint2Code: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1660912/full
- BacktrackAgent: https://aclanthology.org/2025.emnlp-main.212.pdf
- TDFlow: https://arxiv.org/abs/2510.23761

### Datasets (with links)

- R2EGym: 8.1K problems across 13 repos (https://huggingface.co/R2E-Gym)
- SWE-smith: 50K instances from 128 repos (https://github.com/SWE-bench/SWE-smith)
- SWE-Playground: Synthetic projects with diverse tasks (https://github.com/neulab/SWE-Playground)
- SWE-rebench: 21K+ interactive Python SWE tasks (https://openreview.net/forum?id=nMpJoVmRy1)
- SERA training data: Open, model-agnostic format with verification thresholds
- AIDev-pop: 33,596 agentic PRs from 5 coding agents (https://arxiv.org/abs/2602.09185)
- DeepSWE data: 4,500 real-world SWE tasks (https://huggingface.co/agentica-org/DeepSWE-Preview)

### Context Engineering References

- Anthropic: "Effective context engineering for AI agents" (https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- JetBrains: "Smarter Context Management for LLM-Powered Agents" (Dec 2025) (https://blog.jetbrains.com/research/2025/12/efficient-context-management/)
- Observation Masking paper: Simple masking matches LLM summarization at lower cost (https://arxiv.org/abs/2508.21433)

---

## Budget Estimate

### Training Compute (assuming single A100/H100 80GB)

| Phase | GPU Days | Notes |
|-------|----------|-------|
| P0: Navigation + Planning SFT | 5-10 | Additional SFT data mixed with existing |
| P0: Error Recovery GRPO | 10-15 | Requires model inference for trajectory generation |
| P0: Error Recovery SSR self-play | 5-10 | Self-play bug injection/repair (optional, additive) |
| P1: Test Gen + Git + Refactoring SFT | 5-10 | Mostly SFT, some IPO |
| P1: Context Management | 3-5 | Primarily GRPO reward shaping |
| P2: All three areas | 3-5 | SFT only, small datasets |
| **Total** | **31-55 GPU days** | Comparable to SERA (40 GPU days for full training) |

### Data Generation Compute

| Phase | Estimated Cost | Notes |
|-------|---------------|-------|
| Hybrid-Gym synthetic tasks | $5-20 | Cheapest: 0.07 cents per example, need ~5K-10K |
| SWE-Playground synthetic tasks | $20-50 | For test generation tasks |
| SWE-smith task generation | $50-100 | Open-source, local compute |
| Claude API for gold trajectories | $200-500 | 5K trajectories at ~$0.05 each |
| SERA SVG pipeline | $50-100 | Open-source, local compute |
| AIDev-pop (git/docs data) | $0 (free) | Existing dataset, just format conversion |
| Self-play trajectory collection | Included in GPU days | Part of GRPO/SSR training |
| **Total data gen** | **$325-770** | |

---

## Cross-Cutting Considerations

### Harmony Format Compatibility

All 10 capabilities fit naturally into the existing Harmony format. The key tokens are:
- `<|thinking|>`: Plans, verification, judgment, reflection, context summaries
- `<|tool_call|>`: Navigation tools, git tools, test tools, all new tools
- `<|tool_result|>`: Tool outputs (apply observation masking to older results)
- `<|assistant|>`: Final responses, status updates, clarifying questions

No changes to `encode_harmony_messages()` or `validate_harmony_format()` are needed. The new formatters are convenience wrappers for specific trajectory types.

### GRPO Curriculum Integration

The existing GRPO curriculum (4096 -> 8192 -> 16384 -> 32768 tokens) naturally accommodates capability progression:
- 4096: Single-file fixes with simple navigation
- 8192: Multi-file with navigation + basic planning
- 16384: Full module with TDD + error recovery
- 32768: Large repo exploration + git workflow + context management

### Evaluator Dispatch Extension

The existing `pipeline_lib/evaluator_dispatch.py` registry pattern supports adding new reward signals. Each new capability's rewards can be computed by a dedicated evaluator class registered for the appropriate language.

### Training Order Recommendation

Within each priority tier, the recommended training order is:

**P0** (sequential, each builds on previous):
1. Navigation (foundation for all other skills)
2. Planning (requires navigation to plan effectively)
3. Error Recovery (requires planning to know what to backtrack to)

**P1** (can be parallelized):
- Git + Documentation (data is free/cheap, low risk)
- Test Generation (independent of other P1 skills)
- Context Management (can be trained via reward shaping alongside other skills)
- Refactoring (independent)

**P2** (can be done in any order):
- Security, Build Systems, Documentation refinement

---

## Conclusion

The three P0 capabilities (Navigation, Planning, Error Recovery) represent the highest-leverage improvements. Evidence from DeepSWE and SERA shows these capabilities can emerge from RL training on real repo environments, but explicit SFT data accelerates learning and reduces compute requirements.

**Key new insight from Feb 2026 research**: Hybrid-Gym demonstrates that synthetic auxiliary tasks teaching transferable skills (repo-exploration, reasoning, tool use) are more cost-effective than task-specific training data. Combined with SWE-Playground for test generation and Self-Play SWE-RL for error recovery, we have a clear path to training all 10 capabilities at a fraction of the cost of pure RL approaches.

The existing pipeline architecture (Harmony format, evaluator dispatch, GRPO rewards, IPO preferences) is well-suited to absorb all 10 capabilities. The primary investment is in **training data generation** -- the infrastructure for training itself requires only incremental updates (new reward signals, new formatters, new tools).

Recommended next step: Begin Phase 6a with Hybrid-Gym integration for navigation task generation, as this is the cheapest entry point ($5-20 in data generation cost) with the highest proven impact (25.4% SWE-bench improvement).
