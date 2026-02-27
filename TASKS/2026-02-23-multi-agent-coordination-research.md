# Multi-Agent Communication and Coordination Research for GPT-OSS 20B

**Date**: 2026-02-23
**Status**: Research Complete / Pending Implementation
**Goal**: Teach GPT-OSS 20B (MoE, ~3.6B active params) multi-agent coordination capabilities using Harmony format training data

---

## Table of Contents

1. [Multi-Agent Communication Protocols](#1-multi-agent-communication-protocols)
2. [Agent Role Specialization](#2-agent-role-specialization)
3. [Task Decomposition and Delegation](#3-task-decomposition-and-delegation)
4. [Coordination Mechanisms](#4-coordination-mechanisms)
5. [Self-Delegation](#5-self-delegation)
6. [Training Data for Multi-Agent](#6-training-data-for-multi-agent)
7. [Evaluation](#7-evaluation)
8. [Harmony Format Mapping](#8-harmony-format-mapping)
9. [Implementation Plan](#9-implementation-plan)

---

## 1. Multi-Agent Communication Protocols

### Framework Survey

| Framework | Communication Pattern | Message Structure | Agent Discovery | Conflict Resolution |
|-----------|----------------------|-------------------|-----------------|-------------------|
| **AutoGen v0.4** | Async message passing, pub/sub topics | Typed dataclasses (TextMessage, etc.) via JSON-RPC | Runtime agent registration | Orchestrator-mediated |
| **CrewAI** | Sequential/parallel task delegation | Role-content dicts with task context | Crew definition at init | Process-based (sequential/hierarchical) |
| **MetaGPT** | SOP-driven waterfall | Structured role messages with schema outputs | Role registry with SOPs | Standardized Operating Procedures |
| **CAMEL** | Role-play dialogue pairs | Instructor-assistant message tuples | Prompt-defined personalities | Inception prompting |
| **ChatDev** | Chat chain (phase-gated dual-agent) | Instructor-assistant JSON messages | Phase-role assignment | Communicative dehallucination |
| **Google A2A** | JSON-RPC 2.0 over HTTPS | Agent Cards + Task lifecycle | Agent Card discovery | Task state machine |
| **Anthropic MCP** | JSON-RPC 2.0 (stdio/SSE) | Tool/Resource/Prompt primitives | Server capability negotiation | Host-mediated |

### Key Protocols Deep Dive

**AutoGen v0.4 (Microsoft, Jan 2025 redesign)**:
- Rebuilt around async message passing and event-driven patterns
- Two communication types: Direct Messaging (point-to-point) and Broadcast (topic-based pub/sub)
- Messages are serializable dataclass objects (TextMessage, ImageMessage, StructuredMessage)
- StructuredMessageFactory creates typed messages from Pydantic models or JSON schemas
- Follows OpenAI ChatCompletion schema for backward compatibility

**ChatDev (OpenBMB)**:
- Chat chain segments development into Design -> Coding -> Testing phases
- Each phase uses dual-agent instructor-assistant pattern
- Communicative dehallucination: assistant proactively seeks clarification before responding
- Short-term memory (within phase) + long-term memory (cross-phase solution transfer)
- Role pairs: CEO+CTO (design), Programmer+Reviewer (coding), Programmer+Tester (testing)

**MetaGPT (ICLR 2024 Oral)**:
- Mimics software company with SOPs defining inter-role communication
- Roles: Product Manager, Architect, Engineer, QA, Project Manager
- Each role produces structured artifacts (PRD, system design, code, test plans)
- Artifacts serve as communication medium (not just chat messages)

**Google A2A Protocol (April 2025)**:
- Open standard under Linux Foundation, 50+ enterprise partners
- Agent Cards (JSON) describe capabilities, authentication, connection info
- Task lifecycle: submitted -> working -> input-needed -> completed/failed
- Supports streaming, push notifications, multimodal content
- gRPC and REST bindings

**Anthropic MCP (Nov 2024, donated to AAIF Dec 2025)**:
- Host-Client-Server architecture
- JSON-RPC 2.0 transport (stdio for local, SSE for remote)
- Three primitives: Tools (callable functions), Resources (data), Prompts (templates)
- Adopted by OpenAI, Google DeepMind, and major IDEs

### Analysis for GPT-OSS 20B

The most relevant patterns for our model are:
1. **ChatDev's dual-agent instructor-assistant** -- directly maps to Harmony user/assistant turns
2. **MetaGPT's artifact-based communication** -- agents communicate via structured outputs, not just chat
3. **ReDel's delegation-as-tool-call** -- delegation expressed as a special tool call (fits our existing tool_call tokens)
4. **A2A's Agent Cards** -- capability advertisement as structured metadata in developer prompts

---

## 2. Agent Role Specialization

### Key Papers and Decompositions

**AgentCoder (arXiv:2312.13010)** -- Minimal effective decomposition:
- **Programmer Agent**: Code generation and iterative refinement
- **Test Designer Agent**: Independent test case generation (89.6% accuracy on HumanEval with GPT-4)
- **Test Executor Agent**: Runs code against tests, provides structured feedback
- Key insight: Only 3 agents needed. More agents can "overcrowd" for marginal gains.

**MetaGPT (ICLR 2024)** -- Full software company:
- Product Manager -> Architect -> Engineer -> QA -> Project Manager
- Each role produces specific artifacts (PRD, API spec, code, tests, reports)
- Structured output schemas enforce consistency

**ChatDev (ACL 2024)** -- Phased dual-agent:
- Design: CEO + CTO
- Coding: Programmer + Reviewer
- Testing: Programmer + Tester
- Waterfall with chat chain gating between phases

**"Code in Harmony" survey (2025)** -- Key finding:
- PyCapsule achieves strong results with just 2 agents (generator + executor)
- CodeSIM achieved 94.5% HumanEval Pass@1 with ChatGPT-4
- Adding agents beyond 3-4 shows diminishing returns for code generation

**MALT (ICLR 2025)** -- Three-role specialization for reasoning:
- Generator: produces initial answer
- Verifier: critiques the answer
- Refiner: integrates feedback into final output
- Each role trained separately with value iteration reward propagation
- 15.66% improvement on MATH, 7.42% on GSM8K

### Recommended Role Decomposition for GPT-OSS 20B

Based on evidence, a **3-4 role system** is optimal:

| Role | Responsibility | Harmony Channel | Output Artifacts |
|------|---------------|-----------------|------------------|
| **Planner** | Task analysis, decomposition, subtask ordering | thinking + assistant | Task plan, file list, dependency graph |
| **Coder** | Code generation, patching, refactoring | thinking + tool_call + assistant | Patches, new files, cargo commands |
| **Reviewer** | Code review, correctness checking, style | thinking + assistant | Review comments, approval/rejection |
| **Tester** | Test generation, execution, failure analysis | thinking + tool_call + assistant | Test code, execution results, coverage |

The **Planner+Coder+Tester** triple (AgentCoder pattern) is the minimum viable set. Adding Reviewer provides measurable quality improvement at modest training data cost.

---

## 3. Task Decomposition and Delegation

### Approaches from Leading Systems

**OpenHands / OpenDevin (ICLR 2025)**:
- Event-sourced state model with deterministic replay
- `AgentDelegateAction` enables agent-to-sub-agent delegation
- Sandboxed workspace with filesystem, terminal, web interface
- Event log records all commands, edits, results as persistent context
- SDK achieves 72% on SWE-Bench Verified with Claude Sonnet 4.5 + extended thinking

**SWE-agent**:
- Agent-Computer Interface (ACI) with specialized file editing commands
- Linear workflow: read issue -> explore codebase -> edit files -> test -> submit
- No explicit multi-agent decomposition; single agent with rich tool set
- Key insight: good tool design > multi-agent complexity for many tasks

**ADaPT (Allen AI)**:
- Recursive decomposition that adapts to both task complexity and LLM capability
- Decomposes only when needed (vs. always decomposing)
- Sub-tasks inherit context from parent with selective information passing

**SERA (Allen AI, Jan 2026)**:
- Soft Verified Generation (SVG): Teacher generates trajectories, Student learns from them
- 26x cheaper than RL, 57x cheaper than prior synthetic methods
- SERA-32B reaches 49.5% on SWE-bench Verified (32K context), 54.2% at 64K
- Only 8,000 specialized trajectories needed for codebase specialization
- Key relevance: demonstrates efficient trajectory-based training for coding agents

### Task Decomposition Taxonomy

For coding tasks, decomposition follows this hierarchy:

```
Level 0: Full project/feature request
  Level 1: Component-level subtasks (e.g., "implement parser", "add API endpoint")
    Level 2: File-level operations (e.g., "modify src/parser.rs", "create tests/test_parser.rs")
      Level 3: Atomic actions (read file, apply patch, run test)
```

Training data should cover all levels. Current pipeline (15_generate_trajectories.py) only covers Level 2-3. Multi-agent training requires Level 0-1 decomposition data.

---

## 4. Coordination Mechanisms

### Patterns from Literature

**Blackboard Pattern**:
- Shared semantic state that all agents can read/write
- LbMAS (2025): Public space (all agents) + private spaces (debate/verification)
- PC-Agent (2025): Manager maintains global task state, workers handle sub-tasks
- File-level locking prevents concurrent edit conflicts
- Best for: complex tasks requiring shared context across specialists

**Message Passing**:
- Direct agent-to-agent messages (AutoGen Direct Messaging)
- Topic-based pub/sub (AutoGen Broadcast)
- Sequential chain (ChatDev chat chain)
- Best for: well-defined workflows with clear handoff points

**Hierarchical (Tree)**:
- Manager/orchestrator delegates to specialists
- MultiAgentBench found graph topology outperforms star/chain/tree in research scenarios
- Cognitive planning improves milestone achievement by 3%
- Best for: complex multi-file tasks requiring coordination

**Consensus/Voting**:
- Multiple agents generate solutions, majority vote or judge selects
- Used in debate frameworks and self-consistency approaches
- Best for: high-stakes decisions where single-agent reliability is insufficient

### Conflict Resolution for File Editing

Critical for coding agents. Approaches ranked by practicality:

1. **Serialized access** (simplest): Orchestrator assigns file ownership, one agent edits at a time
2. **Lock-based**: Agent acquires file lock before editing, releases after
3. **Merge-based**: Agents edit independently, orchestrator merges (like git merge)
4. **Semantic partitioning**: Different agents own different code regions (functions, modules)

For training: serialized access is easiest to generate data for. The model learns "I should not edit files assigned to another agent."

---

## 5. Self-Delegation

### Key Systems

**ReDel (EMNLP 2024)**:
- Delegation implemented as a special tool call
- `DelegateOne`: Blocks parent until child returns (supports parallel children)
- `DelegateWait`: Non-blocking, separate retrieval function
- Parent spawns child with instruction string, child returns result string
- Configurable depth limit prevents infinite recursion
- Results: GPT-4o achieved 0.687 on FanOutQA, 67.49% TravelPlanner, 0.643 WebArena

**Recursive Language Models (RLMs, 2025)**:
- LLMs decompose input recursively, interact with sub-instances
- Strong performance at 10M+ token scale (double-digit gains over baselines)
- Self-delegation via Python REPL environment

**Voyager (NeurIPS 2023)**:
- Skill library: agent stores learned skills as executable code
- Auto-curriculum: self-generates exploration goals
- Iterative prompting with environment feedback
- Not delegation per se, but self-organization through skill accumulation

**LATS - Language Agent Tree Search (ICML 2024)**:
- Monte Carlo Tree Search for LLM agents
- LM-powered value functions + self-reflections
- 92.7% pass@1 on HumanEval with GPT-4
- Exploration-exploitation for multi-step reasoning/acting

### Self-Delegation as Tool Call (Recommended Approach)

The cleanest implementation for GPT-OSS 20B: delegation is a tool call.

```json
{
  "id": "call_delegate_001",
  "name": "delegate_subtask",
  "arguments": {
    "role": "tester",
    "instructions": "Write unit tests for the parser module in src/parser.rs",
    "context_files": ["src/parser.rs", "src/lib.rs"],
    "return_artifacts": ["test code", "execution results"]
  }
}
```

The model learns when to delegate vs. handle directly. This maps perfectly to existing Harmony tool_call/tool_result tokens.

---

## 6. Training Data for Multi-Agent

### Data Generation Strategies

**Strategy 1: Synthetic Multi-Role Conversations (Primary)**

Use a strong teacher model (GPT-4, Claude) to generate multi-agent dialogues:

1. Give teacher a coding task + role assignment
2. Teacher generates conversation between roles
3. Format as Harmony with role metadata in developer prompt
4. Filter for quality (task completion, code correctness)

Estimated volume: 5,000-10,000 multi-agent trajectories needed based on SERA's finding that 8,000 trajectories suffice for specialization.

**Strategy 2: Trajectory Decomposition from Existing Data**

Take existing single-agent trajectories (from 15_generate_trajectories.py) and decompose:
- Split long trajectories into planner+coder+tester segments
- Add role transitions between segments
- Inject review/feedback turns

This is cheaper than full synthetic generation and reuses existing pipeline output.

**Strategy 3: Preference Pairs for Delegation Quality**

Generate preference data comparing:
- **Chosen**: Agent correctly delegates complex subtask, receives and integrates result
- **Rejected**: Agent attempts everything solo, produces worse code / takes more iterations

Use this for IPO/DPO training to teach the model when delegation improves outcomes.

**Strategy 4: MALT-style Value Iteration**

Based on MALT (ICLR 2025):
1. Sample from Generator, Verifier, Refiner repeatedly to build search tree
2. Grade final outputs against ground truth (cargo test pass/fail)
3. Propagate rewards back through the tree via value iteration
4. Each role learns from both successful and failed trajectories

This is the most sophisticated approach but requires significant compute.

### Concrete Training Data Formats

**Multi-Agent Trajectory Example** (extends format_harmony_agent):
```python
{
    "task": "Implement a caching layer for the HTTP client",
    "roles": ["planner", "coder", "reviewer", "tester"],
    "trajectory": [
        # Planner phase
        {
            "role": "planner",
            "thinking": "This requires: 1) cache data structure, 2) cache invalidation, 3) integration with client...",
            "response": "## Plan\n1. Create CacheEntry struct...\n2. Implement LRU eviction...",
        },
        # Coder phase
        {
            "role": "coder",
            "thinking": "Following the plan, I'll start with the cache struct...",
            "tool_calls": [{"name": "apply_patch", "arguments": {...}}],
            "tool_results": [{"output": "Patch applied."}],
        },
        # Reviewer phase
        {
            "role": "reviewer",
            "thinking": "Let me check the implementation against the plan...",
            "response": "Issue: Missing thread safety. Cache needs Arc<Mutex<>> wrapper.",
        },
        # Coder responds to review
        {
            "role": "coder",
            "thinking": "Good catch. Adding thread safety...",
            "tool_calls": [{"name": "apply_patch", "arguments": {...}}],
            "tool_results": [{"output": "Patch applied."}],
        },
        # Tester phase
        {
            "role": "tester",
            "thinking": "I need to test thread safety, eviction, and basic cache operations...",
            "tool_calls": [
                {"name": "apply_patch", "arguments": {"diff": "...test code..."}},
                {"name": "run_tests", "arguments": {"cmd": "cargo test"}}
            ],
            "tool_results": [
                {"output": "Patch applied."},
                {"output": "test result: ok. 12 passed; 0 failed"}
            ],
            "response": "All tests pass including concurrent access tests.",
        },
    ]
}
```

**Delegation Trajectory Example** (self-delegation):
```python
{
    "task": "Fix 3 failing tests in the parser module",
    "trajectory": [
        {
            "thinking": "Three separate test failures. I should fix them systematically. Let me delegate each one.",
            "tool_calls": [
                {
                    "name": "delegate_subtask",
                    "arguments": {
                        "role": "debugger",
                        "instructions": "Fix test_parse_nested_struct in src/parser.rs",
                        "context_files": ["src/parser.rs"]
                    }
                },
            ],
            "tool_results": [
                {
                    "output": "Fixed: The nested struct parser was not handling lifetime annotations. Applied patch to line 142. cargo test test_parse_nested_struct now passes."
                }
            ],
        },
        # ... more delegation calls for other tests ...
        {
            "thinking": "All subtasks complete. Let me run the full test suite to verify.",
            "tool_calls": [{"name": "run_tests", "arguments": {"cmd": "cargo test"}}],
            "tool_results": [{"output": "test result: ok. 48 passed; 0 failed"}],
            "response": "All 3 failing tests are now fixed. Full suite passes."
        }
    ]
}
```

---

## 7. Evaluation

### Benchmarks

| Benchmark | What It Measures | Multi-Agent Relevance | Difficulty |
|-----------|-----------------|----------------------|------------|
| **SWE-bench Verified** | Real GitHub issue resolution | End-to-end agent capability | High (SOTA: ~75%) |
| **SWE-bench Multilingual** | Multi-language (Java, TS, Go, Rust, C/C++) | Language-specific agent quality | High |
| **DevBench** | Full SDLC (design, setup, implement, test) | Phase-level collaboration | Very High |
| **MultiAgentBench (MARBLE)** | Collaboration + competition scenarios | Direct multi-agent measurement | Medium |
| **HumanEval/MBPP** | Function-level code generation | Baseline coding ability | Low |
| **CrossCodeEval** | Cross-file code completion | Multi-file awareness | Medium |

### Recommended Evaluation Strategy for GPT-OSS 20B

**Tier 1 -- Existing pipeline metrics (extend current eval_rust_agent.py)**:
- cargo_check_pass_rate, cargo_test_pass_rate, clippy_clean_rate
- New: delegation_accuracy (did the model correctly delegate vs. solo?)
- New: plan_quality (does the plan cover all subtasks?)
- New: review_catch_rate (does reviewer role catch real bugs?)

**Tier 2 -- Multi-agent specific metrics**:
- coordination_efficiency: task completion time with delegation vs. solo
- role_adherence: does each role stay within its responsibilities?
- communication_clarity: are inter-role messages actionable?
- conflict_rate: how often do agents produce conflicting edits?

**Tier 3 -- External benchmarks**:
- SWE-bench Multilingual (Rust subset) -- primary external benchmark
- DevBench -- full pipeline quality
- MultiAgentBench -- if model used in multi-instance deployment

### Promotion Gate Additions

New gates for multi-agent capability:

```yaml
# GRPO -> PRODUCTION (Multi-Agent Extension)
grpo_to_production_multiagent:
  # Existing gates (retain all)
  cargo_check_pass_rate: 0.85
  cargo_test_pass_rate: 0.70

  # New: Delegation quality
  delegation_accuracy: 0.70        # >70% appropriate delegation decisions
  plan_completeness: 0.75          # >75% of subtasks in plan are necessary
  role_adherence_rate: 0.80        # >80% responses stay in assigned role
  review_bug_catch_rate: 0.50      # >50% of injected bugs caught by reviewer role

  # New: Coordination
  multi_file_consistency: 0.80     # >80% multi-file edits are consistent
  no_conflict_rate: 0.95           # >95% of delegated tasks don't conflict
```

---

## 8. Harmony Format Mapping

### Current Harmony Tokens

```
<|system|>          -- System prompt
<|developer|>       -- Developer instructions
<|user|>            -- User message
<|assistant|>       -- Assistant response
<|thinking|>        -- Chain of thought (start)
<|/thinking|>       -- Chain of thought (end)
<|tool_call|>       -- Tool invocation
<|tool_result|>     -- Tool output
<|tool_call_id|>    -- Tool call ID reference
<|name|>            -- Tool name reference
<|endoftext|>       -- End of sequence
```

### Proposed Multi-Agent Extensions

**Option A: Role-in-Developer-Prompt (Minimal, No New Tokens)**

Encode the active role in the developer prompt. No tokenizer changes needed.

```
<|developer|>
You are a Rust coding agent operating in ROLE: planner.
Your team consists of: planner, coder, reviewer, tester.
Current phase: planning.
Shared context: {task description, file list, prior outputs}

When you need another role to act, use the delegate_subtask tool.
When receiving delegated work, complete it and return results.
<|user|>
Implement connection pooling for the database module.
<|thinking|>
This is a complex feature. I should plan the subtasks:
1. Design the pool data structure (coder)
2. Implement acquire/release logic (coder)
3. Add timeout handling (coder)
4. Write thread-safety tests (tester)
5. Review for resource leaks (reviewer)
<|/thinking|>
<|assistant|>
## Task Plan

### Subtask 1: Pool Data Structure
- Files: src/db/pool.rs (new)
- Description: Create ConnectionPool<T> struct with...
...

### Subtask 2: Acquire/Release
...
```

**Option B: Role Token (Requires Tokenizer Extension)**

Add a `<|role|>` token that precedes assistant messages in multi-agent context.

```
<|developer|>
Multi-agent coding session. Roles: planner, coder, reviewer, tester.
<|user|>
Implement connection pooling for the database module.
<|role|>planner
<|thinking|>
This is a complex feature requiring careful decomposition...
<|/thinking|>
<|assistant|>
## Plan: Connection Pooling Implementation
...
<|role|>coder
<|thinking|>
Following the plan, I'll implement the pool struct first...
<|/thinking|>
<|tool_call|>
{"id": "call_001", "name": "apply_patch", "arguments": {...}}
<|tool_result|>
<|tool_call_id|>call_001
Patch applied successfully.
<|role|>reviewer
<|thinking|>
Let me review the pool implementation for thread safety...
<|/thinking|>
<|assistant|>
Review: Missing Drop implementation for cleanup. The pool should...
```

**Option C: Delegation as Tool Call (Recommended, No New Tokens)**

Model existing tool_call/tool_result flow. Delegation is just another tool.

```
<|developer|>
You are a Rust coding agent. You can delegate subtasks using delegate_subtask.
<|user|>
Fix the 5 failing tests in the parser module.
<|thinking|>
5 failures across different test functions. I'll delegate each fix.
<|/thinking|>
<|tool_call|>
{"id": "del_001", "name": "delegate_subtask", "arguments": {"role": "debugger", "task": "Fix test_parse_struct: expected Struct got Enum at line 42", "files": ["src/parser.rs"]}}
<|tool_result|>
<|tool_call_id|>del_001
Fixed: Changed match arm on line 42 from Token::Enum to Token::Struct. test_parse_struct now passes. Patch: -    Token::Enum => {...} +    Token::Struct => {...}
<|tool_call|>
{"id": "del_002", "name": "delegate_subtask", "arguments": {"role": "debugger", "task": "Fix test_parse_lifetime: missing lifetime bound", "files": ["src/parser.rs"]}}
<|tool_result|>
<|tool_call_id|>del_002
Fixed: Added lifetime bound 'a to ParseContext. test_parse_lifetime now passes.
...
<|thinking|>
All delegated fixes are complete. Let me run the full suite.
<|/thinking|>
<|tool_call|>
{"id": "call_final", "name": "run_tests", "arguments": {"cmd": "cargo test"}}
<|tool_result|>
<|tool_call_id|>call_final
test result: ok. 87 passed; 0 failed; 0 ignored
<|assistant|>
All 5 failing tests have been fixed. The full test suite now passes.
```

### Recommendation

**Use Option C (delegation-as-tool-call) as the primary approach, with Option A (role-in-developer-prompt) for role context.**

Rationale:
- No tokenizer changes required (critical for MoE model with existing vocabulary)
- Reuses existing tool_call/tool_result token flow that the model already understands
- Role context in developer prompt is natural extension of existing patterns
- Delegation results look identical to other tool results -- model already knows this pattern
- Can be introduced incrementally alongside existing training data

---

## 9. Implementation Plan

### Phase 1: Multi-Role Trajectories (Weeks 1-3)

**Goal**: Generate training data where the model plays different roles sequentially.

**Changes**:
1. Extend `scripts/dataset_formatters/harmony.py` with `format_harmony_multi_agent()` formatter
2. Extend `scripts/15_generate_trajectories.py` with multi-role trajectory templates
3. Add `delegate_subtask` to the tool vocabulary
4. Generate 3,000-5,000 multi-role trajectories from existing mutations data

**New files**:
- `scripts/dataset_formatters/harmony_multi_agent.py` -- multi-agent specific formatters
- `scripts/20_generate_multi_agent_trajectories.py` -- trajectory generation
- `configs/multi_agent_roles.yaml` -- role definitions and templates

### Phase 2: Delegation Training (Weeks 4-6)

**Goal**: Teach the model to use delegate_subtask tool effectively.

**Changes**:
1. Generate delegation trajectories (single-agent delegates to sub-instances)
2. Create preference pairs: good delegation vs. solo attempt on complex tasks
3. Add delegation-specific evaluation metrics to `eval_rust_agent.py`

**Training mix**:
- 60% standard agent trajectories (existing)
- 25% multi-role trajectories (Phase 1)
- 15% delegation trajectories (Phase 2)

### Phase 3: IPO/GRPO for Coordination Quality (Weeks 7-9)

**Goal**: Use preference learning and RL to improve delegation decisions.

**Changes**:
1. Generate preference pairs comparing delegation vs. solo approaches
2. Extend IPO training (scripts/17_ipo_preference.py) with coordination preferences
3. Extend GRPO rewards (scripts/18_grpo_rl.py) with coordination metrics:
   - Reward for successful delegation that reduces total iterations
   - Penalty for unnecessary delegation on simple tasks
   - Reward for accurate role adherence

### Phase 4: Evaluation and Gates (Weeks 10-11)

**Goal**: Measure multi-agent capabilities and set promotion gates.

**Changes**:
1. Extend eval_rust_agent.py with multi-agent evaluation scenarios
2. Add delegation accuracy, plan quality, and role adherence metrics
3. Create multi-file test tasks that benefit from delegation
4. Add promotion gates to configs/promotion_gates.yaml

### Phase 5: External Benchmark Validation (Week 12)

**Goal**: Validate on external benchmarks.

**Changes**:
1. Run SWE-bench Multilingual (Rust subset) with delegation enabled
2. Evaluate on DevBench design+implement+test pipeline
3. Compare delegation vs. solo on matched tasks

### Resource Estimates

| Phase | Data Volume | Compute (A100 hours) | New Code (LoC) |
|-------|-------------|---------------------|-----------------|
| Phase 1 | 5K trajectories | 4h generation, 8h training | ~800 |
| Phase 2 | 3K delegation + 2K preference | 4h generation, 12h training | ~600 |
| Phase 3 | Use Phase 1-2 data | 24h IPO + 48h GRPO | ~400 |
| Phase 4 | 200 eval tasks | 8h evaluation | ~500 |
| Phase 5 | External benchmarks | 16h evaluation | ~200 |
| **Total** | **~10K samples** | **~124 A100 hours** | **~2,500 LoC** |

---

## Key Papers and References

1. **AgentCoder** (2024): Multi-agent code generation with programmer + test designer + executor. arXiv:2312.13010
2. **MetaGPT** (ICLR 2024 Oral): SOP-driven multi-agent software company. arXiv:2308.00352
3. **ChatDev** (ACL 2024): Chat chain with communicative dehallucination. arXiv:2307.07924
4. **MALT** (ICLR 2025): Multi-agent training with generator + verifier + refiner + value iteration. arXiv:2412.01928
5. **ReDel** (EMNLP 2024): Recursive delegation toolkit with DelegateOne/DelegateWait. arXiv:2408.02248
6. **LATS** (ICML 2024): Monte Carlo Tree Search for LLM agents. arXiv:2310.04406
7. **Voyager** (NeurIPS 2023): Lifelong learning agent with skill library. voyager.minedojo.org
8. **SERA** (Jan 2026): Soft-verified trajectory generation, 26x cheaper than RL. arXiv:2601.20789
9. **OpenHands** (ICLR 2025): Event-sourced agent SDK with AgentDelegateAction. arXiv:2407.16741
10. **MultiAgentBench** (ACL 2025): Collaboration/competition benchmark. arXiv:2503.01935
11. **DevBench** (2024): Full SDLC benchmark. arXiv:2403.08604
12. **Google A2A** (April 2025): Agent-to-agent protocol under Linux Foundation
13. **Anthropic MCP** (Nov 2024): Model Context Protocol for tool integration
14. **Multi-Agent Collaboration Mechanisms Survey** (2025): Taxonomy of structures and strategies. arXiv:2501.06322
15. **Recursive Language Models** (2025): Self-delegation via recursive decomposition. arXiv:2512.24601
16. **LLM-based Multi-Agent Blackboard System** (2025): Blackboard pattern for LLM agents. arXiv:2510.01285

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Model confuses roles in single-instance deployment | High | Medium | Strong role-in-prompt formatting, clear role boundaries in training data |
| Delegation loops (infinite recursion) | Medium | High | Depth limit in training data, max_delegation_depth in prompts |
| Training data quality for multi-role | Medium | High | Use SERA-style soft verification on delegation outputs |
| Token overhead from multi-agent context | Medium | Medium | Keep delegation results concise, use summarization |
| Regression on single-agent tasks | Low | High | Maintain 60% single-agent trajectories in training mix |
| MoE routing confusion from role switching | Low | Medium | Monitor expert activation patterns per role |
