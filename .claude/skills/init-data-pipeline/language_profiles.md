# Language Profiles for SOTA Data Pipeline

Reference file mapping each supported language to its evaluation signals, toolchain, and Harmony format usage. The metric keys here match the corresponding `configs/{lang}_eval.yaml` files exactly.

---

## Rust

**Eval config**: `configs/rust_eval.yaml`

| Metric Key | Description |
|-----------|-------------|
| `cargo_check_pass_rate` | Compilation success rate |
| `cargo_test_pass_rate` | Test suite pass rate |
| `clippy_clean_rate` | Lint-clean rate |
| `iterations_to_green_median` | Median debug iterations to passing |
| `diff_size_median` | Median patch size (lines) |
| `tool_call_format_accuracy` | Tool call JSON format validity |
| `hallucinated_api_rate` | Rate of non-existent API references |

**Regression checks**: `humaneval_python`, `mmlu_subset`

**Toolchain**: `cargo`, `cargo-mutants`, `clippy`

**Mutation runner**: `pipeline_lib/cargo_mutants_runner.py`
**Evaluator**: `pipeline_lib/rust_evaluators.py`

---

## Python

**Eval config**: `configs/python_eval.yaml`

| Metric Key | Description |
|-----------|-------------|
| `syntax_check_pass_rate` | Syntax validity rate |
| `pytest_pass_rate` | Test suite pass rate |
| `mypy_clean_rate` | Type-check clean rate |
| `ruff_clean_rate` | Lint-clean rate |
| `iterations_to_green_median` | Median debug iterations to passing |
| `diff_size_median` | Median patch size (lines) |
| `tool_call_format_accuracy` | Tool call JSON format validity |
| `hallucinated_api_rate` | Rate of non-existent API references |

**Regression checks**: `humaneval_python`, `mmlu_subset`

**Toolchain**: `python`, `pytest`, `mypy`, `ruff`

**Mutation runner**: `pipeline_lib/mutmut_runner.py`
**Evaluator**: `pipeline_lib/python_evaluators.py`

---

## TypeScript

**Eval config**: `configs/typescript_eval.yaml`

| Metric Key | Description |
|-----------|-------------|
| `tsc_pass_rate` | TypeScript compilation success rate |
| `jest_pass_rate` | Test suite pass rate |
| `eslint_clean_rate` | Lint-clean rate |
| `iterations_to_green_median` | Median debug iterations to passing |
| `diff_size_median` | Median patch size (lines) |
| `tool_call_format_accuracy` | Tool call JSON format validity |
| `hallucinated_api_rate` | Rate of non-existent API references |

**Regression checks**: `humaneval_python`, `mmlu_subset`

**Toolchain**: `node`, `npx`, `tsc`, `jest`, `eslint`, StrykerJS

**Mutation runner**: `pipeline_lib/stryker_runner.py`
**Evaluator**: `pipeline_lib/typescript_evaluators.py`

---

## Go

**Eval config**: `configs/go_eval.yaml`

| Metric Key | Description |
|-----------|-------------|
| `go_build_pass_rate` | Compilation success rate |
| `go_test_pass_rate` | Test suite pass rate |
| `go_vet_clean_rate` | Vet clean rate |
| `golangci_lint_clean_rate` | Lint-clean rate |
| `iterations_to_green_median` | Median debug iterations to passing |
| `diff_size_median` | Median patch size (lines) |
| `tool_call_format_accuracy` | Tool call JSON format validity |
| `hallucinated_api_rate` | Rate of non-existent API references |

**Regression checks**: `humaneval_python`, `mmlu_subset`

**Toolchain**: `go`, `go-mutesting`, `golangci-lint` (optional)

**Mutation runner**: `pipeline_lib/go_mutesting_runner.py`
**Evaluator**: `pipeline_lib/go_evaluators.py`

---

## Shared Metrics (all languages)

These metrics appear in every language eval config:

| Metric Key | Category |
|-----------|----------|
| `iterations_to_green_median` | Procedural efficiency |
| `diff_size_median` | Procedural efficiency |
| `tool_call_format_accuracy` | Procedural correctness |
| `hallucinated_api_rate` | Knowledge accuracy |
| `humaneval_python` | Regression check |
| `mmlu_subset` | Regression check |

## Harmony Format Handlers

Available format handlers in `dataset_formatters/harmony.py`:

| Key | Function | Use Case |
|-----|----------|----------|
| `harmony_code` | `format_harmony_code()` | Code snippets, knowledge examples |
| `harmony_completion` | `format_harmony_completion()` | Code completion tasks |
| `harmony_agent` | `format_harmony_agent()` | Multi-turn agent traces with tool calls |
| `harmony_preference` | `format_harmony_preference()` | Preference pairs for DPO/IPO |
| `harmony_task` | `format_harmony_task()` | Single-turn task completions |
| `harmony_debug` | `format_harmony_debug()` | Debugging traces |
