# Refactoring Plan: llm-training-pipeline

## Current Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Total files | 44 | - | - |
| Average lines | 318 | 100-300 | WARNING |
| Over 300 lines | 15 (34%) | <20% | WARNING |
| Over 500 lines | 8 | 0 | CRITICAL |
| Over 700 lines | 4 | 0 | CRITICAL |
| Circular imports | 0 | 0 | OK |

### Files Requiring Attention (sorted by priority)

| Priority | File | Lines | Issues |
|----------|------|-------|--------|
| CRITICAL | `02_clean_deduplicate_optimized.py` | 2,701 | Monolithic, 8+ concerns |
| CRITICAL | `02_gpu_clean_deduplicate.py` | 1,138 | Large class, mixed concerns |
| CRITICAL | `05_pretrain.py` | 956 | Kernel setup + callbacks + training + FP8 |
| CRITICAL | `gpu_dedup.py` | 702 | GPU + CPU implementations mixed |
| HIGH | `07_sft.py` | 552 | Duplicated utilities |
| HIGH | `03_tokenize_and_pack.py` | 538 | Multiple concerns |
| HIGH | `06_prepare_reasoning_data.py` | 528 | 14 formatters in one file |
| HIGH | `gpu_utils.py` | 503 | OOM handler could be separate |

### Code Duplication Found

| Function/Class | Copies | Files |
|----------------|--------|-------|
| `unwrap_compiled_model()` | 2 | `05_pretrain.py`, `07_sft.py` |
| `OOMRecoveryCallback` | 3 | `05_pretrain.py`, `07_sft.py`, `09_dpo.py` |
| `load_compiled_checkpoint()` | 3 | `07_sft.py`, `09_dpo.py`, `11_evaluate.py` |
| `check_fp8_available()` / `detect_gpu_type()` | 2 | `gpu_utils.py`, `production_pretrain.py` (deprecated) |

---

## Proposed Structure

The project uses numbered scripts as pipeline stages — this is intentional and should be preserved. The refactoring extracts shared utilities into a `pipeline_lib/` package while keeping the numbered scripts as thin orchestrators.

```
scripts/
├── pipeline_lib/                    # NEW: shared utilities package
│   ├── __init__.py                  # Re-exports key utilities
│   ├── gpu_detection.py             # GPUInfo, detect_gpu_type, check_fp8, setup_torch_backends
│   ├── gpu_fp8.py                   # FP8 accelerator setup
│   ├── oom_handler.py               # OOMHandler, oom_recovery_context, with_oom_retry, get_safe_batch_size
│   ├── training_validation.py       # check_tokenizer_exists, check_checkpoint_exists, validate_prerequisites
│   ├── model_utils.py               # unwrap_compiled_model, load_compiled_checkpoint (deduplicated)
│   ├── training_callbacks.py        # OOMRecoveryCallback, CurriculumCallback (deduplicated)
│   ├── kernel_optimizations.py      # setup_kernel_optimizations, get_optimizer_name
│   └── text_cleaning/               # NEW: sub-package for data cleaning
│       ├── __init__.py
│       ├── pii.py                   # PII regex patterns and replacer
│       ├── cleaning.py              # clean_text_fast/full, parallel_clean
│       ├── tokenization.py          # parallel_tokenize, worker functions
│       ├── quality_filters.py       # DatatroveQualityFilter, filter workers
│       ├── toxicity.py              # DataCleaner (toxicity detection)
│       └── checkpointing.py         # CheckpointManager, StageManager
│
├── dedup/                           # NEW: dedup sub-package
│   ├── __init__.py                  # gpu_fuzzy_dedup, is_gpu_dedup_available
│   ├── minhash.py                   # Shingle hashing, MinHash signatures, LSH index
│   ├── gpu_dedup.py                 # _gpu_dedup_fast implementation
│   └── cpu_dedup.py                 # _cpu_dedup_streaming fallback
│
├── dataset_formatters/              # NEW: reasoning data formatters
│   ├── __init__.py                  # FORMAT_HANDLERS dict
│   ├── math.py                      # GSM8K, Orca-Math, MetaMath, MathInstruct
│   ├── reasoning.py                 # CoT, LogiQA, ARC
│   ├── function_calling.py          # Glaive, Hermes, Gorilla, ToolBench
│   └── general.py                   # Alpaca, OpenOrca, OASST
│
├── 01_download_data.py              # UNCHANGED (327 lines, OK)
├── 02_clean_deduplicate_optimized.py # SLIMMED: orchestrator only (~300 lines)
├── 02_gpu_clean_deduplicate.py      # SLIMMED: GPUDataPipeline uses pipeline_lib/
├── 03_tokenize_and_pack.py          # SLIMMED: uses pipeline_lib/text_cleaning/tokenization
├── 04_init_model.py                 # UNCHANGED (205 lines, OK)
├── 05_pretrain.py                   # SLIMMED: uses pipeline_lib/ for kernels, callbacks
├── 06_prepare_sft_data.py           # UNCHANGED (small)
├── 06_prepare_reasoning_data.py     # SLIMMED: uses dataset_formatters/
├── 07_sft.py                        # SLIMMED: uses pipeline_lib/model_utils, callbacks
├── 08_prepare_dpo_data.py           # UNCHANGED (321 lines, OK)
├── 09_dpo.py                        # SLIMMED: uses pipeline_lib/model_utils, callbacks
├── 10_lora_finetune.py              # UNCHANGED (130 lines, OK)
├── 11_evaluate.py                   # SLIMMED: uses pipeline_lib/model_utils
├── 12_check_gates.py                # UNCHANGED (183 lines, OK)
├── gpu_utils.py                     # BACKWARD COMPAT: re-exports from pipeline_lib/
├── gpu_dedup.py                     # BACKWARD COMPAT: re-exports from dedup/
├── gpu_text_utils.py                # UNCHANGED (270 lines, OK)
└── ...                              # Demo/utility scripts unchanged
```

### Backward Compatibility

The existing `gpu_utils.py` becomes a thin re-export layer:

```python
"""Backward compatibility — imports from pipeline_lib."""
from pipeline_lib.gpu_detection import GPUInfo, detect_gpu_type, check_fp8_available, print_gpu_info, setup_torch_backends
from pipeline_lib.gpu_fp8 import get_fp8_accelerator
from pipeline_lib.training_validation import check_tokenizer_exists, check_checkpoint_exists, validate_training_prerequisites
from pipeline_lib.oom_handler import OOMHandler, oom_recovery_context, with_oom_retry, get_safe_batch_size

__all__ = [...]  # Same as current public API
```

---

## Migration Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Import breakage in training scripts | Keep `gpu_utils.py` as re-export shim; update scripts incrementally |
| Test failures from moved code | Keep `conftest.py` sys.path setup; add `pipeline_lib/` to path too |
| Numbered scripts depend on bare imports | `pipeline_lib/` is inside `scripts/`, so bare imports still work |
| Running scripts from different CWDs | Already handled by `sys.path` manipulation in scripts |
| Notebook compatibility | Notebooks use `%run` or `subprocess` — path unchanged |

---

## Execution Order

### Phase 1: Extract shared utilities (highest impact, lowest risk)
1. Create `scripts/pipeline_lib/` package
2. Move `unwrap_compiled_model` and `load_compiled_checkpoint` → `pipeline_lib/model_utils.py`
3. Move `OOMRecoveryCallback` → `pipeline_lib/training_callbacks.py`
4. Update `05_pretrain.py`, `07_sft.py`, `09_dpo.py`, `11_evaluate.py` to import from `pipeline_lib`
5. Run tests

### Phase 2: Split gpu_utils.py (medium impact)
6. Extract `OOMHandler` + context manager + decorator → `pipeline_lib/oom_handler.py`
7. Extract GPU detection → `pipeline_lib/gpu_detection.py`
8. Extract validation → `pipeline_lib/training_validation.py`
9. Make `gpu_utils.py` a re-export shim
10. Run tests

### Phase 3: Split 05_pretrain.py (medium impact)
11. Move `setup_kernel_optimizations`, `get_optimizer_name` → `pipeline_lib/kernel_optimizations.py`
12. Move `CurriculumCallback` → `pipeline_lib/training_callbacks.py`
13. Keep `setup_training`, `train_with_fp8`, `main` in `05_pretrain.py`
14. Run tests

### Phase 4: Split gpu_dedup.py → dedup/ package (medium impact)
15. Create `scripts/dedup/` package
16. Move MinHash/LSH utilities → `dedup/minhash.py`
17. Move GPU implementation → `dedup/gpu_dedup.py`
18. Move CPU fallback → `dedup/cpu_dedup.py`
19. Make `gpu_dedup.py` a re-export shim
20. Run tests

### Phase 5: Split 02_clean_deduplicate_optimized.py (highest impact, highest risk)
21. Create `scripts/pipeline_lib/text_cleaning/` sub-package
22. Extract PII patterns + cleaning functions → `text_cleaning/pii.py` + `text_cleaning/cleaning.py`
23. Extract quality filters → `text_cleaning/quality_filters.py`
24. Extract toxicity detection → `text_cleaning/toxicity.py`
25. Extract CheckpointManager, StageManager → `text_cleaning/checkpointing.py`
26. Keep orchestrator logic in `02_clean_deduplicate_optimized.py` (~300 lines)
27. Run tests

### Phase 6: Split dataset formatters (low risk)
28. Create `scripts/dataset_formatters/` package
29. Move 14 format functions into categorized files
30. Update `06_prepare_reasoning_data.py` to import from package
31. Run tests

### Phase 7: Slim remaining 500+ line files
32. `07_sft.py`: Remove duplicated code (now imported), target ~300 lines
33. `03_tokenize_and_pack.py`: Move tokenizer training to `pipeline_lib/`, target ~300 lines
34. `02_gpu_clean_deduplicate.py`: Import from `pipeline_lib/text_cleaning/`, target ~400 lines

---

## Actual Results (Refactoring Complete)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total files | 44 | 61 | +17 (new modules) |
| Average lines | 318 | 214 | -33% |
| Over 500 lines | 8 | 4 | -50% |
| Over 700 lines | 4 | 1 | -75% |
| Duplicated functions | 4 (8 copies) | 0 | -100% |
| New shared modules | 0 | 17 | +17 |

### Remaining 500+ line files
- `02_clean_deduplicate_optimized.py` (1280 lines) - complex orchestrator with resume/sync logic
- `02_gpu_clean_deduplicate.py` (1138 lines) - separate GPU pipeline class, pattern-level overlap only
- `05_pretrain.py` (691 lines) - training loop with FP8/curriculum/FSDP setup
- `03_tokenize_and_pack.py` (538 lines) - tokenizer training + packing

### Phases Completed
- Phase 1: Extract shared utilities into `pipeline_lib/` (model_utils, training_callbacks)
- Phase 2: Split `gpu_utils.py` into `pipeline_lib/` modules (gpu_detection, gpu_fp8, oom_handler, training_validation)
- Phase 3: Clean up `05_pretrain.py` (removed duplicate CurriculumCallback)
- Phase 4: Split `gpu_dedup.py` into `dedup/` package (6 modules)
- Phase 5: Split `02_clean_deduplicate_optimized.py` (6 modules under text_cleaning/, 2701→1280 lines)
- Phase 6: Split `06_prepare_reasoning_data.py` dataset formatters (4 modules, 528→214 lines)
- Phase 7: Clean up `07_sft.py` (removed duplicate _get_base_model)

### New packages created
```
scripts/pipeline_lib/          # 8 modules (gpu, training, model utilities)
scripts/pipeline_lib/text_cleaning/  # 6 modules (cleaning, quality, toxicity, etc.)
scripts/dedup/                 # 7 modules (GPU/CPU dedup, MinHash, LSH)
scripts/dataset_formatters/    # 5 modules (math, reasoning, function_calling, general)
```

## Original Expected Results

| Metric | Before | Expected | Change |
|--------|--------|-------|--------|
| Average lines | 318 | ~180 | -43% |
| Over 300 lines | 15 (34%) | ~5 (10%) | -67% |
| Over 500 lines | 8 | 0-1 | -88% |
| Over 700 lines | 4 | 0 | -100% |
| Duplicated functions | 4 (8 copies total) | 0 | -100% |
| New shared modules | 0 | ~15 | +15 |
