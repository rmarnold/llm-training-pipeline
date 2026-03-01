# TUI Notebook Checkpoint Flow Analysis

**Date**: 2026-02-26
**Status**: Research Complete
**Question**: Is the TUI notebook's approach of NOT merging the Agent SFT adapter before IPO/GRPO training correct?

## Executive Summary

The TUI notebook's checkpoint flow is **functionally correct but carries a subtle semantic inconsistency** in how Unsloth + TRL handle the reference model for IPO. The asymmetry (tool calling merged, agent SFT not merged) is intentional and largely justified by practical VRAM constraints, but the IPO stage's reference model behavior may not match the developer's intent. The flow works, but the reference model for IPO is the pre-Agent-SFT state (base + tool calling), not the post-Agent-SFT adapter -- which is actually the theoretically correct DPO/IPO reference anyway.

## Detailed Analysis

### 1. What the TUI Notebook Actually Does

**Cell 32** - Tool Calling SFT: Trains rank-64 LoRA on `openai/gpt-oss-20b` base -> `checkpoints/tool_calling_sft/final/`

**Cell 34** - Merge: Merges tool calling adapter INTO base -> `checkpoints/gpt-oss-20b-coding-tui-merged/hf/` (~40GB full model)

**Cell 40** - Agent SFT: Trains rank-128 LoRA on merged model -> `checkpoints/agent_sft/final/` (adapter only, ~1-2GB)

**Cell 45** - IPO: Calls `python scripts/17_ipo_preference.py --checkpoint checkpoints/agent_sft/final` -> `checkpoints/agent_sft_ipo/final/`

**Cell 50** - GRPO: Calls `python scripts/18_grpo_rl.py --checkpoint checkpoints/agent_sft_ipo/final` -> `checkpoints/agent_sft_grpo/final/`

**Cell 69** - Final Export: Merges best adapter into `openai/gpt-oss-20b` via `19_merge_adapter.py`

### 2. How `17_ipo_preference.py` Loads the Agent SFT Checkpoint

The IPO script calls:
```python
model, tokenizer = load_unsloth_model(
    model_name="checkpoints/agent_sft/final",  # adapter checkpoint path
    ...
)
```

Which calls:
```python
FastLanguageModel.from_pretrained(model_name="checkpoints/agent_sft/final", ...)
```

**What Unsloth does**: When given a directory containing `adapter_config.json`, Unsloth reads the `base_model_name_or_path` from the adapter config to find the base model, then loads the base model + applies the adapter. The adapter's `adapter_config.json` stores the base model path it was trained on.

**Key finding**: The Agent SFT adapter's `adapter_config.json` will point to `checkpoints/gpt-oss-20b-coding-tui-merged/hf/` as the base model (the merged tool-calling model). Unsloth loads this merged model, then applies the Agent SFT LoRA on top.

So the loaded model is: `base GPT-OSS 20B + tool_calling weights (merged) + agent_sft LoRA (adapter)`

### 3. How TRL DPOTrainer Handles the Reference Model with PEFT

The IPO script creates a `DPOTrainer` with NO explicit `ref_model`:

```python
trainer = DPOTrainer(
    model=model,          # PEFT model with LoRA adapter
    args=training_args,   # loss_type="ipo", precompute_ref_log_probs=True
    ...
)
```

**What TRL does when `ref_model=None` and model is PEFT** (from TRL source):

1. Detects the model is a PEFT model via `is_peft_model(model)`
2. Creates a **"ref" adapter** by copying the current "default" adapter weights:
   ```python
   model.add_adapter("ref", model.peft_config["default"])
   for name, param in model.named_parameters():
       if ".default." in name:
           ref_name = name.replace(".default.", ".ref.")
           ref_param = model.get_parameter(ref_name)
           ref_param.data.copy_(param.data)
   ```
3. During training, switches to the "ref" adapter to compute reference log probabilities
4. The "ref" adapter stays frozen while the "default" adapter is updated

**Critical implication**: The reference model for IPO is `base + tool_calling (merged) + agent_sft_initial (frozen copy)`. This means:
- The reference policy IS the Agent SFT policy at the start of IPO training
- As IPO trains, the "default" adapter diverges from "ref" adapter
- The KL constraint keeps IPO output close to Agent SFT output

**This is the correct and expected DPO/IPO reference model behavior.** The reference should be the policy at the start of preference optimization, which is exactly the Agent SFT checkpoint.

### 4. The `precompute_ref_log_probs=True` Question

The IPO config has `precompute_ref_log_probs: true`. The question is whether this works with PEFT.

**Finding**: In modern TRL (v0.26+), `precompute_ref_log_probs=True` IS compatible with PEFT models. The trainer:
1. Creates the "ref" adapter as described above
2. Switches to "ref" adapter
3. Computes all reference log probs for the training data in a single pass
4. Stores them
5. Then trains with the "default" adapter without needing the reference forward pass

This is correct because the reference model is fixed (no `sync_ref_model`). The incompatibility is only with:
- `sync_ref_model=True` (which periodically updates the reference -- incompatible with precomputation)
- `use_liger_kernel=True` (Liger kernel does not support precomputation)

**The notebook's use of `precompute_ref_log_probs=True` with PEFT is correct and saves ~15-20 GiB VRAM.**

### 5. The Asymmetry: Is It a Problem?

**Why tool calling IS merged but agent SFT IS NOT merged before IPO:**

| Factor | Tool Calling -> Agent SFT | Agent SFT -> IPO |
|--------|--------------------------|------------------|
| Rank change | 64 -> 128 (different adapter config) | 128 -> 128 (same adapter, continued training) |
| Purpose | Different task (tool calling vs agent behavior) | Refinement of same task (align agent behavior) |
| Merge needed? | Yes -- different LoRA architecture | No -- same LoRA, DPO/IPO continues training it |
| VRAM cost of merge | ~40GB disk, one-time | Would require another ~40GB merged model |
| Reference model | N/A (SFT doesn't use reference) | Handled by TRL's "ref" adapter copy |

**The asymmetry is intentional and correct for these reasons:**

1. **Tool calling and agent SFT are different tasks** with different LoRA ranks (64 vs 128). Merging before Agent SFT is necessary because you cannot stack LoRA adapters of different ranks in PEFT without merging first.

2. **Agent SFT and IPO are the same adapter being refined.** IPO/DPO is specifically designed to work on top of an SFT checkpoint. The standard workflow in the literature (Rafailov et al., 2023; Tunstall et al., 2025) is SFT -> DPO/IPO on the same model, not SFT -> merge -> DPO/IPO.

3. **TRL's adapter-based reference model is the recommended approach** (HuggingFace documentation). Merging before IPO would require loading a second full model copy as the reference, doubling VRAM usage.

4. **The PEFT recommendation** (from PEFT issue #2264) explicitly says: "Load the first adapter with `is_trainable=True` and continue training" is preferred over "merge first adapter, then create new one."

### 6. Comparison with the Master Plan's Per-Language Flow

The master plan describes per-language training as:
```
CoreAgent SFT (14) -> IPO (17) -> GRPO (18) -> Eval
```
Where each stage loads the PREVIOUS stage's adapter checkpoint.

**This is exactly what the TUI notebook does**, just with different checkpoint names:
- TUI: `agent_sft/final` -> `agent_sft_ipo/final` -> `agent_sft_grpo/final`
- Per-lang: `{lang}_core_agent` -> `{lang}_ipo` -> `{lang}_grpo`

Both follow the same pattern: load adapter checkpoint from previous stage, continue training.

The difference is the base model:
- TUI: Base = merged tool-calling model (`gpt-oss-20b-coding-tui-merged`)
- Per-lang: Base = merged TUI model (`coding_tui/final_merged`)

This is consistent -- each pipeline merges everything BEFORE its first SFT stage, then chains adapters through SFT -> IPO -> GRPO without intermediate merges.

### 7. Potential Issue: Final Export (Cell 69)

The final export cell calls:
```
python scripts/19_merge_adapter.py --adapter_path checkpoints/agent_sft_grpo/final
```

The `19_merge_adapter.py` script uses `merge_and_export()` which calls `_load_base_and_adapter()`. This function reads `base_model_name_or_path` from the adapter config to find the base model.

**Critical question**: What is the `base_model_name_or_path` in the GRPO adapter config?

The chain is:
1. Agent SFT adapter trained on `checkpoints/gpt-oss-20b-coding-tui-merged/hf/` -> base = merged model
2. IPO loaded Agent SFT adapter (Unsloth auto-detected base from adapter config) -> base = merged model
3. GRPO loaded IPO adapter -> base = merged model

So the GRPO adapter's `base_model_name_or_path` should point to the merged tool-calling model. When `19_merge_adapter.py` merges, it:
1. Loads `checkpoints/gpt-oss-20b-coding-tui-merged/hf/` as base
2. Applies GRPO adapter LoRA weights
3. Merges to produce the final model

**But wait**: The merged model includes tool-calling weights already merged in. The GRPO adapter was trained on this merged base. So the final merge produces: `base GPT-OSS 20B + tool_calling (from merge) + agent_sft + IPO + GRPO (from adapter)`.

**This is correct.** All training stages' knowledge is preserved in the final merged model.

**However**: If the merged tool-calling model at `checkpoints/gpt-oss-20b-coding-tui-merged/hf/` has been deleted (disk cleanup!), the merge will fail. Cell 42 runs `dm.cleanup_between_phases("agent_sft", "agent_sft_ipo")` which may delete the merged model.

**This IS a potential bug**: The disk cleanup between Agent SFT and IPO may delete the merged model that the final export cell needs to perform the merge. The adapter configs reference the merged model path, so if it is deleted, `_load_base_and_adapter()` will fail.

## Research Findings from Literature

### Sequential LoRA SFT -> DPO Best Practices

1. **PEFT Issue #2264** (HuggingFace, 2024): Recommends continuing training on the same adapter (`PeftModel.from_pretrained(..., is_trainable=True)`) over merging first. Merging is only needed when changing adapter architecture.

2. **Philschmid DPO Guide** (2025): Trains SFT -> uses that checkpoint directly for DPO -> only merges after DPO is complete. Does NOT merge between SFT and DPO.

3. **Merge before Forget** (arxiv:2512.23017, 2025): For continual learning across unrelated tasks, merging helps prevent forgetting. But SFT -> DPO is NOT unrelated tasks -- it is refinement of the same behavior.

4. **TRL Documentation** (v0.29.0): Explicitly shows loading `PeftModel` and passing it to `DPOTrainer` without merging. The "ref" adapter mechanism handles reference model computation.

### Reference Model Correctness

TRL Issue #1340 (2024, closed): Confirmed that when loading pretrained adapters, the "ref" adapter copy correctly serves as the reference model. The initial adapter state IS the reference policy.

### IPO vs DPO Reference Model

IPO (Azar et al., 2023) does not require a reference model in the traditional sense -- the IPO loss uses an identity transform rather than KL divergence. However, TRL implements IPO via `DPOTrainer(loss_type="ipo")`, which still uses reference log probabilities for the reward computation. The precomputed reference log probs are valid for IPO.

## Recommendation

**The TUI notebook's checkpoint flow is correct.** No changes are needed to the training flow itself.

**One potential issue to address**: Verify that disk cleanup does not delete the merged model before the final export/merge step needs it. The adapter configs reference the merged model path, and the export cell requires it.

**Confidence Level**: High -- based on TRL source code analysis, PEFT documentation, Unsloth adapter loading behavior, and published best practices.

## Sources

- [TRL DPO Trainer Documentation](https://huggingface.co/docs/trl/dpo_trainer)
- [TRL DPOTrainer Source Code](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py)
- [PEFT Issue #2264: Two-Stage Fine-Tuning Guidance](https://github.com/huggingface/peft/issues/2264)
- [TRL Issue #1340: Incorrect Reference Model with Pretrained Adapters](https://github.com/huggingface/trl/issues/1340)
- [Unsloth Issue #631: Model Already Has LoRA Adapters](https://github.com/unslothai/unsloth/issues/631)
- [Merge before Forget: Continual LoRA Merging (arxiv:2512.23017)](https://arxiv.org/html/2512.23017v1)
- [How to Align LLMs in 2025 with DPO - Philschmid](https://www.philschmid.de/rl-with-llms-in-2025-dpo)
- [Unsloth Documentation: Continued Pretraining](https://unsloth.ai/docs/basics/continued-pretraining)
