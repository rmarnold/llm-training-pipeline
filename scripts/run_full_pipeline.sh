#!/bin/bash
# scripts/run_full_pipeline.sh

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     A100 LLM Training Pipeline - Full Execution            ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Activate environment (if using conda)
if command -v conda &> /dev/null; then
    echo "Activating conda environment..."
    conda activate llm-train 2>/dev/null || echo "⚠️  Conda env 'llm-train' not found, using current environment"
else
    echo "ℹ️  Using current Python environment (conda not found)"
fi

# Stage 1: Data Pipeline
echo "\n[STAGE 1/7] Data Pipeline"
python scripts/01_download_data.py
python scripts/02_clean_deduplicate_optimized.py  # Using GPU-accelerated version
python scripts/03_tokenize_and_pack.py
echo "✓ Data pipeline complete"

# Stage 2: Model Initialization
echo "\n[STAGE 2/7] Model Initialization"
python scripts/04_init_model.py
python scripts/profile_memory.py
echo "✓ Model initialized"

# Stage 3: Smoke Test
echo "\n[STAGE 3/7] Smoke Test"
bash scripts/smoke_test.sh
pytest tests/ -v
echo "✓ Smoke test passed"

# Stage 4: Pretraining
echo "\n[STAGE 4/7] Pretraining"
python scripts/05_pretrain.py
python scripts/11_evaluate.py checkpoints/pretrain_final
python scripts/12_check_gates.py pretrain || exit 1
echo "✓ Pretraining complete & gates passed"

# Stage 5: Supervised Fine-Tuning
echo "\n[STAGE 5/7] Supervised Fine-Tuning"
python scripts/06_prepare_sft_data.py
python scripts/07_sft.py
python scripts/11_evaluate.py checkpoints/sft_final
python scripts/12_check_gates.py sft || exit 1
echo "✓ SFT complete & gates passed"

# Stage 6: Preference Optimization
echo "\n[STAGE 6/7] Preference Optimization (DPO)"
python scripts/08_prepare_dpo_data.py
python scripts/09_dpo.py
python scripts/11_evaluate.py checkpoints/dpo_final
python scripts/12_check_gates.py dpo || exit 1
echo "✓ DPO complete & gates passed"

# Stage 7: Domain Fine-Tuning (Optional)
echo "\n[STAGE 7/7] Domain Fine-Tuning (LoRA)"
read -p "Run domain fine-tuning? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  python scripts/10_lora_finetune.py
  python scripts/11_evaluate.py checkpoints/lora_merged
  python scripts/12_check_gates.py lora || exit 1
  echo "✓ LoRA fine-tuning complete"
fi

echo "\n╔════════════════════════════════════════════════════════════╗"
echo "║     PIPELINE COMPLETE - MODEL READY FOR DEPLOYMENT        ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Generate final report
python scripts/generate_report.py
