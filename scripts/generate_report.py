"""Generate training report summarizing all pipeline stages."""
import json
import os
import glob
from datetime import datetime
from pathlib import Path


def load_json_safe(path):
    """Load JSON file safely, returning None if not found."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def get_checkpoint_info(checkpoint_dir):
    """Get information about checkpoints in a directory."""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    if not checkpoints:
        # Check for final checkpoint
        if os.path.exists(os.path.join(checkpoint_dir, "config.json")):
            return {"type": "final", "path": checkpoint_dir}
        return None

    # Get latest checkpoint
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    latest = checkpoints[-1]
    step = int(latest.split("-")[-1])

    return {
        "type": "intermediate",
        "latest_step": step,
        "num_checkpoints": len(checkpoints),
        "path": latest,
    }


def get_tensorboard_metrics(log_dir):
    """Extract key metrics from TensorBoard logs."""
    try:
        from tensorboard.backend.event_processing import event_accumulator

        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()

        metrics = {}
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            if events:
                metrics[tag] = {
                    "final": events[-1].value,
                    "min": min(e.value for e in events),
                    "max": max(e.value for e in events),
                }
        return metrics
    except Exception:
        return None


def generate_report(output_path="training_report.md"):
    """Generate comprehensive training report."""

    report = []
    report.append("# LLM Training Pipeline Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Pipeline stages
    stages = [
        ("Pretraining", "checkpoints/pretrain", "checkpoints/pretrain_final"),
        ("SFT", "checkpoints/sft", "checkpoints/sft_final"),
        ("DPO", "checkpoints/dpo", "checkpoints/dpo_final"),
        ("LoRA", "checkpoints/lora", "checkpoints/lora_final"),
    ]

    report.append("## Training Stages Summary\n")
    report.append("| Stage | Status | Checkpoints | Final Model |")
    report.append("|-------|--------|-------------|-------------|")

    for name, ckpt_dir, final_dir in stages:
        ckpt_info = get_checkpoint_info(ckpt_dir)
        final_exists = os.path.exists(final_dir)

        if final_exists:
            status = "Complete"
        elif ckpt_info:
            status = f"In Progress (step {ckpt_info.get('latest_step', '?')})"
        else:
            status = "Not Started"

        ckpt_count = ckpt_info.get("num_checkpoints", 0) if ckpt_info else 0
        final_status = "Yes" if final_exists else "No"

        report.append(f"| {name} | {status} | {ckpt_count} | {final_status} |")

    # Evaluation results
    report.append("\n## Evaluation Results\n")

    eval_results = load_json_safe("evals/results.json")
    if eval_results:
        report.append("| Metric | Value |")
        report.append("|--------|-------|")

        metric_formats = {
            "perplexity": lambda v: f"{v:.2f}",
            "humaneval_pass@1": lambda v: f"{v:.1%}",
            "mmlu_accuracy": lambda v: f"{v:.1%}",
            "safety_refusal_rate": lambda v: f"{v:.1%}",
        }

        for metric, value in eval_results.items():
            formatter = metric_formats.get(metric, lambda v: f"{v:.4f}")
            report.append(f"| {metric} | {formatter(value)} |")
    else:
        report.append("*No evaluation results found. Run `python scripts/11_evaluate.py` to generate.*\n")

    # Gate check results
    report.append("\n## Promotion Gate Status\n")

    gate_results = load_json_safe("evals/gate_results.json")
    if gate_results:
        for stage, result in gate_results.items():
            status = "PASSED" if result.get("passed", False) else "FAILED"
            report.append(f"- **{stage.title()}**: {status}")
            if "failed_metrics" in result:
                for metric in result["failed_metrics"]:
                    report.append(f"  - Failed: {metric}")
    else:
        report.append("*No gate check results found. Run `python scripts/12_check_gates.py` after evaluation.*\n")

    # Training metrics from TensorBoard
    report.append("\n## Training Metrics\n")

    log_dirs = [
        ("Pretraining", "logs/pretrain"),
        ("Production Pretrain", "logs/production_pretrain"),
        ("SFT", "logs/sft"),
        ("DPO", "logs/dpo"),
    ]

    for name, log_dir in log_dirs:
        if os.path.exists(log_dir):
            metrics = get_tensorboard_metrics(log_dir)
            if metrics:
                report.append(f"### {name}\n")
                report.append("| Metric | Final | Min | Max |")
                report.append("|--------|-------|-----|-----|")
                for tag, values in list(metrics.items())[:10]:  # Limit to 10 metrics
                    report.append(
                        f"| {tag} | {values['final']:.4f} | {values['min']:.4f} | {values['max']:.4f} |"
                    )
                report.append("")

    # Disk usage
    report.append("\n## Disk Usage\n")

    dirs_to_check = [
        ("Checkpoints", "checkpoints"),
        ("Data", "data"),
        ("Logs", "logs"),
    ]

    report.append("| Directory | Size |")
    report.append("|-----------|------|")

    for name, dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(dir_path)
                for filename in filenames
            )
            size_gb = size / (1024**3)
            report.append(f"| {name} | {size_gb:.2f} GB |")
        else:
            report.append(f"| {name} | Not found |")

    # Recommendations
    report.append("\n## Recommendations\n")

    recommendations = []

    if not os.path.exists("checkpoints/pretrain_final"):
        recommendations.append("- Complete pretraining before moving to SFT")
    if not eval_results:
        recommendations.append("- Run evaluation suite to measure model quality")
    if not os.path.exists("data/packed"):
        recommendations.append("- Prepare training data with tokenization and packing")

    if recommendations:
        for rec in recommendations:
            report.append(rec)
    else:
        report.append("*All pipeline stages appear complete. Review metrics above for quality assessment.*")

    # Write report
    report_content = "\n".join(report)

    with open(output_path, "w") as f:
        f.write(report_content)

    print(f"Report generated: {output_path}")
    print("\n" + "=" * 60)
    print(report_content)
    print("=" * 60)

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate training pipeline report")
    parser.add_argument(
        "--output", "-o",
        default="training_report.md",
        help="Output path for report (default: training_report.md)"
    )
    args = parser.parse_args()

    generate_report(args.output)
