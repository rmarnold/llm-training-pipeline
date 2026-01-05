"""Check promotion gates between training stages.

This script validates that model metrics meet the required thresholds
before promoting to the next training stage.

Usage:
    python scripts/12_check_gates.py pretrain
    python scripts/12_check_gates.py sft
    python scripts/12_check_gates.py dpo
"""
import yaml
import json
import sys
import os


def check_gates(stage, results_path="evals/results.json", gates_config_path="configs/promotion_gates.yaml"):
    """Check if model meets promotion criteria for given stage.

    Args:
        stage: Current training stage (pretrain, sft, dpo)
        results_path: Path to evaluation results JSON
        gates_config_path: Path to promotion gates config

    Returns:
        0 if all gates passed, 1 if any failed, 2 if missing files
    """
    # Check if gates config exists
    if not os.path.exists(gates_config_path):
        print(f"Error: Gates config not found: {gates_config_path}")
        print("Create configs/promotion_gates.yaml with threshold definitions.")
        return 2

    # Check if evals directory exists
    evals_dir = os.path.dirname(results_path)
    if evals_dir and not os.path.exists(evals_dir):
        print(f"Error: Evaluations directory not found: {evals_dir}")
        print("")
        print("Run evaluation first:")
        print(f"  python scripts/11_evaluate.py checkpoints/{stage}_final")
        return 2

    # Check if results file exists
    if not os.path.exists(results_path):
        print(f"Error: Evaluation results not found: {results_path}")
        print("")
        print("Run evaluation first:")
        print(f"  python scripts/11_evaluate.py checkpoints/{stage}_final")
        return 2

    # Load configs
    try:
        with open(gates_config_path) as f:
            gates_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in gates config: {e}")
        return 2

    try:
        with open(results_path) as f:
            results = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in results file: {e}")
        return 2

    # Determine gate name
    next_stage = get_next_stage(stage)
    gate_name = f"{stage}_to_{next_stage}"

    # Get gates for this transition
    gates = gates_config.get("gates", {}).get(gate_name, {})

    if not gates:
        # Try alternate config format
        gates = gates_config.get(stage, {}).get("metrics", {})

    if not gates:
        print(f"Warning: No gates defined for {gate_name}")
        print("Allowing promotion by default.")
        return 0

    print(f"\n{'='*60}")
    print(f"CHECKING PROMOTION GATES: {stage} → {next_stage}")
    print(f"{'='*60}\n")

    all_passed = True
    failed_metrics = []

    for gate, threshold in gates.items():
        if gate.endswith("_threshold"):
            metric = gate.replace("_threshold", "")
        else:
            metric = gate

        actual = results.get(metric)

        if actual is None:
            print(f"⚠ SKIP  {metric}: Not found in results")
            continue

        # Determine comparison direction
        # Higher is better for: accuracy, rate, score, pass@k
        # Lower is better for: perplexity, loss
        higher_is_better = any(
            keyword in metric.lower()
            for keyword in ["accuracy", "rate", "score", "pass@", "refusal"]
        )

        if higher_is_better:
            passed = actual >= threshold
            comparison = ">="
        else:
            passed = actual <= threshold
            comparison = "<="

        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}  {metric}: {actual:.4f} {comparison} {threshold}")

        if not passed:
            all_passed = False
            failed_metrics.append(metric)

    print(f"\n{'='*60}")
    if all_passed:
        print(f"✓ ALL GATES PASSED - APPROVED FOR PROMOTION TO {next_stage.upper()}")
        print(f"{'='*60}\n")
        save_gate_result(stage, True, [], results_path)
        return 0
    else:
        print(f"❌ GATES FAILED - PROMOTION BLOCKED")
        print(f"   Failed metrics: {', '.join(failed_metrics)}")
        print(f"{'='*60}\n")
        save_gate_result(stage, False, failed_metrics, results_path)
        return 1


def get_next_stage(current):
    """Get the next training stage."""
    stages = {
        "pretrain": "sft",
        "sft": "dpo",
        "dpo": "production"
    }
    return stages.get(current, "unknown")


def save_gate_result(stage, passed, failed_metrics, results_path):
    """Save gate check results for reporting."""
    gate_results_path = os.path.join(os.path.dirname(results_path), "gate_results.json")

    # Load existing results
    if os.path.exists(gate_results_path):
        with open(gate_results_path) as f:
            gate_results = json.load(f)
    else:
        gate_results = {}

    # Update with current stage
    gate_results[stage] = {
        "passed": passed,
        "failed_metrics": failed_metrics,
        "next_stage": get_next_stage(stage)
    }

    # Save
    os.makedirs(os.path.dirname(gate_results_path), exist_ok=True)
    with open(gate_results_path, "w") as f:
        json.dump(gate_results, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check promotion gates between training stages")
    parser.add_argument("stage", nargs="?", default="pretrain",
                        choices=["pretrain", "sft", "dpo"],
                        help="Training stage to check gates for")
    parser.add_argument("--results", type=str, default="evals/results.json",
                        help="Path to evaluation results JSON")
    parser.add_argument("--gates-config", type=str, default="configs/promotion_gates.yaml",
                        help="Path to promotion gates config")
    args = parser.parse_args()

    sys.exit(check_gates(args.stage, args.results, args.gates_config))
