import yaml
import json
import sys

def check_gates(stage, results_path="evals/results.json"):
    """Check if model meets promotion criteria for given stage"""

    with open("configs/promotion_gates.yaml") as f:
        gates_config = yaml.safe_load(f)

    with open(results_path) as f:
        results = json.load(f)

    gate_name = f"{stage}_to_{get_next_stage(stage)}"
    gates = gates_config["gates"].get(gate_name, {})

    print(f"\n{'='*60}")
    print(f"CHECKING PROMOTION GATES: {gate_name}")
    print(f"{'='*60}\n")

    all_passed = True

    for gate, threshold in gates.items():
        if gate.endswith("_threshold"):
            metric = gate.replace("_threshold", "")
            actual = results.get(metric, float('inf'))

            if metric.endswith("rate") or metric.endswith("score") or metric.endswith("accuracy"):
                passed = actual >= threshold
                comparison = ">="
            else:
                passed = actual <= threshold
                comparison = "<="

            status = "✓ PASS" if passed else "❌ FAIL"
            print(f"{status} {metric}: {actual:.3f} {comparison} {threshold}")

            if not passed:
                all_passed = False

        elif isinstance(threshold, bool) and threshold:
            # Boolean checks
            passed = results.get(gate, False)
            status = "✓ PASS" if passed else "❌ FAIL"
            print(f"{status} {gate}")

            if not passed:
                all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print(f"✓ ALL GATES PASSED - APPROVED FOR PROMOTION")
        print(f"{'='*60}\n")
        return 0
    else:
        print(f"❌ SOME GATES FAILED - PROMOTION BLOCKED")
        print(f"{'='*60}\n")
        return 1

def get_next_stage(current):
    stages = {"pretrain": "sft", "sft": "dpo", "dpo": "production"}
    return stages.get(current, "unknown")

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "pretrain"
    sys.exit(check_gates(stage))
