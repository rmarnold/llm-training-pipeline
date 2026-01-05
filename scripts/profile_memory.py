"""Profile GPU memory usage for training configuration optimization.

This script helps determine optimal batch sizes and identifies memory bottlenecks.
Works with any NVIDIA GPU without requiring additional packages.
"""
import torch
import argparse
import os


def get_gpu_memory_info():
    """Get GPU memory information using PyTorch CUDA functions."""
    if not torch.cuda.is_available():
        return None

    # Get device properties
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    # Memory stats
    total_memory = props.total_memory
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    free = total_memory - reserved

    return {
        "device_name": props.name,
        "total_memory": total_memory,
        "allocated": allocated,
        "reserved": reserved,
        "free": free,
        "compute_capability": f"{props.major}.{props.minor}",
    }


def format_bytes(bytes_value):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(bytes_value) < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def profile_training_memory(model_path="checkpoints/init", batch_size=1, seq_len=2048):
    """Profile memory usage for a training forward/backward pass.

    Args:
        model_path: Path to model checkpoint
        batch_size: Batch size to test
        seq_len: Sequence length to test
    """
    if not torch.cuda.is_available():
        print("Error: CUDA not available. This script requires a GPU.")
        return

    # Get initial GPU info
    gpu_info = get_gpu_memory_info()
    print("=" * 60)
    print("GPU MEMORY PROFILER")
    print("=" * 60)
    print(f"\nDevice: {gpu_info['device_name']}")
    print(f"Compute Capability: {gpu_info['compute_capability']}")
    print(f"Total VRAM: {format_bytes(gpu_info['total_memory'])}")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\nError: Model not found at {model_path}")
        print("Run model initialization first:")
        print("  python scripts/04_init_model.py")
        return

    # Load model
    print(f"\nLoading model from {model_path}...")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).cuda()

    model_params = model.num_parameters()
    print(f"Model Parameters: {model_params / 1e9:.2f}B")

    # Estimate model memory (parameters + gradients + optimizer states)
    param_memory = model_params * 2  # BF16 = 2 bytes per param
    grad_memory = model_params * 2   # Gradients
    optimizer_memory = model_params * 8  # AdamW: 2 states * 4 bytes each
    model_total = param_memory + grad_memory + optimizer_memory

    print(f"\nEstimated Memory Breakdown:")
    print(f"  Parameters: {format_bytes(param_memory)}")
    print(f"  Gradients: {format_bytes(grad_memory)}")
    print(f"  Optimizer States: {format_bytes(optimizer_memory)}")
    print(f"  Total Model Memory: {format_bytes(model_total)}")

    # Reset memory stats for accurate measurement
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Create dummy input
    print(f"\nProfiling with batch_size={batch_size}, seq_len={seq_len}...")
    input_ids = torch.randint(0, 32000, (batch_size, seq_len)).cuda()

    # Forward pass
    print("  Running forward pass...")
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    forward_memory = torch.cuda.max_memory_allocated()
    print(f"  Peak memory after forward: {format_bytes(forward_memory)}")

    # Backward pass
    print("  Running backward pass...")
    loss.backward()

    peak_memory = torch.cuda.max_memory_allocated()
    print(f"  Peak memory after backward: {format_bytes(peak_memory)}")

    # Summary
    total_vram = gpu_info['total_memory']
    utilization = peak_memory / total_vram

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total VRAM: {format_bytes(total_vram)}")
    print(f"Peak Usage: {format_bytes(peak_memory)} ({utilization*100:.1f}%)")
    print(f"Batch Size: {batch_size}")
    print(f"Sequence Length: {seq_len}")

    # Recommendations
    print(f"\n{'=' * 60}")
    print("RECOMMENDATIONS")
    print(f"{'=' * 60}")

    if utilization < 0.70:
        headroom = total_vram - peak_memory
        suggested_batch = int(batch_size * (0.85 / utilization))
        print(f"Low utilization ({utilization*100:.1f}%). Consider:")
        print(f"  - Increase batch size to {suggested_batch}")
        print(f"  - Increase sequence length")
        print(f"  - Available headroom: {format_bytes(headroom)}")

    elif utilization < 0.85:
        print(f"Moderate utilization ({utilization*100:.1f}%).")
        print("  - Current config is safe")
        print("  - Can slightly increase batch size if needed")

    elif utilization <= 0.95:
        print(f"Optimal utilization ({utilization*100:.1f}%).")
        print("  - Good balance of throughput and stability")

    else:
        print(f"High utilization ({utilization*100:.1f}%). Risk of OOM!")
        print("  - Enable gradient checkpointing")
        print("  - Reduce batch size or sequence length")
        print("  - Consider using FSDP for multi-GPU")

    # Cleanup
    del model, input_ids, outputs, loss
    torch.cuda.empty_cache()

    return peak_memory, utilization


def find_optimal_batch_size(model_path="checkpoints/init", seq_len=2048, target_utilization=0.85):
    """Binary search for optimal batch size."""
    print(f"\nSearching for optimal batch size (target: {target_utilization*100}% utilization)...")

    low, high = 1, 64
    best_batch = 1

    while low <= high:
        mid = (low + high) // 2
        print(f"\n  Testing batch_size={mid}...")

        try:
            torch.cuda.empty_cache()
            _, utilization = profile_training_memory(model_path, mid, seq_len)

            if utilization <= target_utilization:
                best_batch = mid
                low = mid + 1
            else:
                high = mid - 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM at batch_size={mid}")
                high = mid - 1
                torch.cuda.empty_cache()
            else:
                raise

    print(f"\n{'=' * 60}")
    print(f"Optimal batch size: {best_batch} (for {target_utilization*100}% target)")
    print(f"{'=' * 60}")

    return best_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile GPU memory usage")
    parser.add_argument("--model", type=str, default="checkpoints/init",
                        help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size to test")
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="Sequence length to test")
    parser.add_argument("--find-optimal", action="store_true",
                        help="Search for optimal batch size")
    parser.add_argument("--target-utilization", type=float, default=0.85,
                        help="Target GPU utilization for optimal batch search")
    args = parser.parse_args()

    if args.find_optimal:
        find_optimal_batch_size(args.model, args.seq_len, args.target_utilization)
    else:
        profile_training_memory(args.model, args.batch_size, args.seq_len)
