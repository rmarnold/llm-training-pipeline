#!/usr/bin/env python3
"""Verify GPU availability and compatibility for A100 training pipeline"""

import sys
import os

def check_nvidia_smi():
    """Check if nvidia-smi is available and working"""
    print("=" * 70)
    print("1. Checking NVIDIA Driver")
    print("=" * 70)

    import subprocess
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ nvidia-smi is available\n")
            print(result.stdout)
            return True
        else:
            print("❌ nvidia-smi failed")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - NVIDIA drivers may not be installed")
        return False

def check_pytorch_cuda():
    """Check PyTorch CUDA availability"""
    print("\n" + "=" * 70)
    print("2. Checking PyTorch CUDA Support")
    print("=" * 70)

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}:")
                print(f"  Name: {torch.cuda.get_device_name(i)}")
                print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")

                # Get memory info
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)  # Convert to GB
                print(f"  Total Memory: {total_memory:.2f} GB")

                # Check if it's an A100
                gpu_name = torch.cuda.get_device_name(i)
                if "A100" in gpu_name:
                    print(f"  ✓ A100 GPU detected!")
                    return True, i
                else:
                    print(f"  ⚠️  Not an A100 GPU (found: {gpu_name})")

            return True, 0
        else:
            print("❌ CUDA not available in PyTorch")
            print("\nPossible reasons:")
            print("  1. No GPU available")
            print("  2. NVIDIA drivers not installed")
            print("  3. PyTorch not compiled with CUDA support")
            print("  4. CUDA version mismatch")
            return False, None

    except ImportError:
        print("❌ PyTorch not installed")
        return False, None

def check_gpu_memory(device_id=0):
    """Check GPU memory availability"""
    print("\n" + "=" * 70)
    print("3. Checking GPU Memory")
    print("=" * 70)

    import torch

    if not torch.cuda.is_available():
        print("❌ No GPU available")
        return False

    torch.cuda.set_device(device_id)

    # Get memory info
    total_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
    reserved = torch.cuda.memory_reserved(device_id) / (1024**3)
    free = total_memory - reserved

    print(f"Total Memory: {total_memory:.2f} GB")
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved: {reserved:.2f} GB")
    print(f"Free: {free:.2f} GB")

    # Check if we have enough memory for 7B model
    min_required = 20  # GB - conservative estimate for 7B model in BF16

    if total_memory >= 70:  # A100 80GB
        print(f"\n✓ Sufficient memory for 7B model training ({total_memory:.0f}GB available)")
        return True
    elif total_memory >= 40:  # A100 40GB
        print(f"\n⚠️  A100 40GB detected - may be tight for 7B model")
        print("   Consider using smaller batch size or gradient checkpointing")
        return True
    else:
        print(f"\n❌ Insufficient memory for 7B model (need ~{min_required}GB, have {total_memory:.2f}GB)")
        return False

def test_gpu_computation():
    """Test basic GPU computation"""
    print("\n" + "=" * 70)
    print("4. Testing GPU Computation")
    print("=" * 70)

    import torch

    if not torch.cuda.is_available():
        print("❌ No GPU available")
        return False

    try:
        # Test basic tensor operations
        print("Creating test tensors on GPU...")
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')

        print("Performing matrix multiplication...")
        z = torch.matmul(x, y)

        print("Testing BF16 support...")
        x_bf16 = x.to(torch.bfloat16)
        y_bf16 = y.to(torch.bfloat16)
        z_bf16 = torch.matmul(x_bf16, y_bf16)

        print("\n✓ GPU computation successful")
        print("✓ BF16 operations working")

        # Check if BF16 is natively supported
        if torch.cuda.is_bf16_supported():
            print("✓ Native BF16 support available (optimal for A100)")
        else:
            print("⚠️  Native BF16 not supported (will use FP16 instead)")

        return True

    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        return False

def test_model_loading():
    """Test loading a small model on GPU"""
    print("\n" + "=" * 70)
    print("5. Testing Model Loading on GPU")
    print("=" * 70)

    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        print("❌ No GPU available")
        return False

    try:
        # Check if demo checkpoint exists
        if os.path.exists("checkpoints/demo_init"):
            print("Loading demo model checkpoint...")
            model = AutoModelForCausalLM.from_pretrained(
                "checkpoints/demo_init",
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )

            print("Moving model to GPU...")
            model = model.to('cuda')

            # Test forward pass
            print("Testing forward pass on GPU...")
            input_ids = torch.randint(0, model.config.vocab_size, (1, 128), device='cuda')

            with torch.no_grad():
                outputs = model(input_ids)

            print(f"\n✓ Model loaded successfully")
            print(f"  Parameters: {model.num_parameters() / 1e6:.1f}M")
            print(f"  Device: {next(model.parameters()).device}")
            print(f"  Dtype: {next(model.parameters()).dtype}")
            print(f"✓ Forward pass successful")
            print(f"  Output shape: {outputs.logits.shape}")

            # Clean up
            del model
            torch.cuda.empty_cache()

            return True
        else:
            print("⚠️  Demo checkpoint not found, skipping model loading test")
            print("   Run: python scripts/demo_init_model.py")
            return True  # Not a critical failure

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_dependencies():
    """Check critical dependencies"""
    print("\n" + "=" * 70)
    print("6. Checking Dependencies")
    print("=" * 70)

    dependencies = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'accelerate': 'Accelerate',
        'peft': 'PEFT',
        'trl': 'TRL',
        'wandb': 'Weights & Biases',
    }

    missing = []

    for module, name in dependencies.items():
        try:
            __import__(module)
            version = __import__(module).__version__
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"❌ {name}: NOT INSTALLED")
            missing.append(name)

    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All critical dependencies installed")
        return True

def main():
    """Run all verification checks"""
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  A100 GPU Verification for Training Pipeline                    ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    results = {}

    # Run all checks
    results['nvidia_driver'] = check_nvidia_smi()
    results['pytorch_cuda'], gpu_id = check_pytorch_cuda()

    if results['pytorch_cuda']:
        results['gpu_memory'] = check_gpu_memory(gpu_id if gpu_id is not None else 0)
        results['gpu_computation'] = test_gpu_computation()
        results['model_loading'] = test_model_loading()
    else:
        results['gpu_memory'] = False
        results['gpu_computation'] = False
        results['model_loading'] = False

    results['dependencies'] = check_dependencies()

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    all_passed = True
    critical_checks = ['pytorch_cuda', 'gpu_memory', 'gpu_computation', 'dependencies']

    for check, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        critical = " (CRITICAL)" if check in critical_checks else ""
        print(f"{status} {check.replace('_', ' ').title()}{critical}")

        if check in critical_checks and not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✅ ALL CRITICAL CHECKS PASSED")
        print("\nThe pipeline is ready to run on GPU!")
        print("\nNext steps:")
        print("  1. Run demo: bash scripts/run_demo_pipeline.sh")
        print("  2. Run full pipeline: bash scripts/run_full_pipeline.sh")
        return 0
    else:
        print("\n❌ SOME CRITICAL CHECKS FAILED")
        print("\nThe pipeline cannot run on GPU in the current environment.")
        print("\nTroubleshooting:")

        if not results['pytorch_cuda']:
            print("\n  GPU Not Available:")
            print("    - Check if you're using a GPU runtime (Colab: Runtime → Change runtime type)")
            print("    - Verify NVIDIA drivers are installed: nvidia-smi")
            print("    - Check PyTorch CUDA installation: pip install torch --index-url https://download.pytorch.org/whl/cu118")

        if not results['gpu_memory']:
            print("\n  Insufficient GPU Memory:")
            print("    - Use a GPU with at least 40GB VRAM for 7B model")
            print("    - Consider using smaller model or batch size")

        if not results['dependencies']:
            print("\n  Missing Dependencies:")
            print("    - Install: pip install -r requirements.txt")

        return 1

if __name__ == "__main__":
    sys.exit(main())
