"""Kernel optimization setup for training (Liger Kernel, Cut Cross-Entropy, optimizers).

IMPORTANT: setup_kernel_optimizations() must be called BEFORE loading the model,
as these optimizations patch the model classes.
"""
from __future__ import annotations

# Kernel optimization flags (set before model loading)
_CCE_PATCHED = False


def setup_kernel_optimizations(
    use_liger_kernel: bool = False,
    use_cce: bool = False,
    model_type: str = "llama"
) -> dict[str, bool]:
    """Setup kernel optimizations before model loading.

    IMPORTANT: Must be called BEFORE loading the model, as these optimizations
    patch the model classes.

    Args:
        use_liger_kernel: Enable Liger Kernel (LinkedIn's Triton kernels)
        use_cce: Enable Cut Cross-Entropy (Apple's memory-efficient CE)
        model_type: Model architecture type (llama, mistral, gemma, etc.)

    Returns:
        Dict with actual enabled status of each optimization
    """
    global _CCE_PATCHED
    enabled = {"liger_kernel": False, "cce": False}

    # Liger Kernel must be applied BEFORE model loading (patches model classes)
    # Liger is Triton-based and works with torch.compile
    # NOTE: fused_linear_cross_entropy=False because it uses .item() which breaks torch.compile
    if use_liger_kernel:
        try:
            if model_type == "llama":
                from liger_kernel.transformers import apply_liger_kernel_to_llama
                apply_liger_kernel_to_llama(
                    rope=True,
                    swiglu=True,
                    rms_norm=True,
                    cross_entropy=True,
                    fused_linear_cross_entropy=False,  # Uses .item() breaking compile
                )
            elif model_type == "mistral":
                from liger_kernel.transformers import apply_liger_kernel_to_mistral
                apply_liger_kernel_to_mistral(fused_linear_cross_entropy=False)
            elif model_type == "gemma":
                from liger_kernel.transformers import apply_liger_kernel_to_gemma
                apply_liger_kernel_to_gemma(fused_linear_cross_entropy=False)
            elif model_type == "qwen2":
                from liger_kernel.transformers import apply_liger_kernel_to_qwen2
                apply_liger_kernel_to_qwen2(fused_linear_cross_entropy=False)
            else:
                print(f"Warning: Liger Kernel not available for model type '{model_type}'")
                use_liger_kernel = False

            if use_liger_kernel:
                enabled["liger_kernel"] = True
                print(f"[Kernel Optimization] Liger Kernel enabled for {model_type}")
                print(f"  - Fused RoPE, SwiGLU, RMSNorm, CrossEntropy")
                print(f"  - ~20% throughput improvement")
                print(f"  - ~60% memory reduction")
                print(f"  - Compatible with torch.compile")
        except ImportError:
            print("Warning: liger-kernel not installed. Install with:")
            print("  pip install liger-kernel")
        except Exception as e:
            print(f"Warning: Failed to enable Liger Kernel: {e}")

    # Cut Cross-Entropy - only if Liger Kernel not enabled
    # (they're mutually exclusive - both optimize cross-entropy computation)
    if use_cce and not enabled["liger_kernel"] and not _CCE_PATCHED:
        try:
            from cut_cross_entropy.transformers import cce_patch
            cce_patch(model_type)
            _CCE_PATCHED = True
            enabled["cce"] = True
            print(f"[Kernel Optimization] Cut Cross-Entropy enabled for {model_type}")
            print(f"  - ~95% memory reduction on cross-entropy loss")
        except ImportError:
            print("Warning: cut-cross-entropy not installed. Install with:")
            print("  pip install cut-cross-entropy")
        except Exception as e:
            print(f"Warning: Failed to enable Cut Cross-Entropy: {e}")
    elif use_cce and _CCE_PATCHED:
        enabled["cce"] = True
    elif use_cce and enabled["liger_kernel"]:
        print("Note: CCE skipped - Liger's CrossEntropy kernel already optimizes CE")

    return enabled


def get_optimizer_name(config_optim: str) -> str:
    """Get optimizer name with fallback for unavailable optimizers.

    Args:
        config_optim: Optimizer name from config (e.g., "adamw_bnb_8bit")

    Returns:
        Actual optimizer name to use (may fallback if dependency missing)
    """
    if config_optim == "adamw_bnb_8bit":
        try:
            import bitsandbytes  # noqa: F401
            print("[Optimizer] Using 8-bit AdamW (bitsandbytes)")
            print("  - ~4x optimizer memory reduction (~30GB saved for 7B model)")
            return "adamw_bnb_8bit"
        except ImportError:
            print("Warning: bitsandbytes not installed, falling back to adamw_torch_fused")
            print("  Install for 4x memory reduction: pip install bitsandbytes")
            return "adamw_torch_fused"
    return config_optim
