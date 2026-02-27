# TransformerEngine Open PRs: SM120/MoE/NVFP4 Relevance Analysis

**Date**: 2026-02-25
**Context**: GPT-OSS 20B MoE (32 experts/layer, top-4 routing, ~3.6B active) QLoRA fine-tuning
**Current target**: H100, planning DGX Spark (SM120/SM121) support
**Known blockers**: #2455 (GroupedGemm NVFP4), #2372 (Hadamard SM120), #2382 (FP8 convergence SM120)

---

## Known Blocker Status

### Issue #2455 - GroupedGemm: NVFP4 via cuBLAS
- **State**: OPEN
- **Description**: Implement GroupedGemm for NVFP4 format using cuBLAS kernels. Ensure integration with GroupedTensor utilities and grouped quantization pathways for Sync-Free MoE training.
- **Impact**: Without this, MoE expert layers cannot use NVFP4 grouped GEMM. Critical for efficient MoE training with FP4 quantization on Blackwell.
- **Related PRs**: #2691, #2555, #2600, #2669

### Issue #2372 - Hadamard Transform Not Working on SM120
- **State**: OPEN
- **Description**: NVFP4BlockScaling with `disable_rht=False` produces CUDA error ("Failed to set Shared Memory size") on SM120. Hadamard transform only works on SM100 (Blackwell datacenter GPUs), not SM120 (DGX Spark/GB10).
- **Workaround**: Set `disable_rht=True` in NVFP4BlockScaling recipe. Training works but with reduced quantization quality.
- **Impact**: CRITICAL for DGX Spark. Must use NVFP4 without Randomized Hadamard Transform.

### Issue #2382 - FP8 Block-Scaled Training Not Converging on SM120
- **State**: OPEN
- **Description**: FP8BlockScaling recipe does not converge on SM120 regardless of scale format (fp32 or E8). NVFP4 converges fine on SM120. FP8BlockScaling converges on SM90 (H100).
- **Comment from reporter**: Problem is related to power-of-2 scaling being forced on SM120.
- **Impact**: CRITICAL. Cannot use FP8 training on DGX Spark at all. Must use NVFP4 or BF16.

---

## TIER 1: CRITICAL - Directly Addresses Our Blockers

### PR #2693 - Enable SM120 support for fused attn if cuDNN is 9.18.1+
- **State**: OPEN
- **Updated**: 2026-02-20
- **What**: Enables fused attention on SM120 when cuDNN >= 9.18.1 is available.
- **How it helps us**: Currently fused attention is disabled on SM120. This PR unblocks one of the key performance paths for attention computation on DGX Spark. Without this, attention falls back to unfused (slower, more VRAM) implementations.
- **Priority**: MUST HAVE for DGX Spark deployment.

### PR #2665 - Improve handling of NVTE_CUDA_ARCHS
- **State**: OPEN
- **Updated**: 2026-02-18
- **What**: Fixes build system to properly handle SM120 as a target architecture. Adds regular architectures to builds with specific architectures, auto-adds sm75 fallback when cmake < 4.0.2 and sm120 is the only selected arch.
- **How it helps us**: Without this, building TransformerEngine targeting SM120 can produce empty CMAKE_CUDA_ARCHITECTURES, causing build failures. Required to compile TE for DGX Spark.
- **Priority**: MUST HAVE for DGX Spark deployment.

### PR #2555 - [NVFP4][Dense/MoE] Integrate Cutlass NVFP4 Row-Cast-Col-RHT-Transpose-Cast Fusion Kernel
- **State**: OPEN (labels: MoE, fp4)
- **Updated**: 2026-02-25
- **What**: Fuses NVFP4 quantization with Randomized Hadamard Transform for dense layers and shared experts. Input only needs to be read once instead of twice. Includes `NVTE_USE_FAST_MATH` env var for further acceleration.
- **How it helps us**: Directly relevant to our GPT-OSS 20B MoE. Shared experts and dense layers get faster NVFP4 quantization. The fast-math toggle eliminates an FP32->BF16->FP32 round trip. On H100 this is a performance win; on SM120 with `disable_rht=True` the fusion is less applicable but the general NVFP4 infrastructure improvements still help.
- **Dependency**: Requires merged PR #2564 (already merged - NVFP4 MOE Bug Fix).
- **Priority**: HIGH for H100 performance; MEDIUM for SM120 (RHT disabled there).

### PR #2691 - NVFP4 Primary Weight Support
- **State**: OPEN
- **Updated**: 2026-02-26 (very active)
- **What**: Adds NVFP4 partial cast for distributed training with ZeRO/FSDP. New kernels: `nvfp4_2d_partial_cast`, `nvfp4_transpose`, `nvfp4_fused_scale`. New API: `cast_master_weights_to_nvfp4()`. Validated on GPT-3 training with bitwise-identical loss.
- **How it helps us**: Enables keeping master weights in FP32 while model weights are NVFP4. Critical for QLoRA-style training where base weights could be NVFP4 instead of NF4. This is the foundation for true NVFP4 primary weight training which is better than NF4 on Blackwell hardware.
- **Priority**: HIGH - potential upgrade path from NF4 to NVFP4 base weights.

---

## TIER 2: HIGH IMPACT - MoE Performance & Correctness

### PR #2600 - [PyTorch] GroupedTensor Integration
- **State**: OPEN (label: MoE)
- **Updated**: 2026-02-25 (very active)
- **What**: Integrates the `GroupedTensor` class into PyTorch bindings. Makes `GroupedLinear` parameters contiguous. Exposes Python `GroupedTensor` class.
- **How it helps us**: Foundation for efficient MoE expert weight management. With 32 experts per layer, contiguous grouped parameters significantly reduce memory fragmentation and improve kernel dispatch efficiency.
- **Priority**: HIGH - core MoE infrastructure improvement.

### PR #2622 - [PyTorch] Add Grouped Linear Op and Experimental Fusion for Grouped MLP
- **State**: OPEN (label: performance)
- **Updated**: 2026-02-12
- **What**: Adds a grouped linear operation for MoE. Includes experimental fused operation for grouped MLP using a CuTe DSL kernel that computes MXFP8 grouped GEMM + SwiGLU in one pass.
- **How it helps us**: Our GPT-OSS 20B uses SwiGLU activation in expert MLPs. Fusing the grouped GEMM with SwiGLU eliminates an entire memory round-trip per expert MLP layer. Currently MXFP8 only, but the grouped linear op benefits all precisions.
- **Priority**: HIGH - major potential speedup for expert MLP forward/backward passes.

### PR #2360 - Add Device-Initiated Grouped GEMM Supporting m_splits on Device
- **State**: OPEN (label: MoE)
- **Updated**: 2026-02-13
- **What**: CUTLASS-based grouped GEMM that reads m_splits directly on GPU. Eliminates device-to-host transfers and synchronization. CUDA Graph compatible. Reduces per-expert quantization from N kernels to 1.
- **How it helps us**: With 32 experts and top-4 routing, the m_splits (tokens per expert) change every forward pass. Currently requires D2H copy + sync. This PR eliminates that CPU bottleneck entirely. CUDA Graph compatibility is a bonus for training throughput.
- **Caveat**: Currently MXFP8 only. Need NVFP4 support (issue #2455) for full benefit on Blackwell.
- **Priority**: HIGH for H100 (MXFP8 works); MEDIUM for SM120 (need NVFP4 path).

### PR #2674 - [Common] MOE Split dBias
- **State**: OPEN (labels: MoE, enhancement, cpu_overhead)
- **Updated**: 2026-02-23
- **What**: New kernel that computes `dbias` separately for each tensor in a group, outputs a grouped dbias tensor. Reduces CPU overhead in MoE backward pass.
- **How it helps us**: With 32 experts, the backward pass bias gradient computation is currently serialized. This fuses it into a single grouped kernel, reducing both CPU overhead and kernel launch latency.
- **Priority**: HIGH - backward pass optimization for MoE.

### PR #2670 - Fix NVFP4 convert_and_update_tensor Shape Check
- **State**: OPEN
- **Updated**: 2026-02-13
- **What**: Fixes bug where NVFP4 columnwise data (enforced 2D) fails shape check when rowwise_data is 3D. Fixes issue #2607.
- **How it helps us**: Bug fix that could cause crashes during NVFP4 quantization with 3D input tensors (common in batched MoE routing). Must be present for reliable NVFP4 training.
- **Priority**: HIGH - correctness fix for NVFP4.

### PR #2633 - [Common][PyTorch] Add sqrtsoftplus Score Function to Fused Router
- **State**: OPEN (label: MoE)
- **Updated**: 2026-02-25 (active)
- **What**: Adds `sqrtsoftplus` scoring function. Switches all fused router math to FP32 (from FP64) for better SM103/SM120 performance. Inputs cast to FP32 at load, all math in FP32, only cast to low-precision for output gradients.
- **How it helps us**: The FP32 math switch is directly relevant to SM120/SM121 where FP64 performance is poor. Even if we don't use sqrtsoftplus, the FP32 precision improvements benefit all router score functions on DGX Spark.
- **Priority**: HIGH for SM120 due to FP32 math change.

---

## TIER 3: MEDIUM IMPACT - Performance & Infrastructure

### PR #2559 - CPU Overhead Optimizations
- **State**: OPEN (label: cpu_overhead)
- **Updated**: 2026-02-24 (active)
- **What**: Comprehensive CPU overhead reduction. Caches Python enum-to-int conversions, adds QuantizedTensor property caching, caches libcuda.so symbol lookups, faster QuantizedTensor construction in C++.
- **How it helps us**: DGX Spark has a less powerful CPU (Grace ARM) compared to x86 servers with H100. CPU overhead that is tolerable on x86 may become a bottleneck on Grace ARM. These optimizations help across all workloads.
- **Priority**: MEDIUM-HIGH, especially important for DGX Spark's Grace ARM CPU.

### PR #2686 - [PyTorch] torch.compile Support for Permutation Functions
- **State**: OPEN
- **Updated**: 2026-02-24
- **What**: Adds `torch.compile(fullgraph=True)` support for MoE permutation ops (`moe_permute`, `moe_unpermute`, `moe_sort_chunks_by_index`). Converts `torch.autograd.Function` to PyTorch custom operators.
- **How it helps us**: Enables torch.compile for MoE token routing. With 32 experts and top-4 routing, permutation is called frequently. torch.compile can fuse surrounding operations and reduce Python overhead.
- **Priority**: MEDIUM - nice performance improvement but not blocking.

### PR #2627 - [PyTorch] SonicMoE Fused Softmax-TopK Integration
- **State**: OPEN
- **Updated**: 2026-02-10
- **What**: Integrates Dao AI Lab's SonicMoE fused softmax-topK kernel. Enabled via `NVTE_USE_SONIC_MOE=1`.
- **How it helps us**: With top-4 routing over 32 experts, the softmax-topK computation is a meaningful portion of the router. Fusing these operations reduces kernel launches and memory traffic.
- **Priority**: MEDIUM - nice to have performance boost.

### PR #2582 - Make router_fusion Adapt for Large num_of_expert (>2048)
- **State**: OPEN
- **Updated**: 2026-01-09
- **What**: Fixes router_fusion CUDA invalid argument error when expert count exceeds 2048.
- **How it helps us**: Our model has 32 experts which is well under 2048, so this is not directly needed. However, if we scale expert count in future architectures, this would be relevant.
- **Priority**: LOW for current model.

### PR #2644 - Add NVTE_BACKWARD_MODE=default|unquant|dequant
- **State**: OPEN
- **Updated**: 2026-02-25 (active)
- **What**: Adds backward mode selection. `unquant` = quantized forward + high-precision backward using unquantized activations/weights. `dequant` = quantized forward + backward from dequantized fprop values.
- **How it helps us**: Could improve convergence of NVFP4 training by using higher precision in backward pass. If SM120 has NVFP4 convergence issues (similar to FP8 issue #2382), this `unquant` mode could be a mitigation strategy.
- **Priority**: MEDIUM - convergence safety net for low-precision training.

### PR #2662 - Add Multi-Precision Training Support to FSDP Script
- **State**: OPEN
- **Updated**: 2026-02-11
- **What**: Adds `--precision` CLI arg to FSDP example supporting fp32, fp16, fp8, mxfp8, nvfp4. Auto-configures recipes per format.
- **How it helps us**: Good reference implementation for setting up NVFP4 training with FSDP. Not directly used in our Unsloth-based pipeline but useful for understanding TE's intended NVFP4 configuration.
- **Priority**: LOW-MEDIUM - reference value.

### PR #2698 - Add fused_adam, quantized_model_init, and fsdp2 Example
- **State**: OPEN
- **Updated**: 2026-02-25 (active)
- **What**: Fixes FusedAdam with FSDP2 + Float8Tensor/QuantizedTensor. Fixes fuse_wgrad_accumulation guard for vanilla FSDP2. Adds quantized_model_init examples.
- **How it helps us**: If we move from Unsloth to native TE for training, this provides the FSDP2 + quantized weight init path. FusedAdam with quantized tensors is directly relevant.
- **Priority**: LOW-MEDIUM - future upgrade path.

### PR #2521 - Enable Post-RHT Amax Estimation with Separate Amax Scale Kernel
- **State**: OPEN (label: fp4)
- **Updated**: 2026-01-08
- **What**: Separates amax estimation after Randomized Hadamard Transform into its own kernel.
- **How it helps us**: Improves NVFP4 quantization accuracy by computing amax after RHT rather than before. Better numerics for FP4 training on H100. On SM120, RHT is disabled so less relevant.
- **Priority**: MEDIUM for H100; LOW for SM120.

### PR #2637 - Fix FP8 Block Scaling with Sequence Parallel
- **State**: OPEN
- **Updated**: 2026-02-12
- **What**: Fixes assertion error when local tensor dimensions are not divisible by 128 during sequence-parallel all-gather with Float8BlockQuantizer.
- **How it helps us**: If we ever use sequence parallelism with FP8 block scaling, this fix prevents crashes. Currently using Unsloth which handles this differently.
- **Priority**: LOW - not directly relevant to current Unsloth pipeline.

### PR #2669 - PyTorch Binding for cuBLAS GEMM + Grouped Linear Integration
- **State**: OPEN
- **Updated**: 2026-02-11
- **What**: Adds cuBLAS GEMM PyTorch binding and integrates with Grouped Linear.
- **How it helps us**: Another piece of the grouped GEMM infrastructure for MoE. cuBLAS path may be needed for NVFP4 grouped GEMM (issue #2455).
- **Priority**: MEDIUM - contributes to #2455 resolution.

### PR #2704 - Fix Flash Attention 3 API Compatibility for Window Size Parameters
- **State**: OPEN (label: 2.14.0)
- **Updated**: 2026-02-26
- **What**: Updates FA3 integration for flash-attn v2.7.0+ API changes (split window_size into window_size_left/right, rename causal to is_causal).
- **How it helps us**: FA3 is Blackwell-optimized. If we use FA3 on DGX Spark (SM120), this compatibility fix is needed for recent flash-attn versions. However, per our CLAUDE.md, FA2/FA3 are incompatible with GPT-OSS (attention sinks cause wrong loss).
- **Priority**: LOW - we use Flex Attention, not FA3.

### PR #2435 - Support CUDA Graph Capture Offloading Module
- **State**: OPEN
- **Updated**: 2026-02-25 (active)
- **What**: Supports offloading modules captured by partial CUDA graphs. Fixes fp8/fp4 tensor allocation for `record_stream()` compatibility. Adds pre/post warmup hooks.
- **How it helps us**: DGX Spark has less VRAM (varies by config). CUDA graph + offloading could help fit our 20B MoE model. The fp4 tensor allocation fix is relevant for NVFP4 training.
- **Priority**: MEDIUM - VRAM optimization for DGX Spark.

### PR #2642 - Add Examples for MoE Models - Mixtral in TE
- **State**: OPEN
- **Updated**: 2026-02-18
- **What**: Tutorial for MoE training in TE using Mixtral 7B as the reference model.
- **How it helps us**: Reference material for understanding TE's MoE integration patterns. Could inform how we integrate TE MoE ops into our GPT-OSS pipeline.
- **Priority**: LOW - documentation/reference only.

### PR #2634 - Add 2D Quant for MXFP8
- **State**: OPEN
- **Updated**: 2026-02-13
- **What**: Adds 2D quantization for MXFP8 format.
- **How it helps us**: MXFP8 is the primary FP8 format on Blackwell. 2D quantization could improve accuracy for MoE where expert weights have different scale distributions.
- **Priority**: MEDIUM for SM120 if FP8 convergence (#2382) gets fixed.

---

## Recently Merged PRs (Relevant)

| PR | Title | Merged | Relevance |
|----|-------|--------|-----------|
| #2655 | [C] NVFP4 quantization for GroupedTensor | 2026-02-11 | Core NVFP4+MoE infrastructure - enables NVFP4 quantization of grouped expert weights |
| #2654 | [PyTorch] Python GroupedTensor | 2026-02-11 | Python bindings for GroupedTensor - prerequisite for #2600 |
| #2664 | [PyTorch] Add ops for MoE grouped MLP | 2026-02-12 | MoE grouped MLP operations - building blocks for #2622 |
| #2564 | [NVFP4][MOE] Bug Fix for NVFP4 Grouped Quant | 2026-01-07 | Critical bug fix for NVFP4 MoE quantization |
| #2411 | [PyTorch][NVFP4][MOE] NVFP4 Grouped Quantize with Hadamard Transform | 2025-12-20 | Foundation for NVFP4 MoE quantization with RHT |
| #2351 | [PyTorch][NVFP4][MOE] NVFP4 Grouped Hadamard Amax Kernel | 2025-11-25 | Amax kernel for NVFP4 MoE |
| #2412 | [Common] Tuned NVFP4 cast kernel | 2026-01-21 | Optimized NVFP4 quantization kernel |
| #2615 | [Common] Disabled the tuned NVFP4 kernels | 2026-01-23 | Tuned kernels had numerics bugs - disabled pending fix |
| #2639 | [Common] Fix NVFP4 tuned-kernel numerics | 2026-02-03 | Fixed the numerics issues from #2412 |
| #2584 | [Common] Enable determinism for cuDNN >= 9.18.1 on Blackwell | 2026-01-20 | Deterministic attention on Blackwell |
| #2589 | (Bug fix) Fix accuracy issue for blockwise scaling+E8 scale on Blackwell | 2026-01-15 | Fixed FP8 blockwise scaling accuracy on Blackwell |
| #2486 | Add logic for block-scaled tensors with GEMM swizzled scales | 2026-01-17 | Block-scaled GEMM with swizzled scales for MoE |
| #2482 | Fix the sm120 compilation with CUDA 12 | 2025-12-09 | SM120 compilation fix - already merged |
| #2320 | [PyTorch] Fix attention backend and tests for sm120 | 2025-10-30 | SM120 attention backend fixes |
| #2279 | Overhaul the compilation for the arch-specific features | 2025-10-23 | Build system overhaul for arch-specific features |
| #2215 | [PyTorch][MOE] Support NVFP4 Grouped Linear | 2025-10-21 | NVFP4 grouped linear for MoE |
| #2631 | Fix minimum version of cublas for grouped gemm | 2026-01-30 | cuBLAS version fix for grouped GEMM |

---

## Summary: DGX Spark Readiness Assessment

### What Works Today on SM120
- NVFP4 training (with `disable_rht=True`)
- BF16 compute
- Basic attention (unfused fallback)

### What Is Blocked on SM120
1. **FP8 training** - does not converge (#2382, no PR fix yet)
2. **Hadamard Transform** - CUDA error (#2372, no PR fix yet)
3. **Fused attention** - disabled, PR #2693 would fix (requires cuDNN 9.18.1+)
4. **Build issues** - PR #2665 fixes NVTE_CUDA_ARCHS handling

### Recommended Watch List (PRs to track for our use case)
1. **#2693** - SM120 fused attention (CRITICAL)
2. **#2665** - SM120 build fix (CRITICAL)
3. **#2691** - NVFP4 primary weights (HIGH)
4. **#2555** - NVFP4 RHT fusion for MoE (HIGH)
5. **#2600** - GroupedTensor integration (HIGH)
6. **#2622** - Grouped MLP fusion (HIGH)
7. **#2360** - Device-initiated grouped GEMM (HIGH)
8. **#2674** - MoE split dBias (HIGH)
9. **#2559** - CPU overhead optimizations (MEDIUM-HIGH for Grace ARM)
10. **#2644** - NVTE_BACKWARD_MODE (MEDIUM - convergence safety)

### Recommended Strategy for DGX Spark
1. Use NVFP4 (not FP8) as the quantization format on SM120
2. Set `disable_rht=True` in NVFP4BlockScaling recipe
3. Wait for PR #2693 + #2665 before attempting DGX Spark builds
4. Monitor #2382 for any FP8 convergence fix on SM120
5. Consider `NVTE_BACKWARD_MODE=unquant` (#2644) if NVFP4 training shows convergence issues
6. The Grace ARM CPU on DGX Spark makes CPU overhead PRs (#2559, #2674) more important than on x86
