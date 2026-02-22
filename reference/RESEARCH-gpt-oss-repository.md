# GPT-OSS Repository Deep Research

**Date**: 2026-02-21
**Repository**: https://github.com/openai/gpt-oss
**Stars**: ~19,800 | **Language**: Python + C + Metal + Triton
**Version**: 0.0.9

## Research Summary

The GPT-OSS repository is OpenAI's official reference implementation for GPT-OSS-20B and GPT-OSS-120B inference. It contains three complete inference backends (PyTorch, Triton/CUDA, Metal/Apple Silicon), a full Responses API server, Harmony chat format handling via the `openai_harmony` library, tool calling (browser, Python sandbox, apply_patch), and custom kernels for MXFP4 dequantization, MoE routing, and Flash Attention with learned sinks. The reinforcement fine-tuning notebook demonstrates GRPO training with Unsloth on a free Colab T4.

---

## 1. Complete File Structure

```
openai/gpt-oss/
|-- README.md
|-- LICENSE
|-- USAGE_POLICY
|-- pyproject.toml                    # Python package config
|-- CMakeLists.txt                    # Metal C library build
|-- MANIFEST.in
|-- awesome-gpt-oss.md
|
|-- _build/                           # Custom build backend
|   |-- gpt_oss_build_backend/
|       |-- __init__.py
|       |-- backend.py
|
|-- examples/
|   |-- reinforcement-fine-tuning.ipynb   # GRPO RL with Unsloth (SAVED LOCALLY)
|   |-- gradio/gradio_chat.py
|   |-- streamlit/streamlit_chat.py
|   |-- agents-sdk-js/                    # OpenAI Agents SDK JS example
|   |-- agents-sdk-python/                # OpenAI Agents SDK Python example
|
|-- gpt_oss/
|   |-- __init__.py
|   |-- chat.py                       # Full Harmony chat w/ tools, StreamableParser
|   |-- generate.py                   # Simple text generation
|   |-- tokenizer.py                  # Tiktoken wrapper
|   |
|   |-- torch/                        # PyTorch reference implementation
|   |   |-- __init__.py
|   |   |-- model.py                  # Full Transformer, MoE, SDPA, RoPE
|   |   |-- weights.py                # MXFP4 checkpoint loading
|   |   |-- utils.py                  # Distributed init
|   |
|   |-- triton/                       # Triton/CUDA optimized implementation
|   |   |-- __init__.py
|   |   |-- model.py                  # Transformer with KV cache, CUDAGraph
|   |   |-- attention.py              # Flash Attention w/ sinks + banded attention
|   |   |-- moe.py                    # MoE with triton_kernels MXFP4 quantization
|   |
|   |-- metal/                        # Apple Silicon Metal implementation
|   |   |-- __init__.py
|   |   |-- CMakeLists.txt
|   |   |-- include/gpt-oss.h
|   |   |-- include/gpt-oss/functions.h   # C API (model, context, sampler)
|   |   |-- include/gpt-oss/types.h       # Status codes, special tokens
|   |   |-- include/gpt-oss/macros.h
|   |   |-- python/                        # CPython bindings
|   |   |   |-- context.c, model.c, module.c, module.h, tokenizer.c
|   |   |-- source/                        # Metal shaders + C runtime
|   |   |   |-- accumulate.metal           # Expert output accumulation
|   |   |   |-- convert.metal              # MXFP4 -> F32 dequantization
|   |   |   |-- embeddings.metal           # BF16 -> F32 embeddings
|   |   |   |-- expert_routing_metadata.metal # MoE expert routing
|   |   |   |-- gather_and_accumulate.metal   # Expert gather + weighted sum
|   |   |   |-- matmul.metal               # Dense matmul (BF16 weights)
|   |   |   |-- moematmul.metal            # MoE matmul with MXFP4 dequant
|   |   |   |-- random.metal               # Squares RNG
|   |   |   |-- rmsnorm.metal              # RMS normalization
|   |   |   |-- rope.metal                 # Rotary Position Embeddings (YaRN)
|   |   |   |-- sample.metal               # Softmax + sampling
|   |   |   |-- scatter.metal              # Token scatter to experts
|   |   |   |-- sdpa.metal                 # Scaled Dot-Product Attention
|   |   |   |-- topk.metal                 # Top-K expert selection (128->4)
|   |   |   |-- context.c                  # Metal context management
|   |   |   |-- generate.c                 # Token generation loop
|   |   |   |-- log.c                      # Logging
|   |   |   |-- metal.m                    # Metal device/queue setup
|   |   |   |-- metal-kernels.c            # Kernel dispatch
|   |   |   |-- model.c                    # Model loading (custom binary format)
|   |   |   |-- tokenizer.c               # Tokenizer from binary format
|   |   |   |-- include/internal/          # Internal headers
|   |   |       |-- kernel-args.h          # All kernel argument structs
|   |   |       |-- datatype.h/hpp, metal.h/hpp, model.h
|   |   |       |-- macros.h, math.h, rng.h/hpp, storage.h, uuid.h, log.h
|   |   |-- benchmark/                    # Metal kernel benchmarks
|   |   |   |-- end-to-end.cc, end-to-end-threadgroup.cc
|   |   |   |-- f32-bf16w-rmsnorm.cc, f32-random.cc
|   |   |   |-- mf4-f32-convert.cc, u32-random.cc
|   |   |-- test/                          # Metal kernel unit tests
|   |   |   |-- bf16-f32-embeddings.cc, f32-bf16w-matmul.cc
|   |   |   |-- f32-bf16w-rmsnorm.cc, f32-random.cc, f32-rope.cc
|   |   |   |-- mf4-f32-convert.cc, u32-random.cc
|   |   |   |-- embeddings-kernel-tester.hpp, matmul-kernel-tester.hpp
|   |   |   |-- fill-random-kernel-tester.hpp, rmsnorm-kernel-tester.hpp, rope-kernel-tester.hpp
|   |   |-- examples/
|   |   |   |-- chat.py, generate.py       # Metal inference examples
|   |   |-- scripts/
|   |       |-- create-local-model.py      # Convert safetensors to Metal format
|   |
|   |-- vllm/                         # vLLM backend
|   |   |-- token_generator.py        # Thin wrapper around vLLM engine
|   |
|   |-- responses_api/                # OpenAI Responses API server
|   |   |-- __init__.py
|   |   |-- api_server.py             # FastAPI server with full Harmony parsing
|   |   |-- events.py                 # SSE event types
|   |   |-- serve.py                  # Server launcher
|   |   |-- types.py                  # Pydantic models
|   |   |-- utils.py
|   |   |-- inference/                # Backend adapters
|   |       |-- __init__.py
|   |       |-- metal.py              # Metal backend (via C API)
|   |       |-- ollama.py             # Ollama backend
|   |       |-- stub.py               # Stub for testing
|   |       |-- transformers.py       # HuggingFace transformers
|   |       |-- triton.py             # Triton/CUDA with CUDAGraph
|   |       |-- vllm.py               # vLLM backend
|   |
|   |-- tools/                        # Tool implementations
|   |   |-- __init__.py
|   |   |-- tool.py                   # Abstract Tool base class
|   |   |-- apply_patch.py            # Patch application tool
|   |   |-- apply_patch.md            # Patch format documentation
|   |   |-- simple_browser/           # Web search tool
|   |   |   |-- __init__.py
|   |   |   |-- backend.py            # YouCom, Exa backends
|   |   |   |-- page_contents.py
|   |   |   |-- simple_browser_tool.py
|   |   |-- python_docker/
|   |       |-- docker_tool.py        # Python sandbox in Docker
|   |
|   |-- evals/                        # Evaluation suite
|       |-- README.md, __init__.py, __main__.py
|       |-- aime_eval.py, basic_eval.py, gpqa_eval.py, healthbench_eval.py
|       |-- chat_completions_sampler.py, responses_sampler.py
|       |-- abcd_grader.py, report.py, types.py
|
|-- gpt-oss-mcp-server/              # MCP server
|   |-- README.md, pyproject.toml
|   |-- browser_server.py, python_server.py
|   |-- build-system-prompt.py, reference-system-prompt.py
|
|-- compatibility-test/               # Provider compatibility tests
|   |-- README.md, package.json, package-lock.json
|   |-- analysis.ts, cases.jsonl, index.ts
|   |-- providers.ts, runCase.ts, tools.ts
|
|-- tests/                            # Unit tests
|   |-- conftest.py, test_api_endpoints.py, test_responses_api.py
|   |-- gpt_oss/tools/simple_browser/test_backend.py
|
|-- tests-data/
    |-- basic-event-stream.txt
    |-- web-search-event-stream.txt
```

---

## 2. Custom Kernel Catalog

### 2.1 Metal Shaders (Apple Silicon) -- 14 kernels across 13 .metal files

| File | Kernel(s) | Purpose |
|------|-----------|---------|
| `matmul.metal` | `gptoss_f32_bf16w_matmul` | MV product: f32 input x bf16 weights, simdgroup reduction |
| `matmul.metal` | `gptoss_f32_bf16w_matmul_qkv` | Fused QKV projection + RoPE + KV cache write |
| `matmul.metal` | `gptoss_f32_bf16w_unembedding` | Unembedding with fused argmax reduction |
| `matmul.metal` | `gptoss_f32_bf16w_dense_matmul_qkv` | Dense tiled matmul (simdgroup_float8x8) for QKV with KV cache |
| `matmul.metal` | `gptoss_f32_bf16w_dense_matmul_attn_output` | Dense matmul for attention output (with residual add) |
| `matmul.metal` | `gptoss_f32_bf16w_dense_matmul_mlp_gate` | Dense matmul for MLP gate projection |
| `moematmul.metal` | `gptoss_f32_mf4w_moe_matmul_swiglu` | MoE MV matmul: MXFP4 dequant + fused SwiGLU activation |
| `moematmul.metal` | `gptoss_f32_mf4w_moe_matmul` | MoE MV matmul: MXFP4 dequant (no activation) |
| `moematmul.metal` | `gptoss_f32_mf4w_moe_dense_matmul_swiglu` | MoE tiled matmul with MXFP4 dequant + SwiGLU (batch mode) |
| `moematmul.metal` | `gptoss_f32_mf4w_moe_dense_matmul` | MoE tiled matmul with MXFP4 dequant (batch mode) |
| `sdpa.metal` | `gptoss_f32_sdpa_q8_d64` | Flash Attention: 8 Q heads / 1 KV head, d=64, sliding window |
| `convert.metal` | `gptoss_mf4_f32_convert` | MXFP4 block -> F32 dequantization |
| `embeddings.metal` | `gptoss_bf16_f32_embeddings` | BF16 embedding lookup -> F32 |
| `expert_routing_metadata.metal` | `gptoss_f32_expert_routing_metadata` | Compute expert offsets for MoE scatter/gather |
| `gather_and_accumulate.metal` | `gptoss_f32_gather_and_accumulate_e4` | Gather from 4 experts + weighted accumulate |
| `scatter.metal` | `gptoss_f32_scatter_e4` | Scatter tokens to 4 experts |
| `accumulate.metal` | `gptoss_f32_accumulate_e4` | Accumulate 4 expert outputs with softmax scores |
| `topk.metal` | `gptoss_f32_topk_softmax_e128_k4` | Top-4 from up to 128 experts + softmax normalization (20B uses 32 experts) |
| `rmsnorm.metal` | `gptoss_f32_bf16w_rmsnorm` | RMS normalization with bf16 weights |
| `rope.metal` | `gptoss_f32_rope` | RoPE with YaRN scaling + KV cache write |
| `sample.metal` | `gptoss_f32_softmax` | Softmax + categorical sampling |
| `random.metal` | `gptoss_u32_fill_random` / `gptoss_f32_fill_random` | Squares RNG |

### 2.2 Triton Kernels (CUDA/H100)

| File | Function | Purpose |
|------|----------|---------|
| `triton/attention.py` | `_attn_fwd` | FlashAttention v2 with **learned attention sinks** + **banded attention** (sliding window). Uses TensorDescriptor for H100 TMA. |
| `triton/moe.py` | `moe()` | MoE dispatch using `triton_kernels` library: routing, scatter, MXFP4 matmul, fused SwiGLU, gather |

**External dependencies for Triton backend:**
- `triton_kernels` (separate library): `matmul_ogs`, `routing`, `swiglu`, `convert_layout`, `InFlexData`
- `triton_kernels.numerics_details.mxfp.downcast_to_mxfp` for MXFP4 quantization
- `triton_kernels.tensor`: `FP4`, `HopperMXValueLayout`, `HopperMXScaleLayout`

### 2.3 No CUDA Kernels

There are NO raw CUDA kernels (.cu files) in the repository. The CUDA path is entirely through Triton JIT compilation.

---

## 3. MXFP4 Quantization/Dequantization

### 3.1 Format Details

MXFP4 (Microscaling FP4) uses:
- **4-bit values**: 16 possible values: `[+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]`
- **8-bit shared exponent**: Per block of 32 elements
- **Storage**: 32 FP4 values packed into 16 bytes (2 nibbles per byte) + 1 scale byte = 17 bytes per 32 elements
- **Checkpoint format**: `.blocks` (packed uint8) + `.scales` (uint8, bias 127)

### 3.2 Metal Dequantization (convert.metal)

The MXFP4 dequant kernel uses a clever bit-manipulation approach:
1. Split packed uint4 block into even/odd nibbles via shift + mask
2. Map nibbles to FP16 exponent form: `nibble * 2` -> add bias `0x70` -> mask to `0x8E` (extracts sign + exponent)
3. Reinterpret as half4 and multiply by shared scale
4. The scale is computed as `as_type<float>((uint(scale_byte) + 14) << 23)` which creates an IEEE float with the right exponent

This is replicated in the MoE matmul kernels (`moematmul.metal`) where dequantization is **fused** with the matrix multiply -- no intermediate buffer needed.

### 3.3 Python Dequantization (weights.py)

The CPU/CUDA path uses a lookup table approach:
```python
FP4_VALUES = [+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
lut = torch.tensor(FP4_VALUES, dtype=dtype)
idx_lo = (blocks & 0x0F).to(torch.long)
idx_hi = (blocks >> 4).to(torch.long)
sub[:, 0::2] = lut[idx_lo]
sub[:, 1::2] = lut[idx_hi]
torch.ldexp(sub, scales, out=sub)
```

### 3.4 Triton MXFP4 (moe.py)

Uses `triton_kernels.numerics_details.mxfp.downcast_to_mxfp` for quantization and `HopperMXValueLayout` / `HopperMXScaleLayout` for optimal Hopper TMA access patterns.

---

## 4. MoE (Mixture of Experts) Implementation

### 4.1 Architecture

Two model sizes exist in the GPT-OSS family:

| | **GPT-OSS 20B** (our training target) | **GPT-OSS 120B** (default in model.py) |
|---|---|---|
| Experts/layer | 32 | 128 |
| Layers | 24 | 36 |
| Hidden size | 2880 | 2880 |
| Intermediate size | 2880 | 2880 |
| Total params | 20.9B | ~120B |
| Active params | 3.6B (top-4 of 32) | ~3.6B (top-4 of 128) |

- **Top-4 routing** (experts_per_token=4) for both sizes
- **Weights stored as MXFP4** (blocks + scales)
- **Biases stored as BF16**
- **NOTE**: The DEFAULT `ModelConfig` in `torch/model.py` is for the 120B model. The 20B config is loaded via `config.json`.

### 4.2 Metal MoE Pipeline (Apple Silicon)

1. `gptoss_f32_topk_softmax_e128_k4` -- Select top-4 from experts (up to 128 for 120B, 32 for 20B), compute softmax scores
2. `gptoss_f32_expert_routing_metadata` -- Build per-expert token counts and offsets (parallel histogram)
3. `gptoss_f32_scatter_e4` -- Scatter input tokens to their 4 assigned expert slots
4. `gptoss_f32_mf4w_moe_matmul_swiglu` / `moe_dense_matmul_swiglu` -- MLP1: MXFP4 matmul + fused SwiGLU
5. `gptoss_f32_mf4w_moe_matmul` / `moe_dense_matmul` -- MLP2: MXFP4 matmul
6. `gptoss_f32_gather_and_accumulate_e4` / `gptoss_f32_accumulate_e4` -- Gather from experts + weighted sum

Two kernel paths:
- **MV path** (single token): `moe_matmul_swiglu` / `moe_matmul` -- each simdgroup handles one output row
- **Dense/batch path** (multiple tokens): `moe_dense_matmul_swiglu` / `moe_dense_matmul` -- tiled matmul with simdgroup_float8x8

### 4.3 SwiGLU Activation

Custom SwiGLU with clamping and +1 linear bias:
```
swish(x) = min(x, limit) * sigmoid(1.702 * min(x, limit))
swiglu(x) = swish(x_glu) * (clamp(x_linear, -limit, limit) + 1)
```
- `alpha = 1.702` (hardcoded)
- `limit = 7.0` (configurable via `swiglu_limit`)
- The `+1` on the linear branch is a GPT-OSS-specific modification

### 4.4 PyTorch MoE (Reference)

Simple einsum-based implementation with `torch.topk` routing:
```python
experts = torch.topk(gate_logits, k=4, sorted=True)
expert_weights = F.softmax(experts.values, dim=1)
mlp1_weight = self.mlp1_weight[expert_indices, ...]
t = einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
t = swiglu(t)
t = einsum("beck,bek->bec", mlp2_weight, t) + mlp2_bias
t = einsum("bec,be->bc", t, expert_weights)
```

### 4.5 Triton MoE

Uses `triton_kernels` library for fused operations:
```python
logits = matmul_ogs(x, wg, bg)                    # Gate projection
rdata, gather_indx, scatter_indx = routing(logits, 4)  # Top-4 routing
x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather_indx,
               fused_activation=FusedActivation("swiglu"))  # MLP1 + SwiGLU
x = matmul_ogs(x, w2, b2, rdata, scatter_indx=scatter_indx,
               gammas=rdata.gate_scal)              # MLP2 + accumulate
```

---

## 5. Attention Implementation

### 5.1 Architecture

- **64 Q heads, 8 KV heads** (GQA ratio 8:1)
- **head_dim = 64**
- **Sliding window = 128** on even layers (0, 2, 4...), full attention on odd layers
- **Learned attention sinks** (`self.sinks` parameter per head) -- trainable scalar bias in softmax
- **RoPE** with YaRN scaling (scaling_factor=32, rope_theta=150000)

### 5.2 Metal SDPA Kernel

`gptoss_f32_sdpa_q8_d64` processes 8 Q heads per KV head simultaneously:
- Each threadgroup handles all 8 Q heads for one KV head group for one query token
- Online softmax (Flash Attention style) with running max `m` and sum `l`
- Sliding window support via `args.window`
- Multi-simdgroup parallelism across KV sequence with cross-simdgroup reduction
- Values stored interleaved in KV cache: `[K_dim, V_dim]` per token

### 5.3 Triton FlashAttention

Extended Flash Attention v2 with:
- **Learned sinks**: Softmax includes a per-head learned bias term
- **Banded attention**: Configurable `BANDWIDTH` parameter for sliding window
- Uses Hopper TensorDescriptor for TMA loads
- `allow_tf32=False` for numerical precision
- Causal masking with sliding window

### 5.4 YaRN RoPE

Both Metal and Python implement NTK-aware YaRN rope scaling:
```
freq = base^(dim_idx / head_dim)
interpolation = 1/(scaling_factor * freq)
extrapolation = 1/freq
ramp = (dim_idx - low) / (high - low)
mask = 1 - clamp(ramp, 0, 1)
inv_freq = interpolation * (1-mask) + extrapolation * mask
concentration = 0.1 * log(scaling_factor) + 1.0
```

---

## 6. Harmony Format Handling

### 6.1 openai_harmony Library

The repository depends on `openai-harmony` (external package) for:
- `Conversation`, `Message`, `Author`, `Role` -- message data structures
- `SystemContent`, `DeveloperContent`, `TextContent` -- content types
- `ToolDescription` -- tool definitions
- `HarmonyEncoding`, `HarmonyEncodingName.HARMONY_GPT_OSS` -- tokenization
- `StreamableParser`, `StreamState` -- streaming token parser
- `ReasoningEffort` -- low/medium/high reasoning control

### 6.2 Special Tokens (from types.h)

```
gptoss_special_token_return     = 1   # <|return|>
gptoss_special_token_start      = 2   # <|start|>
gptoss_special_token_message    = 3   # <|message|>
gptoss_special_token_end        = 4   # <|end|>
gptoss_special_token_refusal    = 5   # <|refusal|>
gptoss_special_token_constrain  = 6   # <|constrain|>
gptoss_special_token_channel    = 7   # <|channel|>
gptoss_special_token_call       = 8   # <|call|>
gptoss_special_token_untrusted  = 9   # <|untrusted|>
gptoss_special_token_end_untrusted = 10 # <|end_untrusted|>
```

### 6.3 Chat Format (from chat.py)

```
<|start|>system<|message|>[system content]<|end|>
<|start|>user<|message|>[user text]<|end|>
<|start|>assistant<|channel|>analysis<|message|>[thinking]<|end|>
<|start|>assistant<|channel|>final<|message|>[response]<|return|>
```

### 6.4 Channels

- `analysis` -- Chain-of-thought / reasoning
- `commentary` -- Tool calls must go to commentary channel
- `final` -- User-facing response

### 6.5 Reasoning Effort

Set via `SystemContent.with_reasoning_effort()`:
- `ReasoningEffort.HIGH` -- verbose reasoning
- `ReasoningEffort.MEDIUM` -- balanced
- `ReasoningEffort.LOW` -- minimal reasoning

---

## 7. Tool Calling

### 7.1 Tool Architecture

Abstract `Tool` base class with:
- `name` property -- routing identifier
- `process(message)` -- async generator yielding response messages
- `instruction()` -- tool documentation for system prompt
- Channel propagation: tool outputs inherit the channel of the triggering message

### 7.2 Available Tools

1. **SimpleBrowserTool** -- Web search via YouCom or Exa backends
   - Recipient: `browser.search`, `browser.open`
   - Returns search results as tool messages

2. **PythonTool** -- Python code execution in Docker sandbox
   - Recipient: `python`
   - Docker-isolated execution environment

3. **apply_patch** -- File patching
   - Recipient: `functions.apply_patch`
   - Uses "Begin Patch / End Patch" format
   - JSON parameter extraction supported

### 7.3 Tool Call Flow in Harmony

```
assistant -> browser.search: [search query]  (via recipient field)
tool (browser.search) -> assistant: [search results]
assistant -> final: [response with citations]
```

### 7.4 Responses API Tool Support

The Responses API server (`api_server.py`) maps Harmony tool calls to OpenAI API types:
- `FunctionCallItem` for custom functions
- `WebSearchCallItem` for browser tool
- `CodeInterpreterCallItem` for Python tool
- Full SSE streaming with progress events

---

## 8. Reinforcement Fine-Tuning Notebook

**Saved to**: `~/Projects/llm-training-pipeline/reference/reinforcement-fine-tuning.ipynb`

### Key Details

- **Algorithm**: GRPO (Group Relative Policy Optimization) via TRL
- **Framework**: Unsloth for memory optimization
- **Hardware**: Free Colab T4 (15GB) -- also supports H100
- **Task**: Teaching GPT-OSS-20B to play 2048 game
- **Model**: `unsloth/gpt-oss-20b` (4-bit quantized for T4)

### Training Configuration

```python
max_seq_length = 768
lora_rank = 4
load_in_4bit = True
offload_embedding = True  # Saves 1GB VRAM
```

### LoRA Targets

```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
lora_alpha = lora_rank * 2  # 2x speeds up training
```

### GRPO Config

```python
temperature = 1.0
learning_rate = 5e-5
per_device_train_batch_size = 1 (auto-adjusted to num_generations=2)
max_steps = 1000
optim = "adamw_8bit"
```

### Reward Functions

1. `function_works` -- +1 if valid Python, -2 if invalid
2. `no_cheating` -- +1 if only stdlib imports, -20 if non-stdlib
3. `strategy_succeeds` -- +20 if wins 2048, +2 if valid but loses, -1 timeout, -3 error

### Save Formats

- `save_method="mxfp4"` -- Native MXFP4 precision
- `save_method="merged_16bit"` -- FP16 merged weights
- `save_method="lora"` -- LoRA adapters only

---

## 9. Inference Backends Summary

| Backend | Hardware | KV Cache | CUDA Graph | Quantization | Batch Mode |
|---------|----------|----------|------------|--------------|------------|
| `torch` | Any GPU | No | No | BF16 weights, MXFP4 MoE | No (sequential) |
| `triton` | NVIDIA (H100) | Yes (BF16) | Yes | MXFP4 via triton_kernels | Yes (prefill) |
| `metal` | Apple Silicon | Yes | N/A | MXFP4 (fused dequant) | Yes (dense kernels) |
| `vllm` | NVIDIA | Yes | Yes | Model-dependent | Yes |
| `ollama` | Any | Via Ollama | Via Ollama | Via Ollama | Via Ollama |

---

## 10. Relevance to Your Pipeline

### For H100 Training (~/Projects/llm-training-pipeline)

1. **MXFP4 format is authoritative**: The `weights.py` LUT approach + `ldexp` scaling is the definitive way to decode GPT-OSS checkpoints. Your training pipeline must produce compatible output.

2. **SwiGLU has non-standard +1 bias**: `out = swish(glu_part) * (linear_part + 1)`. This differs from standard SwiGLU. Any fine-tuning must preserve this.

3. **Expert weights are PACKED**: `[num_experts, hidden_size, intermediate_size*2]` for MLP1 (num_experts=32 for 20B, 128 for 120B), confirming your MEMORY.md note. The Triton path transposes (`mT.contiguous()`) before MXFP4 quantization.

4. **Reinforcement fine-tuning uses GRPO + Unsloth**: The notebook provides a working template. Key insight: reward functions can be arbitrary Python, allowing custom evaluation beyond loss metrics.

5. **triton_kernels is the optimized kernel library**: For H100 training, `matmul_ogs` with `PrecisionConfig` and `FlexCtx` handles the MXFP4 compute. This is a separate installable package.

### For M2 Max Inference (MacLean AI / llama.cpp)

1. **Metal kernels confirm architecture details**: 64 Q heads, 8 KV heads, head_dim=64, MoE top-4 routing (32 experts for 20B, 128 for 120B), sliding_window=128 on even layers.

2. **MXFP4 dequant algorithm**: The Metal kernel's bit manipulation could inform llama.cpp Metal shader optimization if MXFP4 support is added.

3. **Fused MoE + SwiGLU**: OpenAI fuses dequantization directly into the matmul kernel (no intermediate buffer). llama.cpp could benefit from similar fusion.

4. **SDPA with learned sinks**: GPT-OSS attention has learned per-head bias terms (`sinks`) that act like persistent attention to a virtual token. This is unique to GPT-OSS and likely handled by llama.cpp's `--reasoning-format` flags.

5. **RoPE YaRN parameters**: scaling_factor=32, rope_theta=150000, ntk_alpha=1.0, ntk_beta=32.0. These are critical for correct extended context inference.

---

## 11. Dependencies

```toml
[core]
openai-harmony        # Harmony format encoding/decoding
tiktoken >= 0.9.0     # Tokenization
fastapi >= 0.116.1    # Responses API server
pydantic >= 2.11.7    # Data validation

[triton]
triton >= 3.4         # Triton JIT compiler
triton_kernels        # OpenAI's custom kernel library (installed from git)
safetensors >= 0.5.3  # Weight loading
torch >= 2.7.0        # PyTorch

[metal]
numpy, tqdm, safetensors, torch  # For model conversion scripts

[training (from notebook)]
unsloth              # Memory-efficient fine-tuning
trl == 0.22.2        # GRPO trainer
transformers == 4.56.2  # Must match exactly
bitsandbytes         # 4-bit quantization
```
