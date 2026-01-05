import torch
from transformers import AutoModelForCausalLM
import nvidia_smi

def profile_training_memory():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    model = AutoModelForCausalLM.from_pretrained(
        "checkpoints/init",
        torch_dtype=torch.bfloat16
    ).cuda()

    # Simulate forward pass
    batch_size = 1
    seq_len = 2048
    input_ids = torch.randint(0, 32000, (batch_size, seq_len)).cuda()

    torch.cuda.reset_peak_memory_stats()

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    loss.backward()

    # Memory stats
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    peak_allocated = torch.cuda.max_memory_allocated() / 1e9
    total_vram = info.total / 1e9
    used_vram = info.used / 1e9

    print("=" * 60)
    print("VRAM Profile")
    print("=" * 60)
    print(f"Total VRAM: {total_vram:.2f} GB")
    print(f"Used VRAM: {used_vram:.2f} GB ({used_vram/total_vram*100:.1f}%)")
    print(f"Peak Allocated: {peak_allocated:.2f} GB")
    print(f"Model Parameters: {model.num_parameters() / 1e9:.2f}B")
    print(f"Batch Size: {batch_size}")
    print(f"Sequence Length: {seq_len}")
    print("=" * 60)

    # Recommendations
    utilization = used_vram / total_vram
    if utilization < 0.85:
        print("⚠️  VRAM under-utilized. Recommendations:")
        print("   - Increase batch size or gradient accumulation")
        print("   - Increase sequence length")
        print(f"   - Target: >{total_vram * 0.90:.1f}GB ({90}% utilization)")
    elif utilization > 0.95:
        print("⚠️  Risk of OOM. Recommendations:")
        print("   - Enable gradient checkpointing")
        print("   - Reduce batch size")
        print("   - Use FSDP/ZeRO-3")
    else:
        print("✓ VRAM utilization optimal (85-95%)")

if __name__ == "__main__":
    profile_training_memory()
