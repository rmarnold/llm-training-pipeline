from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import yaml

def format_conversation(example, tokenizer):
    """Format multi-turn conversations with special tokens"""
    conversation = []

    if "messages" in example:
        # ShareGPT format
        for msg in example["messages"]:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                conversation.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                conversation.append(f"<|assistant|>\n{content}<|endoftext|>")

    elif "instruction" in example:
        # Alpaca format
        instruction = example["instruction"]
        response = example["output"]

        if example.get("input"):
            instruction = f"{instruction}\n{example['input']}"

        conversation.append(f"<|user|>\n{instruction}\n")
        conversation.append(f"<|assistant|>\n{response}<|endoftext|>")

    text = "".join(conversation)
    return {"text": text}

def prepare_sft_dataset():
    tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer")

    with open("configs/data_sources.yaml") as f:
        config = yaml.safe_load(f)

    datasets = []

    for ds_config in config["datasets"]["instruction_tuning"]:
        print(f"Loading {ds_config['name']}...")

        ds = load_dataset(ds_config["source"], split="train")

        # Format conversations
        ds = ds.map(
            lambda x: format_conversation(x, tokenizer),
            remove_columns=ds.column_names
        )

        # Apply sampling weight
        sample_size = int(len(ds) * ds_config["weight"])
        ds = ds.shuffle(seed=42).select(range(sample_size))

        datasets.append(ds)

    # Combine all datasets
    combined_ds = concatenate_datasets(datasets)
    combined_ds = combined_ds.shuffle(seed=42)

    # Split train/val
    split = combined_ds.train_test_split(test_size=0.05, seed=42)

    # Save
    split["train"].save_to_disk("data/sft/train")
    split["test"].save_to_disk("data/sft/val")

    print(f"✓ SFT data prepared: {len(split['train'])} train, {len(split['test'])} val")

if __name__ == "__main__":
    prepare_sft_dataset()
