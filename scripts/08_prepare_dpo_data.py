from datasets import load_dataset

def prepare_dpo_dataset():
    """Prepare preference pairs for DPO training"""

    # Load HH-RLHF
    hh_rlhf = load_dataset("Anthropic/hh-rlhf", split="train")

    # Format for DPO: {prompt, chosen, rejected}
    def format_preference(example):
        return {
            "prompt": example["chosen"].split("Assistant:")[0] + "Assistant:",
            "chosen": example["chosen"].split("Assistant:")[1].strip(),
            "rejected": example["rejected"].split("Assistant:")[1].strip(),
        }

    dpo_dataset = hh_rlhf.map(format_preference)

    # Add safety filters
    from detoxify import Detoxify
    toxicity_model = Detoxify('original')

    def is_safe(example):
        chosen_toxic = toxicity_model.predict(example["chosen"])
        rejected_toxic = toxicity_model.predict(example["rejected"])

        # Keep if chosen is less toxic than rejected
        return chosen_toxic["toxicity"] < rejected_toxic["toxicity"]

    dpo_dataset = dpo_dataset.filter(is_safe)

    # Split
    split = dpo_dataset.train_test_split(test_size=0.05, seed=42)
    split["train"].save_to_disk("data/dpo/train")
    split["test"].save_to_disk("data/dpo/val")

    print(f"✓ DPO data prepared: {len(split['train'])} pairs")

if __name__ == "__main__":
    prepare_dpo_dataset()
