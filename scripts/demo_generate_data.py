"""Generate synthetic data for demo/testing"""
import os
import json
import random
import pandas as pd

def generate_synthetic_pretraining_data(num_samples=100):
    """Generate synthetic pretraining text"""
    os.makedirs("data/raw", exist_ok=True)

    templates = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Deep learning models require large amounts of training data.",
        "Natural language processing enables computers to understand text.",
    ]

    documents = []
    for i in range(num_samples):
        # Generate random text by combining templates
        text = " ".join([random.choice(templates) for _ in range(random.randint(5, 15))])
        documents.append({"text": text, "id": i})

    df = pd.DataFrame(documents)
    df.to_parquet("data/raw/pretraining_demo.parquet")
    print(f"✓ Generated {num_samples} pretraining documents")
    return df

def generate_synthetic_instruction_data(num_samples=50):
    """Generate synthetic instruction-following data"""

    qa_pairs = [
        ("What is Python?", "Python is a high-level programming language known for its simplicity and readability."),
        ("How do you define a function in Python?", "Use the 'def' keyword followed by the function name and parameters."),
        ("What is machine learning?", "Machine learning is a method of data analysis that automates analytical model building."),
        ("Explain a for loop", "A for loop iterates over a sequence of elements, executing code for each element."),
        ("What is a neural network?", "A neural network is a series of algorithms that recognize patterns in data."),
    ]

    samples = []
    for i in range(num_samples):
        q, a = random.choice(qa_pairs)
        samples.append({
            "instruction": q,
            "output": a,
            "input": ""
        })

    df = pd.DataFrame(samples)
    df.to_parquet("data/raw/instruction_tuning_demo.parquet")
    print(f"✓ Generated {num_samples} instruction samples")
    return df

def generate_synthetic_preference_data(num_samples=30):
    """Generate synthetic preference pairs for DPO"""

    prompts = [
        "Explain what Python is",
        "How do you write a function",
        "What is machine learning",
    ]

    good_responses = [
        "Python is a versatile programming language.",
        "To write a function, use the def keyword.",
        "Machine learning enables computers to learn from data.",
    ]

    bad_responses = [
        "I don't know.",
        "Functions are complicated.",
        "That's too hard to explain.",
    ]

    samples = []
    for i in range(num_samples):
        prompt = random.choice(prompts)
        chosen = random.choice(good_responses)
        rejected = random.choice(bad_responses)

        samples.append({
            "chosen": f"Human: {prompt}\n\nAssistant: {chosen}",
            "rejected": f"Human: {prompt}\n\nAssistant: {rejected}"
        })

    df = pd.DataFrame(samples)
    df.to_parquet("data/raw/preference_data_demo.parquet")
    print(f"✓ Generated {num_samples} preference pairs")
    return df

def create_manifest():
    """Create data manifest"""
    manifest = {
        "pretraining_demo": {
            "path": "data/raw/pretraining_demo.parquet",
            "license": "Synthetic",
            "rows": 100
        },
        "instruction_tuning_demo": {
            "path": "data/raw/instruction_tuning_demo.parquet",
            "license": "Synthetic",
            "rows": 50
        },
        "preference_data_demo": {
            "path": "data/raw/preference_data_demo.parquet",
            "license": "Synthetic",
            "rows": 30
        }
    }

    import yaml
    with open("data/raw/manifest.yaml", "w") as f:
        yaml.dump(manifest, f)

    print("✓ Created data manifest")

if __name__ == "__main__":
    print("Generating synthetic demo data...")
    generate_synthetic_pretraining_data(100)
    generate_synthetic_instruction_data(50)
    generate_synthetic_preference_data(30)
    create_manifest()
    print("\n✓ All demo data generated successfully!")
