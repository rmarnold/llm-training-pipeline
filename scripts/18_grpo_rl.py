"""GRPO (Group Relative Policy Optimization) RL training with execution rewards.

Trains the model using rule-based rewards from actual code execution.
Supports multiple languages via evaluator dispatch:
- Rust: cargo check / cargo test / cargo clippy
- Python: syntax check / pytest / mypy / ruff

GRPO generates N completions per prompt, computes rewards, and optimizes
using group-relative advantages (no critic network needed).

Includes a long-context curriculum that progressively increases sequence
length during training.

Usage:
    python scripts/18_grpo_rl.py
    python scripts/18_grpo_rl.py --config configs/grpo.yaml
    python scripts/18_grpo_rl.py --config configs/grpo_python.yaml --language python
    python scripts/18_grpo_rl.py --max_steps 2000

Requires: pip install -e ".[gpt_oss]"
"""
import os

if "WANDB_MODE" not in os.environ:
    os.environ["WANDB_MODE"] = "offline"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json

import yaml

from pipeline_lib.unsloth_utils import load_unsloth_model, save_adapter, print_trainable_params
from pipeline_lib.evaluator_dispatch import compute_execution_reward


def load_tasks(task_source: str, num_tasks: int = 1000) -> list[dict]:
    """Load GRPO training tasks.

    Args:
        task_source: Path to tasks JSONL file or HF dataset.
        num_tasks: Maximum number of tasks to load.

    Returns:
        List of task dicts with 'description', 'tests', etc.
    """
    tasks = []

    if task_source.endswith(".jsonl"):
        with open(task_source) as f:
            for line in f:
                try:
                    tasks.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if len(tasks) >= num_tasks:
                    break
    else:
        from datasets import load_from_disk
        ds = load_from_disk(task_source)
        for i, example in enumerate(ds):
            if i >= num_tasks:
                break
            tasks.append(dict(example))

    return tasks


def get_curriculum_seq_length(step: int, curriculum: dict) -> int:
    """Get the sequence length for the current training step.

    Args:
        step: Current training step.
        curriculum: Curriculum config with schedule entries.

    Returns:
        Maximum sequence length for this step.
    """
    if not curriculum.get("enabled", False):
        return 32768

    schedule = curriculum.get("schedule", [])
    seq_length = 4096  # default

    for entry in schedule:
        if step <= entry["steps"]:
            seq_length = entry["seq_length"]
            break
        seq_length = entry["seq_length"]

    return seq_length


def compute_rewards_batch(
    completions: list[str],
    reward_config: dict,
    language: str = "rust",
) -> list[float]:
    """Compute execution-based rewards for a batch of completions.

    Args:
        completions: List of generated code strings.
        reward_config: Reward values from config.
        language: Target language for evaluator dispatch.

    Returns:
        List of float rewards.
    """
    rewards = []
    for code in completions:
        reward = compute_execution_reward(code, language=language, reward_config=reward_config)
        rewards.append(reward)
    return rewards


def train_grpo(config_path: str = "configs/grpo.yaml", cli_overrides: dict | None = None) -> None:
    """Train with GRPO using execution-based rewards.

    This implements a simplified GRPO training loop:
    1. Sample N completions per prompt
    2. Compute execution rewards (cargo check/test/clippy)
    3. Use group-relative advantages for policy update

    Args:
        config_path: Path to YAML config file.
        cli_overrides: Dict of CLI overrides.
    """
    if cli_overrides is None:
        cli_overrides = {}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print(f"GRPO RL Training: {config['run_name']}")
    print(f"{'='*60}")

    # Load model
    checkpoint = cli_overrides.get("checkpoint", config["model"]["checkpoint"])
    max_seq_length = config["model"].get("max_seq_length", 32768)

    print(f"\nLoading model from: {checkpoint}")
    model, tokenizer = load_unsloth_model(
        model_name=checkpoint,
        max_seq_length=max_seq_length,
        load_in_4bit=config["model"].get("load_in_4bit", True),
    )
    print_trainable_params(model)

    # Load tasks
    task_source = cli_overrides.get("task_source", config["data"]["task_source"])
    num_tasks = config["data"].get("num_tasks", 1000)

    print(f"\nLoading tasks from: {task_source}")
    tasks = load_tasks(task_source, num_tasks)
    print(f"  Loaded {len(tasks)} tasks")

    if not tasks:
        print("ERROR: No tasks loaded. Cannot train.")
        return

    # Language for evaluator dispatch (defaults to "rust" for backward compat)
    # Check top-level config first, then data section (grpo_python.yaml uses data.language)
    config_language = config.get("language", config.get("data", {}).get("language", "rust"))
    language = cli_overrides.get("language", config_language)

    # Training config
    max_steps = cli_overrides.get("max_steps", config["training"].get("max_steps", 5000))
    num_generations = config["training"].get("num_generations", 4)
    temperature = config["training"].get("temperature", 0.7)
    max_new_tokens = config["training"].get("max_new_tokens", 2048)
    reward_config = config.get("rewards", {})
    curriculum = config.get("curriculum", {})
    output_dir = cli_overrides.get("output_dir", config["checkpointing"]["output_dir"])
    save_steps = cli_overrides.get("save_steps", config["logging"].get("save_steps", 200))

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGRPO Configuration:")
    print(f"  Language: {language}")
    print(f"  Max steps: {max_steps}")
    print(f"  Generations per prompt: {num_generations}")
    print(f"  Temperature: {temperature}")
    print(f"  Curriculum: {'enabled' if curriculum.get('enabled') else 'disabled'}")

    # Try to use TRL's GRPOTrainer if available
    try:
        import trl  # noqa: F401 — availability check
        has_grpo = hasattr(trl, "GRPOTrainer")
    except ImportError:
        has_grpo = False

    if has_grpo:
        print("\nUsing TRL GRPOTrainer...")
        _train_with_trl_grpo(
            model, tokenizer, tasks, config, cli_overrides,
            output_dir, max_steps, reward_config, curriculum, language,
        )
    else:
        print("\nTRL GRPOTrainer not available. Using manual GRPO loop...")
        _train_manual_grpo(
            model, tokenizer, tasks, config,
            output_dir, max_steps, num_generations, temperature,
            max_new_tokens, reward_config, curriculum, save_steps, language,
        )

    # Save final adapter
    final_dir = os.path.join(output_dir, "final")
    save_adapter(model, final_dir, tokenizer)

    print(f"\nGRPO training complete!")
    print(f"  Adapter saved to: {final_dir}")


def _train_with_trl_grpo(
    model, tokenizer, tasks, config, cli_overrides,
    output_dir, max_steps, reward_config, _curriculum, language="rust",
):
    """Train using TRL's GRPOTrainer.

    Note: Curriculum sequence-length scheduling is not supported via
    TRL's GRPOTrainer. Use the manual GRPO loop for curriculum support.
    """
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset

    # Format tasks as prompts
    prompts = []
    for task in tasks:
        desc = task.get("description", "")
        tests = task.get("tests", "")
        prompt = desc
        if tests:
            prompt += f"\n\nTests:\n```rust\n{tests}\n```"
        prompts.append({"prompt": prompt})

    train_dataset = Dataset.from_list(prompts)

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=config["run_name"],
        max_steps=max_steps,
        learning_rate=cli_overrides.get("learning_rate", config["training"]["learning_rate"]),
        per_device_train_batch_size=cli_overrides.get(
            "per_device_train_batch_size",
            config["training"].get("per_device_train_batch_size", 1),
        ),
        gradient_accumulation_steps=cli_overrides.get(
            "gradient_accumulation_steps",
            config["training"].get("gradient_accumulation_steps", 8),
        ),
        bf16=config["training"].get("bf16", True),
        max_grad_norm=config["training"].get("max_grad_norm", 0.5),
        num_generations=config["training"].get("num_generations", 4),
        temperature=config["training"].get("temperature", 0.7),
        max_completion_length=config["training"].get("max_new_tokens", 2048),
        max_prompt_length=config["training"].get("max_prompt_length", 1024),
        logging_steps=cli_overrides.get("logging_steps", config["logging"].get("logging_steps", 1)),
        save_steps=cli_overrides.get("save_steps", config["logging"].get("save_steps", 200)),
        save_total_limit=config["logging"].get("save_total_limit", 3),
        report_to=config["logging"].get("report_to", ["tensorboard"]),
        seed=config["training"].get("seed", 42),
    )

    def reward_fn(completions, **_kwargs):
        """Compute execution rewards for generated completions."""
        return compute_rewards_batch(completions, reward_config, language=language)

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    trainer.train()


def _train_manual_grpo(
    model, tokenizer, tasks, config,
    output_dir, max_steps, num_generations, temperature,
    max_new_tokens, reward_config, curriculum, save_steps, language="rust",
):
    """Manual GRPO training loop (fallback when TRL GRPOTrainer unavailable)."""
    import torch
    from torch.optim import AdamW

    from dataset_formatters.harmony import encode_harmony_messages

    lr = config["training"]["learning_rate"]
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=config["training"].get("weight_decay", 0.0),
    )

    model.train()
    step = 0
    task_idx = 0

    log_path = os.path.join(output_dir, "grpo_log.jsonl")

    print(f"\nStarting manual GRPO loop (max_steps={max_steps})...")

    while step < max_steps:
        task = tasks[task_idx % len(tasks)]
        task_idx += 1

        # Get curriculum sequence length
        seq_length = get_curriculum_seq_length(step, curriculum)

        # Format prompt
        desc = task.get("description", "")
        tests = task.get("tests", "")
        prompt_content = desc
        if tests:
            prompt_content += f"\n\nTests:\n```rust\n{tests}\n```"

        prompt_text = encode_harmony_messages(
            [{"role": "user", "content": prompt_content}],
            developer_instructions="You are a Rust programming expert. Write correct, idiomatic code.",
        )

        # Generate N completions
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=seq_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        completions = []
        with torch.no_grad():
            for _ in range(num_generations):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                )
                completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                completions.append(completion)

        # Compute rewards
        rewards = compute_rewards_batch(completions, reward_config, language=language)

        # Group-relative advantage
        mean_reward = sum(rewards) / len(rewards) if rewards else 0
        std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5 if rewards else 1.0
        std_reward = max(std_reward, 1e-8)

        advantages = [(r - mean_reward) / std_reward for r in rewards]

        # Policy gradient update: use the best completion as a supervised signal
        # weighted by its advantage
        best_idx = max(range(len(rewards)), key=lambda i: rewards[i])
        if advantages[best_idx] > 0:
            best_text = prompt_text + completions[best_idx]
            targets = tokenizer(best_text, return_tensors="pt", truncation=True, max_length=seq_length)
            targets = {k: v.to(model.device) for k, v in targets.items()}

            outputs = model(**targets, labels=targets["input_ids"])
            loss = outputs.loss * advantages[best_idx]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(  # type: ignore[attr-defined]
                [p for p in model.parameters() if p.requires_grad],
                config["training"].get("max_grad_norm", 0.5),
            )
            optimizer.step()
            optimizer.zero_grad()

        step += 1

        # Logging
        if step % config["logging"].get("logging_steps", 1) == 0:
            log_entry = {
                "step": step,
                "mean_reward": mean_reward,
                "max_reward": max(rewards) if rewards else 0,
                "min_reward": min(rewards) if rewards else 0,
                "seq_length": seq_length,
            }
            print(f"  Step {step}/{max_steps} | reward: {mean_reward:.3f} (max: {max(rewards):.3f}) | seq_len: {seq_length}")

            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        # Save checkpoint
        if step % save_steps == 0:
            ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
            save_adapter(model, ckpt_dir, tokenizer)
            print(f"  Saved checkpoint: {ckpt_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GRPO RL training with execution rewards")
    parser.add_argument("--config", type=str, default="configs/grpo.yaml")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--logging_steps", type=int)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--task_source", type=str)
    parser.add_argument("--language", type=str, help="Evaluator language (rust, python)")
    args = parser.parse_args()

    cli_overrides = {}
    for key in ["checkpoint", "max_steps", "learning_rate",
                 "per_device_train_batch_size", "gradient_accumulation_steps",
                 "save_steps", "logging_steps", "output_dir", "task_source",
                 "language"]:
        val = getattr(args, key)
        if val is not None:
            cli_overrides[key] = val

    train_grpo(config_path=args.config, cli_overrides=cli_overrides)
