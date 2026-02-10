"""Reasoning dataset formatters: CoT-Collection, LogiQA, ARC."""
from __future__ import annotations


def format_cot_collection(example):
    """Format CoT-Collection chain-of-thought examples.

    CoT-Collection format: {"source": "...", "rationale": "...", "target": "...", "task": "..."}
    """
    source = example.get("source", "")
    rationale = example.get("rationale", "")
    target = example.get("target", "")

    # Combine rationale (reasoning) with target (answer)
    if rationale:
        response = f"{rationale}\n\nTherefore, the answer is: {target}"
    else:
        response = target

    return {
        "text": f"<|user|>\n{source}\n<|assistant|>\n{response}<|endoftext|>"
    }


def format_logiqa(example):
    """Format LogiQA logical reasoning problems.

    LogiQA format: {"context": "...", "question": "...", "answers": [...], "label": int}
    """
    context = example.get("context", "")
    question = example.get("question", "")
    answers = example.get("answers", [])
    label = example.get("label", 0)

    # Format as multiple choice with reasoning
    choices_text = "\n".join([f"{chr(65+i)}. {ans}" for i, ans in enumerate(answers)])
    correct_answer = chr(65 + label)
    correct_text = answers[label] if label < len(answers) else ""

    prompt = f"Context: {context}\n\nQuestion: {question}\n\nChoices:\n{choices_text}\n\nThink through this step by step and select the correct answer."

    response = f"Let me analyze this logically.\n\nGiven the context, I need to evaluate each option:\n\nThe correct answer is {correct_answer}. {correct_text}"

    return {
        "text": f"<|user|>\n{prompt}\n<|assistant|>\n{response}<|endoftext|>"
    }


def format_arc(example):
    """Format ARC Challenge dataset.

    ARC format: {"question": "...", "choices": {"text": [...], "label": [...]}, "answerKey": "..."}
    """
    question = example.get("question", "")
    choices = example.get("choices", {})
    answer_key = example.get("answerKey", "")

    # Format choices
    choice_texts = choices.get("text", [])
    choice_labels = choices.get("label", [])

    choices_str = "\n".join([f"{label}. {text}" for label, text in zip(choice_labels, choice_texts)])

    # Find correct answer text
    correct_idx = choice_labels.index(answer_key) if answer_key in choice_labels else 0
    correct_text = choice_texts[correct_idx] if correct_idx < len(choice_texts) else ""

    prompt = f"{question}\n\nChoices:\n{choices_str}"
    response = f"The answer is {answer_key}. {correct_text}"

    return {
        "text": f"<|user|>\n{prompt}\n<|assistant|>\n{response}<|endoftext|>"
    }
