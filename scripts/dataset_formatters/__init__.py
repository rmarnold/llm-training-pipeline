"""Dataset formatters for reasoning-focused SFT data.

Each module handles a category of datasets:
- math: GSM8K, Orca-Math, MetaMath, MathInstruct
- reasoning: CoT-Collection, LogiQA, ARC
- function_calling: Glaive, Hermes, Gorilla, ToolBench
- general: Alpaca, OpenOrca, OASST
- harmony: GPT-OSS Harmony format (code, agent, preference, debug)
"""
from dataset_formatters.math import (
    format_gsm8k,
    format_orca_math,
    format_metamath,
    format_mathinstruct,
)
from dataset_formatters.reasoning import (
    format_cot_collection,
    format_logiqa,
    format_arc,
)
from dataset_formatters.function_calling import (
    format_glaive_function_calling,
    format_hermes_function_calling,
    format_gorilla,
    format_toolbench,
)
from dataset_formatters.general import (
    format_alpaca,
    format_openorca,
    format_oasst,
)
from dataset_formatters.harmony import (
    format_harmony_code,
    format_harmony_completion,
    format_harmony_agent,
    format_harmony_preference,
    format_harmony_task,
    format_harmony_debug,
)

FORMAT_HANDLERS = {
    "gsm8k": format_gsm8k,
    "orca-math": format_orca_math,
    "openorca": format_openorca,
    "cot-collection": format_cot_collection,
    "mathinstruct": format_mathinstruct,
    "glaive": format_glaive_function_calling,
    "hermes": format_hermes_function_calling,
    "gorilla": format_gorilla,
    "logiqa": format_logiqa,
    "arc": format_arc,
    "alpaca": format_alpaca,
    "oasst": format_oasst,
    "metamath": format_metamath,
    "toolbench": format_toolbench,
    # Harmony formatters for GPT-OSS
    "harmony_code": format_harmony_code,
    "harmony_completion": format_harmony_completion,
    "harmony_agent": format_harmony_agent,
    "harmony_preference": format_harmony_preference,
    "harmony_task": format_harmony_task,
    "harmony_debug": format_harmony_debug,
}

__all__ = [
    "FORMAT_HANDLERS",
    "format_gsm8k",
    "format_orca_math",
    "format_metamath",
    "format_mathinstruct",
    "format_cot_collection",
    "format_logiqa",
    "format_arc",
    "format_glaive_function_calling",
    "format_hermes_function_calling",
    "format_gorilla",
    "format_toolbench",
    "format_alpaca",
    "format_openorca",
    "format_oasst",
    "format_harmony_code",
    "format_harmony_completion",
    "format_harmony_agent",
    "format_harmony_preference",
    "format_harmony_task",
    "format_harmony_debug",
]
