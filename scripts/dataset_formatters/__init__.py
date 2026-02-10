"""Dataset formatters for reasoning-focused SFT data.

Each module handles a category of datasets:
- math: GSM8K, Orca-Math, MetaMath, MathInstruct
- reasoning: CoT-Collection, LogiQA, ARC
- function_calling: Glaive, Hermes, Gorilla, ToolBench
- general: Alpaca, OpenOrca, OASST
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
]
