"""Text cleaning sub-package for the data processing pipeline.

Provides text cleaning, quality filtering, toxicity detection,
tokenization, and checkpoint management.

Import modules directly where needed:
    from .cleaning import parallel_clean_texts
    from .quality_filter import DatatroveQualityFilter
    from .toxicity import DataCleaner
    from .tokenization import parallel_tokenize
    from .checkpointing import CheckpointManager, StageManager
"""
