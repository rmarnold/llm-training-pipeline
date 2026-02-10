"""Text cleaning sub-package for the data processing pipeline.

Provides text cleaning, quality filtering, toxicity detection,
tokenization, and checkpoint management.

Import modules directly where needed:
    from pipeline_lib.text_cleaning.cleaning import parallel_clean_texts
    from pipeline_lib.text_cleaning.quality_filter import DatatroveQualityFilter
    from pipeline_lib.text_cleaning.toxicity import DataCleaner
    from pipeline_lib.text_cleaning.tokenization import parallel_tokenize
    from pipeline_lib.text_cleaning.checkpointing import CheckpointManager, StageManager
"""
