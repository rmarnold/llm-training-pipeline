"""GPU-accelerated toxicity detection with batched processing."""
from __future__ import annotations

from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import torch
from datasketch import MinHash, MinHashLSH
from detoxify import Detoxify
from tqdm import tqdm

from .tokenization import parallel_tokenize


class DataCleaner:
    """GPU-accelerated data cleaner with batched processing."""

    # Class-level model cache to avoid reloading
    _model_cache: dict = {}
    _model_warmed_up: set = set()
    _tokenizer_cache: dict = {}  # Separate tokenizer cache (no CUDA needed)

    def __init__(self, toxicity_threshold: float = 0.7, use_gpu: bool = True, batch_size: int = None, lazy_load: bool = True):
        """Initialize DataCleaner.

        Args:
            toxicity_threshold: Threshold for toxicity detection
            use_gpu: Use GPU for inference
            batch_size: Batch size for inference (auto-tuned if None)
            lazy_load: If True, defer model loading until first inference (allows parallel tokenization)
        """
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.toxicity_threshold = toxicity_threshold
        self.use_fp16 = False  # Will be set based on GPU capabilities
        self.toxicity_model = None  # Will be loaded lazily
        self._model_loaded = False

        # Auto-tune batch size based on GPU memory
        if batch_size is None:
            if self.device == 'cuda':
                try:
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                    gpu_name = torch.cuda.get_device_name(0)

                    # A100/H100 have excellent fp16/bf16 performance
                    if 'A100' in gpu_name or 'H100' in gpu_name or gpu_mem >= 35:
                        self.use_fp16 = True

                    # Inference batch size (capped at 256 for optimal GPU utilization)
                    if gpu_mem >= 35:  # A100 or similar
                        batch_size = 256
                    elif gpu_mem >= 20:  # RTX 3090/4090
                        batch_size = 192
                    else:
                        batch_size = 128

                    print(f"    Auto-tuned toxicity: batch_size={batch_size}, fp16={self.use_fp16} for {gpu_name} ({gpu_mem:.0f}GB)")
                except Exception as e:
                    print(f"    Warning: Could not auto-tune GPU settings: {e}")
                    batch_size = 128
            else:
                batch_size = 64  # CPU is slower, use smaller batches

        self.batch_size = batch_size
        self.lsh = MinHashLSH(threshold=0.85, num_perm=128)

        # Load model immediately if not lazy loading
        if not lazy_load:
            self._load_model()

    def _load_model(self):
        """Load the toxicity model (call this after tokenization to allow parallel tokenization)."""
        if self._model_loaded:
            return

        # Use cached model if available (avoid re-downloading)
        cache_key = f"detoxify_{self.device}"
        if cache_key not in DataCleaner._model_cache:
            print(f"Loading toxicity model on {self.device}...")
            model = Detoxify('original', device=self.device)

            # Verify model is actually on the correct device
            if self.device == 'cuda':
                try:
                    param_device = next(model.model.parameters()).device
                    if param_device.type != 'cuda':
                        print(f"    Warning: Model loaded on {param_device}, moving to CUDA...")
                        model.model = model.model.cuda()
                    else:
                        print(f"    Model verified on {param_device}")
                except Exception as e:
                    print(f"    Warning: Could not verify model device: {e}")

            # Convert to fp16 for faster inference on A100/H100
            if self.use_fp16 and self.device == 'cuda':
                try:
                    model.model.half()
                    print(f"    Converted model to fp16 for faster inference")
                except Exception as e:
                    print(f"    Warning: Could not convert to fp16: {e}")

            DataCleaner._model_cache[cache_key] = model

        self.toxicity_model = DataCleaner._model_cache[cache_key]
        self._model_loaded = True

        # Warmup GPU with a small batch (compiles CUDA kernels)
        if self.device == 'cuda' and cache_key not in DataCleaner._model_warmed_up:
            self._warmup_gpu()
            DataCleaner._model_warmed_up.add(cache_key)

    @classmethod
    def get_tokenizer(cls):
        """Get the tokenizer without loading the full model (no CUDA initialization)."""
        if 'tokenizer' not in cls._tokenizer_cache:
            from transformers import AutoTokenizer
            cls._tokenizer_cache['tokenizer'] = AutoTokenizer.from_pretrained('bert-base-uncased')
        return cls._tokenizer_cache['tokenizer']

    def _warmup_gpu(self):
        """Warmup GPU by running a small batch through the model."""
        print(f"    Warming up GPU with test batch...")
        warmup_texts = ["This is a test sentence for GPU warmup."] * 16
        try:
            model = self.toxicity_model.model
            tokenizer = self.toxicity_model.tokenizer

            inputs = tokenizer(
                warmup_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                if self.use_fp16:
                    with torch.amp.autocast('cuda'):
                        _ = model(**inputs)
                else:
                    _ = model(**inputs)

            torch.cuda.synchronize()
            print(f"    GPU warmup complete")
        except Exception as e:
            print(f"    Warning: GPU warmup failed: {e}")

    def clean_text(self, text: str) -> str:
        """Clean a single text document."""
        if pd.isna(text):
            return ""
        from ftfy import fix_text
        import re
        text = fix_text(str(text))
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]', text)
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
        return text.strip()

    def filter_quality_batch(self, texts: list[str], min_words: int = 50, max_words: int = 10000) -> list[bool]:
        """Batch quality filtering."""
        results = []
        for text in texts:
            word_count = len(text.split())
            if word_count < min_words or word_count > max_words:
                results.append(False)
                continue
            unique_chars = len(set(text.lower()))
            if unique_chars < 20:
                results.append(False)
                continue
            results.append(True)
        return results

    def is_toxic_batch(self, texts: list[str], show_progress: bool = True) -> list[bool]:
        """Batch toxicity detection with progress bar.

        Optimizations for A100/H100:
        - Pre-tokenize ALL texts first using parallel workers (10-50x faster)
        - Lazy model loading: tokenization runs BEFORE CUDA initialization
        - Uses fp16 (half precision) for 2-3x faster inference
        - Non-blocking data transfer with pin_memory + non_blocking=True
        """
        if not texts:
            return []

        # STAGE 1: Pre-tokenize texts using multiprocessing (bypasses GIL)
        tokenizer = DataCleaner.get_tokenizer()

        n_cpus = cpu_count()
        if show_progress:
            print(f"    Pre-tokenizing {len(texts):,} texts using {n_cpus} CPU cores...")

        all_inputs = parallel_tokenize(
            texts,
            tokenizer,
            max_length=512,
            num_proc=n_cpus,
            batch_size=1000,
            show_progress=show_progress,
        )

        if show_progress:
            print(f"    Tokenization complete. Loading model and running GPU inference...")

        # STAGE 2: Load model AFTER tokenization (this initializes CUDA)
        self._load_model()
        model = self.toxicity_model.model

        # Pin memory for faster GPU transfer
        if self.device == 'cuda':
            all_inputs = {k: v.pin_memory() for k, v in all_inputs.items()}

        # Optimal batch sizes
        infer_batch_size = min(self.batch_size, 256)
        n_batches = (len(texts) + infer_batch_size - 1) // infer_batch_size

        if show_progress:
            desc = f"    Toxicity ({self.device}"
            if self.use_fp16:
                desc += ", fp16"
            desc += f", bs={infer_batch_size}, parallel_tok)"

        # STAGE 2: GPU inference on pre-tokenized batches
        all_results = []

        iterator = range(0, len(texts), infer_batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=desc, unit="batch", total=n_batches)

        for i in iterator:
            batch_inputs = {
                k: v[i:i + infer_batch_size].to(self.device, non_blocking=True)
                for k, v in all_inputs.items()
            }

            with torch.no_grad():
                if self.use_fp16 and self.device == 'cuda':
                    with torch.amp.autocast('cuda'):
                        outputs = model(**batch_inputs)
                else:
                    outputs = model(**batch_inputs)

            predictions = torch.sigmoid(outputs.logits).cpu().numpy()

            batch_size = predictions.shape[0]
            for j in range(batch_size):
                is_toxic = any(predictions[j, k] > self.toxicity_threshold for k in range(predictions.shape[1]))
                all_results.append(is_toxic)

        return all_results

    def compute_minhash(self, text: str) -> MinHash:
        """Compute MinHash for deduplication."""
        m = MinHash(num_perm=128)
        for word in text.split():
            m.update(word.encode('utf8'))
        return m

    def deduplicate_batch(self, texts: list[str], doc_ids: list[str], show_progress: bool = True) -> list[bool]:
        """Batch deduplication with progress bar."""
        keep_mask = []
        iterator = zip(texts, doc_ids)
        if show_progress:
            iterator = tqdm(list(iterator), desc="    Deduplicating", unit="doc")

        for text, doc_id in iterator:
            m = self.compute_minhash(text)
            if self.lsh.query(m):
                keep_mask.append(False)
            else:
                self.lsh.insert(doc_id, m)
                keep_mask.append(True)
        return keep_mask
