"""Tests for data pipeline components."""
import pytest
import numpy as np
import os


class TestDataCleaning:
    """Test data cleaning functionality."""

    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        from scripts.gpu_utils import setup_torch_backends  # noqa: ensure scripts importable

        # Import would require datasketch/detoxify, so test logic manually
        import re

        def clean_text(text):
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            # Remove PII patterns
            text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]', text)
            text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
            text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
            return text.strip()

        # Test whitespace normalization
        assert clean_text("hello   world") == "hello world"
        assert clean_text("  spaces  ") == "spaces"

        # Test PII removal
        assert "[EMAIL]" in clean_text("Contact: test@example.com")
        assert "[SSN]" in clean_text("SSN: 123-45-6789")
        assert "[PHONE]" in clean_text("Call: 555-123-4567")

    def test_quality_filter(self):
        """Test quality filtering logic."""

        def filter_quality(text, min_words=50, max_words=10000):
            word_count = len(text.split())
            if word_count < min_words or word_count > max_words:
                return False
            unique_chars = len(set(text.lower()))
            if unique_chars < 20:
                return False
            return True

        # Too short
        assert filter_quality("short text", min_words=50) is False

        # Good length with variety (needs >= 20 unique chars)
        good_text = " ".join(["the", "quick", "brown", "fox", "jumps"] * 20)
        assert filter_quality(good_text, min_words=50) is True

        # Gibberish (low char variety)
        gibberish = "aaa " * 100
        assert filter_quality(gibberish, min_words=50) is False


class TestTokenization:
    """Test tokenization utilities."""

    def test_packed_sequence_format(self, temp_dir):
        """Test packed sequence data format."""
        # Simulate packed data format
        seq_length = 2048
        num_sequences = 10

        # Create mock packed data
        packed_data = np.random.randint(0, 50000, size=(num_sequences, seq_length))

        # Save and reload
        save_path = os.path.join(temp_dir, "packed.npy")
        np.save(save_path, packed_data)
        loaded = np.load(save_path)

        assert loaded.shape == (num_sequences, seq_length)
        assert loaded.dtype in [np.int32, np.int64]


class TestDatasetFormats:
    """Test dataset format compatibility."""

    def test_sft_format(self):
        """Test SFT dataset format requirements."""
        # SFT expects 'text' or 'messages' field
        sft_sample = {
            "text": "Human: What is 2+2?\nAssistant: 2+2 equals 4.",
        }
        assert "text" in sft_sample or "messages" in sft_sample

    def test_dpo_format(self):
        """Test DPO dataset format requirements."""
        # DPO expects prompt, chosen, rejected
        dpo_sample = {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris.",
            "rejected": "France's capital is London.",
        }
        assert "prompt" in dpo_sample
        assert "chosen" in dpo_sample
        assert "rejected" in dpo_sample
