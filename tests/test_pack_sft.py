"""Tests for pipeline_lib.pack_sft — pre-packing SFT datasets."""
import os
import tempfile
import shutil

import pytest
from datasets import Dataset

from pipeline_lib.pack_sft import pack_sft_dataset, _flush_chunk


class _FakeTokenizer:
    """Minimal tokenizer stub for testing (char-level: each char -> ord)."""

    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        # Simple char-level encoding: each char becomes its ordinal
        return [ord(c) for c in text]


@pytest.fixture
def fake_tokenizer():
    return _FakeTokenizer()


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def _make_dataset(texts, path, column="text"):
    """Create a small HF Dataset on disk."""
    ds = Dataset.from_dict({column: texts})
    ds.save_to_disk(path)
    return path


class TestBasicPacking:
    """Test basic concatenation and chunking."""

    def test_packs_short_examples(self, fake_tokenizer, temp_dir):
        """Short examples should be concatenated into fewer packed sequences."""
        src = os.path.join(temp_dir, "src")
        out = os.path.join(temp_dir, "out")

        # 10 examples of 5 chars each => 50 tokens total
        # With seq_len=20, expect ceil(50/20) = 3 packed sequences (greedy)
        texts = ["hello"] * 10
        _make_dataset(texts, src)

        result = pack_sft_dataset(
            dataset_path=src,
            tokenizer=fake_tokenizer,
            seq_len=20,
            output_path=out,
            val_fraction=0,
        )

        assert result["source_examples"] == 10
        assert result["packed_sequences"] <= 4  # greedy packing
        assert result["packed_sequences"] >= 3
        assert result["compression_ratio"] > 1.0

    def test_single_example_fits(self, fake_tokenizer, temp_dir):
        """A single example shorter than seq_len produces one packed sequence."""
        src = os.path.join(temp_dir, "src")
        out = os.path.join(temp_dir, "out")

        _make_dataset(["abc"], src)

        result = pack_sft_dataset(
            dataset_path=src,
            tokenizer=fake_tokenizer,
            seq_len=10,
            output_path=out,
            val_fraction=0,
        )

        assert result["packed_sequences"] == 1

        # Verify structure
        from datasets import load_from_disk
        ds = load_from_disk(result["train_path"])
        assert len(ds[0]["input_ids"]) == 10
        assert len(ds[0]["attention_mask"]) == 10
        assert len(ds[0]["labels"]) == 10


class TestLabelMasking:
    """Test that padding positions get labels=-100."""

    def test_pad_positions_have_neg100_labels(self, fake_tokenizer, temp_dir):
        """Pad tokens should have labels=-100, real tokens should have actual IDs."""
        src = os.path.join(temp_dir, "src")
        out = os.path.join(temp_dir, "out")

        # 3 chars -> 3 real tokens, rest is padding
        _make_dataset(["abc"], src)

        result = pack_sft_dataset(
            dataset_path=src,
            tokenizer=fake_tokenizer,
            seq_len=10,
            output_path=out,
            val_fraction=0,
        )

        from datasets import load_from_disk
        ds = load_from_disk(result["train_path"])
        row = ds[0]

        # First 3 positions: real tokens
        assert row["labels"][:3] == [ord("a"), ord("b"), ord("c")]
        assert row["attention_mask"][:3] == [1, 1, 1]

        # Remaining 7 positions: padding
        assert row["labels"][3:] == [-100] * 7
        assert row["attention_mask"][3:] == [0] * 7

    def test_flush_chunk_correctness(self):
        """Direct test of _flush_chunk helper."""
        ids, masks, labels = [], [], []
        _flush_chunk([10, 20, 30], 5, 0, ids, masks, labels)

        assert ids == [[10, 20, 30, 0, 0]]
        assert masks == [[1, 1, 1, 0, 0]]
        assert labels == [[10, 20, 30, -100, -100]]

    def test_flush_chunk_with_boundary_offsets(self):
        """Boundary offsets should mask the first token of each subsequent example."""
        ids, masks, labels = [], [], []
        # Simulate chunk: example A=[10, 20], example B=[30, 40], pad
        _flush_chunk([10, 20, 30, 40], 6, 0, ids, masks, labels, boundary_offsets=[2])

        assert ids == [[10, 20, 30, 40, 0, 0]]
        assert masks == [[1, 1, 1, 1, 0, 0]]
        # Position 2 is the boundary — label should be -100
        assert labels == [[10, 20, -100, 40, -100, -100]]

    def test_cross_example_boundary_masking(self, fake_tokenizer, temp_dir):
        """When multiple examples are packed into one chunk, the first token of
        each subsequent example should have labels=-100 to prevent cross-example
        contamination."""
        src = os.path.join(temp_dir, "src")
        out = os.path.join(temp_dir, "out")

        # 3 examples of 3 chars each = 9 tokens total
        # With seq_len=20, all 3 fit in one chunk:
        #   [a, b, c, d, e, f, g, h, i, PAD, ...]
        #    ex1      ex2      ex3
        #   Boundaries at positions 3 and 6
        _make_dataset(["abc", "def", "ghi"], src)

        result = pack_sft_dataset(
            dataset_path=src,
            tokenizer=fake_tokenizer,
            seq_len=20,
            output_path=out,
            val_fraction=0,
        )

        assert result["packed_sequences"] == 1

        from datasets import load_from_disk
        ds = load_from_disk(result["train_path"])
        row = ds[0]

        # Example 1 tokens: all have real labels (no boundary before first example)
        assert row["labels"][0] == ord("a")
        assert row["labels"][1] == ord("b")
        assert row["labels"][2] == ord("c")

        # Position 3: boundary (first token of example 2) — must be -100
        assert row["labels"][3] == -100, (
            f"Cross-example boundary at position 3 should be -100, got {row['labels'][3]}"
        )
        # Rest of example 2 has real labels
        assert row["labels"][4] == ord("e")
        assert row["labels"][5] == ord("f")

        # Position 6: boundary (first token of example 3) — must be -100
        assert row["labels"][6] == -100, (
            f"Cross-example boundary at position 6 should be -100, got {row['labels'][6]}"
        )
        # Rest of example 3 has real labels
        assert row["labels"][7] == ord("h")
        assert row["labels"][8] == ord("i")

        # Padding positions: all -100
        assert row["labels"][9:] == [-100] * 11

        # input_ids should still have all real tokens (boundaries only affect labels)
        assert row["input_ids"][:9] == [ord(c) for c in "abcdefghi"]

    def test_single_example_no_boundary_masking(self, fake_tokenizer, temp_dir):
        """A chunk with only one example should have no boundary masking."""
        src = os.path.join(temp_dir, "src")
        out = os.path.join(temp_dir, "out")

        # One long example that fills most of the chunk
        _make_dataset(["abcdefgh"], src)

        result = pack_sft_dataset(
            dataset_path=src,
            tokenizer=fake_tokenizer,
            seq_len=10,
            output_path=out,
            val_fraction=0,
        )

        from datasets import load_from_disk
        ds = load_from_disk(result["train_path"])
        row = ds[0]

        # All 8 real positions should have real labels (no boundaries)
        assert row["labels"][:8] == [ord(c) for c in "abcdefgh"]
        assert row["labels"][8:] == [-100, -100]


class TestTruncation:
    """Test handling of over-length examples."""

    def test_long_example_truncated(self, fake_tokenizer, temp_dir):
        """Examples longer than seq_len should be truncated, not crash."""
        src = os.path.join(temp_dir, "src")
        out = os.path.join(temp_dir, "out")

        # 20 chars > seq_len=10
        _make_dataset(["a" * 20], src)

        result = pack_sft_dataset(
            dataset_path=src,
            tokenizer=fake_tokenizer,
            seq_len=10,
            output_path=out,
            val_fraction=0,
        )

        assert result["truncated"] == 1
        assert result["packed_sequences"] == 1

        from datasets import load_from_disk
        ds = load_from_disk(result["train_path"])
        assert len(ds[0]["input_ids"]) == 10

    def test_mixed_lengths_with_truncation(self, fake_tokenizer, temp_dir):
        """Mix of short and over-length examples."""
        src = os.path.join(temp_dir, "src")
        out = os.path.join(temp_dir, "out")

        texts = ["ab", "a" * 15, "cd"]  # 2, 15 (truncated to 8), 2
        _make_dataset(texts, src)

        result = pack_sft_dataset(
            dataset_path=src,
            tokenizer=fake_tokenizer,
            seq_len=8,
            output_path=out,
            val_fraction=0,
        )

        assert result["truncated"] == 1
        assert result["source_examples"] == 3


class TestCompressionRatio:
    """Test that packing achieves expected compression."""

    def test_high_compression_for_short_examples(self, fake_tokenizer, temp_dir):
        """Very short examples relative to seq_len should compress well."""
        src = os.path.join(temp_dir, "src")
        out = os.path.join(temp_dir, "out")

        # 100 examples, each 10 chars -> 1000 tokens total
        # seq_len=200 -> ceil(1000/200) = 5 packed sequences
        # compression = 100/5 = 20x
        texts = ["abcdefghij"] * 100
        _make_dataset(texts, src)

        result = pack_sft_dataset(
            dataset_path=src,
            tokenizer=fake_tokenizer,
            seq_len=200,
            output_path=out,
            val_fraction=0,
        )

        assert result["compression_ratio"] >= 15.0  # Conservative bound
        assert result["utilization_pct"] >= 90.0

    def test_utilization_report(self, fake_tokenizer, temp_dir):
        """Utilization should be close to 100% for perfectly divisible data."""
        src = os.path.join(temp_dir, "src")
        out = os.path.join(temp_dir, "out")

        # 4 examples of exactly 5 tokens = 20 tokens, seq_len=10 -> 2 sequences
        texts = ["abcde"] * 4
        _make_dataset(texts, src)

        result = pack_sft_dataset(
            dataset_path=src,
            tokenizer=fake_tokenizer,
            seq_len=10,
            output_path=out,
            val_fraction=0,
        )

        assert result["packed_sequences"] == 2
        assert result["utilization_pct"] == 100.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_examples_skipped(self, fake_tokenizer, temp_dir):
        """Empty strings should be skipped, not cause errors."""
        src = os.path.join(temp_dir, "src")
        out = os.path.join(temp_dir, "out")

        texts = ["", "hello", "", "world", ""]
        _make_dataset(texts, src)

        result = pack_sft_dataset(
            dataset_path=src,
            tokenizer=fake_tokenizer,
            seq_len=20,
            output_path=out,
            val_fraction=0,
        )

        assert result["skipped"] == 3
        assert result["source_examples"] == 5
        assert result["packed_sequences"] >= 1

    def test_train_val_split(self, fake_tokenizer, temp_dir):
        """Validation split should create separate train/val datasets."""
        src = os.path.join(temp_dir, "src")
        out = os.path.join(temp_dir, "out")

        # Enough examples to produce multiple packed sequences
        texts = ["abcdefghij"] * 50
        _make_dataset(texts, src)

        result = pack_sft_dataset(
            dataset_path=src,
            tokenizer=fake_tokenizer,
            seq_len=20,
            output_path=out,
            val_fraction=0.1,
            val_max=5,
        )

        assert result["val_count"] > 0
        assert result["train_count"] + result["val_count"] == result["packed_sequences"]
        assert os.path.exists(result["val_path"])

    def test_default_output_path(self, fake_tokenizer, temp_dir):
        """Output path should default to <input>_packed_<seq_len>."""
        src = os.path.join(temp_dir, "src")
        _make_dataset(["hello"], src)

        result = pack_sft_dataset(
            dataset_path=src,
            tokenizer=fake_tokenizer,
            seq_len=32,
            val_fraction=0,
        )

        assert result["output_path"] == f"{src}_packed_32"

        # Cleanup auto-generated output
        shutil.rmtree(result["output_path"], ignore_errors=True)
