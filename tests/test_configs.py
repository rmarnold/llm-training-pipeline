"""Tests for configuration loading and validation."""
import pytest
import yaml
import os

CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'configs')


class TestPretrainConfig:
    """Test pretraining configuration."""

    def test_pretrain_config_loads(self):
        """Test that pretrain.yaml loads without errors."""
        config_path = os.path.join(CONFIG_DIR, 'pretrain.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert 'model' in config
        assert 'training' in config
        assert 'data' in config

    def test_pretrain_config_required_fields(self):
        """Test that required fields exist in pretrain config."""
        config_path = os.path.join(CONFIG_DIR, 'pretrain.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Model section
        assert 'checkpoint' in config['model']
        assert 'gradient_checkpointing' in config['model']

        # Training section
        assert 'learning_rate' in config['training']
        assert 'per_device_train_batch_size' in config['training']
        assert 'gradient_accumulation_steps' in config['training']

    def test_pretrain_learning_rate_valid(self):
        """Test learning rate is in valid range."""
        config_path = os.path.join(CONFIG_DIR, 'pretrain.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)

        lr = config['training']['learning_rate']
        assert 1e-6 <= lr <= 1e-2, f"Learning rate {lr} outside expected range"


class TestSFTConfig:
    """Test SFT configuration."""

    def test_sft_config_loads(self):
        """Test that sft.yaml loads without errors."""
        config_path = os.path.join(CONFIG_DIR, 'sft.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert 'model' in config
        assert 'training' in config

    def test_sft_packing_enabled(self):
        """Test that sequence packing is configured."""
        config_path = os.path.join(CONFIG_DIR, 'sft.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Packing should be enabled for efficiency
        assert config['data'].get('packing', False) is True


class TestDPOConfig:
    """Test DPO configuration."""

    def test_dpo_config_loads(self):
        """Test that dpo.yaml loads without errors."""
        config_path = os.path.join(CONFIG_DIR, 'dpo.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert 'model' in config
        assert 'training' in config

    def test_dpo_beta_valid(self):
        """Test DPO beta is in valid range."""
        config_path = os.path.join(CONFIG_DIR, 'dpo.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)

        beta = config['training']['beta']
        assert 0.01 <= beta <= 1.0, f"DPO beta {beta} outside expected range"

    def test_dpo_learning_rate_conservative(self):
        """Test DPO uses conservative learning rate."""
        config_path = os.path.join(CONFIG_DIR, 'dpo.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)

        lr = config['training']['learning_rate']
        # DPO should use very low LR
        assert lr <= 1e-5, f"DPO learning rate {lr} should be <= 1e-5"


class TestLoRAConfig:
    """Test LoRA configuration."""

    def test_lora_config_loads(self):
        """Test that lora_finetune.yaml loads without errors."""
        config_path = os.path.join(CONFIG_DIR, 'lora_finetune.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert 'lora' in config

    def test_lora_rank_valid(self):
        """Test LoRA rank is reasonable."""
        config_path = os.path.join(CONFIG_DIR, 'lora_finetune.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)

        r = config['lora']['r']
        assert 4 <= r <= 128, f"LoRA rank {r} outside expected range"


class TestPromotionGates:
    """Test promotion gates configuration."""

    def test_gates_config_loads(self):
        """Test that promotion_gates.yaml loads."""
        config_path = os.path.join(CONFIG_DIR, 'promotion_gates.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Config uses 'gates' top-level key with transition names
        assert 'gates' in config
        gates = config['gates']
        assert 'pretrain_to_sft' in gates
        assert 'sft_to_dpo' in gates
        assert 'dpo_to_production' in gates

    def test_gates_have_thresholds(self):
        """Test that each gate has metric thresholds."""
        config_path = os.path.join(CONFIG_DIR, 'promotion_gates.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)

        gates = config['gates']

        # Check pretrain_to_sft gate
        pretrain_gate = gates['pretrain_to_sft']
        assert 'perplexity_threshold' in pretrain_gate

        # Check sft_to_dpo gate
        sft_gate = gates['sft_to_dpo']
        assert 'instruction_following_score' in sft_gate

        # Check dpo_to_production gate
        dpo_gate = gates['dpo_to_production']
        assert 'safety_refusal_rate' in dpo_gate

    def test_gates_thresholds_valid_ranges(self):
        """Test that gate thresholds are in valid ranges."""
        config_path = os.path.join(CONFIG_DIR, 'promotion_gates.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)

        gates = config['gates']

        # Perplexity should be positive
        assert gates['pretrain_to_sft']['perplexity_threshold'] > 0

        # Scores should be between 0 and 1
        assert 0 <= gates['sft_to_dpo']['instruction_following_score'] <= 1
        assert 0 <= gates['dpo_to_production']['safety_refusal_rate'] <= 1
