"""Pytest configuration and shared fixtures."""
import pytest
import os
import sys
import tempfile
import shutil

# Add scripts directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def sample_config():
    """Sample training configuration for tests."""
    return {
        'model': {
            'checkpoint': 'gpt2',  # Small model for testing
            'gradient_checkpointing': True,
            'use_flash_attention': False,
        },
        'training': {
            'num_train_epochs': 1,
            'max_steps': 10,
            'learning_rate': 1e-4,
            'per_device_train_batch_size': 2,
            'gradient_accumulation_steps': 1,
            'bf16': False,
            'tf32': False,
            'max_grad_norm': 1.0,
            'weight_decay': 0.01,
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
            'warmup_steps': 0,
            'lr_scheduler': 'cosine',
            'optim': 'adamw_torch',
            'fsdp': '',
            'fsdp_transformer_layer_cls_to_wrap': None,
        },
        'data': {
            'max_seq_length': 512,
            'num_workers': 0,
            'pin_memory': False,
            'persistent_workers': False,
        },
        'logging': {
            'logging_dir': './logs',
            'logging_steps': 1,
            'report_to': [],
            'save_steps': 5,
            'save_total_limit': 2,
        },
        'eval': {
            'evaluation_strategy': 'no',
            'eval_steps': 5,
            'per_device_eval_batch_size': 2,
        },
        'checkpointing': {
            'output_dir': './checkpoints',
            'save_strategy': 'steps',
            'save_steps': 5,
        },
        'curriculum': {
            'enabled': False,
        },
        'run_name': 'test-run',
    }


@pytest.fixture
def sample_texts():
    """Sample text data for testing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Large language models can generate human-like text.",
        "Training neural networks requires significant compute resources.",
        "Transformers have revolutionized natural language processing.",
    ]


@pytest.fixture
def mock_gpu_info():
    """Mock GPU information for testing without GPU."""
    return {
        'gpu_name': 'NVIDIA A100-SXM4-80GB',
        'compute_capability': '8.0',
        'is_h100': False,
        'is_a100': True,
        'fp8_available': False,
        'compile_mode': 'default',
        'batch_size': 8,
    }
