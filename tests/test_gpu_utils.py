"""Tests for GPU detection and utilities."""
import pytest
from unittest.mock import patch, MagicMock


class TestGPUDetection:
    """Test GPU detection functionality."""

    def test_detect_gpu_type_no_cuda(self):
        """Test detection when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            from gpu_utils import detect_gpu_type
            result = detect_gpu_type()

            assert result['gpu_name'] == 'CPU'
            assert result['compute_capability'] == 'N/A'
            assert result['is_h100'] is False
            assert result['is_a100'] is False
            assert result['fp8_available'] is False
            assert result['compile_mode'] == 'default'

    def test_detect_gpu_type_a100(self):
        """Test A100 detection."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_name', return_value='NVIDIA A100-SXM4-80GB'), \
             patch('torch.cuda.get_device_capability', return_value=(8, 0)):
            from gpu_utils import detect_gpu_type
            result = detect_gpu_type()

            assert 'A100' in result['gpu_name']
            assert result['is_a100'] is True
            assert result['is_h100'] is False
            assert result['compile_mode'] == 'default'

    def test_detect_gpu_type_h100(self):
        """Test H100 detection."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_name', return_value='NVIDIA H100-SXM5-80GB'), \
             patch('torch.cuda.get_device_capability', return_value=(9, 0)):
            from gpu_utils import detect_gpu_type
            result = detect_gpu_type()

            assert 'H100' in result['gpu_name']
            assert result['is_h100'] is True
            assert result['is_a100'] is False
            assert result['compile_mode'] == 'max-autotune'

    def test_check_fp8_no_transformer_engine(self):
        """Test FP8 check when transformer-engine not installed."""
        with patch.dict('sys.modules', {'transformer_engine': None}):
            from gpu_utils import check_fp8_available
            # Force reimport to pick up patched module
            import importlib
            import gpu_utils
            importlib.reload(gpu_utils)
            result = gpu_utils.check_fp8_available()
            assert result is False

    def test_setup_torch_backends_no_cuda(self):
        """Test backend setup when CUDA not available."""
        with patch('torch.cuda.is_available', return_value=False):
            from gpu_utils import setup_torch_backends
            # Should not raise
            setup_torch_backends()


class TestGPUInfo:
    """Test GPU info printing."""

    def test_print_gpu_info(self, mock_gpu_info, capsys):
        """Test GPU info printing."""
        from gpu_utils import print_gpu_info
        print_gpu_info(mock_gpu_info)
        captured = capsys.readouterr()

        assert 'A100' in captured.out
        assert '8.0' in captured.out
