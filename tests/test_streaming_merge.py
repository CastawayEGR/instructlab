# SPDX-License-Identifier: Apache-2.0
"""Tests for memory-efficient streaming LoRA merge functionality."""

# Standard
from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import tempfile

# Third Party
import pytest


class TestMergeLoraIntoLayer:
    """Tests for the merge_lora_into_layer function."""

    def test_basic_merge(self):
        """Test basic LoRA merge math."""
        # Third Party
        import torch

        # First Party
        from instructlab.model.merge import merge_lora_into_layer

        # Create test tensors
        base = torch.randn(768, 768)
        rank = 8
        lora_a = torch.randn(rank, 768)  # (rank, in_features)
        lora_b = torch.randn(768, rank)  # (out_features, rank)

        # Merge with default scale
        merged = merge_lora_into_layer(base, lora_a, lora_b, alpha=32, rank=8)

        # Verify shape is preserved
        assert merged.shape == base.shape

        # Verify merge actually changed the weights
        assert not torch.allclose(merged, base)

    def test_merge_with_scale(self):
        """Test LoRA merge with explicit scale factor."""
        # Third Party
        import torch

        # First Party
        from instructlab.model.merge import merge_lora_into_layer

        base = torch.zeros(64, 64)
        lora_a = torch.ones(4, 64)
        lora_b = torch.ones(64, 4)

        # Merge with scale=2.0
        merged = merge_lora_into_layer(base, lora_a, lora_b, scale=2.0)

        # With zeros base and ones in LoRA, result should be 2.0 * (64x4 @ 4x64)
        # Each element should be 2.0 * 4 = 8.0
        expected_value = 2.0 * 4  # scale * rank (sum of 4 ones)
        assert torch.allclose(merged, torch.full((64, 64), expected_value))

    def test_merge_preserves_dtype(self):
        """Test that merge preserves the original dtype."""
        # Third Party
        import torch

        # First Party
        from instructlab.model.merge import merge_lora_into_layer

        base = torch.randn(32, 32, dtype=torch.float16)
        lora_a = torch.randn(4, 32, dtype=torch.float32)
        lora_b = torch.randn(32, 4, dtype=torch.float32)

        merged = merge_lora_into_layer(base, lora_a, lora_b)

        assert merged.dtype == base.dtype


class TestGetLoraKeysForLayer:
    """Tests for the get_lora_keys_for_layer function."""

    def test_finds_standard_keys(self):
        """Test finding standard PEFT LoRA keys."""
        # First Party
        from instructlab.model.merge import get_lora_keys_for_layer

        adapter_weights = {
            "model.layers.0.self_attn.q_proj.lora_a": None,
            "model.layers.0.self_attn.q_proj.lora_b": None,
        }

        result = get_lora_keys_for_layer(
            "model.layers.0.self_attn.q_proj.weight", adapter_weights
        )

        assert result is not None
        assert "lora_a" in result[0]
        assert "lora_b" in result[1]

    def test_finds_alternate_naming(self):
        """Test finding alternate PEFT naming convention."""
        # First Party
        from instructlab.model.merge import get_lora_keys_for_layer

        adapter_weights = {
            "model.layers.5.self_attn.v_proj.lora_A.weight": None,
            "model.layers.5.self_attn.v_proj.lora_B.weight": None,
        }

        result = get_lora_keys_for_layer(
            "model.layers.5.self_attn.v_proj.weight", adapter_weights
        )

        assert result is not None

    def test_returns_none_for_non_lora_layer(self):
        """Test that non-LoRA target layers return None."""
        # First Party
        from instructlab.model.merge import get_lora_keys_for_layer

        adapter_weights = {
            "model.layers.0.self_attn.q_proj.lora_a": None,
            "model.layers.0.self_attn.q_proj.lora_b": None,
        }

        # MLP layers typically aren't LoRA targets
        result = get_lora_keys_for_layer(
            "model.layers.0.mlp.up_proj.weight", adapter_weights
        )

        assert result is None

    def test_returns_none_for_non_weight_key(self):
        """Test that non-weight keys return None."""
        # First Party
        from instructlab.model.merge import get_lora_keys_for_layer

        adapter_weights = {
            "model.layers.0.self_attn.q_proj.lora_a": None,
            "model.layers.0.self_attn.q_proj.lora_b": None,
        }

        # Bias instead of weight
        result = get_lora_keys_for_layer(
            "model.layers.0.self_attn.q_proj.bias", adapter_weights
        )

        assert result is None


class TestLoadLoraAdapter:
    """Tests for the load_lora_adapter function."""

    def test_load_safetensors_adapter(self, tmp_path):
        """Test loading a safetensors adapter file."""
        # Third Party
        import torch
        from safetensors.torch import save_file

        # First Party
        from instructlab.model.merge import load_lora_adapter

        # Create a test adapter file
        adapter_data = {
            "lora_a": torch.randn(8, 768),
            "lora_b": torch.randn(768, 8),
        }
        adapter_path = tmp_path / "adapter_model.safetensors"
        save_file(adapter_data, str(adapter_path))

        # Load it
        loaded = load_lora_adapter(adapter_path)

        assert "lora_a" in loaded
        assert "lora_b" in loaded
        assert loaded["lora_a"].shape == (8, 768)

    def test_load_from_directory(self, tmp_path):
        """Test loading adapter from directory."""
        # Third Party
        import torch
        from safetensors.torch import save_file

        # First Party
        from instructlab.model.merge import load_lora_adapter

        # Create adapter file in directory
        adapter_data = {"test_weight": torch.randn(32, 32)}
        adapter_path = tmp_path / "adapter_model.safetensors"
        save_file(adapter_data, str(adapter_path))

        # Load from directory
        loaded = load_lora_adapter(tmp_path)

        assert "test_weight" in loaded


class TestMemoryEstimate:
    """Tests for memory estimation utilities."""

    def test_get_merge_memory_estimate(self, tmp_path):
        """Test memory estimation for model merge."""
        # Third Party
        import torch
        from safetensors.torch import save_file

        # First Party
        from instructlab.model.merge import get_merge_memory_estimate

        # Create a mock model with a few layers
        model_data = {
            f"layer_{i}": torch.randn(1024, 1024) for i in range(10)
        }
        save_file(model_data, str(tmp_path / "model.safetensors"))

        estimate = get_merge_memory_estimate(tmp_path, chunk_size=5)

        assert "total_model_size" in estimate
        assert "num_layers" in estimate
        assert "streaming_peak_memory" in estimate
        assert "full_load_memory" in estimate
        assert "memory_savings_ratio" in estimate

        # Streaming should use less memory than full load
        assert estimate["streaming_peak_memory"] < estimate["full_load_memory"]


class TestIterSafetensorLayers:
    """Tests for the safetensor layer iterator."""

    def test_iterates_all_layers(self, tmp_path):
        """Test that all layers are iterated."""
        # Third Party
        import torch
        from safetensors.torch import save_file

        # First Party
        from instructlab.model.merge import iter_safetensor_layers

        # Create model files
        layers = {f"layer_{i}": torch.randn(32, 32) for i in range(5)}
        save_file(layers, str(tmp_path / "model.safetensors"))

        # Iterate and count
        count = 0
        for name, tensor in iter_safetensor_layers(tmp_path):
            count += 1
            assert tensor.shape == (32, 32)

        assert count == 5

    def test_raises_on_missing_files(self, tmp_path):
        """Test that FileNotFoundError is raised for empty directory."""
        # First Party
        from instructlab.model.merge import iter_safetensor_layers

        with pytest.raises(FileNotFoundError):
            list(iter_safetensor_layers(tmp_path))


class TestStreamMergeLora:
    """Integration tests for the streaming merge function."""

    @pytest.fixture
    def mock_model_and_adapter(self, tmp_path):
        """Create mock model and adapter files for testing."""
        # Third Party
        import torch
        from safetensors.torch import save_file

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        output_dir = tmp_path / "output"

        # Create mock model weights
        model_weights = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(768, 768),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(768, 768),
            "model.layers.0.mlp.up_proj.weight": torch.randn(768, 768),
            "model.embed_tokens.weight": torch.randn(32000, 768),
        }
        save_file(model_weights, str(model_dir / "model.safetensors"))

        # Create config.json
        config = {
            "model_type": "llama",
            "hidden_size": 768,
            "num_hidden_layers": 1,
        }
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)

        # Create mock adapter weights
        adapter_weights = {
            "model.layers.0.self_attn.q_proj.lora_a": torch.randn(8, 768),
            "model.layers.0.self_attn.q_proj.lora_b": torch.randn(768, 8),
            "model.layers.0.self_attn.v_proj.lora_a": torch.randn(8, 768),
            "model.layers.0.self_attn.v_proj.lora_b": torch.randn(768, 8),
        }
        save_file(adapter_weights, str(adapter_dir / "adapter_model.safetensors"))

        # Create adapter config
        adapter_config = {"lora_alpha": 32, "r": 8}
        with open(adapter_dir / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f)

        return model_dir, adapter_dir, output_dir

    def test_stream_merge_creates_output(self, mock_model_and_adapter):
        """Test that streaming merge creates output files."""
        # First Party
        from instructlab.model.merge import stream_merge_lora

        model_dir, adapter_dir, output_dir = mock_model_and_adapter

        stream_merge_lora(
            base_model_path=model_dir,
            adapter_path=adapter_dir / "adapter_model.safetensors",
            output_path=output_dir,
            chunk_size=2,
        )

        # Check output exists
        assert output_dir.exists()
        assert any(output_dir.glob("*.safetensors"))
        assert (output_dir / "model.safetensors.index.json").exists()

    def test_stream_merge_preserves_non_lora_weights(self, mock_model_and_adapter):
        """Test that non-LoRA weights are preserved unchanged."""
        # Third Party
        import torch
        from safetensors import safe_open

        # First Party
        from instructlab.model.merge import stream_merge_lora

        model_dir, adapter_dir, output_dir = mock_model_and_adapter

        # Load original embed weight
        with safe_open(model_dir / "model.safetensors", framework="pt") as f:
            original_embed = f.get_tensor("model.embed_tokens.weight").clone()

        stream_merge_lora(
            base_model_path=model_dir,
            adapter_path=adapter_dir / "adapter_model.safetensors",
            output_path=output_dir,
            chunk_size=2,
        )

        # Load merged embed weight
        merged_files = list(output_dir.glob("model-*.safetensors"))
        for mf in merged_files:
            with safe_open(mf, framework="pt") as f:
                if "model.embed_tokens.weight" in f.keys():
                    merged_embed = f.get_tensor("model.embed_tokens.weight")
                    assert torch.allclose(original_embed, merged_embed)
                    return

        # If we get here, embed_tokens wasn't found
        pytest.fail("embed_tokens.weight not found in merged output")

    def test_stream_merge_modifies_lora_weights(self, mock_model_and_adapter):
        """Test that LoRA target weights are modified."""
        # Third Party
        import torch
        from safetensors import safe_open

        # First Party
        from instructlab.model.merge import stream_merge_lora

        model_dir, adapter_dir, output_dir = mock_model_and_adapter

        # Load original q_proj weight
        with safe_open(model_dir / "model.safetensors", framework="pt") as f:
            original_q_proj = f.get_tensor(
                "model.layers.0.self_attn.q_proj.weight"
            ).clone()

        stream_merge_lora(
            base_model_path=model_dir,
            adapter_path=adapter_dir / "adapter_model.safetensors",
            output_path=output_dir,
            chunk_size=2,
        )

        # Load merged q_proj weight
        merged_files = list(output_dir.glob("model-*.safetensors"))
        for mf in merged_files:
            with safe_open(mf, framework="pt") as f:
                if "model.layers.0.self_attn.q_proj.weight" in f.keys():
                    merged_q_proj = f.get_tensor(
                        "model.layers.0.self_attn.q_proj.weight"
                    )
                    # Should be different after merge
                    assert not torch.allclose(original_q_proj, merged_q_proj)
                    return

        pytest.fail("q_proj.weight not found in merged output")


class TestMLXStreamingMerge:
    """Tests for MLX-specific streaming merge functions."""

    @pytest.mark.skipif(
        not _mlx_available(), reason="MLX not available on this platform"
    )
    def test_maybe_merge_lora_mlx(self):
        """Test MLX LoRA merge helper function."""
        # Third Party
        import mlx.core as mx

        # First Party
        from instructlab.train.lora_mlx.fuse import _maybe_merge_lora_mlx

        tensor = mx.zeros((768, 768))
        adapters = {
            "model.layers.5.self_attn.q_proj.lora_a": mx.ones((8, 768)),
            "model.layers.5.self_attn.q_proj.lora_b": mx.ones((768, 8)),
        }

        merged, was_merged = _maybe_merge_lora_mlx(
            key="model.layers.5.self_attn.q_proj.weight",
            tensor=tensor,
            adapters=adapters,
            lora_start=0,
            num_layers=32,
            de_quantize=False,
            scale=1.0,
        )

        assert was_merged
        # Should be different from zeros
        assert not mx.allclose(merged, tensor)


def _mlx_available():
    """Check if MLX is available."""
    try:
        # Third Party
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False
