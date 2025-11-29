# SPDX-License-Identifier: Apache-2.0
"""Memory-efficient LoRA merge utilities.

This module provides streaming/layer-by-layer LoRA merging to minimize
memory usage during the merge process. Instead of loading the entire model
into RAM, it processes layers one at a time.

Peak memory usage: ~2-3 layers worth of weights instead of entire model.
"""

from pathlib import Path
from typing import Optional, Iterator, Tuple, Dict, Any, Callable, List
import gc
import json
import logging
import os
import re
import shutil

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CHUNK_SIZE = 10  # Number of layers to process before saving
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate"]


def iter_safetensor_layers(
    model_path: Path,
) -> Iterator[Tuple[str, Any]]:
    """
    Lazily iterate over layers in safetensors files without loading all into RAM.

    Args:
        model_path: Path to model directory containing safetensors files

    Yields:
        Tuple of (layer_name, tensor)
    """
    # Third Party
    from safetensors import safe_open
    import glob as glob_module

    weight_files = sorted(glob_module.glob(str(model_path / "*.safetensors")))

    if not weight_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    for wf in weight_files:
        with safe_open(wf, framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


def load_lora_adapter(adapter_path: Path) -> Dict[str, Any]:
    """
    Load LoRA adapter weights. These are small enough to fit in RAM.

    Args:
        adapter_path: Path to adapter file (.npz, .safetensors, or .bin)

    Returns:
        Dict mapping layer names to lora_a and lora_b tensors
    """
    # Third Party
    import torch

    adapter_path = Path(adapter_path)

    if adapter_path.suffix == ".npz":
        # MLX format
        try:
            # Third Party
            import mlx.core as mx

            adapters = dict(mx.load(str(adapter_path)))
            # Convert to torch tensors
            import numpy as np

            return {k: torch.from_numpy(np.array(v)) for k, v in adapters.items()}
        except ImportError:
            # Fallback to numpy if MLX not available
            import numpy as np

            data = np.load(str(adapter_path))
            return {k: torch.from_numpy(data[k]) for k in data.files}

    elif adapter_path.suffix == ".safetensors":
        # Safetensors format
        # Third Party
        from safetensors.torch import load_file

        return load_file(str(adapter_path))

    elif adapter_path.suffix == ".bin":
        # PyTorch format
        return torch.load(str(adapter_path), map_location="cpu")

    else:
        # Try to find adapter file in directory
        if adapter_path.is_dir():
            for ext in [".safetensors", ".bin", ".npz"]:
                candidate = adapter_path / f"adapter_model{ext}"
                if candidate.exists():
                    return load_lora_adapter(candidate)
            raise FileNotFoundError(
                f"No adapter file found in {adapter_path}. "
                "Expected adapter_model.safetensors, adapter_model.bin, or adapter_model.npz"
            )
        raise ValueError(f"Unsupported adapter format: {adapter_path.suffix}")


def get_lora_keys_for_layer(
    layer_name: str, adapter_weights: Dict[str, Any]
) -> Optional[Tuple[str, str]]:
    """
    Find matching LoRA weights for a given layer.

    Args:
        layer_name: Name of the base model layer
        adapter_weights: Dict of adapter weights

    Returns:
        Tuple of (lora_a_key, lora_b_key) or None if not a LoRA target
    """
    for target in LORA_TARGET_MODULES:
        if target in layer_name and "weight" in layer_name:
            base_name = layer_name.replace(".weight", "")

            # Check various naming conventions
            naming_patterns = [
                # Standard PEFT naming
                (f"{base_name}.lora_a", f"{base_name}.lora_b"),
                # Alternative with .weight suffix
                (f"{base_name}.lora_A.weight", f"{base_name}.lora_B.weight"),
                # HuggingFace PEFT naming
                (
                    f"base_model.model.{base_name}.lora_A.weight",
                    f"base_model.model.{base_name}.lora_B.weight",
                ),
                # Shortened path
                (f"{base_name}.lora_A.default.weight", f"{base_name}.lora_B.default.weight"),
            ]

            for lora_a_key, lora_b_key in naming_patterns:
                if lora_a_key in adapter_weights and lora_b_key in adapter_weights:
                    return lora_a_key, lora_b_key

            # Try fuzzy matching - find keys containing the target module
            for key in adapter_weights.keys():
                if target in key and "lora_a" in key.lower():
                    # Extract the corresponding lora_b key
                    lora_b_key = key.replace("lora_a", "lora_b").replace(
                        "lora_A", "lora_B"
                    )
                    if lora_b_key in adapter_weights:
                        # Verify this is for the right layer
                        layer_match = re.search(r"layers\.(\d+)\.", layer_name)
                        key_match = re.search(r"layers\.(\d+)\.", key)
                        if layer_match and key_match:
                            if layer_match.group(1) == key_match.group(1):
                                if target in key:
                                    return key, lora_b_key

    return None


def merge_lora_into_layer(
    base_weight: Any,
    lora_a: Any,
    lora_b: Any,
    scale: float = 1.0,
    alpha: Optional[int] = None,
    rank: Optional[int] = None,
) -> Any:
    """
    Merge LoRA weights into base layer weights.

    Formula: W_merged = W_base + (alpha/rank) * (lora_B @ lora_A)

    Args:
        base_weight: Original layer weights
        lora_a: LoRA A matrix (typically shape: rank x in_features)
        lora_b: LoRA B matrix (typically shape: out_features x rank)
        scale: Scaling factor (default 1.0)
        alpha: LoRA alpha (if provided, used with rank for scaling)
        rank: LoRA rank (if provided, used with alpha for scaling)

    Returns:
        Merged weight tensor
    """
    # Third Party
    import torch

    # Compute scaling
    if alpha is not None and rank is not None and rank > 0:
        scale = alpha / rank

    # Ensure correct dtype
    dtype = base_weight.dtype
    device = base_weight.device

    lora_a = lora_a.to(dtype=dtype, device=device)
    lora_b = lora_b.to(dtype=dtype, device=device)

    # LoRA merge: W' = W + scale * (B @ A)
    # Shapes: lora_a is (rank, in_features) or (in_features, rank)
    #         lora_b is (out_features, rank) or (rank, out_features)
    # We need to produce (out_features, in_features) to match base_weight

    # Handle different shape conventions
    if lora_a.shape[0] == lora_b.shape[1]:
        # lora_a: (rank, in_features), lora_b: (out_features, rank)
        delta = scale * (lora_b @ lora_a)
    elif lora_a.shape[1] == lora_b.shape[0]:
        # lora_a: (in_features, rank), lora_b: (rank, out_features)
        delta = scale * (lora_b @ lora_a).T
    else:
        # Try transposing to make it work
        try:
            delta = scale * (lora_b @ lora_a.T)
        except RuntimeError:
            delta = scale * (lora_b.T @ lora_a)

    # Ensure delta matches base_weight shape
    if delta.shape != base_weight.shape:
        if delta.T.shape == base_weight.shape:
            delta = delta.T
        else:
            raise ValueError(
                f"Shape mismatch: base_weight {base_weight.shape}, "
                f"delta {delta.shape}, lora_a {lora_a.shape}, lora_b {lora_b.shape}"
            )

    return base_weight + delta


def stream_merge_lora(
    base_model_path: Path,
    adapter_path: Path,
    output_path: Path,
    scale: float = 1.0,
    alpha: Optional[int] = None,
    rank: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    de_quantize: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """
    Memory-efficient layer-by-layer LoRA merge.

    This function streams through the base model one layer at a time,
    merges LoRA weights where applicable, and writes to output incrementally.

    Peak memory usage: ~2-3 layers worth of weights instead of entire model.

    Args:
        base_model_path: Path to base model directory
        adapter_path: Path to LoRA adapter file or directory
        output_path: Path to write merged model
        scale: LoRA scaling factor
        alpha: LoRA alpha parameter
        rank: LoRA rank parameter
        chunk_size: Number of layers to buffer before writing
        de_quantize: Whether to de-quantize quantized layers
        progress_callback: Optional callback(current, total) for progress
    """
    # Third Party
    import torch
    from safetensors.torch import save_file

    base_model_path = Path(base_model_path)
    adapter_path = Path(adapter_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load adapter weights (these are small, ~10-100MB typically)
    logger.info(f"Loading LoRA adapter from {adapter_path}")
    adapter_weights = load_lora_adapter(adapter_path)
    logger.info(f"Loaded {len(adapter_weights)} adapter weight tensors")

    # Get adapter config for alpha/rank if not provided
    adapter_config_paths = [
        adapter_path / "adapter_config.json" if adapter_path.is_dir() else adapter_path.parent / "adapter_config.json",
        base_model_path / "adapter_config.json",
    ]

    for config_path in adapter_config_paths:
        if config_path.exists() and (alpha is None or rank is None):
            with open(config_path) as f:
                config = json.load(f)
                alpha = alpha if alpha is not None else config.get("lora_alpha", 32)
                rank = rank if rank is not None else config.get("r", 8)
                logger.info(f"Loaded adapter config: alpha={alpha}, rank={rank}")
                break

    # Default values if not found
    if alpha is None:
        alpha = 32
    if rank is None:
        rank = 8

    # Copy non-weight files (config, tokenizer, etc.)
    _copy_model_metadata(base_model_path, output_path)

    # Process layers in streaming fashion
    buffer: Dict[str, torch.Tensor] = {}
    shard_index = 0
    total_layers = _count_layers(base_model_path)
    processed = 0
    merged_count = 0

    logger.info(f"Starting layer-by-layer merge ({total_layers} layers)")
    print(f"Streaming LoRA merge: {total_layers} layers, chunk_size={chunk_size}")

    for layer_name, base_tensor in iter_safetensor_layers(base_model_path):
        # Check if this layer needs LoRA merge
        lora_keys = get_lora_keys_for_layer(layer_name, adapter_weights)

        if lora_keys:
            lora_a_key, lora_b_key = lora_keys
            lora_a = adapter_weights[lora_a_key]
            lora_b = adapter_weights[lora_b_key]

            # Handle quantized weights
            if de_quantize and _is_quantized(base_tensor):
                base_tensor = _dequantize_tensor(base_tensor)

            # Merge LoRA
            try:
                merged_tensor = merge_lora_into_layer(
                    base_tensor,
                    lora_a,
                    lora_b,
                    scale=scale,
                    alpha=alpha,
                    rank=rank,
                )
                buffer[layer_name] = merged_tensor
                merged_count += 1
                logger.debug(f"Merged LoRA into {layer_name}")
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Failed to merge {layer_name}: {e}, using original")
                buffer[layer_name] = base_tensor
        else:
            # No LoRA for this layer, pass through
            buffer[layer_name] = base_tensor

        processed += 1
        if progress_callback:
            progress_callback(processed, total_layers)

        # Print progress
        if processed % 50 == 0 or processed == total_layers:
            print(f"  Processed {processed}/{total_layers} layers ({merged_count} merged)")

        # Write chunk to disk when buffer is full
        if len(buffer) >= chunk_size:
            _write_shard(output_path, buffer, shard_index)
            shard_index += 1
            buffer.clear()
            gc.collect()  # Force garbage collection

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Write remaining layers
    if buffer:
        _write_shard(output_path, buffer, shard_index)

    # Write safetensors index file
    _write_index(output_path)

    logger.info(f"Merge complete. Merged {merged_count} layers. Output saved to {output_path}")
    print(f"Streaming merge complete: {merged_count} layers merged, output at {output_path}")


def _copy_model_metadata(src: Path, dst: Path) -> None:
    """Copy config, tokenizer, and other metadata files."""
    metadata_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "generation_config.json",
        "vocab.json",
        "merges.txt",
    ]

    for filename in metadata_files:
        src_file = src / filename
        if src_file.exists():
            shutil.copy2(src_file, dst / filename)
            logger.debug(f"Copied {filename}")


def _count_layers(model_path: Path) -> int:
    """Count total layers in model for progress reporting."""
    # Third Party
    from safetensors import safe_open
    import glob as glob_module

    count = 0
    for wf in glob_module.glob(str(model_path / "*.safetensors")):
        with safe_open(wf, framework="pt") as f:
            count += len(f.keys())
    return count


def _write_shard(output_path: Path, tensors: Dict[str, Any], shard_index: int) -> None:
    """Write a shard of tensors to disk."""
    # Third Party
    from safetensors.torch import save_file

    shard_name = f"model-{shard_index:05d}-of-{99999:05d}.safetensors"
    save_file(tensors, str(output_path / shard_name))
    logger.debug(f"Wrote shard {shard_name} ({len(tensors)} tensors)")


def _write_index(output_path: Path) -> None:
    """Write model.safetensors.index.json for sharded model."""
    # Third Party
    from safetensors import safe_open
    import glob as glob_module

    weight_map = {}
    total_size = 0
    shard_files = sorted(glob_module.glob(str(output_path / "model-*.safetensors")))
    num_shards = len(shard_files)

    # Rename shards with correct total count
    for i, old_path in enumerate(shard_files):
        old_path = Path(old_path)
        new_name = f"model-{i + 1:05d}-of-{num_shards:05d}.safetensors"
        new_path = output_path / new_name
        if old_path != new_path:
            old_path.rename(new_path)

    # Build index
    shard_files = sorted(glob_module.glob(str(output_path / "model-*.safetensors")))
    for shard_file in shard_files:
        shard_name = Path(shard_file).name
        with safe_open(shard_file, framework="pt") as f:
            for key in f.keys():
                weight_map[key] = shard_name
                tensor = f.get_tensor(key)
                total_size += tensor.numel() * tensor.element_size()

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}

    with open(output_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    logger.debug(f"Wrote index with {len(weight_map)} weights across {num_shards} shards")


def _is_quantized(tensor: Any) -> bool:
    """Check if tensor is quantized."""
    # Third Party
    import torch

    return tensor.dtype in [torch.int8, torch.uint8, torch.int4, torch.qint8, torch.quint8]


def _dequantize_tensor(tensor: Any) -> Any:
    """De-quantize a quantized tensor to float16."""
    # Third Party
    import torch

    if hasattr(tensor, "dequantize"):
        return tensor.dequantize().to(torch.float16)
    # Fallback: just cast (may lose precision)
    return tensor.to(torch.float16)


def get_merge_memory_estimate(model_path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Dict[str, int]:
    """
    Estimate memory requirements for merging.

    Args:
        model_path: Path to model directory
        chunk_size: Chunk size for streaming merge

    Returns:
        Dict with memory estimates in bytes
    """
    # Third Party
    from safetensors import safe_open
    import glob as glob_module

    layer_sizes: List[int] = []
    total_size = 0

    for wf in glob_module.glob(str(model_path / "*.safetensors")):
        with safe_open(wf, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                size = tensor.numel() * tensor.element_size()
                layer_sizes.append(size)
                total_size += size

    if not layer_sizes:
        return {"error": "No safetensors found"}

    avg_layer_size = sum(layer_sizes) / len(layer_sizes)
    max_layer_size = max(layer_sizes)

    return {
        "total_model_size": total_size,
        "num_layers": len(layer_sizes),
        "avg_layer_size": int(avg_layer_size),
        "max_layer_size": max_layer_size,
        "streaming_peak_memory": int(max_layer_size * chunk_size * 2.5),  # 2.5x for safety
        "full_load_memory": int(total_size * 2.5),  # Original approach
        "memory_savings_ratio": total_size / (max_layer_size * chunk_size * 2.5),
    }
