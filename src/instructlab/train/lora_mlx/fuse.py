# SPDX-License-Identifier: MIT
# Copyright © 2023 Apple Inc.

# Standard
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import gc
import glob as glob_module
import json
import logging
import re
import shutil

# Third Party
from mlx.utils import tree_flatten, tree_unflatten
import mlx.core as mx
import mlx.nn as nn

# Local
from . import utils
from .models.lora import LoRALinear

logger = logging.getLogger(__name__)

# Constants for streaming merge
DEFAULT_CHUNK_SIZE = 10
LORA_TARGET_MODULES = ["q_proj", "v_proj", "gate"]


def fine_tune(
    model: str = "mlx_model",
    save_path: str = "lora_fused_model",
    adapter_file: str = "adapters.npz",
    hf_path: Optional[str] = "None",
    upload_name: Optional[str] = None,
    de_quantize: bool = False,
):
    """LoRA or QLoRA fine tuning."""
    print("Loading pretrained model")

    loaded_model, tokenizer, config = utils.load(model)

    # Load adapters and get number of LoRA layers
    adapters = list(mx.load(adapter_file).items())
    lora_layers = len([m for m in adapters if "q_proj.lora_a" in m[0]])

    # Freeze all layers other than LORA linears
    loaded_model.freeze()
    for l in loaded_model.model.layers[len(loaded_model.model.layers) - lora_layers :]:
        l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj)
        l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj)
        if hasattr(l, "block_sparse_moe"):
            l.block_sparse_moe.gate = LoRALinear.from_linear(l.block_sparse_moe.gate)

    loaded_model.update(tree_unflatten(adapters))
    fused_linears = [
        (n, m.to_linear())
        for n, m in loaded_model.named_modules()
        if isinstance(m, LoRALinear)
    ]

    loaded_model.update_modules(tree_unflatten(fused_linears))

    if de_quantize:
        de_quantize_layers = []
        for n, m in loaded_model.named_modules():
            if isinstance(m, nn.QuantizedLinear):
                bias = "bias" in m
                weight = m.weight
                weight = mx.dequantize(
                    weight,
                    m.scales,
                    m.biases,
                    m.group_size,
                    m.bits,
                ).astype(mx.float16)
                output_dims, input_dims = weight.shape
                linear = nn.Linear(input_dims, output_dims, bias=bias)
                linear.weight = weight
                if bias:
                    linear.bias = m.bias
                de_quantize_layers.append((n, linear))

        loaded_model.update_modules(tree_unflatten(de_quantize_layers))

    weights = dict(tree_flatten(loaded_model.parameters()))
    if de_quantize:
        config.pop("quantization", None)
    utils.save_model(save_path, weights, tokenizer, config)

    if upload_name is not None:
        if not Path(model).exists():
            # If the model path doesn't exist, assume it's an HF repo
            hf_path = model
        elif hf_path is None:
            raise ValueError(
                "Must provide original Hugging Face repo to upload local model."
            )
        utils.upload_to_hub(save_path, upload_name, hf_path)


def fine_tune_streaming(
    model: str = "mlx_model",
    save_path: str = "lora_fused_model",
    adapter_file: str = "adapters.npz",
    de_quantize: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress_callback: Optional[Callable[[int, int], None]] = None,
):
    """
    Memory-efficient LoRA fusion using layer streaming.

    Unlike fine_tune(), this processes one layer at a time,
    reducing peak memory from ~3x model size to ~1 layer.

    Args:
        model: Path to the base model directory
        save_path: Path to save the fused model
        adapter_file: Path to the LoRA adapter file (.npz)
        de_quantize: Whether to de-quantize quantized layers
        chunk_size: Number of layers to buffer before writing to disk
        progress_callback: Optional callback(current, total) for progress updates
    """
    # Third Party
    from safetensors import safe_open
    import numpy as np

    model_path = Path(model)
    save_path_obj = Path(save_path)
    save_path_obj.mkdir(parents=True, exist_ok=True)

    # Load adapter (small, fits in RAM)
    print(f"Loading adapter from {adapter_file}")
    adapters = dict(mx.load(adapter_file))
    lora_layers = len([m for m in adapters if "q_proj.lora_a" in m])
    print(f"Found {lora_layers} LoRA layers in adapter")

    # Load config to determine model architecture
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    # Determine which layers have LoRA
    num_layers = config.get("num_hidden_layers", 32)
    lora_start = num_layers - lora_layers

    # Get scale from adapter if available
    scale_value = 20.0  # Default MLX LoRA scale
    for key in adapters:
        if "scale" in key:
            scale_value = float(adapters[key])
            break

    # Find all weight files
    weight_files = sorted(glob_module.glob(str(model_path / "*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    # Count total layers for progress
    total_layers = 0
    for wf in weight_files:
        with safe_open(wf, framework="np") as f:
            total_layers += len(f.keys())

    print(f"Streaming merge: {total_layers} layers, chunk_size={chunk_size}")

    # Process weight files one at a time
    output_tensors: Dict[str, Any] = {}
    shard_idx = 0
    processed = 0
    merged_count = 0

    for wf in weight_files:
        with safe_open(wf, framework="np") as f:
            for key in f.keys():
                # Load tensor as numpy, then convert to MLX
                np_tensor = f.get_tensor(key)
                tensor = mx.array(np_tensor)

                # Check if this layer needs LoRA merge
                merged_tensor, was_merged = _maybe_merge_lora_mlx(
                    key, tensor, adapters, lora_start, num_layers, de_quantize, scale_value
                )

                if was_merged:
                    merged_count += 1

                output_tensors[key] = merged_tensor
                processed += 1

                if progress_callback:
                    progress_callback(processed, total_layers)

                # Print progress periodically
                if processed % 50 == 0 or processed == total_layers:
                    print(f"  Processed {processed}/{total_layers} layers ({merged_count} merged)")

                # Write shard when buffer full
                if len(output_tensors) >= chunk_size:
                    _write_mlx_shard(save_path_obj, output_tensors, shard_idx)
                    shard_idx += 1
                    output_tensors.clear()
                    gc.collect()

    # Write remaining tensors
    if output_tensors:
        _write_mlx_shard(save_path_obj, output_tensors, shard_idx)
        shard_idx += 1

    # Rename shards with correct total count and write index
    _finalize_mlx_shards(save_path_obj, shard_idx)

    # Copy tokenizer and config
    _copy_mlx_metadata(model_path, save_path_obj, config, de_quantize)

    print(f"Streaming merge complete: {merged_count} layers merged, output at {save_path_obj}")


def _maybe_merge_lora_mlx(
    key: str,
    tensor: mx.array,
    adapters: Dict[str, mx.array],
    lora_start: int,
    num_layers: int,
    de_quantize: bool,
    scale: float,
) -> tuple:
    """
    Merge LoRA weights into a layer if applicable.

    Returns:
        Tuple of (merged_tensor, was_merged)
    """
    # Parse layer number from key like "model.layers.15.self_attn.q_proj.weight"
    match = re.search(r"layers\.(\d+)\.", key)
    if not match:
        return tensor, False

    layer_num = int(match.group(1))
    if layer_num < lora_start:
        return tensor, False  # No LoRA for this layer

    # Check for LoRA target modules
    for target in LORA_TARGET_MODULES:
        if target in key and "weight" in key:
            base_key = key.replace(".weight", "")
            lora_a_key = f"{base_key}.lora_a"
            lora_b_key = f"{base_key}.lora_b"

            if lora_a_key in adapters and lora_b_key in adapters:
                lora_a = adapters[lora_a_key]
                lora_b = adapters[lora_b_key]

                # De-quantize if needed
                original_dtype = tensor.dtype
                if de_quantize:
                    # Check if this is quantized data
                    if tensor.dtype in [mx.int8, mx.uint8]:
                        tensor = tensor.astype(mx.float16)

                # Ensure consistent dtype for merge
                dtype = tensor.dtype if tensor.dtype in [mx.float16, mx.bfloat16, mx.float32] else mx.float16

                # Merge: W' = W + scale * (B.T @ A.T)
                lora_a_t = lora_a.T.astype(dtype)
                lora_b_t = lora_b.T.astype(dtype)
                tensor_f = tensor.astype(dtype)

                delta = scale * (lora_b_t @ lora_a_t)

                # Handle shape mismatches
                if delta.shape != tensor_f.shape:
                    if delta.T.shape == tensor_f.shape:
                        delta = delta.T

                merged = tensor_f + delta
                logger.debug(f"Merged LoRA into {key}")
                return merged, True

    return tensor, False


def _write_mlx_shard(save_path: Path, tensors: Dict[str, mx.array], shard_idx: int) -> None:
    """Write a shard of MLX tensors to disk."""
    # Use placeholder for total count, will rename later
    shard_name = f"model-{shard_idx:05d}-of-99999.safetensors"
    shard_path = save_path / shard_name

    # Convert to format saveable by MLX
    mx.save_safetensors(str(shard_path), tensors)
    logger.debug(f"Wrote shard {shard_name} ({len(tensors)} tensors)")


def _finalize_mlx_shards(save_path: Path, num_shards: int) -> None:
    """Rename shards with correct count and write index file."""
    # Find and rename all shards
    shard_files = sorted(glob_module.glob(str(save_path / "model-*-of-99999.safetensors")))

    weight_map = {}
    total_size = 0

    for i, old_path in enumerate(shard_files):
        old_path = Path(old_path)
        new_name = f"model-{i + 1:05d}-of-{num_shards:05d}.safetensors"
        new_path = save_path / new_name
        old_path.rename(new_path)

        # Build weight map for index
        weights = mx.load(str(new_path))
        for key, value in weights.items():
            weight_map[key] = new_name
            total_size += value.size * value.itemsize

    # Write index file
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map
    }

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)


def _copy_mlx_metadata(src: Path, dst: Path, config: Dict, de_quantize: bool) -> None:
    """Copy tokenizer and config files."""
    # Third Party
    import transformers

    # Update config if de-quantized
    if de_quantize:
        config.pop("quantization", None)

    # Write config
    with open(dst / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Copy tokenizer files
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]

    for filename in tokenizer_files:
        src_file = src / filename
        if src_file.exists():
            shutil.copy2(src_file, dst / filename)

    # Try to save tokenizer properly if available
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(src))
        tokenizer.save_pretrained(str(dst))
    except Exception as e:
        logger.debug(f"Could not save tokenizer via transformers: {e}")
