#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
YAMNet Weight Mapping and Conversion Utilities.

Provides mapping between PyTorch (torch_audioset) and MLX weight names,
plus utilities for weight shape validation.
"""

from typing import Dict, Tuple


# YAMNet Architecture Specification
# Based on MobileNet-v1 with 14 convolutional layers
YAMNET_LAYER_SPECS = [
    # (in_channels, out_channels, stride)
    # Layer 1: Standard Conv2D
    (1, 32, (2, 2)),
    # Layers 2-14: Depthwise Separable Convolutions
    (32, 64, (1, 1)),
    (64, 128, (2, 2)),
    (128, 128, (1, 1)),
    (128, 256, (2, 2)),
    (256, 256, (1, 1)),
    (256, 512, (2, 2)),
    (512, 512, (1, 1)),
    (512, 512, (1, 1)),
    (512, 512, (1, 1)),
    (512, 512, (1, 1)),
    (512, 512, (1, 1)),
    (512, 1024, (2, 2)),
    (1024, 1024, (1, 1)),
]


def get_pytorch_to_mlx_mapping() -> Dict[str, str]:
    """Get complete mapping from PyTorch weight names to MLX names.

    Returns:
        Dictionary mapping PyTorch keys to MLX keys
    """
    mapping = {}

    # Layer 1: Standard convolution + batch norm
    mapping['layer1.fused.conv.weight'] = 'conv1.weight'
    mapping['layer1.fused.bn.weight'] = 'bn1.weight'
    mapping['layer1.fused.bn.bias'] = 'bn1.bias'
    mapping['layer1.fused.bn.running_mean'] = 'bn1.running_mean'
    mapping['layer1.fused.bn.running_var'] = 'bn1.running_var'
    # Skip num_batches_tracked (PyTorch-specific, not needed for MLX)

    # Layers 2-14: Depthwise Separable Convolutions (13 blocks)
    for i in range(13):
        pytorch_layer_idx = i + 2  # PyTorch layers 2-14
        mlx_block_idx = i           # MLX conv_blocks 0-12

        # Depthwise convolution
        mapping[f'layer{pytorch_layer_idx}.depthwise_conv.conv.weight'] = \
            f'conv_blocks.{mlx_block_idx}.depthwise.weight'
        mapping[f'layer{pytorch_layer_idx}.depthwise_conv.bn.weight'] = \
            f'conv_blocks.{mlx_block_idx}.bn_depthwise.weight'
        mapping[f'layer{pytorch_layer_idx}.depthwise_conv.bn.bias'] = \
            f'conv_blocks.{mlx_block_idx}.bn_depthwise.bias'
        mapping[f'layer{pytorch_layer_idx}.depthwise_conv.bn.running_mean'] = \
            f'conv_blocks.{mlx_block_idx}.bn_depthwise.running_mean'
        mapping[f'layer{pytorch_layer_idx}.depthwise_conv.bn.running_var'] = \
            f'conv_blocks.{mlx_block_idx}.bn_depthwise.running_var'

        # Pointwise convolution (1x1)
        mapping[f'layer{pytorch_layer_idx}.pointwise_conv.conv.weight'] = \
            f'conv_blocks.{mlx_block_idx}.pointwise.weight'
        mapping[f'layer{pytorch_layer_idx}.pointwise_conv.bn.weight'] = \
            f'conv_blocks.{mlx_block_idx}.bn_pointwise.weight'
        mapping[f'layer{pytorch_layer_idx}.pointwise_conv.bn.bias'] = \
            f'conv_blocks.{mlx_block_idx}.bn_pointwise.bias'
        mapping[f'layer{pytorch_layer_idx}.pointwise_conv.bn.running_mean'] = \
            f'conv_blocks.{mlx_block_idx}.bn_pointwise.running_mean'
        mapping[f'layer{pytorch_layer_idx}.pointwise_conv.bn.running_var'] = \
            f'conv_blocks.{mlx_block_idx}.bn_pointwise.running_var'

    # Embedding layer (1024 → embedding_size, typically 1024)
    # Note: Some PyTorch versions may not have this layer
    mapping['embedding.weight'] = 'embedding.weight'
    mapping['embedding.bias'] = 'embedding.bias'

    # Classifier head (embedding_size → 521 classes)
    mapping['classifier.weight'] = 'classifier.weight'
    mapping['classifier.bias'] = 'classifier.bias'

    return mapping


def get_expected_weight_shapes() -> Dict[str, Tuple[int, ...]]:
    """Get expected shapes for all MLX weights.

    Returns:
        Dictionary mapping MLX weight names to expected shapes
    """
    shapes = {}

    # Layer 1: Standard conv
    shapes['conv1.weight'] = (32, 1, 3, 3)
    shapes['bn1.weight'] = (32,)
    shapes['bn1.bias'] = (32,)
    shapes['bn1.running_mean'] = (32,)
    shapes['bn1.running_var'] = (32,)

    # Depthwise separable blocks
    for i in range(13):
        in_ch, out_ch, _ = YAMNET_LAYER_SPECS[i + 1]

        # Depthwise: groups = in_ch, each input channel gets its own filter
        shapes[f'conv_blocks.{i}.depthwise.weight'] = (in_ch, in_ch, 3, 3)
        shapes[f'conv_blocks.{i}.bn_depthwise.weight'] = (in_ch,)
        shapes[f'conv_blocks.{i}.bn_depthwise.bias'] = (in_ch,)
        shapes[f'conv_blocks.{i}.bn_depthwise.running_mean'] = (in_ch,)
        shapes[f'conv_blocks.{i}.bn_depthwise.running_var'] = (in_ch,)

        # Pointwise: 1x1 conv to change channels
        shapes[f'conv_blocks.{i}.pointwise.weight'] = (out_ch, in_ch, 1, 1)
        shapes[f'conv_blocks.{i}.bn_pointwise.weight'] = (out_ch,)
        shapes[f'conv_blocks.{i}.bn_pointwise.bias'] = (out_ch,)
        shapes[f'conv_blocks.{i}.bn_pointwise.running_mean'] = (out_ch,)
        shapes[f'conv_blocks.{i}.bn_pointwise.running_var'] = (out_ch,)

    # Embedding layer
    shapes['embedding.weight'] = (1024, 1024)
    shapes['embedding.bias'] = (1024,)

    # Classifier
    shapes['classifier.weight'] = (521, 1024)
    shapes['classifier.bias'] = (521,)

    return shapes


def validate_weight_shapes(weights: Dict, strict: bool = True) -> Tuple[bool, list]:
    """Validate that weight shapes match expected dimensions.

    Args:
        weights: Dictionary of weight tensors
        strict: If True, all weights must be present

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    expected_shapes = get_expected_weight_shapes()
    errors = []

    # Check all expected weights are present
    if strict:
        missing = set(expected_shapes.keys()) - set(weights.keys())
        if missing:
            errors.append(f"Missing weights: {missing}")

    # Check shapes match
    for name, expected_shape in expected_shapes.items():
        if name in weights:
            actual_shape = tuple(weights[name].shape)
            if actual_shape != expected_shape:
                errors.append(
                    f"Shape mismatch for {name}: "
                    f"expected {expected_shape}, got {actual_shape}"
                )

    # Check for unexpected weights
    unexpected = set(weights.keys()) - set(expected_shapes.keys())
    if unexpected:
        # Filter out PyTorch-specific keys
        unexpected = {k for k in unexpected if 'num_batches_tracked' not in k}
        if unexpected:
            errors.append(f"Unexpected weights: {unexpected}")

    is_valid = len(errors) == 0
    return is_valid, errors


def count_parameters(weights: Dict) -> int:
    """Count total number of parameters in weights.

    Args:
        weights: Dictionary of weight tensors

    Returns:
        Total parameter count
    """
    total = 0
    for name, tensor in weights.items():
        # Skip PyTorch tracking variables
        if 'num_batches_tracked' in name:
            continue
        if hasattr(tensor, 'size'):
            total += tensor.size
        elif hasattr(tensor, 'shape'):
            import numpy as np
            total += np.prod(tensor.shape)
    return total


__all__ = [
    'YAMNET_LAYER_SPECS',
    'get_pytorch_to_mlx_mapping',
    'get_expected_weight_shapes',
    'validate_weight_shapes',
    'count_parameters',
]
