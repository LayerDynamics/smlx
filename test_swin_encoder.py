#!/usr/bin/env python3
"""Quick test of Swin Transformer encoder implementation."""

import mlx.core as mx
from smlx.models.Donut_base.config import SwinConfig
from smlx.models.Donut_base.model import SwinEncoder

def test_swin_encoder():
    """Test Swin encoder produces correct output shape."""
    print("Testing Swin Transformer Encoder...")

    # Create config
    config = SwinConfig(
        image_size=(224, 224),
        patch_size=4,
        num_channels=3,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        window_size=7,
    )

    # Create encoder
    hidden_size = 1024
    encoder = SwinEncoder(config, hidden_size)

    # Create dummy input in MLX format (batch_size=2, height=224, width=224, channels=3)
    pixel_values = mx.random.normal((2, 224, 224, 3))

    print(f"Input shape: {pixel_values.shape}")

    # Forward pass
    features = encoder(pixel_values)

    print(f"Output shape: {features.shape}")

    # Expected output shape calculation:
    # After 4 stages with 3 patch mergings:
    # Initial: 224 / 4 = 56 (patch embedding)
    # After stage 1 + merge: 56 / 2 = 28
    # After stage 2 + merge: 28 / 2 = 14
    # After stage 3 + merge: 14 / 2 = 7
    # After stage 4: 7 (no merge)
    # Sequence length: 7 * 7 = 49

    expected_shape = (2, 49, hidden_size)

    if features.shape == expected_shape:
        print(f"✓ SUCCESS: Output shape matches expected {expected_shape}")
        return True
    else:
        print(f"✗ FAILED: Expected {expected_shape}, got {features.shape}")
        return False

if __name__ == "__main__":
    success = test_swin_encoder()
    exit(0 if success else 1)
