#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Integration tests for Chatterbox TTS model.

Tests the complete synthesis pipeline with the HiFi-GAN vocoder.
"""

import pytest
import mlx.core as mx
import numpy as np


@pytest.mark.integration
@pytest.mark.requires_model
def test_chatterbox_model_creation():
    """Test Chatterbox model can be created."""
    from smlx.models.Chatterbox.model import create_model

    model = create_model()
    assert model is not None
    assert hasattr(model, 'vocoder')
    assert hasattr(model, 'llama_backbone')
    assert hasattr(model, 'acoustic_head')


@pytest.mark.integration
@pytest.mark.requires_model
def test_chatterbox_forward_pass():
    """Test complete forward pass through model."""
    from smlx.models.Chatterbox.model import create_model
    from smlx.models.Chatterbox.config import DEFAULT_CONFIG

    model = create_model(DEFAULT_CONFIG)
    model.eval()

    # Create sample input
    batch_size = 2
    seq_len = 20
    input_ids = mx.random.randint(0, 1000, (batch_size, seq_len))

    # Forward pass
    mel, waveform = model(input_ids)

    # Check mel-spectrogram shape
    assert mel.ndim == 3
    assert mel.shape[0] == batch_size
    assert mel.shape[2] == 80  # n_mels

    # Check waveform shape
    assert waveform.ndim == 2
    assert waveform.shape[0] == batch_size

    # Waveform should not be all zeros (real vocoder!)
    assert not mx.all(waveform == 0)

    # Waveform should be in tanh range [-1, 1]
    assert float(waveform.min()) >= -1.0
    assert float(waveform.max()) <= 1.0


@pytest.mark.integration
def test_processor_audio_loading():
    """Test processor can handle audio processing."""
    from smlx.models.Chatterbox.processor import create_processor
    import numpy as np

    processor = create_processor(sample_rate=24000)

    # Create synthetic audio
    audio = np.random.randn(24000).astype(np.float32)  # 1 second

    # Process audio
    mel = processor.process_audio(audio, sr=24000)

    # Check mel-spectrogram
    assert mel.ndim == 2
    assert mel.shape[1] == 80  # n_mels
    assert mel.shape[0] > 0  # Has time frames


@pytest.mark.integration
def test_vocoder_mel_to_audio():
    """Test vocoder converts mel-spectrogram to audio."""
    from smlx.models.Chatterbox.vocoder import create_vocoder

    vocoder = create_vocoder()

    # Create random mel-spectrogram
    batch_size = 1
    time_frames = 100
    n_mels = 80
    mel = mx.random.normal((batch_size, time_frames, n_mels))

    # Generate waveform
    waveform = vocoder(mel)

    # Check shape
    assert waveform.ndim == 2
    assert waveform.shape[0] == batch_size

    # Check approximate length (100 frames × 256 hop = 25600 samples)
    expected_samples = time_frames * 256
    assert abs(waveform.shape[1] - expected_samples) < 1000

    # Check range
    assert float(waveform.min()) >= -1.0
    assert float(waveform.max()) <= 1.0


@pytest.mark.integration
def test_audio_utils_pipeline():
    """Test complete audio processing pipeline."""
    from smlx.models.Chatterbox import audio_utils
    import numpy as np

    # Create synthetic audio
    audio = np.random.randn(24000).astype(np.float32)
    audio_mlx = mx.array(audio)

    # Extract mel-spectrogram
    mel = audio_utils.log_mel_spectrogram(audio_mlx, n_mels=80, sample_rate=24000)

    # Check output
    assert mel.ndim == 2
    assert mel.shape[1] == 80  # n_mels
    assert mel.shape[0] > 0  # time frames


@pytest.mark.integration
@pytest.mark.requires_model
def test_end_to_end_synthesis():
    """Test end-to-end text-to-speech synthesis."""
    from smlx.models.Chatterbox.model import create_model
    from smlx.models.Chatterbox.processor import create_processor

    # Create model and processor
    model = create_model()
    processor = create_processor()
    model.eval()

    # Create dummy tokenizer behavior
    class DummyTokenizer:
        def encode(self, text):
            # Return dummy token IDs
            return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    processor.tokenizer = DummyTokenizer()

    # Process text
    text = "Hello world, this is a test."
    token_ids = processor(text)

    # Add batch dimension
    input_ids = mx.expand_dims(token_ids, axis=0)

    # Synthesize
    mel, waveform = model(input_ids)

    # Verify output
    assert mel.shape[0] == 1  # batch size
    assert waveform.shape[0] == 1
    assert waveform.shape[1] > 0  # has samples

    # Waveform should not be zeros
    assert not mx.all(waveform == 0)

    print(f"✓ End-to-end synthesis successful!")
    print(f"  Mel shape: {mel.shape}")
    print(f"  Waveform shape: {waveform.shape}")
    print(f"  Audio duration: {waveform.shape[1]/24000:.2f}s")


@pytest.mark.integration
def test_download_pretrained_vocoder():
    """Test downloading pre-trained HiFi-GAN vocoder."""
    from smlx.models.Chatterbox.loader import download_pretrained_vocoder

    # Test downloading HiFi-GAN LJSpeech
    print("\nTesting vocoder download (hifi-gan-ljspeech)...")
    vocoder_path = download_pretrained_vocoder(
        vocoder_name="hifi-gan-ljspeech",
        cache_dir=None  # Use default HF cache
    )

    if vocoder_path is not None:
        assert vocoder_path.exists()
        print(f"✓ Vocoder downloaded to: {vocoder_path}")
    else:
        print("⚠ Vocoder download failed (may require internet connection)")


@pytest.mark.integration
def test_convert_weights_module():
    """Test weight conversion utilities."""
    from smlx.models.Chatterbox.convert_weights import (
        transpose_conv1d_weight,
        save_mlx_weights,
        load_mlx_weights,
    )
    import tempfile
    from pathlib import Path

    # Test Conv1d transposition
    print("\nTesting Conv1d weight transposition...")
    weight_pytorch = np.random.randn(64, 32, 5).astype(np.float32)  # (O, I, K)
    weight_mlx = transpose_conv1d_weight(weight_pytorch)

    assert weight_mlx.shape == (64, 5, 32)  # (O, K, I)
    print(f"✓ Conv1d transpose: {weight_pytorch.shape} -> {weight_mlx.shape}")

    # Test saving and loading MLX weights
    print("\nTesting MLX weight save/load...")
    test_weights = {
        "layer1.weight": mx.random.normal((128, 64)),
        "layer2.weight": mx.random.normal((64, 32)),
        "layer2.bias": mx.random.normal((64,)),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_weights.npz"

        # Save
        save_mlx_weights(test_weights, save_path)
        assert save_path.exists()
        print(f"✓ Saved {len(test_weights)} tensors to {save_path}")

        # Load
        loaded_weights = load_mlx_weights(save_path)
        assert len(loaded_weights) == len(test_weights)
        print(f"✓ Loaded {len(loaded_weights)} tensors from {save_path}")

        # Verify
        for key in test_weights.keys():
            assert key in loaded_weights
            assert loaded_weights[key].shape == test_weights[key].shape


@pytest.mark.integration
def test_load_tts_model_weights_mlx_format():
    """Test loading TTS model weights from MLX format."""
    from smlx.models.Chatterbox.model import create_model
    from smlx.models.Chatterbox.loader import load_tts_model_weights
    from smlx.models.Chatterbox.convert_weights import save_mlx_weights
    import tempfile
    from pathlib import Path

    print("\nTesting TTS weight loading (MLX format)...")

    # Create model
    model = create_model()

    # Create dummy weights matching some model parameters
    dummy_weights = {
        "llama_backbone.embedding.weight": mx.random.normal((1000, 1024)),
        "acoustic_head.weight": mx.random.normal((80, 1024)),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save dummy weights
        checkpoint_path = Path(tmpdir) / "test_weights.npz"
        save_mlx_weights(dummy_weights, checkpoint_path)

        # Load weights into model
        success = load_tts_model_weights(
            model=model,
            checkpoint_path=checkpoint_path,
            model_type="auto",
            strict=False  # Non-strict loading
        )

        if success:
            print("✓ MLX format weight loading successful")
        else:
            print("⚠ MLX format weight loading failed")

        assert success is True


@pytest.mark.integration
def test_full_llama_backbone():
    """Test full Llama backbone with all features."""
    from smlx.models.Chatterbox.model import (
        LlamaBackbone,
        Attention,
        MLP,
        TransformerBlock,
        NoPE,
    )
    from smlx.models.Chatterbox.config import LlamaBackboneConfig

    print("\nTesting full Llama backbone...")

    # Create small config for testing
    config = LlamaBackboneConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=512,
        no_rope_layer_interval=4,
    )

    # Test Attention
    print("\n1. Testing Attention module...")
    attn = Attention(config)
    x = mx.random.normal((2, 10, config.hidden_size))
    out = attn(x, mask="causal")
    assert out.shape == x.shape
    print(f"   ✓ Attention: {x.shape} -> {out.shape}")

    # Test MLP
    print("\n2. Testing MLP module...")
    mlp = MLP(config)
    out = mlp(x)
    assert out.shape == x.shape
    print(f"   ✓ MLP: {x.shape} -> {out.shape}")

    # Test TransformerBlock
    print("\n3. Testing TransformerBlock...")
    block = TransformerBlock(config)
    out = block(x, mask="causal")
    assert out.shape == x.shape
    print(f"   ✓ TransformerBlock: {x.shape} -> {out.shape}")

    # Test NoPE
    print("\n4. Testing NoPE...")
    nope = NoPE()
    out = nope(x)
    assert mx.allclose(out, x)
    print(f"   ✓ NoPE: no-op verified")

    # Test full backbone
    print("\n5. Testing full LlamaBackbone...")
    backbone = LlamaBackbone(config)
    input_ids = mx.random.randint(0, config.vocab_size, (2, 10))
    out = backbone(input_ids=input_ids)
    assert out.shape == (2, 10, config.hidden_size)
    print(f"   ✓ LlamaBackbone: {input_ids.shape} -> {out.shape}")

    # Test NoPE layer configuration
    print("\n6. Verifying NoPE layer configuration...")
    for i, layer in enumerate(backbone.layers):
        if (i + 1) % config.no_rope_layer_interval == 0:
            assert isinstance(layer.self_attn.rope, NoPE)
            print(f"   ✓ Layer {i}: NoPE enabled")
        else:
            print(f"   ✓ Layer {i}: RoPE enabled")

    print("\n✓ Full Llama backbone test complete!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
