# YAMNet Weight Loading and Conversion

This document describes the YAMNet weight loading system, conversion process, and troubleshooting.

## Overview

YAMNet in SMLX uses real pre-trained weights from Google's AudioSet-trained model. The weights are automatically downloaded and converted from PyTorch format to MLX format on first load.

**Weight Pipeline:**
```
Original TensorFlow (Google)
         ↓
PyTorch (w-hc/torch_audioset)  ← Auto-downloaded
         ↓
MLX (SMLX)                     ← Auto-converted & cached
```

## Quick Start

### Basic Usage

```python
from smlx.models.YAMNet import load

# First load: downloads PyTorch weights and converts to MLX
model = load()  # Requires: pip install torch

# Subsequent loads: uses cached MLX weights (fast)
model = load()  # No conversion needed
```

### Dependencies

**Required for first-time load:**
- `torch` - For loading PyTorch weights during conversion

```bash
pip install torch
```

**Required for audio processing:**
- `librosa` - For mel spectrogram computation
- `soundfile` - For audio file loading

```bash
pip install librosa soundfile
```

**Optional:**
- `huggingface_hub` - If using pre-converted MLX weights from HuggingFace Hub

## Weight Sources

### Primary: PyTorch (torch_audioset)

- **Source**: https://github.com/w-hc/torch_audioset
- **URL**: `https://github.com/w-hc/torch_audioset/releases/download/v0.1/yamnet.pth`
- **Size**: ~15 MB
- **Format**: PyTorch state_dict (.pth)
- **License**: MIT (torch_audioset), Apache 2.0 (original YAMNet)

### Original: TensorFlow Hub

- **Source**: https://tfhub.dev/google/yamnet/1
- **URL**: `https://storage.googleapis.com/audioset/yamnet.h5`
- **Format**: Keras H5 / TensorFlow SavedModel
- **Note**: Requires manual conversion (not recommended)

### Future: HuggingFace MLX

If pre-converted MLX weights are uploaded to HuggingFace Hub:

```python
model = load("mlx-community/yamnet")  # No PyTorch required
```

## Weight Conversion Process

### Automatic Conversion (Default)

The `load()` function handles conversion automatically:

1. **Check cache**: Look for cached MLX weights at `~/.cache/smlx/yamnet/yamnet_mlx.npz`
2. **Download PyTorch**: If not cached, download from torch_audioset
3. **Convert**: Map PyTorch layer names to MLX names
4. **Validate**: Check weight shapes match expected
5. **Cache**: Save MLX weights for future use

### Manual Conversion

Use the standalone conversion tool for manual conversion:

```bash
# Convert and save locally
python -m smlx.tools.convert_yamnet --output ./yamnet_mlx

# Convert and upload to HuggingFace Hub
python -m smlx.tools.convert_yamnet \
    --upload mlx-community/yamnet

# Force re-download of source weights
python -m smlx.tools.convert_yamnet \
    --force-download \
    --output ./yamnet_mlx
```

### Conversion Script Options

```bash
python -m smlx.tools.convert_yamnet --help
```

**Options:**
- `--output DIR` - Output directory (default: `./yamnet_mlx`)
- `--cache-dir DIR` - Cache directory for downloads (default: `~/.cache/smlx/yamnet`)
- `--force-download` - Force re-download of PyTorch weights
- `--upload REPO` - Upload to HuggingFace Hub (e.g., `mlx-community/yamnet`)
- `--no-config` - Don't save config.json
- `--quiet` - Suppress progress messages

## Weight Mapping

### PyTorch → MLX Layer Mapping

The conversion maps PyTorch layer names to MLX structure:

**Layer 1 (Standard Convolution):**
```
layer1.fused.conv.weight       → conv1.weight
layer1.fused.bn.weight         → bn1.weight
layer1.fused.bn.bias           → bn1.bias
layer1.fused.bn.running_mean   → bn1.running_mean
layer1.fused.bn.running_var    → bn1.running_var
```

**Layers 2-14 (Depthwise Separable Convolutions):**
```
layer{2-14}.depthwise_conv.conv.weight  → conv_blocks.{0-12}.depthwise.weight
layer{2-14}.depthwise_conv.bn.*         → conv_blocks.{0-12}.bn_depthwise.*
layer{2-14}.pointwise_conv.conv.weight  → conv_blocks.{0-12}.pointwise.weight
layer{2-14}.pointwise_conv.bn.*         → conv_blocks.{0-12}.bn_pointwise.*
```

**Final Layers:**
```
embedding.weight    → embedding.weight
embedding.bias      → embedding.bias
classifier.weight   → classifier.weight
classifier.bias     → classifier.bias
```

### Weight Structure

Total weights: ~140 tensors

**Breakdown:**
- Layer 1: 5 tensors (conv + bn parameters)
- Layers 2-14 (13 blocks): 10 tensors each = 130 tensors
  - Depthwise conv: weight
  - Depthwise BN: weight, bias, running_mean, running_var
  - Pointwise conv: weight
  - Pointwise BN: weight, bias, running_mean, running_var
- Embedding layer: 2 tensors (weight, bias)
- Classifier: 2 tensors (weight, bias)

### Expected Weight Shapes

Key weight shapes (see `smlx/models/YAMNet/weights.py` for complete list):

```python
conv1.weight:        (32, 1, 3, 3)      # Initial conv
bn1.weight:          (32,)              # Batch norm gamma
bn1.bias:            (32,)              # Batch norm beta

# Example depthwise separable block (0)
conv_blocks.0.depthwise.weight:  (64, 32, 3, 3)   # Depthwise
conv_blocks.0.pointwise.weight:  (128, 64, 1, 1)  # Pointwise

embedding.weight:    (1024, 1024)       # Embedding layer
embedding.bias:      (1024,)

classifier.weight:   (521, 1024)        # 521 AudioSet classes
classifier.bias:     (521,)
```

## Cache Management

### Cache Location

Default cache directory: `~/.cache/smlx/yamnet/`

**Files:**
- `yamnet.pth` - Downloaded PyTorch weights (~15 MB)
- `yamnet_mlx.npz` - Converted MLX weights (~15 MB)

### Clearing Cache

To force re-download and re-conversion:

```python
from smlx.models.YAMNet import load

# Force re-download and re-conversion
model = load(force_download=True)
```

Or manually delete cache:

```bash
rm -rf ~/.cache/smlx/yamnet/
```

### Custom Cache Directory

```python
from pathlib import Path

# Use custom cache directory
model = load(cache_dir=Path("./my_cache"))
```

## Verification

### Verify Weights Loaded

```python
from smlx.models.YAMNet import load
from smlx.models.YAMNet.model import count_parameters

model = load()

# Check parameter count (~3.7M)
param_count = count_parameters(model)
print(f"Parameters: {param_count:,}")  # Should be ~3,700,000

# Check model is in eval mode
assert not model.training
```

### Validate Weight Shapes

```python
from smlx.models.YAMNet.loader import load_weights
from smlx.models.YAMNet.weights import validate_weight_shapes

weights = load_weights()

is_valid, errors = validate_weight_shapes(weights, strict=True)

if is_valid:
    print("✓ All weight shapes are correct")
else:
    print("Weight validation errors:")
    for error in errors:
        print(f"  - {error}")
```

### Test Inference

```python
import numpy as np
from smlx.models.YAMNet import load, classify

model = load()

# Create synthetic audio (3 seconds, 16kHz)
audio = np.sin(2 * np.pi * 440 * np.linspace(0, 3, 16000 * 3))
audio = audio.astype(np.float32)

# Classify
predictions = classify(model, audio, top_k=5)

print("Top predictions:")
for pred in predictions:
    print(f"  {pred.label}: {pred.score:.3f}")
```

## Troubleshooting

### "PyTorch is required for weight conversion"

**Problem**: PyTorch not installed

**Solution**:
```bash
pip install torch
```

Or use pre-converted MLX weights (if available):
```python
model = load("mlx-community/yamnet")
```

### "Failed to download PyTorch weights"

**Problem**: Network error or URL changed

**Solutions:**
1. Check internet connection
2. Try manual download:
   ```bash
   curl -L -o ~/.cache/smlx/yamnet/yamnet.pth \
       https://github.com/w-hc/torch_audioset/releases/download/v0.1/yamnet.pth
   ```
3. Use local weights:
   ```python
   from pathlib import Path
   from smlx.models.YAMNet.loader import load_weights

   weights = load_weights(model_path=Path("./my_weights"))
   ```

### "Weight validation warnings"

**Problem**: Some weight shapes don't match expected

**Solutions:**
- If only a few warnings and model works: Likely harmless differences
- If many errors: Re-download and re-convert weights with `force_download=True`

### "Model outputs look random"

**Problem**: Weights not loaded correctly

**Checks:**
1. Verify cache exists: `ls ~/.cache/smlx/yamnet/`
2. Check parameter count matches 3.7M
3. Force reload: `model = load(force_download=True)`

### Permission errors writing to cache

**Problem**: Cannot write to `~/.cache/smlx/yamnet/`

**Solution**: Use custom cache directory
```python
model = load(cache_dir=Path("/tmp/yamnet_cache"))
```

## Advanced Usage

### Programmatic Conversion

```python
from pathlib import Path
from smlx.models.YAMNet.loader import (
    download_pytorch_weights,
    convert_pytorch_to_mlx,
)
import mlx.core as mx

# Download PyTorch weights
pytorch_path = download_pytorch_weights()

# Convert to MLX
mlx_weights = convert_pytorch_to_mlx(pytorch_path)

# Save to custom location
output_path = Path("./my_yamnet_weights")
output_path.mkdir(exist_ok=True)
mx.savez(str(output_path / "yamnet.npz"), **mlx_weights)
```

### Inspect Weight Mapping

```python
from smlx.models.YAMNet.weights import get_pytorch_to_mlx_mapping

mapping = get_pytorch_to_mlx_mapping()

# Show some mappings
for pt_key, mlx_key in list(mapping.items())[:10]:
    print(f"{pt_key:40} → {mlx_key}")
```

### Upload to HuggingFace

After converting, you can upload to HuggingFace Hub for community use:

```bash
# Convert and upload
python -m smlx.tools.convert_yamnet \
    --output ./yamnet_mlx \
    --upload mlx-community/yamnet
```

Or programmatically:

```python
from huggingface_hub import HfApi
from pathlib import Path

api = HfApi()

# Create repo
api.create_repo(repo_id="mlx-community/yamnet", exist_ok=True)

# Upload weights
api.upload_folder(
    folder_path="./yamnet_mlx",
    repo_id="mlx-community/yamnet",
    repo_type="model",
)

print("✓ Uploaded to https://huggingface.co/mlx-community/yamnet")
```

## References

### Official Sources

- **Original YAMNet**: https://tfhub.dev/google/yamnet/1
- **TensorFlow Models**: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
- **AudioSet**: https://research.google.com/audioset/

### Converted Versions

- **PyTorch (torch_audioset)**: https://github.com/w-hc/torch_audioset
- **SMLX MLX**: This implementation

### Related Documentation

- [QUICKSTART.md](../../QUICKSTART.md) - Quick start guide
- [CLAUDE.md](../../CLAUDE.md) - Project overview
- [Quant.md](../Quant.md) - Quantization guide

## License

- **Original YAMNet**: Apache 2.0 (Google)
- **torch_audioset**: MIT License
- **SMLX Implementation**: Project license

When using YAMNet weights, please cite the original work:

```
@inproceedings{hershey2017cnn,
  title={CNN architectures for large-scale audio classification},
  author={Hershey, Shawn and Chaudhuri, Sourish and Ellis, Daniel PW and Gemmeke, Jort F and Jansen, Aren and Moore, R Channing and Plakal, Manoj and Platt, Devin and Saurous, Rif A and Seybold, Bryan and others},
  booktitle={2017 ieee international conference on acoustics, speech and signal processing (icassp)},
  pages={131--135},
  year={2017},
  organization={IEEE}
}
```
