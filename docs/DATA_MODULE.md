# SMLX Data Module Documentation

## Overview

The `smlx.data` module provides centralized, comprehensive data handling for all modalities in SMLX. It consolidates previously scattered data utilities and follows patterns from the MLX ecosystem (MLX-LM, MLX-VLM).

**Created Files:**
- [smlx/data/loaders.py](smlx/data/loaders.py:1) - Data loading for images, audio, text, video
- [smlx/data/datasets.py](smlx/data/datasets.py:1) - Dataset classes for different tasks
- [smlx/data/preprocessing.py](smlx/data/preprocessing.py:1) - Preprocessing pipelines
- [smlx/data/batch.py](smlx/data/batch.py:1) - DataLoader and collation utilities
- [smlx/data/hf.py](smlx/data/hf.py:1) - HuggingFace integration
- [smlx/data/augmentation.py](smlx/data/augmentation.py:1) - Data augmentation
- [smlx/data/__init__.py](smlx/data/__init__.py:1) - Public API exports

---

## Quick Start

```python
from smlx.data import (
    load_image, load_audio,
    TextDataset, DataLoader,
    ImagePreprocessor,
    collate_text
)

# Load single items
image = load_image("photo.jpg")  # Works with files, URLs, BytesIO, base64
audio = load_audio("speech.wav", sr=16000)

# Create dataset
dataset = TextDataset(data, tokenizer)

# Batch processing
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_text
)

for batch in dataloader:
    outputs = model(batch)
```

---

## Module Structure

### 1. **loaders.py** - Data Loading

Load data from various sources (files, URLs, BytesIO, base64).

**Functions:**
- `load_image(source, timeout=10)` → PIL.Image
  - Sources: file path, URL, BytesIO, base64 data URI, PIL Image
  - Automatically applies EXIF orientation and converts to RGB

- `load_audio(file, sr=16000, mono=True, timeout=10)` → mx.array
  - Sources: file path, URL
  - Uses soundfile or falls back to ffmpeg
  - Automatic resampling to target sample rate

- `load_text(file, encoding="utf-8")` → str
  - Load text files

- `load_video(file, fps=None, max_frames=None)` → list[PIL.Image]
  - Extract frames from video using ffmpeg

- `resample_audio(audio, orig_sr, target_sr)` → np.ndarray
  - Resample audio using linear interpolation

**Example:**
```python
from smlx.data import load_image, load_audio

# Load from file
img = load_image("cat.jpg")

# Load from URL
img = load_image("https://example.com/dog.jpg")

# Load from base64
img = load_image("data:image/jpeg;base64,/9j/4AAQ...")

# Load audio
audio = load_audio("speech.wav", sr=16000, mono=True)
```

---

### 2. **datasets.py** - Dataset Classes

Dataset classes following MLX-LM patterns.

**Classes:**
- `TextDataset` - Plain text with tokenization
- `ChatDataset` - Chat/conversation format (OpenAI messages)
- `CompletionsDataset` - Prompt-completion pairs
- `VisionLanguageDataset` - Image + text for VLMs
- `AudioDataset` - Audio with optional transcriptions
- `ConcatenatedDataset` - Combine multiple datasets
- `CacheDataset` - Cache processed items in memory
- `SubsetDataset` - Take subset by indices or percentage

**Example:**
```python
from smlx.data import TextDataset, ChatDataset, VisionLanguageDataset

# Text dataset
text_data = [
    {"text": "Hello world"},
    {"text": "How are you?"}
]
dataset = TextDataset(text_data, tokenizer)

# Chat dataset
chat_data = [{
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"}
    ]
}]
dataset = ChatDataset(chat_data, tokenizer, mask_prompt=True)

# Vision-language dataset
vlm_data = [
    {"image": "cat.jpg", "question": "What animal?", "answer": "A cat"},
    {"image": "dog.jpg", "text": "A dog playing"}
]
dataset = VisionLanguageDataset(vlm_data, tokenizer, image_processor)
```

---

### 3. **preprocessing.py** - Preprocessing Pipelines

Standard preprocessing for different modalities.

**Classes:**
- `ImagePreprocessor` - Resize, normalize, convert to tensor
- `AudioPreprocessor` - Mel-spectrogram computation
- `TextPreprocessor` - Tokenization with padding/truncation
- `MultimodalPreprocessor` - Combined image + text processing

**Example:**
```python
from smlx.data import ImagePreprocessor, AudioPreprocessor, TextPreprocessor

# Image preprocessing (CLIP-style)
img_processor = ImagePreprocessor(
    size=224,
    resize_mode="shortest_edge",
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
pixel_values = img_processor(image)  # Returns MLX array [C, H, W]

# Audio preprocessing
audio_processor = AudioPreprocessor(
    sample_rate=16000,
    n_mels=80
)
mel_spec = audio_processor(audio)

# Text preprocessing
text_processor = TextPreprocessor(
    tokenizer,
    max_length=512,
    padding="max_length"
)
result = text_processor("Hello world!")
# Returns: {"input_ids": mx.array(...), "attention_mask": mx.array(...)}
```

---

### 4. **batch.py** - Batching and DataLoader

Efficient batching without PyTorch dependency.

**Classes:**
- `DataLoader` - Simple dataloader for batching

**Functions:**
- `collate_text()` - Collate text with padding
- `collate_images()` - Collate images with padding
- `collate_audio()` - Collate audio with padding
- `collate_vlm()` - Collate vision-language samples
- `pad_sequences()` - Pad sequences to same length
- `batch_images()` - Batch images with padding
- `create_batches()` - Simple batch creation
- `dynamic_batching()` - Variable-size batching

**Example:**
```python
from smlx.data import DataLoader, collate_text, dynamic_batching

# Basic DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_text,
    drop_last=False
)

for batch in dataloader:
    # batch is a dict with padded tensors
    outputs = model(batch["input_ids"], batch["attention_mask"])

# Dynamic batching for variable-length sequences
def get_length(item):
    return len(item["input_ids"])

for batch in dynamic_batching(
    items,
    get_size_fn=get_length,
    max_batch_tokens=2048,
    max_batch_size=32
):
    process_batch(batch)
```

---

### 5. **hf.py** - HuggingFace Integration

Load datasets from HuggingFace Hub or local files.

**Functions:**
- `create_dataset()` - Auto-detect format and create dataset
- `load_local_dataset()` - Load from JSONL files
- `load_hf_dataset()` - Load from HuggingFace Hub
- `load_dataset_splits()` - Load with custom split specs
- `download_from_hub()` - Download dataset to local cache
- `save_dataset_to_jsonl()` - Save dataset to JSONL

**Example:**
```python
from smlx.data import load_hf_dataset, load_local_dataset, load_dataset_splits

# Load from HuggingFace Hub
train, valid, test = load_hf_dataset(
    "HuggingFaceTB/smoltalk",
    tokenizer
)

# Load from local JSONL files
train, valid, test = load_local_dataset(
    Path("./data/my_dataset"),
    tokenizer
)

# Custom splits
train, valid, test = load_dataset_splits(
    "HuggingFaceTB/smoltalk",
    tokenizer,
    train_split="train[:90%]",
    valid_split="train[90%:]",
    test_split=None
)
```

---

### 6. **augmentation.py** - Data Augmentation

Training data augmentation for improved robustness.

**Classes:**
- `ImageAugmentation` - Image transforms (flip, crop, brightness, rotation, etc.)
- `AudioAugmentation` - Audio transforms (noise, time stretch, pitch shift, etc.)
- `Compose` - Compose multiple transforms
- `RandomApply` - Apply transform with probability
- `RandomChoice` - Apply one of several transforms

**Example:**
```python
from smlx.data import ImageAugmentation, AudioAugmentation, Compose, RandomApply

# Image augmentation
img_aug = ImageAugmentation(
    random_flip=0.5,
    random_brightness=(0.8, 1.2),
    random_rotation=15,
    random_blur=0.1
)
augmented_img = img_aug(image)

# Audio augmentation
audio_aug = AudioAugmentation(
    add_noise=True,
    noise_level=0.01,
    volume_change=(0.8, 1.2)
)
augmented_audio = audio_aug(audio)

# Compose multiple transforms
composed = Compose([
    ImageAugmentation(random_flip=0.5),
    ImageAugmentation(random_brightness=(0.8, 1.2))
])

# Apply with probability
maybe_rotate = RandomApply(
    ImageAugmentation(random_rotation=30),
    p=0.5
)
```

---

## Common Patterns

### Pattern 1: Training Pipeline

```python
from smlx.data import (
    load_local_dataset,
    DataLoader,
    collate_text,
    ImageAugmentation,
    CacheDataset
)

# Load dataset
train, valid, _ = load_local_dataset(
    Path("./data/train"),
    tokenizer,
    mask_prompt=True
)

# Add caching for expensive preprocessing
train = CacheDataset(train)

# Create dataloader
train_loader = DataLoader(
    train,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_text
)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        loss = model.train_step(batch)
```

### Pattern 2: Vision-Language Model

```python
from smlx.data import (
    VisionLanguageDataset,
    ImagePreprocessor,
    DataLoader,
    collate_vlm
)

# Create image processor
image_processor = ImagePreprocessor(
    size=224,
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

# Create VLM dataset
dataset = VisionLanguageDataset(
    data,
    tokenizer,
    image_processor,
    image_key="image",
    question_key="question",
    answer_key="answer"
)

# Create dataloader
loader = DataLoader(
    dataset,
    batch_size=8,
    collate_fn=collate_vlm
)

# Inference
for batch in loader:
    outputs = vlm_model(
        batch["pixel_values"],
        batch["input_ids"],
        batch["attention_mask"]
    )
```

### Pattern 3: Audio Processing

```python
from smlx.data import (
    AudioDataset,
    AudioPreprocessor,
    load_audio,
    collate_audio
)

# Load audio
audio = load_audio("speech.wav", sr=16000, mono=True)

# Preprocess
processor = AudioPreprocessor(
    sample_rate=16000,
    n_mels=80,
    normalize=True
)
features = processor(audio)  # Mel-spectrogram

# Dataset
dataset = AudioDataset(
    data,
    audio_key="file",
    text_key="transcription",
    sample_rate=16000
)
```

---

## Migration from Old Code

The data module consolidates utilities previously scattered across the codebase:

**Before (scattered):**
```python
from smlx.utils.vision import load_image, preprocess_image
from smlx.utils.batch import create_batches, pad_batch
```

**After (centralized):**
```python
from smlx.data import load_image, ImagePreprocessor, DataLoader, pad_sequences
```

**Backward Compatibility:**
- Old utilities in `smlx/utils/vision.py` and `smlx/utils/batch.py` still work
- Gradually deprecate in favor of `smlx.data` module
- Update examples and models to use new module

---

## Testing

Basic tests are in [tests/data/test_loaders.py](tests/data/test_loaders.py:1).

Run tests:
```bash
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest tests/data/ -v
```

---

## Source References

The data module was adapted from:

1. **[resources/mlx-lm/mlx_lm/tuner/datasets.py](resources/mlx-lm/mlx_lm/tuner/datasets.py:1)** (lines 11-323)
   - TextDataset, ChatDataset, CompletionsDataset
   - HuggingFace loading patterns

2. **[resources/mlx-vlm/mlx_vlm/utils.py](resources/mlx-vlm/mlx_vlm/utils.py:580)** (lines 580-896)
   - load_image with all source types
   - load_audio with resampling
   - VLM preprocessing patterns

3. **[smlx/utils/vision.py](smlx/utils/vision.py:1)** (existing SMLX code)
   - Image loading and preprocessing
   - Batch processing utilities

4. **[smlx/utils/batch.py](smlx/utils/batch.py:1)** (existing SMLX code)
   - Batch creation and processing
   - Dynamic batching

5. **[resources/mlx-examples/cifar/dataset.py](resources/mlx-examples/cifar/dataset.py:1)**
   - Simple dataset patterns
   - Image augmentation

---

## Next Steps

1. **Update existing models** to use `smlx.data` module
2. **Add more tests** for datasets, preprocessing, batching
3. **Create examples** demonstrating new data module
4. **Add deprecation warnings** to old utilities
5. **Extend augmentation** with more transforms as needed

---

## Summary

The `smlx.data` module provides a **comprehensive, centralized data handling system** for SMLX, bringing together:

✅ **Loaders** - Load any modality from any source
✅ **Datasets** - MLX-LM compatible dataset classes
✅ **Preprocessing** - Standard pipelines for all modalities
✅ **Batching** - Efficient DataLoader without PyTorch
✅ **HuggingFace** - Seamless Hub integration
✅ **Augmentation** - Training data transforms

All following MLX ecosystem patterns and optimized for "smol" models on Apple Silicon!
