# SMLX Data Download Status

**Last Updated**: 2025-01-14

## Downloaded Datasets (Ready to Use)

### ✅ Benchmark Datasets (~62 MB)

#### WikiText-2 Language Modeling (~13 MB)

- **Location**: `data/benchmark/wikitext/test/`
- **Splits**: test
- **Samples**: 4,358
- **Status**: ✅ Complete
- **Usage**: Language model perplexity evaluation

#### GLUE Natural Language Understanding (~1 MB)

- **Location**: `data/benchmark/glue/`
- **Tasks**:
  - SST-2 (sentiment analysis) validation - 872 samples
  - MRPC (paraphrase detection) validation - 408 samples
- **Status**: ✅ Complete
- **Usage**: NLU task evaluation

#### MMStar Vision-Language (~43 MB)

- **Location**: `data/benchmark/mmstar/val/`
- **Split**: val (1,500 samples)
- **Status**: ✅ Complete
- **Usage**: Vision-indispensable multimodal evaluation
- **Note**: Fixed split name from `val_sample` to `val`

#### MMMU Multimodal Understanding (~5 MB)

- **Location**: `data/benchmark/mmmu/dev/`
- **Split**: dev (150 samples across 30 college subjects)
- **Status**: ✅ Complete
- **Usage**: Expert-level multimodal understanding evaluation
- **Note**: Automatically downloads and combines all 30 subjects (Accounting, Agriculture, Architecture, Art, Biology, Chemistry, Computer Science, Design, Economics, Electronics, Engineering, Finance, Geography, History, Literature, Management, Marketing, Materials, Math, Medicine, Music, Pharmacy, Physics, Psychology, Public Health, Sociology, etc.)

### ✅ Training/Testing Datasets (~508 KB)

#### COCO8 Images

- **Location**: `data/datasets/coco8/`
- **Content**: 8 diverse COCO images
- **Status**: ✅ Complete
- **Usage**: Quick VLM sanity checks

## ⏸️ Pending Downloads (Large Audio Files)

These datasets are very large (several GB) and download slowly. Download manually as needed:

### LibriSpeech Speech Recognition (~340 MB)

- **Command**: `python -m smlx.tools.download_data --dataset librispeech --split test_clean`
- **Location**: Will be saved to `data/benchmark/librispeech/test_clean/`
- **Note**: Audio files are large; download takes 30-60 minutes

### LJSpeech TTS Samples (~50 MB)

- **Command**: `python -m smlx.tools.download_data --dataset ljspeech_sample`
- **Location**: Will be saved to `data/datasets/ljspeech_sample/`

### ESC-50 Audio Classification (~25 MB)

- **Command**: `python -m smlx.tools.download_data --dataset esc50_sample`
- **Location**: Will be saved to `data/datasets/esc50_sample/`

## ⏭ Datasets Automatically Skipped

These datasets are automatically skipped due to compatibility issues with HuggingFace datasets library v3.x. The download script detects and handles these gracefully:

### MathVista (~100 MB)

- **Issue**: Incompatible with `datasets>=3.0` (uses deprecated 'List' feature type)
- **Status**: ⏭ Automatically skipped during download
- **Behavior**: Download script continues with other datasets without errors
- **To Download**:

  ```bash
  # Temporarily use older datasets library
  pip install datasets==2.14.0
  python -m smlx.tools.download_data --dataset mathvista --split testmini
  pip install datasets>=3.0  # Upgrade back if needed
  ```

- **Note**: Dataset maintainers need to update to use 'LargeList' feature type

### OCRBench (~150 MB)

- **Issue**: Incompatible with `datasets>=3.0` (uses deprecated 'List' feature type)
- **Status**: ⏭ Automatically skipped during download
- **Behavior**: Download script continues with other datasets without errors
- **To Download**:

  ```bash
  # Temporarily use older datasets library
  pip install datasets==2.14.0
  python -m smlx.tools.download_data --dataset ocrbench --split test
  pip install datasets>=3.0  # Upgrade back if needed
  ```

- **Note**: Dataset maintainers need to update to use 'LargeList' feature type

## 📊 Summary

**Successfully Downloaded**: ~62 MB across:

- 1 language dataset (WikiText-2 test - 4,358 samples)
- 2 NLU tasks (GLUE: SST-2, MRPC - 1,280 samples total)
- 2 vision-language datasets:
  - MMStar val (1,500 samples)
  - MMMU dev (150 samples across 30 subjects)

**Ready for Use**:

- ✅ SmolLM2 language model evaluation (WikiText, GLUE)
- ✅ SmolVLM vision-language evaluation (MMStar, MMMU)
- ✅ Multimodal understanding across 30 college-level subjects (MMMU)
- ✅ Basic model testing and development

**Pending (Download as Needed)**:

- ⏸️ Audio benchmarks (LibriSpeech, LJSpeech, ESC-50) - ~415 MB

**Automatically Skipped** (requires datasets==2.x):

- ⏭ MathVista (~100 MB) - automatically skipped, download with datasets==2.14.0
- ⏭ OCRBench (~150 MB) - automatically skipped, download with datasets==2.14.0

**Download Summary**:

- ✓ Successfully Downloaded: ~62 MB (4 benchmark datasets)
- ⏸️ Pending: ~415 MB (audio benchmarks - download as needed)
- ⏭ Skipped: ~250 MB (2 datasets - require datasets==2.x)
- **Grand Total Available**: ~727 MB when all downloaded

## Quick Start

### Run Evaluations on Downloaded Data

```bash
# Language model evaluation
python -m smlx.evals.text_eval --dataset wikitext-2

# NLU task evaluation
python -m smlx.evals.glue_eval --task sst2
python -m smlx.evals.glue_eval --task mrpc

# Vision-language evaluation
python -m smlx.evals.mmstar_eval
python -m smlx.evals.mmmu_eval
```

### Download Additional Datasets

```bash
# Download specific dataset
python -m smlx.tools.download_data --dataset librispeech --split test.clean

# Download all benchmarks (warning: several GB)
python -m smlx.tools.download_data --benchmarks

# Download MMMU with specific split
python -m smlx.tools.download_data --dataset mmmu --split validation

# List all available
python -m smlx.tools.download_data --list
```

### Workaround for MathVista and OCRBench

If you need these datasets, use one of these approaches:

```bash
# Option 1: Downgrade datasets library temporarily
pip install datasets==2.14.0
python -m smlx.tools.download_data --dataset mathvista --split testmini
pip install datasets>=3.0  # Upgrade back if needed

# Option 2: Download manually from HuggingFace Hub
# Visit https://huggingface.co/datasets/AI4Math/MathVista
# Visit https://huggingface.co/datasets/echo840/OCRBench
```

## Notes

- **Automatic compatibility handling**: The download script automatically detects and skips incompatible datasets (MathVista, OCRBench), continuing with other downloads without errors
- **datasets library version**: This project uses `datasets>=3.0`, which has better performance but some older datasets are incompatible and get automatically skipped
- **MMMU download**: Automatically downloads and combines all 30 college subjects into a single dataset
- **Audio datasets**: Large and slow to download (30-60 minutes for LibriSpeech) - download as needed
- **Split naming**: MMStar uses `val` (not `val_sample`), fixed in this version
- All downloaded datasets include README files with usage instructions
- Dataset licenses are documented in [DATASETS.md](DATASETS.md)

## Next Steps

1. ✅ Test language models on WikiText and GLUE datasets
2. ✅ Test vision-language models on MMStar and MMMU
3. ✅ Multi-subject MMMU download working (30 subjects combined automatically)
4. ✅ Automatic skip handling for incompatible datasets (MathVista, OCRBench)
5. ⏸️ Download audio datasets when needed for Whisper testing
6. ⏭ For MathVista/OCRBench: Use datasets==2.14.0 if these datasets are needed

---

For full dataset documentation, see [DATASETS.md](DATASETS.md) and [data/README.md](data/README.md).
