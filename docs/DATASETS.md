# SMLX Datasets Reference

Comprehensive guide to datasets used in SMLX for evaluation, training, and testing.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Benchmark Datasets](#benchmark-datasets)
- [Training/Testing Datasets](#trainingtesting-datasets)
- [Dataset Licenses](#dataset-licenses)
- [Download Instructions](#download-instructions)
- [Usage Examples](#usage-examples)
- [Citations](#citations)

## Overview

SMLX uses a carefully curated collection of datasets optimized for small models:

- **Benchmark Datasets** (`data/benchmark/`): Evaluation datasets for measuring model performance
- **Training/Testing Datasets** (`data/datasets/`): Small samples for development and testing
- **Document Samples** (`data/documents/`): OCR and document understanding samples
- **Image Samples** (`data/images/`): Vision-language model testing
- **Audio Samples** (`data/audio/`): Speech and sound model testing

**Total default data size**: ~220 MB (included in repository)
**Full benchmark size**: ~3-5 GB (downloaded separately)

## Quick Start

### Download Default Data

```bash
# Download all default datasets (~220 MB)
python -m smlx.tools.download_data --default-data
```

### Download Specific Dataset

```bash
# Download specific benchmark
python -m smlx.tools.download_data --dataset mathvista --split testmini

# Download all benchmarks (several GB)
python -m smlx.tools.download_data --benchmarks
```

### List Available Datasets

```bash
python -m smlx.tools.download_data --list
```

## Benchmark Datasets

### Vision-Language Model Benchmarks

#### MathVista: Mathematical Reasoning with Visual Contexts

**Source**: [AI4Math/MathVista](https://huggingface.co/datasets/AI4Math/MathVista)

**Description**: Evaluates mathematical reasoning capabilities in visual contexts including charts, diagrams, and geometric figures.

**Splits**:

- `testmini`: 1,000 samples (~100 MB) - **INCLUDED BY DEFAULT**
- `test`: 5,141 samples (~500 MB) - Download via script

**Models**: SmolVLM, nanoVLM, moondream3, TinyLLaVA

**Citation**:

```bibtex
@article{lu2024mathvista,
  title={MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts},
  author={Lu, Pan and Bansal, Hritik and Xia, Tony and Liu, Jiacheng and Li, Chunyuan and Hajishirzi, Hannaneh and Cheng, Hao and Chang, Kai-Wei and Galley, Michel and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2310.02255},
  year={2024}
}
```

**License**: CC BY-SA 4.0

---

#### MMMU: Massive Multi-discipline Multimodal Understanding

**Source**: [MMMU/MMMU](https://huggingface.co/datasets/MMMU/MMMU)

**Description**: Expert-level multimodal understanding across 30 college-level subjects spanning art, business, science, health, and humanities.

**Splits**:

- `dev`: 150 samples (~15 MB) - **INCLUDED BY DEFAULT**
- `validation`: 900 samples (~90 MB) - Download via script
- `test`: 10,500 samples (~1 GB) - Download via script

**Subjects**: 30 subjects across 6 core disciplines

**Models**: All vision-language models

**Citation**:

```bibtex
@article{yue2024mmmu,
  title={MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI},
  author={Yue, Xiang and Ni, Yuansheng and Zhang, Kai and Zheng, Tianyu and Liu, Ruoqi and Zhang, Ge and Stevens, Samuel and Jiang, Dongfu and Ren, Weiming and Sun, Yuxuan and others},
  journal={arXiv preprint arXiv:2311.16502},
  year={2024}
}
```

**License**: Apache 2.0

---

#### MMStar: Elite Vision-Indispensable Benchmark

**Source**: [Lin-Chen/MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar)

**Description**: Carefully curated benchmark focusing on vision-indispensable tasks across 6 core capabilities and 18 subcategories.

**Splits**:

- `val_sample`: 100 samples (~20 MB) - **INCLUDED BY DEFAULT**
- `val`: 1,500 samples (~250 MB) - Download via script

**Capabilities**:

1. Coarse Perception (450 samples)
2. Fine-grained Perception (450 samples)
3. Instance Reasoning (150 samples)
4. Logical Reasoning (150 samples)
5. Science & Technology (150 samples)
6. Math (150 samples)

**Models**: All vision-language models

**Citation**:

```bibtex
@article{chen2024mmstar,
  title={Are We on the Right Way for Evaluating Large Vision-Language Models?},
  author={Chen, Lin and Li, Jisong and Dong, Xiaoyi and Zhang, Pan and He, Conghui and Wang, Jiaqi and Zhao, Feng and Lin, Dahua},
  journal={arXiv preprint arXiv:2403.20330},
  year={2024}
}
```

**License**: Apache 2.0

---

#### OCRBench: Comprehensive OCR Evaluation

**Source**: [echo840/OCRBench](https://huggingface.co/datasets/echo840/OCRBench)

**Description**: Comprehensive evaluation of OCR capabilities across 5 task types and 29 source datasets.

**Splits**:

- `test_sample`: 50 samples (~10 MB) - **INCLUDED BY DEFAULT**
- `test`: 1,000 samples (~150 MB) - Download via script

**Task Types**:

1. Text Recognition (400 samples)
2. Scene Text-centric VQA (200 samples)
3. Document-oriented VQA (200 samples)
4. Key Information Extraction (100 samples)
5. Handwritten Math Expression Recognition (100 samples)

**Models**: OCR (SmolVLM via mlx-vlm), nanoVLM, moondream3

**Citation**:

```bibtex
@article{liu2023hidden,
  title={On the Hidden Mystery of OCR in Large Multimodal Models},
  author={Liu, Yuliang and Zhang, Biao and Guo, Chunyuan and Zhou, Xiang and Zhang, Hao and Li, Junda and Jia, Canjie and Zhang, Zihan and Cui, Cheng and Li, Xinyu and Liu, Xingzheng and Luo, Cong and Bai, Xiang},
  journal={arXiv preprint arXiv:2305.07895},
  year={2023}
}
```

**License**: CC BY 4.0

---

### Language Model Benchmarks

#### WikiText: Language Modeling Benchmark

**Source**: [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)

**Description**: Language modeling evaluation on Wikipedia text.

**Variants**:

- **WikiText-2**: Smaller variant (~4 MB) - **INCLUDED BY DEFAULT**
- **WikiText-103**: Larger variant (~500 MB) - Download via script

**Models**: SmolLM2-135M, SmolLM2-360M

**Citation**:

```bibtex
@article{merity2016pointer,
  title={Pointer sentinel mixture models},
  author={Merity, Stephen and Xiong, Caiming and Bradbury, James and Socher, Richard},
  journal={arXiv preprint arXiv:1609.07843},
  year={2016}
}
```

**License**: CC BY-SA 4.0

---

#### GLUE: General Language Understanding Evaluation

**Source**: [nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue)

**Description**: Natural language understanding benchmark suite with 9 tasks.

**Tasks Included by Default**:

- **SST-2**: Sentiment analysis (872 validation samples) - **INCLUDED**
- **MRPC**: Paraphrase detection (408 validation samples) - **INCLUDED**

**Additional Tasks** (download via script):

- CoLA, STS-B, QQP, MNLI, QNLI, RTE, WNLI

**Models**: SmolLM2-135M, SmolLM2-360M

**Citation**:

```bibtex
@inproceedings{wang2018glue,
  title={GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding},
  author={Wang, Alex and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel R},
  booktitle={Proceedings of the 2018 EMNLP Workshop BlackboxNLP},
  year={2018}
}
```

**License**: Various (see individual tasks)

---

### Audio Model Benchmarks

#### LibriSpeech: Speech Recognition Benchmark

**Source**: [openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr)

**Description**: Clean read English speech for ASR evaluation.

**Splits**:

- `test_clean_sample`: 20 clips (~5 MB) - **INCLUDED BY DEFAULT**
- `test_clean`: 2,620 utterances (~340 MB) - Download via script
- `test_other`: 2,939 utterances (~360 MB) - Download via script

**Audio Format**: 16 kHz, mono, FLAC

**Models**: Whisper-tiny

**Citation**:

```bibtex
@inproceedings{panayotov2015librispeech,
  title={Librispeech: an ASR corpus based on public domain audio books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5206--5210},
  year={2015},
  organization={IEEE}
}
```

**License**: CC BY 4.0

---

## Training/Testing Datasets

### COCO8: Quick Vision Testing

**Source**: [Ultralytics/COCO8](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip)

**Description**: 8 diverse COCO images for quick VLM sanity checks.

**Size**: ~2 MB

**Models**: All vision-language models

**License**: CC BY 4.0

**Included by Default**: Yes

---

### LJSpeech Sample: TTS Reference Audio

**Source**: [keithito/lj_speech](https://huggingface.co/datasets/keithito/lj_speech)

**Description**: 100 single-speaker TTS clips for reference and testing.

**Size**: ~50 MB

**Audio Format**: 22.05 kHz, mono, WAV

**Models**: Kokoro (TTS, mlx-audio)

**License**: Public Domain

**Included by Default**: Yes

---

### ESC-50 Sample: Environmental Sound Classification

**Source**: [ashraq/esc50](https://huggingface.co/datasets/ashraq/esc50)

**Description**: 50 environmental sound clips (10 per category) for audio classification testing.

**Size**: ~25 MB

**Categories**: Dog, rain, crying baby, door knock, helicopter, chainsaw, rooster, fire, car horn, church bells

**Models**: AST / AudioSet (audio classification, transformers)

**License**: CC BY-NC 3.0 (research/education use)

**Included by Default**: Yes

---

## Dataset Licenses

All default datasets have permissive licenses suitable for research and education:

| Dataset | License | Commercial Use | Attribution Required |
|---------|---------|----------------|---------------------|
| MathVista | CC BY-SA 4.0 | Yes | Yes |
| MMMU | Apache 2.0 | Yes | Yes |
| MMStar | Apache 2.0 | Yes | Yes |
| OCRBench | CC BY 4.0 | Yes | Yes |
| WikiText | CC BY-SA 4.0 | Yes | Yes |
| GLUE | Various | Varies | Yes |
| LibriSpeech | CC BY 4.0 | Yes | Yes |
| COCO8 | CC BY 4.0 | Yes | Yes |
| LJSpeech | Public Domain | Yes | No |
| ESC-50 | CC BY-NC 3.0 | No | Yes |

**Note**: Always review specific dataset licenses before commercial use.

## Download Instructions

### Prerequisites

```bash
# Install HuggingFace datasets library
pip install datasets

# Or install all SMLX dependencies
pip install -e ".[all]"
```

### Download Commands

#### Default Data (Included in Repo)

```bash
# Download ~220 MB of default datasets
python -m smlx.tools.download_data --default-data
```

This includes:

- MathVista testmini (1,000 samples)
- MMMU dev (150 samples)
- MMStar sample (100 samples)
- OCRBench sample (50 samples)
- WikiText-2 test
- GLUE SST-2 & MRPC validation
- COCO8 images
- LibriSpeech samples
- ESC-50 samples
- LJSpeech samples

#### Full Benchmarks

```bash
# Download all benchmark datasets (several GB)
python -m smlx.tools.download_data --benchmarks
```

#### Specific Datasets

```bash
# Download specific dataset and split
python -m smlx.tools.download_data --dataset mathvista --split testmini
python -m smlx.tools.download_data --dataset mmmu --split validation
python -m smlx.tools.download_data --dataset wikitext --split test

# Download training dataset
python -m smlx.tools.download_data --dataset coco8
```

#### Models

```bash
# Download all test models
python -m smlx.tools.download_data --models

# Download specific model
python -m smlx.tools.download_data --model smolvlm-256m
python -m smlx.tools.download_data --model whisper-tiny
```

## Usage Examples

### Running Evaluations

#### Vision-Language Model Evaluation

```python
from smlx.evals.math_vista import MathVistaEval
from smlx.models import load

# Load model
bm = load("smolvlm-256m")
model, processor = bm.model, bm.processor

# Run evaluation
evaluator = MathVistaEval(data_dir="data/benchmark/mathvista")
results = evaluator.evaluate(model, processor, split="testmini")
print(f"Accuracy: {results['accuracy']:.2%}")
```

#### Language Model Evaluation

```python
from smlx.evals.text_eval import WikiTextEval
from smlx.models import load

# Load model
bm = load("smollm2-135m")
model, tokenizer = bm.model, bm.processor

# Run evaluation
evaluator = WikiTextEval(data_dir="data/benchmark/wikitext")
results = evaluator.evaluate(model, tokenizer, variant="wikitext-2")
print(f"Perplexity: {results['perplexity']:.2f}")
```

#### Audio Model Evaluation

```python
from smlx.evals.asr_eval import LibriSpeechEval

# ASR runs through mlx-whisper (e.g. runner.produce("whisper-tiny", audio=...))

# Run benchmark evaluation
evaluator = LibriSpeechEval(data_dir="data/benchmark/librispeech")
results = evaluator.evaluate_test_clean()
print(f"Word Error Rate (WER): {results['wer']:.2%}")
```

### Loading Datasets

#### Using HuggingFace Datasets

```python
from datasets import load_dataset

# Load from HuggingFace Hub (cached locally)
dataset = load_dataset("AI4Math/MathVista", split="testmini")

# Load from local data directory
from datasets import load_from_disk
dataset = load_from_disk("data/benchmark/mathvista/testmini")
```

#### Manual Loading

```python
import json
from pathlib import Path

# Load dataset with metadata
data_dir = Path("data/benchmark/mathvista/testmini")

# Load metadata
with open(data_dir / "metadata.json") as f:
    metadata = json.load(f)

print(f"Dataset: {metadata['dataset']}")
print(f"Count: {metadata['count']}")
```

## Citations

If you use these datasets in your research, please cite the original papers. See individual dataset sections above for BibTeX citations.

### Citing SMLX

If you use SMLX in your research, please cite:

```bibtex
@software{smlx2025,
  title={SMLX: Small Models for MLX},
  author={SMLX Contributors},
  year={2025},
  url={https://github.com/yourusername/smlx}
}
```

## Additional Resources

- **Main Data README**: [data/README.md](data/README.md)
- **Download Script**: [smlx/tools/download_data.py](smlx/tools/download_data.py)
- **Evaluation Guide**: [docs/Eval.md](docs/Eval.md)
- **Benchmark Results**: [BENCHMARKS.md](BENCHMARKS.md)
- **Contributing Guide**: [CONTRIBUTING.md](CONTRIBUTING.md)

## Support

For issues with datasets:

1. Check dataset-specific READMEs in `data/` subdirectories
2. Review download script documentation
3. Verify HuggingFace Datasets is installed: `pip install datasets`
4. Check dataset licenses for usage restrictions
5. Report issues at: <https://github.com/yourusername/smlx/issues>

---

**Last Updated**: 2025-01-14
**Maintained By**: SMLX Contributors

For the latest information, see the [online documentation](https://smlx.readthedocs.io).
