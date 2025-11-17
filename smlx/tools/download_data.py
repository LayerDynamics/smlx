#!/usr/bin/env python3
"""
Command-line interface for downloading SMLX models and datasets.

Usage:
    # List available datasets
    python -m smlx.tools.download_data --list

    # Download benchmarks
    python -m smlx.tools.download_data --benchmarks
    python -m smlx.tools.download_data --dataset mathvista --split testmini

    # Download all default data
    python -m smlx.tools.download_data --default-data

    # Download specific datasets
    python -m smlx.tools.download_data --dataset ocrbench --split test
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: 'datasets' library not installed. Run: pip install datasets")

from .download import (
    download_from_url,
    download_model,
    get_cache_dir,
)

# Get project root (where data/ folder is)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ============================================================================
# BENCHMARK DATASETS
# ============================================================================

BENCHMARK_DATASETS = {
    "mathvista": {
        "repo_id": "AI4Math/MathVista",
        "splits": {
            "testmini": {"count": 1000, "size_mb": 100, "default": True},
            "test": {"count": 5141, "size_mb": 500, "default": False},
        },
        "output_dir": "benchmark/mathvista",
        "description": "Mathematical reasoning with visual contexts",
    },
    "mmmu": {
        "repo_id": "MMMU/MMMU",
        "splits": {
            "dev": {"count": 150, "size_mb": 15, "default": True},
            "validation": {"count": 900, "size_mb": 90, "default": False},
            "test": {"count": 10500, "size_mb": 1000, "default": False},
        },
        "output_dir": "benchmark/mmmu",
        "description": "Multimodal understanding across college subjects (30 subjects combined)",
        "note": "Downloads all 30 college subjects and combines them into a single dataset",
    },
    "mmstar": {
        "repo_id": "Lin-Chen/MMStar",
        "splits": {
            "val": {"count": 1500, "size_mb": 43, "default": True},
        },
        "output_dir": "benchmark/mmstar",
        "description": "Vision-indispensable multimodal benchmark",
    },
    "ocrbench": {
        "repo_id": "echo840/OCRBench",
        "splits": {
            "test": {"count": 1000, "size_mb": 150, "default": True},
        },
        "output_dir": "benchmark/ocrbench",
        "description": "OCR capabilities evaluation",
    },
    "wikitext": {
        "repo_id": "Salesforce/wikitext",
        "config": "wikitext-2-raw-v1",
        "splits": {
            "test": {"size_mb": 4, "default": True},
            "train": {"size_mb": 4, "default": False},
        },
        "output_dir": "benchmark/wikitext",
        "description": "Language modeling benchmark",
    },
    "wikitext-103": {
        "repo_id": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "splits": {
            "test": {"size_mb": 500, "default": False},
        },
        "output_dir": "benchmark/wikitext",
        "description": "Large language modeling benchmark",
    },
    "glue": {
        "repo_id": "nyu-mll/glue",
        "splits": {
            "sst2_validation": {"task": "sst2", "split": "validation", "default": True},
            "mrpc_validation": {"task": "mrpc", "split": "validation", "default": True},
            "cola_validation": {"task": "cola", "split": "validation", "default": False},
            "stsb_validation": {"task": "stsb", "split": "validation", "default": False},
        },
        "output_dir": "benchmark/glue",
        "description": "Natural language understanding benchmark",
    },
    "librispeech": {
        "repo_id": "openslr/librispeech_asr",
        "splits": {
            "test.clean": {"count": 2620, "size_mb": 340, "default": True},
            "test.other": {"count": 2939, "size_mb": 360, "default": False},
            "validation.clean": {"count": 2703, "size_mb": 337, "default": False},
        },
        "output_dir": "benchmark/librispeech",
        "description": "Standard ASR benchmark (~1000 hours of audiobooks)",
        "note": "Recommended for ASR evaluation. Full dataset is ~60GB.",
    },
    "fleurs": {
        "repo_id": "google/fleurs",
        "config": "en_us",
        "splits": {
            "validation": {"count": 647, "size_mb": 50, "default": True},
            "test": {"count": 647, "size_mb": 50, "default": False},
        },
        "output_dir": "benchmark/fleurs",
        "description": "Few-shot multilingual speech recognition (102 languages)",
        "note": "Config 'en_us' for English. Supports 102 languages for low-resource ASR evaluation.",
    },
    "speech_commands": {
        "repo_id": "google/speech_commands",
        "config": "v0.02",
        "splits": {
            "validation": {"count": 6798, "size_mb": 150, "default": True},
            "test": {"count": 4890, "size_mb": 100, "default": False},
        },
        "output_dir": "benchmark/speech_commands",
        "description": "Keyword spotting and audio classification benchmark",
        "note": "35 short command words for keyword detection tasks.",
    },
    "gigaspeech": {
        "repo_id": "speechcolab/gigaspeech",
        "config": "xs",
        "splits": {
            "test": {"count": 2000, "size_mb": 200, "default": False},
        },
        "output_dir": "benchmark/gigaspeech",
        "description": "Large-scale English ASR corpus",
        "note": "Config 'xs' (10 hours). Also available: s, m, l, xl (up to 10,000 hours).",
    },
    # ========== Document Understanding Benchmarks ==========
    "docvqa": {
        "repo_id": "lmms-lab/DocVQA",
        "splits": {
            "val": {"count": 1000, "size_mb": 500, "default": True},
            "test": {"count": 5349, "size_mb": 2500, "default": False},
        },
        "output_dir": "benchmark/docvqa",
        "description": "Document Visual Question Answering benchmark",
        "note": "For TrOCR, Donut, and VLM document understanding evaluation.",
    },
    "funsd": {
        "repo_id": "nielsr/funsd",
        "splits": {
            "train": {"count": 149, "size_mb": 50, "default": True},
            "test": {"count": 50, "size_mb": 20, "default": False},
        },
        "output_dir": "benchmark/funsd",
        "description": "Form Understanding in Noisy Scanned Documents",
        "note": "For document layout and form understanding tasks.",
    },
    "rvl_cdip": {
        "repo_id": "aharley/rvl_cdip",
        "splits": {
            "test": {"count": 40000, "size_mb": 15000, "default": False},
        },
        "output_dir": "benchmark/rvl_cdip",
        "description": "Document classification (400K images, 16 classes)",
        "note": "Large dataset for document image classification. Test split only.",
    },
}

# ============================================================================
# TRAINING/TESTING DATASETS
# ============================================================================

TRAINING_DATASETS = {
    "coco8": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip",
        "output_dir": "datasets/coco8",
        "size_mb": 2,
        "description": "8 COCO images for quick testing",
        "default": True,
    },
    # ========== Text-to-Speech (TTS) Datasets ==========
    "ljspeech_sample": {
        "repo_id": "keithito/lj_speech",
        "output_dir": "audio/tts/ljspeech_sample",
        "split": "train",
        "count": 100,
        "size_mb": 50,
        "description": "LJSpeech TTS dataset sample (100 samples)",
        "default": True,
    },
    "ljspeech": {
        "repo_id": "keithito/lj_speech",
        "output_dir": "audio/tts/ljspeech",
        "split": "train",
        "count": None,  # Full dataset
        "size_mb": 2500,
        "description": "Full LJSpeech TTS dataset (~13k samples, single speaker)",
        "default": False,
    },
    "libritts": {
        "repo_id": "mythicinfinity/libritts",
        "output_dir": "datasets/libritts",
        "count": 5000,
        "size_mb": 3000,
        "description": "LibriTTS multi-speaker TTS dataset (subset)",
        "default": False,
        "note": "Full dataset is ~585 hours (202GB). This downloads a 5000 sample subset.",
    },
    "vctk": {
        "repo_id": "vctk",
        "output_dir": "datasets/vctk",
        "count": None,
        "size_mb": 11000,
        "description": "VCTK multi-speaker English corpus (110 speakers)",
        "default": False,
        "note": "High-quality multi-speaker dataset for TTS research.",
    },
    # ========== Audio Classification Datasets ==========
    "esc50_sample": {
        "repo_id": "ashraq/esc50",
        "output_dir": "audio/environmental/esc50_sample",
        "split": "train",
        "count": 100,
        "size_mb": 80,
        "description": "ESC-50 environmental sound sample (100 clips, 50 classes)",
        "default": True,
    },
    "esc50": {
        "repo_id": "ashraq/esc50",
        "output_dir": "audio/environmental/esc50",
        "split": "train",
        "count": None,  # Full 2000 samples
        "size_mb": 773,
        "description": "Full ESC-50 environmental sound dataset (2000 clips, 50 classes)",
        "default": False,
    },
    "urbansound8k": {
        "repo_id": "danavery/urbansound8k",
        "output_dir": "datasets/urbansound8k",
        "count": None,
        "size_mb": 6000,
        "description": "UrbanSound8K urban sound classification (8732 samples, 10 classes)",
        "default": False,
        "note": "Common urban sounds: car horn, dog bark, drilling, etc.",
    },
    "gtzan": {
        "repo_id": "marsyas/gtzan",
        "output_dir": "datasets/gtzan",
        "count": None,
        "size_mb": 1200,
        "description": "GTZAN music genre classification (1000 tracks, 10 genres)",
        "default": False,
        "note": "Classic music genre classification dataset.",
    },
    # ========== Speech/Audio Processing ==========
    "librispeech_sample": {
        "repo_id": "openslr/librispeech_asr",
        "config": "clean",
        "output_dir": "audio/speech/librispeech_sample",
        "split": "train.100",
        "count": 500,
        "size_mb": 100,
        "description": "LibriSpeech ASR sample (500 clean speech samples)",
        "default": True,
        "note": "Clean speech recordings for quick ASR testing.",
    },
    # ========== Document Datasets ==========
    "sroie": {
        "repo_id": "darentang/sroie",
        "output_dir": "datasets/sroie",
        "count": None,
        "size_mb": 300,
        "description": "SROIE receipts for OCR and information extraction (973 samples)",
        "default": False,
        "note": "Scanned receipts with company, date, address, total annotations.",
    },
    "sroie_sample": {
        "repo_id": "darentang/sroie",
        "output_dir": "datasets/sroie_sample",
        "count": 50,
        "size_mb": 20,
        "description": "SROIE receipts sample (50 samples)",
        "default": False,
    },
    # ========== Vision/Image Datasets ==========
    "coco128": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
        "output_dir": "datasets/coco128",
        "size_mb": 7,
        "description": "COCO128 subset (128 images for quick VLM testing)",
        "default": False,
    },
    "flickr30k_sample": {
        "repo_id": "nlphuji/flickr30k",
        "output_dir": "datasets/flickr30k_sample",
        "count": 500,
        "split": "test",
        "size_mb": 300,
        "description": "Flickr30k image captioning sample (500 images)",
        "default": False,
        "note": "Images with 5 reference captions each for VLM evaluation.",
    },
    "visual_genome_sample": {
        "repo_id": "visual_genome",
        "config": "region_descriptions_v1.2.0",
        "output_dir": "datasets/visual_genome_sample",
        "count": 1000,
        "size_mb": 500,
        "description": "Visual Genome sample (1000 images with region descriptions)",
        "default": False,
        "note": "Dense scene graphs with objects, attributes, and relationships.",
    },
}

# ============================================================================
# PREDEFINED MODELS
# ============================================================================

TEST_MODELS = {
    "smollm2-135m": "mlx-community/SmolLM2-135M-Instruct",
    "smollm2-360m": "mlx-community/SmolLM2-360M-Instruct",
    "smolvlm-256m": "mlx-community/SmolVLM-256M-Instruct",
    "smolvlm-500m": "mlx-community/SmolVLM-500M-Instruct",
    "nanovlm": "mlx-community/nanoVLM",
    "moondream2": "vikhyatk/moondream2",
    "whisper-tiny": "mlx-community/whisper-tiny",
}


# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================


def download_mmmu_all_subjects(split: str = "dev", save_to_data_dir: bool = True) -> Path:
    """
    Download MMMU dataset across all 30 subjects and combine them.

    Args:
        split: Split to download (dev, validation, test)
        save_to_data_dir: Save to data directory

    Returns:
        Path to combined dataset
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library required. Run: pip install datasets")

    # All 30 MMMU subjects
    subjects = [
        'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory',
        'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science',
        'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics',
        'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage',
        'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy',
        'Physics', 'Psychology', 'Public_Health', 'Sociology'
    ]

    output_dir = DATA_DIR / "benchmark" / "mmmu" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Downloading MMMU dataset: {split} split")
    print(f"Subjects: {len(subjects)} college-level disciplines")
    print(f"{'='*70}\n")

    all_datasets = []
    failed_subjects = []

    for i, subject in enumerate(subjects, 1):
        print(f"  [{i}/{len(subjects)}] {subject}...", end=" ", flush=True)
        try:
            dataset = load_dataset("MMMU/MMMU", subject, split=split, trust_remote_code=True)
            all_datasets.append(dataset)
            print(f"✓ ({len(dataset)} samples)")
        except Exception as e:
            print(f"✗ Failed: {str(e).split(chr(10))[0]}")
            failed_subjects.append(subject)

    if not all_datasets:
        print("\n✗ Failed to download any MMMU subjects")
        return None

    # Combine all datasets
    print(f"\nCombining {len(all_datasets)} subjects...")
    from datasets import concatenate_datasets
    combined_dataset = concatenate_datasets(all_datasets)

    # Save combined dataset
    combined_dataset.save_to_disk(str(output_dir))

    # Save metadata
    metadata = {
        "dataset": "mmmu",
        "repo_id": "MMMU/MMMU",
        "split": split,
        "subjects": [s for s in subjects if s not in failed_subjects],
        "count": len(combined_dataset),
        "failed_subjects": failed_subjects,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✓ MMMU {split} downloaded! ({len(combined_dataset)} total samples)")
    if failed_subjects:
        print(f"  ⚠ {len(failed_subjects)} subjects failed: {', '.join(failed_subjects)}")
    print(f"  Saved to: {output_dir}")
    print(f"{'='*70}\n")

    return output_dir


def download_benchmark_dataset(
    dataset_name: str,
    split: Optional[str] = None,
    save_to_data_dir: bool = True,
) -> Path:
    """
    Download a benchmark dataset.

    Args:
        dataset_name: Name of benchmark dataset (e.g., 'mathvista')
        split: Specific split to download (defaults to 'default' splits)
        save_to_data_dir: If True, save to data/ folder; if False, use cache

    Returns:
        Path to downloaded dataset
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library required. Run: pip install datasets")

    if dataset_name not in BENCHMARK_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(BENCHMARK_DATASETS.keys())}"
        )

    # Special handling for MMMU (requires downloading all 30 subjects)
    if dataset_name == "mmmu":
        # Determine which split to download
        if split:
            return download_mmmu_all_subjects(split=split, save_to_data_dir=save_to_data_dir)
        else:
            # Download default split (dev)
            dataset_info = BENCHMARK_DATASETS[dataset_name]
            default_splits = [
                s for s, info in dataset_info["splits"].items()
                if info.get("default", False)
            ]
            if default_splits:
                return download_mmmu_all_subjects(split=default_splits[0], save_to_data_dir=save_to_data_dir)
            else:
                # No default, download dev
                return download_mmmu_all_subjects(split="dev", save_to_data_dir=save_to_data_dir)

    dataset_info = BENCHMARK_DATASETS[dataset_name]
    repo_id = dataset_info["repo_id"]
    output_dir = DATA_DIR / dataset_info["output_dir"]

    print(f"\n{'='*70}")
    print(f"Downloading: {dataset_name}")
    print(f"Repository: {repo_id}")
    print(f"Description: {dataset_info['description']}")
    if "note" in dataset_info:
        print(f"\nNOTE: {dataset_info['note']}")
    print(f"{'='*70}\n")

    # Determine which splits to download
    if split:
        if split not in dataset_info["splits"]:
            raise ValueError(
                f"Unknown split '{split}' for {dataset_name}. "
                f"Available: {list(dataset_info['splits'].keys())}"
            )
        splits_to_download = [split]
    else:
        # Download default splits
        splits_to_download = [
            s for s, info in dataset_info["splits"].items()
            if info.get("default", False)
        ]

    if not splits_to_download:
        print("No splits specified and no default splits. Skipping.")
        return output_dir

    # Download each split and track results
    successful_splits = []
    failed_splits = []
    skipped_splits = []  # For known compatibility issues

    for split_name in splits_to_download:
        split_info = dataset_info["splits"][split_name]
        size_mb = split_info.get("size_mb", "Unknown")
        count = split_info.get("count", "")

        print(f"\n  Downloading split: {split_name}")
        print(f"  Size: ~{size_mb} MB")
        if count:
            print(f"  Samples: {count}")

        try:
            # Load dataset using HuggingFace datasets
            config = dataset_info.get("config")

            if config:
                dataset = load_dataset(repo_id, config, split=split_name, trust_remote_code=True)
            else:
                # Handle GLUE's task-based structure
                if "task" in split_info:
                    task = split_info["task"]
                    actual_split = split_info["split"]
                    dataset = load_dataset(repo_id, task, split=actual_split, trust_remote_code=True)
                    split_name = f"{task}_{actual_split}"
                else:
                    dataset = load_dataset(repo_id, split=split_name, trust_remote_code=True)

            # Save to data directory if requested
            if save_to_data_dir:
                split_output_dir = output_dir / split_name
                split_output_dir.mkdir(parents=True, exist_ok=True)

                # Save dataset
                dataset.save_to_disk(str(split_output_dir))

                # Save metadata
                metadata = {
                    "dataset": dataset_name,
                    "repo_id": repo_id,
                    "split": split_name,
                    "count": len(dataset),
                    "size_mb": size_mb,
                }
                with open(split_output_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                print(f"  ✓ Saved to: {split_output_dir}")
                successful_splits.append(split_name)
            else:
                print("  ✓ Cached (use datasets.load_dataset to access)")
                successful_splits.append(split_name)

        except Exception as e:
            error_msg = str(e).split('\n')[0]  # Get first line of error

            # Check if this is the known "Feature type 'List'" compatibility issue
            if "Feature type 'List' not found" in str(e):
                print("  ⏭ Skipped: Incompatible with datasets>=3.0 (deprecated 'List' feature type)")
                print("    Workaround: pip install datasets==2.14.0 (then upgrade back)")
                skipped_splits.append((split_name, "datasets v3.x compatibility"))
            else:
                print(f"  ✗ Failed: {error_msg}")
                failed_splits.append((split_name, error_msg))

    # Report results
    print(f"\n{'='*70}")
    if successful_splits and not failed_splits and not skipped_splits:
        print(f"✓ {dataset_name} download complete! ({len(successful_splits)} split(s))")
    elif successful_splits:
        print(f"⚠ {dataset_name} partially downloaded:")
        print(f"  ✓ Success: {', '.join(successful_splits)}")
        if skipped_splits:
            print(f"  ⏭ Skipped: {', '.join(s for s, _ in skipped_splits)} (compatibility issues)")
        if failed_splits:
            print(f"  ✗ Failed: {', '.join(s for s, _ in failed_splits)}")
    elif skipped_splits and not failed_splits:
        print(f"⏭ {dataset_name} skipped - incompatible with datasets>=3.0")
        print(f"  Splits: {', '.join(s for s, _ in skipped_splits)}")
        print("  Run with datasets==2.14.0 to download this dataset")
    else:
        print(f"✗ {dataset_name} download FAILED - all splits failed")
    print(f"{'='*70}\n")

    return output_dir if successful_splits else None


def download_training_dataset(dataset_name: str) -> Path:
    """
    Download a training/testing dataset.

    Args:
        dataset_name: Name of training dataset (e.g., 'coco8')

    Returns:
        Path to downloaded dataset
    """
    if dataset_name not in TRAINING_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(TRAINING_DATASETS.keys())}"
        )

    dataset_info = TRAINING_DATASETS[dataset_name]
    output_dir = DATA_DIR / dataset_info["output_dir"]

    print(f"\n{'='*70}")
    print(f"Downloading: {dataset_name}")
    print(f"Description: {dataset_info['description']}")
    print(f"Size: ~{dataset_info['size_mb']} MB")
    if "note" in dataset_info:
        print(f"NOTE: {dataset_info['note']}")
    print(f"{'='*70}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if "url" in dataset_info:
            # Direct URL download
            url = dataset_info["url"]
            zip_path = output_dir / "download.zip"
            download_from_url(url, zip_path, desc=dataset_name)

            # Extract if zip
            if zip_path.suffix == ".zip":
                import zipfile
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(output_dir)
                zip_path.unlink()  # Remove zip file

        elif "repo_id" in dataset_info:
            # HuggingFace dataset
            if not DATASETS_AVAILABLE:
                raise ImportError("datasets library required")

            repo_id = dataset_info["repo_id"]
            config = dataset_info.get("config")
            split = dataset_info.get("split", "train")
            count = dataset_info.get("count")

            # Load dataset with optional config
            if config:
                dataset = load_dataset(repo_id, config, split=split, trust_remote_code=True)
            else:
                dataset = load_dataset(repo_id, split=split, trust_remote_code=True)

            # Take subset if count specified
            if count and len(dataset) > count:
                import random
                indices = random.sample(range(len(dataset)), count)
                dataset = dataset.select(indices)

            dataset.save_to_disk(str(output_dir))

        print(f"\n✓ Saved to: {output_dir}")

    except Exception as e:
        print(f"✗ Failed: {e}")
        raise

    return output_dir


def download_default_data() -> None:
    """Download all default datasets (included in repo)."""
    print("\n" + "="*70)
    print("DOWNLOADING DEFAULT DATA FOR SMLX")
    print("="*70)
    print("\nThis will download ~220 MB of default datasets:")
    print("  • Benchmark samples (MathVista, MMMU, MMStar, OCRBench)")
    print("  • Language benchmarks (WikiText-2, GLUE samples)")
    print("  • Audio samples (LibriSpeech, ESC-50, LJSpeech)")
    print("  • Image samples (COCO8)")
    print("\n" + "="*70 + "\n")

    # Download benchmark datasets (default splits only)
    print("\n[1/4] Downloading Benchmark Datasets...")
    print("-" * 70)
    for dataset_name, info in BENCHMARK_DATASETS.items():
        has_default = any(s.get("default", False) for s in info["splits"].values())
        if has_default:
            try:
                download_benchmark_dataset(dataset_name, split=None)
            except Exception as e:
                print(f"Warning: Failed to download {dataset_name}: {e}")

    # Download training datasets
    print("\n[2/4] Downloading Training/Testing Datasets...")
    print("-" * 70)
    for dataset_name, info in TRAINING_DATASETS.items():
        if info.get("default", False):
            try:
                download_training_dataset(dataset_name)
            except Exception as e:
                print(f"Warning: Failed to download {dataset_name}: {e}")

    # Note about image and audio samples
    print("\n[3/4] Document Samples")
    print("-" * 70)
    print("Document samples (IAM, SROIE, etc.) require manual download")
    print("due to licensing requirements. See data/documents/README.md")

    print("\n[4/4] Image Samples")
    print("-" * 70)
    print("Curated image test collection requires manual curation.")
    print("See data/images/README.md for instructions.")

    print("\n" + "="*70)
    print("✓ DEFAULT DATA DOWNLOAD COMPLETE!")
    print("="*70)
    print(f"\nData location: {DATA_DIR}")
    print("\nNext steps:")
    print("  1. Review data/README.md for usage examples")
    print("  2. Run evaluations: python -m smlx.evals.<eval_name>")
    print("  3. Download full datasets: python -m smlx.tools.download_data --benchmarks")


def download_all_benchmarks() -> None:
    """Download all benchmark datasets (including non-default splits)."""
    print("\n" + "="*70)
    print("DOWNLOADING ALL BENCHMARK DATASETS")
    print("="*70)
    print("\nThis will download several GB of benchmark data.")
    print("This may take a while...\n")

    successful_datasets = []
    failed_datasets = []
    skipped_datasets = []

    for dataset_name in BENCHMARK_DATASETS.keys():
        try:
            # Download all splits
            dataset_info = BENCHMARK_DATASETS[dataset_name]
            for split_name in dataset_info["splits"].keys():
                result = download_benchmark_dataset(dataset_name, split=split_name)

                # Track status based on result
                if result is None:
                    if dataset_name not in failed_datasets and dataset_name not in skipped_datasets:
                        # Could be failed or skipped - check output
                        failed_datasets.append(dataset_name)
                elif dataset_name not in successful_datasets and dataset_name not in skipped_datasets and dataset_name not in failed_datasets:
                    successful_datasets.append(dataset_name)
        except Exception as e:
            error_msg = str(e).split('\n')[0]

            # Check if it's a known compatibility issue
            if "Feature type 'List' not found" in str(e):
                if dataset_name not in skipped_datasets:
                    skipped_datasets.append(dataset_name)
            else:
                print(f"Warning: Failed to download {dataset_name}: {error_msg}")
                if dataset_name not in failed_datasets:
                    failed_datasets.append(dataset_name)
            continue

    # Report aggregate results
    print("\n" + "="*70)
    print("BENCHMARK DOWNLOAD SUMMARY")
    print("="*70)

    if successful_datasets:
        print(f"\n✓ Successfully downloaded ({len(successful_datasets)}):")
        for ds in successful_datasets:
            print(f"  • {ds}")

    if skipped_datasets:
        print(f"\n⏭ Skipped - incompatible with datasets>=3.0 ({len(skipped_datasets)}):")
        for ds in skipped_datasets:
            print(f"  • {ds} (use datasets==2.14.0 to download)")

    if failed_datasets:
        print(f"\n✗ Failed to download ({len(failed_datasets)}):")
        for ds in failed_datasets:
            print(f"  • {ds}")

    total = len(successful_datasets) + len(skipped_datasets) + len(failed_datasets)
    if not failed_datasets and not skipped_datasets:
        print("\n✓ ALL BENCHMARKS DOWNLOADED!")
    elif successful_datasets:
        print(f"\n⚠ {len(successful_datasets)}/{total} benchmarks downloaded")
        if skipped_datasets:
            print(f"  ({len(skipped_datasets)} skipped due to compatibility)")

    print("="*70)


def download_test_models(model_ids: Optional[list[str]] = None) -> None:
    """Download test models for integration testing."""
    if model_ids is None:
        model_ids = list(TEST_MODELS.keys())

    print(f"Downloading {len(model_ids)} test model(s)...")
    print("=" * 70)

    for model_id in model_ids:
        # Resolve shortcuts
        if model_id in TEST_MODELS:
            repo_id = TEST_MODELS[model_id]
            print(f"\n[{model_id}] -> {repo_id}")
        else:
            repo_id = model_id
            print(f"\n{repo_id}")

        try:
            model_path = download_model(
                repo_id,
                allow_patterns=[
                    "*.safetensors",
                    "*.json",
                    "*.txt",
                    "*.py",
                    "*.model",
                ],
                ignore_patterns=[
                    "*.bin",
                    "*.pt",
                    "*.onnx",
                ],
            )
            print(f"✓ Saved to: {model_path}")

        except Exception as e:
            print(f"✗ Failed to download {repo_id}: {e}")
            continue

    print("\n" + "=" * 70)
    print("✓ Model downloads complete!")


def list_available() -> None:
    """List available datasets and models."""
    print("\n" + "="*70)
    print("AVAILABLE BENCHMARK DATASETS")
    print("="*70)
    for name, info in BENCHMARK_DATASETS.items():
        print(f"\n  {name}")
        print(f"    {info['description']}")
        print(f"    Repo: {info['repo_id']}")
        print(f"    Splits: {', '.join(info['splits'].keys())}")
        if "note" in info:
            print(f"    {info['note']}")

    print("\n" + "="*70)
    print("AVAILABLE TRAINING/TESTING DATASETS")
    print("="*70)
    for name, info in TRAINING_DATASETS.items():
        print(f"\n  {name}")
        print(f"    {info['description']}")
        print(f"    Size: ~{info['size_mb']} MB")
        if "note" in info:
            print(f"    {info['note']}")

    print("\n" + "="*70)
    print("AVAILABLE MODELS")
    print("="*70)
    for shortcut, repo_id in TEST_MODELS.items():
        print(f"  {shortcut:20s} -> {repo_id}")

    print("\n" + "="*70)
    print("USAGE EXAMPLES")
    print("="*70)
    print("""
  # Download default data (included in repo)
  python -m smlx.tools.download_data --default-data

  # Download all benchmarks
  python -m smlx.tools.download_data --benchmarks

  # Download specific dataset
  python -m smlx.tools.download_data --dataset mathvista --split testmini
  python -m smlx.tools.download_data --dataset mmmu --split validation

  # Download model
  python -m smlx.tools.download_data --model smolvlm-256m

  # List everything
  python -m smlx.tools.download_data --list
""")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download SMLX models and datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default data for repo
  python -m smlx.tools.download_data --default-data

  # Download all benchmarks
  python -m smlx.tools.download_data --benchmarks

  # Download specific dataset
  python -m smlx.tools.download_data --dataset mathvista --split testmini

  # List available datasets and models
  python -m smlx.tools.download_data --list
""",
    )

    # Main download options
    parser.add_argument(
        "--default-data",
        action="store_true",
        help="Download default datasets (included in repo, ~220 MB)",
    )
    parser.add_argument(
        "--benchmarks",
        action="store_true",
        help="Download all benchmark datasets (full versions, several GB)",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Download all predefined test models",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download everything (models + all benchmarks)",
    )

    # Specific downloads
    parser.add_argument(
        "--dataset",
        type=str,
        metavar="NAME",
        help="Download specific dataset (e.g., mathvista, mmmu)",
    )
    parser.add_argument(
        "--split",
        type=str,
        metavar="SPLIT",
        help="Specific split to download (e.g., testmini, validation)",
    )
    parser.add_argument(
        "--model",
        type=str,
        action="append",
        dest="model_ids",
        metavar="MODEL_ID",
        help="Download specific model (shortcut or repo ID)",
    )

    # Utilities
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and models",
    )
    parser.add_argument(
        "--cache-dir",
        action="store_true",
        help="Print cache directory location",
    )

    args = parser.parse_args()

    # Handle list command
    if args.list:
        list_available()
        return 0

    # Handle cache-dir command
    if args.cache_dir:
        print(get_cache_dir())
        return 0

    # Handle download commands
    downloaded_something = False

    if args.all:
        download_default_data()
        download_all_benchmarks()
        download_test_models()
        return 0

    if args.default_data:
        download_default_data()
        downloaded_something = True

    if args.benchmarks:
        download_all_benchmarks()
        downloaded_something = True

    if args.dataset:
        # Try benchmark datasets first
        if args.dataset in BENCHMARK_DATASETS:
            download_benchmark_dataset(args.dataset, split=args.split)
        elif args.dataset in TRAINING_DATASETS:
            download_training_dataset(args.dataset)
        else:
            print(f"Error: Unknown dataset '{args.dataset}'")
            print("Run --list to see available datasets")
            return 1
        downloaded_something = True

    if args.models:
        download_test_models()
        downloaded_something = True

    if args.model_ids:
        download_test_models(args.model_ids)
        downloaded_something = True

    # If no commands specified, show help
    if not downloaded_something:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
