"""Fixtures for model smoke tests — real inputs drawn from the bundled data/ tree.

Inputs come from ``smlx.data.local`` so smoke tests exercise the same in-repo
data shipped to users. Fixtures are session-scoped; the autouse MLX-memory
purge in the top-level conftest still runs per test.
"""

from __future__ import annotations

import os

# Audio deps can load a second OpenMP runtime on macOS; allow it before import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="session")
def giraffe_image_path():
    """Path to a bundled COCO8 image of two giraffes (known content)."""
    from smlx.data import local

    if not local.is_available("coco8"):
        pytest.skip("coco8 not present in data/")
    tree = local.load("coco8", split="train")
    for p in tree.images:
        if p.name == "000000000025.jpg":
            return p
    pytest.skip("expected coco8 giraffe image not found")


@pytest.fixture(scope="session")
def librispeech_clip():
    """A (audio_array_16k, reference_text) pair from the bundled LibriSpeech sample."""
    from smlx.data import local

    if not local.is_available("librispeech_sample"):
        pytest.skip("librispeech_sample not present in data/")
    ds = local.load("librispeech_sample")
    row = ds[0]
    audio = np.asarray(row["audio"]["array"], dtype=np.float32)
    return audio, row["text"]


@pytest.fixture(scope="session")
def semantic_sentences():
    """Sentences with a clearly related pair (0,1) and an unrelated pair (0,2)."""
    return [
        "A small cat is sleeping on the warm windowsill.",
        "A kitten naps in the sunshine by the window.",
        "Quarterly interest rates were raised by the central bank.",
    ]
