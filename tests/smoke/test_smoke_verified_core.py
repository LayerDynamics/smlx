"""Smoke tests for the verified-good core (LM / VLM / ASR / embeddings).

These establish the real-output contract on models already confirmed to work,
so the contract itself is validated before it is extended to the rest of the
zoo. Each loads real weights, runs on a bundled input, and asserts correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.smoke import assertions as A

pytestmark = [pytest.mark.smoke, pytest.mark.requires_model]


def test_smollm2_135m_real_text():
    from smlx.models.SmolLM2_135M import generate, load

    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
    out = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="List three primary colors:",
        max_tokens=40,
        temperature=0.0,
    )
    A.assert_text_coherent(out, context="SmolLM2-135M")


def test_smolvlm_256m_describes_giraffes(giraffe_image_path):
    from PIL import Image

    from smlx.models.SmolVLM_256M import generate, load

    model, processor = load("HuggingFaceTB/SmolVLM-256M-Instruct")
    with Image.open(giraffe_image_path) as im:
        image = im.convert("RGB")
    out = generate(
        model,
        processor,
        prompt="Describe this image in one sentence.",
        image=image,
        max_tokens=48,
        temperature=0.0,
    )
    A.assert_text_coherent(out, context="SmolVLM-256M")
    A.assert_contains_any(
        out, ["giraffe", "giraffes", "animal", "tree"], context="SmolVLM-256M caption"
    )


def test_whisper_tiny_transcribes(librispeech_clip):
    from smlx.models.Whisper_tiny import load, transcribe

    audio, reference = librispeech_clip
    model, tokenizer = load("mlx-community/whisper-tiny")
    result = transcribe(audio, model, tokenizer, language="en", verbose=None)
    A.assert_transcription(
        reference, (result.get("text") or "").strip(), max_wer=0.4, context="Whisper-tiny"
    )


def test_minilm_embeddings_semantic(semantic_sentences):
    from smlx.models.MiniLM import encode, load

    model, tokenizer = load("all-MiniLM-L6-v2")
    emb = np.asarray(encode(model, tokenizer, semantic_sentences), dtype=np.float32)
    A.assert_embeddings_semantic(
        emb,
        expected_dim=384,
        related_pair=(0, 1),
        unrelated_pair=(0, 2),
        context="all-MiniLM-L6-v2",
    )
