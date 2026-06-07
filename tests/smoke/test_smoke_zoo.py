"""Smoke coverage for the rest of the model zoo (beyond the verified core).

Running ``pytest tests/smoke -m smoke -ra`` produces a live status board:
PASSED  = model loads real weights and produces correct output;
SKIPPED = a WS-2 task remains (the skip reason is the exact work needed).

As each WS-2 conversion/wiring/training task lands, its skip is removed and the
test must pass — there is no placeholder-passing test here.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.smoke import assertions as A

pytestmark = [pytest.mark.smoke, pytest.mark.requires_model]


# ---------------------------------------------------------------------------
# Language models
# ---------------------------------------------------------------------------


def test_smollm2_360m_real_text():
    from smlx.models.SmolLM2_360M import generate, load

    model, tokenizer = load("mlx-community/SmolLM2-360M-Instruct")
    out = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="Name three planets in our solar system:",
        max_tokens=40,
        temperature=0.0,
    )
    A.assert_text_coherent(out, context="SmolLM2-360M")


# ---------------------------------------------------------------------------
# Vision-language models
# ---------------------------------------------------------------------------


def test_smolvlm_500m_describes_giraffes(giraffe_image_path):
    from PIL import Image

    from smlx.models.SmolVLM_500M_Instruct import generate, load

    model, processor = load("HuggingFaceTB/SmolVLM-500M-Instruct")
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
    A.assert_text_coherent(out, context="SmolVLM-500M")
    A.assert_contains_any(out, ["giraffe", "giraffes", "animal", "tree"], context="SmolVLM-500M")


def test_nanovlm_describes_giraffes(giraffe_image_path):
    pytest.skip(
        "WS-2: nanoVLM loads real lusxvr/nanoVLM-222M weights correctly (verified: 0 "
        "missing / 0 extra / 0 shape-mismatched params; the loader's '3 unmatched' warning "
        "is spurious) and is image-conditioned (output differs per image) BUT inaccurate "
        "(giraffe photo -> 'white background'; food photo -> 'a dog'). Preprocessing is "
        "standard SigLIP (224, 0.5 mean/std) and image_size/patch/posemb are consistent "
        "(224/16/196). The defect is in the vision-encoder forward / modality projector "
        "(pixel-shuffle token count num_image_tokens=49) or prompt template — needs "
        "activation-level comparison against the HF nanoVLM reference, not a guessed fix."
    )


def test_moondream2_describes_giraffes(giraffe_image_path):
    pytest.skip(
        "WS-2: Moondream2 (~1.8B) is IN SCOPE under the performance-based inclusion policy "
        "(fits the M4 memory budget; size is a guideline, not a gate). Remaining: verify "
        "real-weights load + image-conditioned generation produce a correct description."
    )


def test_tinyllava_describes_giraffes(giraffe_image_path):
    pytest.skip(
        "WS-2: TinyLLaVA-1.5B is IN SCOPE under the performance-based inclusion policy "
        "(~1.5B fits the M4 memory budget; size is a guideline, not a gate). Remaining "
        "fix: it currently echoes the prompt back ('What is in this image?' -> same) "
        "instead of generating an answer — a generation/prompt-template bug to resolve "
        "before it can assert real output."
    )


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def test_all_minilm_l6_v2_semantic(semantic_sentences):
    from smlx.models.all_MiniLM_L6_v2 import encode, load

    model, tokenizer = load("all-MiniLM-L6-v2")
    emb = np.asarray(encode(model, tokenizer, semantic_sentences), dtype=np.float32)
    A.assert_embeddings_semantic(
        emb,
        expected_dim=384,
        related_pair=(0, 1),
        unrelated_pair=(0, 2),
        context="all_MiniLM_L6_v2",
    )


# ---------------------------------------------------------------------------
# OCR / document
# ---------------------------------------------------------------------------


def test_trocr_small_reads_text():
    pytest.skip(
        "WS-2: convert microsoft/trocr-small-printed weights to MLX in the loader "
        "(default path currently can fall back to random)."
    )


def test_donut_base_reads_document():
    pytest.skip(
        "WS-2: convert naver-clova-ix/donut-base and wire BARTDecoder weights "
        "before asserting real OCR/doc output."
    )


# ---------------------------------------------------------------------------
# Audio: classification / VAD / TTS
# ---------------------------------------------------------------------------


def test_yamnet_classifies_sound():
    pytest.skip(
        "WS-2: confirm mlx-community/yamnet exists and convert (original is TF) "
        "before asserting top-1 label on a known ESC-50 clip."
    )


def test_silerovad_detects_speech(librispeech_clip):
    pytest.skip(
        "WS-2: implement ONNX->MLX conversion for silero/silero-vad "
        "(loader downloads ONNX but conversion is a stub -> random weights)."
    )


def test_chatterbox_synthesizes_intelligible_speech():
    pytest.skip(
        "WS-2: wire the real HiFi-GAN vocoder (placeholder today) and verify "
        "ResembleAI/chatterbox model weights, then assert ASR-round-trip intelligibility."
    )


def test_orpheus_synthesizes_intelligible_speech():
    pytest.skip(
        "WS-2 (gated): source real Orpheus-150M weights on HF (or swap to a smol TTS "
        "with weights); loader runs on random init today -> noise."
    )


# ---------------------------------------------------------------------------
# Text -> CAD
# ---------------------------------------------------------------------------


def test_smolgencad_generates_valid_cad():
    pytest.skip(
        "WS-2 (gated, highest effort): smolGenCad has no pretrained weights and must "
        "be trained (dataset + training run) before it can produce a valid CAD program."
    )
