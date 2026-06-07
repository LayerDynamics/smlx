"""Real-output assertions for smoke tests — the anti-placeholder contract.

Each helper raises ``AssertionError`` when a model's output looks like the
product of random / placeholder weights rather than a correctly loaded model.
The checks are deliberately stronger than "output is non-empty": coherence for
text, word-error-rate for ASR, semantic ranking for embeddings, intelligibility
(via ASR round-trip) for TTS, normalized match for OCR, and label match for
classification.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Text (LM / VLM)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def assert_text_coherent(text: str, *, min_words: int = 4, context: str = "") -> None:
    """Assert generated text reads like real language, not random-weight gibberish.

    Random/untrained language models collapse to one of a few failure modes:
    empty/whitespace, a single repeated token, or non-word character soup. This
    checks for a minimum number of distinct dictionary-shaped words and guards
    against degenerate repetition.
    """
    where = f" ({context})" if context else ""
    assert text is not None, f"output is None{where}"
    stripped = text.strip()
    assert stripped, f"output is empty/whitespace{where}"

    words = _WORD_RE.findall(stripped)
    assert (
        len(words) >= min_words
    ), f"only {len(words)} word-like tokens (need >= {min_words}){where}: {stripped[:120]!r}"

    # Guard against degenerate repetition (e.g. "the the the ..." or "AAAA").
    lowered = [w.lower() for w in words]
    most_common, count = Counter(lowered).most_common(1)[0]
    assert (
        count / len(lowered) < 0.6
    ), f"token {most_common!r} is {count}/{len(lowered)} of output — degenerate repetition{where}"

    # Printable, mostly-ASCII-letters content (random weights often emit control/CJK soup).
    letters = sum(c.isalpha() for c in stripped)
    assert (
        letters / max(len(stripped), 1) > 0.45
    ), f"only {letters}/{len(stripped)} chars are letters — likely not real text{where}"


def assert_contains_any(text: str, expected: Sequence[str], *, context: str = "") -> None:
    """Assert the (lowercased) text contains at least one expected keyword.

    Used where we know roughly what the answer should mention (e.g. a caption of
    a giraffe photo should say 'giraffe')."""
    where = f" ({context})" if context else ""
    low = (text or "").lower()
    assert any(
        kw.lower() in low for kw in expected
    ), f"none of {list(expected)} found in output{where}: {text[:160]!r}"


# ---------------------------------------------------------------------------
# ASR (speech -> text)
# ---------------------------------------------------------------------------


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Normalized word error rate via SMLX's own evaluator."""
    from smlx.evals.audio_eval import AudioEvaluator

    return AudioEvaluator().evaluate(reference, hypothesis).wer


def assert_transcription(
    reference: str, hypothesis: str, *, max_wer: float, context: str = ""
) -> float:
    """Assert an ASR hypothesis matches the reference within a WER ceiling."""
    where = f" ({context})" if context else ""
    wer = word_error_rate(reference, hypothesis)
    assert (
        wer <= max_wer
    ), f"WER {wer:.2%} > ceiling {max_wer:.2%}{where}\n  ref: {reference!r}\n  hyp: {hypothesis!r}"
    return wer


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-12
    return float(np.dot(a, b) / denom)


def assert_embeddings_semantic(
    embeddings: np.ndarray,
    *,
    expected_dim: int,
    related_pair: tuple[int, int],
    unrelated_pair: tuple[int, int],
    context: str = "",
) -> None:
    """Assert embeddings have the right dimension and capture real semantics.

    A correctly loaded embedding model places a related sentence pair closer
    (higher cosine) than an unrelated pair; random weights do not.
    """
    where = f" ({context})" if context else ""
    emb = np.asarray(embeddings, dtype=np.float32)
    assert emb.ndim == 2, f"embeddings must be 2-D{where}, got shape {emb.shape}"
    assert (
        emb.shape[1] == expected_dim
    ), f"embedding dim {emb.shape[1]} != expected {expected_dim}{where}"
    assert np.isfinite(emb).all(), f"embeddings contain non-finite values{where}"

    related = _cos(emb[related_pair[0]], emb[related_pair[1]])
    unrelated = _cos(emb[unrelated_pair[0]], emb[unrelated_pair[1]])
    assert related > unrelated, (
        f"related similarity {related:.3f} !> unrelated {unrelated:.3f} — "
        f"embeddings not capturing meaning{where}"
    )


# ---------------------------------------------------------------------------
# Audio / TTS
# ---------------------------------------------------------------------------


def audio_rms(audio: np.ndarray) -> float:
    a = np.asarray(audio, dtype=np.float32).ravel()
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(a**2)))


def assert_audio_has_signal(audio: np.ndarray, *, min_rms: float = 1e-3, context: str = "") -> None:
    """Assert audio is not silence/DC — a cheap first gate (does NOT prove speech)."""
    where = f" ({context})" if context else ""
    a = np.asarray(audio, dtype=np.float32).ravel()
    assert a.size > 0, f"empty audio{where}"
    assert np.isfinite(a).all(), f"audio has non-finite samples{where}"
    rms = audio_rms(a)
    assert rms > min_rms, f"audio RMS {rms:.2e} below {min_rms:.2e} — silence{where}"
    assert float(a.std()) > 1e-4, f"audio is ~constant (DC), not a waveform{where}"


def assert_speech_intelligible(
    audio: np.ndarray,
    sampling_rate: int,
    expected_text: str,
    *,
    max_wer: float = 0.6,
    context: str = "",
) -> float:
    """Assert synthesized speech is intelligible by transcribing it with Whisper.

    This is the real anti-noise gate for TTS: placeholder/random vocoders emit
    energy (so an RMS check passes) but no words. Transcribing the synthesized
    clip and requiring a bounded WER against the intended text proves the model
    produced actual speech, not noise.
    """
    where = f" ({context})" if context else ""
    assert_audio_has_signal(audio, context=context)

    import librosa

    from smlx.models.Whisper_tiny import load as load_whisper
    from smlx.models.Whisper_tiny import transcribe

    a = np.asarray(audio, dtype=np.float32).ravel()
    if sampling_rate != 16000:
        a = librosa.resample(a, orig_sr=sampling_rate, target_sr=16000)

    model, tokenizer = load_whisper("mlx-community/whisper-tiny")
    result = transcribe(a, model, tokenizer, language="en", verbose=None)
    hypothesis = (result.get("text") or "").strip()
    assert hypothesis, f"synthesized audio transcribed to nothing — not speech{where}"
    return assert_transcription(expected_text, hypothesis, max_wer=max_wer, context=context)


# ---------------------------------------------------------------------------
# OCR / classification
# ---------------------------------------------------------------------------


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def assert_ocr_match(
    prediction: str, expected: str, *, min_overlap: float = 0.6, context: str = ""
) -> None:
    """Assert an OCR prediction matches expected text (token-overlap tolerant)."""
    where = f" ({context})" if context else ""
    pred = _normalize_text(prediction)
    exp = _normalize_text(expected)
    assert pred, f"OCR produced empty text{where}"
    if exp in pred or pred in exp:
        return
    exp_tokens = set(exp.split())
    pred_tokens = set(pred.split())
    overlap = len(exp_tokens & pred_tokens) / max(len(exp_tokens), 1)
    assert (
        overlap >= min_overlap
    ), f"OCR overlap {overlap:.0%} < {min_overlap:.0%}{where}\n  exp: {exp!r}\n  got: {pred!r}"


def assert_label_in(prediction, expected_labels: Sequence[str], *, context: str = "") -> None:
    """Assert a predicted class label is among the acceptable ground-truth labels."""
    where = f" ({context})" if context else ""
    pred = _normalize_text(str(prediction))
    accepted = {_normalize_text(x) for x in expected_labels}
    assert any(
        pred == a or a in pred or pred in a for a in accepted
    ), f"predicted label {prediction!r} not in {list(expected_labels)}{where}"


def assert_perf_floor(measured: float, floor: float, *, unit: str, context: str = "") -> None:
    """Assert a measured throughput/speed meets its committed floor."""
    where = f" ({context})" if context else ""
    assert math.isfinite(measured), f"non-finite perf measurement{where}"
    assert measured >= floor, f"perf {measured:.2f} {unit} below floor {floor:.2f} {unit}{where}"
