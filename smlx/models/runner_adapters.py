"""Adapters registering every implemented model with the unified runner.

Each :class:`~smlx.models.runner.RunEntry` here wires a model's *own real*
``load`` + inference function into the runner. Imports are done lazily inside the
loader/runner closures so ``smlx run --list`` (and importing this module) does not
pull every model's heavy dependencies.

Every registered entry routes to a maintained upstream implementation (mlx-lm,
mlx-vlm, mlx-whisper, mlx-embeddings, mlx-audio, onnxruntime, transformers) or a
real deterministic implementation — there are **no bespoke hand-written forward
passes** in any path, so every entry reports ``TRAINED``. The bespoke SMLX impls
that mishandled real weights (Orpheus/Chatterbox TTS, TrOCR/Donut OCR, YAMNet
audio-cls, the untrained smolGenCad) are **quarantined** — not wired here — and
documented in ``docs/MODEL_STATUS.md``.
"""

from __future__ import annotations

from .runner import RunEntry, RunOutput, WeightStatus, register

# All runner entries now route to real upstream impls (mlx-lm/mlx-vlm/mlx-whisper/
# mlx-embeddings/mlx-audio/onnxruntime/transformers) or a real deterministic
# implementation, so every entry reports WeightStatus.TRAINED. The old
# weights_loaded-signal helper is gone with the bespoke adapters it served.


# --------------------------------------------------------------------------- #
# Language models                                                             #
# --------------------------------------------------------------------------- #


# Language models run through mlx-lm (correct upstream forward + chat templating),
# never the bespoke SmolLM2 forward.
_LM_BACKEND = {"smollm2-135m": "smollm2-135m", "smollm2-360m": "smollm2-360m"}


def _make_backend_lm(repo: str):
    def _load():
        from smlx.models import mlx_backend

        return mlx_backend.load(repo, backend=mlx_backend.Backend.MLX_LM)

    def _run(loaded, *, text, image=None, audio=None, document=None, max_tokens=64, **opts):
        from smlx.models import mlx_backend

        out = mlx_backend.generate(
            loaded, text, max_tokens=max_tokens, temperature=opts.get("temperature", 0.0)
        )
        return RunOutput(
            kind="text", status=WeightStatus.TRAINED, reason="mlx-lm", text=(out or "").strip()
        )

    return _load, _run


for _key, _repo in _LM_BACKEND.items():
    _lload, _lrun = _make_backend_lm(_repo)
    register(RunEntry(_key, "language", ("text",), _lload, _lrun, note=f"{_repo} (mlx-lm)"))


# --------------------------------------------------------------------------- #
# Vision-language models                                                       #
# --------------------------------------------------------------------------- #


# Every VLM entry runs through the real mlx-vlm backend (correct upstream
# forward + real weights), NOT the bespoke SMLX VLM code (which mangles even real
# weights). Each alias maps to a real, mlx-vlm-loadable checkpoint, verified to
# produce real output. moondream2's real arch isn't supported by mlx-vlm, so that
# entry honestly runs Qwen2-VL (note says so) rather than emit gibberish.
# moondream2 is QUARANTINED: real Moondream2 isn't supported by mlx-vlm, and the
# Qwen2-VL-4bit substitute is degenerate on basic vision (answers "blue" for every
# solid color) — it fails the correctness gate, so it is not wired here.
_VLM_BACKEND = {
    "smolvlm-256m": ("smolvlm-256m", "SmolVLM-256M (mlx-vlm)"),
    "smolvlm-500m": ("smolvlm-500m", "SmolVLM-500M (mlx-vlm)"),
    "nanovlm": ("mlx-community/nanoLLaVA-1.5-4bit", "nanoLLaVA-1.5 (mlx-vlm)"),
    "tinyllava": ("qnguyen3/nanoLLaVA", "nanoLLaVA / TinyLLaVA-class (mlx-vlm)"),
}


def _make_backend_vlm(repo: str, real_note: str):
    def _load():
        from smlx.models import mlx_backend

        return mlx_backend.load(repo, backend=mlx_backend.Backend.MLX_VLM)

    def _run(loaded, *, text, image, audio=None, document=None, max_tokens=64, **opts):
        from smlx.models import mlx_backend

        out = mlx_backend.generate(loaded, text, image=image, max_tokens=max_tokens)
        return RunOutput(
            kind="text", status=WeightStatus.TRAINED, reason=real_note, text=(out or "").strip()
        )

    return _load, _run


for _key, (_repo, _note) in _VLM_BACKEND.items():
    _vload, _vrun = _make_backend_vlm(_repo, _note)
    register(RunEntry(_key, "vlm", ("image", "text"), _vload, _vrun, note=_note))


# --------------------------------------------------------------------------- #
# ASR                                                                          #
# --------------------------------------------------------------------------- #


# ASR runs through mlx-whisper (Apple's maintained Whisper), never the bespoke
# Whisper_tiny forward. mlx_whisper.transcribe loads+caches the model by repo, so
# the loader just carries the repo id.
_WHISPER_REPO = "mlx-community/whisper-tiny"


def _whisper_loader():
    return _WHISPER_REPO


def _whisper_runner(loaded, *, text=None, image=None, audio, document=None, **opts):
    import mlx_whisper

    result = mlx_whisper.transcribe(
        audio, path_or_hf_repo=loaded, language=opts.get("language", "en")
    )
    return RunOutput(
        kind="text",
        status=WeightStatus.TRAINED,
        reason="mlx-whisper",
        text=(result.get("text") or "").strip(),
    )


register(
    RunEntry(
        "whisper-tiny",
        "asr",
        ("audio",),
        _whisper_loader,
        _whisper_runner,
        note="Whisper speech-to-text (mlx-whisper)",
    )
)


# --------------------------------------------------------------------------- #
# TTS                                                                          #
# --------------------------------------------------------------------------- #


# QUARANTINED: the bespoke Orpheus_150M / Chatterbox TTS impls have no public
# checkpoint and emit noise. They are NOT wired into the runner — `kokoro` is the
# real TTS. See docs/MODEL_STATUS.md "Quarantined".


# Real, trained-weights TTS via mlx-audio's Kokoro-82M. Unlike the SMLX
# Orpheus/Chatterbox impls (which have no public checkpoint and emit noise), this
# produces intelligible speech — verified by transcribing the output back with
# Whisper and matching the input text.
_KOKORO_REPO = "mlx-community/Kokoro-82M-bf16"
_kokoro_patched = False


def _patch_kokoro_istftnet() -> None:
    """Fix an upstream mlx-audio shape bug in Kokoro's harmonic source generator.

    ``SineGen.__call__`` builds ``sine_waves`` via ``_f02sine`` (which up- then
    down-samples) and ``uv`` directly from f0; the resample rounding can leave
    ``sine_waves`` a few frames longer than ``uv``, so ``sine_waves * uv + noise``
    raises ``broadcast_shapes`` on many inputs (e.g. "Hello there."). Align both
    to the shorter length before combining — a ~12 ms trim that makes synthesis
    robust. Verified: previously-failing prompts now transcribe back correctly.
    """
    global _kokoro_patched
    if _kokoro_patched:
        return
    import mlx.core as mx
    from mlx_audio.tts.models.kokoro import istftnet

    def _aligned_call(self, f0):
        fn = f0 * mx.arange(1, self.harmonic_num + 2)[None, None, :]
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        t = min(sine_waves.shape[1], uv.shape[1])
        sine_waves, uv = sine_waves[:, :t, :], uv[:, :t, :]
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * mx.random.normal(sine_waves.shape)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise

    istftnet.SineGen.__call__ = _aligned_call
    _kokoro_patched = True


def _kokoro_loader():
    from mlx_audio.tts.utils import load_model

    _patch_kokoro_istftnet()
    return load_model(_KOKORO_REPO)


def _kokoro_runner(loaded, *, text, image=None, audio=None, document=None, **opts):
    import contextlib
    import glob
    import io
    import os
    import tempfile

    import numpy as np
    import soundfile as sf
    from mlx_audio.tts.generate import generate_audio

    with tempfile.TemporaryDirectory() as td:
        # file_prefix is a path prefix; join_audio=True writes a single
        # "<prefix>.wav". (output_path is not the directory control here.)
        prefix = os.path.join(td, "kokoro")
        with contextlib.redirect_stdout(io.StringIO()):
            generate_audio(
                text=text,
                model=loaded,
                voice=opts.get("voice", "af_heart"),
                file_prefix=prefix,
                audio_format="wav",
                join_audio=True,
                verbose=False,
            )
        wavs = sorted(glob.glob(os.path.join(td, "*.wav")))
        if not wavs:
            raise RuntimeError("Kokoro produced no audio file")
        waveform, sr = sf.read(wavs[0])

    return RunOutput(
        kind="audio",
        status=WeightStatus.TRAINED,
        reason="Kokoro-82M (mlx-audio): intelligible speech",
        audio=(np.asarray(waveform, dtype=np.float32), int(sr)),
    )


register(
    RunEntry(
        "kokoro",
        "tts",
        ("text",),
        _kokoro_loader,
        _kokoro_runner,
        note="Kokoro-82M real TTS (mlx-audio) — trained, intelligible speech",
    )
)


# --------------------------------------------------------------------------- #
# OCR                                                                          #
# --------------------------------------------------------------------------- #


# Real OCR via a real VLM (mlx-vlm SmolVLM-500M) prompted to transcribe text.
# The bespoke TrOCR/Donut impls diverge architecturally from real TrOCR/BART and
# can't be made correct by weight-mapping; this is the working OCR path.
_OCR_REPO = "smolvlm-500m"
_OCR_PROMPT = (
    "Transcribe all the text in this image exactly, preserving line breaks. Output only the text."
)


def _ocr_vlm_loader():
    from smlx.models import mlx_backend

    return mlx_backend.load(_OCR_REPO)


def _ocr_vlm_runner(loaded, *, text=None, image=None, audio=None, document, **opts):
    from smlx.models import mlx_backend

    out = mlx_backend.generate(
        loaded, _OCR_PROMPT, image=document, max_tokens=opts.get("max_tokens", 256)
    )
    return RunOutput(
        kind="text",
        status=WeightStatus.TRAINED,
        reason="SmolVLM-500M OCR (mlx-vlm)",
        text=(out or "").strip(),
    )


register(
    RunEntry(
        "ocr",
        "ocr",
        ("document",),
        _ocr_vlm_loader,
        _ocr_vlm_runner,
        note="Real OCR via SmolVLM-500M (mlx-vlm) — transcribes document text",
    )
)


# QUARANTINED: the bespoke TrOCR_small / Donut_base impls diverge architecturally
# from real TrOCR/BART (extra layers with no checkpoint source, residual encoder
# bugs) and produce gibberish even with real weights. They are NOT wired into the
# runner — the `ocr` entry above (SmolVLM via mlx-vlm) is the real OCR path.
# See docs/MODEL_STATUS.md "Quarantined".


# --------------------------------------------------------------------------- #
# Embeddings                                                                   #
# --------------------------------------------------------------------------- #


# Embeddings run through mlx-embeddings (maintained MLX sentence-transformers),
# never the bespoke MiniLM forward. Both aliases are the same real model.
_EMBED_REPO = "mlx-community/all-MiniLM-L6-v2-4bit"


def _embed_loader():
    from mlx_embeddings.utils import load

    return load(_EMBED_REPO)  # (model, tokenizer)


def _embed_runner(loaded, *, text, image=None, audio=None, document=None, **opts):
    import numpy as np

    model, tokenizer = loaded
    sentences = text if isinstance(text, list) else [text]
    out = model(**tokenizer.batch_encode_plus(sentences, return_tensors="mlx", padding=True))
    vecs = np.asarray(out.text_embeds if hasattr(out, "text_embeds") else out[0])
    return RunOutput(
        kind="embeddings", status=WeightStatus.TRAINED, reason="mlx-embeddings", data=vecs
    )


for _ekey in ("minilm", "all-minilm-l6-v2"):
    register(
        RunEntry(
            _ekey,
            "embeddings",
            ("text",),
            _embed_loader,
            _embed_runner,
            note="all-MiniLM-L6-v2 sentence embeddings (mlx-embeddings)",
        )
    )


# --------------------------------------------------------------------------- #
# VAD                                                                          #
# --------------------------------------------------------------------------- #


# Real Silero VAD v5 via onnxruntime — the bespoke SMLX SileroVAD runs on random
# weights, and the `silero-vad` Python package imports torchaudio (broken ABI in
# some envs). Locate the package's bundled ONNX model WITHOUT importing it (so no
# torch is pulled), and run it directly. The v5 model requires a 64-sample context
# prepended to each 512-sample frame; without it the probabilities are ~0.


def _silero_onnx_path():
    import importlib.util
    import os

    spec = importlib.util.find_spec("silero_vad")  # does not execute the package
    for base in getattr(spec, "submodule_search_locations", None) or []:
        cand = os.path.join(base, "data", "silero_vad.onnx")
        if os.path.exists(cand):
            return cand
    # Fallback: fetch the ONNX from the Hub (no torch dependency).
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id="onnx-community/silero-vad", filename="onnx/model.onnx")


def _silero_loader():
    import onnxruntime as ort

    return ort.InferenceSession(_silero_onnx_path())


def _load_audio_16k(audio):
    import librosa
    import numpy as np

    if isinstance(audio, str):
        import soundfile as sf

        arr, sr = sf.read(audio)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
    else:
        arr = np.asarray(audio, dtype=np.float32)
        sr = 16000
    if sr != 16000:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    return arr, 16000


def _silero_runner(loaded, *, text=None, image=None, audio, document=None, **opts):
    import numpy as np

    sess = loaded
    arr, sr = _load_audio_16k(audio)
    win, ctx, thresh = 512, 64, float(opts.get("threshold", 0.5))
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, ctx), dtype=np.float32)
    probs = []
    for i in range(0, len(arr) - win + 1, win):
        chunk = arr[i : i + win][None, :].astype(np.float32)
        x = np.concatenate([context, chunk], axis=1)  # v5 needs 64-sample context
        out = sess.run(None, {"input": x, "state": state, "sr": np.array(sr, dtype=np.int64)})
        probs.append(float(out[0].reshape(-1)[0]))
        state = out[1]
        context = chunk[:, -ctx:]
    # Group consecutive speech frames into (start, end) second segments.
    speech = np.array(probs) > thresh
    segments, start = [], None
    for j, s in enumerate(speech):
        t = j * win / sr
        if s and start is None:
            start = t
        elif not s and start is not None:
            segments.append({"start": round(start, 2), "end": round(t, 2)})
            start = None
    if start is not None:
        segments.append({"start": round(start, 2), "end": round(len(speech) * win / sr, 2)})
    return RunOutput(
        kind="segments",
        status=WeightStatus.TRAINED,
        reason=f"Silero VAD v5 (onnx): {len(segments)} speech segment(s)",
        data=segments,
    )


register(
    RunEntry(
        "silero-vad",
        "vad",
        ("audio",),
        _silero_loader,
        _silero_runner,
        note="Real Silero VAD v5 (onnxruntime) — speech segments",
    )
)


# --------------------------------------------------------------------------- #
# Audio classification                                                         #
# --------------------------------------------------------------------------- #


# Real audio-event classification via the maintained AST AudioSet model
# (transformers). The bespoke YAMNet's log-mel front-end didn't match its weights
# (classified speech as "Timpani"); AST classifies speech as "Speech" with high
# confidence. The YAMNet package is quarantined (not wired here).
_AST_REPO = "MIT/ast-finetuned-audioset-10-10-0.4593"


def _ast_loader():
    from transformers import ASTForAudioClassification, AutoFeatureExtractor

    fe = AutoFeatureExtractor.from_pretrained(_AST_REPO)
    model = ASTForAudioClassification.from_pretrained(_AST_REPO).eval()
    return (model, fe)


def _ast_runner(loaded, *, text=None, image=None, audio, document=None, **opts):
    import torch

    model, fe = loaded
    arr, sr = _load_audio_16k(audio)  # reuse the VAD helper (16k mono float)
    inputs = fe(arr, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    top = torch.topk(probs, int(opts.get("top_k", 5)))
    payload = [
        {"label": model.config.id2label[int(i)], "score": round(float(p), 4)}
        for p, i in zip(top.values, top.indices)
    ]
    return RunOutput(
        kind="labels",
        status=WeightStatus.TRAINED,
        reason="AST AudioSet (transformers)",
        data=payload,
    )


register(
    RunEntry(
        "ast",
        "audio_cls",
        ("audio",),
        _ast_loader,
        _ast_runner,
        note="AST AudioSet classifier (transformers) — real audio-event labels",
    )
)


# --------------------------------------------------------------------------- #
# CAD                                                                          #
# --------------------------------------------------------------------------- #


# Real text->CAD via a deterministic parser that emits valid CadQuery (verified
# by executing it). The smolGenCad neural model has no public checkpoint (random
# output); this produces genuine correct CAD for the supported primitives. An
# unsupported spec raises CADParseError, surfaced honestly as an ERROR result.
def _cad_loader():
    return "text_to_cad"  # stateless parser; nothing heavy to load


def _cad_runner(loaded, *, text, image=None, audio=None, document=None, **opts):
    import json

    from smlx.models.smolGenCad.text_to_cad import generate as cad_generate

    r = cad_generate(text, validate=True)
    payload = {
        "sequence_json": json.dumps(
            {"primitive": r["primitive"], "params": r["params"], "bbox": r["bbox"]}, indent=2
        ),
        "python": r["python"],
        "n_commands": 1,
        "summary": f"{r['primitive']} {r['params']} bbox={r['bbox']}",
    }
    return RunOutput(
        kind="cad",
        status=WeightStatus.TRAINED,
        reason=f"deterministic text->CadQuery ({r['primitive']})",
        data=payload,
    )


register(
    RunEntry(
        "cad",
        "cad",
        ("text",),
        _cad_loader,
        _cad_runner,
        note="Real text->CAD: deterministic parser -> valid CadQuery (cylinder/box/sphere/cone)",
    )
)
