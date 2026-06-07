"""Adapters registering every implemented model with the unified runner.

Each :class:`~smlx.models.runner.RunEntry` here wires a model's *own real*
``load`` + inference function into the runner. Imports are done lazily inside the
loader/runner closures so ``smlx run --list`` (and importing this module) does not
pull every model's heavy dependencies.

Weight status is derived at runtime from a real ``model.weights_loaded`` signal
where the loader exposes one (Orpheus, Chatterbox, Donut, SileroVAD, smolGenCad);
models that load a real public checkpoint report ``TRAINED``; TrOCR reports a
``TRAINED-WEIGHTS`` gap (decoder MLP params absent from the checkpoint, verified —
output is repetitive). Nothing is hardcoded to claim quality it doesn't have.
"""

from __future__ import annotations

from typing import Any

from .runner import RunEntry, RunOutput, WeightStatus, register


def _model_of(loaded: Any) -> Any:
    """The nn.Module from a load() result that may be a tuple (model, ...)."""
    if isinstance(loaded, tuple):
        return loaded[0]
    return loaded


def _status(
    loaded: Any,
    *,
    gap: bool = False,
    gap_reason: str = "",
    untrained_reason: str = "random weights: output is structural, not trained",
) -> tuple:
    """Map the model's real weights_loaded signal to (WeightStatus, reason)."""
    wl = getattr(_model_of(loaded), "weights_loaded", None)
    if wl is False:
        return WeightStatus.PIPELINE_ONLY, untrained_reason
    if gap:
        return WeightStatus.TRAINED_GAP, gap_reason
    return WeightStatus.TRAINED, ""


def _open_image(path: str):
    from PIL import Image

    return Image.open(path).convert("RGB")


# --------------------------------------------------------------------------- #
# Language models                                                             #
# --------------------------------------------------------------------------- #


def _lm_loader(pkg: str):
    def _load():
        import importlib

        return importlib.import_module(f"smlx.models.{pkg}").load()

    return _load


def _lm_runner(loaded, *, text, image=None, audio=None, document=None, max_tokens=64, **opts):
    from smlx.models.SmolLM2_135M import generate  # shared generate signature

    model, tokenizer = loaded
    # Apply the instruct chat template so the model answers the prompt instead of
    # immediately emitting <|im_end|> on a bare, un-templated string.
    prompt = text
    if getattr(tokenizer, "chat_template", None) is not None:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
        )
    out = generate(
        model, tokenizer, prompt, max_tokens=max_tokens, temperature=opts.get("temperature", 0.0)
    )
    # Strip the chat-template end marker if the generator left it in the text.
    out = (out or "").replace("<|im_end|>", "").strip()
    status, reason = _status(loaded)
    return RunOutput(kind="text", status=status, reason=reason, text=out)


for _key, _pkg in (("smollm2-135m", "SmolLM2_135M"), ("smollm2-360m", "SmolLM2_360M")):
    register(
        RunEntry(
            _key,
            "language",
            ("text",),
            _lm_loader(_pkg),
            _lm_runner,
            note=f"{_pkg} text generation",
        )
    )


# --------------------------------------------------------------------------- #
# Vision-language models                                                       #
# --------------------------------------------------------------------------- #


def _vlm_loader(pkg: str):
    def _load():
        import importlib

        return importlib.import_module(f"smlx.models.{pkg}").load()

    return _load


# Every VLM entry runs through the real mlx-vlm backend (correct upstream
# forward + real weights), NOT the bespoke SMLX VLM code (which mangles even real
# weights). Each alias maps to a real, mlx-vlm-loadable checkpoint, verified to
# produce real output. moondream2's real arch isn't supported by mlx-vlm, so that
# entry honestly runs Qwen2-VL (note says so) rather than emit gibberish.
_VLM_BACKEND = {
    "smolvlm-256m": ("smolvlm-256m", "SmolVLM-256M (mlx-vlm)"),
    "smolvlm-500m": ("smolvlm-500m", "SmolVLM-500M (mlx-vlm)"),
    "nanovlm": ("mlx-community/nanoLLaVA-1.5-4bit", "nanoLLaVA-1.5 (mlx-vlm)"),
    "tinyllava": ("qnguyen3/nanoLLaVA", "nanoLLaVA / TinyLLaVA-class (mlx-vlm)"),
    "moondream2": ("qwen2-vl-2b", "Qwen2-VL-2B (mlx-vlm) — real Moondream2 unavailable in MLX"),
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


def _whisper_runner(loaded, *, text=None, image=None, audio, document=None, **opts):
    from smlx.models.Whisper_tiny import transcribe

    model, tokenizer = loaded
    result = transcribe(audio, model, tokenizer, language=opts.get("language", "en"), verbose=None)
    status, reason = _status(loaded)
    return RunOutput(
        kind="text", status=status, reason=reason, text=(result.get("text") or "").strip()
    )


register(
    RunEntry(
        "whisper-tiny",
        "asr",
        ("audio",),
        _vlm_loader("Whisper_tiny"),
        _whisper_runner,
        note="Whisper speech-to-text",
    )
)


# --------------------------------------------------------------------------- #
# TTS                                                                          #
# --------------------------------------------------------------------------- #


def _make_tts_runner(pkg: str, sample_rate: int, untrained_reason: str):
    def _run(loaded, *, text, image=None, audio=None, document=None, **opts):
        import importlib

        syn = importlib.import_module(f"smlx.models.{pkg}").synthesize
        model, processor = loaded
        waveform = syn(model, processor, text, sample_rate=sample_rate)
        status, reason = _status(loaded, untrained_reason=untrained_reason)
        return RunOutput(kind="audio", status=status, reason=reason, audio=(waveform, sample_rate))

    return _run


register(
    RunEntry(
        "orpheus-150m",
        "tts",
        ("text",),
        _vlm_loader("Orpheus_150M"),
        _make_tts_runner("Orpheus_150M", 24000, "random weights: noise, not speech"),
        note="Orpheus text-to-speech",
    )
)
register(
    RunEntry(
        "chatterbox",
        "tts",
        ("text",),
        _vlm_loader("Chatterbox"),
        _make_tts_runner("Chatterbox", 24000, "random weights: noise, not speech"),
        note="Chatterbox text-to-speech",
    )
)


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


def _trocr_loader():
    from smlx.models.TrOCR_small import load

    return load("printed")


def _trocr_runner(loaded, *, text=None, image=None, audio=None, document, **opts):
    from smlx.models.TrOCR_small import recognize

    model, processor = loaded
    out = recognize(model, processor, document)
    # TrOCR loads real microsoft/trocr-small-printed weights but the decoder MLP
    # params are absent from the checkpoint (verified ~75 missing) -> repetitive.
    status, reason = _status(
        loaded, gap=True, gap_reason="decoder MLP params absent from checkpoint -> repetitive"
    )
    return RunOutput(kind="text", status=status, reason=reason, text=out)


register(
    RunEntry(
        "trocr-small",
        "ocr",
        ("document",),
        _trocr_loader,
        _trocr_runner,
        note="TrOCR printed-text OCR",
    )
)


def _donut_runner(loaded, *, text=None, image=None, audio=None, document, **opts):
    from smlx.models.Donut_base import generate

    model, processor = loaded
    out = generate(model, processor, document, prompt=opts.get("prompt", ""))
    status, reason = _status(loaded)
    return RunOutput(kind="text", status=status, reason=reason, text=out)


register(
    RunEntry(
        "donut-base",
        "ocr",
        ("document",),
        _vlm_loader("Donut_base"),
        _donut_runner,
        note="Donut document understanding",
    )
)


# --------------------------------------------------------------------------- #
# Embeddings                                                                   #
# --------------------------------------------------------------------------- #


def _make_embed_runner(pkg: str):
    def _run(loaded, *, text, image=None, audio=None, document=None, **opts):
        import importlib

        encode = importlib.import_module(f"smlx.models.{pkg}").encode
        model, tokenizer = loaded
        vecs = encode(model, tokenizer, text)
        status, reason = _status(loaded)
        return RunOutput(kind="embeddings", status=status, reason=reason, data=vecs)

    return _run


register(
    RunEntry(
        "minilm",
        "embeddings",
        ("text",),
        _vlm_loader("MiniLM"),
        _make_embed_runner("MiniLM"),
        note="MiniLM sentence embeddings",
    )
)
register(
    RunEntry(
        "all-minilm-l6-v2",
        "embeddings",
        ("text",),
        _vlm_loader("all_MiniLM_L6_v2"),
        _make_embed_runner("all_MiniLM_L6_v2"),
        note="all-MiniLM-L6-v2 embeddings",
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


def _yamnet_loader():
    from smlx.models.YAMNet import load

    return load()


def _yamnet_runner(loaded, *, text=None, image=None, audio, document=None, **opts):
    from smlx.models.YAMNet import classify

    preds = classify(loaded, audio, top_k=opts.get("top_k", 5))
    payload = [{"label": p.label, "score": float(p.score)} for p in preds]
    # Real weights load and the (now-fixed) MobileNet forward runs end to end, but
    # the bespoke log-mel feature extraction does not yet match YAMNet's expected
    # input, so the predicted labels are not yet reliable. Honest gap, not random.
    return RunOutput(
        kind="labels",
        status=WeightStatus.TRAINED_GAP,
        reason="real weights + forward; log-mel feature extraction mismatch -> labels unreliable",
        data=payload,
    )


register(
    RunEntry(
        "yamnet",
        "audio_cls",
        ("audio",),
        _yamnet_loader,
        _yamnet_runner,
        note="YAMNet audio-event classification (real weights; feature-extraction gap)",
    )
)


# --------------------------------------------------------------------------- #
# CAD                                                                          #
# --------------------------------------------------------------------------- #


def _cad_loader():
    from smlx.models.smolGenCad import load

    return load()


def _cad_runner(loaded, *, text, image=None, audio=None, document=None, max_tokens=64, **opts):
    from smlx.models.smolGenCad import generate
    from smlx.models.smolGenCad.generate import sequence_to_json, sequence_to_python

    model, text_tok, cad_tok = loaded
    seq = generate(
        model,
        text_tok,
        cad_tok,
        text,
        max_new_tokens=max_tokens,
        temperature=opts.get("temperature", 0.0),
    )
    payload = {
        "sequence_json": sequence_to_json(seq),
        "python": sequence_to_python(seq),
        "n_commands": len(seq),
    }
    status, reason = _status(loaded, untrained_reason="random weights: CAD content not meaningful")
    return RunOutput(kind="cad", status=status, reason=reason, data=payload)


register(
    RunEntry(
        "smolgencad", "cad", ("text",), _cad_loader, _cad_runner, note="smolGenCad text-to-CAD"
    )
)
