# Plan: 100% real output — zero bespoke forward passes, fail-closed correctness gate

**Date:** 2026-06-08
**Branch (suggested):** `feat/all-models-real`
**Status:** Planned (awaiting execution)

## Thesis

Every entry in `smlx run --list` must produce **real, verified-correct** output, or
be **quarantined** (excluded from the runner and the verified list) — never faked.
The root cause of garbage output is the **bespoke hand-reimplemented forward passes**
in `smlx/models/<Name>/`: they load real weights but misplace them (wrong key
names), leave slots empty (extra layers with no checkpoint), feed them wrong
(tensor layout / missing `groups`), or describe the input wrong (feature
extraction). The fix is the pivot, applied to **100%** of entries: route to a
maintained upstream implementation (mlx-lm / mlx-whisper / mlx-vlm / mlx-audio /
mlx-embeddings / onnxruntime / transformers) or a **real deterministic**
implementation — and **no entry may depend on a bespoke forward pass.**

A hard, fail-closed `smlx run verify` gate (run in CI) proves correctness per
modality and **fails** if any listed model isn't verified-correct.

## Decisions (from planning)

1. **Route everything to upstream** — including the currently-working-but-bespoke
   LM / ASR / embeddings paths. Zero hand-written forwards in any runner path.
2. **Audio-cls → a real classifier** (not the bespoke YAMNet feature front-end).
3. **CAD → a real deterministic text→CAD** parser emitting valid CadQuery.
4. **Hard fail-closed gate** on `smlx run --list`.

## Current state (19 entries on `main`)

| Entry | Modality | Backing today | Verdict |
|-------|----------|---------------|---------|
| smollm2-135m/360m | language | **bespoke** `SmolLM2_135M.generate` | re-route → mlx-lm |
| whisper-tiny | asr | **bespoke** `Whisper_tiny.transcribe` | re-route → mlx-whisper |
| minilm / all-minilm-l6-v2 | embeddings | **bespoke** `MiniLM.encode` | re-route → mlx-embeddings |
| smolvlm-256m/500m, nanovlm, tinyllava, moondream2 | vlm | mlx-vlm ✓ | keep (real) |
| ocr | ocr | mlx-vlm (SmolVLM-500M) ✓ | keep (real) |
| kokoro | tts | mlx-audio ✓ | keep (real) |
| silero-vad | vad | onnxruntime ✓ | keep (real) |
| yamnet | audio_cls | bespoke (feature gap) | replace → real classifier |
| smolgencad | cad | bespoke, no weights | replace → rule-based CAD |
| trocr-small, donut-base | ocr | bespoke, broken | **quarantine** (ocr entry covers OCR) |
| orpheus-150m, chatterbox | tts | bespoke, random noise | **quarantine** (kokoro covers TTS) |

## Workstreams

### WS-1 — Route LM / ASR / embeddings to upstream (kill bespoke forwards)
1. **LM**: runner LM adapter → `mlx_backend` (mlx-lm). Remove `from smlx.models.SmolLM2_135M import generate` from the runner path (`runner_adapters.py:66`).
2. **ASR**: add `mlx-whisper` (Apple's maintained Whisper). Route `whisper-tiny` adapter + `mlx_backend` ASR (`Backend.SMLX`→mlx-whisper). Replace bespoke `Whisper_tiny.transcribe`.
3. **Embeddings**: add `mlx-embeddings`; route `minilm`/`all-minilm` + `mlx_backend` embeddings to it. Replace bespoke `MiniLM.encode`.
4. After WS-1, grep proves **no runner adapter imports a bespoke `smlx.models.<Model>` forward** (only mlx-lm/vlm/audio/whisper/embeddings/onnx).

### WS-2 — Audio-cls: real classifier
5. Wire a real audio-event classifier. Primary: **transformers AST**
   (`MIT/ast-finetuned-audioset-10-10-0.4593`) — torch is installed; compute the
   mel/fbank features with numpy/librosa to **avoid the broken torchaudio**.
   Verify on known sounds (speech→"Speech", etc.). If AST can't run torch-free,
   fall back to a real MLX audio classifier or quarantine (documented).
6. Replace the `yamnet` adapter; the bespoke YAMNet package is quarantined.

### WS-3 — CAD: real deterministic text→CAD
7. Add a **real parser** `smlx/models/smolGenCad/text_to_cad.py`: parse a text spec
   (primitives + dims: cylinder/box/sphere/cone, radius/height/width/…, fillet,
   extrude) into a **valid CadQuery** program and JSON. Deterministic, correct,
   real output (not ML, but not fake). Verify the emitted Python executes under
   `cadquery` (add as dep) and yields a solid with expected bounding box.
8. Route the `smolgencad` runner entry to this parser; the untrained model is
   quarantined.

### WS-4 — Quarantine the un-real bespoke entries
9. Remove `trocr-small`, `donut-base`, `orpheus-150m`, `chatterbox` from the runner
   registry (OCR is covered by `ocr`; TTS by `kokoro`). Keep the packages on disk
   but **not** in `smlx run --list` / the verified set. Document them in
   MODEL_STATUS under "Quarantined — superseded, not wired (reason)".
10. `smlx run --list` shows ONLY verified-real entries.

### WS-5 — Hard correctness gate (`smlx run verify`)
11. New `smlx run verify [--all]` runs a **real per-modality correctness check**,
    not just "output happens":
    - **LM/VLM**: answer-contains assertion on a known prompt (e.g. capital-of-France; giraffe image → "giraffe").
    - **TTS**: synth → **Whisper transcribes back** → fuzzy-match the input text.
    - **ASR**: transcribe a known clip → match reference (WER threshold).
    - **OCR**: render known text → exact/normalized match.
    - **VAD**: labeled speech+silence → segments overlap speech, exclude silence.
    - **audio-cls**: known sound (speech) → expected top label.
    - **embeddings**: paraphrase pair more similar than unrelated pair.
    - **CAD**: parse spec → CadQuery executes → expected solid/bbox.
12. Exits non-zero if **any** listed model fails. Quarantined models are excluded
    (never run, never faked). Status taxonomy stays honest (TRAINED only when the
    correctness check passes).

### WS-6 — Deps + docs
13. Add to `audio`/core extras: `mlx-whisper`, `mlx-embeddings`, `cadquery`, and the
    AST classifier path (transformers already present). Keep a coherent set
    (`pip check` clean).
14. Regenerate `docs/MODEL_STATUS.md` from the verified gate output; update README;
    list quarantined models with reasons.

### WS-7 — Tests
15. `tests/models/test_correctness.py`: one real correctness test per modality
    (the WS-5 checks), marked `requires_model`. Offline unit tests assert the
    registry contains only verified-real entries and that no adapter imports a
    bespoke forward.

## Correctness definitions (how "real" is proven)

| Modality | Proof |
|----------|-------|
| LM | known-answer contains (deterministic, temp 0) |
| VLM | image→answer contains expected entity |
| ASR | WER ≤ threshold vs reference transcript |
| TTS | synth→Whisper round-trip fuzzy-matches input |
| OCR | rendered text exact/normalized match |
| VAD | segment IoU vs labeled speech regions |
| audio-cls | top-label matches known sound |
| embeddings | cos(paraphrase) > cos(unrelated) |
| CAD | emitted CadQuery executes; bbox within tolerance |

## Risks / unknowns (resolved empirically in execution)

- **AST torch-free features**: AST's default extractor uses torchaudio (broken
  ABI). Must compute fbank via numpy/librosa or find a real MLX classifier; if
  neither works, audio-cls is quarantined with an honest note (not faked).
- **Dependency coherence**: new deps (mlx-whisper, mlx-embeddings, cadquery) must
  not re-break the transformers-5 / mlx-vlm-0.6 set; verify `pip check` after each.
- **Downloads/time**: the gate downloads several models; keep `requires_model`
  separate from offline unit tests.

## Acceptance criteria

- `grep` proves **no runner adapter calls a bespoke `smlx.models.<Model>` forward**.
- `smlx run verify --all` is **green**: every listed model passes its real
  correctness check; exits non-zero if any fails.
- `smlx run --list` contains only verified-real entries; quarantined models
  (trocr/donut/orpheus/chatterbox, untrained smolgencad/yamnet) are excluded and
  documented with reasons.
- audio-cls and CAD produce real correct output (real classifier; real CadQuery),
  or are honestly quarantined if a real path is proven impossible.
- `pip check` clean; MODEL_STATUS/README regenerated from the gate.
