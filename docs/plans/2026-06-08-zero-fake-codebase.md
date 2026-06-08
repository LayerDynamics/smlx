# Plan: Zero fake / not-real code in the SMLX codebase

**Date:** 2026-06-08
**Branch (suggested):** `feat/zero-fake-codebase`
**Status:** Planned (awaiting execution)

## Mandate

There is to be **zero fake / not-real code anywhere** in the repo — not just in
`smlx run`. Every public code path that loads or runs a model must produce real,
correct output or fail honestly; nothing may return noise/gibberish/placeholder
data. The bespoke hand-reimplemented model packages (which mishandle real weights)
are **deleted**, every API re-points to the real upstream/deterministic path, and a
fail-closed correctness gate covers all public surfaces.

**Hard constraint:** do **not** add rule/marketing comments to the code ("this is
real", "no fake", "NO PLACEHOLDERS", etc.). The code is just correct; it doesn't
announce it.

## Decisions (from planning)

1. **Delete bespoke, keep real** for examples/tests (all 46 examples + 60 tests use
   bespoke per-model APIs → replace with real-API examples; keep/fix only real tests).
2. **`load_model`** becomes a thin real wrapper; deleting the fakes removes the dead
   aliases, so the registry only contains real entries (no garbage to route).
3. **Moondream2** — investigate a real MLX checkpoint/quant; wire only if it passes
   the gate, else leave out.
4. **Gate covers all public paths** (`smlx run`, `load_model`, `mlx_backend`,
   server audio/embeddings routes) — one fail-closed source of truth.

## Current real surface (keep)

- `smlx/models/runner.py` + `runner_adapters.py` + `runner_verify.py` — the verified
  runner (14/14). Routes to mlx-lm/mlx-vlm/mlx-whisper/mlx-embeddings/mlx-audio/
  onnxruntime/transformers + the deterministic CAD parser.
- `smlx/models/mlx_backend.py` — real for LM/VLM; **must fix** ASR/embeddings.
- `smlx/models/smolGenCad/text_to_cad.py` — standalone real CAD parser (move it).

## To delete (bespoke, garbage-or-hand-written forward passes)

`smlx/models/`: `SmolLM2_135M`, `SmolLM2_360M`, `SmolVLM_256M`,
`SmolVLM_500M_Instruct`, `nanoVLM`, `Moondream2`, `TinyLLaVA`, `Whisper_tiny`,
`Chatterbox`, `Orpheus_150M`, `TrOCR_small`, `Donut_base`, `MiniLM`,
`all_MiniLM_L6_v2`, `SileroVAD`, `YAMNet`, `smolGenCad` (after rescuing
`text_to_cad.py`). Plus `smlx/models/common/` only if unused after deletion.

## Workstreams

### WS-1 — Rescue the real CAD parser
1. Move `smolGenCad/text_to_cad.py` → `smlx/models/cad.py` (it only needs `re`
   + `cadquery`). Update the runner import (`runner_adapters._cad_runner`).

### WS-2 — Route mlx_backend ASR/embeddings to upstream
2. `mlx_backend.load`/`transcribe`/`embed`: ASR → mlx-whisper, embeddings →
   mlx-embeddings (drop the `Backend.SMLX` → `Whisper_tiny`/`MiniLM` branch).
   Keep the BackendModel API stable.

### WS-3 — Fix server routes
3. `server/routes/audio.py` (`Whisper_tiny.transcribe`) → mlx-whisper;
   `server/routes/embeddings.py` (`MiniLM.encode_single`) → mlx-embeddings.
   Remove the `NotImplementedError` fallbacks; the real path always works.

### WS-4 — Rewrite the legacy loader
4. `registry.py`: `MODEL_REGISTRY` keeps only real aliases; `load_model()` becomes
   a thin wrapper that delegates to `mlx_backend.load` (LM/VLM/ASR/embeddings) and
   the runner for the deterministic/real-extra modalities. Unknown/removed aliases
   raise a clear error (no garbage). Update `infer_model_type`.
5. Update `smlx/models/__init__.py` (stop exporting bespoke; export runner +
   real API), `smlx_manager.py`, `smlx_router.py` (remove bespoke references; route
   capability detection through the real backend/runner).

### WS-5 — Delete the bespoke packages
6. Delete the 17 packages (post WS-1 rescue). Run the import/lint/test suite to
   surface every remaining reference and fix it (registry, __init__, manager,
   router, server, bench/eval helpers).

### WS-6 — Examples & tests
7. Delete every example under `examples/` tied to a deleted package; add a small
   set of **real** examples driving `smlx run` / `load_model` / `mlx_backend`.
8. Delete bespoke per-model tests; keep/fix tests that exercise the real API
   (runner, mlx_backend, packaging, inclusion, correctness gate). The suite must
   pass (no import errors from deleted packages).

### WS-7 — Moondream2 (real or out)
9. Try real Moondream via mlx-vlm-supported checkpoints (and/or quantizing one);
   wire a `moondream2` entry only if it passes the correctness gate, else leave it
   out (documented).

### WS-8 — Non-model fake-code audit
10. Review `bench/suites/vlm.py:408` (NotImplementedError) and the handful of
    other markers; make each path real or remove the dead branch. Confirm
    `quant/ agents/ evals/ tools/ kv_cache/ utils/ config/ data/` have no
    stub/placeholder/simulated code (grep + read the hits).

### WS-9 — Extend the fail-closed gate to all public paths
11. Extend `smlx run --verify` (or a shared verifier) so the same per-modality
    correctness checks also assert: `load_model(alias)` returns a working model,
    `mlx_backend` ASR/embeddings produce correct output, and the server
    audio/embeddings route handlers transcribe/embed correctly. Exit non-zero on
    any failure. Keep the existing 14/14 green. Wire into CI.

### WS-10 — Docs
12. Reconcile `docs/MODEL_STATUS.md`, `README.md`, and `CLAUDE.md` to the new
    reality (real entries, deleted packages, gate). **No rule-comments in code.**

## Acceptance criteria

- The 17 bespoke packages are gone; `git grep "smlx.models.<Bespoke>"` returns
  nothing outside docs/history.
- `load_model`, `mlx_backend`, and the server audio/embeddings routes produce real
  correct output or raise honestly — verified by the extended gate.
- `smlx run --verify` (all public paths) is green and exits non-zero on any
  injected failure.
- The full non-`requires_model` test suite imports and passes (no references to
  deleted packages).
- No stub/placeholder/simulated/NotImplemented-as-stub code remains in
  `server/ quant/ bench/ gym/ agents/ evals/ tools/`.
- No "this is real / no fake" style comments anywhere in the code.

## Risks

- **Breadth:** deleting 17 packages cascades into registry/__init__/manager/router/
  server + 106 example/test files. Execute WS-5 iteratively (delete → run → fix the
  next import error) rather than all at once.
- **Moondream2** real checkpoint may not exist in mlx-vlm; acceptable to leave out.
- **Server tests** may need real model downloads; keep those `requires_model`.
