#!/usr/bin/env python3
"""Showcase: speech recognition + word-error-rate on local audio.

Transcribes clips straight from the bundled LibriSpeech sample
(``data/audio/speech/librispeech_sample``) with Whisper-tiny (≈39M params) and
scores them against the ground-truth transcripts using SMLX's own
:class:`~smlx.evals.audio_eval.AudioEvaluator`. The audio arrays are fed
directly from the in-repo dataset -- no files are written -- so this is a real,
measured ASR evaluation that runs end to end on-device.

Run::

    python examples/showcase/whisper_librispeech_wer.py
    python examples/showcase/whisper_librispeech_wer.py --num-clips 10
"""

from __future__ import annotations

import os

# Audio deps (librosa/soundfile) can pull in a second OpenMP runtime on macOS,
# which aborts the process. Allow the duplicate before those imports load.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402

import numpy as np  # noqa: E402

from smlx.data import local  # noqa: E402
from smlx.evals.audio_eval import AudioEvaluator  # noqa: E402
from smlx.models.Whisper_tiny import load, transcribe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--num-clips", type=int, default=5, help="Clips to transcribe")
    args = parser.parse_args()

    if not local.is_available("librispeech_sample"):
        print(
            "LibriSpeech sample not present. Fetch with:\n"
            "  python -m smlx.tools.download_data --dataset librispeech_sample"
        )
        return 1

    ds = local.load("librispeech_sample")
    clips = [ds[i] for i in range(min(args.num_clips, len(ds)))]
    print(f"Loaded {len(clips)} local LibriSpeech clip(s).")

    print("Loading Whisper-tiny (≈39M params) ...")
    model, tokenizer = load("mlx-community/whisper-tiny")

    references: list[str] = []
    hypotheses: list[str] = []
    for idx, clip in enumerate(clips, 1):
        audio = np.asarray(clip["audio"]["array"], dtype=np.float32)
        reference = clip["text"]
        result = transcribe(audio, model, tokenizer, language="en", verbose=None)
        hypothesis = result["text"].strip()

        references.append(reference)
        hypotheses.append(hypothesis)
        print("\n" + "=" * 72)
        print(f"[{idx}/{len(clips)}] {clip.get('id', '')}  ({len(audio) / 16000:.1f}s)")
        print(f"  reference : {reference}")
        print(f"  whisper   : {hypothesis}")

    evaluator = AudioEvaluator()
    summary = evaluator.evaluate_batch(references, hypotheses)
    print("\n" + "=" * 72)
    print(
        f"Aggregate over {summary['num_samples']} clips:  "
        f"WER = {summary['wer']:.2%}   CER = {summary['cer']:.2%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
