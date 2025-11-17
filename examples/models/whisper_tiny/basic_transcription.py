"""
Basic Whisper-tiny transcription example.

Demonstrates how to load the Whisper-tiny model and transcribe audio files.

Usage:
    python examples/whisper_tiny/basic_transcription.py audio.wav
    python examples/whisper_tiny/basic_transcription.py audio.wav --language es --task translate
"""

import argparse
import sys
from pathlib import Path

# Add smlx to path if running from examples directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from smlx.models.Whisper_tiny import load, transcribe


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using Whisper-tiny",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription with auto-detected language
  python basic_transcription.py speech.wav

  # Transcribe Spanish audio
  python basic_transcription.py speech.wav --language es

  # Translate Spanish to English
  python basic_transcription.py speech.wav --language es --task translate

  # Use higher temperature for creative transcription
  python basic_transcription.py speech.wav --temperature 0.5

  # Show detailed output with timestamps
  python basic_transcription.py speech.wav --verbose --show-segments
        """,
    )

    parser.add_argument(
        "audio",
        type=str,
        help="Path to audio file (supports most formats via ffmpeg)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/whisper-tiny",
        help="Model path or HuggingFace repo (default: mlx-community/whisper-tiny)",
    )

    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Audio language (e.g., en, es, fr). Auto-detected if not specified.",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task: transcribe (X->X) or translate (X->English) (default: transcribe)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy, higher = more creative) (default: 0.0)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress and segment information",
    )

    parser.add_argument(
        "--show-segments",
        action="store_true",
        help="Show individual segments with timestamps",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use float16 precision (default: True)",
    )

    args = parser.parse_args()

    # Check if audio file exists
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    print("=" * 70)
    print("WHISPER-TINY TRANSCRIPTION")
    print("=" * 70)
    print(f"Audio file: {audio_path}")
    print(f"Model: {args.model}")
    print(f"Language: {args.language or 'auto-detect'}")
    print(f"Task: {args.task}")
    print(f"Temperature: {args.temperature}")
    print("=" * 70)
    print()

    # Load model
    print("Loading model...")
    model, tokenizer = load(args.model)
    print(f"Model loaded: {model.config.n_audio_layer}-layer encoder, "
          f"{model.config.n_text_layer}-layer decoder")
    print()

    # Transcribe
    print("Transcribing...")
    result = transcribe(
        str(audio_path),
        model,
        tokenizer,
        language=args.language,
        task=args.task,
        temperature=args.temperature,
        verbose=args.verbose,
        fp16=args.fp16,
    )

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Language: {result['language']}")
    print()
    print("Text:")
    print(result["text"])
    print()

    # Show segments if requested
    if args.show_segments and result["segments"]:
        print("-" * 70)
        print("SEGMENTS")
        print("-" * 70)
        for i, segment in enumerate(result["segments"]):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            print(f"[{start:6.2f}s -> {end:6.2f}s] {text}")
        print("-" * 70)

    print("=" * 70)


if __name__ == "__main__":
    main()
