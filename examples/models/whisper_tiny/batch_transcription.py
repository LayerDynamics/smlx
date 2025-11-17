"""
Batch transcription example for Whisper-tiny.

Demonstrates how to transcribe multiple audio files efficiently.

Usage:
    python examples/whisper_tiny/batch_transcription.py audio1.wav audio2.wav audio3.wav
    python examples/whisper_tiny/batch_transcription.py audio_dir/*.wav --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Add smlx to path if running from examples directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from smlx.models.Whisper_tiny import load, transcribe


def transcribe_batch(
    audio_files: List[Path],
    model,
    tokenizer,
    **kwargs,
) -> List[dict]:
    """Transcribe multiple audio files.

    Args:
        audio_files: List of audio file paths
        model: Whisper model
        tokenizer: Whisper tokenizer
        **kwargs: Additional arguments for transcribe()

    Returns:
        List of transcription results
    """
    results = []

    for i, audio_file in enumerate(audio_files):
        print(f"\n[{i + 1}/{len(audio_files)}] Processing: {audio_file.name}")
        print("-" * 70)

        try:
            result = transcribe(str(audio_file), model, tokenizer, **kwargs)
            results.append(
                {
                    "file": str(audio_file),
                    "text": result["text"],
                    "language": result["language"],
                    "segments": result["segments"],
                    "status": "success",
                }
            )
            print(f"✓ Success: {result['text'][:100]}...")

        except Exception as e:
            print(f"✗ Error: {e}")
            results.append(
                {
                    "file": str(audio_file),
                    "text": "",
                    "language": None,
                    "segments": [],
                    "status": "error",
                    "error": str(e),
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch transcribe audio files using Whisper-tiny",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe multiple files
  python batch_transcription.py audio1.wav audio2.wav audio3.wav

  # Transcribe all WAV files in directory
  python batch_transcription.py audio_dir/*.wav

  # Save results to JSON
  python batch_transcription.py audio_dir/*.wav --output results.json

  # Transcribe with specific language
  python batch_transcription.py *.wav --language es --output spanish_results.json
        """,
    )

    parser.add_argument(
        "audio_files",
        nargs="+",
        type=str,
        help="Audio files to transcribe",
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
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    # Validate audio files
    audio_files = [Path(f) for f in args.audio_files]
    audio_files = [f for f in audio_files if f.exists()]

    if not audio_files:
        print("Error: No valid audio files found")
        sys.exit(1)

    print("=" * 70)
    print("WHISPER-TINY BATCH TRANSCRIPTION")
    print("=" * 70)
    print(f"Files to process: {len(audio_files)}")
    print(f"Model: {args.model}")
    print(f"Language: {args.language or 'auto-detect'}")
    print(f"Task: {args.task}")
    print("=" * 70)

    # Load model once for all files
    print("\nLoading model...")
    model, tokenizer = load(args.model)
    print("Model loaded successfully")

    # Transcribe all files
    results = transcribe_batch(
        audio_files,
        model,
        tokenizer,
        language=args.language,
        task=args.task,
        verbose=args.verbose,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    print(f"Total files: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print("=" * 70)

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    # Print transcriptions
    print("\n" + "=" * 70)
    print("TRANSCRIPTIONS")
    print("=" * 70)
    for result in results:
        file_name = Path(result["file"]).name
        status = result["status"]
        print(f"\n{file_name} [{status}]:")
        if status == "success":
            print(f"  Language: {result['language']}")
            print(f"  Text: {result['text']}")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
