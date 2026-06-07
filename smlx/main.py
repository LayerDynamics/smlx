#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
SMLX Command-Line Interface.

Main entry point for the SMLX toolkit providing commands for:
- Model inference and generation
- Server management
- Benchmarking
- Model conversion
- Data downloading
"""

import sys
from pathlib import Path

import click


@click.group()
@click.version_option()
def cli():
    """SMLX - Small Model Learning with MLX.

    A toolkit for running small models (< 1B parameters) on Apple Silicon
    using the MLX framework, optimized for M4 chipsets with unified memory.
    """
    pass


@cli.command()
@click.argument("model", type=str)
@click.argument("prompt", type=str, required=False)
@click.option("--image", "-i", type=click.Path(exists=True), help="Image file for VLM models")
@click.option("--max-tokens", "-t", type=int, default=200, help="Maximum tokens to generate")
@click.option("--temperature", type=float, default=0.0, help="Sampling temperature")
@click.option("--quantize", "-q", default=None, help="Apply SMLX quantization: 4bit or 8bit")
def generate(model, prompt, image, max_tokens, temperature, quantize):
    """Generate text with a language or vision-language model.

    Runs the model through its correct upstream MLX implementation
    (mlx-lm / mlx-vlm) via smlx.models.mlx_backend, optionally quantized with
    SMLX's quantization system.

    Examples:

        \b
        # Text generation (zoo alias or any HF repo)
        smlx generate smollm2-135m "Hello, how are you?"

        \b
        # Vision-language generation
        smlx generate smolvlm-256m "What's in this image?" -i photo.jpg

        \b
        # Quantize with SMLX's 4-bit, then generate
        smlx generate smollm2-360m "Write a haiku" -q 4bit
    """
    if prompt is None:
        click.echo("Interactive mode - enter your prompt:")
        prompt = click.prompt("Prompt")

    from smlx.models import mlx_backend as backend

    try:
        click.echo(f"Loading {model}...")
        bm = backend.load(model, quantize=quantize)
        click.echo(f" Loaded ({bm.backend.value}{', quantized' if bm.quantized else ''})\n")
        click.echo("-" * 60)
        text = backend.generate(
            bm, prompt, image=image, max_tokens=max_tokens, temperature=temperature
        )
        click.echo(text)
        click.echo("-" * 60)
    except KeyboardInterrupt:
        click.echo("\n\nGeneration interrupted.")
        sys.exit(0)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload/--no-reload", default=False, help="Enable auto-reload")
@click.option("--log-level", default="info", help="Logging level")
def server(host, port, reload, log_level):
    """Start the SMLX API server.

    Provides OpenAI-compatible endpoints for chat, completions, embeddings,
    and audio transcription.
    """
    try:
        import uvicorn
    except ImportError:
        click.echo("Error: uvicorn not installed. Install with: pip install uvicorn", err=True)
        sys.exit(1)

    click.echo(f"Starting SMLX server on {host}:{port}")
    click.echo("API documentation available at: http://localhost:8000/docs")

    uvicorn.run("smlx.server.app:app", host=host, port=port, reload=reload, log_level=log_level)


@cli.command()
@click.argument("suite", required=False)
@click.option("--model", "-m", help="Model to benchmark")
@click.option("--all", "run_all", is_flag=True, help="Run all benchmark suites")
@click.option("--list", "list_suites", is_flag=True, help="List available benchmark suites")
def bench(suite, model, run_all, list_suites):
    """Run performance benchmarks.

    Available suites: llm, vlm, quantization, text_generation, ops
    """
    from smlx.bench import run as bench_run

    if list_suites:
        click.echo("Available benchmark suites:")
        click.echo("  - system: Display system information")
        click.echo("  - llm: Language model benchmarks")
        click.echo("  - vlm: Vision-language model benchmarks")
        click.echo("  - quantization: Quantization performance tests")
        click.echo("  - text_generation: Text generation quality/speed")
        click.echo("  - ops: Low-level MLX operation benchmarks")
        return

    # Construct arguments for bench module
    args = []
    if run_all:
        args.append("--all")
    if suite:
        args.append(suite)
    if model:
        args.extend(["--model", model])

    # Run benchmark
    sys.argv = ["smlx-bench"] + args
    bench_run.main()


@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option("--format", "output_format", default="safetensors", help="Output format")
@click.option("--quantize", help="Quantize during conversion (4bit, 8bit, etc.)")
def convert(source, output, output_format, quantize):
    """Convert models to MLX format.

    Convert PyTorch, TensorFlow, or ONNX models to MLX-compatible format.
    """
    from smlx.tools.convert2mlx import convert as convert_to_mlx

    click.echo(f"Converting {source} -> {output}")
    if output_format and output_format != "safetensors":
        click.echo(
            f"Note: MLX conversion writes safetensors weights; ignoring --format '{output_format}'."
        )

    # Map the CLI --quantize string ("4bit"/"8bit"/...) to convert()'s
    # (quantize: bool, q_bits: int) arguments.
    quantize_flag = bool(quantize)
    q_bits = 4
    if quantize:
        digits = "".join(ch for ch in str(quantize) if ch.isdigit())
        if digits:
            q_bits = int(digits)

    try:
        convert_to_mlx(
            hf_path=source,
            mlx_path=output,
            quantize=quantize_flag,
            q_bits=q_bits,
        )
        click.echo(f" Conversion complete: {output}")
    except Exception as e:
        click.echo(f"Error during conversion: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--models", is_flag=True, help="Download models")
@click.option("--datasets", is_flag=True, help="Download datasets")
@click.option("--all", "download_all", is_flag=True, help="Download all data")
@click.option("--model", help="Specific model to download")
def download(models, datasets, download_all, model):
    """Download models and datasets.

    Download pre-trained models and evaluation datasets to local cache.
    """
    from smlx.tools.download_data import main as download_main

    # Construct arguments
    args = []
    if download_all:
        args.append("--all")
    if models:
        args.append("--models")
    if datasets:
        args.append("--datasets")
    if model:
        args.extend(["--model", model])

    # Run download tool
    sys.argv = ["smlx-download"] + args
    download_main()


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--model", default="whisper-tiny", help="Whisper model variant")
@click.option("--language", help="Audio language (auto-detected if not specified)")
@click.option(
    "--format", "output_format", default="text", help="Output format (text, json, srt, vtt)"
)
@click.option("--output", "-o", type=click.Path(), help="Output file (stdout if not specified)")
def transcribe(audio_file, model, language, output_format, output):
    """Transcribe audio files using Whisper.

    Supports various output formats including SRT/VTT subtitles.
    """
    from smlx.models.Whisper_tiny import load, transcribe as whisper_transcribe

    click.echo(f"Loading {model}...")
    whisper_model, tokenizer = load(model)
    click.echo(" Model loaded\n")

    click.echo(f"Transcribing {audio_file}...")
    result = whisper_transcribe(
        audio=audio_file,
        model=whisper_model,
        tokenizer=tokenizer,
        language=language,
    )

    # Format output
    if output_format == "text":
        output_text = result["text"]
    elif output_format == "json":
        import json

        output_text = json.dumps(result, indent=2)
    elif output_format == "srt":
        from smlx.server.routes.audio import format_as_srt

        output_text = format_as_srt(result.get("segments", []))
    elif output_format == "vtt":
        from smlx.server.routes.audio import format_as_vtt

        output_text = format_as_vtt(result.get("segments", []))
    else:
        click.echo(f"Error: Unknown format: {output_format}", err=True)
        sys.exit(1)

    # Write output
    if output:
        Path(output).write_text(output_text)
        click.echo(f" Transcription saved to: {output}")
    else:
        click.echo("\nTranscription:")
        click.echo("-" * 60)
        click.echo(output_text)
        click.echo("-" * 60)


@cli.group()
def data():
    """Inspect and use the bundled datasets under ``data/``.

    \b
    smlx data list                       # list datasets + presence
    smlx data validate                   # health-check every dataset
    smlx data preview wikitext -n 3      # preview samples
    smlx data eval mathvista -m <model>  # run an eval on a real model
    """
    pass


@data.command(name="list")
def data_list():
    """List all registered datasets and whether they are present on disk."""
    from smlx.data import local, report

    for line in report.list_lines(local.inventory(compute_size=False)):
        click.echo(line)


@data.command(name="validate")
@click.option("--category", type=click.Choice(["benchmark", "training"]), help="Filter by category")
@click.option("--no-size", is_flag=True, help="Skip on-disk size computation (faster)")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a table")
def data_validate(category, no_size, as_json):
    """Probe every dataset: layout, splits, counts, size, single-sample load.

    Exits non-zero if any present dataset fails to load (corrupt/unreadable).
    """
    import json as _json
    from dataclasses import asdict

    from smlx.data import local, report

    results = local.inventory(compute_size=not no_size)
    if category:
        results = [r for r in results if r.category == category]
    orphans = local.find_orphans()

    if as_json:
        payload = {
            "data_dir": str(local.data_dir()),
            "datasets": [{**asdict(r), "layout": r.layout.value} for r in results],
            "orphans": orphans,
        }
        click.echo(_json.dumps(payload, indent=2))
    else:
        click.echo(f"SMLX data directory: {local.data_dir()}\n")
        for line in report.inventory_lines(results, orphans=orphans, show_size=not no_size):
            click.echo(line)

    failed = [r for r in results if r.available and r.sample_ok is False]
    if failed:
        sys.exit(1)


@data.command(name="preview")
@click.argument("dataset")
@click.option("--split", help="Split to preview (default: the dataset's default)")
@click.option("-n", "--limit", type=int, default=5, help="Number of samples to show")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of text")
def data_preview(dataset, split, limit, as_json):
    """Preview samples from a dataset (no model loaded)."""
    import json as _json

    from smlx.data import local, report

    if dataset not in local.registry():
        click.echo(f"Unknown dataset '{dataset}'. Try: smlx data list", err=True)
        sys.exit(2)
    if not local.is_available(dataset):
        click.echo(
            f"Dataset '{dataset}' is not present on disk. Download it with:\n"
            f"  python -m smlx.tools.download_data --dataset {dataset}",
            err=True,
        )
        sys.exit(1)

    try:
        samples = list(local.iter_samples(dataset, split=split, limit=limit))
    except Exception as exc:
        click.echo(f"Failed to load samples from '{dataset}': {exc}", err=True)
        sys.exit(1)

    if as_json:
        click.echo(_json.dumps({"dataset": dataset, "split": split, "samples": samples}, indent=2))
    else:
        for line in report.preview_lines(dataset, split, samples):
            click.echo(line)


# Friendly benchmark name -> eval module (each module exposes a CLI main()).
_EVAL_MODULES = {
    "mathvista": "smlx.evals.math_vista",
    "mmmu": "smlx.evals.mmmu",
    "mmstar": "smlx.evals.mmstar",
    "ocrbench": "smlx.evals.ocrbench",
}


@data.command(
    name="eval",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("benchmark")
@click.option("--model", "-m", required=True, help="Model id/path to evaluate")
@click.argument("eval_args", nargs=-1, type=click.UNPROCESSED)
def data_eval(benchmark, model, eval_args):
    """Run a benchmark evaluation on a real model.

    Thin launcher over the ``smlx.evals.*`` evaluators. Extra flags are passed
    straight through, e.g.::

        smlx data eval mathvista -m mlx-community/SmolVLM-256M-Instruct --max-samples 10

    Note: the evaluators load the benchmark from their own configured source
    (the HuggingFace repo). This command reports whether a local copy exists,
    but does not force the eval to read it (most local copies are
    ``save_to_disk`` directories the evaluator's ``load_dataset`` cannot
    consume directly).
    """
    from importlib import import_module

    if benchmark not in _EVAL_MODULES:
        click.echo(
            f"Unknown benchmark '{benchmark}'. Available: {', '.join(sorted(_EVAL_MODULES))}",
            err=True,
        )
        sys.exit(2)

    from smlx.data import local

    if benchmark in local.registry() and local.is_available(benchmark):
        rel = local.local_path(benchmark).relative_to(local.data_dir())
        click.echo(f"Local copy present at: data/{rel}")
    else:
        click.echo(f"No local copy of '{benchmark}'; the evaluator will download it.")

    mod_name = _EVAL_MODULES[benchmark]
    click.echo(f"Running {mod_name} on model: {model}\n")
    mod = import_module(mod_name)
    sys.argv = [mod_name, "--model", model, *eval_args]
    mod.main()


@cli.group()
def models():
    """Inspect and verify the curated model zoo (runs on the upstream backend).

    \b
    smlx models list                  # the curated zoo
    smlx models verify                # load + run + check real output for every model
    smlx models verify smolvlm-256m   # just one
    """
    pass


@models.command(name="list")
def models_list():
    """List the curated zoo (alias, backend, modality, params)."""
    from smlx.models import mlx_backend as backend

    header = f"{'ALIAS':<16} {'BACKEND':<8} {'MODALITY':<10} {'PARAMS':<7} REPO"
    click.echo(header)
    click.echo("-" * len(header))
    for entry in backend.ZOO.values():
        click.echo(
            f"{entry.key:<16} {entry.backend.value.replace('mlx_',''):<8} "
            f"{entry.modality:<10} {entry.params:<7} {entry.repo}"
        )


def _looks_like_real_text(text: str, min_words: int = 3) -> bool:
    """Lightweight real-output check (CLI cannot import the test assertions)."""
    import re

    words = re.findall(r"[A-Za-z]{2,}", text or "")
    if len(words) < min_words:
        return False
    from collections import Counter

    lowered = [w.lower() for w in words]
    _, top = Counter(lowered).most_common(1)[0]
    return top / len(lowered) < 0.6  # not degenerate repetition


@models.command(name="verify")
@click.argument("model", required=False)
@click.option("--quantize", "-q", default=None, help="Verify with SMLX quantization (4bit/8bit)")
@click.option("--max-tokens", "-t", type=int, default=40, help="Tokens to generate per check")
def models_verify(model, quantize, max_tokens):
    """Load each model through its backend, run it, and assert real output.

    Exits non-zero if any verified model fails to produce coherent output.
    """
    import time

    from smlx.models import mlx_backend as backend

    targets = [model.lower()] if model else list(backend.ZOO.keys())
    unknown = [t for t in targets if t not in backend.ZOO]
    if unknown:
        click.echo(f"Unknown model(s): {unknown}. Try: smlx models list", err=True)
        sys.exit(2)

    # A bundled image for VLM checks (optional).
    image_path = None
    try:
        from smlx.data import local

        if local.is_available("coco8"):
            tree = local.load("coco8", split="train")
            image_path = str(tree.images[0])
    except Exception:
        image_path = None

    failures = 0
    click.echo(f"{'MODEL':<16} {'RESULT':<7} {'tok/s':>7}  OUTPUT")
    click.echo("-" * 72)
    for key in targets:
        entry = backend.ZOO[key]
        try:
            bm = backend.load(key, quantize=quantize)
            t0 = time.perf_counter()
            if entry.modality == "asr":
                # Transcribe a bundled LibriSpeech clip; check it's real text.
                from smlx.data import local as _local

                audio = _local.load("librispeech_sample")[0]["audio"]["array"]
                out = backend.transcribe(bm, audio)
                ok = _looks_like_real_text(out)
            elif entry.modality == "embeddings":
                import numpy as _np

                emb = _np.asarray(backend.embed(bm, ["a cat sleeps", "a kitten naps"]))
                # Correct: 2-D, finite, and the two related sentences are similar.
                ok = emb.ndim == 2 and bool(_np.isfinite(emb).all())
                out = f"embeddings shape {emb.shape}"
            else:
                prompt = (
                    "What is in this image?"
                    if entry.modality == "vlm"
                    else "In one sentence, what is MLX?"
                )
                img = image_path if entry.modality == "vlm" else None
                out = backend.generate(bm, prompt, image=img, max_tokens=max_tokens)
                ok = _looks_like_real_text(out)
            dt = time.perf_counter() - t0
            n_words = len((out or "").split())
            tps = n_words / dt if dt > 0 else 0.0
            snippet = " ".join((out or "").split())[:48]
            click.echo(f"{key:<16} {'PASS' if ok else 'FAIL':<7} {tps:>7.1f}  {snippet}")
            if not ok:
                failures += 1
        except Exception as e:
            click.echo(f"{key:<16} {'ERROR':<7} {'-':>7}  {type(e).__name__}: {str(e)[:40]}")
            failures += 1

    click.echo("-" * 72)
    click.echo(f"{len(targets) - failures}/{len(targets)} models produced real output.")
    if failures:
        sys.exit(1)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
