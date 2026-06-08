# Copyright © 2025 SMLX Project

"""
Audio transcription endpoints (OpenAI compatible).
"""

import asyncio

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from ..dependencies import get_model_manager
from ..model_manager import ModelManager
from ..schemas import AudioTranscriptionResponse

router = APIRouter()


def _format_timestamp(seconds: float, use_comma: bool = False) -> str:
    """Format timestamp as HH:MM:SS.mmm or HH:MM:SS,mmm.

    Args:
        seconds: Timestamp in seconds
        use_comma: Use comma separator (SRT format) instead of period (VTT format)

    Returns:
        Formatted timestamp string
    """
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    secs = milliseconds // 1_000
    milliseconds -= secs * 1_000

    hours_marker = f"{hours:02d}:" if hours > 0 else "00:"
    separator = "," if use_comma else "."
    return f"{hours_marker}{minutes:02d}:{secs:02d}{separator}{milliseconds:03d}"


def format_as_srt(segments: list[dict]) -> str:
    """Format transcription segments as SRT subtitle file.

    Args:
        segments: List of segment dictionaries with start, end, text

    Returns:
        SRT formatted string
    """
    srt_output = []
    for i, segment in enumerate(segments, 1):
        start = _format_timestamp(segment["start"], use_comma=True)
        end = _format_timestamp(segment["end"], use_comma=True)
        text = segment["text"].strip()

        srt_output.append(f"{i}\n{start} --> {end}\n{text}\n")

    return "\n".join(srt_output)


def format_as_vtt(segments: list[dict]) -> str:
    """Format transcription segments as WebVTT subtitle file.

    Args:
        segments: List of segment dictionaries with start, end, text

    Returns:
        WebVTT formatted string
    """
    vtt_output = ["WEBVTT\n"]
    for segment in segments:
        start = _format_timestamp(segment["start"], use_comma=False)
        end = _format_timestamp(segment["end"], use_comma=False)
        text = segment["text"].strip()

        vtt_output.append(f"{start} --> {end}\n{text}\n")

    return "\n".join(vtt_output)


@router.post("/audio/transcriptions", response_model=AudioTranscriptionResponse)
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-tiny"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    manager: ModelManager = Depends(get_model_manager),
):
    """
    Transcribe audio file.

    OpenAI-compatible endpoint for audio transcription using Whisper models.
    """
    try:
        # Validate model
        if "whisper" not in model.lower():
            raise ValueError(f"Model {model} is not a Whisper model")

        # Load Whisper model
        bm = await manager.load_model(model, model_type="whisper")

        # Read audio file
        audio_bytes = await file.read()

        # Transcribe audio
        transcription = await transcribe_audio(
            bm=bm,
            audio_bytes=audio_bytes,
            language=language,
            prompt=prompt,
            temperature=temperature,
        )

        # Format response based on requested format
        if response_format == "json":
            return AudioTranscriptionResponse(
                text=transcription["text"],
                language=transcription.get("language"),
                duration=transcription.get("duration"),
                segments=transcription.get("segments"),
            )
        elif response_format == "text":
            return {"text": transcription["text"]}
        elif response_format == "srt":
            segments = transcription.get("segments", [])
            srt_text = format_as_srt(segments)
            return Response(content=srt_text, media_type="text/plain")
        elif response_format == "vtt":
            segments = transcription.get("segments", [])
            vtt_text = format_as_vtt(segments)
            return Response(content=vtt_text, media_type="text/plain")
        else:
            raise ValueError(f"Invalid response format: {response_format}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def transcribe_audio(
    bm,
    audio_bytes: bytes,
    language: str | None,
    prompt: str | None,
    temperature: float,
) -> dict:
    """
    Transcribe audio through mlx-whisper.

    Args:
        bm: Loaded ASR BackendModel (carries the Whisper repo)
        audio_bytes: Audio file bytes
        language: Optional language code
        prompt: Optional transcription prompt
        temperature: Sampling temperature

    Returns:
        Dictionary with transcription results
    """
    # Save audio bytes to temporary file (Whisper expects a file path)
    import os
    import tempfile

    import mlx_whisper

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name

    try:
        # Run transcription in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: mlx_whisper.transcribe(
                tmp_path,
                path_or_hf_repo=bm.repo,
                language=language,
                temperature=temperature,
                initial_prompt=prompt,
            ),
        )

        return {
            "text": result.get("text", ""),
            "language": result.get("language"),
            "duration": result.get("duration"),
            "segments": result.get("segments"),
        }

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
