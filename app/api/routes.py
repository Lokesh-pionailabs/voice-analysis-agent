"""API route definitions."""

import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.dependencies import (
    AnalysisServiceDep,
    ScoringServiceDep,
    SettingsDep,
    SpeechServiceDep,
    StorageServiceDep,
    TranscriptServiceDep,
    verify_api_key,
)
from app.models.response import AnalyzeAudioResponse
from app.services.analysis_service import AnalysisServiceError
from app.services.speech_service import SpeechServiceError
from app.services.storage_service import StorageServiceError
from app.utils.helpers import bytes_to_mb, get_file_extension, new_correlation_id
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["voice-analysis"])

_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "m4a": "audio/mp4",
    "flac": "audio/flac",
}


@router.post(
    "/analyze",
    response_model=AnalyzeAudioResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze a call recording end-to-end",
    dependencies=[Depends(verify_api_key)],
)
async def analyze_audio(
    settings: SettingsDep,
    storage_service: StorageServiceDep,
    speech_service: SpeechServiceDep,
    transcript_service: TranscriptServiceDep,
    analysis_service: AnalysisServiceDep,
    scoring_service: ScoringServiceDep,
    audio_file: UploadFile = File(..., description="Audio file: mp3, wav, m4a, or flac"),
    language_code: str | None = Form(default=None),
) -> AnalyzeAudioResponse:
    """Run the full pipeline: upload -> transcribe -> clean -> analyze -> score.

    Args:
        audio_file: Multipart audio upload.
        language_code: Optional BCP-47 language override.

    Returns:
        A complete structured analysis of the call.
    """
    request_id = new_correlation_id()
    started_at = time.monotonic()
    logger.info("Received analyze request %s for file '%s'", request_id, audio_file.filename)

    _validate_upload(settings, audio_file)
    content = await audio_file.read()
    _validate_size(settings, content)

    extension = get_file_extension(audio_file.filename or "")
    content_type = _CONTENT_TYPES.get(extension, audio_file.content_type or "application/octet-stream")

    gcs_uri: str | None = None
    raw_result = None

    if settings.gcs_bucket_name:
        # Step 1 & 2: stage audio in GCS.
        date_prefix = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        destination_path = f"uploads/{date_prefix}/{request_id}.{extension}"
        try:
            gcs_uri = await storage_service.upload_audio(
                destination_path=destination_path, content=content, content_type=content_type
            )
        except StorageServiceError as exc:
            logger.error("[%s] Storage upload failed: %s", request_id, exc)
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Failed to store audio file.") from exc

        # Step 3: transcribe.
        try:
            raw_result = await speech_service.transcribe_gcs_uri(
                gcs_uri, language_codes=[language_code] if language_code else None
            )
        except SpeechServiceError as exc:
            logger.error("[%s] Transcription failed: %s", request_id, exc)
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Speech-to-text transcription failed.") from exc
    else:
        try:
            raw_result = await speech_service.transcribe_audio_content(
                content, language_codes=[language_code] if language_code else None
            )
        except SpeechServiceError as exc:
            logger.error("[%s] Transcription failed: %s", request_id, exc)
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Speech-to-text transcription failed.") from exc

    if not raw_result.full_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No speech could be detected in the provided audio file.",
        )

    # Step 4: clean + structure transcript.
    segments = transcript_service.build_segments(raw_result)
    llm_transcript = transcript_service.render_for_llm(segments)

    # Step 5 & 6: Gemini analysis + deterministic scoring rules.
    try:
        analysis = await analysis_service.analyze_transcript(llm_transcript)
    except AnalysisServiceError as exc:
        logger.error("[%s] Gemini analysis failed: %s", request_id, exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Conversation analysis failed.") from exc

    analysis = scoring_service.normalize(analysis)

    _save_transcript_locally(settings.local_transcript_path, request_id, llm_transcript)

    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    logger.info("[%s] Completed analysis in %d ms", request_id, elapsed_ms)

    return AnalyzeAudioResponse(
        request_id=request_id,
        audio_filename=audio_file.filename or "unknown",
        audio_duration_seconds=raw_result.duration_seconds,
        gcs_uri=gcs_uri,
        transcript=llm_transcript,
        transcript_segments=segments,
        analysis=analysis,
        processing_time_ms=elapsed_ms,
    )


def _save_transcript_locally(base_path: str, request_id: str, transcript: str) -> None:
    path = Path(base_path)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"{request_id}.txt"
    file_path.write_text(transcript, encoding="utf-8")


def _validate_upload(settings: SettingsDep, audio_file: UploadFile) -> None:
    if not audio_file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided.")
    extension = get_file_extension(audio_file.filename)
    if extension not in settings.allowed_audio_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format '{extension}'. Allowed: {settings.allowed_audio_extensions}.",
        )


def _validate_size(settings: SettingsDep, content: bytes) -> None:
    if len(content) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded audio file is empty.")
    size_mb = bytes_to_mb(len(content))
    if size_mb > settings.max_audio_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file is {size_mb} MB, exceeding the {settings.max_audio_size_mb} MB limit.",
        )
