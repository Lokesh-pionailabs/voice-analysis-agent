"""API route definitions."""

import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status, BackgroundTasks

from app.api.dependencies import (
    AnalysisServiceDep,
    ScoringServiceDep,
    SettingsDep,
    SpeechServiceDep,
    StorageServiceDep,
    TranscriptServiceDep,
    verify_api_key,
)
from app.models.response import AnalyzeAudioResponse, TranscriptSegment
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
    "weba": "audio/webm",
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
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio file: mp3, wav, m4a, flac, or weba"),
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

    import asyncio

    async def _do_upload() -> str | None:
        if settings.gcs_bucket_name:
            date_prefix = datetime.now(timezone.utc).strftime("%Y/%m/%d")
            destination_path = f"uploads/{date_prefix}/{request_id}.{extension}"
            try:
                return await storage_service.upload_audio(
                    destination_path=destination_path, content=content, content_type=content_type
                )
            except StorageServiceError as exc:
                logger.error("[%s] Storage upload failed: %s", request_id, exc)
                raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Failed to store audio file.") from exc
        return None

    async def _do_transcribe():
        try:
            return await speech_service.transcribe_audio_content(
                content, language_codes=[language_code] if language_code else None, format=extension
            )
        except SpeechServiceError as exc:
            logger.error("[%s] Transcription failed: %s", request_id, exc)
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Speech-to-text transcription failed.") from exc

    gcs_uri, raw_result = await asyncio.gather(_do_upload(), _do_transcribe())

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

    background_tasks.add_task(_save_transcript_locally, request_id, segments)

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


def _save_transcript_locally(request_id: str, segments: list[TranscriptSegment]) -> None:
    import json
    from datetime import timedelta
    
    path = Path("transcripts")
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"{request_id}.json"
    
    base_time = datetime.now(timezone.utc)
    
    json_data = []
    for seg in segments:
        role = "assistant" if seg.speaker == "Speaker 1" else "user" if seg.speaker == "Speaker 2" else "unknown"
        start_sec = seg.start_seconds or 0.0
        end_sec = seg.end_seconds or 0.0
        
        start_td = timedelta(seconds=start_sec)
        end_td = timedelta(seconds=end_sec)
        
        current_ts = base_time + start_td
        end_ts = base_time + end_td
        
        def format_td(td: timedelta) -> str:
            total_seconds = int(td.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{td.microseconds:06d}"
        
        iso_str = current_ts.isoformat()
        end_iso_str = end_ts.isoformat()
        
        epoch_ms = int(current_ts.timestamp() * 1000)
        end_epoch_ms = int(end_ts.timestamp() * 1000)
        
        json_data.append({
            "role": role,
            "text": seg.text,
            "finished": True,
            "timestamp": iso_str,
            "tsIso": iso_str,
            "tsEpochMs": epoch_ms,
            "startTsIso": iso_str,
            "startTsEpochMs": epoch_ms,
            "endTsIso": end_iso_str,
            "endTsEpochMs": end_epoch_ms,
            "startTime": format_td(start_td),
            "endTime": format_td(end_td)
        })
        
    file_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")


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
