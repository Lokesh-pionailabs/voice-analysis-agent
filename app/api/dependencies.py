"""FastAPI dependency-injection providers.

Services are constructed lazily on the first request to ensure they are
instantiated within the running asyncio event loop. The underlying Google
Cloud SDK clients (especially gRPC ones) require an active event loop at
instantiation time.
"""

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.services.analysis_service import AnalysisService
from app.services.scoring_service import ScoringService
from app.services.speech_service import SpeechService
from app.services.storage_service import StorageService
from app.services.transcript_service import TranscriptService
from app.utils.config import Settings, get_settings

_storage_service: StorageService | None = None
_speech_service: SpeechService | None = None
_transcript_service: TranscriptService | None = None
_analysis_service: AnalysisService | None = None
_scoring_service: ScoringService | None = None


async def get_storage_service() -> StorageService:
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService(settings=get_settings())
    return _storage_service


async def get_speech_service() -> SpeechService:
    global _speech_service
    if _speech_service is None:
        _speech_service = SpeechService(settings=get_settings())
    return _speech_service


async def get_transcript_service() -> TranscriptService:
    global _transcript_service
    if _transcript_service is None:
        _transcript_service = TranscriptService()
    return _transcript_service


async def get_analysis_service() -> AnalysisService:
    global _analysis_service
    if _analysis_service is None:
        _analysis_service = AnalysisService(settings=get_settings())
    return _analysis_service


async def get_scoring_service() -> ScoringService:
    global _scoring_service
    if _scoring_service is None:
        _scoring_service = ScoringService()
    return _scoring_service


SettingsDep = Annotated[Settings, Depends(get_settings)]
StorageServiceDep = Annotated[StorageService, Depends(get_storage_service)]
SpeechServiceDep = Annotated[SpeechService, Depends(get_speech_service)]
TranscriptServiceDep = Annotated[TranscriptService, Depends(get_transcript_service)]
AnalysisServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]
ScoringServiceDep = Annotated[ScoringService, Depends(get_scoring_service)]


async def verify_api_key(
    settings: SettingsDep,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """Optional static API-key gate.

    No-ops when `API_KEY` is not configured. When configured, requires a matching header.
    """
    if settings.api_key is None:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key.")
