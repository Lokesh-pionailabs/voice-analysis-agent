"""Google Cloud Speech-to-Text V2 integration.

Uses the v2 `SpeechAsyncClient` against a GCS URI, which scales to long
audio files (Speech-to-Text v2 supports batch recognition for large
files without the ~1-minute sync-request ceiling of v1's inline mode).
"""

from dataclasses import dataclass, field

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPICallError, RetryError
from google.api_core.retry_async import AsyncRetry
from google.cloud.speech_v2 import (
    AutoDetectDecodingConfig,
    RecognitionConfig,
    RecognitionFeatures,
    RecognizeRequest,
    SpeakerDiarizationConfig,
    SpeechAsyncClient,
)

from app.utils.config import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SpeechServiceError(Exception):
    """Raised when speech recognition fails after all retries."""


@dataclass
class TranscriptWord:
    word: str
    start_seconds: float
    end_seconds: float
    speaker_tag: int | None = None


@dataclass
class RawTranscriptionResult:
    """Raw output of speech recognition, before cleaning/structuring."""

    full_text: str
    words: list[TranscriptWord] = field(default_factory=list)
    duration_seconds: float | None = None


class SpeechService:
    """Wraps Speech-to-Text V2 for async, retry-safe transcription."""

    def __init__(self, settings: Settings, client: SpeechAsyncClient | None = None) -> None:
        """Initialize the service.

        Args:
            settings: Application settings.
            client: Optional pre-built async client, primarily for testing.
        """
        self._settings = settings
        self._client = client or SpeechAsyncClient(
            client_options=ClientOptions(api_endpoint=f"{settings.gcp_region}-speech.googleapis.com")
            if settings.gcp_region != "global"
            else None
        )
        self._recognizer_path = (
            f"projects/{settings.gcp_project_id}/locations/{settings.gcp_region}"
            f"/recognizers/{settings.speech_recognizer_id}"
        )

    def _build_recognition_config(self, *, language_codes: list[str] | None = None) -> RecognitionConfig:
        diarization = None
        if self._settings.enable_speaker_diarization:
            diarization = SpeakerDiarizationConfig(
                min_speaker_count=self._settings.diarization_min_speakers,
                max_speaker_count=self._settings.diarization_max_speakers,
            )

        return RecognitionConfig(
            auto_decoding_config=AutoDetectDecodingConfig(),
            language_codes=language_codes or self._settings.speech_language_codes,
            model=self._settings.speech_model,
            features=RecognitionFeatures(
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True,
                enable_word_confidence=True,
                diarization_config=diarization,
            ),
        )

    async def transcribe_gcs_uri(
        self,
        gcs_uri: str,
        *,
        language_codes: list[str] | None = None,
        max_retries: int = 3,
    ) -> RawTranscriptionResult:
        """Transcribe an audio file already staged in GCS.

        Args:
            gcs_uri: `gs://bucket/path` URI of the audio to transcribe.
            language_codes: Optional override of the configured language codes.
            max_retries: Number of retry attempts on transient failures.

        Returns:
            The raw transcription result (full text plus word-level timing
            and speaker tags, when diarization is enabled).

        Raises:
            SpeechServiceError: If transcription fails after all retries.
        """
        return await self._transcribe_recognize_request(
            RecognizeRequest(recognizer=self._recognizer_path, config=self._build_recognition_config(language_codes=language_codes), uri=gcs_uri),
            retry_target=gcs_uri,
        )

    async def transcribe_audio_content(
        self,
        content: bytes,
        *,
        language_codes: list[str] | None = None,
        max_retries: int = 3,
    ) -> RawTranscriptionResult:
        """Transcribe raw audio bytes without staging in GCS, automatically chunking long audio.

        Args:
            content: Raw audio bytes.
            language_codes: Optional override of the configured language codes.
            max_retries: Number of retry attempts on transient failures.

        Returns:
            The raw transcription result.

        Raises:
            SpeechServiceError: If transcription fails after all retries.
        """
        import io
        import asyncio
        from pydub import AudioSegment

        audio = AudioSegment.from_file(io.BytesIO(content))
        duration_seconds = len(audio) / 1000.0

        if duration_seconds <= 55:
            return await self._transcribe_recognize_request(
                RecognizeRequest(
                    recognizer=self._recognizer_path, 
                    config=self._build_recognition_config(language_codes=language_codes),
                    content=content
                ),
                retry_target="inline audio",
            )

        chunk_length_ms = 50 * 1000
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        
        async def process_chunk(index: int, chunk: AudioSegment) -> tuple[int, RawTranscriptionResult]:
            buf = io.BytesIO()
            chunk.export(buf, format="wav")
            chunk_bytes = buf.getvalue()
            
            res = await self._transcribe_recognize_request(
                RecognizeRequest(
                    recognizer=self._recognizer_path, 
                    config=self._build_recognition_config(language_codes=language_codes),
                    content=chunk_bytes
                ),
                retry_target=f"chunk {index}",
            )
            return index, res

        semaphore = asyncio.Semaphore(5)
        async def sem_process(index: int, chunk: AudioSegment) -> tuple[int, RawTranscriptionResult]:
            async with semaphore:
                return await process_chunk(index, chunk)
                
        tasks = [sem_process(i, c) for i, c in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])
        
        full_text_parts = []
        all_words = []
        for index, res in results:
            offset_seconds = index * 50.0
            if res.full_text.strip():
                full_text_parts.append(res.full_text)
            for w in res.words:
                w.start_seconds += offset_seconds
                w.end_seconds += offset_seconds
                all_words.append(w)
                
        return RawTranscriptionResult(
            full_text=" ".join(full_text_parts),
            words=all_words,
            duration_seconds=duration_seconds
        )

    async def _transcribe_recognize_request(
        self,
        request: RecognizeRequest,
        *,
        retry_target: str,
    ) -> RawTranscriptionResult:
        retry = AsyncRetry(predicate=_is_retryable, initial=1.0, maximum=10.0, multiplier=2.0, timeout=180.0)
        try:
            response = await self._client.recognize(request=request, retry=retry, timeout=180.0)
        except (GoogleAPICallError, RetryError) as exc:
            logger.error("Speech-to-Text recognition failed for %s: %s", retry_target, exc)
            raise SpeechServiceError(f"Speech recognition failed: {exc}") from exc
        return self._parse_response(response)

    @staticmethod
    def _parse_response(response) -> RawTranscriptionResult:  # noqa: ANN001 - SDK response type
        words: list[TranscriptWord] = []
        text_parts: list[str] = []
        last_end_seconds = 0.0

        for result in response.results:
            if not result.alternatives:
                continue
            top_alternative = result.alternatives[0]
            text_parts.append(top_alternative.transcript)

            for word_info in top_alternative.words:
                end_seconds = word_info.end_offset.total_seconds()
                last_end_seconds = max(last_end_seconds, end_seconds)
                words.append(
                    TranscriptWord(
                        word=word_info.word,
                        start_seconds=word_info.start_offset.total_seconds(),
                        end_seconds=end_seconds,
                        speaker_tag=getattr(word_info, "speaker_label", None) or None,
                    )
                )

        full_text = " ".join(part.strip() for part in text_parts if part.strip())
        return RawTranscriptionResult(full_text=full_text, words=words, duration_seconds=last_end_seconds or None)


def _is_retryable(exc: Exception) -> bool:
    """Predicate for retrying only transient Speech-to-Text errors."""
    from google.api_core import exceptions as gexc

    return isinstance(exc, (gexc.ServiceUnavailable, gexc.DeadlineExceeded, gexc.InternalServerError, gexc.TooManyRequests))
