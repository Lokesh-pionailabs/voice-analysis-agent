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

    def _build_recognition_config(
        self,
        *,
        language_codes: list[str] | None = None,
        inline_content: bool = False,
    ) -> RecognitionConfig:
        # Speaker diarization is NOT supported with inline Recognize requests
        # for any current model (long, chirp_2, etc.) — only with
        # BatchRecognize (GCS). Skip it for inline content to avoid 400 errors.
        diarization = None
        if self._settings.enable_speaker_diarization and not inline_content:
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
        format: str | None = None,
    ) -> RawTranscriptionResult:
        """Transcribe raw audio bytes without staging in GCS, automatically chunking long audio.

        Uses temp files for decoding (critical for formats like weba/webm that
        require seeking) and splits long audio into overlapping chunks to avoid
        cutting words at boundaries.

        Args:
            content: Raw audio bytes.
            language_codes: Optional override of the configured language codes.
            max_retries: Number of retry attempts on transient failures.
            format: Audio format hint (e.g. 'weba', 'wav', 'mp3').

        Returns:
            The raw transcription result.

        Raises:
            SpeechServiceError: If transcription fails after all retries.
        """
        import asyncio
        import io
        import os
        import tempfile

        from pydub import AudioSegment

        # --- Decode audio in a threadpool (pydub/ffmpeg is blocking I/O) ---
        def _load_audio() -> AudioSegment:
            # Always write to a temp file first: weba/webm containers require
            # seeking which BytesIO can't provide reliably to ffmpeg.
            decode_ext = format or "wav"
            if decode_ext == "weba":
                decode_ext = "webm"

            with tempfile.NamedTemporaryFile(suffix=f".{decode_ext}", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                audio = AudioSegment.from_file(tmp_path, format=decode_ext)
                # Normalize to 16 kHz 16-bit mono — optimal for Google STT
                audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)
                return audio
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        audio = await asyncio.to_thread(_load_audio)
        duration_seconds = len(audio) / 1000.0
        logger.info("Audio decoded: %.1f seconds, format=%s", duration_seconds, format)

        # --- Short audio: send in a single request ---
        if duration_seconds <= 55:
            buf = io.BytesIO()
            audio.export(buf, format="wav")
            wav_bytes = buf.getvalue()
            result = await self._transcribe_recognize_request(
                RecognizeRequest(
                    recognizer=self._recognizer_path,
                    config=self._build_recognition_config(language_codes=language_codes, inline_content=True),
                    content=wav_bytes,
                ),
                retry_target="inline audio",
            )
            result.duration_seconds = duration_seconds
            return result

        # --- Long audio: split into overlapping chunks ---
        chunk_length_ms = 58 * 1000   # 58 s per chunk (under the 60 s inline limit)
        overlap_ms = 3 * 1000          # 3 s overlap to avoid cutting words

        step_ms = chunk_length_ms - overlap_ms  # effective advance per chunk

        chunks: list[tuple[int, AudioSegment, float]] = []  # (index, segment, offset_seconds)
        pos = 0
        idx = 0
        while pos < len(audio):
            end = min(pos + chunk_length_ms, len(audio))
            chunks.append((idx, audio[pos:end], pos / 1000.0))
            pos += step_ms
            idx += 1
            # Avoid a tiny trailing chunk (< 2 s) that often produces garbage
            if len(audio) - pos < 2000:
                # Extend the last chunk to cover the remainder
                last_idx, _, last_offset = chunks[-1]
                chunks[-1] = (last_idx, audio[int(last_offset * 1000):], last_offset)
                break

        logger.info(
            "Splitting %.1f s audio into %d overlapping chunks (chunk=%.0f s, overlap=%.0f s)",
            duration_seconds, len(chunks), chunk_length_ms / 1000, overlap_ms / 1000,
        )

        async def _transcribe_chunk(index: int, chunk: AudioSegment, offset_s: float) -> tuple[int, float, RawTranscriptionResult]:
            def _export():
                buf = io.BytesIO()
                chunk.export(buf, format="wav")
                return buf.getvalue()
            
            chunk_bytes = await asyncio.to_thread(_export)

            res = await self._transcribe_recognize_request(
                RecognizeRequest(
                    recognizer=self._recognizer_path,
                    config=self._build_recognition_config(language_codes=language_codes, inline_content=True),
                    content=chunk_bytes,
                ),
                retry_target=f"chunk {index}",
            )
            return index, offset_s, res

        # Use a semaphore to limit concurrency (10 at a time instead of 2)
        # to dramatically speed up transcription of long audio files
        semaphore = asyncio.Semaphore(10)

        async def _sem_transcribe(index: int, chunk: AudioSegment, offset_s: float) -> tuple[int, float, RawTranscriptionResult]:
            async with semaphore:
                return await _transcribe_chunk(index, chunk, offset_s)

        tasks = [_sem_transcribe(i, seg, off) for i, seg, off in chunks]
        results = await asyncio.gather(*tasks)
        results_sorted = sorted(results, key=lambda x: x[0])

        # --- Merge chunk results, de-duplicating the overlap region ---
        full_text_parts: list[str] = []
        all_words: list[TranscriptWord] = []
        global_end_seconds = 0.0  # tracks the latest word end across all merged chunks

        for index, offset_s, res in results_sorted:
            # Shift word timestamps by the chunk's offset in the full audio
            for w in res.words:
                w.start_seconds += offset_s
                w.end_seconds += offset_s

            if index == 0:
                # First chunk: take everything
                if res.full_text.strip():
                    full_text_parts.append(res.full_text)
                all_words.extend(res.words)
                if res.words:
                    global_end_seconds = res.words[-1].end_seconds
            else:
                # Subsequent chunks: skip words that fall within the overlap
                # region (i.e., words whose start_seconds < global_end_seconds)
                # to avoid duplicating words at the boundary.
                new_words = [w for w in res.words if w.start_seconds >= global_end_seconds - 0.15]
                new_text_parts = [w.word for w in new_words]

                if new_text_parts:
                    full_text_parts.append(" ".join(new_text_parts))
                all_words.extend(new_words)
                if new_words:
                    global_end_seconds = max(global_end_seconds, new_words[-1].end_seconds)

        return RawTranscriptionResult(
            full_text=" ".join(full_text_parts),
            words=all_words,
            duration_seconds=duration_seconds,
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
