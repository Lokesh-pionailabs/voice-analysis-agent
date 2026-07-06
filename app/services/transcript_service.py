"""Transcript cleaning and structuring.

Turns raw Speech-to-Text output (which may contain filler words, stutter
duplications, noise labels, and word-level timing) into a clean,
speaker-labeled transcript suitable for LLM analysis.
"""

import re

from app.models.response import TranscriptSegment
from app.services.speech_service import RawTranscriptionResult, TranscriptWord
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Common English filler words / verbal tics to strip. Kept conservative to
# avoid removing words that carry meaning (e.g. "well" is ambiguous and
# intentionally excluded).
_FILLER_WORDS = {
    "um", "umm", "uh", "uhh", "uhm", "erm", "ah", "hmm", "mhm",
    "like", "you know", "i mean", "sort of", "kind of",
}

# Noise / non-speech labels Speech-to-Text or upstream tooling may emit.
_NOISE_LABEL_PATTERN = re.compile(
    r"\[(noise|music|silence|inaudible|crosstalk|laughter|background noise)\]",
    re.IGNORECASE,
)

_TIMESTAMP_PATTERN = re.compile(r"\(?\d{1,2}:\d{2}(:\d{2})?\)?")

_MULTI_SPACE_PATTERN = re.compile(r"\s{2,}")


class TranscriptService:
    """Cleans raw transcription output and structures it for analysis."""

    def clean_text(self, text: str) -> str:
        """Remove noise labels, timestamps, and filler/duplicated words.

        Args:
            text: Raw transcript text.

        Returns:
            A cleaned, single-spaced transcript string.
        """
        cleaned = _NOISE_LABEL_PATTERN.sub("", text)
        cleaned = _TIMESTAMP_PATTERN.sub("", cleaned)
        cleaned = self._remove_filler_words(cleaned)
        cleaned = self._collapse_duplicated_words(cleaned)
        cleaned = _MULTI_SPACE_PATTERN.sub(" ", cleaned).strip()
        return cleaned

    def _remove_filler_words(self, text: str) -> str:
        for filler in sorted(_FILLER_WORDS, key=len, reverse=True):
            pattern = re.compile(rf"(?<!\w){re.escape(filler)}(?!\w)[,]?", re.IGNORECASE)
            text = pattern.sub("", text)
        return text

    def _collapse_duplicated_words(self, text: str) -> str:
        """Collapse immediate word repetitions caused by stutter/ASR artifacts.

        e.g. "I I want to to cancel" -> "I want to cancel".
        Case-insensitive comparison; keeps the first occurrence's casing.
        """
        tokens = text.split()
        result: list[str] = []
        for token in tokens:
            normalized = re.sub(r"[^\w']", "", token).lower()
            if result:
                prev_normalized = re.sub(r"[^\w']", "", result[-1]).lower()
                if normalized and normalized == prev_normalized:
                    continue
            result.append(token)
        return " ".join(result)

    def build_segments(self, raw: RawTranscriptionResult) -> list[TranscriptSegment]:
        """Group word-level timing/speaker data into speaker-labeled segments.

        Consecutive words from the same speaker are merged into a single
        segment. Falls back to a single unlabeled segment if no word-level
        data is available.

        Args:
            raw: The raw transcription result from `SpeechService`.

        Returns:
            A list of `TranscriptSegment` ordered by time.
        """
        if not raw.words:
            cleaned = self.clean_text(raw.full_text)
            return [TranscriptSegment(speaker=None, start_seconds=None, end_seconds=None, text=cleaned)] if cleaned else []

        segments: list[TranscriptSegment] = []
        current_speaker: int | str | None = None
        current_words: list[TranscriptWord] = []

        def flush() -> None:
            if not current_words:
                return
            text = self.clean_text(" ".join(w.word for w in current_words))
            if not text:
                return
            segments.append(
                TranscriptSegment(
                    speaker=f"Speaker {current_speaker}" if current_speaker is not None else None,
                    start_seconds=current_words[0].start_seconds,
                    end_seconds=current_words[-1].end_seconds,
                    text=text,
                )
            )

        for word in raw.words:
            if word.speaker_tag != current_speaker and current_words:
                flush()
                current_words = []
            current_speaker = word.speaker_tag
            current_words.append(word)
        flush()

        return segments

    def render_for_llm(self, segments: list[TranscriptSegment]) -> str:
        """Render speaker-labeled segments into plain text for the LLM prompt.

        Args:
            segments: Structured transcript segments.

        Returns:
            A human-readable transcript, e.g.:
                "Speaker 1: Thanks for calling, how can I help?
                 Speaker 2: Hi, I have a billing question."
        """
        lines = []
        for seg in segments:
            prefix = f"{seg.speaker}: " if seg.speaker else ""
            lines.append(f"{prefix}{seg.text}")
        return "\n".join(lines)
