"""Unit tests for `TranscriptService`."""

from app.services.speech_service import RawTranscriptionResult, TranscriptWord
from app.services.transcript_service import TranscriptService


def test_clean_text_removes_noise_labels_and_timestamps():
    service = TranscriptService()
    raw = "Hello (00:01) [noise] there, how are you [background noise] doing today?"
    cleaned = service.clean_text(raw)
    assert "noise" not in cleaned.lower()
    assert "00:01" not in cleaned
    assert "how are you" in cleaned.lower()


def test_clean_text_removes_filler_words():
    service = TranscriptService()
    raw = "um so I uh wanted to like ask about my bill you know"
    cleaned = service.clean_text(raw)
    for filler in ("um", "uh", "like", "you know"):
        assert filler not in cleaned.lower().split()


def test_collapse_duplicated_words():
    service = TranscriptService()
    raw = "I I want to to cancel my my subscription"
    cleaned = service.clean_text(raw)
    assert cleaned.lower().split().count("i") == 1
    assert cleaned.lower().split().count("to") == 1


def test_build_segments_groups_by_speaker():
    service = TranscriptService()
    words = [
        TranscriptWord(word="Hello", start_seconds=0.0, end_seconds=0.5, speaker_tag=1),
        TranscriptWord(word="there", start_seconds=0.5, end_seconds=1.0, speaker_tag=1),
        TranscriptWord(word="Hi", start_seconds=1.1, end_seconds=1.3, speaker_tag=2),
    ]
    raw_result = RawTranscriptionResult(full_text="Hello there Hi", words=words, duration_seconds=1.3)
    segments = service.build_segments(raw_result)

    assert len(segments) == 2
    assert segments[0].speaker == "Speaker 1"
    assert segments[1].speaker == "Speaker 2"


def test_build_segments_falls_back_to_single_segment_without_words():
    service = TranscriptService()
    raw_result = RawTranscriptionResult(full_text="hello world", words=[], duration_seconds=None)
    segments = service.build_segments(raw_result)
    assert len(segments) == 1
    assert segments[0].speaker is None


def test_render_for_llm_includes_speaker_prefix():
    service = TranscriptService()
    from app.models.response import TranscriptSegment

    segments = [TranscriptSegment(speaker="Speaker 1", start_seconds=0, end_seconds=1, text="Hello there")]
    rendered = service.render_for_llm(segments)
    assert rendered == "Speaker 1: Hello there"
