"""
Integration test for `POST /api/v1/analyze`.

Exercises the full FastAPI request/response cycle with the Google Cloud
SDK clients replaced by fakes via dependency overrides, so the test
suite runs with no network access and no real credentials.
"""

import io

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import dependencies as deps
from app.models.response import (
    CommunicationSkills,
    Compliance,
    ConversationQuality,
    CustomerHandling,
    Emotion,
    EmotionLabel,
    IntentLabel,
    LanguageQuality,
    Risk,
    RiskLevel,
    Sentiment,
    SentimentLabel,
    VoiceAnalysisResult,
)
from app.services.speech_service import RawTranscriptionResult, TranscriptWord
from app.utils.config import Settings


class FakeStorageService:
    async def upload_audio(self, *, destination_path: str, content: bytes, content_type: str) -> str:
        return f"gs://test-bucket/{destination_path}"


class FakeSpeechService:
    async def transcribe_gcs_uri(self, gcs_uri: str, *, language_codes=None, max_retries: int = 3):
        return self._fake_result()

    async def transcribe_audio_content(self, content: bytes, *, language_codes=None, max_retries: int = 3):
        return self._fake_result()

    def _fake_result(self):
        return RawTranscriptionResult(
            full_text="Hello thanks for calling how can I help you today",
            words=[
                TranscriptWord(word="Hello", start_seconds=0.0, end_seconds=0.4, speaker_tag=1),
                TranscriptWord(word="thanks", start_seconds=0.4, end_seconds=0.8, speaker_tag=1),
            ],
            duration_seconds=12.5,
        )


class FakeAnalysisService:
    async def analyze_transcript(self, transcript: str) -> VoiceAnalysisResult:
        return VoiceAnalysisResult(
            overall_score=88,
            overall_grade="B",
            overall_explanation="Solid, professional call.",
            communication=CommunicationSkills(score=88, clarity=90, fluency=85, confidence=88, professionalism=90),
            customer_handling=CustomerHandling(
                score=87, greeting=90, active_listening=85, empathy=86, understanding=88, politeness=90
            ),
            conversation_quality=ConversationQuality(
                score=89, logical_flow=90, relevance=88, completeness=87, problem_solving=90
            ),
            language_quality=LanguageQuality(score=90, grammar=92, vocabulary=88, tone=90, sentence_quality=90),
            compliance=Compliance(
                score=95,
                proper_greeting=True,
                verification_performed=True,
                proper_closing=True,
                professional_language=True,
                no_abusive_language=True,
                issues=[],
            ),
            sentiment=Sentiment(label=SentimentLabel.POSITIVE, confidence=0.92),
            emotion=Emotion(primary=EmotionLabel.CALM, confidence=0.88),
            intent=[IntentLabel.SUPPORT],
            risk=Risk(
                level=RiskLevel.LOW,
                escalation_risk=False,
                customer_dissatisfaction=False,
                compliance_issues=False,
                potential_churn=False,
                false_information=False,
                aggressive_language=False,
                issues=[],
            ),
            topics=["billing"],
            strengths=["Clear greeting"],
            weaknesses=["Could confirm resolution more explicitly"],
            recommendations=["Summarize next steps before closing"],
            summary="Customer's billing question was resolved professionally.",
            detailed_summary="The agent greeted the customer, verified their account, and resolved a billing question.",
            action_items=["Send confirmation email"],
        )


@pytest.fixture
def app_with_fakes():
    from app.main import app

    app.dependency_overrides[deps.get_storage_service] = lambda: FakeStorageService()
    app.dependency_overrides[deps.get_speech_service] = lambda: FakeSpeechService()
    app.dependency_overrides[deps.get_analysis_service] = lambda: FakeAnalysisService()
    yield app
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_analyze_endpoint_returns_full_analysis(app_with_fakes):
    transport = ASGITransport(app=app_with_fakes)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"audio_file": ("call.wav", io.BytesIO(b"fake audio bytes"), "audio/wav")}
        response = await client.post("/api/v1/analyze", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["analysis"]["overall_score"] == 88
    assert body["analysis"]["overall_grade"] == "B"
    assert body["gcs_uri"].startswith("gs://test-bucket/uploads/")
    assert "Speaker 1" in body["transcript"]


@pytest.mark.asyncio
async def test_analyze_endpoint_rejects_unsupported_extension(app_with_fakes):
    transport = ASGITransport(app=app_with_fakes)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"audio_file": ("call.txt", io.BytesIO(b"not audio"), "text/plain")}
        response = await client.post("/api/v1/analyze", files=files)

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_health_check(app_with_fakes):
    transport = ASGITransport(app=app_with_fakes)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_analyze_endpoint_saves_transcript_locally(tmp_path):
    from app.api import dependencies as deps
    from app.utils.config import Settings

    settings = Settings(
        gcp_project_id="test-project",
        gcs_bucket_name=None,
        local_transcript_path=str(tmp_path / "transcripts"),
        api_key=None,
    )

    from app.main import app

    app.dependency_overrides[deps.get_storage_service] = lambda: FakeStorageService()
    app.dependency_overrides[deps.get_speech_service] = lambda: FakeSpeechService()
    app.dependency_overrides[deps.get_analysis_service] = lambda: FakeAnalysisService()
    app.dependency_overrides[deps.get_settings] = lambda: settings

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"audio_file": ("call.wav", io.BytesIO(b"fake audio bytes"), "audio/wav")}
        response = await client.post("/api/v1/analyze", files=files)

    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["gcs_uri"] is None
    transcript_file = tmp_path / "transcripts" / f"{body['request_id']}.txt"
    assert transcript_file.exists()
    assert transcript_file.read_text(encoding="utf-8") == body["transcript"]
