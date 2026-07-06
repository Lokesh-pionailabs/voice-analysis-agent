"""Unit tests for `ScoringService`."""

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
from app.services.scoring_service import ScoringService


def _base_result(**overrides) -> VoiceAnalysisResult:
    defaults = dict(
        overall_score=95,
        overall_grade="A",
        overall_explanation="Great call.",
        communication=CommunicationSkills(score=90, clarity=90, fluency=90, confidence=90, professionalism=90),
        customer_handling=CustomerHandling(
            score=90, greeting=90, active_listening=90, empathy=90, understanding=90, politeness=90
        ),
        conversation_quality=ConversationQuality(
            score=90, logical_flow=90, relevance=90, completeness=90, problem_solving=90
        ),
        language_quality=LanguageQuality(score=90, grammar=90, vocabulary=90, tone=90, sentence_quality=90),
        compliance=Compliance(
            score=90,
            proper_greeting=True,
            verification_performed=True,
            proper_closing=True,
            professional_language=True,
            no_abusive_language=True,
            issues=[],
        ),
        sentiment=Sentiment(label=SentimentLabel.POSITIVE, confidence=0.9),
        emotion=Emotion(primary=EmotionLabel.CALM, confidence=0.9),
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
        topics=[],
        strengths=[],
        weaknesses=[],
        recommendations=[],
        summary="Good call.",
        detailed_summary="Detailed summary.",
        action_items=[],
    )
    defaults.update(overrides)
    return VoiceAnalysisResult(**defaults)


def test_grade_matches_score_after_normalization():
    result = _base_result(overall_score=85, overall_grade="F")
    normalized = ScoringService().normalize(result)
    assert normalized.overall_grade == "B"


def test_score_capped_when_abusive_language_present():
    result = _base_result(overall_score=95)
    result.compliance.no_abusive_language = False
    normalized = ScoringService().normalize(result)
    assert normalized.overall_score <= 40
    assert normalized.overall_grade == "F"


def test_score_capped_on_critical_risk():
    result = _base_result(overall_score=95)
    result.risk.level = RiskLevel.CRITICAL
    normalized = ScoringService().normalize(result)
    assert normalized.overall_score <= 40


def test_score_unchanged_for_clean_call():
    result = _base_result(overall_score=95, overall_grade="F")
    normalized = ScoringService().normalize(result)
    assert normalized.overall_score == 95
    assert normalized.overall_grade == "A"
