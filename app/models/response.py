"""
Pydantic response models.

These models define the exact contract returned by `POST /analyze`. They
double as the schema Gemini's structured output is validated against
before being sent to the client, guaranteeing the API never returns
malformed or partial JSON.
"""

from enum import Enum

from pydantic import BaseModel, Field


class SentimentLabel(str, Enum):
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"
    MIXED = "Mixed"


class EmotionLabel(str, Enum):
    HAPPY = "Happy"
    CALM = "Calm"
    FRUSTRATED = "Frustrated"
    ANGRY = "Angry"
    CONFUSED = "Confused"
    EXCITED = "Excited"
    NEUTRAL = "Neutral"


class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class IntentLabel(str, Enum):
    COMPLAINT = "Complaint"
    SUPPORT = "Support"
    INQUIRY = "Inquiry"
    PURCHASE = "Purchase"
    CANCELLATION = "Cancellation"
    FEEDBACK = "Feedback"
    FOLLOW_UP = "Follow-up"
    OTHER = "Other"


class CommunicationSkills(BaseModel):
    score: int = Field(ge=0, le=100)
    clarity: int = Field(ge=0, le=100)
    fluency: int = Field(ge=0, le=100)
    confidence: int = Field(ge=0, le=100)
    professionalism: int = Field(ge=0, le=100)


class CustomerHandling(BaseModel):
    score: int = Field(ge=0, le=100)
    greeting: int = Field(ge=0, le=100)
    active_listening: int = Field(ge=0, le=100)
    empathy: int = Field(ge=0, le=100)
    understanding: int = Field(ge=0, le=100)
    politeness: int = Field(ge=0, le=100)


class ConversationQuality(BaseModel):
    score: int = Field(ge=0, le=100)
    logical_flow: int = Field(ge=0, le=100)
    relevance: int = Field(ge=0, le=100)
    completeness: int = Field(ge=0, le=100)
    problem_solving: int = Field(ge=0, le=100)


class LanguageQuality(BaseModel):
    score: int = Field(ge=0, le=100)
    grammar: int = Field(ge=0, le=100)
    vocabulary: int = Field(ge=0, le=100)
    tone: int = Field(ge=0, le=100)
    sentence_quality: int = Field(ge=0, le=100)


class Compliance(BaseModel):
    score: int = Field(ge=0, le=100)
    proper_greeting: bool
    verification_performed: bool
    proper_closing: bool
    professional_language: bool
    no_abusive_language: bool
    issues: list[str] = Field(default_factory=list)


class Sentiment(BaseModel):
    label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)


class Emotion(BaseModel):
    primary: EmotionLabel
    confidence: float = Field(ge=0.0, le=1.0)
    secondary_emotions: list[EmotionLabel] = Field(default_factory=list)


class Risk(BaseModel):
    level: RiskLevel
    escalation_risk: bool
    customer_dissatisfaction: bool
    compliance_issues: bool
    potential_churn: bool
    false_information: bool
    aggressive_language: bool
    issues: list[str] = Field(default_factory=list)


class Summary(BaseModel):
    executive_summary: str
    detailed_summary: str
    action_items: list[str] = Field(default_factory=list)


class VoiceAnalysisResult(BaseModel):
    """Full structured analysis result produced by Gemini for one call."""

    overall_score: int = Field(ge=0, le=100)
    overall_grade: str = Field(description="Letter grade, e.g. 'A', 'B+', 'C'")
    overall_explanation: str

    communication: CommunicationSkills
    customer_handling: CustomerHandling
    conversation_quality: ConversationQuality
    language_quality: LanguageQuality
    compliance: Compliance
    sentiment: Sentiment
    emotion: Emotion
    intent: list[IntentLabel]
    risk: Risk

    topics: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    summary: str
    detailed_summary: str
    action_items: list[str] = Field(default_factory=list)


class TranscriptSegment(BaseModel):
    speaker: str | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    text: str


class AnalyzeAudioResponse(BaseModel):
    """Top-level response returned by `POST /analyze`."""

    request_id: str
    audio_filename: str
    audio_duration_seconds: float | None = None
    gcs_uri: str | None = None
    transcript: str
    transcript_segments: list[TranscriptSegment] = Field(default_factory=list)
    analysis: VoiceAnalysisResult
    processing_time_ms: int


class ErrorResponse(BaseModel):
    """Standard error envelope returned for all handled failures."""

    request_id: str
    error: str
    detail: str
