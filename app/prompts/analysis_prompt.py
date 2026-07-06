"""
Prompt template for the Gemini conversation-analysis step.

Kept isolated from `analysis_service.py` so prompt engineering changes
don't require touching orchestration code, and so prompts can be
versioned / A-B tested independently.
"""

SYSTEM_INSTRUCTION = """You are an expert call-quality auditor and conversation analyst \
with two decades of experience in contact-center quality assurance, linguistics, and \
customer experience management. You analyze call transcripts with rigor, objectivity, \
and consistency. You never invent facts that are not supported by the transcript. \
You always respond with a single valid JSON object and nothing else - no markdown \
fences, no commentary, no preamble."""

# The response schema is passed separately to the model via
# `generation_config.response_schema` for guaranteed structured output.
# This template only carries the analytical instructions.
ANALYSIS_PROMPT_TEMPLATE = """\
Analyze the following customer service call transcript and produce a complete, \
objective quality assessment. Base every judgment strictly on evidence present in \
the transcript - do not assume information that was not said.

## Scoring rules
- All numeric scores are integers from 0 to 100.
- Be discriminating: reserve 90+ for genuinely excellent performance, use the full \
  range, and justify the overall score in `overall_explanation`.
- `overall_grade` should be a letter grade consistent with `overall_score` \
  (A: 90-100, B: 80-89, C: 70-79, D: 60-69, F: <60).

## Dimensions to evaluate
1. Communication skills: clarity, fluency, confidence, professionalism.
2. Customer handling: greeting, active listening, empathy, understanding, politeness.
3. Conversation quality: logical flow, relevance, completeness, problem solving.
4. Language quality: grammar, vocabulary, tone, sentence quality.
5. Compliance: proper greeting, identity/account verification, proper closing, \
   professional language throughout, absence of abusive language. List any specific \
   compliance issues found.
6. Sentiment: overall call sentiment (Positive/Neutral/Negative/Mixed) with confidence.
7. Emotion: the primary emotion expressed by the customer, with confidence, plus any \
   notable secondary emotions.
8. Intent: one or more customer intents from: Complaint, Support, Inquiry, Purchase, \
   Cancellation, Feedback, Follow-up, Other.
9. Risk detection: escalation risk, customer dissatisfaction, compliance issues, \
   potential churn, false information given by the agent, aggressive language by \
   either party. Set an overall risk level (Low/Medium/High/Critical) and list \
   specific issues found, if any.
10. Key topics discussed.
11. Strengths demonstrated by the agent.
12. Weaknesses or missed opportunities.
13. Actionable improvement recommendations.
14. Summary: a 1-2 sentence executive summary, a more detailed multi-sentence \
    summary, and a list of concrete action items arising from the call.

## Transcript
Speaker labels (if present) indicate different participants but may not always \
map perfectly to "Agent" vs "Customer" - infer roles from context.

\"\"\"
{transcript}
\"\"\"

Return ONLY the JSON object matching the required schema.
"""


def build_analysis_prompt(transcript: str) -> str:
    """Render the analysis prompt for a given cleaned transcript.

    Args:
        transcript: The cleaned, speaker-labeled transcript text.

    Returns:
        The fully rendered prompt string ready to send to Gemini.
    """
    return ANALYSIS_PROMPT_TEMPLATE.format(transcript=transcript.strip())
