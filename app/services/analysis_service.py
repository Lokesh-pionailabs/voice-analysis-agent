"""Vertex AI Gemini integration for deep conversation analysis.

Uses the `google-genai` SDK's Vertex AI backend (the current, non-deprecated
client for Gemini on Vertex AI) with a JSON response schema so the model is
constrained to return output matching `VoiceAnalysisResult` directly,
eliminating brittle prompt-only JSON parsing.
"""

import time

from google import genai
from google.api_core.exceptions import GoogleAPICallError
from google.genai import types
from pydantic import ValidationError

from app.models.response import VoiceAnalysisResult
from app.prompts.analysis_prompt import SYSTEM_INSTRUCTION, build_analysis_prompt
from app.utils.config import Settings
from app.utils.helpers import extract_json_object
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AnalysisServiceError(Exception):
    """Raised when Gemini analysis fails after all retries."""


class AnalysisService:
    """Runs transcript analysis through Gemini 2.5 Pro on Vertex AI."""

    def __init__(self, settings: Settings, client: genai.Client | None = None) -> None:
        """Initialize the service.

        Args:
            settings: Application settings.
            client: Optional pre-built `genai.Client`, primarily for testing.
                When omitted, a Vertex AI-backed client is built using
                Application Default Credentials.
        """
        self._settings = settings
        self._client = client or genai.Client(
            vertexai=True,
            project=settings.gcp_project_id,
            location=settings.gcp_region,
        )

    async def analyze_transcript(self, transcript: str) -> VoiceAnalysisResult:
        """Send a cleaned transcript to Gemini and return a validated analysis.

        Args:
            transcript: The cleaned, speaker-labeled transcript text.

        Returns:
            A validated `VoiceAnalysisResult`.

        Raises:
            AnalysisServiceError: If Gemini fails, or returns output that
                cannot be parsed/validated after all retries.
        """
        if not transcript.strip():
            raise AnalysisServiceError("Cannot analyze an empty transcript.")

        prompt = build_analysis_prompt(transcript)
        generation_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=self._settings.vertex_temperature,
            top_p=self._settings.vertex_top_p,
            max_output_tokens=self._settings.vertex_max_output_tokens,
            response_mime_type="application/json",
            response_schema=VoiceAnalysisResult,
        )

        last_error: Exception | None = None
        for attempt in range(1, self._settings.vertex_max_retries + 2):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self._settings.vertex_model_name,
                    contents=prompt,
                    config=generation_config,
                )
                return self._parse_response(response)
            except GoogleAPICallError as exc:
                last_error = exc
                logger.warning("Gemini call failed (attempt %d): %s", attempt, exc)
            except (ValidationError, ValueError) as exc:
                last_error = exc
                logger.warning("Gemini returned invalid analysis JSON (attempt %d): %s", attempt, exc)

            if attempt <= self._settings.vertex_max_retries:
                await self._backoff(attempt)

        logger.error("Gemini analysis failed after all retries: %s", last_error)
        raise AnalysisServiceError(f"Transcript analysis failed: {last_error}") from last_error

    @staticmethod
    async def _backoff(attempt: int) -> None:
        import asyncio

        await asyncio.sleep(min(2**attempt, 20))

    @staticmethod
    def _parse_response(response) -> VoiceAnalysisResult:  # noqa: ANN001 - SDK response type
        # Prefer the SDK's parsed object when response_schema parsing succeeded.
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, VoiceAnalysisResult):
            return parsed

        raw_text = response.text
        if not raw_text:
            raise ValueError("Gemini returned an empty response.")

        payload = extract_json_object(raw_text)
        return VoiceAnalysisResult.model_validate(payload)
