"""
Deterministic scoring validation and normalization.

Gemini produces the primary scores, but this service applies deterministic,
auditable business rules on top of the model output so scoring stays
consistent and defensible even as prompts evolve. This keeps "the model
decides everything" out of the loop for anything that should be a hard
rule (e.g. a grade must match its numeric score; compliance failures
should cap the overall score).
"""

from app.models.response import VoiceAnalysisResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

_GRADE_THRESHOLDS: list[tuple[int, str]] = [
    (90, "A"),
    (80, "B"),
    (70, "C"),
    (60, "D"),
    (0, "F"),
]


class ScoringService:
    """Applies deterministic normalization rules to Gemini's raw analysis."""

    def normalize(self, result: VoiceAnalysisResult) -> VoiceAnalysisResult:
        """Reconcile the letter grade with the numeric score and apply caps.

        Args:
            result: The raw, validated analysis returned by Gemini.

        Returns:
            The same result with `overall_grade` corrected to match
            `overall_score`, and the overall score capped when critical
            compliance or risk issues are present.
        """
        adjusted_score = result.overall_score

        if not result.compliance.no_abusive_language or result.risk.level.value == "Critical":
            capped = min(adjusted_score, 40)
            if capped != adjusted_score:
                logger.info(
                    "Capping overall_score from %d to %d due to abusive language / critical risk",
                    adjusted_score,
                    capped,
                )
            adjusted_score = capped

        result.overall_score = adjusted_score
        result.overall_grade = self._grade_for_score(adjusted_score)
        return result

    @staticmethod
    def _grade_for_score(score: int) -> str:
        for threshold, grade in _GRADE_THRESHOLDS:
            if score >= threshold:
                return grade
        return "F"
