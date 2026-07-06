"""Small, dependency-free helper utilities shared across services."""

import json
import re
import uuid
from typing import Any


def new_correlation_id() -> str:
    """Generate a short unique ID used to correlate logs for one request."""
    return uuid.uuid4().hex[:12]


def get_file_extension(filename: str) -> str:
    """Return the lowercase file extension (without the dot) of a filename."""
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Extract and parse the first valid JSON object found in `raw_text`.

    Gemini is instructed to return raw JSON, but models occasionally wrap
    output in markdown code fences or add stray prose. This function
    defensively strips fences and locates the outermost `{...}` block
    before parsing, raising a clear error if no valid JSON is found.

    Args:
        raw_text: The raw text returned by the model.

    Returns:
        The parsed JSON object as a dict.

    Raises:
        ValueError: If no valid JSON object could be parsed.
    """
    text = raw_text.strip()

    # Strip ```json ... ``` or ``` ... ``` fences if present.
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: locate the outermost braces and try again.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model output was not valid JSON: {exc}") from exc

    raise ValueError("No JSON object could be located in model output.")


def bytes_to_mb(size_bytes: int) -> float:
    """Convert a byte count to megabytes, rounded to 2 decimal places."""
    return round(size_bytes / (1024 * 1024), 2)
