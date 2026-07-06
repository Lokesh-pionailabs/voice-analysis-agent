"""Pydantic request models for the API layer."""

from pydantic import BaseModel, Field


class AnalyzeAudioMetadata(BaseModel):
    """Optional metadata that can accompany an audio upload.

    Sent as additional form fields alongside the multipart file upload.
    All fields are optional so the endpoint remains simple to call.
    """

    language_code: str | None = Field(
        default=None, description="BCP-47 language code override, e.g. 'en-US'. Defaults to server configuration."
    )
    agent_name: str | None = Field(default=None, max_length=200)
    call_reference_id: str | None = Field(default=None, max_length=200)
    min_speakers: int | None = Field(default=None, ge=1, le=10)
    max_speakers: int | None = Field(default=None, ge=1, le=10)
