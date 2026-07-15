"""
Application configuration.

All configuration is loaded from environment variables (optionally via a
`.env` file in local development). No secrets are ever hardcoded here.
Credentials for Google Cloud are resolved through Application Default
Credentials (ADC) - e.g. a service account attached to the Cloud Run
service, `GOOGLE_APPLICATION_CREDENTIALS` pointing at a mounted secret,
or a secret pulled at startup from Secret Manager. See README.md for the
recommended production setup.
"""

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly typed application settings.

    Values are populated from environment variables (case-insensitive).
    Never hardcode credentials or secrets in this file - only variable
    names and safe defaults belong here.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Google Cloud project -------------------------------------------------
    gcp_project_id: str = Field(..., description="GCP project ID")
    gcp_region: str = Field(default="us-central1", description="GCP region for Vertex AI / Speech-to-Text")

    # Credentials are OPTIONAL. In production on Cloud Run, prefer attaching a
    # runtime service account and leaving this unset so ADC is used. This var
    # exists only to support local development or Secret-Manager-mounted files.
    google_application_credentials: str | None = Field(
        default=None,
        description="Path to a service-account JSON key file (local dev only). "
        "Never commit this file or its contents to source control.",
    )

    # --- Cloud Storage ----------------------------------------------------------
    gcs_bucket_name: str | None = Field(
        default=None,
        description="GCS bucket used for staging uploaded audio. When unset, audio is processed inline without Cloud Storage.",
    )
    gcs_signed_url_expiry_seconds: int = Field(default=3600, ge=60, le=86400)
    local_transcript_path: str = Field(
        default="transcripts",
        description="Directory where cleaned transcripts are saved locally.",
    )

    # --- Speech-to-Text -----------------------------------------------------------
    speech_recognizer_id: str = Field(
        default="_", description="Speech-to-Text v2 recognizer ID ('_' uses the implicit default recognizer)"
    )
    speech_language_codes: List[str] = Field(default_factory=lambda: ["en-US"])
    speech_model: str = Field(default="long", description="Speech-to-Text v2 model: 'long', 'short', 'chirp_2', etc.")
    enable_speaker_diarization: bool = Field(default=True)
    diarization_min_speakers: int = Field(default=1, ge=1)
    diarization_max_speakers: int = Field(default=2, ge=1)

    # --- Vertex AI / Gemini -----------------------------------------------------
    vertex_model_name: str = Field(default="gemini-2.5-pro")
    vertex_max_output_tokens: int = Field(default=8192)
    vertex_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    vertex_top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    vertex_request_timeout_seconds: int = Field(default=120, ge=1)
    vertex_max_retries: int = Field(default=3, ge=0, le=10)

    # --- Upload constraints --------------------------------------------------
    max_audio_size_mb: int = Field(default=200, ge=1)
    allowed_audio_extensions: List[str] = Field(default_factory=lambda: ["mp3", "wav", "m4a", "flac", "weba"])

    # --- App / server ----------------------------------------------------------
    app_env: str = Field(default="production")
    log_level: str = Field(default="INFO")
    api_key: str | None = Field(default=None, description="Optional static API key for inbound auth")
    cors_allow_origins: List[str] = Field(default_factory=lambda: ["*"])

    @field_validator("allowed_audio_extensions", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [item.strip().lower() for item in v.split(",") if item.strip()]
        return v

    @field_validator("gcs_bucket_name", mode="before")
    @classmethod
    def _normalize_bucket_name(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("speech_language_codes", "cors_allow_origins", mode="before")
    @classmethod
    def _split_generic_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide Settings instance.

    Using `lru_cache` means environment variables are read once, at first
    access, avoiding repeated (and potentially inconsistent) environment
    reads across the request lifecycle.
    """
    return Settings()
