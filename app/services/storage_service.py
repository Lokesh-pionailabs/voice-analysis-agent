"""Google Cloud Storage integration for staging uploaded audio files."""

import asyncio
from datetime import timedelta

from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError

from app.utils.config import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class StorageServiceError(Exception):
    """Raised when a Cloud Storage operation fails after retries."""


class StorageService:
    """Handles uploading audio files to GCS for durable, resumable access.

    Speech-to-Text v2 (and Gemini, for large files) work far more reliably
    against a `gs://` URI than an inline byte payload, so every uploaded
    audio file is first staged in Cloud Storage.
    """

    def __init__(self, settings: Settings, client: storage.Client | None = None) -> None:
        """Initialize the service.

        Args:
            settings: Application settings (bucket name, etc).
            client: Optional pre-built `storage.Client`, primarily for testing.
                When omitted, a client is built using Application Default
                Credentials, which is the recommended production pattern.
        """
        self._settings = settings
        self._bucket_name = settings.gcs_bucket_name
        if self._bucket_name:
            self._client = client or storage.Client(project=settings.gcp_project_id)
        else:
            self._client = client

    async def upload_audio(self, *, destination_path: str, content: bytes, content_type: str) -> str:
        """Upload raw audio bytes to GCS and return the `gs://` URI.

        The blocking `google-cloud-storage` SDK call is offloaded to a
        thread so it doesn't block the FastAPI event loop.

        Args:
            destination_path: Object path within the bucket, e.g.
                "uploads/2026/07/03/<request_id>.wav".
            content: Raw audio bytes.
            content_type: MIME type of the audio content.

        Returns:
            The `gs://bucket/path` URI of the uploaded object.

        Raises:
            StorageServiceError: If the upload fails.
        """
        if not self._bucket_name:
            raise StorageServiceError("GCS bucket is not configured for this deployment.")

        try:
            return await asyncio.to_thread(self._upload_sync, destination_path, content, content_type)
        except GoogleCloudError as exc:
            logger.error("GCS upload failed for %s: %s", destination_path, exc)
            raise StorageServiceError(f"Failed to upload audio to GCS: {exc}") from exc

    def _upload_sync(self, destination_path: str, content: bytes, content_type: str) -> str:
        bucket = self._client.bucket(self._bucket_name)
        blob = bucket.blob(destination_path)
        blob.upload_from_string(content, content_type=content_type)
        gcs_uri = f"gs://{self._bucket_name}/{destination_path}"
        logger.info("Uploaded audio to %s (%d bytes)", gcs_uri, len(content))
        return gcs_uri

    async def generate_signed_url(self, gcs_uri: str) -> str:
        """Generate a time-limited signed URL for a stored object, if needed.

        Useful for allowing downstream systems or dashboards to fetch the
        original audio without granting them direct bucket access.

        Args:
            gcs_uri: The `gs://bucket/path` URI previously returned by
                `upload_audio`.

        Returns:
            A signed HTTPS URL valid for `gcs_signed_url_expiry_seconds`.
        """
        if not self._bucket_name:
            raise StorageServiceError("GCS bucket is not configured for this deployment.")
        return await asyncio.to_thread(self._generate_signed_url_sync, gcs_uri)

    def _generate_signed_url_sync(self, gcs_uri: str) -> str:
        _, _, path = gcs_uri.partition(f"gs://{self._bucket_name}/")
        blob = self._client.bucket(self._bucket_name).blob(path)
        return blob.generate_signed_url(
            expiration=timedelta(seconds=self._settings.gcs_signed_url_expiry_seconds),
            method="GET",
        )

    async def delete_object(self, gcs_uri: str) -> None:
        """Best-effort cleanup of a staged object (e.g. after processing).

        Failures are logged but never raised, since cleanup should not
        fail an otherwise-successful analysis request.
        """
        if not self._bucket_name:
            logger.debug("Skipping delete_object because GCS bucket is not configured.")
            return

        try:
            await asyncio.to_thread(self._delete_sync, gcs_uri)
        except GoogleCloudError as exc:
            logger.warning("Failed to delete GCS object %s: %s", gcs_uri, exc)

    def _delete_sync(self, gcs_uri: str) -> None:
        _, _, path = gcs_uri.partition(f"gs://{self._bucket_name}/")
        self._client.bucket(self._bucket_name).blob(path).delete()
        logger.info("Deleted staged audio object %s", gcs_uri)
