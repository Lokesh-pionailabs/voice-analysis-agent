# AI Voice Analysis Agent

Production-grade, end-to-end call analysis pipeline built on **Google Cloud
Vertex AI**. Upload an audio recording; get back a structured JSON quality
assessment covering communication skills, customer handling, compliance,
sentiment, emotion, intent, risk, and actionable recommendations.

> ⚠️ **Security note:** never commit service-account key files or paste them
> into chat/tickets/logs. This project is designed to run entirely on
> Application Default Credentials (a runtime service account on Cloud Run,
> or `gcloud auth application-default login` locally). If a key was ever
> exposed, revoke it in IAM immediately and rotate.

## Pipeline

```
Audio upload (mp3/wav/m4a/flac)
        │
        ▼
Cloud Storage (staging)              storage_service.py
        │
        ▼
Speech-to-Text V2 (batch recognize)  speech_service.py
        │
        ▼
Transcript cleaning + diarization    transcript_service.py
  (filler words, duplicates,
   noise labels, timestamps)
        │
        ▼
Gemini 2.5 Pro on Vertex AI           analysis_service.py
  (structured JSON output,
   schema-validated)
        │
        ▼
Deterministic scoring rules           scoring_service.py
  (grade/score reconciliation,
   compliance/risk caps)
        │
        ▼
Structured JSON response              models/response.py
```

## Project structure

```
app/
├── api/
│   ├── routes.py            # POST /api/v1/analyze
│   └── dependencies.py      # DI providers, optional API-key auth
├── services/
│   ├── speech_service.py    # Speech-to-Text V2
│   ├── transcript_service.py# Cleaning + speaker segmentation
│   ├── analysis_service.py  # Vertex AI Gemini
│   ├── scoring_service.py   # Deterministic score/grade rules
│   └── storage_service.py   # GCS staging
├── models/
│   ├── request.py
│   └── response.py          # Full response schema (also the Gemini schema)
├── prompts/
│   └── analysis_prompt.py
├── utils/
│   ├── logger.py            # Structured JSON logging
│   ├── config.py            # Pydantic settings from env vars
│   └── helpers.py
└── main.py                  # FastAPI app, error handlers, health check
tests/
├── unit/
└── integration/
Dockerfile
docker-compose.yml
requirements.txt
```

## Requirements

- Python 3.12
- A GCP project with billing enabled
- APIs enabled: `speech.googleapis.com`, `aiplatform.googleapis.com`, `storage.googleapis.com`
- A GCS bucket for staging audio uploads
- A Speech-to-Text v2 recognizer (or use the implicit default `_` recognizer)

## Local setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set GCP_PROJECT_ID and GCS_BUCKET_NAME

# Authenticate locally with ADC (no key file needed):
gcloud auth application-default login

uvicorn app.main:app --reload --port 8080
```

Open `http://localhost:8080/docs` for interactive API documentation (Swagger UI).

### Local development with Docker

```bash
# Only if you truly need a key file locally (prefer ADC above):
mkdir -p secrets && cp /path/to/your/local-key.json secrets/sa-key.json

docker compose up --build
```

## API

### `POST /api/v1/analyze`

`multipart/form-data`:

| Field           | Type   | Required | Description                         |
|-----------------|--------|----------|--------------------------------------|
| `audio_file`    | file   | yes      | mp3, wav, m4a, or flac              |
| `language_code` | string | no       | BCP-47 override, e.g. `en-US`       |

**Response 200** — see `app/models/response.py::AnalyzeAudioResponse`. Example shape:

```json
{
  "request_id": "a02dd9e5b66f",
  "audio_filename": "call.wav",
  "audio_duration_seconds": 184.2,
  "gcs_uri": "gs://your-bucket/uploads/2026/07/03/a02dd9e5b66f.wav",
  "transcript": "Speaker 1: Thanks for calling...\nSpeaker 2: Hi, I have a question...",
  "transcript_segments": [ { "speaker": "Speaker 1", "start_seconds": 0.0, "end_seconds": 3.2, "text": "..." } ],
  "analysis": {
    "overall_score": 91,
    "overall_grade": "A",
    "overall_explanation": "...",
    "communication": { "score": 90, "clarity": 94, "fluency": 90, "confidence": 88, "professionalism": 95 },
    "customer_handling": { "score": 89, "greeting": 92, "active_listening": 88, "empathy": 91, "understanding": 88, "politeness": 93 },
    "conversation_quality": { "score": 93, "logical_flow": 94, "relevance": 92, "completeness": 90, "problem_solving": 93 },
    "language_quality": { "score": 90, "grammar": 92, "vocabulary": 88, "tone": 91, "sentence_quality": 90 },
    "compliance": { "score": 95, "proper_greeting": true, "verification_performed": true, "proper_closing": true, "professional_language": true, "no_abusive_language": true, "issues": [] },
    "sentiment": { "label": "Positive", "confidence": 0.96 },
    "emotion": { "primary": "Calm", "confidence": 0.91, "secondary_emotions": [] },
    "intent": ["Support"],
    "risk": { "level": "Low", "escalation_risk": false, "customer_dissatisfaction": false, "compliance_issues": false, "potential_churn": false, "false_information": false, "aggressive_language": false, "issues": [] },
    "topics": ["billing dispute"],
    "strengths": ["Clear, warm greeting"],
    "weaknesses": ["Did not confirm resolution explicitly"],
    "recommendations": ["Summarize next steps before closing the call"],
    "summary": "...",
    "detailed_summary": "...",
    "action_items": ["Send confirmation email"]
  },
  "processing_time_ms": 4231
}
```

**Error envelope** (4xx/5xx): `{"request_id": "...", "error": "...", "detail": "..."}`

### `GET /health`

Liveness/readiness probe for Cloud Run.

## Authentication

- **Inbound (client → API):** optional static `X-API-Key` header, enabled by
  setting `API_KEY`. In production, prefer fronting the service with
  Identity-Aware Proxy, an API Gateway, or Cloud Run's built-in IAM auth
  instead of (or in addition to) this.
- **Outbound (API → Google Cloud):** Application Default Credentials only.
  - **Cloud Run (recommended):** attach a dedicated runtime service account
    via `--service-account`. No key file, no `GOOGLE_APPLICATION_CREDENTIALS`.
  - **Local dev:** `gcloud auth application-default login`, or a key file
    referenced by `GOOGLE_APPLICATION_CREDENTIALS` and never committed.
  - **Secret Manager:** if a key file is unavoidable (e.g. cross-cloud
    federation), fetch it at startup from Secret Manager into memory —
    never bake it into the image or environment variables in plaintext.

### Minimal IAM roles for the runtime service account

| Role                                | Purpose                          |
|--------------------------------------|-----------------------------------|
| `roles/storage.objectAdmin` (bucket-scoped) | Upload/read/delete staged audio |
| `roles/speech.client`                | Speech-to-Text v2 recognition    |
| `roles/aiplatform.user`              | Vertex AI Gemini inference        |
| `roles/logging.logWriter`            | Structured logs to Cloud Logging  |

## Deployment guide (Cloud Run)

```bash
export PROJECT_ID=your-gcp-project-id
export REGION=us-central1
export REPO=voice-analysis
export SERVICE=voice-analysis-agent
export BUCKET=your-audio-staging-bucket
export SA_EMAIL=voice-analysis-runtime@${PROJECT_ID}.iam.gserviceaccount.com

# One-time setup
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  speech.googleapis.com aiplatform.googleapis.com storage.googleapis.com

gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION

gcloud iam service-accounts create voice-analysis-runtime \
  --display-name="Voice Analysis Agent runtime"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/aiplatform.user"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/speech.client"
gsutil iam ch serviceAccount:${SA_EMAIL}:roles/storage.objectAdmin gs://$BUCKET

# Build & push
gcloud builds submit --tag ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:latest

# Deploy — no key file, no plaintext secrets
gcloud run deploy $SERVICE \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:latest \
  --region=$REGION \
  --service-account=$SA_EMAIL \
  --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION},GCS_BUCKET_NAME=${BUCKET},VERTEX_MODEL_NAME=gemini-2.5-pro" \
  --memory=1Gi --cpu=2 --timeout=300 --concurrency=10 \
  --no-allow-unauthenticated   # remove only if the endpoint should be public
```

If you'd rather store `API_KEY` or other sensitive values in Secret Manager:

```bash
gcloud secrets create voice-analysis-api-key --data-file=-
gcloud run deploy $SERVICE --update-secrets=API_KEY=voice-analysis-api-key:latest ...
```

## Testing

```bash
pip install -r requirements.txt
pytest -q                      # unit + integration (mocked GCP clients, no network)
pytest --cov=app -q            # with coverage
```

Integration tests override the DI providers in `app/api/dependencies.py`
with fakes, so the whole request/response cycle (validation, error
handling, transcript cleaning, response shape) is exercised without any
real GCP calls or credentials.

## Best practices implemented

- **Async everywhere**: blocking GCS SDK calls are offloaded via
  `asyncio.to_thread`; Speech-to-Text and Gemini use their native async
  clients.
- **Retries**: exponential backoff on transient Speech-to-Text errors
  (`AsyncRetry`) and on Gemini calls, both configurable via env vars.
- **Structured, schema-validated LLM output**: Gemini is called with
  `response_schema=VoiceAnalysisResult`, so malformed JSON is caught by the
  SDK/Pydantic rather than regex-parsed from free text.
- **Deterministic guardrails on top of the LLM**: `ScoringService` enforces
  hard rules (grade must match score; abusive language / critical risk
  caps the score) so scoring isn't purely at the model's discretion.
- **Clean architecture / SOLID**: routes only orchestrate; each service has
  a single responsibility and is injected via FastAPI `Depends`, making
  every layer independently testable and swappable.
- **Structured JSON logging** with correlation IDs, ready for Cloud Logging.
- **No hardcoded secrets** anywhere in source; credentials resolved via ADC.
- **Non-root, multi-stage Docker image** with a minimal runtime footprint.
- **Input validation**: file extension and size limits enforced before any
  paid API calls are made, minimizing wasted cost on invalid uploads.
- **Idempotent, unique staging paths** (`uploads/YYYY/MM/DD/<request_id>.ext`)
  to avoid collisions and simplify lifecycle-based cleanup (set a GCS
  lifecycle rule to auto-delete staged audio after N days).

## Cost & latency notes

- Speech-to-Text v2's `long` model batches efficiently for calls of any
  length; for very large files, consider the async `BatchRecognize` API
  and a webhook/polling pattern instead of synchronous `Recognize`.
- Gemini `max_output_tokens` and `temperature` are tuned for consistent,
  concise structured output — raise `max_output_tokens` if you extend the
  schema with longer free-text fields.
- Add a GCS lifecycle policy (e.g. delete after 7 days) on the staging
  bucket, or call `StorageService.delete_object` post-analysis if you don't
  need to retain the original audio.

## Extending the system

- Swap `VERTEX_MODEL_NAME` to any Gemini model available in your region.
- Add a `BatchRecognize`-based path in `speech_service.py` for audio over
  ~1 hour.
- Add a persistence layer (e.g. Firestore/BigQuery) by adding a new service
  and calling it from `routes.py` after scoring — no other layer changes.
