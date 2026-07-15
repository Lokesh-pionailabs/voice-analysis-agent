import asyncio
import os
import io
import traceback
from pydub import AudioSegment
from app.utils.config import get_settings
from app.services.speech_service import SpeechService
from google.cloud.speech_v2 import SpeechAsyncClient, RecognizeRequest, RecognitionConfig, AutoDetectDecodingConfig, RecognitionFeatures

async def main():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("voice-ai.json")
    os.environ["GCP_PROJECT_ID"] = "voice-v2-team"
    os.environ["GCP_REGION"] = "us-central1"
    os.environ["SPEECH_MODEL"] = "long"
    
    settings = get_settings()
    service = SpeechService(settings)
    
    # Create a simple 1 second wav file in memory
    audio = AudioSegment.silent(duration=1000)
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    wav_bytes = buf.getvalue()

    print("Testing inline audio (<= 55s)...")
    try:
        res = await service.transcribe_audio_content(wav_bytes)
        print("Inline Success!")
    except Exception as e:
        print("Inline Failed!")
        traceback.print_exc()

    # Now create a 60 second wav file to test chunking (> 55s)
    print("\nTesting chunked inline audio (> 55s)...")
    audio = AudioSegment.silent(duration=60000)
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    long_wav_bytes = buf.getvalue()
    
    try:
        res = await service.transcribe_audio_content(long_wav_bytes)
        print("Chunked Success!")
    except Exception as e:
        print("Chunked Failed!")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
