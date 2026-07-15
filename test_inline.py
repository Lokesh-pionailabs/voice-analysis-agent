import asyncio
import io
import sys
from pydub import AudioSegment
from google.cloud.speech_v2 import SpeechAsyncClient, RecognizeRequest, RecognitionConfig, AutoDetectDecodingConfig, RecognitionFeatures
from app.utils.config import get_settings
from app.services.speech_service import SpeechService

async def main():
    import os
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("voice-ai.json")
    settings = get_settings()
    service = SpeechService(settings)
    
    with open("dummy.wav", "rb") as f:
        content = f.read()
    
    try:
        res = await service.transcribe_audio_content(content)
        print("Success!")
        print(res.full_text)
    except Exception as e:
        print("Error!")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
