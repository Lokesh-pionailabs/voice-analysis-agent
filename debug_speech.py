import asyncio
from app.utils.config import get_settings
from app.services.speech_service import SpeechService
from google.cloud.speech_v2 import SpeechAsyncClient

async def main():
    settings = get_settings()
    service = SpeechService(settings)
    try:
        # A tiny valid 44-byte wav file with 0 data length
        wav_header = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\xbb\x00\x00\x00w\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        res = await service.transcribe_audio_content(wav_header)
        print("Success:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
