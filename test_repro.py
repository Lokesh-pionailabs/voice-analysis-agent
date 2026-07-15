import asyncio
import os
import io
from pydub import AudioSegment
from app.utils.config import get_settings
from app.services.speech_service import SpeechService

async def main():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("voice-ai.json")
    settings = get_settings()
    service = SpeechService(settings)
    
    with open("dummy.weba", "rb") as f:
        weba_content = f.read()

    # Reproduce exactly what routes.py does
    audio = AudioSegment.from_file(io.BytesIO(weba_content))
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    wav_content = buf.getvalue()

    with open("debug.wav", "wb") as f:
        f.write(wav_content)

    print("Transcoded to WAV, length:", len(wav_content))
    
    try:
        res = await service.transcribe_audio_content(wav_content)
        print("Success!")
    except Exception as e:
        print("Failed:", e)

if __name__ == "__main__":
    asyncio.run(main())
