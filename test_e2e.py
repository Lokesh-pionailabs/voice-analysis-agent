import os
import wave
import struct
import requests
import json

def test_stt_via_api():
    print("\n--- TESTING PIPELINE VIA API ---")
    file_name = "dummy.wav"
    sample_rate = 16000
    duration_seconds = 1
    num_samples = sample_rate * duration_seconds

    with wave.open(file_name, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for _ in range(num_samples):
            wav_file.writeframes(struct.pack('h', 0))

    url = "http://localhost:8080/api/v1/analyze"
    try:
        with open(file_name, "rb") as f:
            files = {"audio_file": (file_name, f, "audio/wav")}
            response = requests.post(url, files=files)
            
        print(f"Status: {response.status_code}")
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(response.text)
            
        if response.status_code == 422 and "No speech could be detected" in response.text:
            print("STT Test: PASSED (Properly reached STT, STT processed audio, returned empty transcript as expected)")
        else:
            print("STT Test: UNEXPECTED RESULT")
    except Exception as e:
        print(f"STT Test: FAILED with {e}")
        
    if os.path.exists(file_name):
        os.remove(file_name)

if __name__ == "__main__":
    test_stt_via_api()
