import requests
import json
import time
import subprocess
import os
import sys

# Start the uvicorn server in a subprocess
print("Starting server...")
env = os.environ.copy()
env["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("voice-ai.json")
process = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8081"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Wait for server to start
time.sleep(3)

print("Sending request...")
try:
    with open("dummy.weba", "rb") as f:
        files = {"audio_file": ("dummy.weba", f, "audio/webm")}
        response = requests.post("http://localhost:8081/api/v1/analyze", files=files)
        
    print(f"Status Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print("Response Text:", response.text)
finally:
    # Kill the server and get logs
    process.terminate()
    stdout, stderr = process.communicate()
    print("\n--- SERVER LOGS ---")
    print(stdout.decode('utf-8'))
    print(stderr.decode('utf-8'))
