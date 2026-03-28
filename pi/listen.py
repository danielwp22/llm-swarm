import vosk
import subprocess
import tempfile
import os
import json

model = vosk.Model("vosk-model-small-en-us-0.15")
rec = vosk.KaldiRecognizer(model, 16000)

print("Listening... (Ctrl+C to stop)")
while True:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmpfile = f.name

    subprocess.run([
        "arecord", "-D", "plughw:2,0", "-f", "S16_LE",
        "-r", "16000", "-c", "1", "-t", "wav", "-d", "4", tmpfile
    ], stderr=subprocess.DEVNULL)

    with open(tmpfile, "rb") as f:
        f.read(44)  # skip WAV header
        data = f.read()

    if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        text = result.get("text", "").strip()
        if text:
            print(f"Heard: {text}")
    else:
        partial = json.loads(rec.PartialResult())
        print(f"Partial: {partial.get('partial', '')}")

    os.unlink(tmpfile)
