"""Generate one narration clip per scene with ElevenLabs. Sends only the lines below. Key comes from .env."""
import json, sys, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY = next(l.split("=", 1)[1].strip() for l in open(ROOT / ".env") if l.startswith("ELEVENLABS_API_KEY="))
LINES = [
    "No car, but still paying for parking? This Pyrmont two-bedroom is about a thousand and ten dollars a week. How much of that is empty concrete?",
    "Our model prices both, for every flat. Without the space, about nine hundred and thirty. That's eighty dollars a week, over four thousand a year.",
    "A tenant with no car can ask for the flat without it. An owner can lease the space to someone who needs it.",
    "Tested on two thousand six hundred later adverts, our typical miss was about forty dollars a week, and eighty-two percent landed within a hundred.",
    "And it changes by area. Across Pyrmont, a space is worth about sixty-seven dollars. In Waterloo, about twenty-six.",
    "Know what you are paying for. Two Bulbs, One Source.",
]
H = {"xi-api-key": KEY, "User-Agent": "predict-flat-price-pitch", "Content-Type": "application/json"}


def call(url, data=None):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data else None, headers=H)
    try:
        return urllib.request.urlopen(req, timeout=60).read()
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}")


voices = json.loads(call("https://api.elevenlabs.io/v1/voices"))["voices"]
print("voices:", [(v["name"].split(" - ")[0], v.get("labels", {}).get("accent")) for v in voices][:24])
pick = next((v for v in voices if v.get("labels", {}).get("accent") in ("australian", "Australian")), None) \
    or next((v for v in voices if v["name"].startswith(("Charlotte", "Alice", "Matilda", "Sarah"))), voices[0])
print("using:", pick["name"], pick.get("labels", {}).get("accent"))
for i, t in enumerate(LINES):
    b = call(f"https://api.elevenlabs.io/v1/text-to-speech/{pick['voice_id']}?output_format=mp3_44100_128",
             dict(text=t, model_id="eleven_multilingual_v2", voice_settings=dict(stability=.5, similarity_boost=.75, speed=1.1)))
    (ROOT / f"video/audio/scene_{i}.mp3").write_bytes(b)
print("clips:", len(LINES), "chars:", sum(map(len, LINES)))
