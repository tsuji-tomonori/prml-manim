"""Generate WhiteCUL narration, with independently padded storyboard beats."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import urllib.parse
import urllib.request
import wave
from pathlib import Path

from narration_content import SCENES, SYNTHESIS_SETTINGS, estimated_duration, script_hash

SCENE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCENE_DIR / "assets" / "voicevox"
MANIFEST = OUTPUT_DIR / "manifest.json"
CACHE_DIR = SCENE_DIR.parents[2] / ".working" / "voicevox-lines"
SPEAKER = {"label": "VOICEVOX:WhiteCUL", "id": 23, "speed_scale": 1.08}


def wav_duration(path):
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def valid_entry(scene, entry):
    path = OUTPUT_DIR / f"{scene['id']}.wav"
    if entry.get("status") != "generated" or entry.get("script_sha256") != script_hash(scene):
        return False
    if not path.exists() or entry.get("wav_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        return False
    beats = entry.get("beat_durations", [])
    if len(beats) != len(scene["beats"]) or any(d <= 0 for d in beats):
        return False
    cues = entry.get("subtitle_cues", [])
    if [(c.get("id"), c.get("display"), c.get("speech")) for c in cues] != [
            (s["id"], s["display"], s["speech"]) for b in scene["beats"] for s in b["segments"]]:
        return False
    if len(entry.get("beat_speech_ends", [])) != len(beats):
        return False
    try:
        return abs(sum(beats) - wav_duration(path)) < 0.02
    except (wave.Error, EOFError):
        return False


def pending_entry(scene):
    return {"id": scene["id"], "title": scene["title"], "status": "pending",
            "path": None, "script_sha256": script_hash(scene),
            "beat_durations": [estimated_duration(b) for b in scene["beats"]]}


def save_manifest(entries):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix(".tmp.json")
    temporary.write_text(json.dumps({"version": 4, "speaker": SPEAKER, "scenes": entries},
                                    ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(MANIFEST)


def prepare_manifest():
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    old = {e["id"]: e for e in previous.get("scenes", [])}
    entries = [old[s["id"]] if valid_entry(s, old.get(s["id"], {})) else pending_entry(s)
               for s in SCENES]
    valid_ids = {e["id"] for e in entries if e["status"] == "generated"}
    # Only narration assets belonging to this feature; old scene10..13 also go.
    for path in OUTPUT_DIR.glob("scene*.wav"):
        if path.stem not in valid_ids:
            path.unlink()
    save_manifest(entries)
    return entries


def post_json(base, endpoint, params, body=None):
    request = urllib.request.Request(
        f"{base.rstrip('/')}/{endpoint}?{urllib.parse.urlencode(params)}",
        data=body, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def sentence_audio(base, text):
    key = hashlib.sha256(json.dumps([base, "0.25.2", 23, SYNTHESIS_SETTINGS, text],
                                  ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{key}.wav"
    if cache.exists():
        return cache.read_bytes()
    query = json.loads(post_json(base, "audio_query", {"speaker": 23, "text": text}))
    query.update(SYNTHESIS_SETTINGS)
    data = post_json(base, "synthesis", {"speaker": 23}, json.dumps(query).encode())
    cache.write_bytes(data)
    return data


def generate_scene(base, scene):
    path = OUTPUT_DIR / f"{scene['id']}.wav"
    temporary = path.with_suffix(".tmp.wav")
    durations, speech_ends, cues = [], [], []
    expected = None
    total_frames = 0
    with wave.open(str(temporary), "wb") as output:
        for i, beat in enumerate(scene["beats"]):
            print(f"{scene['id']} beat {i + 1}", flush=True)
            beat_start = total_frames
            for segment in beat["segments"]:
                data = sentence_audio(base, segment["speech"])
                with wave.open(io.BytesIO(data), "rb") as source:
                    params = (source.getnchannels(), source.getsampwidth(), source.getframerate())
                    if expected is None:
                        expected = params
                        output.setnchannels(params[0])
                        output.setsampwidth(params[1])
                        output.setframerate(params[2])
                    elif params != expected:
                        raise RuntimeError("VOICEVOX changed WAV format within a scene")
                    count = source.getnframes()
                    cues.append({"beat_index": i, **segment,
                                 "start": total_frames / params[2],
                                 "end": (total_frames + count) / params[2]})
                    output.writeframes(source.readframes(count))
                    total_frames += count
            speech_ends.append((total_frames - beat_start) / params[2])
            # Only a short breath after actual speech; never pad to the old
            # silent-storyboard duration. Align to the 15fps preview boundaries.
            target_seconds = math.ceil(((total_frames - beat_start) / params[2] + .35) * 15) / 15
            target_frames = round(target_seconds * params[2])
            padding = target_frames - (total_frames - beat_start)
            output.writeframes(b"\0" * (padding * params[0] * params[1]))
            total_frames += padding
            durations.append(target_frames / params[2])
    temporary.replace(path)
    return {"id": scene["id"], "title": scene["title"], "status": "generated",
            "path": str(path.relative_to(SCENE_DIR)), "script_sha256": script_hash(scene),
            "wav_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "duration": wav_duration(path), "beat_durations": durations,
            "beat_speech_ends": speech_ends, "subtitle_cues": cues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:50021")
    parser.add_argument("--from-scene", choices=[s["id"] for s in SCENES], default="scene01")
    parser.add_argument("--prepare-only", action="store_true", help="Invalidate stale audio without contacting Engine")
    args = parser.parse_args()
    if not args.prepare_only:
        with urllib.request.urlopen(f"{args.base_url.rstrip('/')}/version", timeout=5) as response:
            print("VOICEVOX Engine", response.read().decode(), flush=True)
    entries = prepare_manifest()
    if args.prepare_only:
        print("Manifest prepared. Audio not regenerated.")
        return
    start = next(i for i, s in enumerate(SCENES) if s["id"] == args.from_scene)
    for i in range(start, len(SCENES)):
        entries[i] = generate_scene(args.base_url, SCENES[i])
        save_manifest(entries)  # Safe to resume after each completed scene.
    print(f"Saved {MANIFEST}")


if __name__ == "__main__":
    main()
