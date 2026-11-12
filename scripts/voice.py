"""v2 Phase 9 (low priority): a voice-over that reads the caption lines aloud, for an optional player toggle (O).

Free, local text-to-speech: Piper (piper-tts, no API key), voice en_US-ryan-high. Piper lives in its own venv so its
onnxruntime can't disturb the OCR's: config.yaml `TTS_PYTHON: ~/.venvs/mm-tts/Scripts/python.exe`. This script only
needs the standard library + piper (+ numpy, which piper brings).
  outputs/<job>/<L>/voice.m4a   each line starts at its caption's start; a line longer than its slot is sped up a
                                little (<= 1.15x) so it never runs into the next one; loudnorm I=-14 (speech over music)
  outputs/<job>/<L>/voice.json  [[start, end], ...] when the voice speaks: the player turns the song (and the game
                                sound) down there while the voice is on
usage: <TTS_PYTHON> scripts/voice.py inputs/<job> --len 60|120
"""
import hashlib, json, subprocess, sys, wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VOICE = ROOT / "work" / "_tts" / "en_US-ryan-high.onnx"
SR = 44100
TEMPO_MAX = 1.15
GAP = 0.25           # keep this much air before the next line


def synth(voice, text, cache):
    f = cache / (hashlib.sha1(text.encode()).hexdigest()[:16] + ".wav")
    if not f.exists():
        with wave.open(str(f), "wb") as w:
            voice.synthesize_wav(text, w)
    return f


def load(f, tempo=1.0):
    af = f"aresample={SR}" + (f",atempo={tempo:.4f}" if tempo > 1.001 else "")
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(f), "-af", af, "-f", "f32le", "-ac", "1", "-"],
                       capture_output=True, check=True).stdout
    return np.frombuffer(r, np.float32)


def main():
    from piper import PiperVoice
    job = Path(sys.argv[1]).name
    L = int(sys.argv[sys.argv.index("--len") + 1])
    caps = json.loads((ROOT / "captions" / f"{job}_{L}.json").read_text(encoding="utf-8"))
    out = ROOT / "outputs" / job / str(L)
    total = json.loads((out / "timeline.json").read_text(encoding="utf-8"))["duration"]
    if not VOICE.exists():
        VOICE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "piper.download_voices", VOICE.stem, "--download-dir", str(VOICE.parent)],
                       check=True)
    voice = PiperVoice.load(str(VOICE))
    cache = ROOT / "work" / "_tts" / "lines"
    cache.mkdir(parents=True, exist_ok=True)
    buf = np.zeros(int((total + 1) * SR), np.float32)
    spans, rows = [], []
    for i, c in enumerate(caps):
        f = synth(voice, c["text"], cache)
        with wave.open(str(f)) as w:
            dur = w.getnframes() / w.getframerate()
        room = (caps[i + 1]["start"] if i + 1 < len(caps) else total) - c["start"] - GAP
        tempo = min(TEMPO_MAX, max(1.0, dur / max(0.5, room)))
        y = load(f, tempo)
        a = int(c["start"] * SR)
        n = min(len(y), len(buf) - a)
        buf[a: a + n] += y[:n]
        spans.append([round(c["start"], 3), round(c["start"] + n / SR, 3)])
        rows.append((c["start"], dur, tempo))
    buf = buf[: int(total * SR)]
    wav = out / "voice.wav"
    pcm = (np.clip(buf, -1, 1) * 32767).astype("<i2")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-af", "loudnorm=I=-14:TP=-1.5:LRA=7", "-ar", str(SR),
                    "-ac", "2", "-c:a", "aac", "-b:a", "128k", "-t", f"{total:.3f}", str(out / "voice.m4a")], check=True)
    wav.unlink()
    (out / "voice.json").write_text(json.dumps(spans), encoding="utf-8")
    over = [r for r in rows if r[2] > 1.001]
    print(f"voice: {len(caps)} lines -> outputs/{job}/{L}/voice.m4a; {len(over)} sped up "
          f"(max {max([r[2] for r in rows]):.2f}x); speech {sum(r[1] for r in rows):.0f} s of {total:.0f} s")


if __name__ == "__main__":
    main()
