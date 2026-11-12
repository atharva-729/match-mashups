"""Thin adapter to the music library (v2). The ONLY module that reads the music library; nothing here writes to it.

Two sources:
  demo    the sister repo's demo recordings (site/demo/*.mp3, classicals.de), via sister.py
  engine  Soundtrack Engine's library (596 Kevin MacLeod tracks, CC BY 4.0), if SOUNDTRACK_REPO is set in config.yaml:
          library/tracks.csv + library/curves.npz (one energy value per second). Its single-act window search
          (engine/stitch.py `candidates`: correlation pre-filter over every window, then banded DTW) is imported, not
          rewritten. The final ranking re-scores the shortlist with the sister repo's matcher (same DTW).
Only tracks with a recorded licence AND credit line are usable.

  tracks()                       -> [{id, title, artist, licence, credit, kind, ...}]
  windows(level, L, k, only)     -> best L-second windows of the library for a game curve, ranked by DTW cost
  window_in_file(level, path, L) -> best L-second window of one audio file
  ensure_audio(track)            -> local path of a track's audio (an engine track is downloaded once from incompetech)
usage: python scripts/library.py            (lists the tracks with credits; Checkpoint 0)
"""
import csv, functools, json, subprocess, sys, urllib.request
from pathlib import Path

import numpy as np

import sister
from common import ROOT

CACHE = ROOT / "work" / "_library"
LOCAL = {"cancan": ("Can-Can (Galop infernal)", "Jacques Offenbach", "*Can-Can*.mp3"),
         "tetris": ("Korobeiniki (Tetris theme), arranged for strings", "Traditional", "*Tetris*.mp3"),
         "winter": ("The Four Seasons: Winter, I. Allegro non molto", "Antonio Vivaldi", "*Vivaldi*.mp3"),
         "william_tell": ("William Tell Overture: Finale, arranged for strings", "Gioachino Rossini", "*William Tell*.mp3")}
STEP = 2            # seconds between candidate window starts
SNAP_START = 2.0    # a window's start moves to the quietest nearby beat within this many seconds
SNAP_END = 0.5      # its end to the nearest beat within this (so the length stays L +- 0.5 s)


@functools.lru_cache(None)
def _engine():
    repo = sister.CFG.get("SOUNDTRACK_REPO")
    if not repo:
        return None
    root = (ROOT / Path(repo).expanduser()).resolve()
    if not (root / "library" / "tracks.csv").exists():
        return None
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from engine import energy, match, stitch          # Soundtrack Engine's own code
    return {"root": root, "energy": energy, "match": match, "stitch": stitch}


@functools.lru_cache(None)
def tracks():
    out = []
    for d in json.loads((sister.MATCH_REPO / "site/demo/demos.json").read_text(encoding="utf-8")):
        if not d.get("licence"):
            continue
        out.append({"id": "demo:" + d["key"], "kind": "demo", "title": d["piece"].split(":")[-1].strip()
                    if "Mountain King" in d["piece"] else d["piece"].split(",")[0], "artist": d["composer"],
                    "performer": d["performer"], "licence": d["licence"], "excerpt": d.get("excerpt"),
                    "credit": f"{d['piece']} ({d['composer']}), {d['performer']}, {d['licence']}, via www.classicals.de",
                    "file": str(sister.MATCH_REPO / "site" / d["file"])})
    # local recordings Atharva downloaded from classicals.de (inputs/_songs/, gitignored): CC BY-NC 4.0, internal use
    for key, (title, artist, pattern) in LOCAL.items():
        hits = sorted((ROOT / "inputs" / "_songs").glob(pattern))
        if hits:
            out.append({"id": "local:" + key, "kind": "demo", "title": title, "artist": artist, "performer": "classicals.de",
                        "licence": "CC BY-NC 4.0", "credit": f"{title} ({artist}), via www.classicals.de, CC BY-NC 4.0",
                        "file": str(hits[0])})
    eng = _engine()
    if eng:
        with open(eng["root"] / "library/tracks.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["licence"] and r["credit"]:
                    for d in out:      # a Kevin MacLeod demo recording: his full CC BY 4.0 line from tracks.csv
                        if d["kind"] == "demo" and d["performer"] == "Kevin MacLeod" and d["title"] == r["title"]:
                            d["credit"] = r["credit"] + ", via www.classicals.de"
                    out.append({"id": r["id"], "kind": "engine", "title": r["title"], "artist": r["artist"],
                                "performer": r["artist"], "licence": r["licence"], "credit": r["credit"],
                                "audio_url": r["audio_url"], "genre": r["genre"], "feel": r["feel"]})
    return out


def track(tid):
    return next(t for t in tracks() if t["id"] == tid)


def ensure_audio(t):
    if t["kind"] == "demo":
        return Path(t["file"])
    p = CACHE / "audio" / f"{t['id']}.mp3"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(t["audio_url"], headers={"User-Agent": "Mozilla/5.0"})
        p.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    return p


def _file_curve(path):
    """Per-second energy (Soundtrack Engine's energy code, so it compares with its library) + beats, cached."""
    eng = _engine()
    path = Path(path)
    key = CACHE / "curves" / (path.stem + f"_{int(path.stat().st_size)}.json")
    if key.exists():
        return json.loads(key.read_text())
    key.parent.mkdir(parents=True, exist_ok=True)
    y = eng["energy"].load_audio(path)
    t0, ps = eng["energy"].per_second(*eng["energy"].energy_from_signal(y))
    _, beats, _ = eng["energy"].rhythm(y)
    out = {"t0": t0, "curve": [float(v) for v in ps], "beats": [float(b) for b in beats],
           "duration": len(y) / eng["energy"].SR}
    key.write_text(json.dumps(out))
    return out


def _snap(start, L, beats, raw, t0, dur):
    """Start on the quietest beat within +-SNAP_START (avoid starting inside a loud passage), end on the beat nearest
    start + L within +-SNAP_END; clamp to the file."""
    beats = np.asarray(beats, float)
    c = beats[np.abs(beats - start) <= SNAP_START]
    if len(c):
        e = np.interp(c, t0 + np.arange(len(raw)), raw)
        score = np.abs(c - start) / 4.0 + (e - e.min()) / max(1e-6, np.ptp(e) or 1)
        start = float(c[np.argmin(score)])
    start = float(np.clip(start, 0, max(0.0, dur - L)))
    end = start + L
    c = beats[np.abs(beats - end) <= SNAP_END]
    if len(c):
        end = float(c[np.argmin(np.abs(c - end))])
    return round(start, 3), round(min(end, dur), 3)


def _score(level, raw, t0, L):
    """Every L-second window of a per-second curve (step STEP), scored with the sister matcher."""
    q = sister.normalise(list(level))
    W = int(round(L))
    out = []
    for s in range(0, max(1, len(raw) - W + 1), STEP):
        w = raw[s: s + W]
        if len(w) < W - 1 or np.std(w) < 1e-6:
            continue
        d, _ = sister.dtw_path(q, sister.normalise(list(w)))
        out.append((float(d), t0 + s))
    return out


def window_in_file(level, path, L):
    """Best L-second window of one file -> {start, end, cost}."""
    c = _file_curve(path)
    raw = np.asarray(c["curve"])
    if c["duration"] <= L + 0.5:
        return {"start": 0.0, "end": round(c["duration"], 3), "cost": _score(level, raw, c["t0"], len(raw))[0][0]}
    d, s = min(_score(level, raw, c["t0"], L))
    a, b = _snap(s, L, c["beats"], raw, c["t0"], c["duration"])
    return {"start": a, "end": b, "cost": d}


def windows(level, L, k=3, only=None):
    """Best L-second window per track over the whole library, top k tracks by sister-DTW cost.
    -> [{track, title, start, end, cost}]"""
    cands = []
    for t in tracks():
        if only and t["id"] not in only:
            continue
        if t["kind"] == "demo":
            c = _file_curve(t["file"])
            raw = np.asarray(c["curve"])
            if len(raw) >= L:
                d, s = min(_score(level, raw, c["t0"], L))
                cands.append((d, t["id"], s))
    eng = _engine()
    if eng and only:          # named tracks: every window of each, directly (no library-wide shortlist)
        lib = eng["match"].library()
        for tid in only:
            if tid in lib["ids"]:
                ti = lib["ids"].index(tid)
                raw, t0 = np.asarray(lib["raw"][ti]), float(lib["t0"][ti])
                if len(raw) >= L:
                    d, s = min(_score(level, raw, t0, L))
                    cands.append((d, tid, s))
    elif eng:
        lib = eng["match"].library()
        top = eng["stitch"].candidates(np.asarray(level, float), int(round(L)), step=STEP, per_track=2, topw=60)
        q = sister.normalise(list(level))
        for c in top:
            if only and c["track"] not in only:
                continue
            ti = c["ti"]; raw = lib["raw"][ti]; t0 = float(lib["t0"][ti])
            s = int(round(c["start"] - t0))
            d, _ = sister.dtw_path(q, sister.normalise(list(raw[s: s + int(round(L))])))
            cands.append((float(d), c["track"], c["start"]))
    best = {}
    for d, tid, s in sorted(cands):
        best.setdefault(tid, (d, s))
    out = []
    for tid, (d, s) in sorted(best.items(), key=lambda x: x[1][0])[:k]:
        t = track(tid)
        if t["kind"] == "demo":
            c = _file_curve(t["file"]); raw, t0, beats, dur = np.asarray(c["curve"]), c["t0"], c["beats"], c["duration"]
        else:
            lib = eng["match"].library(); ti = lib["ids"].index(tid)
            raw, t0, beats, dur = lib["raw"][ti], float(lib["t0"][ti]), lib["beats"][ti], float(lib["duration"][ti])
        a, b = _snap(s, L, beats, raw, t0, dur)
        out.append({"track": tid, "title": t["title"], "start": a, "end": b, "cost": round(d, 4)})
    return out


if __name__ == "__main__":
    ts = tracks()
    print(f"{len(ts)} usable tracks ({sum(t['kind'] == 'demo' for t in ts)} sister demos, "
          f"{sum(t['kind'] == 'engine' for t in ts)} Soundtrack Engine)")
    for t in ts[:8] + ts[-3:]:
        print(f"  {t['id']:<34} {t['licence']:<22} {t['credit'][:110]}")
