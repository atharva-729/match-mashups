"""Song search for a story cut with excitement pacing: one song per length, matched to the cut's own excitement curve.

The curve is excitement.cut_curve over the cut (each beat a spike of its surprise, fading to the suspense on screen),
on the beats' natural footage pacing (work/<job>/<L>/story_pace.json, written by plan_edit.py: run align + plan once
first). Every window of every library track is scored with the sister matcher (library.windows: normalise + banded
DTW), as pick_song.py does for the whole game. Writes song_*_<L> keys into job.yaml (one song per length) and logs the
top tracks, plus the earlier songs' best windows against the same curve for comparison.
usage: python scripts/pick_song_excite.py inputs/<job>
"""
import json, sys

import numpy as np

import excitement
import library
import sister
from common import ROOT, Job, log, stamp

LENGTHS = (60, 120)
BLOCK = "# --- excitement song per length (written by scripts/pick_song_excite.py; edit to override) ---"
COMPARE = ("demo:mountain-king", "km_danse_macabre")
# the matcher compares shape only (when the energy rises and falls), not mood: a calm waltz can have the right shape.
# A game like this needs a dramatic track, so library tracks must carry one of these feel tags (Soundtrack Engine's
# tracks.csv); the classical recordings (kind "demo") are all dramatic pieces and stay eligible.
MOOD = {"Intense", "Epic", "Action", "Aggressive", "Driving", "Suspenseful"}


def dramatic(tid):
    t = library.track(tid)
    return t["kind"] == "demo" or bool(MOOD & {f.strip() for f in t.get("feel", "").split(",")})


def cut_level(job, L):
    pace = json.loads((job.work / str(L) / "story_pace.json").read_text(encoding="utf-8"))
    by = {b["step"]: b for b in job.story["cuts"][L]}
    beats = [by.get(s, {"step": s}) for s in pace["steps"]]
    _, cc = excitement.cut_curve(job.gid, beats, pace["e0"], L)
    return np.asarray(sister.resample(np.asarray(cc, float), sister.N), float)


def main():
    name = sys.argv[1]
    keys, lines = {}, [f"### {stamp()} · Excitement song search ({Job(name).name})"]
    for L in LENGTHS:
        job = Job(name, L)
        level = cut_level(job, L)
        ranked = library.windows(level, L, k=60)
        top = [w for w in ranked if dramatic(w["track"])][:8]
        skipped = [w for w in ranked[:8] if not dramatic(w["track"])]
        lines.append(f"- {L} s: top dramatic tracks by their best window vs the cut's excitement curve (sister DTW, lower = closer):")
        for i, w in enumerate(top, 1):
            lines.append(f"  {i}. {w['title']} (`{w['track']}`) · {w['start']:.1f}–{w['end']:.1f} s · DTW {w['cost']:.4f}")
        if skipped:
            lines.append("  (closer shape but not dramatic, skipped: " + "; ".join(
                f"{w['title']} {w['cost']:.4f} [{library.track(w['track']).get('feel', '')}]" for w in skipped) + ")")
        for tid in COMPARE:
            try:
                c = library.windows(level, L, k=1, only=[tid])[0]
                lines.append(f"  (for comparison: {c['title']} best window {c['start']:.1f}–{c['end']:.1f} s · DTW {c['cost']:.4f})")
            except (IndexError, StopIteration, ValueError):
                pass
        w = top[0]
        t = library.track(w["track"])
        path = library.ensure_audio(t).resolve()
        rel = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        keys.update({f"song_track_{L}": w["track"], f"song_file_{L}": rel, f"song_title_{L}": t["title"],
                     f"song_artist_{L}": t["artist"], f"song_start_{L}": float(w["start"]), f"song_end_{L}": float(w["end"]),
                     f"song_dtw_{L}": float(w["cost"]), f"song_credit_full_{L}": t["credit"]})
        lines.append(f"- {L} s chosen: {t['title']} ({t['artist']}), {w['start']:.1f}–{w['end']:.1f} s")
    f = ROOT / "inputs" / Job(name).name / "job.yaml"
    txt = f.read_text(encoding="utf-8")
    if BLOCK in txt:
        txt = txt[: txt.index(BLOCK)].rstrip() + "\n"
    body = [BLOCK] + [f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in keys.items()]
    f.write_text(txt.rstrip() + "\n" + "\n".join(body) + "\n", encoding="utf-8")
    log("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    sys.argv += [] if "--len" in sys.argv else ["--len", "120"]   # log to progress_v2.md
    main()
