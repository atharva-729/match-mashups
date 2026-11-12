"""v2 Phase 1: the song + its best 60 s and 120 s windows for one game.

The game curve is the story level (story.py, unchanged). Every L-second window (step 2 s) of every library track is
scored with the sister repo's matcher (resample to 200, z-score, banded DTW +-10); the window's start is moved to a
quiet beat within 2 s and its end to the nearest beat (so it stays L +- 0.5 s).
One track per game: the track whose best 120 s window wins, then that track's best 60 s window.
A job can pin its track (`song_fixed: true` + `song_file`): then only the windows are searched (in that file).
Writes the choice into job.yaml (song_file, song_track, song_start_60/120, song_end_60/120, song_credit_full) and
logs the top 3 tracks with their DTW costs so the choice can be overridden by editing job.yaml.
usage: python scripts/pick_song.py inputs/<job> [--force]
"""
import re, sys

import library
import story
from common import ROOT, job_from_argv, log, stamp

LENGTHS = (60, 120)
BLOCK = "# --- v2 song (written by scripts/pick_song.py; edit to override) ---"


def write_keys(job, keys):
    f = job.inp / "job.yaml"
    txt = f.read_text(encoding="utf-8")
    if BLOCK in txt:
        txt = txt[: txt.index(BLOCK)].rstrip() + "\n"
    lines = [BLOCK] + [f"{k}: {v!r}" if isinstance(v, str) else f"{k}: {v}" for k, v in keys.items()]
    f.write_text(txt.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def main():
    job = job_from_argv()
    cfg = job.cfg
    if all(f"song_start_{L}" in cfg for L in LENGTHS) and "--force" not in sys.argv:
        print("song already chosen (job.yaml); --force to search again")
        return
    _, level, _ = story.story(job.gid, job.league)
    lines = [f"### {stamp()} · Song ({job.name})"]
    if cfg.get("song_fixed"):
        tid = cfg["song_track"]
        t = library.track(tid)
        path = ROOT / cfg["song_file"]
        win = {L: library.window_in_file(level, path, L) for L in LENGTHS}
        lines.append(f"- Track fixed in job.yaml: {t['title']} (`{cfg['song_file']}`)")
        for L in LENGTHS:
            w = win[L]
            lines.append(f"- {L} s window: {w['start']:.1f}–{w['end']:.1f} s ({w['end'] - w['start']:.1f} s) · DTW {w['cost']:.4f}")
    else:
        top = library.windows(level, 120, k=3)
        lines.append("- Top 3 tracks by their best 120 s window (sister DTW cost, lower = closer):")
        for i, w in enumerate(top, 1):
            lines.append(f"  {i}. {w['title']} (`{w['track']}`) · {w['start']:.1f}–{w['end']:.1f} s · DTW {w['cost']:.4f}")
        tid = top[0]["track"]
        t = library.track(tid)
        path = library.ensure_audio(t)
        win = {120: top[0], 60: library.windows(level, 60, k=1, only=[tid])[0]}
        lines.append(f"- Chosen: {t['title']}; 60 s window {win[60]['start']:.1f}–{win[60]['end']:.1f} s · DTW {win[60]['cost']:.4f}")
    keys = {} if cfg.get("song_fixed") else {
        "song_track": tid, "song_file": path.resolve().relative_to(ROOT).as_posix()
        if path.resolve().is_relative_to(ROOT) else str(path)}
    for L in LENGTHS:
        keys[f"song_start_{L}"], keys[f"song_end_{L}"] = float(win[L]["start"]), float(win[L]["end"])
        keys[f"song_dtw_{L}"] = round(float(win[L]["cost"]), 4)
    keys["song_credit_full"] = t["credit"]
    write_keys(job, keys)
    lines.append(f"- Credit: {t['credit']}")
    log("\n".join(lines))


if __name__ == "__main__":
    sys.argv += [] if "--len" in sys.argv else ["--len", "120"]   # log to progress_v2.md
    main()
