"""Build the job's footage from a full-game replay instead of a highlight reel (NFL).

  1. Locate: OCR the score bug of a low-resolution copy (inputs/<job>/replay/full_360p.mp4) at every keyframe (~5 s) and turn each
     reading into a play (the same NFLParser as Phase 3). Each play gets its snap time in the replay: the first reading
     at or below its snap clock, minus the seconds the clock has already run.
  2. Select: every scoring play, turnover and big win-probability swing, then the next most important plays until the
     selection holds SELECT_FACTOR x the song's length of footage.
  3. Fetch: only those sections, at 720p, 6 at a time, stream-copied (yt-dlp --download-sections).
  4. Join them without re-encoding, in game order, into inputs/<job>/part1.mp4 (a highlight reel made from the full game); the old reel is
     kept as reel_highlights.mp4. inputs/<job>/replay/reel.json lists where every section came from.
Phases 2-5 then run on part1.mp4 unchanged.

job.yaml:  replay_url: <YouTube link>     replay_box_360: [x, y, w, h]   (the bug in the 640x360 copy)
usage: python scripts/fetch_replay.py inputs/<job> [--locate-only]
"""
import json, os, subprocess, sys, time

import cv2
import numpy as np

import sister
import story
from common import job_from_argv, log, probe_duration, stamp
from tag_shots import NFLParser, bar_present, ocr_frames, strip_split

SELECT_FACTOR = 3.5
PRE, POST, POST_KEY = 6.0, 12.0, 24.0     # seconds of replay around the snap (key plays keep their replays)
FORMAT_HQ = "136"                          # 1280x720 H.264, video only (the mashup's audio is the song)


def locate(job):
    rdir = job.inp / "replay"
    cache = job.wpath("replay_ocr.json")
    if cache.exists():
        return json.loads(cache.read_text())
    src = rdir / "full_360p.mp4"
    if not src.exists():                     # the low-resolution copy for locating plays (360p: the bug is legible)
        rdir.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "-f", "134", "-o", str(src),
                        job.cfg["replay_url"]], check=True)
    x, y, w, h = job.cfg["replay_box_360"]
    cdir = job.wpath("replay_crops")
    times_f = job.wpath("replay_keyframes.txt")
    if not (cdir.exists() and any(cdir.iterdir()) and times_f.exists()):
        # keyframes only (every ~5 s in a YouTube encode): decoding just those is ~10x faster than decoding every
        # frame of a 2+ hour video, and a play (~40 s apart) is still seen several times; the 720p pass pins it down
        cdir.mkdir(exist_ok=True)
        r = subprocess.run(["ffprobe", "-v", "error", "-skip_frame", "nokey", "-select_streams", "v", "-show_entries",
                            "frame=pts_time", "-of", "csv=p=0", str(src)], capture_output=True, text=True, check=True)
        times_f.write_text(r.stdout)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-skip_frame", "nokey", "-i", str(src), "-vf",
                        f"crop={w}:{h}:{x}:{y}", "-fps_mode", "passthrough", str(cdir / "%05d.png")], check=True)
    times = [float(v) for v in times_f.read_text().split()]
    crops = sorted(cdir.glob("*.png"))
    assert len(crops) == len(times), f"{len(crops)} keyframe crops vs {len(times)} keyframe times"
    present = bar_present([cv2.imread(str(f)) for f in crops], "NFL")
    t0 = time.time()
    toks, n = ocr_frames([str(f) for f in crops], present, "NFL", 20, strip_split(crops, "NFL"))
    out = {"sampling": "keyframes", "seconds": round(time.time() - t0), "ocr_calls": n,
           "frames": [{"t": round(times[k], 2), "tokens": toks[k]} for k in range(len(crops))]}
    cache.write_text(json.dumps(out))
    return out


def snap_times(job, ocr):
    """play row p -> estimated snap time in the replay (seconds).
    Every bug reading becomes a game time g = elapsed game seconds (quarter + clock; a reading without a legible
    quarter takes the last one seen within 2 minutes). Readings that disagree with the replay running forward are
    dropped (max-weight non-decreasing subsequence). A play's snap is when the game clock first runs past its snap
    clock: the first reading with g > g_snap, minus the game seconds already run."""
    from tag_shots import lis_keep
    parser = NFLParser(job.gid)
    sb = parser.sb
    rd, last_q = [], (None, -1e9)
    for f in ocr["frames"]:
        r = parser.raw(f["tokens"])
        if not r:
            continue
        q = r["q"]
        if q is None and f["t"] - last_q[1] <= 120:
            q = last_q[0]
        if q is None or q > 4:
            continue
        if r["q"] is not None:
            last_q = (r["q"], f["t"])
        rd.append((f["t"], (q - 1) * 900 + (900 - r["c"])))
    keep = lis_keep([g for _, g in rd], [1.0] * len(rd))
    rd = [x for k, x in enumerate(rd) if k in keep]
    ts, gs = np.array([t for t, _ in rd]), np.array([g for _, g in rd])
    snaps, wins = {}, {}
    for p in range(len(sb["c"]) - 1):
        q = sb["q"][p]
        if q is None or q > 4:
            continue
        gp = (q - 1) * 900 + (900 - sb["c"][p])
        k = int(np.searchsorted(gs, gp + 0.5, side="left"))     # first reading with the clock past the snap clock
        if k >= len(gs) or gs[k] - gp > 120:                     # nothing read near this play
            continue
        t_after = float(ts[k])
        # the clock may have stopped since the snap (incompletion, turnover, score), so the snap lies between the
        # last reading at/above its snap clock and t_after; the estimate assumes the clock ran all the way
        t_before = float(ts[k - 1]) if k > 0 and gp - gs[k - 1] <= 120 else None
        est = t_after - (gs[k] - gp)
        lo = max(t_before, est) if t_before is not None else est
        snaps[p] = lo
        wins[p] = (t_before if t_before is not None else est, t_after)   # the snap lies in between
    return snaps, wins


def select(job, snaps, song_len):
    imp = story.importance(job.gid, "NFL")
    ev = {e["step"]: e for e in sister.events(job.gid)}
    wp = np.asarray(sister.curve(job.gid)["wp"], float)
    key = []
    for p in snaps:                                  # play row p -> its result lands on step p + 1
        e = ev.get(p + 1, {})
        if e.get("td") or e.get("turnover") or e.get("fg") or abs(wp[p + 1] - wp[p]) >= story.SWING_FULL["NFL"]:
            key.append(p)
    chosen = set(key)
    total = lambda: sum(PRE + (POST_KEY if p in key else POST) for p in chosen)
    rest = sorted((p for p in snaps if p not in chosen and ev.get(p + 1, {}).get("play_type") in ("pass", "run")),
                  key=lambda p: -imp[p + 1])
    for p in rest:
        if total() >= SELECT_FACTOR * song_len:
            break
        chosen.add(p)
    return sorted(chosen), set(key)


def sections(snaps, wins, chosen, key, dur):
    """[(a, b, [plays])] in replay time, merging overlaps. A key play's section spans its whole bracket (the 720p
    pass finds the play inside it); an ordinary play's starts at most 15 s before its bracket ends."""
    out = []
    for p in chosen:
        lo, hi = wins[p]
        if p not in key:
            lo = max(lo, hi - 15)
        a, b = max(0.0, min(lo, snaps[p]) - 3), min(dur, max(hi, snaps[p]) + (POST_KEY if p in key else POST))
        nxt = [snaps[q] for q in snaps if snaps[q] > snaps[p]]
        if p not in key and nxt:                     # an ordinary play stops before the next snap's run-up
            b = min(b, max(snaps[p] + 6, min(nxt) - 2))
        if out and a <= out[-1][1] + 1.0:
            out[-1] = (out[-1][0], max(out[-1][1], b), out[-1][2] + [p])
        else:
            out.append((a, b, [p]))
    return out


def fetch(job, secs):
    """Download the sections in parallel (6 at a time), stream-copied: a section starts at the keyframe at or before
    its requested start (a second or few of extra run-up) instead of being re-encoded to cut exactly."""
    from concurrent.futures import ThreadPoolExecutor
    rdir = job.inp / "replay" / "sections"
    rdir.mkdir(parents=True, exist_ok=True)
    files = [rdir / f"s{k:03d}_{int(a)}_{int(b)}.mp4" for k, (a, b, _) in enumerate(secs)]

    def get(k):
        a, b, _ = secs[k]
        if not files[k].exists():
            subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "-f", FORMAT_HQ,
                            "--download-sections", f"*{a:.2f}-{b:.2f}", "-o", str(files[k]), job.cfg["replay_url"]],
                           check=True)
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(get, range(len(secs))))
    return files


def join(job, files, secs):
    """Concatenate the sections without re-encoding (they share one encode: same codec, size and frame rate)."""
    tmp = job.wpath("reel_parts"); tmp.mkdir(exist_ok=True)
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{f.resolve().as_posix()}'\n" for f in files), encoding="utf-8")
    old = job.inp / "part1.mp4"
    keep = job.inp / "reel_highlights.mp4"
    if old.exists() and not keep.exists():
        old.rename(keep)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-an", "-c", "copy",
                    str(job.inp / "part1.mp4")], check=True)
    manifest, at = [], 0.0
    for f, (a, b, plays) in zip(files, secs):
        d = probe_duration(f)
        manifest.append({"reel_in": round(at, 3), "reel_out": round(at + d, 3), "replay_in": round(a, 2),
                         "replay_out": round(b, 2), "plays": plays, "file": f.name})
        at += d
    (job.inp / "replay" / "reel.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return at


def main():
    job = job_from_argv()
    t0 = time.time()
    ocr = locate(job)
    snaps, wins = snap_times(job, ocr)
    al = job.load("alignment.json")
    song_len = al["song_len"]
    n_plays = len(sister.nfl_plays(job.gid))
    chosen, key = select(job, snaps, song_len)
    dur = probe_duration(job.inp / "replay" / "full_360p.mp4")
    secs = sections(snaps, wins, chosen, key, dur)
    lines = [f"### {stamp()} · Replay ({job.name}): located {len(snaps)}/{n_plays} plays in the full replay",
             f"- Bug OCR at every keyframe (~5 s): {len(ocr['frames'])} frames, {ocr['ocr_calls']} strip reads, {ocr['seconds']} s",
             f"- Selected {len(chosen)} plays ({len(key)} scoring / turnover / big swing) in {len(secs)} sections, "
             f"{sum(b - a for a, b, _ in secs):.0f} s of replay"]
    if "--locate-only" in sys.argv:
        log("\n".join(lines)); return
    files = fetch(job, secs)
    reel = join(job, files, secs)
    lines.append(f"- Downloaded at 720p and joined into `inputs/{job.name}/part1.mp4` ({reel:.0f} s; the old reel is "
                 f"`reel_highlights.mp4`) in {time.time() - t0:.0f} s total")
    log("\n".join(lines))


if __name__ == "__main__":
    main()
