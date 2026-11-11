"""Build the job's footage from a full-game replay instead of a highlight reel (NFL).

  1. Locate: OCR the score bug of a low-resolution copy (inputs/<job>/replay/full_360p.mp4) every 2 s and turn each
     reading into a play (the same NFLParser as Phase 3). Each play gets its snap time in the replay: the first reading
     at or below its snap clock, minus the seconds the clock has already run.
  2. Select: every scoring play, turnover and big win-probability swing, then the next most important plays until the
     selection holds SELECT_FACTOR x the song's length of footage.
  3. Fetch: only those sections, at 720p, with yt-dlp --download-sections (inputs/<job>/replay/sections/).
  4. Join them, in game order, into inputs/<job>/part1.mp4 (a highlight reel made from the full game); the old reel is
     kept as reel_highlights.mp4. inputs/<job>/replay/reel.json lists where every section came from.
Phases 2-5 then run on part1.mp4 unchanged.

job.yaml:  replay_url: <YouTube link>     replay_box_360: [x, y, w, h]   (the bug in the 640x360 copy)
usage: python scripts/fetch_replay.py inputs/<job> [--locate-only]
"""
import json, multiprocessing as mp, os, subprocess, sys, time

import cv2
import numpy as np

import sister
import story
from common import job_from_argv, log, probe_duration, stamp
from tag_shots import NFLParser, _ocr_chunk, _worker_init, bar_present

FPS_LOC = 0.5
SELECT_FACTOR = 3.5
PRE, POST, POST_KEY = 6.0, 12.0, 24.0     # seconds of replay around the snap (key plays keep their replays)
FORMAT_HQ = "136"                          # 1280x720 H.264, video only (the mashup's audio is the song)


def locate(job):
    rdir = job.inp / "replay"
    cache = job.wpath("replay_ocr.json")
    if cache.exists():
        return json.loads(cache.read_text())
    src = rdir / "full_360p.mp4"
    x, y, w, h = job.cfg["replay_box_360"]
    cdir = job.wpath("replay_crops")
    if not (cdir.exists() and any(cdir.iterdir())):
        cdir.mkdir(exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", f"fps={FPS_LOC},crop={w}:{h}:{x}:{y}",
                        str(cdir / "%05d.png")], check=True)
    crops = sorted(cdir.glob("*.png"))
    present = bar_present([cv2.imread(str(f)) for f in crops], "NFL")
    nw = max(1, (os.cpu_count() or 2) - 1)
    bounds = np.linspace(0, len(crops), nw * 6 + 1).astype(int)
    t0 = time.time()
    with mp.get_context("spawn").Pool(nw, initializer=_worker_init) as pool:
        res = pool.map(_ocr_chunk, [([str(f) for f in crops[a:b]], present[a:b], False, 0)
                                    for a, b in zip(bounds, bounds[1:]) if b > a], chunksize=1)
    toks = [t for r in res for t in r[0]]
    out = {"fps": FPS_LOC, "seconds": round(time.time() - t0), "ocr_calls": sum(r[1] for r in res),
           "frames": [{"t": round(k / FPS_LOC, 2), "tokens": toks[k]} for k in range(len(crops))]}
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
        wins[p] = (max(t_before, t_after - 75) if t_before is not None else est, t_after)
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
    rdir = job.inp / "replay" / "sections"
    rdir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    files = []
    for k, (a, b, plays) in enumerate(secs):
        f = rdir / f"s{k:03d}_{int(a)}_{int(b)}.mp4"
        if not f.exists():
            subprocess.run([py, "-m", "yt_dlp", "-q", "--no-warnings", "-f", FORMAT_HQ, "--force-keyframes-at-cuts",
                            "--download-sections", f"*{a:.2f}-{b:.2f}", "-o", str(f), job.cfg["replay_url"]], check=True)
        files.append(f)
    return files


def join(job, files, secs):
    """Re-encode every section to one 1280x720 30 fps stream and concatenate (frame-exact cut points)."""
    tmp = job.wpath("reel_parts"); tmp.mkdir(exist_ok=True)
    norm = []
    for f in files:
        g = tmp / f.name
        if not g.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(f), "-an", "-vf", "scale=1280:720,fps=30000/1001",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(g)], check=True)
        norm.append(g)
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{g.as_posix()}'\n" for g in norm), encoding="utf-8")
    old = job.inp / "part1.mp4"
    keep = job.inp / "reel_highlights.mp4"
    if old.exists() and not keep.exists():
        old.rename(keep)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    str(job.inp / "part1.mp4")], check=True)
    manifest, at = [], 0.0
    for g, (a, b, plays) in zip(norm, secs):
        d = probe_duration(g)
        manifest.append({"reel_in": round(at, 3), "reel_out": round(at + d, 3), "replay_in": round(a, 2),
                         "replay_out": round(b, 2), "plays": plays})
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
             f"- Bug OCR every {1 / FPS_LOC:.0f} s: {len(ocr['frames'])} frames, {ocr['ocr_calls']} OCR calls, {ocr['seconds']} s",
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
