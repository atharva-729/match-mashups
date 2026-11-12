"""Build the job's footage from a full-game replay instead of a highlight reel (NFL).

  1. Locate: OCR the score bug of a low-resolution copy (inputs/<job>/replay/full_360p.mp4) at every keyframe (~5 s) and turn each
     reading into a play (the same NFLParser as Phase 3). Each play gets its snap time in the replay: the first reading
     at or below its snap clock, minus the seconds the clock has already run.
  2. Select: every scoring play, turnover, big win-probability swing and page moment, then the next most important
     ordinary plays until the selection holds SELECT_FACTOR x the song's length of footage.
  3. Fetch: only those sections, at 720p, 6 at a time, stream-copied (yt-dlp --download-sections).
  4. Join them without re-encoding, in game order, into inputs/<job>/part1.mp4 (a highlight reel made from the full game); the old reel is
     kept as reel_highlights.mp4. inputs/<job>/replay/reel.json lists where every section came from; each section's
     exact start in the replay (`replay_true_in`) is measured by matching its frames against the 360p copy, so the
     broadcast audio (inputs/<job>/replay/full_audio.m4a, downloaded whole: v2 game sound) lines up with the picture.
Phases 2-5 then run on part1.mp4 unchanged.

job.yaml:  replay_url: <YouTube link>     replay_box_360: [x, y, w, h]   (the bug in the 640x360 copy)
           replay_locate: 720   (optional) a bug too small to read at 360p (FOX 2016's stacked box): download the
                                whole 720p copy once, locate on it (box x2) and cut the sections from it locally
usage: python scripts/fetch_replay.py inputs/<job> [--locate-only | --offsets-only]
"""
import json, os, subprocess, sys, time

import cv2
import numpy as np

import sister
import story
from common import job_from_argv, log, probe_duration, stamp
from tag_shots import NFLParser, bar_present, layout, ocr_frames

SELECT_FACTOR = 7.0          # v2: was 3.5, which held only the key plays (no ordinary plays between scores)
PRE, POST, POST_KEY = 6.0, 12.0, 24.0     # seconds of replay around the snap (key plays keep their replays)
FORMAT_HQ = "136"                          # 1280x720 H.264, video only (the audio comes whole, below)
FORMAT_AUDIO = "140"                       # AAC 128k: the broadcast sound of the whole game (commentary + crowd)


def locate(job):
    rdir = job.inp / "replay"
    cache = job.wpath("replay_ocr.json")
    if cache.exists():
        return json.loads(cache.read_text())
    hq = str(job.cfg.get("replay_locate", "")) == "720"
    src = rdir / ("full_720p.mp4" if hq else "full_360p.mp4")
    if not src.exists():                     # the copy for locating plays (360p unless the bug needs 720p)
        rdir.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "-f", FORMAT_HQ if hq else "134",
                        "-o", str(src), job.cfg["replay_url"]], check=True)
    x, y, w, h = [2 * v for v in job.cfg["replay_box_360"]] if hq else job.cfg["replay_box_360"]
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
    times = [float(v.split(",")[0]) for v in times_f.read_text().split() if v.split(",")[0]]   # some files: '0.000000,'
    crops = sorted(cdir.glob("*.png"))
    assert len(crops) == len(times), f"{len(crops)} keyframe crops vs {len(times)} keyframe times"
    present = bar_present([cv2.imread(str(f)) for f in crops], "NFL")
    t0 = time.time()
    toks, n = ocr_frames([str(f) for f in crops], present, "NFL", 20, layout(job, crops))
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
    parser = NFLParser(job.gid, job.cfg.get("bug_style"))
    sb = parser.sb
    rd, last_q = [], (None, -1e9)
    for f in ocr["frames"]:
        r = parser.raw(f["tokens"]) if f["tokens"] else None
        if not r:
            continue
        q = r["q"]
        if q is None and f["t"] - last_q[1] <= 120:
            q = last_q[0]
        if q is None or q > 5:                       # 5 = overtime
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
        if q is None or q > 5:                       # 5 = overtime
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
    moments = {m["step"] for m in sister.game_json(job.gid)["moments"]}
    key = []
    for p in snaps:                                  # play row p -> its result lands on step p + 1
        e = ev.get(p + 1, {})
        if e.get("td") or e.get("turnover") or e.get("fg") or abs(wp[p + 1] - wp[p]) >= story.SWING_FULL["NFL"]                 or p + 1 in moments:                 # v2: every page moment (v1 dropped the blocked FG at Q2 0:01)
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


def keyframes(src, cache):
    if not cache.exists():
        r = subprocess.run(["ffprobe", "-v", "error", "-skip_frame", "nokey", "-select_streams", "v", "-show_entries",
                            "frame=pts_time", "-of", "csv=p=0", str(src)], capture_output=True, text=True, check=True)
        cache.write_text(r.stdout)
    return np.array([float(v.split(",")[0]) for v in cache.read_text().split() if v.split(",")[0]])


def cut_local(job, secs):
    """The whole 720p copy is here (replay_locate: 720): cut each section from it, stream-copied, starting exactly on
    a keyframe at or before its requested start, so its true start in the replay is known exactly."""
    rdir = job.inp / "replay" / "sections"
    rdir.mkdir(parents=True, exist_ok=True)
    src = job.inp / "replay" / "full_720p.mp4"
    kf = keyframes(src, job.wpath("replay_keyframes_720.txt"))
    files, starts = [], []
    for k, (a, b, _) in enumerate(secs):
        a0 = float(kf[max(0, np.searchsorted(kf, a + 1e-3) - 1)])
        f = rdir / f"s{k:03d}_{int(a0)}_{int(b)}.mp4"
        if not f.exists():
            # seek a hair past the keyframe: '-ss 1275.374' is before its real time 1275.3741, and a stream copy then
            # starts one keyframe (up to ~5 s) earlier than intended
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a0 + 0.02:.3f}", "-i", str(src), "-t", f"{b - a0:.3f}",
                            "-an", "-c", "copy", "-avoid_negative_ts", "make_zero", str(f)], check=True)
        files.append(f); starts.append(round(a0, 3))
    return files, starts


def fetch(job, secs):
    """Download the sections in parallel (6 at a time), stream-copied: a section starts at the keyframe at or before
    its requested start (a second or few of extra run-up) instead of being re-encoded to cut exactly."""
    from concurrent.futures import ThreadPoolExecutor
    rdir = job.inp / "replay" / "sections"
    rdir.mkdir(parents=True, exist_ok=True)
    files = [rdir / f"s{k:03d}_{int(a)}_{int(b)}.mp4" for k, (a, b, _) in enumerate(secs)]

    def get(k):
        a, b, _ = secs[k]
        for attempt in range(4):              # YouTube occasionally refuses one of 6 parallel requests: retry
            if files[k].exists():
                return
            r = subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "-f", FORMAT_HQ,
                                "--download-sections", f"*{a:.2f}-{b:.2f}", "-o", str(files[k]), job.cfg["replay_url"]],
                               capture_output=True, text=True)
            if r.returncode == 0:
                return
            time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"section {k} ({a:.0f}-{b:.0f} s) failed 4 times: {r.stderr[-500:]}")
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(get, range(len(secs))))
    return files


def true_start(job, f, a):
    """Where section file f really starts in the replay (its frames matched against the 360p copy near a)."""
    def frames(src, ss, dur, fps):
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, ss):.3f}", "-t", f"{dur:.3f}", "-i", str(src),
                            "-vf", f"fps={fps},scale=64:36,format=gray", "-f", "rawvideo", "-"],
                           capture_output=True, check=True).stdout
        return np.frombuffer(r, np.uint8).reshape(-1, 36, 64).astype(np.float32)
    probe = frames(f, 0.0, 10.0, 2)[2::4]                     # section frames at 1, 3, 5, 7, 9 s
    if len(probe) == 0:
        return a
    lo = max(0.0, a - 8.0)
    ref = frames(job.inp / "replay" / "full_360p.mp4", lo, 30.0, 30000 / 1001)
    dt = 1001 / 30000
    best = (np.inf, 0.0)
    for k0 in range(len(ref)):                                 # candidate start: lo + k0 * dt
        cost, n = 0.0, 0
        for i, fr in enumerate(probe):
            k = k0 + int(round((1.0 + 2.0 * i) / dt))
            if k < len(ref):
                cost += float(np.mean(np.abs(ref[k] - fr))); n += 1
        if n == len(probe) and cost / n < best[0]:
            best = (cost / n, lo + k0 * dt)
    return round(best[1], 3)


def fetch_audio(job):
    """The whole game's broadcast audio (~135 MB for 2.5 h), once: game sound is cut from it with pauses chosen freely,
    not limited to the video sections."""
    f = job.inp / "replay" / "full_audio.m4a"
    if not f.exists():
        subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "-f", FORMAT_AUDIO, "-o", str(f),
                        job.cfg["replay_url"]], check=True)
    return f


def join(job, files, secs, starts=None):
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
                         "replay_out": round(b, 2), "plays": plays,
                         "replay_true_in": starts[len(manifest)] if starts else true_start(job, f, a),
                         "file": f.name})
        at += d
    (job.inp / "replay" / "reel.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return at


def main():
    job = job_from_argv()
    t0 = time.time()
    ocr = locate(job)
    snaps, wins = snap_times(job, ocr)
    al = job.alignment_any()
    song_len = max(120.0, al["song_len"])       # v2: the reel serves the 120 s summary (the 60 s one cuts from it)
    n_plays = len(sister.nfl_plays(job.gid))
    chosen, key = select(job, snaps, song_len)
    dur = probe_duration(job.inp / "replay" / ("full_360p.mp4" if (job.inp / "replay" / "full_360p.mp4").exists()
                                               else "full_720p.mp4"))
    secs = sections(snaps, wins, chosen, key, dur)
    lines = [f"### {stamp()} · Replay ({job.name}): located {len(snaps)}/{n_plays} plays in the full replay",
             f"- Bug OCR at every keyframe (~5 s): {len(ocr['frames'])} frames, {ocr['ocr_calls']} strip reads, {ocr['seconds']} s",
             f"- Selected {len(chosen)} plays ({len(key)} scoring / turnover / big swing) in {len(secs)} sections, "
             f"{sum(b - a for a, b, _ in secs):.0f} s of replay"]
    if "--locate-only" in sys.argv:
        log("\n".join(lines)); return
    if "--offsets-only" in sys.argv:          # measure replay_true_in for an existing reel.json
        f = job.inp / "replay" / "reel.json"
        m = json.loads(f.read_text(encoding="utf-8"))
        for x in m:
            x["replay_true_in"] = true_start(job, job.inp / "replay" / "sections" / x["file"], x["replay_in"])
            print(x["file"], x["replay_in"], "->", x["replay_true_in"], flush=True)
        f.write_text(json.dumps(m, indent=1), encoding="utf-8"); return
    if (job.inp / "replay" / "full_720p.mp4").exists():
        files, starts = cut_local(job, secs)
    else:
        files, starts = fetch(job, secs), None
    fetch_audio(job)
    reel = join(job, files, secs, starts)
    lines.append(f"- Downloaded at 720p and joined into `inputs/{job.name}/part1.mp4` ({reel:.0f} s; the old reel is "
                 f"`reel_highlights.mp4`) in {time.time() - t0:.0f} s total")
    log("\n".join(lines))


if __name__ == "__main__":
    main()
