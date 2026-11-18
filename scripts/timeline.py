"""v2 Phase 5: what the player needs to draw over the video, as lookups from the EDL.

outputs/<job>/<L>/timeline.json
  samples   10 per second: [step, home win probability, tension (smoothed, 0-1)]
  teams     home/away abbreviation, name, colour;  title, final, season/week, song credit, YouTube link
  bug       the broadcast scoreboard box in video pixels (overlays keep clear of it)
  captions  copied from captions/<job>_<L>.json (if written)
outputs/gallery.json: one card per job with v2 outputs (matchup, final, song, link, available lengths).
Nothing here is committed: it holds the YouTube links (job.yaml is private).
usage: python scripts/timeline.py inputs/<job> --len 60|120      python scripts/timeline.py --gallery
"""
import json, sys

import numpy as np

import library
import sister
from common import ROOT, Job, job_from_argv, log, stamp

RATE = 10
# primary colours (a lighter alternate where the primary is too dark to read on a dark UI)
NFL_COLOURS = {
    "ARI": "#97233F", "ATL": "#A71930", "BAL": "#6A4FB0", "BUF": "#2F6FD6", "CAR": "#0085CA", "CHI": "#E05A1B",
    "CIN": "#FB4F14", "CLE": "#FF3C00", "DAL": "#4A78C2", "DEN": "#FB4F14", "DET": "#0076B6", "GB": "#2E7D4F",
    "HOU": "#C8102E", "IND": "#2C6FD1", "JAX": "#006778", "KC": "#E31837", "LA": "#2F5FD0", "STL": "#B3995D",
    "LAC": "#0080C6", "SD": "#0080C6", "LV": "#A5ACAF", "OAK": "#A5ACAF", "MIA": "#008E97", "MIN": "#7A4FC0",
    "NE": "#C60C30", "NO": "#D3BC8D", "NYG": "#3157C6", "NYJ": "#2E8B57", "PHI": "#1F8A86", "PIT": "#FFB612",
    "SEA": "#69BE28", "SF": "#C9243F", "TB": "#D50A0A", "TEN": "#4B92DB", "WAS": "#B5374A",
    "GT": "#C9A64A", "LSG": "#3BB4E6",   # IPL
}


def source_label(job):
    """Credit for the footage: the full broadcast, or the highlights video it was cut from."""
    if job.cfg.get("replay_url"):
        return "Original game on YouTube"
    u = job.cfg.get("highlights_url", "")
    return "Highlights on iplt20.com" if "iplt20.com" in u else "Highlights on YouTube" if u else ""


def story_curve(job, n=80):
    """The game's story curve (what the song was matched to), n points, smoothed, 0-1: the gallery's sparkline."""
    if job.league == "NASCAR":
        import nascar
        lv = np.asarray(nascar.race(job)["level"], float)
    else:
        import story
        lv = np.asarray(story.story(job.gid, job.league)[1], float)
    lv = np.convolve(np.pad(lv, (4, 4), mode="edge"), np.ones(9) / 9, mode="same")[4:-4]
    v = np.interp(np.linspace(0, len(lv) - 1, n), np.arange(len(lv)), lv)
    v = (v - v.min()) / max(1e-6, np.ptp(v))
    return [round(float(x), 3) for x in v]


def match_curves(job):
    """The song <-> game match, as the sister page draws it: the song window's energy curve, the game's curve (story
    curve, or tension in page mode), the DTW path between them, and each step's x on the game curve (0-1)."""
    import align as A
    d, _ = A.song_curve(job)
    a = sister.normalise(d["curve"])
    if job.league == "NASCAR":
        import nascar
        r = nascar.race(job)
        level, tau = r["level"], r["tau"]
    elif job.cfg.get("alignment", "page") == "story":
        import story
        tau, level, _ = story.story(job.gid, job.league)
    else:
        full = sister.curve(job.gid)
        level = list(sister.resample(np.asarray(full["tension"], float), A.N))
        tau = list(np.linspace(0, 1, full["n_steps"]))
    b = sister.normalise(list(level))
    _, path = sister.dtw_path(a, b)
    r3 = lambda v: [round(float(x), 3) for x in v]
    mm = lambda v: (np.asarray(v, float) - np.min(v)) / max(1e-9, float(np.ptp(v)))      # 0-1 for drawing
    return {"song": r3(mm(a)), "game": r3(mm(b)), "path": [[int(i), int(j)] for k, (i, j) in enumerate(path) if k % 3 == 0],
            "t0": round(float(d["t0"]), 3), "t1": round(float(d["t1"]), 3), "tau": r3(tau)}


def song_live(job, n):
    """The song stem as it plays, per sample: loudness (RMS in dB, 0-1 between its quiet and loud ends, ~0.5 s
    smoothing) and hits (percussive onset strength, 0-1). The hits drive the glow's pulse; build() draws the meter's
    song line from the matcher's song energy instead of this loudness."""
    import librosa, subprocess
    sr = 22050                          # decode with ffmpeg: librosa >= 1.0 has no audioread fallback for AAC/m4a
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(job.out_l / "song.m4a"), "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    y = np.frombuffer(pcm, dtype=np.float32).copy()
    hop = sr // RATE
    rms = librosa.feature.rms(y=y, frame_length=2 * hop, hop_length=hop)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-5))
    lo, hi = np.percentile(db, 5), np.percentile(db, 98)
    env = np.clip((db - lo) / max(1e-6, hi - lo), 0, 1)
    env = np.convolve(np.pad(env, (2, 2), mode="edge"), np.ones(5) / 5, mode="same")[2:-2]
    on = librosa.onset.onset_strength(y=librosa.effects.percussive(y), sr=sr, hop_length=hop)
    hit = np.clip(on / max(1e-6, np.percentile(on, 97)), 0, 1)
    fit = lambda v: [round(float(x), 3) for x in np.interp(np.arange(n), np.arange(len(v)), v)]
    return fit(env), fit(hit)


CLOCK_LEAD, CLOCK_CLEAR = 3.4, 0.35   # a clock callout: from this long before the play's result to just before it


def clock_callouts(job):
    """"Time is running out" callouts for the story beats marked `clock: "<line>"`: the game clock at the snap of the
    play (sister scoreboard data: quarter + seconds left) as a headline, the story's line under it, and the scoreboard
    at the snap (score, quarter, clock: drawn by the player, so it agrees with the headline; the broadcast's own clock
    in the footage has often run on past the snap). Shown during the
    run-up, cleared just before the result so the play itself lands on the music. -> [{start, end, step, head, sub}]"""
    if not job.story:
        return []
    beats = [b for b in job.story["cuts"].get(job.L, []) if b.get("clock")]
    if not beats:
        return []
    rep = {int(k): v for k, v in job.load("plan_report.json").items()}
    slow = {x["step"]: x for x in job.load("edl.json") if x.get("why") == "slowmo"}
    sb = sister.curve(job.gid)["sb"]
    out, prev_end = [], 0.0
    for b in beats:
        s, e = b["step"], (rep.get(b["step"]) or {}).get("planned")
        if e is None:
            continue
        q, c = sb["q"][s - 1], sb["c"][s - 1]
        when = "OVERTIME" if q >= 5 else ("4TH QUARTER" if q == 4 else f"Q{q}")
        start = max(prev_end + 0.3, e - CLOCK_LEAD)
        if s in slow:                    # with slow motion: exactly over the slowed run-up
            start = max(prev_end + 0.3, slow[s]["song_in"] - 0.25)
        out.append({"start": round(start, 2), "end": round(e - CLOCK_CLEAR, 2), "step": s,
                    "head": f"{c // 60}:{c % 60:02d} LEFT", "when": when, "sub": b["clock"],
                    "sa": int(sb["a"][s - 1]), "sh": int(sb["h"][s - 1]), "q": "OT" if q >= 5 else f"Q{q}",
                    "clk": f"{c // 60}:{c % 60:02d}"})
        prev_end = e - CLOCK_CLEAR
    return out


def game_live(job, n):
    """For an excitement-paced story cut: the game line the music was matched to, i.e. the cut's excitement curve
    (excitement.cut_curve) on the beats' planned result times, 0-1, per sample. The meter's game line and the glow's
    strength use it instead of the whole game's tension. None for other cuts."""
    if not (job.story and job.story.get("pacing") == "excitement"):
        return None
    import excitement
    rep = {int(k): v for k, v in job.load("plan_report.json").items()}
    by = {b["step"]: b for b in job.story["cuts"][job.L]}
    steps = [s for s in by if (rep.get(s) or {}).get("planned") is not None]
    t, cc = excitement.cut_curve(job.gid, [by[s] for s in steps], [rep[s]["planned"] for s in steps], (n - 1) / RATE)
    cc = (cc - cc.min()) / max(1e-9, float(np.ptp(cc)))
    return [round(float(x), 3) for x in np.interp(np.arange(n) / RATE, t, cc)]


def why_song(job):
    """The story's "why this song" line; a story with one song per length keys it by length ({60: ..., 120: ...})."""
    w = (job.story or {}).get("why_song") or job.cfg.get("why_song", "")
    return w.get(job.L, "") if isinstance(w, dict) else w


def build(job):
    al, edl = job.load("alignment.json"), job.load("edl.json")
    if job.league == "NASCAR":          # a race: no win probability; the glow follows the race's tension
        import nascar
        r = nascar.race(job)
        g = {"title": r["meta"]["title"], "final": r["meta"]["final"], "season": job.cfg.get("season"),
             "home": "", "home_name": "", "away": "", "away_name": ""}
        c = {"wp": [0.5] * (r["laps"] + 1), "tension": r["tension"], "week": None}
    else:
        g = sister.game_json(job.gid)
        c = sister.curve(job.gid)
    L = al["song_len"]
    wp = np.asarray(c["wp"], float)
    ten = np.asarray(c["tension"], float)
    ten = ten / max(1e-6, ten.max())
    n = int(np.floor(L * RATE)) + 1
    steps = np.zeros(n, int)
    for k in range(n):
        t = k / RATE
        e = next((e for e in edl if e["song_in"] <= t < e["song_out"]), edl[-1])
        steps[k] = e["step"] if e["step"] is not None else 0
    # the step on screen jumps around (replays, run-ups): the visual follows a running max so it never runs backwards,
    # and the tension is smoothed over ~3 s so nothing flickers
    shown = np.maximum.accumulate(steps)
    k = int(3 * RATE)
    tv = np.convolve(np.pad(ten[shown], (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]
    wv = np.convolve(np.pad(wp[shown], (5, 5), mode="edge"), np.ones(11) / 11, mode="same")[5:-5]
    cap_f = ROOT / "captions" / f"{job.name}_{job.L}.json"
    t = library.track(job.cfg["song_track"])
    out = {
        "job": job.name, "length": job.L, "duration": round(L, 3), "rate": RATE,
        "title": g["title"], "final": g["final"], "season": g["season"], "sport": job.league,
        "week": c.get("week"), "league": job.league,
        "teams": {side: {"abbr": g[side], "name": g[f"{side}_name"], "colour": NFL_COLOURS.get(g[side], "#888888")}
                  for side in ("home", "away")},
        "song": {"title": t["title"], "artist": t["artist"], "credit": job.cfg.get("song_credit_full", t["credit"])},
        "youtube": job.cfg.get("replay_url") or job.cfg.get("highlights_url"),
        "source_label": source_label(job),
        "bug": (json.loads(job.wpath("box.json").read_text())["part1.mp4"]["box"] if job.wpath("box.json").exists()
                else [2 * v for v in job.cfg["replay_box_360"]] if "replay_box_360" in job.cfg else None),
        "video_size": [1280, 720],
        "samples": [[int(steps[i]), round(float(wv[i]), 4), round(float(tv[i]), 4)] for i in range(n)],
        "captions": json.loads(cap_f.read_text(encoding="utf-8")) if cap_f.exists() else [],
        "files": {"video": "video.mp4", "game": "game.m4a", "song": "song.m4a"},
        "story": (job.story or {}).get("line") or job.cfg.get("story_line", ""),
        "story_title": (job.story or {}).get("title", ""),
        "why_song": why_song(job),
        "curve": story_curve(job),
        "match": match_curves(job),
    }
    _, out["song_hit"] = song_live(job, n)
    out["clock"] = clock_callouts(job)
    ge = game_live(job, n)
    if ge is not None:
        out["game_env"] = ge
    # the meter's song line = the song energy the game was matched to (the right panel's top curve), on the cut's
    # clock. The rendered stem's own loudness is flattened by loudnorm (the quiet opening read as loud).
    m = out["match"]
    out["song_env"] = [round(float(x), 3) for x in
                       np.interp(np.arange(n) / RATE, np.linspace(m["t0"], m["t1"], len(m["song"])), m["song"])]
    (job.out_l / "timeline.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def gallery():
    cards = []
    for d in sorted((ROOT / "outputs").iterdir()):
        lens = sorted(int(x.name) for x in d.glob("*") if x.is_dir() and x.name.isdigit()
                      and (x / "timeline.json").exists() and (x / "video.mp4").exists())
        if not lens:
            continue
        tl = json.loads((d / str(lens[-1]) / "timeline.json").read_text(encoding="utf-8"))
        cards.append({"job": d.name, "title": tl["title"], "final": tl["final"], "season": tl["season"],
                      "week": tl["week"], "teams": tl["teams"], "song": tl["song"], "youtube": tl["youtube"],
                      "story": tl.get("story", ""), "story_title": tl.get("story_title", ""), "curve": tl.get("curve"), "sport": tl.get("sport", "NFL"),
                      "source_label": tl.get("source_label") or "Original game on YouTube",
                      "lengths": lens})
    (ROOT / "outputs" / "gallery.json").write_text(json.dumps(cards, ensure_ascii=False, indent=1), encoding="utf-8")
    return cards


if __name__ == "__main__":
    if "--gallery" in sys.argv:
        print(f"gallery: {len(gallery())} games")
    else:
        job = job_from_argv()
        tl = build(job)
        gallery()
        print(f"timeline: {len(tl['samples'])} samples, {len(tl['captions'])} captions -> outputs/{job.name}/{job.L}/timeline.json")
