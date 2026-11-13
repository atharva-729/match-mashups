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
        "story": job.cfg.get("story_line", ""),
        "curve": story_curve(job),
    }
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
                      "story": tl.get("story", ""), "curve": tl.get("curve"), "sport": tl.get("sport", "NFL"),
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
