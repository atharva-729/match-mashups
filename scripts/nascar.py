"""NASCAR adapter (v2): a race's lap-by-lap story curve from NASCAR Race Center's unofficial public feeds.

  https://cf.nascar.com/cacher/<season>/<series>/<race_id>/lap-times.json   every car's lap time + running position
  https://cf.nascar.com/cacher/<season>/<series>/<race_id>/weekend-feed.json   cautions, lead changes, results
Steps = laps (step 0 = the green flag, step L = lap L completed). Per lap:
  tension     how close the lead is: exp(-gap(P1, P2) / 1 s), x0.15 under yellow, rising over the final 20% of the race
  importance  lead change 1.0 (0.4 if it happens under yellow / in a pit cycle), crash caution 0.6, restart 0.5,
              the final lap 1.5 (a close finish is the story of the race)
  story curve story.py's construction: each lap lasts 1 + 3 x importance, Gaussian-smoothed importance + tension
Writes work/<job>/race.json: {laps, wp (P1-P2 gap as a 0-1 'closeness'), tension, importance, events, level, tau, meta}.
usage: python scripts/nascar.py inputs/<job>          (job.yaml: league: NASCAR, season, series, race_id)
"""
import json, urllib.request

import numpy as np

from common import job_from_argv

N = 200
SIGMA = 0.025
STRETCH = 3.0
API = "https://cf.nascar.com/cacher/{season}/{series}/{race}/{feed}.json"


def fetch(job, feed):
    f = job.wpath(f"nascar_{feed}.json")
    if not f.exists():
        url = API.format(season=job.cfg["season"], series=job.cfg.get("series", 1), race=job.cfg["race_id"], feed=feed)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        f.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    return json.loads(f.read_text(encoding="utf-8"))


def build(job):
    lt, wk = fetch(job, "lap-times"), fetch(job, "weekend-feed")["weekend_race"][0]
    n = int(wk["actual_laps"])
    cars = {}
    for c in lt["laps"]:
        t, cum = {}, 0.0
        for l in c["Laps"]:
            if l["Lap"] and l["LapTime"]:
                cum += l["LapTime"]; t[l["Lap"]] = cum
        cars[c["Number"]] = (c["FullName"], t)
    yellow = np.zeros(n + 1, bool)
    for f in lt.get("flags", []):
        if f["LapsCompleted"] <= n and f["FlagState"] in (2, 3):
            yellow[f["LapsCompleted"]] = True
    gap = np.full(n + 1, 5.0)
    for lap in range(1, n + 1):
        ts = sorted(t[lap] for _, t in cars.values() if lap in t)
        if len(ts) >= 2:
            gap[lap] = ts[1] - ts[0]
    closeness = np.exp(-np.clip(gap, 0, 30) / 1.0)
    late = np.clip((np.arange(n + 1) / n - 0.8) / 0.2, 0, 1)
    tension = closeness * np.where(yellow, 0.15, 1.0)
    tension = np.clip(0.75 * tension + 0.25 * late, 0, 1)
    tension = np.convolve(np.pad(tension, (3, 3), mode="edge"), np.ones(7) / 7, mode="same")[3:-3]
    name = {num: nm for num, (nm, _) in cars.items()}
    imp, events = np.zeros(n + 1), {}
    for k, ld in enumerate(wk["race_leaders"]):
        lap = int(ld["start_lap"])
        if k == 0 or lap > n:
            continue
        w = 0.4 if (yellow[max(0, lap - 1)] or yellow[min(n, lap)]) else 1.0
        imp[lap] += w
        events.setdefault(lap, []).append(f"{name.get(ld['car_number'], '#' + ld['car_number'])} takes the lead")
    for cs in wk["caution_segments"]:
        a, b = int(cs["start_lap"]), int(cs["end_lap"])
        if a <= n and cs["reason"] != "Competition":
            imp[a] += 0.6
            events.setdefault(a, []).append(f"Caution: {cs['comment']}")
        if b + 1 <= n:
            imp[b + 1] += 0.5
            events.setdefault(b + 1, []).append("Restart" + (" after the stage break" if cs["reason"] == "Competition" else ""))
    imp[n] += 1.5
    win = wk["results"][0]
    events.setdefault(n, []).append(f"{win['driver_fullname']} wins by {wk['margin_of_victory']} s")
    imp[0] = 0.0
    dur = 1.0 + STRETCH * np.minimum(imp, 1.5)
    tau = np.concatenate([[0.0], np.cumsum(dur[1:])]); tau /= tau[-1]
    grid = np.linspace(0, 1, N)
    tl = np.interp(grid, tau, tension)
    level = 0.5 * tl / max(1e-6, tl.max())
    for s in np.nonzero(imp)[0]:
        level += 0.5 * imp[s] * np.exp(-0.5 * ((grid - tau[s]) / SIGMA) ** 2)
    out = {"laps": n, "tension": [round(float(v), 4) for v in tension], "closeness": [round(float(v), 4) for v in closeness],
           "importance": [round(float(v), 3) for v in imp], "events": {str(k): v for k, v in events.items()},
           "level": [round(float(v), 4) for v in level], "tau": [round(float(v), 5) for v in tau],
           "yellow": [bool(v) for v in yellow], "gap": [round(float(v), 3) for v in gap],
           "meta": {"title": f"{wk['race_name']} · {wk['track_name']} · {wk['race_date'][:10]}",
                    "winner": win["driver_fullname"], "margin": wk["margin_of_victory"],
                    "lead_changes": wk["number_of_lead_changes"], "cautions": wk["number_of_cautions"],
                    "final": f"{win['driver_fullname']} wins by {wk['margin_of_victory']} s"}}
    job.save("race.json", out)
    return out


# ---------------------------------------------------------------- v2 pipeline pieces for a NASCAR job
def race(job):
    f = job.wpath("race.json")
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else build(job)


def keys(r):
    """lap -> why: 'finish' (last lap), 'lead' (green-flag lead change), 'caution' (crash), 'restart'."""
    out = {}
    for lap, evs in r["events"].items():
        lap = int(lap); why = []
        for e in evs:
            if "takes the lead" in e and not r["yellow"][min(lap, r["laps"])]:
                why.append("lead")
            elif e.startswith("Caution"):
                why.append("caution")
            elif e.startswith("Restart"):
                why.append("restart")
            elif " wins by " in e:
                why.append("finish")
        if why:
            out[lap] = sorted(set(why))
    return out


def align(job):
    """alignment.json for a NASCAR job + length, in align.py's format (story mode: the race's story curve)."""
    import librosa
    import align as A
    import sister
    r = race(job)
    d, song_len = A.song_curve(job)
    t0, t1 = d["t0"], d["t1"]
    a = sister.normalise(d["curve"])
    n = r["laps"] + 1
    dist, i_of_j = A.mapping(a, sister.normalise(r["level"]))
    tau = np.asarray(r["tau"])
    t = np.maximum.accumulate(np.array([t0 + A.lerp(i_of_j, tau[s] * (A.N - 1)) / (A.N - 1) * (t1 - t0) for s in range(n)]))
    y, sr0 = A.song_window(job)
    y, sr = librosa.resample(np.asarray(y, np.float32), orig_sr=sr0, target_sr=22050), 22050
    _, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    env = librosa.onset.onset_strength(y=librosa.effects.percussive(y), sr=sr)
    pk = librosa.onset.onset_detect(onset_envelope=env, sr=sr, units="frames")
    hs = env[pk] / (np.percentile(env[pk], 95) if len(pk) else 1.0)
    k = keys(r)
    steps = [{"i": s, "t": round(float(t[s]), 3), "t_page": round(float(t[s]), 3), "wp": r["closeness"][s],
              "tension": r["tension"][s], "importance": r["importance"][s], "is_key": s in k, "why": k.get(s, []),
              "label": f"Lap {s}", "desc": "; ".join(r["events"].get(str(s), []))} for s in range(n)]
    job.save("alignment.json", {"match_id": job.gid, "title": r["meta"]["title"], "final": r["meta"]["final"],
                                "league": "NASCAR", "mode": "story", "song_len": song_len, "t0": t0, "t1": t1,
                                "dtw_dist": dist, "steps": steps, "beats": [round(float(v), 3) for v in beats],
                                "onsets": [round(float(v), 3) for v in onsets],
                                "hits": [[round(float(x), 3), round(float(min(1.0, v)), 3)]
                                         for x, v in zip(librosa.frames_to_time(pk, sr=sr), hs)]})
    print(f"aligned {job.tag}: {n} laps -> {song_len:.1f} s, DTW {dist:.4f}, {len(k)} key laps")


class LapParser:
    """FOX NASCAR leaderboard header -> lap: '40 OF 267 | STAGE 1 - 40 OF 80', '85 LAPS TO GO',
    'OVERTIME - 2 LAPS TO GO' (counts down to the actual finish), 'FINAL LAP', 'UNOFFICIAL RESULTS' (the finish)."""
    def __init__(self, total, actual=None):
        self.total, self.actual = total, actual or total

    def __call__(self, tokens):
        import re
        txt = " ".join(t[0] for t in tokens).upper().replace("0F", "OF")
        if re.search(r"FINA\s*L?\s*LAP|UNOFFICIAL|RESULTS", txt):
            return {"state": self.actual, "cost": 0.0, "read": "final lap"}
        if "OVERTIME" in txt:
            m = re.search(r"(\d)\s*LAPS?\s*T[O0]\s*G[O0]", txt)
            return {"state": self.actual - int(m.group(1)), "cost": 0.0, "read": "overtime"} if m else None
        m = re.search(r"(\d{1,3})\s*OF\s*(\d{3})", txt)
        if m and abs(int(m.group(2)) - self.total) <= 5:
            return {"state": int(m.group(1)), "cost": 0.0, "read": f"lap {m.group(1)}"}
        m = re.search(r"(\d{1,3})\s*LAPS?\s*T[O0]\s*G[O0]", txt)
        if m:
            return {"state": self.total - int(m.group(1)), "cost": 0.0, "read": f"{m.group(1)} to go"}
        return None

    def event_step(self, state):
        return state


def tag(job):
    """clips.json for a NASCAR highlights reel: shots, lap counter OCR at 2 fps, shot -> lap (the NFL shot logic)."""
    import tag_shots as T
    r = race(job)
    box = job.load("box.json")["part1.mp4"]["box"]
    part = job.parts[0]
    shots = T.detect_shots(job, part)
    crops = T.extract_crops(job, part, box)
    cache = job.wpath("ocr_part1.json")
    if cache.exists():
        toks = json.loads(cache.read_text())
    else:
        toks, n = T.ocr_frames([str(f) for f in crops], [True] * len(crops), "NFL", 6, [[0.0, 0.0, 1.0, 1.0]])
        cache.write_text(json.dumps(toks))
    parser = LapParser(int(job.cfg.get("scheduled_laps", 267)), r["laps"])
    frames = [{"t": round(k / T.FPS_OCR, 2), "tokens": toks[k]} for k in range(len(crops))]
    rows = T.tag_part(job, part, shots, frames, parser, "NFL")
    clips = [{"part": part.name, "start": x["start"], "end": x["end"], "step": x["step"], "kind": x["kind"],
              "event_t": round(x["start"] + 0.4 * (x["end"] - x["start"]), 2), "clock_stop": None,
              "conf": round(len(x["reads"]) / max(1, (x["end"] - x["start"]) * T.FPS_OCR), 2)} for x in rows]
    job.save("clips.json", clips)
    k = keys(r)
    live = {c["step"] for c in clips if c["kind"] == "live"}
    nread = sum(1 for f in frames if f["tokens"] and parser(f["tokens"]) is not None)
    print(f"tagged: {len(clips)} shots, {nread}/{len(frames)} frames read a lap; "
          f"key laps with a live shot: {sum(s in live for s in k)}/{len(k)}")


if __name__ == "__main__":
    r = build(job_from_argv())
    print(r["meta"], "laps", r["laps"], "events", len(r["events"]))
