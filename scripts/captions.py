"""v2 Phase 4: captions. Claude Code writes them during the build (no API call); this script gives it the data and
checks the result.

  --brief   work/<job>/<L>/caption_brief.md: the caption slots (every key moment on screen, starting ~0.3 s before its
            play, and one slot every ~6-10 s in between) with the rows for the step on screen: play text, score and
            margin, quarter and clock, down and distance, win probability before/after, tension around it, and the
            plays just before (to explain a stretch where nothing much happens).
  --check   Checkpoint 4 on captions/<job>_<L>.json ([{start, end, step, text}]): <= 6 words, no banned words, no
            all-caps words over 3 letters, at most one "!" per video, 2-3.5 s each, no overlaps, step within +-2 of
            the step on screen at `start`, count in range.
usage: python scripts/captions.py inputs/<job> --len 60|120 --brief|--check
"""
import json, re, sys

import numpy as np

import sister
from common import ROOT, job_from_argv, log, mmss, stamp

BANNED = re.compile(r"music|song|sound|quiet|lull|crescendo|beat|tempo", re.I)
COUNT = {60: (7, 12), 120: (11, 18)}     # few: the game is the focus (Atharva, 2026-11-12)
GAP = (6.0, 10.0)


def on_screen(edl, t):
    e = next((e for e in edl if e["song_in"] <= t < e["song_out"]), edl[-1])
    return e["step"]


def row_info(job, s, plays, wp, ten, g):
    """The data for step s (the play whose result lands on s = row s-1)."""
    r = s - 1
    if not 0 <= r < len(plays):
        return {}
    p = plays.iloc[r]
    home, away = g["home"], g["away"]
    # nflverse's total_*_score is the score AFTER the play: before = the previous row's
    hs2, as2 = int(p.total_home_score), int(p.total_away_score)
    prv = plays.iloc[r - 1] if r > 0 else None
    hs, as_ = (int(prv.total_home_score), int(prv.total_away_score)) if prv is not None else (0, 0)
    dd = f"{int(p.down)}&{int(p.ydstogo)}" if p.down == p.down and p.down else ""
    return {"step": s, "q": f"Q{int(p.qtr)}" if p.qtr < 5 else "OT", "clock": p.time, "posteam": p.posteam,
            "down": dd, "score_before": f"{away} {as_} - {home} {hs}", "score_after": f"{away} {as2} - {home} {hs2}",
            "wp_home_before": round(float(wp[s - 1]), 2), "wp_home_after": round(float(wp[s]), 2),
            "tension": round(float(ten[s]), 2), "type": p.play_type, "desc": sister.clean_desc(p.desc, 200)}


def brief(job):
    al, edl = job.load("alignment.json"), job.load("edl.json")
    rep = {int(k): v for k, v in job.load("plan_report.json").items()}
    plays = sister.nfl_plays(job.gid)
    g = sister.game_json(job.gid)
    c = sister.curve(job.gid)
    wp, ten = np.asarray(c["wp"], float), np.asarray(c["tension"], float)
    ten = ten / ten.max()
    L = al["song_len"]
    steps = al["steps"]
    keys = sorted((v["planned"], s) for s, v in rep.items() if v.get("planned") is not None
                  and any(e["step"] == s and e.get("kind") == "live" for e in edl))
    slots = [(max(0.0, t - 2.0), s, "key") for t, s in keys]   # planned = the result; the play starts ~2-4 s before
    # fill: a slot every ~8 s where nothing is captioned
    t, out = 0.3, []
    for ts, s, kind in sorted(slots) + [(L, None, "end")]:
        while ts - t > GAP[1]:
            t += GAP[0] + 2.0
            if ts - t >= 3.0:
                out.append((t, on_screen(edl, t), "fill"))
        if kind != "end":
            out.append((ts, s, kind))
            t = ts
    lines = [f"# Caption brief: {job.name} {job.L} s ({g['title']}); home = {g['home']}, away = {g['away']}",
             f"Window {L:.1f} s. Write {COUNT[job.L][0]}-{COUNT[job.L][1]} captions, 2-3.5 s each, <= 6 words.", ""]
    for ts, s, kind in out:
        s_on = on_screen(edl, ts)
        info = row_info(job, s, plays, wp, ten, g)
        prev = [row_info(job, x, plays, wp, ten, g) for x in range(max(1, s - 4), s)]
        lines.append(f"## {mmss(ts)} ({ts:.1f} s) · {kind} · step {s} (on screen at start: {s_on})")
        lines.append("```json\n" + json.dumps(info, ensure_ascii=False) + "\n```")
        lines.append("before: " + " | ".join(f"{x.get('q')} {x.get('clock')} {x.get('posteam')} {x.get('type')}: "
                                             f"{x.get('desc', '')[:70]}" for x in prev))
        lines.append("")
    lines.append("## On screen (EDL): song time · step · kind · speed · play")
    for e in edl:
        info = row_info(job, e["step"], plays, wp, ten, g) if e["step"] else {}
        lines.append(f"- {e['song_in']:6.1f}-{e['song_out']:6.1f} · {e['step']} · {e['kind']} · {e['speed']:.2f}x · "
                     f"{info.get('q', '')} {info.get('clock', '')} {info.get('score_before', '')} · {info.get('desc', '')[:80]}")
    f = job.work / str(job.L) / "caption_brief.md"
    f.write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(out)} slots -> {f}")


def check(job):
    edl = job.load("edl.json")
    f = ROOT / "captions" / f"{job.name}_{job.L}.json"
    caps = json.loads(f.read_text(encoding="utf-8"))
    probs = []
    bangs = sum(c["text"].count("!") for c in caps)
    story = bool(job.story)        # story cuts: longer lines (Atharva, 2026-11-13: "longer, but not too long"), timed by script
    max_w, dur = (10, (1.5, 4.5)) if story else (6, (2.0, 3.5))
    if story:                      # a story file may allow longer lines (e.g. the dramatized cut)
        max_w, dur = job.story.get("max_words", max_w), (dur[0], job.story.get("caption_max_s", dur[1]))
    for i, c in enumerate(caps):
        w = c["text"].split()
        tag = f"#{i} {mmss(c['start'])} “{c['text']}”"
        if len(w) > max_w:
            probs.append(f"{tag}: {len(w)} words")
        if BANNED.search(c["text"]):
            probs.append(f"{tag}: banned word '{BANNED.search(c['text']).group(0)}'")
        caps_w = [x for x in re.findall(r"[A-Za-z]+", c["text"]) if len(x) > 3 and x.isupper()]
        if caps_w:
            probs.append(f"{tag}: all caps {caps_w}")
        d = c["end"] - c["start"]
        if not dur[0] - 1e-6 <= d <= dur[1] + 1e-6:
            probs.append(f"{tag}: lasts {d:.1f} s")
        if i and c["start"] < caps[i - 1]["end"] - 1e-6:
            probs.append(f"{tag}: overlaps the previous caption")
        s_on = on_screen(edl, c["start"])
        if not story and (s_on is None or abs(s_on - c["step"]) > 2):
            probs.append(f"{tag}: step {c['step']} but step {s_on} is on screen")
    lo, hi = COUNT[job.L]
    n_ok = story or lo <= len(caps) <= hi
    ok = not probs and bangs <= 1 and n_ok
    log("\n".join([f"### {stamp()} · Checkpoint 4 ({job.tag}): {'PASS' if ok else 'FAIL'}",
                   f"- `captions/{job.name}_{job.L}.json`: {len(caps)} captions ({lo}-{hi}: {n_ok}); '!' used {bangs}× (≤1: {bangs <= 1})",
                   f"- Problems: {len(probs)}", *[f"  - {p}" for p in probs]]))
    return ok


if __name__ == "__main__":
    job = job_from_argv()
    brief(job) if "--brief" in sys.argv else sys.exit(0 if check(job) else 1)
