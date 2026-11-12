"""Phase 4: the edit plan (EDL). Exactly covers [0, song_len], game order, the big moments on their song times.

Model: the usable footage (non-junk shots, in order) is one virtual source timeline. Each step with a live shot is
a unit (its consecutive live shots) with an event point E in the source and a target song time e = t_step:
IPL E = the delivery; NFL E = the play's result, i.e. the last moment the game clock ticked in its shots (the README
says the snap, but the song moment belongs to the result, and the snap is ~5 s earlier), else ~4 s after the snap.
  1. Merge: drop the weaker unit (non-key, lower tension) when two targets are closer than MIN_GAP.
  2. Key units' targets snap to the nearest beat/onset within 0.5 s.
  3. Place: choose song times e' close to e (key units weigh 10x) so every stretch can be filled at >= 0.8x
     (song gap <= source gap / 0.8) and none is shorter than MIN_GAP (SLSQP).
  4. Fill each stretch [e'_a, e'_b] from the source [E_a, E_b]: if it all fits at the allowed speeds, play it all;
     otherwise keep a's follow-through and b's run-up first, then a key moment's replay, then other live shots,
     then filler, each trimmed. Speeds: key live <= 1.25x, normal 1.25-2x, low-tension montage up to 2.5x,
     never below 0.8x (0.75x for a key replay).
  5. Snap cuts to a beat within 0.25 s.
Writes work/<job>/edl.json: [{part, src_in, src_out, speed, song_in, song_out, step, why}].
v2 (--len 60|120): runs against that length's alignment. When the key moments can't all fit at 1.0x (a 60 s summary),
the lowest-priority ones stop being key (PRIORITY: page moments > TDs > turnovers > lead-changing FGs > other FGs >
swing-only), and the drops are logged. The +-1 s check is advisory (sync precision isn't the goal).
usage: python scripts/plan_edit.py inputs/<job> [--len 60|120]
"""
import bisect
import numpy as np
from scipy.optimize import minimize

import sister
from common import job_from_argv, log, mmss, stamp

MIN_GAP, MIN_GAP_KEY = 0.9, 0.6
VMIN, VMIN_KEY_REPLAY = 0.8, 0.75
# speed caps: (normal, emergency). The emergency caps are used only when a stretch cannot show its must-keep moments
# (every wicket / TD / turnover with its delivery and result) at the normal caps.
CAP = {"key": (1.0, 1.25), "live": (1.25, 1.5), "other": (1.25, 1.75)}
VMAX = CAP["other"][1]
MUST = {"IPL": ("wicket", "six"), "NFL": ("td", "turnover", "fg"), "NASCAR": ("finish", "lead")}
HIGH = {"IPL": ("swing", "moment"), "NFL": ("swing", "moment"), "NASCAR": ("caution", "restart")}
HIT_WINDOW = 1.5    # how far a key moment may move to land on a strong hit
END_HOLD = 4.0      # the last moment lands at least this long before the song ends (its result plays out)
# the core of a key moment that must be shown: before its event point / after it (seconds of source)
CORE = {"IPL": (1.2, {"must": 3.5, "high": 3.0}), "NFL": (4.0, {"must": 2.0, "high": 1.5}),
        "NASCAR": (2.5, {"must": 2.5, "high": 1.5})}
CORE_SHORT = {"IPL": (1.2, 3.0), "NFL": (3.0, 1.5), "NASCAR": (2.0, 1.5)}    # v2 60 s summaries


class Source:
    """Non-junk shots laid end to end: virtual time u <-> (part, t)."""
    def __init__(self, clips):
        self.c = [dict(c) for c in clips if c["kind"] != "junk"]
        u = 0.0
        for c in self.c:
            c["u0"], c["u1"] = u, u + (c["end"] - c["start"])
            u = c["u1"]
        self.total = u

    def at(self, u):
        k = min(len(self.c) - 1, max(0, bisect.bisect_right([c["u0"] for c in self.c], u) - 1))
        return k

    def pieces(self, ua, ub):
        out = []
        for k, c in enumerate(self.c):
            a, b = max(ua, c["u0"]), min(ub, c["u1"])
            if b - a > 1e-6:
                out.append({"k": k, "u0": a, "u1": b})
        return out

    def real(self, k, u):
        c = self.c[k]
        return c["start"] + (u - c["u0"])


def main():
    job = job_from_argv()
    al, clips = job.load("alignment.json"), job.load("clips.json")
    L = job.league
    steps = al["steps"]
    t = np.array([s["t"] for s in steps])
    ten = np.array([s["tension"] for s in steps])
    low_thr = float(np.percentile(ten, 30))
    song_len = al["song_len"]
    beats = np.array(al["beats"])
    hits = np.array(sorted(set(al["beats"]) | set(al["onsets"])))
    tier = {}
    for st in steps:
        if any(w in MUST[L] for w in st["why"]) or ((job.L or L == "NFL") and "moment" in st["why"]):   # v2: page moments are must-keep (NFL)
            tier[st["i"]] = "must"
        elif any(w in HIGH[L] for w in st["why"]):
            tier[st["i"]] = "high"
    key_steps = set(tier)
    src = Source(clips)
    pri = priorities(job, steps) if job.L else {}

    # ---- units: one per step with live footage (its consecutive live shots) ----
    units, seen = [], set()
    for k, c in enumerate(src.c):
        s = c["step"]
        if c["kind"] != "live" or s in seen or s is None or not 0 < s < len(steps) or s not in key_steps:
            continue
        seen.add(s)
        grp = [k]
        while grp[-1] + 1 < len(src.c) and src.c[grp[-1] + 1]["kind"] == "live" and src.c[grp[-1] + 1]["step"] == s:
            grp.append(grp[-1] + 1)
        if L == "NFL":   # the result: the last clock stop among the play's shots, else ~4 s after the snap
            stops = [j for j in grp if src.c[j].get("clock_stop") is not None]
            if stops:
                j = stops[-1]; E = src.c[j]["u0"] + (src.c[j]["clock_stop"] - src.c[j]["start"])
            else:
                E = c["u0"] + min(4.0, 0.5 * (c["u1"] - c["u0"]))
        else:            # the delivery
            E = c["u0"] + (c["event_t"] - c["start"])
        # how much of the moment's own footage exists either side of its event point (for the minimum spacing)
        own = [j for j in range(len(src.c)) if src.c[j]["step"] == s and src.c[j]["kind"] == "live"]
        pre_av = E - min(src.c[j]["u0"] for j in own)
        post_av = max(src.c[j]["u1"] for j in own) - E
        core_pre, core_post = min(pre_av, CORE[L][0]), min(post_av, CORE[L][1][tier[s]])
        short = (min(core_pre, CORE_SHORT[L][0]), min(core_post, CORE_SHORT[L][1]))
        units.append({"step": s, "k": src.at(E), "E": E, "e": float(t[s]), "key": True, "tier": tier[s],
                      "low": ten[s] < low_thr, "core": (core_pre, core_post), "core_short": short})
    units.sort(key=lambda u: u["E"])
    prio_dropped = []
    if job.L:      # v2: drop key moments by priority until the rest fit at 1.0x
        units, prio_dropped = prune(units, pri, song_len, ten)
        for u in prio_dropped:
            tier.pop(u["step"], None); key_steps.discard(u["step"])

    # 1. merge units that are too close in song time
    dropped = []
    changed = True
    while changed:
        changed = False
        for a, b in zip(units, units[1:]):
            gap = min(b["e"] - a["e"], b["E"] - a["E"])
            need = MIN_GAP_KEY if (a["key"] and b["key"]) else MIN_GAP
            if gap < need and not (a["tier"] == "must" and b["tier"] == "must"):
                lose = min((a, b), key=lambda u: (u["tier"] == "must", ten[u["step"]]))
                units.remove(lose); dropped.append(lose); changed = True
                break

    # 2. key targets onto the music's strongest percussive hit nearby (drums, timpani, accents): the strongest hit
    #    within +-1.5 s, preferring nearer ones; else the nearest beat / onset within 0.5 s
    ph = np.array(al.get("hits", []), float).reshape(-1, 2)
    for u in units:
        u["target"] = u["e"]
        near = ph[np.abs(ph[:, 0] - u["e"]) <= HIT_WINDOW] if len(ph) else ph
        if len(near):
            score = near[:, 1] - 0.15 * np.abs(near[:, 0] - u["e"])
            best = near[int(np.argmax(score))]
            if best[1] >= 0.35:
                u["target"] = float(best[0]); u["hit"] = round(float(best[1]), 2)
                continue
        if len(hits):
            h = hits[np.argmin(np.abs(hits - u["e"]))]
            if abs(h - u["e"]) <= 0.5:
                u["target"] = float(h)

    # 3. place
    n = len(units)
    E = np.array([u["E"] for u in units]); tg = np.array([u["target"] for u in units])
    w = np.array([10.0 if u["tier"] == "must" else 5.0 for u in units])
    cons = []
    for i in range(n - 1):
        cons.append({"type": "ineq", "fun": lambda x, i=i: (E[i + 1] - E[i]) / VMIN - (x[i + 1] - x[i])})
        need = max(MIN_GAP_KEY, (units[i]["core"][1] + units[i + 1]["core"][0]) / CAP["key"][1])
        cons.append({"type": "ineq", "fun": lambda x, i=i, need=need: (x[i + 1] - x[i]) - need})
    cons.append({"type": "ineq", "fun": lambda x: E[0] / VMIN - x[0]})
    cons.append({"type": "ineq", "fun": lambda x: x[0] - max(0.5, units[0]["core"][0] / CAP["key"][1])})
    cons.append({"type": "ineq", "fun": lambda x: (src.total - E[-1]) / VMIN - (song_len - x[-1])})
    cons.append({"type": "ineq", "fun": lambda x: song_len - END_HOLD - x[-1]})
    res = minimize(lambda x: float(np.sum(w * (x - tg) ** 2)), np.clip(tg, 0.5, song_len - 1), jac=lambda x: 2 * w * (x - tg),
                   constraints=cons, method="SLSQP", options={"maxiter": 500, "ftol": 1e-9})
    ep = np.maximum.accumulate(res.x)
    for u, x in zip(units, ep):
        u["place"] = float(x)

    # 4. fill every stretch
    edl = []
    bounds = [(None, 0.0, 0.0)] + [(u, u["place"], u["E"]) for u in units] + [(None, song_len, src.total)]
    for (ua, sa, Ua), (ub, sb, Ub) in zip(bounds, bounds[1:]):
        edl += fill(src, ua, ub, sa, sb, Ua, Ub, ten, low_thr, key_steps, L, tier)

    # merge contiguous source at equal speed, then 5. snap cuts to beats
    edl = merge(edl)
    snap_cuts(edl, beats)
    edl = drop_slivers(merge(edl))
    for e in edl:
        e["song_in"], e["song_out"] = round(e["song_in"], 4), round(e["song_out"], 4)
        e["src_in"], e["src_out"], e["speed"] = round(e["src_in"], 4), round(e["src_out"], 4), round(e["speed"], 4)
    job.save("edl.json", edl)

    # ---- checkpoint 4 ----
    total = sum(e["song_out"] - e["song_in"] for e in edl)
    gaps = [round(b["song_in"] - a["song_out"], 4) for a, b in zip(edl, edl[1:]) if abs(b["song_in"] - a["song_out"]) > 1e-3]
    cuts = sum(1 for a, b in zip(edl, edl[1:]) if not (a["part"] == b["part"] and abs(a["src_out"] - b["src_in"]) < 0.05))
    rep, ok, n_key, limited_n, report = [], 0, 0, 0, {}
    placed = {u["step"]: u for u in units}
    for s in sorted(key_steps):
        u = placed.get(s)
        intended = float(u["target"]) if u is not None else float(t[s])   # aligned time, moved onto a strong hit
        if u is None:
            if s not in {d["step"] for d in dropped}:
                rep.append(f"  - step {s} ({steps[s]['label']}, {'+'.join(steps[s]['why'])}): intended {mmss(intended)} · "
                           f"not in the footage")
                report[s] = {"intended": intended, "planned": None, "status": "not in footage"}
                continue
            # merged into a neighbour: its shots may still be on screen; judge by where they play
            on = [e for e in edl if e["step"] == s and e["kind"] == "live"]
            planned = (on[0]["song_in"] + on[0]["song_out"]) / 2 if on else None
            good = planned is not None and abs(planned - intended) <= 1.0
            n_key += 1; ok += good
            report[s] = {"intended": intended, "planned": planned, "status": "ok" if good else "merged"}
            rep.append(f"  - step {s} ({steps[s]['label']}, {'+'.join(steps[s]['why'])}): intended {mmss(intended)} · "
                       + (f"merged into a neighbour, its shot plays at {mmss(planned)} · Δ {planned - intended:+.2f} s "
                          f"{'✓' if good else '✗'}" if on else "merged into a neighbour and cut for time ✗"))
            continue
        planned = song_time_of(edl, src, u)
        good = planned is not None and abs(planned - intended) <= 1.0
        # reachable at all? all footage before the event at the slowest speed / all footage after it likewise
        latest, earliest = u["E"] / VMIN, song_len - (src.total - u["E"]) / VMIN
        limited = intended > latest + 1.0 or intended < earliest - 1.0
        if limited and not good:
            limited_n += 1
            note = (f"footage-limited: too little footage {'before' if intended > latest else 'after'} it, so it can land "
                    f"{'no later than ' + mmss(latest) if intended > latest else 'no earlier than ' + mmss(earliest)}")
        else:
            n_key += 1; ok += good; note = ""
        report[s] = {"intended": intended, "planned": planned,
                     "status": "ok" if good else ("footage-limited" if limited else "miss")}
        rep.append(f"  - step {s} ({steps[s]['label']}, {'+'.join(steps[s]['why'])}): intended {mmss(intended)} · planned "
                   f"{mmss(planned) if planned is not None else '—'} · Δ "
                   f"{f'{planned - intended:+.2f}' if planned is not None else '?'} s {'✓' if good else '✗'} {note}")
    job.save("plan_report.json", report)
    speeds = [e["speed"] for e in edl]
    c_ok = abs(total - song_len) <= 0.05 and not gaps and abs(edl[0]["song_in"]) < 1e-6
    # the +-1 s check is advisory since v2: the song/game correlation doesn't need to be tight ("the human
    # mind still tends to do a good job of finding patterns"), so it is reported but never fails a run
    k_ok = ok / max(1, n_key) >= 0.8
    if job.L:
        return checkpoint2(job, edl, units, prio_dropped + dropped, pri, steps, song_len, total, gaps, cuts, rep, ok, n_key)
    log("\n".join([
        f"### {stamp()} · Checkpoint 4 ({job.name}): {'PASS' if c_ok else 'FAIL'}",
        f"- EDL entries {len(edl)} · cuts {cuts} · average shot {song_len / (cuts + 1):.2f} s · speeds "
        f"{min(speeds):.2f}–{max(speeds):.2f}× (median {np.median(speeds):.2f}×)",
        f"- Covers the song: sum {total:.3f} s vs song {song_len:.3f} s · gaps/overlaps {gaps or 'none'} ({c_ok})",
        f"- Units {len(units)} (dropped as too close: {len(dropped)}) · placement solver: {res.message}",
        f"- Key moments within ±1 s (advisory): {ok}/{n_key} reachable ones with live footage = {ok / max(1, n_key):.0%}; "
        f"{limited_n} more are footage-limited (the highlights hold too little footage before/after them)",
        *rep]))


PRIORITY_NAMES = {1: "page moment", 2: "TD", 3: "turnover", 4: "lead-changing FG", 5: "FG", 6: "swing only",
                  7: "other"}


def priorities(job, steps):
    """v2 Phase 2: step -> priority (1 = keep first). NFL: page moments, TDs, turnovers, lead-changing FGs, other FGs,
    then key steps that are only a win-probability swing. IPL (no v2 runs): wickets 2, sixes 3."""
    sb = sister.curve(job.gid)["sb"] if job.league != "NASCAR" else None
    out = {}
    for st in steps:
        w, s = st["why"], st["i"]
        if not w:
            continue
        if job.league == "NASCAR":
            out[s] = 1 if "finish" in w else 2 if "lead" in w else 3 if "caution" in w else 5
            continue
        if "moment" in w:
            out[s] = 1
        elif "td" in w or "wicket" in w:
            out[s] = 2
        elif "turnover" in w or "six" in w:
            out[s] = 3
        elif "fg" in w:
            if job.league == "NFL" and 0 < s < len(sb["h"]):
                before = np.sign(sb["h"][s - 1] - sb["a"][s - 1])
                after = np.sign(sb["h"][s] - sb["a"][s])
                out[s] = 4 if before != after else 5
            else:
                out[s] = 5
        elif "swing" in w:
            out[s] = 6
        else:
            out[s] = 7
    return out


def prune(units, pri, song_len, ten):
    """Drop key moments, lowest priority first (least tense first within a priority), until every remaining one gets
    its core (run-up + result) at 1.0x: start + sum of pairwise spacing + END_HOLD <= the song."""
    def need(us):
        if not us:
            return 0.0
        tot = max(0.5, us[0]["core"][0] / CAP["key"][0]) + END_HOLD
        for a, b in zip(us, us[1:]):
            tot += max(MIN_GAP_KEY, (a["core"][1] + b["core"][0]) / CAP["key"][0])
        return tot
    units, dropped = list(units), []
    if need(units) > song_len:          # first: shorter cores (run-up 3 s, result 1.5 s) instead of faster ones
        for u in units:
            u["core"] = u["core_short"]
    while units and need(units) > song_len:
        lose = max(units, key=lambda u: (pri.get(u["step"], 7), -ten[u["step"]]))
        units.remove(lose); dropped.append(lose)
    return units, dropped


def checkpoint2(job, edl, units, prio_dropped, pri, steps, song_len, total, gaps, cuts, rep, ok, n_key):
    """v2 Checkpoint 2: length, continuity, speed, key moments on screen (by priority for 60 s)."""
    speeds = [e["speed"] for e in edl]
    on_screen = {e["step"] for e in edl if e.get("kind") == "live"}
    in_footage = {u["step"] for u in units} | {u["step"] for u in prio_dropped}
    keys = [s for s in pri if s in in_footage]
    shown = [s for s in keys if s in on_screen]
    len_ok = abs(total - song_len) <= 0.1 and abs(song_len - job.L) <= 0.5
    cont_ok = not gaps and abs(edl[0]["song_in"]) < 1e-6
    sp_ok = max(speeds) <= 1.35
    if job.L >= 120:
        k_ok = len(shown) >= 0.8 * len(keys)
        k_txt = f"key moments in the footage on screen: {len(shown)}/{len(keys)} (≥80%: {k_ok})"
    else:
        must = [s for s in keys if pri[s] <= 2]
        k_ok = all(s in on_screen for s in must)
        k_txt = (f"page moments + TDs in the footage on screen: {sum(s in on_screen for s in must)}/{len(must)} "
                 f"(all: {k_ok}); all key moments on screen {len(shown)}/{len(keys)}")
    ok_all = len_ok and cont_ok and sp_ok and k_ok
    import json
    (job.work / str(job.L) / "check2.json").write_text(json.dumps({
        "pass": bool(ok_all), "speed_min": round(min(speeds), 3), "speed_max": round(max(speeds), 3),
        "keys_shown": len(shown), "keys_in_footage": len(keys), "dropped": [u["step"] for u in prio_dropped]}), encoding="utf-8")
    drops = [f"  - dropped step {u['step']} ({steps[u['step']]['label']}, {PRIORITY_NAMES[pri.get(u['step'], 7)]}): "
             f"{'still on screen as ordinary footage' if u['step'] in on_screen else 'cut'} (no room at 1.0x)"
             for u in prio_dropped]
    log("\n".join([
        f"### {stamp()} · Checkpoint 2 ({job.tag}): {'PASS' if ok_all else 'FAIL'}",
        f"- EDL {total:.3f} s vs window {song_len:.3f} s (L ± 0.1: {len_ok}) · gaps/overlaps {gaps or 'none'} ({cont_ok}) · "
        f"{len(edl)} entries, {cuts} cuts",
        f"- Speeds {min(speeds):.2f}–{max(speeds):.2f}× (median {np.median(speeds):.2f}×; max ≤ 1.35: {sp_ok})",
        f"- {k_txt}",
        f"- Key moments kept as anchors: {len(units)}; dropped by priority: {len(prio_dropped)}", *drops,
        f"- ±1 s of the aligned time (advisory): {ok}/{n_key}", *rep]))


def song_time_of(edl, src, u):
    """Song time at which the unit's source event plays."""
    c = src.c[u["k"]]
    real = src.real(u["k"], u["E"])
    for e in edl:
        if e["part"] == c["part"] and e["src_in"] - 1e-3 <= real <= e["src_out"] + 1e-3:
            return e["song_in"] + (real - e["src_in"]) / e["speed"]
    return None


def piece_role(p, src, ua, ub, Ua, Ub):
    """post = a's own live footage after its event; pre = b's own live footage before its event."""
    c = src.c[p["k"]]
    if c["kind"] == "live" and ub is not None and c["step"] == ub["step"]:
        return "pre"
    if c["kind"] == "live" and ua is not None and c["step"] == ua["step"]:
        return "post"
    return c["kind"]


def fill(src, ua, ub, sa, sb, Ua, Ub, ten, low_thr, key_steps, L, tier):
    S = sb - sa
    if S <= 1e-6:
        return []
    pieces = src.pieces(Ua, Ub)
    for p in pieces:
        c = src.c[p["k"]]
        p["role"] = piece_role(p, src, ua, ub, Ua, Ub)
        p["step"] = c["step"]
        key, low = c["step"] in key_steps, c["step"] is not None and ten[c["step"]] < low_thr
        p["tier"] = tier.get(c["step"])
        r = p["role"]
        if r in ("post", "pre"):
            cap, pref = (CAP["key"], 0.95) if key else (CAP["live"], 1.1)
        elif r == "live":
            cap, pref = (CAP["key"], 1.0) if key else (CAP["live"], 1.15)
        elif r == "intro":                       # pre-game cold open (only exists before the first live shot)
            cap, pref = CAP["other"], 1.0
        else:
            cap, pref = CAP["other"], 1.2
        lo = VMIN_KEY_REPLAY if (r == "replay" and key) else VMIN
        p.update(lo=lo, hi=cap[0], hi2=cap[1], pref=pref, key=key, low=low)
        name = {"post": "after event", "pre": "up to event"}.get(r, r)
        p["why"] = ("key " if key else "") + name
    if sum((p["u1"] - p["u0"]) / p["hi"] for p in pieces) > S:      # too much footage: choose and trim
        pieces = choose(pieces, S, L)
    items = [p for p in pieces if p["u1"] - p["u0"] > 0.05]
    if not items:   # nothing usable: extend the neighbouring event shot (should not happen; constraints prevent it)
        return []
    speeds = solve_speeds(items, S)
    out, song = [], sa
    for p, v in zip(items, speeds):
        d = (p["u1"] - p["u0"]) / v
        c = src.c[p["k"]]
        out.append({"part": c["part"], "src_in": src.real(p["k"], p["u0"]), "src_out": src.real(p["k"], p["u1"]),
                    "speed": v, "song_in": song, "song_out": song + d, "step": p["step"], "why": p["why"],
                    "kind": c["kind"]})
        song += d
    out[-1]["song_out"] = sb                                           # absorb float error
    out[-1]["speed"] = (out[-1]["src_out"] - out[-1]["src_in"]) / (sb - out[-1]["song_in"])
    return out


def choose(pieces, S, L):
    """Too much footage for this stretch. Priorities:
      0  the moment on each side: a's result after its event, b's delivery / snap up to its event (always kept;
         trimmed to their core only if even the emergency speed caps can't fit them)
      2  the follow-up of a key moment (replays, the batter's name card, the celebration)
      3  other balls / plays (their own action, from the start of their shot)
      4  replays and reactions of ordinary balls, filler
    Pieces are first taken short (capped) in priority order; any room left is then given back as longer pieces,
    again in priority order, so the stretch is filled with substance rather than slowed down."""
    core_pre = CORE[L][0]
    for p in pieces:
        p["o0"], p["o1"] = p["u0"], p["u1"]
        r, ln, key = p["role"], p["u1"] - p["u0"], p["key"]
        if r in ("post", "pre"):
            pri = 0
        elif r == "replay" and key:
            pri = 2
        elif r == "live":
            pri = 3
        else:
            pri = 4
        p["pri"] = pri
    # short versions
    post_left, pre_left = {}, {}
    for p in pieces:                                       # post: from a's event forward
        if p["role"] == "post":
            cap = 5.0 if p["key"] else 3.0
            left = post_left.setdefault("a", cap)
            ln = min(p["u1"] - p["u0"], left); post_left["a"] = left - ln
            p["s0"], p["s1"] = p["u0"], p["u0"] + ln
    for p in reversed(pieces):                             # pre: back from b's event
        if p["role"] == "pre":
            cap = 2.5 if L == "IPL" else 7.0
            left = pre_left.setdefault("b", cap)
            ln = min(p["u1"] - p["u0"], left); pre_left["b"] = left - ln
            p["s0"], p["s1"] = p["u1"] - ln, p["u1"]
    for p in pieces:
        if p["role"] in ("post", "pre"):
            continue
        ln = p["u1"] - p["u0"]
        short = {2: 3.0, 3: 2.6 if L == "IPL" else 4.0, 4: 1.2}[p["pri"]]
        p["s0"], p["s1"] = p["u0"], p["u0"] + min(ln, short)
    dur = lambda p, a, b, hi: (b - a) / hi
    keep, used = [], 0.0
    for p in sorted(pieces, key=lambda p: (p["pri"], p["u0"])):
        need = dur(p, p["s0"], p["s1"], p["hi"])
        if p["pri"] == 0 or used + need <= S - 0.05:
            keep.append(p); used += need
    for p in keep:
        p["u0"], p["u1"] = p["s0"], p["s1"]
    # room left: grow kept pieces back toward their full length, by priority
    room = S - used
    for p in sorted(keep, key=lambda p: (p["pri"], p["u0"])):
        if room <= 0.05:
            break
        if p["role"] == "pre":
            grow = min(p["u0"] - p["o0"], room * p["hi"]); p["u0"] -= grow
        else:
            grow = min(p["o1"] - p["u1"], room * p["hi"]); p["u1"] += grow
        room -= grow / p["hi"]
    keep = [p for p in pieces if p in keep]
    tot = lambda ps, k="hi": sum((p["u1"] - p["u0"]) / p[k] for p in ps)
    # the must-keep pieces alone don't fit: emergency speed caps
    if tot(keep) > S + 1e-6:
        for p in keep:
            p["hi"] = p["hi2"]
    # still too long: drop optional pieces, then trim a's result from its end and b's run-up from its start, keeping
    # b's delivery / snap up to its event (the core) whole as long as possible
    if tot(keep) > S + 1e-6:
        keep = [p for p in keep if p["pri"] == 0]
    if tot(keep) > S + 1e-6:
        pre = [p for p in keep if p["role"] == "pre"]
        post = [p for p in keep if p["role"] == "post"]
        pre_len = sum(p["u1"] - p["u0"] for p in pre)
        hi = max(p["hi"] for p in keep)
        budget = S * hi                                            # source seconds that fit
        pre_keep = min(pre_len, max(core_pre, budget - sum(p["u1"] - p["u0"] for p in post)), budget * 0.6)
        post_keep = max(0.0, budget - pre_keep)
        for p in reversed(pre):                                    # from the event backwards
            ln = min(p["u1"] - p["u0"], pre_keep); pre_keep -= ln; p["u0"] = p["u1"] - ln
        for p in post:                                             # from the event forwards
            ln = min(p["u1"] - p["u0"], post_keep); post_keep -= ln; p["u1"] = p["u0"] + ln
        for p in keep:
            p["hi"] = hi
    # drop slivers (< 0.3 s on screen) unless they carry the event
    return [p for p in keep if (p["u1"] - p["u0"]) / p["hi"] >= 0.3 or p["role"] in ("post", "pre")]


def solve_speeds(items, S):
    """Speeds v_i = clip(lam * pref_i, lo_i, hi_i) with sum(len_i / v_i) = S (bisection on lam)."""
    ln = np.array([p["u1"] - p["u0"] for p in items])
    lo, hi, pref = (np.array([p[k] for p in items]) for k in ("lo", "hi", "pref"))
    f = lambda lam: float(np.sum(ln / np.clip(lam * pref, lo, hi)))
    a, b = 1e-3, 1e3
    if f(b) > S:   # even the caps can't fit (trimming should have prevented this): emergency caps, then uniform
        hi = np.array([p["hi2"] for p in items])
        f = lambda lam: float(np.sum(ln / np.clip(lam * pref, lo, hi)))
    if f(a) < S or f(b) > S:   # can't fit inside the speed limits either way: one uniform speed (shows in the report)
        return [float(ln.sum() / S)] * len(items)
    for _ in range(80):
        m = (a * b) ** 0.5
        a, b = (m, b) if f(m) > S else (a, m)
    return list(np.clip(((a * b) ** 0.5) * pref, lo, hi))


def merge(edl):
    out = []
    for e in edl:
        if out and out[-1]["part"] == e["part"] and abs(out[-1]["src_out"] - e["src_in"]) < 1e-3 \
                and abs(out[-1]["speed"] - e["speed"]) < 1e-3 and out[-1]["step"] == e["step"]:
            out[-1]["src_out"], out[-1]["song_out"] = e["src_out"], e["song_out"]
        else:
            out.append(dict(e))
    return out


def drop_slivers(edl, min_dur=0.25):
    """Fold entries too short to read on screen into their neighbour (which plays a touch slower)."""
    out = []
    for e in edl:
        if e["song_out"] - e["song_in"] < min_dur and out:
            p = out[-1]
            p["song_out"] = e["song_out"]
            p["speed"] = (p["src_out"] - p["src_in"]) / (p["song_out"] - p["song_in"])
        elif out and out[-1]["song_out"] - out[-1]["song_in"] < min_dur:
            p = out.pop()
            e = dict(e, song_in=p["song_in"])
            e["speed"] = (e["src_out"] - e["src_in"]) / (e["song_out"] - e["song_in"])
            out.append(e)
        else:
            out.append(dict(e))
    return out


def snap_cuts(edl, beats):
    """Move a hard cut onto a beat within 0.25 s by retiming the two neighbours (speeds stay in range)."""
    if not len(beats):
        return
    for a, b in zip(edl, edl[1:]):
        if a["part"] == b["part"] and abs(a["src_out"] - b["src_in"]) < 0.05:
            continue                                                       # continuous footage: not a cut
        x = a["song_out"]
        bt = beats[np.argmin(np.abs(beats - x))]
        if abs(bt - x) > 0.25 or bt - a["song_in"] < 0.3 or b["song_out"] - bt < 0.3:
            continue
        va = (a["src_out"] - a["src_in"]) / (bt - a["song_in"])
        vb = (b["src_out"] - b["src_in"]) / (b["song_out"] - bt)
        # a beat-snap may nudge a speed by a few percent, never past the piece's own normal cap (or its current speed
        # if it was already in emergency territory)
        lim = lambda e, v: 0.75 <= v <= 1.04 * max(e["speed"], CAP["key" if e["why"].startswith("key") else "live"][0])
        if lim(a, va) and lim(b, vb):
            a["song_out"] = b["song_in"] = float(bt)
            a["speed"], b["speed"] = va, vb


if __name__ == "__main__":
    main()
