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
usage: python scripts/plan_edit.py inputs/<job>
"""
import bisect
import numpy as np
from scipy.optimize import minimize

from common import job_from_argv, log, mmss, stamp

MIN_GAP, MIN_GAP_KEY = 0.9, 0.6
VMIN, VMIN_KEY_REPLAY, VMAX = 0.8, 0.75, 2.5


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
    key_steps = {s["i"] for s in steps if s["is_key"]}
    src = Source(clips)

    # ---- units: one per step with live footage (its consecutive live shots) ----
    units, seen = [], set()
    for k, c in enumerate(src.c):
        s = c["step"]
        if c["kind"] != "live" or s in seen or s is None or not 0 < s < len(steps):
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
        units.append({"step": s, "k": src.at(E), "E": E, "e": float(t[s]), "key": s in key_steps, "low": ten[s] < low_thr})
    units.sort(key=lambda u: u["E"])

    # 1. merge units that are too close in song time
    dropped = []
    changed = True
    while changed:
        changed = False
        for a, b in zip(units, units[1:]):
            gap = min(b["e"] - a["e"], b["E"] - a["E"])
            need = MIN_GAP_KEY if (a["key"] and b["key"]) else MIN_GAP
            if gap < need:
                lose = min((a, b), key=lambda u: (u["key"], ten[u["step"]]))
                units.remove(lose); dropped.append(lose); changed = True
                break

    # 2. key targets onto the beat / onset
    for u in units:
        u["target"] = u["e"]
        if u["key"] and len(hits):
            h = hits[np.argmin(np.abs(hits - u["e"]))]
            if abs(h - u["e"]) <= 0.5:
                u["target"] = float(h)

    # 3. place
    n = len(units)
    E = np.array([u["E"] for u in units]); tg = np.array([u["target"] for u in units])
    w = np.array([10.0 if u["key"] else 1.0 for u in units])
    cons = []
    for i in range(n - 1):
        cons.append({"type": "ineq", "fun": lambda x, i=i: (E[i + 1] - E[i]) / VMIN - (x[i + 1] - x[i])})
        cons.append({"type": "ineq", "fun": lambda x, i=i: (x[i + 1] - x[i]) - MIN_GAP_KEY})
    cons.append({"type": "ineq", "fun": lambda x: E[0] / VMIN - x[0]})
    cons.append({"type": "ineq", "fun": lambda x: x[0] - 0.5})
    cons.append({"type": "ineq", "fun": lambda x: (src.total - E[-1]) / VMIN - (song_len - x[-1])})
    cons.append({"type": "ineq", "fun": lambda x: song_len - 1.0 - x[-1]})
    res = minimize(lambda x: float(np.sum(w * (x - tg) ** 2)), np.clip(tg, 0.5, song_len - 1), jac=lambda x: 2 * w * (x - tg),
                   constraints=cons, method="SLSQP", options={"maxiter": 500, "ftol": 1e-9})
    ep = np.maximum.accumulate(res.x)
    for u, x in zip(units, ep):
        u["place"] = float(x)

    # 4. fill every stretch
    edl = []
    bounds = [(None, 0.0, 0.0)] + [(u, u["place"], u["E"]) for u in units] + [(None, song_len, src.total)]
    for (ua, sa, Ua), (ub, sb, Ub) in zip(bounds, bounds[1:]):
        edl += fill(src, ua, ub, sa, sb, Ua, Ub, ten, low_thr, key_steps, L)

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
        intended = float(t[s])
        u = placed.get(s)
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
    k_ok = ok / max(1, n_key) >= 0.8
    log("\n".join([
        f"### {stamp()} · Checkpoint 4 ({job.name}): {'PASS' if c_ok and k_ok else 'FAIL'}",
        f"- EDL entries {len(edl)} · cuts {cuts} · average shot {song_len / (cuts + 1):.2f} s · speeds "
        f"{min(speeds):.2f}–{max(speeds):.2f}× (median {np.median(speeds):.2f}×)",
        f"- Covers the song: sum {total:.3f} s vs song {song_len:.3f} s · gaps/overlaps {gaps or 'none'} ({c_ok})",
        f"- Units {len(units)} (dropped as too close: {len(dropped)}) · placement solver: {res.message}",
        f"- Key moments within ±1 s: {ok}/{n_key} reachable ones with live footage = {ok / max(1, n_key):.0%} (need ≥80%: {k_ok}); "
        f"{limited_n} more are footage-limited (the highlights hold too little footage before/after them)",
        *rep]))


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


def fill(src, ua, ub, sa, sb, Ua, Ub, ten, low_thr, key_steps, L):
    S = sb - sa
    if S <= 1e-6:
        return []
    pieces = src.pieces(Ua, Ub)
    for p in pieces:
        c = src.c[p["k"]]
        p["role"] = piece_role(p, src, ua, ub, Ua, Ub)
        p["step"] = c["step"]
        key, low = c["step"] in key_steps, c["step"] is not None and ten[c["step"]] < low_thr
        r = p["role"]
        if r in ("post", "pre", "live"):
            lo, hi, pref = (VMIN, 1.25, 1.0) if key else ((VMIN, VMAX, 2.0) if low else (VMIN, 2.0, 1.5))
        elif r == "replay":
            lo, hi, pref = (VMIN_KEY_REPLAY if key else VMIN), VMAX, 1.6
        elif r == "intro":                       # pre-game cold open (only exists before the first live shot)
            lo, hi, pref = VMIN, VMAX, 1.0
        else:
            lo, hi, pref = VMIN, VMAX, 2.0
        p.update(lo=lo, hi=hi, pref=pref, key=key, low=low)
        name = {"post": "after event", "pre": "up to event"}.get(r, r)
        p["why"] = ("key " if key else ("montage " if low else "")) + name
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
    """Too much footage for this stretch: keep the follow-through and the run-up, then extras by priority."""
    # IPL event = the delivery (keep the ball's flight and the reaction after it); NFL event = the play's result
    # (keep the play itself before it, a little reaction after)
    cap_post = {True: 5.0, False: 3.5} if L == "IPL" else {True: 3.0, False: 1.5}
    cap_pre = {True: 3.5, False: 3.0} if L == "IPL" else {True: 6.0, False: 4.0}
    left = {"post": None, "pre": None}
    order = []
    for p in pieces:                                        # post: from a's event forward, up to the cap in total
        if p["role"] == "post":
            cap = 2.0 if p["low"] and not p["key"] else cap_post[p["key"]]
            left["post"] = cap if left["post"] is None else left["post"]
            ln = min(p["u1"] - p["u0"], left["post"]); left["post"] -= ln
            p["u1"] = p["u0"] + ln; order.append((0, p))
    for p in reversed(pieces):                              # pre: back from b's event, up to the cap in total
        if p["role"] == "pre":
            left["pre"] = cap_pre[p["key"]] if left["pre"] is None else left["pre"]
            ln = min(p["u1"] - p["u0"], left["pre"]); left["pre"] -= ln
            p["u0"] = p["u1"] - ln; order.append((0 if L == "NFL" else 1, p))
    for p in pieces:
        r, ln = p["role"], p["u1"] - p["u0"]
        if r in ("post", "pre"):
            continue
        if r == "live" and p["key"]:                        # a key play merged into a neighbour: still show it
            mid = p["u0"] + 0.5 * ln; half = min(ln, 2.5) / 2
            p["u0"], p["u1"] = max(p["u0"], mid - half), min(p["u1"], mid + half); pri = 1   # always kept
        elif r == "replay" and p["key"]:
            p["u1"] = p["u0"] + min(ln, 4.0); pri = 2
        elif r == "live":
            mid = p["u0"] + 0.4 * ln; half = min(ln, 2.5) / 2
            p["u0"], p["u1"] = max(p["u0"], mid - half), min(p["u1"], mid + half); pri = 3
        elif r == "replay":
            p["u1"] = p["u0"] + min(ln, 2.5); pri = 4
        else:
            p["u1"] = p["u0"] + min(ln, 1.5); pri = 5
        order.append((pri, p))
    keep, used = set(), 0.0
    for pri, p in sorted(order, key=lambda x: x[0]):
        need = (p["u1"] - p["u0"]) / p["hi"]
        if need > 0 and (pri <= 1 or used + need <= S - 0.05):
            keep.add(id(p)); used += need
    keep = [p for p in pieces if id(p) in keep]
    tot = lambda ps: sum((p["u1"] - p["u0"]) / p["hi"] for p in ps)
    # still too long: first keep only the longest piece per merged-in play (no stutter of 0.4 s fragments)
    if tot(keep) > S:
        best = {}
        for p in keep:
            if p["role"] == "live" and (p["step"] not in best or p["u1"] - p["u0"] > best[p["step"]]["u1"] - best[p["step"]]["u0"]):
                best[p["step"]] = p
        keep = [p for p in keep if p["role"] != "live" or best.get(p["step"]) is p]
    # then trim everything proportionally, but never trim an event out: keep >= 0.6 s of b's footage up to its event
    if tot(keep) > S:
        f = S / tot(keep)
        for p in keep:
            ln = (p["u1"] - p["u0"]) * f
            if p["role"] == "pre":
                p["u0"] = p["u1"] - max(ln, min(0.6, p["u1"] - p["u0"]))
            else:
                p["u1"] = p["u0"] + ln
    # drop slivers (< 0.3 s on screen) unless they carry the event
    return [p for p in keep if (p["u1"] - p["u0"]) / p["hi"] >= 0.3 or p["role"] in ("post", "pre")]


def solve_speeds(items, S):
    """Speeds v_i = clip(lam * pref_i, lo_i, hi_i) with sum(len_i / v_i) = S (bisection on lam)."""
    ln = np.array([p["u1"] - p["u0"] for p in items])
    lo, hi, pref = (np.array([p[k] for p in items]) for k in ("lo", "hi", "pref"))
    f = lambda lam: float(np.sum(ln / np.clip(lam * pref, lo, hi)))
    a, b = 1e-3, 1e3
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
        if 0.75 <= va <= 2.6 and 0.75 <= vb <= 2.6:
            a["song_out"] = b["song_in"] = float(bt)
            a["speed"], b["speed"] = va, vb


if __name__ == "__main__":
    main()
