"""Phase 1: song-to-game alignment. Which second of the song each game step (play / ball) belongs to.

Uses the sister repo's matcher exactly as its page does (site/index.html buildMaps + timeOfGameFrac, ported 1:1):
  song = normalise(demo curve), game = dequantised index curve, path = dtw_path(song, game)
  iOfJ[j] = mean song index of the path cells at game index j
  t(step) = t0 + lerp(iOfJ, step/(n-1) * (N-1)) / (N-1) * (t1 - t0)
so the moment timestamps equal the page's. Then: monotonic, beats + onsets, key steps.
usage: python scripts/align.py inputs/<job>
"""
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sister
from common import ROOT, job_from_argv, log, mmss, stamp

N = sister.N


def lerp(arr, x):
    x = min(N - 1, max(0.0, x))
    a = int(np.floor(x)); b = min(N - 1, a + 1)
    return arr[a] + (arr[b] - arr[a]) * (x - a)


def song_curve(job):
    """The curve the page matched with (demos.json), checked against a fresh analysis of inputs/<job>/song.mp3."""
    d = sister.demo(job.cfg["demo_key"])
    x, sr = sister.load_mono(job.song)
    e = sister.energy_curve(x, sr)
    fresh = sister.resample(np.asarray(e["curve"]), N)
    diff = float(np.max(np.abs(np.asarray(fresh) - np.asarray(d["curve"]))))
    assert diff < 1e-3 and abs(e["t0"] - d["t0"]) < 0.05 and abs(e["t1"] - d["t1"]) < 0.05, \
        f"song.mp3 is not the audio the page matched (max curve diff {diff:.4f})"
    return d, len(x) / sr


def main():
    job = job_from_argv()
    gid = job.gid
    d, song_len = song_curve(job)
    t0, t1 = d["t0"], d["t1"]
    a = sister.normalise(d["curve"])
    b = sister.index_curve(gid)
    dist, path = sister.dtw_path(a, b)
    si, ci = np.zeros(N), np.zeros(N)
    for i, j in path:
        si[j] += i; ci[j] += 1
    i_of_j = si / ci

    full = sister.curve(gid)
    n = full["n_steps"]
    t = np.array([t0 + lerp(i_of_j, s / (n - 1) * (N - 1)) / (N - 1) * (t1 - t0) for s in range(n)])
    t = np.maximum.accumulate(t)

    # beats and onsets
    y, sr = librosa.load(str(job.song), sr=22050, mono=True)
    _, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False)

    # key steps: the page's 5 moments + every wicket/six (IPL) or TD/turnover/FG (NFL)
    g = sister.game_json(gid)
    ev = {e["step"]: e for e in sister.events(gid)}
    moments = {m["step"]: m for m in g["moments"]}
    key = {}
    for s, e in ev.items():
        why = [k for k in (("wicket", "six") if job.league == "IPL" else ("td", "turnover", "fg")) if e.get(k)]
        if why:
            key[s] = why
    for s in moments:
        key.setdefault(s, []).insert(0, "moment")

    steps = []
    for s in range(n):
        e = ev.get(s, {})
        steps.append({"i": s, "t": round(float(t[s]), 3), "wp": full["wp"][s], "tension": full["tension"][s],
                      "is_key": s in key, "why": key.get(s, []), "label": e.get("label", "start" if s == 0 else "final"),
                      "desc": e.get("desc", "")})
    job.save("alignment.json", {"match_id": gid, "title": g["title"], "final": g["final"], "league": job.league,
                                "song_len": song_len, "t0": t0, "t1": t1, "dtw_dist": dist,
                                "steps": steps, "beats": [round(float(v), 3) for v in beats],
                                "onsets": [round(float(v), 3) for v in onsets]})

    # ---- checkpoint 1 ----
    mono = bool(np.all(np.diff(t) >= 0))
    start_ok, end_ok = t[0] <= 0.05 * song_len, t[-1] >= 0.95 * song_len
    ms = [m["step"] for m in g["moments"]]
    js = sister.page_moment_times(job.cfg["demo_key"], gid, ms)      # the page's own JS, run under node
    page_diff = max(abs(t[s] - v) for s, v in zip(ms, js))
    page_ok = page_diff <= 1.0
    plot(job, d, full, t, key, song_len)
    ok = mono and start_ok and end_ok and page_ok
    lines = [f"### {stamp()} · Checkpoint 1 ({job.name}): {'PASS' if ok else 'FAIL'}",
             f"- {g['title']} ↔ “{job.cfg['song_title']}” · DTW dist {dist:.4f} · {n} steps · song {song_len:.1f} s "
             f"(matched span {t0:.2f}–{t1:.2f} s) · {len(beats)} beats, {len(onsets)} onsets",
             f"- t_i monotonic: {mono}; starts at {t[0]:.2f} s (≤5%: {start_ok}); ends at {t[-1]:.2f} s (≥95%: {end_ok})",
             f"- Plot: `logs/{job.name}_alignment.png`",
             "- Key steps (song time · step · why · event):"]
    for s in sorted(key):
        lines.append(f"  - {mmss(t[s])} · step {s} · {'+'.join(key[s])} · {ev[s]['label']}: {ev[s]['desc'][:90]}")
    lines.append("- Page moments, here vs the page's own JS (core.js + index.html under node): "
                 + ", ".join(f"step {s} {mmss(t[s])} vs {mmss(v)}" for s, v in zip(ms, js))
                 + f" · max diff {page_diff:.3f} s (≤1 s: {page_ok})")
    log("\n".join(lines))


def plot(job, d, full, t, key, song_len):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True, gridspec_kw={"hspace": 0.08})
    st = np.linspace(d["t0"], d["t1"], N)
    ax1.plot(st, d["curve"], color="#7b3fe4", lw=1.8, label="song energy")
    ax1.set_ylabel("song energy"); ax1.legend(loc="upper left", fontsize=8)
    ten = np.asarray(full["tension"])
    ax2.plot(t, ten, color="#e4572e", lw=1.4, label="game tension, at its aligned song time")
    for s in sorted(key):
        ax2.axvline(t[s], color="#2a6fdb" if "moment" in key[s] else "#999", lw=1.2 if "moment" in key[s] else 0.5,
                    alpha=0.9 if "moment" in key[s] else 0.5)
    for s in range(0, len(t), max(1, len(t) // 25)):   # alignment lines: step s (bottom) → its song time (top)
        ax1.axvline(t[s], color="#bbb", lw=0.5, zorder=0)
    ax2.set_ylabel("tension"); ax2.set_xlabel("song time (s)"); ax2.set_xlim(0, song_len)
    ax2.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{job.name}: grey lines = every ~4% of game steps; blue = page moments; light = wickets/sixes or TDs/turnovers/FGs",
                 fontsize=9)
    fig.savefig(ROOT / "logs" / f"{job.name}_alignment.png", dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    main()
