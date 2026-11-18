"""Game excitement curve: how a viewer feels the game, play by play, from the research on suspense and surprise.

Why: the earlier curves were levels. Tension (sister repo: leverage + swing, smoothed) stays high for as long as a late
game is close, and the story curve smooths a run of scores into one long plateau. Viewers respond to changes:
  - Ely, Frankel & Kamenica (2015, JPE), "Suspense and Surprise": surprise = how far beliefs just moved; suspense =
    how far they may move next.
  - Minute-by-minute TV audiences rise with both (Wimbledon: Bizzozero, Flepp & Franck 2016; Premier League: Buraimo
    et al. 2020; Champions League: Richardson et al. 2023), with surprise the larger effect (Wimbledon: 2-5x).
  - Fans' heart rate spikes on goals and then settles (e.g. the 2026 Scientific Reports smartwatch study).
So a score that ties a game is a spike, and the tied stretch after it is a lower, held level (suspense), not a peak.

Per step s of the sister repo's indexing (step s = the state after play s-1; the appended last step = the outcome):
  surprise[s]   = |wp[s] - wp[s-1]|                   the win-probability jump just seen
  suspense[s]   = E|dWP| of the next play, given the state (sister.nfl_suspense: the sister repo's leverage model)
  afterglow[s]  = surprise, each jump fading with a half-life of HALF_LIFE plays (attack instant)
  excitement[s] = afterglow[s] + W_SUSPENSE * suspense[s]
Both terms are in win-probability units (suspense is the expected surprise of the next play), so W_SUSPENSE < 1 says
directly that a jump seen counts more than a jump that may come.
usage: python scripts/excitement.py <match_id>       (plots logs/excitement_<match_id>.png and prints the tie check)
"""
import sys

import numpy as np

import sister

W_SUSPENSE = 0.5      # surprise weighs more than suspense (Wimbledon: surprise effect 2-5x the suspense effect)
HALF_LIFE = 2.0       # plays: a jump's afterglow halves every 2 plays (~1-2 minutes of real game time)


def afterglow(x, half_life=HALF_LIFE):
    k = 0.5 ** (1.0 / half_life)
    out, cur = np.zeros(len(x)), 0.0
    for i, v in enumerate(x):
        cur = max(v, cur * k)          # a new jump replaces a smaller fading one; it doesn't stack on it
        out[i] = cur
    return out


def curve(gid):
    """-> {surprise, suspense, afterglow, excitement}: arrays over the game's steps (sister indexing)."""
    wp = np.asarray(sister.curve(gid)["wp"], float)
    surprise = np.abs(np.diff(wp, prepend=wp[0]))
    suspense = sister.nfl_suspense(gid)
    glow = afterglow(surprise)
    return {"surprise": surprise, "suspense": suspense, "afterglow": glow, "excitement": glow + W_SUSPENSE * suspense}


HALF_LIFE_S = 3.0     # in a story cut: a moment's afterglow halves every 3 s of the cut (beats are 4-12 s apart)


def cut_curve(gid, beats, e, length, rate=10):
    """The excitement a viewer feels along a story cut (not the whole game): beats [{step, show?}] with their results
    at cut seconds e[]. Each result is a spike of its own surprise; a shown play before it (show) spikes on the way;
    between spikes the curve fades (HALF_LIFE_S) onto the suspense of the play on screen (the state before the beat's
    play); after the last result the game is over, so it fades to nothing.  -> (t[], curve[]) at `rate` per second."""
    c = curve(gid)
    t = np.arange(int(round(length * rate)) + 1) / rate
    floor, spikes, prev = np.zeros(len(t)), [], 0.0
    for b, eb in zip(beats, e):
        shows = [s for s in b.get("show", []) if s != b["step"]]
        for i, s in enumerate(shows):
            spikes.append((eb - (eb - prev) * (len(shows) - i) / (len(shows) + 1), c["surprise"][s]))
        spikes.append((eb, c["surprise"][b["step"]]))
        floor[(t >= prev) & (t <= eb)] = W_SUSPENSE * c["suspense"][b["step"] - 1]
        prev = eb
    k = np.log(2) / HALF_LIFE_S
    glow = np.zeros(len(t))
    for ts, h in spikes:
        m = t >= ts
        glow[m] = np.maximum(glow[m], h * np.exp(-k * (t[m] - ts)))
    return t, glow + floor


def ties(gid):
    """Steps whose play tied the score (the scoreboard goes level from not level)."""
    sb = sister.curve(gid)["sb"]
    h, a = np.asarray(sb["h"]), np.asarray(sb["a"])
    return [s for s in range(1, len(h)) if h[s] == a[s] and h[s - 1] != a[s - 1]]


def report(gid, out_png=None):
    c, full = curve(gid), sister.curve(gid)
    old = np.asarray(full["tension"], float)
    mm = lambda v: (v - v.min()) / max(1e-9, np.ptp(v))
    new_n, old_n = mm(c["excitement"]), mm(old)
    lines = ["tie check (0-1 scale): value on the tying play -> mean of the next 3 plays (a drop = the held breath)"]
    for s in ties(gid):
        nxt = slice(s + 1, min(len(old), s + 4))
        lines.append(f"  step {s:3d} {full['sb']['a'][s]}-{full['sb']['h'][s]}:  old tension {old_n[s]:.2f} -> "
                     f"{old_n[nxt].mean():.2f}   new excitement {new_n[s]:.2f} -> {new_n[nxt].mean():.2f}")
    if out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
        x = np.arange(len(old))
        ax[0].plot(x, full["wp"], color="#444"); ax[0].set_ylabel("home WP"); ax[0].axhline(0.5, color="#bbb", lw=0.8)
        ax[1].plot(x, old_n, color="#3a7bd5"); ax[1].set_ylabel("old tension")
        ax[2].plot(x, new_n, color="#e8590c", label="excitement")
        ax[2].plot(x, mm(c["excitement"]) * 0 + W_SUSPENSE * c["suspense"] / max(1e-9, np.ptp(c["excitement"])),
                   color="#999", lw=1, ls="--", label="suspense part")
        ax[2].set_ylabel("new excitement"); ax[2].legend(loc="upper left", fontsize=8)
        for a in ax:
            for s in ties(gid):
                a.axvline(s, color="#2b8a3e", lw=0.9, alpha=0.6)
        ax[0].set_title(f"{full['title']}  ·  green = plays that tied the game")
        ax[2].set_xlabel("play (step)")
        fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
        lines.append(f"plot: {out_png}")
    return "\n".join(lines)


if __name__ == "__main__":
    from common import ROOT
    gid = sys.argv[1] if len(sys.argv) > 1 else "nfl_2014_06_CAR_CIN"
    print(report(gid, ROOT / "logs" / f"excitement_{gid}.png"))
