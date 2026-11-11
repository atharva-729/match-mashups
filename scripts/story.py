"""Story alignment: an alternative to the page's tension curve, for the mashup only.

Tension measures how undecided a game is. That hides some stories: in a collapsing chase every wicket makes the game
LESS undecided, so the biggest moments sit on falling tension. The story curve instead measures what happens:

  importance(step) = league weights for the event that produced the step (IPL: wicket, six, four; NFL: TD, turnover,
                     FG) + the size of its win-probability swing.
  story time       = each step lasts 1 + STRETCH * importance, so a wicket ball gets several times the room of a dot
                     ball and a dense collapse is spread out instead of crammed.
  story level      = importance smoothed over story time (a Gaussian), i.e. "how much is happening around here".

The level curve is matched to the song with the sister repo's own matcher (normalise + banded DTW), exactly as the
page matches tension; only the game curve and its time axis differ.
"""
import numpy as np

import sister

N = sister.N
WEIGHTS = {"IPL": {"wicket": 1.0, "six": 0.6, "four": 0.15},
           "NFL": {"td": 1.0, "turnover": 0.9, "fg": 0.5}}
SWING_FULL = {"IPL": 0.08, "NFL": 0.15}     # a win-probability swing this big counts as a full event
SWING_W = 0.6
STRETCH = {"IPL": 4.0, "NFL": 3.0}
SIGMA = 0.025                                # smoothing, as a fraction of the story-time axis


def importance(gid, league):
    c = sister.curve(gid)
    n, wp = c["n_steps"], np.asarray(c["wp"], float)
    imp = np.zeros(n)
    for e in sister.events(gid):
        s = e["step"]
        imp[s] += max([w for k, w in WEIGHTS[league].items() if e.get(k)], default=0.0)
    swing = np.abs(np.diff(wp, prepend=wp[0]))
    imp += SWING_W * np.minimum(1.0, swing / SWING_FULL[league])
    imp[0] = imp[-1] = 0.0
    return imp


def story(gid, league):
    """-> (tau[n_steps] in [0, 1]: story time of each step, level[N]: game curve on a uniform story-time grid)."""
    imp = importance(gid, league)
    dur = 1.0 + STRETCH[league] * imp
    tau = np.concatenate([[0.0], np.cumsum(dur[1:])])
    tau /= tau[-1]
    grid = np.linspace(0, 1, N)
    level = np.zeros(N)
    for s in np.nonzero(imp)[0]:
        level += imp[s] * np.exp(-0.5 * ((grid - tau[s]) / SIGMA) ** 2)
    return tau, level, imp
