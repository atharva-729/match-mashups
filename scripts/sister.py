"""Thin adapter to the sister repo (which-match-is-your-song, local folder from config.yaml MATCH_REPO).

This is the ONLY module that imports from, or reads paths in, the sister repo. Everything else calls it.
Nothing here writes to the sister repo.

Step conventions (the sister repo's, used everywhere in this project):
  IPL  step 0 = pre-match, step k = state after delivery row k-1 (Cricsheet parse order), last step = outcome.
       sb = {i, r, w, b}: innings, runs, wickets, legal balls after each step.
  NFL  step p = state before play row p (home_wp not null & play_type not null, sorted by play_id);
       the result of play p lands on step p+1; last step = outcome.
       sb = {q, c, h, a}: quarter and clock (s) at play p's snap, home/away score before play p.
  So in both leagues: the footage of event row r belongs to step r + 1 (where its swing shows on the page).
"""
import functools, json, math, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    p = ROOT / "config.yaml"
    if not p.exists():
        p = ROOT / "config.example.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


CFG = _cfg()
MATCH_REPO = (ROOT / Path(CFG["MATCH_REPO"]).expanduser()).resolve()
if str(MATCH_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(MATCH_REPO / "scripts"))

# --- the sister repo's own code ---
import match as _match                      # scripts/match.py: resample, zscore, normalise, dtw_path, N, W
import song_energy as _song_energy          # scripts/song_energy.py: energy_curve, load_mono (Python twin of core.js)

N, W = _match.N, _match.W
normalise, resample, dtw_path = _match.normalise, _match.resample, _match.dtw_path
energy_curve, load_mono = _song_energy.energy_curve, _song_energy.load_mono


@functools.lru_cache(None)
def _build_nfl():
    import build_nfl                        # clean_desc (play text cleaner used for the page's moments)
    return build_nfl


@functools.lru_cache(None)
def _build_ipl():
    import build_ipl                        # ball_label, is_wicket (moment text used on the page)
    return build_ipl


def clean_desc(d, limit=140):
    return _build_nfl().clean_desc(d, limit)


def ball_label(row):
    return _build_ipl().ball_label(row)


# --- site data: what the live page uses ---
def game_json(gid):
    return json.loads((MATCH_REPO / "site/games" / f"{gid}.json").read_text(encoding="utf-8"))


@functools.lru_cache(None)
def _index():
    idx = json.loads((MATCH_REPO / "site/index.json").read_text(encoding="utf-8"))
    rows = [dict(zip(idx["fields"], r)) for r in idx["rows"]]
    Q = np.fromfile(MATCH_REPO / "site/index.bin", dtype=np.int8).reshape(-1, N)
    return idx, rows, Q


def index_curve(gid):
    """The 200-point normalised tension curve exactly as the page's curveOf(i) dequantises it."""
    idx, rows, Q = _index()
    i = next(k for k, r in enumerate(rows) if r["id"] == gid)
    return [float(v) / idx["scale"] for v in Q[i]]


def demo(key):
    d = json.loads((MATCH_REPO / "site/demo/demos.json").read_text(encoding="utf-8"))
    return next(x for x in d if x["key"] == key)


def demo_audio(key):
    return MATCH_REPO / "site/demo" / f"{key}.mp3"


def page_moment_times(demo_key, gid, steps):
    """Song times of `steps` computed by the page's own JS (core.js dtwPath + index.html's mapping), via node."""
    import subprocess
    d = demo(demo_key)
    q = {"song": d["curve"], "game": index_curve(gid), "t0": d["t0"], "t1": d["t1"],
         "n_steps": curve(gid)["n_steps"], "steps": list(steps)}
    r = subprocess.run(["node", str(ROOT / "scripts/page_moments.js"), str(MATCH_REPO / "site/core.js")],
                       input=json.dumps(q), capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


# --- full-resolution curves (data/curves/{league}.pkl) ---
@functools.lru_cache(None)
def _curves(league):
    with open(MATCH_REPO / "data/curves" / f"{league.lower()}.pkl", "rb") as f:
        return {c["id"]: c for c in pickle.load(f)}


def curve(gid):
    """Full-resolution dict: wp[], tension[], moments, sb, n_steps, ..."""
    return _curves("IPL" if gid.startswith("ipl_") else "NFL")[gid]


# --- event rows, in the sister repo's indexing (row r -> step r + 1) ---
NFL_COLS = ["game_id", "play_id", "home_wp", "play_type", "qtr", "time", "desc", "down", "ydstogo",
            "total_home_score", "total_away_score", "posteam", "touchdown", "interception", "fumble_lost",
            "field_goal_result", "safety", "sp", "game_seconds_remaining"]


def nfl_plays(gid):
    """Plays of one game, filtered and ordered exactly like build_nfl.load()."""
    game_id = gid.removeprefix("nfl_")
    season = int(game_id[:4])
    d = pd.read_parquet(MATCH_REPO / "data/raw/nfl" / f"play_by_play_{season}.parquet", columns=NFL_COLS,
                        filters=[("game_id", "==", game_id)])
    d = d[d.home_wp.notna() & d.play_type.notna()].sort_values("play_id", kind="stable").reset_index(drop=True)
    assert len(d) + 1 == curve(gid)["n_steps"], "NFL play indexing drifted from the sister repo"
    return d


def nfl_suspense(gid):
    """Suspense per step of an NFL game: the expected |dWP| of the next play given the game state (the sister repo's
    own leverage model, build_nfl.fit_leverage: LightGBM on WP distance from 50%, time, quarter, down, distance, field
    position, margin), in win-probability units; the appended outcome step gets 0. Fitted once on every season the
    sister repo holds, the same data its page uses; cached per game in work/_excitement/nfl_suspense.npz."""
    from common import ROOT
    cache = ROOT / "work" / "_excitement" / "nfl_suspense.npz"
    gkey = gid.removeprefix("nfl_")
    if not cache.exists():
        bn = _build_nfl()
        pbp = bn.load()
        lev, _ = bn.fit_leverage(pbp)
        cache.parent.mkdir(parents=True, exist_ok=True)
        pbp = pbp.assign(lev=lev)
        np.savez_compressed(cache, **{k: g.lev.to_numpy(np.float32) for k, g in pbp.groupby("game_id", sort=False)})
    lev = np.load(cache)[gkey].astype(float)
    assert len(lev) + 1 == curve(gid)["n_steps"], "NFL play indexing drifted from the sister repo"
    return np.append(lev, 0.0)


def ipl_balls(gid):
    """Deliveries of one match, in the order build_ipl used (parse order within the match)."""
    mid = gid.removeprefix("ipl_")
    b = pd.read_parquet(MATCH_REPO / "data/raw/t20_balls.parquet", filters=[("match_id", "==", mid)]).reset_index(drop=True)
    assert len(b) + 2 == curve(gid)["n_steps"], "IPL ball indexing drifted from the sister repo"
    return b


def events(gid):
    """One row per step 1..n-2 with the event that produced it: {step, desc, label, kind flags}."""
    out = []
    if gid.startswith("ipl_"):
        bi = _build_ipl()
        for r, row in enumerate(ipl_balls(gid).itertuples()):
            lab = bi.ball_label(row)
            out.append({"step": r + 1, "row": r, "clock": f"{row.innings}-{row.legal // 6}.{row.legal % 6}",
                        "label": f"{'1st' if row.innings == 1 else '2nd'} inns {row.over}.{row.ball}",
                        "desc": f"{row.bowler} to {row.batter}: {lab} ({row.runs}/{row.wickets})",
                        "wicket": bool(bi.is_wicket(row)), "six": int(row.batter_runs) == 6,
                        "four": int(row.batter_runs) == 4})
    else:
        p = nfl_plays(gid)
        for r, row in enumerate(p.itertuples()):
            q = "OT" if row.qtr >= 5 else f"Q{int(row.qtr)}"
            out.append({"step": r + 1, "row": r, "clock": f"{q} {row.time}", "label": f"{q} {row.time}",
                        "desc": clean_desc(row.desc), "td": row.touchdown == 1,
                        "turnover": row.interception == 1 or row.fumble_lost == 1,
                        "fg": row.field_goal_result == "made", "play_type": row.play_type})
    return out


if __name__ == "__main__":
    # smoke test (Checkpoint 0)
    print("MATCH_REPO", MATCH_REPO)
    for gid in ("ipl_1304103", "nfl_2014_13_CLE_BUF"):
        g, c = game_json(gid), curve(gid)
        ev = events(gid)
        print(gid, "|", g["title"], "| n_steps", c["n_steps"], "| tension", len(c["tension"]),
              "| events", len(ev), "| index curve", len(index_curve(gid)))
    a, b = normalise(demo("mountain-king")["curve"]), index_curve("ipl_1304103")
    dist, path = dtw_path(a, b)
    print("matcher dtw_path ok: dist", round(dist, 4), "path len", len(path))
