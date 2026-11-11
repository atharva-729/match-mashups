"""Phase 3: cut the footage into shots and tag each shot with its play / ball, from the on-screen scoreboard.

1. Shots: PySceneDetect ContentDetector (threshold 27), per part, in order.
2. Scoreboard: 2 frames a second, crop the confirmed box, upscale x3, grayscale (+ adaptive threshold if that
   reads better on a 100-frame sample), easyocr with an allowlist. Raw OCR is cached in work/<job>/ocr_<part>.json.
3. Parse -> a scoreboard state in the sister repo's indexing:
     IPL  state s = sb index whose (innings, runs, wickets, legal balls) matches the bar ("GT 139-4 19.4").
          The bar updates AFTER a ball, so the ball that produced state s is event step s.
     NFL  state p = the play whose quarter matches and whose snap clock is the nearest at or above the bug's
          clock (score and down & distance as tie-breakers). Its result is event step p + 1.
4. Per shot:
   IPL  driven by delivery views (bowler's-end camera: pitch strip in the centre, grass both sides, found at 4 fps).
        The bar during a delivery still shows the state before the ball, so a delivery seen at state X is the ball
        of event step X + 1. Shots are split at each delivery; the delivery shot and the next 6 s are live, later
        shots of the same ball (replays, name card, walk back) are its replay / filler. The bar is read only in its
        left third (team, score, overs), where the bowler's figures can't be mistaken for the score.
   NFL  first readable play in the shot -> its result step; a non-decreasing (LIS) pass over live shots, then
        live / replay / filler / junk.
Writes work/<job>/shots.json, clips.json, contact_sheet.jpg.
usage: python scripts/tag_shots.py inputs/<job> [--reocr]
"""
import json, random, re, subprocess, sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import sister
from common import job_from_argv, log, stamp

FPS_OCR = 2.0
ALLOW = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZv-/:.& "


# ---------------------------------------------------------------- shots
def detect_shots(job, part):
    cache = job.wpath(f"shots_{part.stem}.json")
    if cache.exists():
        return json.loads(cache.read_text())
    from scenedetect import detect, ContentDetector
    scenes = detect(str(part), ContentDetector(threshold=27))
    shots = [[round(a.get_seconds(), 3), round(b.get_seconds(), 3)] for a, b in scenes]
    cache.write_text(json.dumps(shots))
    return shots


# ---------------------------------------------------------------- delivery shots (IPL)
FPS_VIS = 4.0


def delivery_features(job, part):
    """Per frame (4 fps): how much the picture looks like the bowler's-end delivery view: a light pitch strip down the
    centre with grass both sides. Columns: t, pitch share in the centre, grass left, grass right, grass in the centre."""
    cache = job.wpath(f"deliv_{part.stem}.json")
    if cache.exists():
        return json.loads(cache.read_text())
    cap = cv2.VideoCapture(str(part))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step, out, k = max(1, round(fps / FPS_VIS)), [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if k % step == 0:
            _, f = cap.retrieve()
            s = cv2.resize(f, (160, 90), interpolation=cv2.INTER_AREA)
            H, S, V = (cv2.cvtColor(s, cv2.COLOR_BGR2HSV).astype(int)[..., i] for i in range(3))
            green = (H >= 30) & (H <= 90) & (S > 50) & (V > 50)
            pitch = (~green) & (V > 100) & (S < 120) & (H >= 8) & (H <= 40)
            y0, y1 = 27, 72                               # above the score bar
            out.append([round(k / fps, 3), round(float(pitch[y0:y1, 68:92].mean()), 3),
                        round(float(green[y0:y1, 10:45].mean()), 3), round(float(green[y0:y1, 115:150].mean()), 3),
                        round(float(green[y0:y1, 68:92].mean()), 3)])
        k += 1
    cache.write_text(json.dumps(out))
    return out


def delivery_onsets(feats, min_gap=3.0):
    """Start times of delivery views: >= 2 consecutive frames (0.5 s) of pitch-in-the-middle, grass-both-sides."""
    hit = [(t, c >= 0.4 and l >= 0.6 and r >= 0.6 and cg <= 0.45) for t, c, l, r, cg in feats]
    onsets, run = [], []
    for t, h in hit + [(1e9, False)]:
        if h:
            run.append(t); continue
        if len(run) >= 2 and (not onsets or run[0] - onsets[-1] >= min_gap):
            onsets.append(run[0])
        run = []
    return onsets


# ---------------------------------------------------------------- OCR
def extract_crops(job, part, box):
    d = job.wpath(f"crops_{part.stem}")
    if d.exists() and any(d.iterdir()):
        return sorted(d.glob("*.png"))
    d.mkdir(exist_ok=True)
    x, y, w, h = box
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(part), "-vf", f"fps={FPS_OCR},crop={w}:{h}:{x}:{y}",
                    str(d / "%05d.png")], check=True)
    return sorted(d.glob("*.png"))


def prep(img, threshold):
    g = cv2.cvtColor(cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
    if threshold:
        g = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
    return g


def ocr(reader, img, threshold):
    res = reader.readtext(prep(img, threshold), allowlist=ALLOW, detail=1, paragraph=False)
    res.sort(key=lambda r: (round(min(p[1] for p in r[0]) / 40), min(p[0] for p in r[0])))   # rows, then left->right
    return [[t, round(float(c), 3), [int(min(p[0] for p in b)), int(min(p[1] for p in b))]] for b, t, c in res]


def bar_present(imgs, league):
    """Is the bar on screen? Pixels that sit at their median in most frames are the bar's background. IPL: judged on
    the left third only (team, score, overs), because the middle panel swaps between layouts ('NEXT TO BAT',
    'REVIEWS REMAINING', a speed chart) and would otherwise make the bar look absent."""
    gray = np.stack([cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs]).astype(np.int16)
    if league == "IPL":
        gray = gray[:, :, : int(gray.shape[2] * IPLParser.ZONE)]
    med = np.median(gray, axis=0)
    close = np.abs(gray - med) < 28
    mask = close.mean(axis=0) > 0.45
    use_mask = mask.mean() > 0.10
    return [(not use_mask) or bool(close[k][mask].mean() > 0.55) for k in range(len(imgs))]


def topup_ocr(job, part, box, out, cache):
    """OCR frames the zone-based presence check finds but the cached pass skipped (cached results stay as they are)."""
    import multiprocessing as mp, os, time
    crops = extract_crops(job, part, box)
    present = bar_present([cv2.imread(str(f)) for f in crops], job.league)
    todo = [k for k in range(len(crops)) if present[k] and not out["frames"][k]["tokens"]]
    t0 = time.time()
    if todo:
        th = out["preprocess"] == "adaptive"
        with mp.get_context("spawn").Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_worker_init) as pool:
            res = pool.map(_ocr_chunk, [([str(crops[k])], [True], th, 0) for k in todo], chunksize=1)
        for k, r in zip(todo, res):
            out["frames"][k]["tokens"] = r[0][0]
    out["topup"] = {"frames": len(todo), "seconds": round(time.time() - t0)}
    out["bar_on_screen"] = round(sum(present) / len(present), 3)
    print(f"  top-up OCR: {len(todo)} frames the first pass skipped, {time.time() - t0:.0f} s", flush=True)
    cache.write_text(json.dumps(out))
    return out


def run_ocr(job, part, box, parser, reocr):
    cache = job.wpath(f"ocr_{part.stem}.json")
    if cache.exists() and not reocr:
        out = json.loads(cache.read_text())
        if job.league == "IPL" and "topup" not in out:
            out = topup_ocr(job, part, box, out, cache)
        return out
    crops = extract_crops(job, part, box)
    imgs = [cv2.imread(str(f)) for f in crops]
    present = bar_present(imgs, job.league)
    # OCR in parallel worker processes (one thread each; all cores). The README's adaptive threshold first; its success
    # rate on a 100-frame sample of bar-on-screen frames decides whether to redo with plain gray (Tesseract isn't
    # installed; easyocr's second mode is the fallback).
    import multiprocessing as mp, os, time
    max_reuse = 20 if job.league == "IPL" else 2         # the NFL clock ticks every second
    nw = max(1, (os.cpu_count() or 2) - 1)
    bounds = np.linspace(0, len(crops), nw * 3 + 1).astype(int)
    rng = random.Random(0)
    on = [k for k in range(len(imgs)) if present[k]]
    sample = sorted(rng.sample(on, min(100, len(on))))
    rates = {}
    with mp.get_context("spawn").Pool(nw, initializer=_worker_init) as pool:
        # 1. the README's 100-frame sample, per preprocessing mode (adaptive first; gray only if adaptive < 60%)
        for th in (True, False):
            t0 = time.time()
            res = pool.map(_ocr_chunk, [([str(crops[i])], [True], th, 0) for i in sample], chunksize=1)
            ok = sum(parser(r[0][0]) is not None for r in res)
            rates["adaptive" if th else "gray"] = round(ok / len(sample), 3)
            print(f"  sample ({'adaptive' if th else 'gray'}): {ok / len(sample):.0%} parsed, {time.time() - t0:.0f} s", flush=True)
            if ok / len(sample) >= 0.6:
                break
        th = rates["adaptive"] >= rates.get("gray", 0)
        # 2. one full pass with the better mode
        t0 = time.time()
        tasks = [([str(f) for f in crops[a:b]], present[a:b], th, max_reuse) for a, b in zip(bounds, bounds[1:]) if b > a]
        res = pool.map(_ocr_chunk, tasks, chunksize=1)
    toks = [t for r in res for t in r[0]]
    n_ocr = sum(r[1] for r in res)
    print(f"  full pass ({'adaptive' if th else 'gray'}): {n_ocr} OCR calls on {nw} workers in {time.time() - t0:.0f} s", flush=True)
    out = {"engine": "easyocr", "preprocess": "adaptive" if th else "gray", "sample_rates": rates,
           "bar_on_screen": round(sum(present) / len(present), 3), "ocr_calls": n_ocr, "topup": {"frames": 0},
           "frames": [{"t": round(k / FPS_OCR, 2), "tokens": toks[k]} for k in range(len(crops))]}
    cache.write_text(json.dumps(out))
    return out


_READER = None


def _worker_init():
    global _READER
    import torch, easyocr
    torch.set_num_threads(1)
    _READER = easyocr.Reader(["en"], gpu=False, verbose=False)


def _ocr_chunk(task):
    """One contiguous run of crops: skip frames without the bar, reuse the reading while the bar is unchanged."""
    paths, present, th, max_reuse = task
    out, last, n = [], None, 0
    for k, p in enumerate(paths):
        if not present[k]:
            out.append([]); continue
        img = cv2.imread(p)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
        if last is not None and k - last[0] <= max_reuse and (np.abs(g - last[2]) > 40).mean() < 0.004:
            out.append(last[1]); continue
        toks = ocr(_READER, img, th); n += 1
        last = (k, toks, g)
        out.append(toks)
    return out, n


# ---------------------------------------------------------------- parsing
def text_of(tokens):
    return " ".join(t[0] for t in tokens).upper().replace("O", "0") if tokens else ""


class IPLParser:
    """'GT 139-4 19.4 V LSG' / 'LSG 67-6 11.5 V GT TARGET 145' -> sb state index.
    Only tokens in the bar's left third are read (team, score, overs, target): the batter and bowler panels to the
    right hold look-alike numbers (bowler figures '3-24', bowler overs '3.4') that otherwise get read as the score."""
    ZONE = 0.33

    def __init__(self, gid, box_w=None):
        self.zone_px = self.ZONE * 3 * box_w if box_w else None     # OCR runs on a x3 upscale of the crop
        g = sister.game_json(gid)
        self.sb, self.n, self.brk = g["sb"], g["n_steps"], g["break_step"]
        self.first, self.second = g["home"], g["away"]           # 'home' = batted first on the page
        self.states = [(self.sb["i"][s], self.sb["r"][s], self.sb["w"][s], self.sb["b"][s]) for s in range(self.n - 1)]
        self.states.append((2, 0, 0, 0))                         # virtual: innings 2 not started (-> brk - 1)
        self.target = (g.get("chase") or {}).get("target")
        self.eventful = {e["step"] for e in sister.events(gid) if e["wicket"] or e["six"] or e["four"]}

    def raw(self, tokens):
        if self.zone_px:
            tokens = [t for t in (tokens or []) if t[2][0] < self.zone_px]
        if not tokens:
            return None
        words = [t[0].upper().strip() for t in tokens]
        txt = " ".join(words)
        txt0 = txt.replace("O", "0")
        m = re.search(r"(?<![\d.])(\d{1,3})\s*[-/]\s*(\d{1,2})(?![\d.])", txt0)
        if not m:
            return None
        runs, wk = int(m.group(1)), int(m.group(2))
        rest = txt0[m.end():]
        o = re.search(r"(?<!\d)(\d{1,2})\s*[.,]\s*([0-5])(?!\d)", rest) or re.search(r"(?<!\d)(\d{1,2})\s*[.,]\s*([0-5])(?!\d)", txt0)
        if o:
            ov, bl = int(o.group(1)), int(o.group(2))
        else:   # the bar shows a completed over as a bare "6", right after the score
            o = re.match(r"\s*(\d{1,2})(?![\d.,-])", rest)
            if o:
                ov, bl = int(o.group(1)), 0
            elif runs == 0 and wk == 0:                    # "LSG 0-0 v GT TARGET 145": innings not started
                ov, bl = 0, 0
            else:
                return None
        if wk > 10 or ov > 20:
            return None
        legal = ov * 6 + bl
        head = txt[:m.start()]
        inn = None
        if self.second in head.split() or "TARGET" in txt or "NEED" in txt:
            inn = 2
        elif self.first in head.split():
            inn = 1
        return {"inn": inn, "r": runs, "w": wk, "b": legal, "text": txt}

    def __call__(self, tokens):
        rd = self.raw(tokens)
        if rd is None:
            return None
        best, bc = None, 1e9
        for s, (i, r, w, b) in enumerate(self.states):
            if rd["inn"] is not None and i != rd["inn"]:
                continue
            c = abs(r - rd["r"]) + 6 * abs(w - rd["w"]) + 0.3 * abs(b - rd["b"])   # overs are the smallest digits: trust them least
            if c < bc:
                best, bc = s, c
        if best is None or bc > 6:
            return None
        if best == len(self.states) - 1:
            best = self.brk - 1
        i, r, w, b = self.states[best]
        return {"state": best, "cost": round(bc, 2), "read": f"{rd['r']}-{rd['w']} {rd['b'] // 6}.{rd['b'] % 6}",
                "label": f"inn{i} {r}/{w} {b // 6}.{b % 6}"}

    @staticmethod
    def event_step(state):
        return state


class NFLParser:
    """'1ST & 10 CLE 3 BUF 0 3RD 8:07' -> play row p (state p; its result is event step p + 1)."""
    ORD = {"1ST": 1, "2ND": 2, "3RD": 3, "4TH": 4, "0T": 5, "OT": 5}

    def __init__(self, gid):
        g = sister.game_json(gid)
        self.sb, self.n = g["sb"], g["n_steps"]
        self.home, self.away = g["home"], g["away"]
        plays = sister.nfl_plays(gid)
        self.down = plays.down.fillna(0).astype(int).tolist()
        self.togo = plays.ydstogo.fillna(0).astype(int).tolist()
        # with the clock stopped, a TD, its extra point and the kickoff share one snap clock; highlights rarely show
        # the last two (or timeouts), so they only win a tie when nothing else fits
        self.minor = [pt in ("extra_point", "kickoff", "no_play") for pt in plays.play_type]

    def raw(self, tokens):
        if not tokens:
            return None
        # common misreads of the ordinals (more of them at low resolution: '3R0', 'BR0', 'ZN0', 'I5T', '7ST')
        fix = lambda s: re.sub(r"\b(IST|LST|1S7|I5T|15T|7ST|1ST)\b", "1ST", re.sub(r"\b(ZND|ZN0|2N0|2ND)\b", "2ND",
                       re.sub(r"\b(ATH|4TN|ITH|4TH)\b", "4TH", re.sub(r"\b[3B8]R[O0DV]\b", "3RD", s))))
        whole = [fix(t[0].upper().strip()) for t in tokens]
        txt = " ".join(whole)
        # OCR sometimes glues the quarter and the clock into one token ('3RD  14:19'): look at the parts as well
        words = whole + [w for t in whole if re.search(r"\s", t) for w in t.split()]
        # OCR comes back in a jumbled order, so read token by token: the clock is its own token, the colon is
        # sometimes read as 3, 4, 8 or H ("7459" = 7:59, "12830" = 12:30, "1H02" = 1:02), maybe with a stray digit after
        mm = ss = None
        for w in words:
            m = (re.fullmatch(r"(\d{1,2})\s*[:.H]\s*(\d{2})\d?(\s.*)?", w) or re.fullmatch(r"(\d{1,2})[348](\d{2})\d?", w)
                 or re.fullmatch(r"(\d)[348](\d{2})\d?", w))
            if m and int(m.group(1)) <= 15 and int(m.group(2)) <= 59:
                mm, ss = int(m.group(1)), int(m.group(2))
                break
        if mm is None:
            return None
        q = None                                   # the quarter is an ordinal standing alone (not a down "1ST & 10")
        for w in whole:
            if w in self.ORD:
                q = self.ORD[w]
        if q is None:                              # glued tokens: an ordinal not followed by '&' ('3RD  14:19')
            for t in whole:
                parts = t.split()
                for a, b in zip(parts, parts[1:] + [""]):
                    if a in self.ORD and not b.startswith("&"):
                        q = self.ORD[a]
        dn = re.search(r"(1ST|2ND|3RD|4TH)\s*&\s*(\d{1,2}|G[O0]AL)", txt)
        hs = re.search(rf"\b{self.home}\s*(\d{{1,2}})\b", txt)
        as_ = re.search(rf"\b{self.away}\s*(\d{{1,2}})\b", txt)
        return {"q": q, "c": mm * 60 + ss, "down": self.ORD.get(dn.group(1)) if dn else None,
                "togo": (0 if dn.group(2).startswith("G") else int(dn.group(2))) if dn else None,
                "h": int(hs.group(1)) if hs else None, "a": int(as_.group(1)) if as_ else None, "text": txt}

    def __call__(self, tokens):
        rd = self.raw(tokens)
        if rd is None or rd["q"] is None:
            return None
        best, bc = None, 1e9
        for p in range(self.n - 1):
            if self.sb["q"][p] != rd["q"]:
                continue
            dc = self.sb["c"][p] - rd["c"]
            if dc >= 0:
                c = dc / 15.0                  # play p in progress (its clock runs down from its snap)
                if dc > 12:                    # no play runs this long: more likely the next play's pre-snap clock
                    c += 0.1 * (dc - 12)
            elif dc >= -40 and rd["down"] is not None and self.down[p] == rd["down"] and self.togo[p] == rd["togo"]:
                c = -dc / 15.0 + 0.3           # clock running before play p's snap: the bug already shows p's down
            elif dc >= -40 and rd["down"] is None:
                c = -dc / 15.0 + 0.8           # same, but the down wasn't legible
            else:
                continue
            if rd["h"] is not None and rd["h"] != self.sb["h"][p]:
                c += 2
            if rd["a"] is not None and rd["a"] != self.sb["a"][p]:
                c += 2
            if rd["down"] is not None and self.down[p] and (rd["down"] != self.down[p] or rd["togo"] != self.togo[p]):
                c += 1.0
            if self.minor[p]:
                c += 1.0
            if c < bc:
                best, bc = p, c
        if best is None or bc > 6:
            return None
        q, c = self.sb["q"][best], self.sb["c"][best]
        return {"state": best, "cost": round(bc, 2), "read": f"Q{rd['q']} {rd['c'] // 60}:{rd['c'] % 60:02d}",
                "label": f"Q{q} {c // 60}:{c % 60:02d}", "clock": rd["c"]}

    def event_step(self, state):
        """Play `state` -> the step its result lands on. A bug frozen on an extra point / kickoff / timeout is the
        aftermath (replay, celebration) of the real play before it, so it belongs to that play."""
        p = state
        while p > 0 and self.minor[p]:
            p -= 1
        return p + 1


# ---------------------------------------------------------------- shot logic
def lis_keep(vals, weights):
    """Indices of the max-weight non-decreasing subsequence."""
    n = len(vals)
    best, prev = [0.0] * n, [-1] * n
    for i in range(n):
        best[i] = weights[i]
        for j in range(i):
            if vals[j] <= vals[i] and best[j] + weights[i] > best[i]:
                best[i], prev[i] = best[j] + weights[i], j
    if not n:
        return set()
    i = int(np.argmax(best)); keep = set()
    while i != -1:
        keep.add(i); i = prev[i]
    return keep


def smooth_states(reads):
    """Drop single-frame OCR blips: keep readings that agree with a neighbour within 1 state."""
    out = []
    for k, (t, s) in enumerate(reads):
        nb = [reads[j][1] for j in (k - 1, k + 1) if 0 <= j < len(reads) and abs(reads[j][0] - t) <= 1.6]
        if not nb or any(abs(s - v) <= 1 for v in nb):
            out.append((t, s))
    return out


def ipl_runs(rows, parser):
    """Which ball does a shot show when the bar holds steady at state X: the reaction to ball X, or ball X+1 being
    bowled? Judged per run of consecutive shots at X (shots whose bar ticks inside them are already ball Y):
      - the bar ticked into X live (+1, from the shot before) -> the run opens with reactions to ball X;
        otherwise (a jump at a cut = a new highlight) the run is ball X+1 being bowled
      - the bar ticks to X+1 right after the run -> its last shot is ball X+1 being bowled
      - highlight editors show wickets and boundaries: if ball X was nothing special and ball X+1 is, it's X+1."""
    rd = [k for k, r in enumerate(rows) if r["reads"]]
    runs, cur = [], []
    for k in rd:
        r = rows[k]
        if r["change"]:
            if cur: runs.append(cur); cur = []
            continue
        if cur and rows[cur[-1]]["state"] != r["state"]:
            runs.append(cur); cur = []
        cur.append(k)
    if cur:
        runs.append(cur)
    for run in runs:
        X = rows[run[0]]["state"]
        before = [k for k in rd if k < run[0]]
        after = [k for k in rd if k > run[-1]]
        P = (rows[before[-1]]["Y"] if before else None)
        Y = (rows[after[0]]["X"] if after else None)
        live_into = P is not None and X - P == 1
        live_out = Y is not None and Y - X == 1
        if live_into and X not in parser.eventful and X + 1 in parser.eventful:
            live_into = False
        for n, k in enumerate(run):
            last = n == len(run) - 1
            if live_out and last and (len(run) > 1 or not live_into):
                rows[k]["step"] = X + 1
            else:
                rows[k]["step"] = X if live_into else X + 1


def tag_part(job, part, shots, frames, parser, league):
    for f in frames:
        f["p"] = parser(f["tokens"])
    reads = smooth_states([(f["t"], f["p"]["state"]) for f in frames if f["p"]])
    rows = []
    for si, (a, b) in enumerate(shots):
        rs = [s for t, s in reads if a <= t < b]
        rows.append({"part": part.name, "start": a, "end": b, "reads": rs})
    # event step per shot
    prev_state = None
    for k, r in enumerate(rows):
        rs = r["reads"]
        r["step"], r["event_t"], r["change"] = None, None, False
        if rs:
            med = int(np.median(rs))
            X, Y = int(rs[0]), int(rs[-1])     # first and last reading (single-frame blips are already smoothed out)
            if league == "IPL":
                r["X"], r["Y"] = X, Y
                if Y > X:                                          # bar ticked inside the shot: this is ball Y
                    r["step"], r["change"] = Y, True
                    tc = next(t for t, s in reads if r["start"] <= t < r["end"] and s >= Y)
                    r["event_t"] = max(r["start"], tc - 2.5)
                else:
                    r["event_t"] = r["start"] + 0.4 * (r["end"] - r["start"])
            else:
                first = int(np.median(rs[: max(1, min(3, len(rs)))]))
                r["step"] = parser.event_step(first)
                r["event_t"] = r["start"]                          # the snap (fallback)
                # the play's result: the last moment the game clock ticked down inside this shot (it stops on
                # incompletions, scores, turnovers, out of bounds)
                clk = [(f["t"], f["p"]["clock"]) for f in frames
                       if f["p"] and r["start"] <= f["t"] < r["end"] and "clock" in f["p"]]
                ticks = [t for (t0, c0), (t, c) in zip(clk, clk[1:]) if c < c0 and c0 - c <= 15]
                r["clock_stop"] = ticks[-1] if ticks else None
            r["state"] = med
            prev_state = int(rs[-1])
    if league == "IPL":
        ipl_runs(rows, parser)
    # non-decreasing pass over shots with readings
    idx = [k for k, r in enumerate(rows) if r["step"] is not None]
    keep = lis_keep([rows[k]["step"] for k in idx], [rows[k]["end"] - rows[k]["start"] for k in idx])
    for n, k in enumerate(idx):
        rows[k]["kind"] = "live" if n in keep else "backwards"
    # classify the rest
    live_t = [(r["start"], r["end"]) for r in rows if r.get("kind") == "live"]
    last_live = None
    for k, r in enumerate(rows):
        near = any(min(abs(r["start"] - b), abs(r["end"] - a), 0 if (a < r["end"] and b > r["start"]) else 1e9) <= 20
                   for a, b in live_t)
        if r.get("kind") == "live":
            last_live = r
            continue
        if not near or last_live is None:
            r["kind"], r["step"] = "junk", None
            continue
        if r.get("kind") == "backwards" or rows[k - 1] is last_live or rows[k - 1].get("kind") == "replay":
            r["kind"] = "replay" if (r.get("kind") == "backwards" or r["end"] - r["start"] >= 2.0) else "filler"
        else:
            r["kind"] = "filler"
        r["step"] = last_live["step"]
        r["event_t"] = r["start"] + 0.4 * (r["end"] - r["start"])
    # a run of 'replay' only directly follows its live shot
    for k in range(1, len(rows)):
        if rows[k]["kind"] == "replay" and rows[k - 1]["kind"] not in ("live", "replay"):
            rows[k]["kind"] = "filler"
    # junk: no valid reading within 20 s either side (intro, outro, studio)
    return rows


HIT = 1.2          # the bat meets the ball ~1.2 s after a delivery view starts (run-up already under way at the cut)
LIVE_FOLLOW = 6.0  # shots starting within this long after the delivery are the live action (the catch, the chase)
REPLAY_SPAN = 15.0 # later shots of the same ball within this long are its replays / reactions / name card


def tag_part_ipl(job, part, shots, frames, parser):
    """IPL, driven by delivery views. During a delivery the bar still shows the state BEFORE the ball, so a delivery
    seen with the bar at state X is ball row X, whose event step is X + 1. Every later shot (the catch, replays,
    the batter's name card, the walk back) belongs to that ball until the next delivery."""
    for f in frames:
        f["p"] = parser(f["tokens"])
    reads = smooth_states([(f["t"], f["p"]["state"]) for f in frames if f["p"]])
    onsets = delivery_onsets(delivery_features(job, part))

    def ball_of(o, o_next):
        now = [s for t, s in reads if o <= t <= o + 2.0]            # the previous ball's update can lag into the cut
        if now:
            return max(now) + 1, "bar"
        after = [(t, s) for t, s in reads if o + 2.0 < t < min(o_next, o + 12)]
        if after:                    # first reading after the delivery: still the state before it, or already its result?
            X = after[0][1]
            if any(s > X for _, s in after):                         # it ticks later on: X was the state before
                return X + 1, "bar after, ticks"
            return (X if X in parser.eventful else X + 1), "bar after"
        before = [s for t, s in reads if o - 3 <= t < o]             # only right before the cut (a jump is a new clip)
        if before:
            return max(before) + 1, "bar before"
        return None, None

    balls, last = [], 0
    for o, o_next in zip(onsets, onsets[1:] + [1e9]):
        b, how = ball_of(o, o_next)
        if b is None or b < last:
            continue
        balls.append({"o": o, "step": b, "how": how, "replay": b == last})
        last = b
    # split shots at delivery onsets so each delivery starts its own shot
    cuts = sorted({round(x, 3) for a, b in shots for x in (a, b)} | {round(d["o"], 3) for d in balls})
    pieces = [(a, b) for a, b in zip(cuts, cuts[1:]) if b - a > 0.05]
    rows = []
    for a, b in pieces:
        rs = [s for t, s in reads if a <= t < b]
        rows.append({"part": part.name, "start": a, "end": b, "reads": rs, "step": None, "event_t": None,
                     "kind": None, "state": int(np.median(rs)) if rs else None})
    has_info = [(t, 1) for t, _ in reads] + [(d["o"], 1) for d in balls]
    bi = 0
    cur = None
    for r in rows:
        while bi < len(balls) and balls[bi]["o"] <= r["start"] + 1e-3:
            cur = balls[bi]; bi += 1
        near = any(abs(t - r["start"]) <= 20 or abs(t - r["end"]) <= 20 or r["start"] <= t <= r["end"] for t, _ in has_info)
        if cur is None or not near:
            r["kind"] = "junk"
            continue
        r["step"] = cur["step"]
        dt = r["start"] - cur["o"]
        if abs(dt) < 1e-3:
            r["kind"] = "replay" if cur["replay"] else "live"
            r["event_t"] = min(r["end"], cur["o"] + HIT)
            r["delivery"] = cur["how"]
        elif dt < LIVE_FOLLOW and not cur["replay"]:
            r["kind"] = "live"
        elif dt < REPLAY_SPAN:
            r["kind"] = "replay"
        else:
            r["kind"] = "filler"
        if r["event_t"] is None:
            r["event_t"] = r["start"] + 0.4 * (r["end"] - r["start"])
    return rows


# ---------------------------------------------------------------- contact sheet
def contact_sheet(job, clips, ev):
    tagged = [c for c in clips if c["kind"] in ("live", "replay")]
    random.Random(1).shuffle(tagged)
    pick = sorted(tagged[:16], key=lambda c: (c["part"], c["start"]))
    tiles = []
    for c in pick:
        cap = cv2.VideoCapture(str(job.inp / c["part"]))
        cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * (c["start"] + 0.5 * (c["end"] - c["start"])))
        ok, fr = cap.read(); cap.release()
        fr = cv2.resize(fr, (480, 270))
        cap_h = 64
        tile = np.zeros((270 + cap_h, 480, 3), np.uint8); tile[:270] = fr
        e = ev.get(c["step"], {})
        lines = [f"{c['kind']} step {c['step']} | {e.get('label', '')} | bar {c.get('read', '-')}",
                 e.get("desc", "")[:62], e.get("desc", "")[62:124]]
        for li, s in enumerate(lines):
            cv2.putText(tile, s.encode("ascii", "replace").decode(), (6, 288 + 19 * li), cv2.FONT_HERSHEY_SIMPLEX,
                        0.46, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(job.wpath("contact_sheet.jpg")), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------------------------------------------------------------- main
def main():
    job = job_from_argv()
    reocr = "--reocr" in sys.argv
    boxes = job.load("box.json")
    clips, ocr_stats = [], {}
    for part in job.parts:
        box = boxes[part.name]["box"]
        parser = IPLParser(job.gid, box[2]) if job.league == "IPL" else NFLParser(job.gid)
        shots = detect_shots(job, part)
        print(part.name, len(shots), "shots", flush=True)
        o = run_ocr(job, part, box, parser, reocr)
        ok = sum(parser(f["tokens"]) is not None for f in o["frames"])
        ocr_stats[part.name] = {"preprocess": o["preprocess"], "sample_rates": o["sample_rates"],
                                "frames": len(o["frames"]), "parsed": ok}
        if job.league == "IPL":
            rows = tag_part_ipl(job, part, shots, o["frames"], parser)
        else:
            rows = tag_part(job, part, shots, o["frames"], parser, job.league)
        if part == job.parts[0]:      # pre-game footage before the first live shot: usable only as the cold open
            for r in rows:
                if r["kind"] == "live":
                    break
                if r["kind"] == "junk":
                    r["kind"], r["step"] = "intro", 0
        for r in rows:
            c = {"part": r["part"], "start": r["start"], "end": r["end"], "step": r["step"], "kind": r["kind"],
                 "event_t": round(r["event_t"], 2) if r.get("event_t") is not None else None,
                 "clock_stop": r.get("clock_stop"),
                 "conf": round(len(r["reads"]) / max(1, (r["end"] - r["start"]) * FPS_OCR), 2)}
            if r.get("state") is not None:
                c["state"] = r["state"]
                fr = [f for f in o["frames"] if r["start"] <= f["t"] < r["end"] and f["p"]]
                if fr:
                    c["read"] = Counter(f["p"]["read"] for f in fr).most_common(1)[0][0]
            clips.append(c)
    job.save("clips.json", clips)

    # ---- checkpoint 3 ----
    al = job.load("alignment.json")
    ev = {e["step"]: e for e in sister.events(job.gid)}
    contact_sheet(job, clips, ev)
    dur = lambda c: c["end"] - c["start"]
    nonjunk = sum(dur(c) for c in clips if c["kind"] not in ("junk", "intro"))
    lr = sum(dur(c) for c in clips if c["kind"] in ("live", "replay"))
    live_steps = {c["step"] for c in clips if c["kind"] == "live"}
    key_all = [s["i"] for s in al["steps"] if s["is_key"]]
    # a highlight package only covers part of the game: judge tagging on the key steps inside the footage's range
    lo, hi = (min(live_steps), max(live_steps)) if live_steps else (0, -1)
    key = [s for s in key_all if lo - 1 <= s <= hi + 1]
    key_hit = [s for s in key if s in live_steps]
    lv = [c["step"] for c in clips if c["kind"] == "live"]
    backwards = sum(1 for a, b in zip(lv, lv[1:]) if b < a)
    n = len(al["steps"])
    covered = {c["step"] for c in clips if c["kind"] in ("live", "replay")}
    kinds = Counter(c["kind"] for c in clips)
    tl = {k: round(sum(dur(c) for c in clips if c["kind"] == k), 1) for k in kinds}
    a_ok, b_ok, c_ok = lr / nonjunk >= 0.70, len(key_hit) / max(1, len(key)) >= 0.80, backwards == 0
    log("\n".join([
        f"### {stamp()} · Checkpoint 3 ({job.name}): {'PASS' if a_ok and b_ok and c_ok else 'FAIL'}",
        f"- OCR: {json.dumps(ocr_stats)}",
        f"- Shots {len(clips)} · by kind {dict(kinds)} · seconds by kind {tl}",
        f"- Live+replay share of non-junk time: {lr / nonjunk:.0%} (need ≥70%: {a_ok})",
        f"- Footage covers steps {lo}–{hi} of {n - 1}. Key steps in that range with a live shot: {len(key_hit)}/{len(key)} = "
        f"{len(key_hit) / max(1, len(key)):.0%} (need ≥80%: {b_ok}); missing: {[s for s in key if s not in live_steps]}; "
        f"outside the footage: {[s for s in key_all if s not in key]}",
        f"- Backwards steps among live shots: {backwards} (need 0: {c_ok})",
        f"- Coverage (steps with live/replay footage / total steps): {len(covered)}/{n} = {len(covered) / n:.0%}",
        f"- Contact sheet: `work/{job.name}/contact_sheet.jpg`"]))


if __name__ == "__main__":
    main()
