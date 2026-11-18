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
import json, os, random, re, subprocess, sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import sister
from common import job_from_argv, log, stamp

FPS_OCR = 2.0
STRIP_SAME = 0.03   # a strip counts as unchanged if < 3% of its pixels moved by > 40 (the NFL bug is see-through)
STRIP_SAME_FIELDS = 0.008   # the same for a style's small opaque fields (FOX 2016)
ALLOW = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZv-/:.& "


# ---------------------------------------------------------------- shots
def detect_shots(job, part):
    cache = job.wpath(f"shots_{part.stem}.json")
    if cache.exists():
        return json.loads(cache.read_text())
    from scenedetect import detect, ContentDetector
    scenes = detect(str(part), ContentDetector(threshold=27))
    # scenedetect counts frames and assumes a constant rate; a reel joined from downloaded sections can be variable
    # frame rate (avg 32.7 fps on a nominal 30), and then every cut drifts later through the reel. Map frame numbers
    # to the frames' real presentation times instead (packet pts, sorted: no decoding needed).
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                        "-of", "csv=p=0", str(part)], capture_output=True, text=True, check=True)
    pts = np.sort(np.array([float(v.split(",")[0]) for v in r.stdout.split() if v.split(",")[0] not in ("", "N/A")]))
    pts -= pts[0] if len(pts) else 0.0
    at = lambda fn: float(pts[min(fn, len(pts) - 1)]) if fn < len(pts) else float(pts[-1]) + (fn - len(pts) + 1) / 30
    shots = [[round(at(a.get_frames()), 3), round(at(b.get_frames()), 3)] for a, b in scenes]
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


# Score-bug styles (v2 Phase 7.5): one shared OCR + parser core, one small config per broadcast style. A style lists
# the bug's text fields as rectangles (fractions of the box: x0, y0, x1, y1) and how the parser reads them.
#   cbs2014 (default)  one line '3RD & 6 | CLE 3 | BUF 0 | 2ND 13:02': cut at the quiet column before the clock
#   fox2016            a small stacked box: team names / scores (away left, home right) / 'quarter clock' / 'down'
BUG_STYLES = {
    # CBS 2012: top bar '2ND & 10 | BAL 0 | SD 7 | 2ND 3:53' (see-through; fields give the strict reuse test)
    "cbs2012": {"stacked": True,
                "fields": [[0.470, 0.0, 0.540, 1.0], [0.710, 0.0, 0.775, 1.0], [0.785, 0.0, 1.0, 1.0], [0.090, 0.0, 0.290, 1.0]]},
    # CBS 2021: bottom bar 'CBS | LAC 10 | WSH 6 | 2ND 8:57 | 40 | 2ND & 10' (team codes differ from nflverse: WSH)
    "cbs2021": {"stacked": True,
                "fields": [[0.240, 0.0, 0.300, 1.0], [0.478, 0.0, 0.535, 1.0], [0.540, 0.0, 0.720, 1.0], [0.800, 0.0, 0.980, 1.0]]},
    # ESPN 2024 (MNF): one bottom bar, team logos instead of names: '[logo] 7-4  14 | 13  7-3 [logo] | 3RD 13:35 | 40 | 3RD & 9'
    "espn2024": {"stacked": True,
                 "fields": [[0.012, 0.0, 0.084, 1.0],     # away score
                            [0.114, 0.0, 0.199, 1.0],     # home score
                            [0.464, 0.0, 0.657, 1.0],     # quarter + clock
                            [0.753, 0.0, 0.994, 1.0]]},   # down & distance
    "fox2016": {"stacked": True,
                "fields": [[0.0, 0.24, 0.5, 0.47],      # away score
                           [0.5, 0.24, 1.0, 0.47],      # home score
                           [0.0, 0.48, 1.0, 0.67],      # quarter + clock
                           [0.0, 0.67, 1.0, 0.97]]},    # down ('2ND DOWN' / '2ND & 10')
}


def layout(job, crops):
    """What strips() cuts the bug into for this job: the style's field rectangles, else (CBS-like line) the split
    column found from the frames."""
    st = BUG_STYLES.get(job.cfg.get("bug_style", ""))
    return st["fields"] if st else strip_split(crops, job.league)


def strip_split(crops, league):
    """NFL: the column where the bug's fast-changing right end (game clock) begins: the quietest column (least text
    edge, averaged over frames) in 70-90% of the width, i.e. a gap between two fields. None for IPL."""
    if league != "NFL" or not crops:
        return None
    fs = crops[:: max(1, len(crops) // 300)]
    imgs = np.stack([cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY) for f in fs]).astype(float)
    gx = np.convolve(np.abs(np.diff(imgs, axis=2)).mean(axis=(0, 1)), np.ones(3) / 3, mode="same")
    w = imgs.shape[2]
    lo, hi = int(w * 0.70), int(w * 0.90)
    return lo + int(np.argmin(gx[lo:hi]))


def strips(img, league, split=None):
    """The bar's text lines at fixed positions (it never moves, so no text detection is needed): IPL = the two rows
    of its left third (team, score, overs / opponent, target); NFL = the bug cut at `split` into its slow part
    (down, teams, scores) and its fast part (the game clock). Small strips are upscaled toward the recogniser's 48 px
    line height."""
    h, w = img.shape[:2]
    if isinstance(split, list):                # a style's field rectangles (BUG_STYLES)
        out = [img[int(y0 * h): max(int(y0 * h) + 1, int(y1 * h)), int(x0 * w): max(int(x0 * w) + 1, int(x1 * w))]
               for x0, y0, x1, y1 in split]
    elif league == "IPL":
        z = int(w * IPLParser.ZONE)
        out = [img[: h // 2, :z], img[h // 2:, :z]]
    else:
        out = [img[:, :split], img[:, split:]] if split else [img]
    return [cv2.resize(s, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC) if (f := 40 / s.shape[0]) > 1.2 else s
            for s in out]


def rec_strip(reader, st, row):
    res, _ = reader(st, use_det=False, use_cls=False, use_rec=True)
    return [[w, round(float(c), 3), [0, row * 60]] for t, c in (res or []) if t and c >= 0.5 for w in t.split()]


def ocr(reader, img, league, split=None):
    """Recognition only (RapidOCR / PaddleOCR models on onnxruntime), one call per strip. Tokens keep the old format
    [text, conf, [x, y]]; the words of a strip share its position (x = 0, y = row * 60)."""
    return [t for row, st in enumerate(strips(img, league, split)) for t in rec_strip(reader, st, row)]


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
    import time
    crops = extract_crops(job, part, box)
    present = bar_present([cv2.imread(str(f)) for f in crops], job.league)
    todo = [k for k in range(len(crops)) if present[k] and not out["frames"][k]["tokens"]]
    t0 = time.time()
    if todo:
        split = layout(job, crops)
        for k in todo:
            out["frames"][k]["tokens"] = ocr_frames([str(crops[k])], [True], job.league, 0, split)[0][0]
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
    import time
    max_reuse = 20
    t0 = time.time()
    split = layout(job, crops)
    toks, n_ocr = ocr_frames([str(f) for f in crops], present, job.league, max_reuse, split)
    res = [(toks, n_ocr)]
    nw = 1
    secs = time.time() - t0
    on = [k for k in range(len(crops)) if present[k]]
    rate = sum(parser(toks[k]) is not None for k in on) / max(1, len(on))
    print(f"  OCR: {n_ocr} strip reads in {secs:.0f} s; {rate:.0%} of bar frames parsed", flush=True)
    out = {"engine": "rapidocr-rec", "preprocess": "strips", "sample_rates": {"strips": round(rate, 3)},
           "bar_on_screen": round(sum(present) / len(present), 3), "ocr_calls": n_ocr, "seconds": round(secs),
           "topup": {"frames": 0}, "frames": [{"t": round(k / FPS_OCR, 2), "tokens": toks[k]} for k in range(len(crops))]}
    cache.write_text(json.dumps(out))
    return out


_READER = None


def _worker_init(threads=1):
    global _READER
    from rapidocr_onnxruntime import RapidOCR
    _READER = RapidOCR(**({"intra_op_num_threads": threads, "inter_op_num_threads": 1} if threads else {}))


def ocr_frames(paths, present, league, max_reuse, split=None):
    """OCR in this process with onnxruntime's own multi-threading. On this laptop one process using all cores
    (~40 ms a strip) beats a pool of single-thread workers (~70-100 ms a strip overall): the cores can't all run flat
    out at once, so extra processes only fight."""
    if _READER is None:
        _worker_init(threads=None)
    return _ocr_chunk((paths, present, league, max_reuse, split))


def _ocr_chunk(task):
    """One contiguous run of crops: skip frames without the bar; per strip, reuse the last reading while that strip
    is unchanged (the scores rarely change, the clock every second), so only what changed is recognised."""
    paths, present, league, max_reuse = task[:4]
    split = task[4] if len(task) > 4 else None
    out, last, n = [], {}, 0
    for k, p in enumerate(paths):
        if not present[k]:
            out.append([]); continue
        toks = []
        for row, st in enumerate(strips(cv2.imread(p), league, split)):
            g = cv2.cvtColor(st, cv2.COLOR_BGR2GRAY).astype(np.int16)
            prev = last.get(row)
            # a style's fields (BUG_STYLES) are small and opaque: a clock going 14:26 -> 14:08 moves < 3% of the strip,
            # so they get a much stricter threshold (else the old clock is reused and the play is placed late)
            same = STRIP_SAME_FIELDS if isinstance(split, list) else STRIP_SAME
            if prev is not None and k - prev[0] <= max(max_reuse, 20 if row == 0 and split else 0)                     and prev[2].shape == g.shape and (np.abs(g - prev[2]) > 40).mean() < same:
                toks += prev[1]; continue
            t = rec_strip(_READER, st, row); n += 1
            last[row] = (k, t, g)
            toks += t
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
        """All plausible readings of the bar: [{inn, r, w, b, text}]. A recogniser without word detection glues the
        fields ('GT51-39.1' = 51-3 after 9.1 overs; 'LSG70-813' = 70-8 after 13), so every split of the digits after
        the dash that makes cricket sense is a candidate; the scoreboard match picks one."""
        if self.zone_px:
            tokens = [t for t in (tokens or []) if t[2][0] < self.zone_px]
        if not tokens:
            return []
        words = [t[0].upper().strip() for t in tokens]
        txt = " ".join(words)
        txt0 = txt.replace("O", "0")
        m = re.search(r"(?<![\d.])(\d{1,3})\s*[-/]\s*(\d)", txt0)
        if not m:
            return []
        runs = int(m.group(1))
        head = txt[:m.start()]
        inn = None
        if self.second in head.split() or head.endswith(self.second) or "TARGET" in txt or "NEED" in txt:
            inn = 2
        elif self.first in head.split() or head.endswith(self.first):
            inn = 1
        tail = re.match(r"[\d\s.,]*", txt0[m.start(2):]).group(0).rstrip()
        cands = []
        parts = tail.split()
        splits = []
        if len(parts) >= 2:                                   # spaced: wickets, then overs
            splits.append((parts[0], "".join(parts[1:2])))
        glued = tail.replace(" ", "")
        for k in (1, 2):
            splits.append((glued[:k], glued[k:]))
        for wk_s, ov_s in splits:
            if not wk_s.isdigit() or int(wk_s) > 10 or (len(wk_s) == 2 and wk_s[0] == "0"):
                continue
            o = re.fullmatch(r"(\d{1,2})(?:[.,]([0-5]))?", ov_s)
            if o and not (len(o.group(1)) == 2 and o.group(1)[0] == "0") and int(o.group(1)) <= 20:
                ov, bl = int(o.group(1)), int(o.group(2) or 0)
            elif ov_s == "" and runs == 0 and int(wk_s) == 0:  # 'LSG 0-0 v GT TARGET 145': innings not started
                ov, bl = 0, 0
            else:
                continue
            cands.append({"inn": inn, "r": runs, "w": int(wk_s), "b": ov * 6 + bl, "text": txt})
        if not cands:                                         # older OCR: overs elsewhere in the zone ('30-2 P 5.2')
            mw = re.match(r"(\d{1,2})(?![\d.])", txt0[m.start(2):])
            o = re.search(r"(?<!\d)(\d{1,2})\s*[.,]\s*([0-5])(?!\d)", txt0[m.end():])
            if mw and o and int(mw.group(1)) <= 10 and int(o.group(1)) <= 20:
                cands.append({"inn": inn, "r": runs, "w": int(mw.group(1)), "b": int(o.group(1)) * 6 + int(o.group(2)),
                              "text": txt})
        return cands

    def __call__(self, tokens):
        best, bc, brd = None, 1e9, None
        for rd in self.raw(tokens):
            for s, (i, r, w, b) in enumerate(self.states):
                if rd["inn"] is not None and i != rd["inn"]:
                    continue
                c = abs(r - rd["r"]) + 6 * abs(w - rd["w"]) + 0.3 * abs(b - rd["b"])   # overs: the smallest digits
                if c < bc:
                    best, bc, brd = s, c, rd
        if best is None or bc > 6:
            return None
        if best == len(self.states) - 1:
            best = self.brk - 1
        i, r, w, b = self.states[best]
        rd = brd
        return {"state": best, "cost": round(bc, 2), "read": f"{rd['r']}-{rd['w']} {rd['b'] // 6}.{rd['b'] % 6}",
                "label": f"inn{i} {r}/{w} {b // 6}.{b % 6}"}

    @staticmethod
    def event_step(state):
        return state


class NFLParser:
    """'1ST & 10 CLE 3 BUF 0 3RD 8:07' -> play row p (state p; its result is event step p + 1)."""
    ORD = {"1ST": 1, "2ND": 2, "3RD": 3, "4TH": 4, "0T": 5, "OT": 5}

    def __init__(self, gid, style=None):
        g = sister.game_json(gid)
        self.stacked = BUG_STYLES.get(style or "", {}).get("stacked", False)
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
        if self.stacked:                           # fields by row: away score, home score, quarter + clock, down
            by = {}
            for t in tokens:
                by.setdefault(int(t[2][1] // 60), []).append(t[0])
            num = lambda w: re.sub(r"[^\dO]", "", w.upper()).replace("O", "0")      # 'O' read for a zero
            dig = lambda r: next((num(w) for w in by.get(r, []) if re.fullmatch(r"\d{1,2}", num(w))), "")
            a, h = dig(0), dig(1)
            # the quarter row always opens with the quarter: a digit + two letters, lower case on FOX ('3rd', read as
            # '3pD', '3p0', '380'), glued to the clock: digit -> ordinal, keep the rest ('38011:14' = 3RD 11:14)
            row2 = " ".join(by.get(2, [])).upper().strip()
            m = None if re.match(r"\d{1,2}\s*[:.]\s*\d{2}", row2) else re.match(r"([1-4])([^\s:.]{2})\s*(.*)", row2)
            o = re.match(r"[O0]T\s*(.*)", row2)
            if m:
                row2 = f"{['1ST', '2ND', '3RD', '4TH'][int(m.group(1)) - 1]} {m.group(3)}"
            elif o:
                row2 = f"OT {o.group(1)}"
            tokens = ([[" ".join(by.get(3, []))]] + ([[f"{self.away} {a}"]] if a else []) + ([[f"{self.home} {h}"]] if h else [])
                      + [[w] for w in row2.split()])
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
        if mm is None:                             # glued: the last m:ss anywhere in the line
            g = re.findall(r"(?<![\d])([\dO]{1,2})[:.]([\dO]{2})", txt)          # 'O:52' = 0:52
            g = [(int(a.replace("O", "0")), int(b.replace("O", "0"))) for a, b in g]
            g = [(a, b) for a, b in g if a <= 15 and b <= 59]
            if not g:                              # colon lost: three digits right after the quarter ('4TH 332' = 3:32)
                g = [(int(a), int(b)) for a, b in re.findall(r"(?:1ST|2ND|3RD|4TH)\s*(\d)(\d{2})(?!\d)", txt)
                     if int(b) <= 59]
            if not g:                              # under a minute FOX shows ':51', read glued as '2ND51' = 0:51
                g = [(0, int(b)) for b in re.findall(r"(?:1ST|2ND|3RD|4TH|OT)\s*:?(\d{2})(?![\d&])", txt) if int(b) <= 59]
            if g:
                mm, ss = g[-1]
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
        qm = list(re.finditer(r"(1ST|2ND|3RD|4TH)(?!\s*&)", txt))   # an ordinal that isn't a down
        if q is None and qm:                       # glued line ('BUF73RD8:04' = away 7, 3RD): the last such ordinal
            q = self.ORD[qm[-1].group(1)]
        head = txt[: qm[-1].start()] if qm else txt                 # the scores sit before the quarter
        dn = re.search(r"(1ST|2ND|3RD|4TH)\s*(?:&\s*(\d{1,2}|G[O0]AL)|D[O0]WN)", txt)   # FOX: '2ND DOWN' (no distance)
        hs = re.search(rf"{self.home}\s*(\d{{1,2}})(?!\d)", head)
        as_ = re.search(rf"{self.away}\s*(\d{{1,2}})(?!\d)", head)
        return {"q": q, "c": mm * 60 + ss, "down": self.ORD.get(dn.group(1)) if dn else None,
                "togo": (None if dn.group(2) is None else 0 if dn.group(2).startswith("G") else int(dn.group(2))) if dn else None,
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
                    c += 0.2 * (dc - 12)
            elif dc >= -40 and (p == 0 or self.sb["q"][p - 1] != rd["q"] or rd["c"] <= self.sb["c"][p - 1]):
                # clock running before play p's snap; it can't read more than the previous play's snap clock
                match = rd["down"] is not None and self.down[p] == rd["down"] and rd["togo"] in (None, self.togo[p])
                c = -dc / 15.0 + (0.3 if match else 0.5)   # a stale down (the bug updates it late) costs a little
            else:
                continue
            if rd["h"] is not None and rd["h"] != self.sb["h"][p]:
                c += 1.0                       # the bug updates the score late after a TD
            if rd["a"] is not None and rd["a"] != self.sb["a"][p]:
                c += 1.0                       # the bug updates the score late after a TD
            if rd["down"] is not None and self.down[p] and (rd["down"] != self.down[p] or rd["togo"] not in (None, self.togo[p])):
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
    global OFFBUG
    # shots without the bug count as replays / reactions only this close to a live shot; a fan upload that keeps its
    # commercial breaks needs a tight window (job.yaml offbug_window), else the ads come in as 'replays'
    OFFBUG = float(job.cfg.get("offbug_window", 20))
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
        near = any(min(abs(r["start"] - b), abs(r["end"] - a), 0 if (a < r["end"] and b > r["start"]) else 1e9) <= OFFBUG
                   for a, b in live_t) if r.get("kind") != "backwards" else any(
                   min(abs(r["start"] - b), abs(r["end"] - a), 0 if (a < r["end"] and b > r["start"]) else 1e9) <= 20
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
    if job.league == "NASCAR":
        import nascar
        return nascar.tag(job)
    reocr = "--reocr" in sys.argv
    boxes = job.load("box.json")
    clips, ocr_stats = [], {}
    for part in job.parts:
        box = boxes[part.name]["box"]
        parser = IPLParser(job.gid, box[2]) if job.league == "IPL" else NFLParser(job.gid, job.cfg.get("bug_style"))
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
    al = job.alignment_any()
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
