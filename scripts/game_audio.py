"""v2 game sound: the broadcast's own audio (commentary + crowd) under the summary, following the edit.

Rules (Atharva, 2026-11-12):
  * never cut a commentator mid-word or mid-scream: every audio cut lands in a pause (a low point of the sound, found
    in the audio itself);
  * the commentary is NOT sped up with the picture: each piece plays at 1.0x from where its video starts, laid over the
    (possibly faster) video, so it drifts a little behind on sped-up stretches;
  * a piece ends at the last pause that fits before the next piece; only if no pause fits does it speed up, at most
    1.2x (atempo, pitch kept).
Source: inputs/<job>/replay/full_audio.m4a (the whole game), positioned with reel.json's replay_true_in, so a piece can
start a moment before its video (to catch the start of a phrase) and isn't limited to the downloaded video sections.

  build(job, edl, L, out) -> stats; also writes work/<job>/<L>/audio_plan.json (per piece: where it was cut, tempo,
  drift from its video)
"""
import json, subprocess

import numpy as np

SR = 44100
LEAD_MAX = 0.6         # a piece may start this much before its video, to begin at a pause
LATE_MAX = 0.4         # ... or this much after
MIN_RUN = 2.0          # video runs shorter than this don't get their own audio: the previous piece plays over them
TEMPO_MAX = 1.2
GAP_OK = 0.8           # a silent gap up to this long before the next piece is fine at 1.0x; longer -> try tempo
TIERS = (-6.0, -4.5)   # what counts as a pause: a dip this far below the sound around it (-4.5: inside crowd roar)
FILL_MAX = 2.5         # a gap left anyway: the next piece may start up to this early (at a pause) to fill it
SOFT_FADE = 0.3        # the fade-out when a piece has to end on a soft dip rather than a clear pause


def pause_penalty(depth):
    """How bad it is to cut here, in 'seconds of silence' units: a clear pause 0, a dip in crowd roar 0.4,
    a slight dip 1.5, no dip 3+."""
    if depth <= -6:
        return 0.0
    if depth <= -4.5:
        return 0.4
    if depth <= -3:
        return 1.5
    return 3.0 + 0.3 * (depth + 3)
FADE = 0.04            # 40 ms fades at every audio cut
ENV_HOP = 0.01


def read_span(src, a, b):
    """Stereo float32 samples of src in [a, b] (seconds)."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, a):.3f}", "-t", f"{b - max(0.0, a):.3f}", "-i", str(src),
                        "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(r, np.float32).reshape(-1, 2)


class Pauses:
    """Pause finder on one span of the broadcast audio: a 'pause' is a local low of the 120 ms loudness that sits well
    below the loudness around it (the gaps between a commentator's phrases, breaths, the dip after a call)."""
    def __init__(self, y, t0):
        m = y.mean(axis=1)
        hop, win = int(ENV_HOP * SR), int(0.12 * SR)
        n = max(1, (len(m) - win) // hop)
        c = np.concatenate([[0.0], np.cumsum(m.astype(np.float64) ** 2)])
        idx = np.arange(n) * hop
        rms = np.sqrt((c[idx + win] - c[idx]) / win)
        self.db = 20 * np.log10(rms + 1e-6)
        self.t = t0 + (idx + win / 2) / SR
        k = int(2.0 / ENV_HOP)                       # loudness around: 80th percentile over +-1 s
        pad = np.pad(self.db, (k // 2, k // 2), mode="edge")
        from numpy.lib.stride_tricks import sliding_window_view
        self.ref = np.percentile(sliding_window_view(pad, k + 1)[: len(self.db)], 80, axis=1)
        self.depth = self.db - self.ref              # <= -6 dB: a clear pause

    def best(self, lo, hi, prefer, weight=0.5):
        """The best pause in [lo, hi]: deep and near `prefer`. -> (time, depth_db, clean?)"""
        sel = np.nonzero((self.t >= lo) & (self.t <= hi))[0]
        if not len(sel):
            return float(np.clip(prefer, lo, hi)), 0.0, False
        d = self.depth[sel]
        score = d + weight * np.abs(self.t[sel] - prefer) * 6
        i = sel[int(np.argmin(score))]
        return float(self.t[i]), float(self.depth[i]), bool(self.depth[i] <= -6)

    def latest(self, lo, hi, thr=-6.0):
        """The latest pause (dip <= thr dB) in [lo, hi] (end a piece as late as possible). -> (time, depth, clean?)"""
        sel = np.nonzero((self.t >= lo) & (self.t <= hi) & (self.depth <= thr))[0]
        if len(sel):
            # among the clear pauses in the last 0.4 s before the latest, take the deepest
            last = self.t[sel[-1]]
            near = sel[self.t[sel] >= last - 0.4]
            i = near[int(np.argmin(self.depth[near]))]
            return float(self.t[i]), float(self.depth[i]), True
        return None


def replay_map(job):
    """reel time -> replay time, per section."""
    f = job.inp / "replay" / "reel.json"
    if not f.exists():                         # footage is a highlights video: its own sound, 1:1
        return [(0.0, 1e9, 0.0)]
    m = json.loads(f.read_text(encoding="utf-8"))
    return [(x["reel_in"], x["reel_out"], x.get("replay_true_in", x["replay_in"])) for x in m]


def runs(edl, secs):
    """Output-time stretches over which the replay time is continuous: [(T0, T1, R0, R1_video, speed_avg)]."""
    pieces = []
    for e in edl:
        a, b, v = e["src_in"], e["src_out"], e["speed"]
        for r0, r1, rt in secs:                       # split an entry at section boundaries
            lo, hi = max(a, r0), min(b, r1)
            if hi - lo > 1e-3:
                T0 = e["song_in"] + (lo - a) / v
                pieces.append({"T0": T0, "T1": T0 + (hi - lo) / v, "R0": rt + (lo - r0), "R1": rt + (hi - r0),
                               "step": e["step"]})
    out = []
    for p in pieces:
        if out and abs(p["R0"] - out[-1]["R1"]) < 0.06:
            out[-1]["T1"], out[-1]["R1"] = p["T1"], p["R1"]
        else:
            out.append(dict(p))
    return out


def build(job, edl, L, out_path):
    src = job.inp / "replay" / "full_audio.m4a"
    if not src.exists():                       # highlights video: the sound is in part1.mp4
        src = job.parts[0]
    secs = replay_map(job)
    rs = runs(edl, secs)
    total = edl[-1]["song_out"]
    # group: a run shorter than MIN_RUN joins the previous piece (its audio keeps playing over it)
    groups = []
    for r in rs:
        if groups and r["T1"] - r["T0"] < MIN_RUN:
            groups[-1]["runs"].append(r)
        else:
            groups.append({"runs": [r]})
    # analyse each group's neighbourhood of the broadcast audio once
    plan, buf = [], np.zeros((int(np.ceil(total * SR)) + SR, 2), np.float32)
    starts = []
    for g in groups:
        r = g["runs"][0]
        lo, hi = r["R0"] - FILL_MAX - LEAD_MAX - 1.5, r["R0"] + (total - r["T0"]) * TEMPO_MAX + 1.5
        hi = min(hi, r["R0"] + 60.0)
        g["y0"] = lo
        g["y"] = read_span(src, lo, hi)
        g["P"] = Pauses(g["y"], max(0.0, lo))
        t, d, clean = g["P"].best(r["R0"] - LEAD_MAX, r["R0"] + LATE_MAX, r["R0"])
        g["Ra"], g["start_depth"], g["start_clean"] = t, d, clean
        g["Ta"] = max(0.0, r["T0"] - (r["R0"] - t))
        starts.append(g["Ta"])
    for i, g in enumerate(groups):           # keep starts ordered and at least 0.3 s apart
        if i and g["Ta"] < groups[i - 1]["Ta"] + 0.3:
            shift = groups[i - 1]["Ta"] + 0.3 - g["Ta"]
            g["Ta"] += shift; g["Ra"] += shift
    for i, g in enumerate(groups):
        S = (groups[i + 1]["Ta"] if i + 1 < len(groups) else total) - g["Ta"]
        last = i + 1 == len(groups)
        P, Ra = g["P"], g["Ra"]
        tempo, end, how, depth = 1.0, None, "", -99.0
        if last:                                           # the final piece runs to the end (faded out below)
            end, how = Ra + S, "end of the summary"
        else:
            # every possible end point, scored: seconds of silence it leaves before the next piece + how unclear a
            # pause it is (a clear pause costs nothing; inside a long crowd roar there may be none, and a soft dip
            # late beats a clear pause that leaves seconds of silence) + speeding up (only past the slot, <= 1.2x)
            sel = np.nonzero((P.t >= Ra + 0.3) & (P.t <= Ra + S * TEMPO_MAX))[0]
            tt, dd = P.t[sel], P.depth[sel]
            tem = np.maximum(1.0, (tt - Ra) / S)
            silence = np.maximum(0.0, S - (tt - Ra) / tem)
            cost = silence + np.vectorize(pause_penalty)(dd) + (tem - 1.0) * 8.0
            k = int(np.argmin(cost))
            end, tempo, depth = float(tt[k]), float(tem[k]), float(dd[k])
            how = (f"pause {depth:.0f} dB" if depth <= TIERS[-1] else f"soft dip {depth:.0f} dB (long fade)") + \
                  (f" at {tempo:.2f}x" if tempo > 1.001 else "") + (f" (gap {silence[k]:.1f} s)" if silence[k] > GAP_OK else "")
            gap = S - (end - Ra) / tempo
            if gap > GAP_OK:                               # fill the gap: start the next piece earlier, at a pause
                nx = groups[i + 1]; r1 = nx["runs"][0]
                want = min(FILL_MAX, gap - 0.15) + (r1["T0"] - nx["Ta"])
                lo_r = r1["R0"] - want
                for thr in TIERS:
                    sel = np.nonzero((nx["P"].t >= lo_r) & (nx["P"].t <= nx["Ra"] - 0.3) & (nx["P"].depth <= thr))[0]
                    if len(sel):
                        t_new = float(nx["P"].t[sel[0]])
                        nx["Ta"] -= nx["Ra"] - t_new; nx["Ra"] = t_new
                        nx["start_depth"], nx["start_clean"] = float(nx["P"].depth[sel[0]]), True
                        nx["filled"] = round(nx["Ra"] - t_new + (r1["R0"] - nx["Ra"]), 2)
                        how += f"; next piece starts {r1['T0'] - nx['Ta']:.1f} s early to fill it"
                        break
        y = g["y"][int(round((Ra - max(0.0, g["y0"])) * SR)): int(round((end - max(0.0, g["y0"])) * SR))]
        if tempo > 1.0 + 1e-3:
            y = atempo(y, tempo)
        n = len(y)
        f = min(int((FADE if g["start_clean"] else SOFT_FADE) * SR), n // 2)   # an unclear start fades in longer
        fo = min(int((FADE if depth <= TIERS[-1] else SOFT_FADE) * SR), n // 2)   # an unclear end fades out longer
        if f:
            y = y.copy()
            y[:f] *= np.linspace(0, 1, f, dtype=np.float32)[:, None]
            y[-fo:] *= np.linspace(1, 0, fo, dtype=np.float32)[:, None]
        a = int(round(g["Ta"] * SR))
        n = min(n, len(buf) - a)
        buf[a: a + n] += y[:n]
        r0 = g["runs"][0]
        t_end = min(r0["T1"], g["Ta"] + n / SR)            # drift at the end of the piece's first video run
        vid = r0["R0"] + (t_end - r0["T0"]) * (r0["R1"] - r0["R0"]) / max(1e-6, r0["T1"] - r0["T0"])
        aud = Ra + (t_end - g["Ta"]) * tempo
        plan.append({"out_in": round(g["Ta"], 3), "out_out": round(g["Ta"] + n / SR, 3),
                     "replay_in": round(Ra, 3), "replay_out": round(end, 3), "tempo": round(tempo, 3),
                     "lead": round(r0["T0"] - g["Ta"], 3), "start": f"pause {g['start_depth']:.0f} dB"
                     if g["start_clean"] else f"quietest nearby ({g['start_depth']:.0f} dB)", "end": how,
                     "drift_end": round(vid - aud, 2), "runs": len(g["runs"]), "step": r0["step"]})
    buf = buf[: int(round(total * SR))]
    # the summary's last 1.5 s fade out
    f = int(1.5 * SR)
    buf[-f:] *= np.linspace(1, 0, f, dtype=np.float32)[:, None]
    wav = out_path.with_suffix(".wav")
    write_wav(wav, buf)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                    "-ar", str(SR), "-c:a", "aac", "-b:a", "160k", "-t", f"{total:.3f}", str(out_path)], check=True)
    wav.unlink()
    return plan


def atempo(y, tempo):
    r = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-i", "-",
                        "-af", f"atempo={tempo:.4f}", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
                       input=np.ascontiguousarray(y, np.float32).tobytes(), capture_output=True, check=True).stdout
    return np.frombuffer(r, np.float32).reshape(-1, 2)


def write_wav(path, y):
    import wave
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def click_check(path, cuts):
    """Max sample jump within +-50 ms of each cut vs the file's 99.9th-percentile jump (ratio <= 1.5 passes)."""
    y = read_span(path, 0, 10_000).mean(axis=1)
    j = np.abs(np.diff(y))
    p999 = float(np.percentile(j, 99.9))
    w = int(0.05 * SR)
    worst = 0.0
    for c in cuts:
        k = int(c * SR)
        seg = j[max(0, k - w): k + w]
        if len(seg):
            worst = max(worst, float(seg.max()) / max(1e-9, p999))
    return worst
