"""Phase 5: render the EDL with ffmpeg, in ONE encode.

One filter graph: every EDL entry is an input that seeks straight into its source (-ss/-t), is retimed (setpts),
resampled to the song timeline's frame grid and cut to an exact frame count (so nothing drifts), scaled/padded to
1280x720; cuts inside one step get a short crossfade (round(0.15*fps) frames, the outgoing shot extended to pay for
it), every other cut is hard. The same graph adds the song as the only audio (0.5 s fade-in, 2 s fade-out), a 2.5 s
title card and a 3 s end card (drawtext over a dark gradient band, no logos).
--debug also writes mashup_debug.mp4 with a corner label per entry (step · over.ball/clock · desc) and reads the label
back at every key moment (Checkpoint 5's OCR check); without it, Checkpoint 5 checks duration, audio and the EDL.
usage: python scripts/render.py inputs/<job> [--debug] [--check-only]
"""
import json, shutil, subprocess, sys
from pathlib import Path

import cv2
import numpy as np

import sister
from common import job_from_argv, log, mmss, probe_duration, stamp

W, H = 1280, 720
FONT_DIR = Path("C:/Windows/Fonts")


def fps_of(part):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                        "-of", "csv=p=0", str(part)], capture_output=True, text=True, check=True).stdout.strip()
    a, b = r.split("/")
    return round(int(a) / int(b))


def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"ffmpeg failed: {' '.join(map(str, cmd))[:400]}\n{r.stderr[-2000:]}")


def is_cut(a, b):
    return not (a["part"] == b["part"] and abs(a["src_out"] - b["src_in"]) < 0.05)


def video_graph(job, edl, fps):
    """Inputs and filter chains that turn the EDL into one video stream [vcat] on the song's frame grid."""
    xf = max(1, round(0.15 * fps))
    F = lambda x: int(round(x * fps))
    inputs, chain = [], []
    for i, e in enumerate(edl):
        n = F(e["song_out"]) - F(e["song_in"])
        nxt = edl[i + 1] if i + 1 < len(edl) else None
        e["_fade"] = nxt is not None and is_cut(e, nxt) and nxt["step"] == e["step"]   # a cut inside one step
        e["_n"] = n + (xf if e["_fade"] else 0)
        src_len = e["_n"] / fps * e["speed"] + 0.5
        inputs += ["-ss", f"{e['src_in']:.4f}", "-t", f"{src_len:.4f}", "-i", str(job.inp / e["part"])]
        chain.append(f"[{i}:v]setpts=(PTS-STARTPTS)/{e['speed']:.6f},fps={fps},"
                     f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
                     f"tpad=stop_mode=clone:stop_duration=3,trim=end_frame={e['_n']},setpts=PTS-STARTPTS,"
                     f"format=yuv420p,settb=AVTB[s{i}]")
    last, length = "[s0]", edl[0]["_n"] / fps
    for i in range(1, len(edl)):
        prev = edl[i - 1]
        if prev["_fade"]:
            off = length - xf / fps
            chain.append(f"{last}[s{i}]xfade=transition=fade:duration={xf / fps:.4f}:offset={off:.4f},settb=AVTB[c{i}]")
            length = off + edl[i]["_n"] / fps
        else:
            chain.append(f"{last}[s{i}]concat=n=2:v=1:a=0,settb=AVTB[c{i}]")
            length += edl[i]["_n"] / fps
        last = f"[c{i}]"
    chain.append(f"{last}null[vcat]")
    return inputs, chain


def gradient_png(path):
    """Transparent at the top, dark at the bottom: a subtle band for the cards."""
    a = np.zeros((H, W, 4), np.uint8)
    y = np.arange(H)[:, None]
    alpha = np.clip((y - 0.42 * H) / (0.58 * H), 0, 1) ** 1.3 * 215
    a[..., 3] = alpha.astype(np.uint8)
    cv2.imwrite(str(path), a)


def esc(s):
    return s.replace("\\", "\\\\").replace("'", "\u2019").replace(":", "\\:").replace("%", "\\%")


def cards(job, al, song_len, work):
    """drawtext filters for the title and end cards; text goes through textfile= to dodge escaping."""
    cfg = job.cfg
    files = {
        "t1": cfg["song_title"], "t2": cfg["song_artist"], "t3": "matched to", "t4": al["title"],
        "e1": al["final"],
        "e2": f"Music: {cfg['song_title']}, {cfg['song_artist']} ({cfg['song_performer']}), {cfg['song_credit']}",
        "e3": "Match data: " + ("Cricsheet" if job.league == "IPL" else "nflverse") + " · Matched by Which Match Is Your Song?",
    }
    for k, v in files.items():
        (work / f"card_{k}.txt").write_text(v, encoding="utf-8")
    for f in ("segoeui.ttf", "segoeuib.ttf", "segoeuil.ttf"):
        shutil.copy(FONT_DIR / f, work / f)
    T0, T1 = 0.0, 2.5
    E0, E1 = song_len - 3.0, song_len
    fade = lambda a, b: (f"alpha='if(lt(t,{a + 0.35:.2f}),(t-{a:.2f})/0.35,if(gt(t,{b - 0.4:.2f}),({b:.2f}-t)/0.4,1))'")
    dt = lambda k, font, size, y, a, b, color="white": (
        f"drawtext=fontfile={font}:textfile=card_{k}.txt:fontsize={size}:fontcolor={color}:x=(w-text_w)/2:y={y}:"
        f"enable='between(t,{a:.2f},{b:.2f})':{fade(a, b)}:shadowcolor=black@0.5:shadowx=1:shadowy=1")
    return ",".join([
        dt("t1", "segoeuib.ttf", 46, "h-250", T0, T1), dt("t2", "segoeui.ttf", 28, "h-192", T0, T1, "white@0.85"),
        dt("t3", "segoeuil.ttf", 22, "h-140", T0, T1, "white@0.7"), dt("t4", "segoeuib.ttf", 34, "h-106", T0, T1),
        dt("e1", "segoeuib.ttf", 40, "h-170", E0, E1), dt("e2", "segoeui.ttf", 19, "h-104", E0, E1, "white@0.85"),
        dt("e3", "segoeui.ttf", 19, "h-74", E0, E1, "white@0.7"),
    ]), (T0, T1, E0, E1)


def srt(edl, ev, path):
    ts = lambda x: f"{int(x // 3600):02d}:{int(x % 3600 // 60):02d}:{int(x % 60):02d},{int(round((x % 1) * 1000)) % 1000:03d}"
    lines = []
    for k, e in enumerate(edl, 1):
        d = ev.get(e["step"], {})
        txt = (f"{e['step']} · {d.get('label', '')} · {d.get('desc', '')[:72]}  [{e['kind']} {e['speed']:.2f}x]"
               if e["step"] else f"pre-game intro  [{e['speed']:.2f}x]")
        lines += [str(k), f"{ts(e['song_in'])} --> {ts(e['song_out'])}", txt, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]   # QSV tested: no faster here


def final(job, edl, fps, al, song_len, work, out, debug):
    inputs, chain = video_graph(job, edl, fps)
    n = len(edl)
    draw, (T0, T1, E0, E1) = cards(job, al, song_len, work)
    gradient_png(work / "band.png")
    chain += [f"[{n + 1}:v]format=rgba,split[ga][gb]",
              f"[ga]fade=t=in:st={T0}:d=0.35:alpha=1,fade=t=out:st={T1 - 0.4}:d=0.4:alpha=1[g1]",
              f"[gb]fade=t=in:st={E0}:d=0.35:alpha=1[g2]",
              f"[vcat][g1]overlay=enable='between(t,{T0},{T1})'[o1]",
              f"[o1][g2]overlay=enable='between(t,{E0},{E1})'[o2]"]
    last = f"[o2]{draw}"
    if debug:
        last += (",subtitles=debug.srt:force_style='FontName=Consolas,FontSize=11,PrimaryColour=&H00FFFFFF,"
                 "BackColour=&H99000000,BorderStyle=4,Outline=0,Shadow=0,Alignment=7,MarginL=12,MarginV=10'")
    chain += [last + "[v]", f"[{n}:a]afade=t=in:d=0.5,afade=t=out:st={song_len - 2:.3f}:d=2[a]"]
    script = work / ("graph_debug.txt" if debug else "graph.txt")
    script.write_text(";\n".join(chain), encoding="utf-8")
    run(["ffmpeg", "-v", "error", "-y", *inputs, "-i", str(job.song), "-loop", "1", "-framerate", str(fps),
         "-t", f"{song_len:.3f}", "-i", "band.png", "-filter_complex_script", script.name, "-map", "[v]", "-map", "[a]",
         "-t", f"{song_len:.3f}", *ENC, "-r", str(fps), "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
         str(out)], cwd=work)


def main():
    job = job_from_argv()
    edl, al = job.load("edl.json"), job.load("alignment.json")
    song_len = al["song_len"]
    if "--check-only" in sys.argv:
        return check(job, al, edl, song_len, {e["step"]: e for e in sister.events(job.gid)},
                     (job.out / "mashup_debug.mp4").exists())
    fps = fps_of(job.parts[0])
    ev = {e["step"]: e for e in sister.events(job.gid)}
    debug = "--debug" in sys.argv
    import time
    t0 = time.time()
    final(job, edl, fps, al, song_len, job.work, job.out / "mashup.mp4", debug=False)
    print(f"  rendered mashup.mp4 in {time.time() - t0:.0f} s", flush=True)
    if debug:
        srt(edl, ev, job.work / "debug.srt")
        final(job, edl, fps, al, song_len, job.work, job.out / "mashup_debug.mp4", debug=True)
    check(job, al, edl, song_len, ev, debug)


def check(job, al, edl, song_len, ev, debug):
    """Checkpoint 5: duration, one audio stream, and per key moment the step on screen at its planned time: read back
    from the debug video's label by OCR (--debug), else from the EDL the video was rendered from."""
    out, dbg = job.out / "mashup.mp4", job.out / "mashup_debug.mp4"
    dur = probe_duration(out)
    na = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0",
                         str(out)], capture_output=True, text=True).stdout.split()
    kdir = job.wpath("keyframes"); shutil.rmtree(kdir, ignore_errors=True); kdir.mkdir()
    import re
    if debug:
        from rapidocr_onnxruntime import RapidOCR
        reader = RapidOCR()
    rows, ok, n = [], 0, 0
    plan = {int(k): v for k, v in job.load("plan_report.json").items()}
    for s in al["steps"]:
        if not s["is_key"]:
            continue
        i = s["i"]
        st = plan.get(i, {}).get("status", "not in footage")
        if st == "not in footage":
            rows.append(f"  - step {i} ({ev[i]['label']}): n/a, not in the highlight footage")
            continue
        # read the label where the planner put the moment (it may move a moment off its aligned time to give every
        # must-keep moment room); the plan report says how far it moved
        t = plan[i]["planned"] if plan.get(i, {}).get("planned") is not None else s["t"]
        in_edl = next((e["step"] for e in edl if e["song_in"] <= t < e["song_out"]), None)
        if debug:
            f = kdir / f"step{i:03d}_{t:06.2f}.jpg"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(dbg), "-frames:v", "1", str(f)],
                           check=True)
            res, _ = reader(cv2.imread(str(f))[:70, : W // 2])
            toks = [w for r in sorted(res or [], key=lambda r: r[0][0][0]) for w in r[1].split()]
            m = next((re.match(r"(\d{1,3})(?!\d)", x) for x in toks[:3] if re.match(r"\d", x)), None)
            shown = int(m.group(1)) if m else None                 # the label opens with the step ('55·' glued)
        else:
            shown = in_edl
        good = shown is not None and abs(shown - i) <= 2
        reachable = st != "footage-limited"
        if reachable:
            n += 1; ok += good
        why = "" if reachable else (f" (footage-limited: the highlights can't reach this song time; it plays at "
                                    f"{mmss(plan[i]['planned'])})" if plan[i]["planned"] is not None else " (footage-limited)")
        rows.append(f"  - step {i} ({ev[i]['label']}, {'+'.join(s['why'])}) at {mmss(t)} (aligned {mmss(s['t'])}): label reads {shown} "
                    f"(EDL says {in_edl}) {'PASS' if good else 'FAIL'}{why}")
    d_ok, a_ok = abs(dur - song_len) <= 0.1, len(na) == 1
    log("\n".join([f"### {stamp()} · Checkpoint 5 ({job.name}): {'PASS' if d_ok and a_ok and ok == n else 'FAIL'}",
                   f"- `outputs/{job.name}/mashup.mp4`: {dur:.3f} s vs song {song_len:.3f} s ({d_ok}); audio streams: {len(na)} ({a_ok})",
                   f"- Key moments on screen within ±2 steps: {ok}/{n} reachable key moments ("
                   + (f"label OCR, frames in `work/{job.name}/keyframes/`)" if debug else "from the EDL; --debug reads the labels)"),
                   *rows]))


if __name__ == "__main__":
    main()
