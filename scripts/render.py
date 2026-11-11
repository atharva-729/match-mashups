"""Phase 5: render the EDL with ffmpeg only.

Segments: each EDL entry is trimmed, retimed with setpts, scaled/padded to 1280x720 (the sources are 720p) and
encoded frame-exact (frame counts from the song timeline, so nothing drifts). Cuts inside one step get a short
crossfade (round(0.15*fps) frames, the outgoing shot extended to pay for it); every other cut is hard.
Final pass: the song as the only audio (0.5 s fade-in, 2 s fade-out), a 2.5 s title card and a 3 s end card
(drawtext over a dark gradient band, no logos). mashup_debug.mp4 adds a corner label per entry: step · over.ball/clock · desc.
usage: python scripts/render.py inputs/<job> [--keep-segments]
"""
import json, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
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


def seg_cmd(job, e, n_frames, fps, out):
    src_len = n_frames / fps * e["speed"] + 0.5
    vf = (f"setpts=(PTS-STARTPTS)/{e['speed']:.6f},fps={fps},"
          f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
          f"tpad=stop_mode=clone:stop_duration=3,format=yuv420p")
    return ["ffmpeg", "-v", "error", "-y", "-ss", f"{e['src_in']:.4f}", "-t", f"{src_len:.4f}", "-i", str(job.inp / e["part"]),
            "-vf", vf, "-frames:v", str(n_frames), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
            "-r", str(fps), str(out)]


def is_cut(a, b):
    return not (a["part"] == b["part"] and abs(a["src_out"] - b["src_in"]) < 0.05)


def build_video(job, edl, fps, segdir):
    """Render segments, crossfade cuts within a step, concat everything. Returns the silent full video path."""
    xf = max(1, round(0.15 * fps))
    F = lambda x: int(round(x * fps))
    # groups: runs of entries joined by within-step cuts (crossfaded) or continuous footage
    groups, cur = [], [0]
    for i in range(1, len(edl)):
        a, b = edl[i - 1], edl[i]
        if is_cut(a, b) and a["step"] != b["step"]:
            groups.append(cur); cur = [i]
        else:
            cur.append(i)
    groups.append(cur)
    jobs = []
    for g in groups:
        for j, i in enumerate(g):
            e = edl[i]
            n = F(e["song_out"]) - F(e["song_in"])
            fade_next = j + 1 < len(g) and is_cut(e, edl[g[j + 1]])
            e["_n"], e["_fade"] = n + (xf if fade_next else 0), fade_next
            jobs.append((e, segdir / f"s{i:04d}.mp4"))
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda je: run(seg_cmd(job, je[0], je[0]["_n"], fps, je[1])), jobs))
    gfiles = []
    for gi, g in enumerate(groups):
        out = segdir / f"g{gi:04d}.mp4"
        if len(g) == 1:
            shutil.copy(segdir / f"s{g[0]:04d}.mp4", out)
        else:
            inputs, chain, last, t_acc = [], [], "[n0]", 0.0
            for j, i in enumerate(g):      # one timebase and frame rate for every link (xfade insists)
                inputs += ["-i", str(segdir / f"s{i:04d}.mp4")]
                chain.append(f"[{j}:v]fps={fps},settb=AVTB,setpts=PTS-STARTPTS,format=yuv420p[n{j}]")
            for j in range(1, len(g)):
                prev = edl[g[j - 1]]
                t_acc += prev["_n"] / fps
                if prev["_fade"]:
                    t_acc -= xf / fps
                    chain.append(f"{last}[n{j}]xfade=transition=fade:duration={xf / fps:.4f}:offset={t_acc:.4f},"
                                 f"settb=AVTB[v{j}]")
                else:
                    chain.append(f"{last}[n{j}]concat=n=2:v=1:a=0,settb=AVTB[v{j}]")
                last = f"[v{j}]"
            run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(chain), "-map", last,
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-r", str(fps), "-pix_fmt", "yuv420p", str(out)])
        gfiles.append(out)
    lst = segdir / "list.txt"
    lst.write_text("".join(f"file '{f.name}'\n" for f in gfiles))
    full = segdir / "full.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", "full.mp4"], cwd=segdir)
    return full


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


def final(job, full, al, song_len, work, out, debug):
    draw, (T0, T1, E0, E1) = cards(job, al, song_len, work)
    gradient_png(work / "band.png")
    fc = (f"[2:v]format=rgba,split[ga][gb];"
          f"[ga]fade=t=in:st={T0}:d=0.35:alpha=1,fade=t=out:st={T1 - 0.4}:d=0.4:alpha=1[g1];"
          f"[gb]fade=t=in:st={E0}:d=0.35:alpha=1[g2];"
          f"[0:v][g1]overlay=enable='between(t,{T0},{T1})'[o1];[o1][g2]overlay=enable='between(t,{E0},{E1})'[o2];"
          f"[o2]{draw}")
    if debug:
        fc += (",subtitles=debug.srt:force_style='FontName=Consolas,FontSize=11,PrimaryColour=&H00FFFFFF,"
               "BackColour=&H99000000,BorderStyle=4,Outline=0,Shadow=0,Alignment=7,MarginL=12,MarginV=10'")
    fc += f"[v];[1:a]afade=t=in:d=0.5,afade=t=out:st={song_len - 2:.3f}:d=2[a]"
    run(["ffmpeg", "-v", "error", "-y", "-i", str(full), "-i", str(job.song), "-loop", "1", "-framerate", "25",
         "-t", f"{song_len:.3f}", "-i", "band.png", "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
         "-t", f"{song_len:.3f}", "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)], cwd=work)


def main():
    job = job_from_argv()
    edl, al = job.load("edl.json"), job.load("alignment.json")
    song_len = al["song_len"]
    if "--check-only" in sys.argv:
        return check(job, al, edl, song_len, {e["step"]: e for e in sister.events(job.gid)})
    fps = fps_of(job.parts[0])
    segdir = job.wpath("segments")
    shutil.rmtree(segdir, ignore_errors=True); segdir.mkdir()
    full = build_video(job, edl, fps, segdir)
    ev = {e["step"]: e for e in sister.events(job.gid)}
    srt(edl, ev, job.work / "debug.srt")
    final(job, full, al, song_len, job.work, job.out / "mashup.mp4", debug=False)
    final(job, full, al, song_len, job.work, job.out / "mashup_debug.mp4", debug=True)
    if "--keep-segments" not in sys.argv:
        shutil.rmtree(segdir, ignore_errors=True)
    check(job, al, edl, song_len, ev)


def check(job, al, edl, song_len, ev):
    """Checkpoint 5: duration, one audio stream, and the burned-in label at each key moment (read back by OCR)."""
    out, dbg = job.out / "mashup.mp4", job.out / "mashup_debug.mp4"
    dur = probe_duration(out)
    na = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0",
                         str(out)], capture_output=True, text=True).stdout.split()
    kdir = job.wpath("keyframes"); shutil.rmtree(kdir, ignore_errors=True); kdir.mkdir()
    import easyocr, re
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
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
        t = s["t"]
        f = kdir / f"step{i:03d}_{t:06.2f}.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(dbg), "-frames:v", "1", str(f)], check=True)
        img = cv2.imread(str(f))
        toks = [r[1].strip() for r in reader.readtext(cv2.resize(img[:70], None, fx=1.5, fy=1.5))]
        shown = next((int(x) for x in toks[:3] if re.fullmatch(r"\d{1,3}", x)), None)   # the label opens with the step
        in_edl = next((e["step"] for e in edl if e["song_in"] <= t < e["song_out"]), None)
        good = shown is not None and abs(shown - i) <= 2
        reachable = st != "footage-limited"
        if reachable:
            n += 1; ok += good
        why = "" if reachable else (f" (footage-limited: the highlights can't reach this song time; it plays at "
                                    f"{mmss(plan[i]['planned'])})" if plan[i]["planned"] is not None else " (footage-limited)")
        rows.append(f"  - step {i} ({ev[i]['label']}, {'+'.join(s['why'])}) at {mmss(t)}: label reads {shown} "
                    f"(EDL says {in_edl}) {'PASS' if good else 'FAIL'}{why}")
    d_ok, a_ok = abs(dur - song_len) <= 0.1, len(na) == 1
    log("\n".join([f"### {stamp()} · Checkpoint 5 ({job.name}): {'PASS' if d_ok and a_ok and ok == n else 'FAIL'}",
                   f"- `outputs/{job.name}/mashup.mp4`: {dur:.3f} s vs song {song_len:.3f} s ({d_ok}); audio streams: {len(na)} ({a_ok})",
                   f"- Key-moment labels within ±2 steps: {ok}/{n} reachable key moments "
                   f"(frames in `work/{job.name}/keyframes/`)", *rows]))


if __name__ == "__main__":
    main()
