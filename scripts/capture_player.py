"""Export a shareable MP4 that looks exactly like the site: the real player page, captured frame by frame.

Headless Edge (Playwright) opens the local player for one cut and steps it one video frame at a time: the player's
animation loop is paused (requestAnimationFrame queued) and run by hand after each seek, so captions, the meter, the
edge glow and the right panel ("Why this song" + the song <-> game chart) are drawn by the player's own code; CSS
fades are slowed through DevTools (Animation.setPlaybackRate) to match the pace of the capture, so they last as long
as on the site. Frames: the title card (as before play), the summary, the end card (as after the end), each 1920x1080
(the page's player row, centred). Audio = the song (the player's default). Seeks use a short-GOP copy of the cut
(outputs/<job>/<L>/video_cap.mp4) so every frame is exact and quick to reach.
Needs the local player server (python run.py --serve) and Playwright (pip install playwright; it uses installed Edge).
Writes outputs/<job>/<L>/share_site.mp4.
usage: python scripts/capture_player.py inputs/<job> --len 60|120 [--port 8765]
"""
import json, subprocess, sys, time

from playwright.sync_api import sync_playwright

from common import job_from_argv

FPS = 30
SLOT = 0.28                    # real seconds per captured frame (CSS fades are slowed to match; a 1080p JPEG shot takes ~0.25 s)
INTRO, OUTRO = 3.5, 4.0
VW, VH = 1920, 1080

INIT = """
window.__rafq = []; window.__origRAF = window.requestAnimationFrame.bind(window);
window.requestAnimationFrame = (cb) => { window.__rafq.push(cb); return window.__rafq.length; };
window.cancelAnimationFrame = () => {};
window.__flush = (n) => { for (let i = 0; i < n; i++) { const q = window.__rafq; window.__rafq = [];
  const now = performance.now(); for (const cb of q) { try { cb(now); } catch (e) { console.error(e); } } } };
"""
STEP = """async (t) => {
  const v = document.getElementById('video');
  if (Math.abs(v.currentTime - t) > 1e-4) { v.currentTime = t; await new Promise(r => v.addEventListener('seeked', r, {once: true})); }
  window.__flush(2);                                   // the player's frame loop, twice (it runs at ~60 fps on the site)
  await new Promise(r => window.__origRAF(() => window.__origRAF(r)));
}"""


def main():
    job = job_from_argv()
    port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else "8765"
    out = job.out_l
    tl = json.loads((out / "timeline.json").read_text(encoding="utf-8"))
    L = tl["duration"]
    cap = out / "video_cap.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out / "video.mp4"), "-c:v", "libx264", "-crf", "10",
                    "-preset", "veryfast", "-g", "4", "-an", str(cap)], check=True)
    dst = out / "share_site.mp4"
    enc = None
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True, args=["--autoplay-policy=no-user-gesture-required"])
        pg = b.new_page(viewport={"width": VW, "height": VH})
        pg.add_init_script(INIT)
        pg.goto(f"http://127.0.0.1:{port}/player/#/play/{job.name}/{job.L}")
        pg.wait_for_function("window.__mm && window.__mm.tl", timeout=90000)
        pg.evaluate(f"""async () => {{
            const v = document.getElementById('video');
            for (const id of ['game', 'song']) document.getElementById(id).muted = true;
            v.pause(); v.src = '../outputs/{job.name}/{job.L}/video_cap.mp4';
            await new Promise(r => v.addEventListener('loadeddata', r, {{once: true}}));
            document.getElementById('bigPlay').hidden = true; }}""")
        pg.evaluate(STEP, 0.0)
        box = pg.evaluate("() => { const r = document.querySelector('.playrow').getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; }")
        top = max(0, int((VH - box[3]) / 2) - int(box[1]))      # centre the player row in the frame
        cdp = pg.context.new_cdp_session(pg)
        cdp.send("Animation.enable")
        cdp.send("Animation.setPlaybackRate", {"playbackRate": (1 / FPS) / SLOT})
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "image2pipe", "-c:v", "mjpeg", "-r", str(FPS), "-i", "-",
                                "-i", str(out / "song.m4a"),
                                "-filter_complex", f"[0:v]pad={VW}:{VH}:0:{top}:color=0x0a0c0f[v];"
                                                   f"[1:a]adelay={int(INTRO * 1000)}:all=1,apad=pad_dur={OUTRO}[a]",
                                "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "16", "-preset", "slow",
                                "-tune", "film", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                                "-movflags", "+faststart", "-t", f"{INTRO + L + OUTRO:.3f}", str(dst)], stdin=subprocess.PIPE)
        clip = {"x": 0, "y": 0, "width": VW, "height": VH - top}
        shot = lambda: pg.screenshot(clip=clip, type="jpeg", quality=95)
        # title card, as the page shows it before play
        png = shot()
        for _ in range(int(INTRO * FPS)):
            enc.stdin.write(png)
        # the summary: the card fades as on play, then every frame
        pg.evaluate("() => { document.getElementById('titleCard').classList.add('fade'); window.__mm.started = true; }")
        n = int(round(L * FPS))
        t_start = time.perf_counter()
        for i in range(n):
            t0 = time.perf_counter()
            pg.evaluate(STEP, min(L - 0.02, i / FPS + 0.001))
            enc.stdin.write(shot())
            if i % 150 == 0:
                el = time.perf_counter() - t_start
                print(f"  {i / FPS:5.1f} / {L:.1f} s   ({el:.0f} s elapsed)", flush=True)
            rest = SLOT - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)
        # the end card, as after the end
        pg.evaluate("""() => { window.__mm.ended = true; document.getElementById('endCard').hidden = false;
                               document.getElementById('cap').classList.remove('on'); window.__flush(1); }""")
        for k in range(int(OUTRO * FPS)):
            t0 = time.perf_counter()
            enc.stdin.write(shot())
            rest = SLOT - (time.perf_counter() - t0)
            if rest > 0 and k < 30:            # let the card's fade-in play out, then it is still
                time.sleep(rest)
        b.close()
    enc.stdin.close(); enc.wait()
    cap.unlink(missing_ok=True)
    print(f"site-look share video -> {dst}")


if __name__ == "__main__":
    main()
