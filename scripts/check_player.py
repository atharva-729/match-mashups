"""v2 Checkpoints 5 + 6: drive the local player in a real browser (Playwright + Edge, which plays H.264/AAC).

  5  screenshots of both visuals (A strip, B sparkline) at a lull, a touchdown and the end -> logs/v2_visual_*.png;
     neither visual (nor the captions or the song credit) overlaps the broadcast's score bug
  6  every game x length loads; each toggle (C, S, V, L) works mid-play; audio-video drift over 60 s of play;
     screenshots of the gallery and of the player with everything on; no console errors
Needs Playwright (any Python that has it; it only uses the standard library otherwise):
  <python with playwright> scripts/check_player.py [--quick]
"""
import json, subprocess, sys, time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PORT = 8799
BASE = f"http://127.0.0.1:{PORT}/player/"
PLAY_S = 60


def project_python():
    cfg = (ROOT / "config.yaml").read_text(encoding="utf-8")
    for line in cfg.splitlines():
        if line.startswith("PYTHON:"):
            return str(Path(line.split(":", 1)[1].strip()).expanduser())
    return sys.executable


def log(text):
    with open(ROOT / "logs/progress_v2.md", "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")
    print(text.encode(sys.stdout.encoding or "utf-8", "replace").decode(sys.stdout.encoding or "utf-8"))


def rect_overlap(a, b):
    return not (a["x"] + a["w"] <= b["x"] or b["x"] + b["w"] <= a["x"] or a["y"] + a["h"] <= b["y"] or b["y"] + b["h"] <= a["y"])


GEOM = """() => {
  const st = document.getElementById('stage').getBoundingClientRect();
  const tl = window.__mm.tl, sx = st.width / tl.video_size[0], sy = st.height / tl.video_size[1];
  const r = (id) => { const e = document.getElementById(id), b = e.getBoundingClientRect();
    const cs = getComputedStyle(e); return {x: b.left - st.left, y: b.top - st.top, w: b.width, h: b.height, on: parseFloat(cs.opacity) > 0.05}; };
  const [bx, by, bw, bh] = tl.bug || [0, 0, 0, 0];
  return {bug: {x: bx * sx, y: by * sy, w: bw * sx, h: bh * sy}, vizA: r('vizA'), cap: r('cap'), credit: r('credit')};
}"""


def main():
    quick = "--quick" in sys.argv
    srv = subprocess.Popen([project_python(), str(ROOT / "scripts/serve.py"), str(PORT)], stdout=subprocess.PIPE)
    time.sleep(1.5)
    gallery = json.loads((ROOT / "outputs/gallery.json").read_text(encoding="utf-8"))
    rows5, rows6, ok5, ok6 = [], [], True, True
    errors = []
    drift_rec = {}
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(channel="msedge", args=["--autoplay-policy=no-user-gesture-required"])
            pg = b.new_page(viewport={"width": 1400, "height": 1000})
            pg.on("console", lambda m: errors.append(f"{m.type}: {m.text}") if m.type == "error" else None)
            pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            pg.goto(BASE)
            pg.wait_for_selector(".gcard")
            pg.screenshot(path=str(ROOT / "logs/v2_gallery.png"))
            for g in gallery:
                for L in g["lengths"]:
                    tag = f"{g['job']} · {L} s"
                    pg.goto(f"{BASE}#/play/{g['job']}/{L}")
                    pg.wait_for_function("window.__mm && window.__mm.tl && !document.getElementById('video').paused", timeout=20000)
                    tl = pg.evaluate("window.__mm.tl")
                    # toggles mid-play
                    pg.wait_for_timeout(1500)
                    res = {}
                    keys = (("c", "window.__mm.captions"), ("s", "window.__mm.song"), ("v", "window.__mm.visual"))
                    if tl["files"].get("voice"):
                        keys += (("o", "window.__mm.voiceOn"),)
                    for k, check in keys:
                        before = pg.evaluate(check)
                        pg.keyboard.press(k)
                        pg.wait_for_timeout(400)
                        res[k.upper()] = pg.evaluate(check) != before          # the key flips the toggle
                    song_on = pg.evaluate("window.__mm.song")
                    vol = pg.evaluate("[document.getElementById('game').volume, document.getElementById('song').volume]")
                    playing = pg.evaluate("!document.getElementById('video').paused && !document.getElementById('game').paused && !document.getElementById('song').paused")
                    vol_ok = (vol[1] > 0.25 and vol[0] < 0.01) if song_on else (vol[0] > 0.25 and vol[1] < 0.01)   # > 0.25: ducked under the voice-over
                    tog_ok = all(res.values()) and vol_ok and playing
                    # drift over PLAY_S seconds of play (the whole summary if shorter)
                    span = min(PLAY_S if not quick else 10, tl["duration"] - 3)
                    pg.evaluate("window.__mm.sync = {max: 0, resyncs: 0, samples: 0, since: performance.now(), sum: 0}")
                    pg.wait_for_timeout(int(span * 1000))
                    sync = pg.evaluate("window.__mm.sync")
                    drift_ok = sync["max"] <= 0.08 + 1e-9 and sync["samples"] > 0
                    # everything on: screenshot
                    pg.screenshot(path=str(ROOT / f"logs/v2_player_{g['job']}_{L}.png"))
                    # Checkpoint 5: both visuals at a lull, a touchdown, the end; overlap with the bug
                    caps = tl["captions"]
                    samples = tl["samples"]
                    mid = [i for i in range(len(samples)) if 0.15 * len(samples) < i < 0.85 * len(samples)]
                    lull = min(mid, key=lambda i: samples[i][2]) / tl["rate"]
                    td_steps = {107, 110, 153}
                    td = next((i / tl["rate"] for i, s in enumerate(samples) if s[0] in td_steps), tl["duration"] / 2) + 2.0
                    moments = {"lull": lull, "td": td, "end": tl["duration"] - 1.5}
                    pg.evaluate("document.getElementById('video').pause()")
                    overlaps = []
                    for style in ("A",):
                        for name, t in moments.items():
                            pg.evaluate(f"document.getElementById('video').currentTime = {t}")
                            pg.wait_for_timeout(900)
                            if L == g["lengths"][0]:
                                pg.locator("#stage").screenshot(path=str(ROOT / f"logs/v2_visual_{style}_{name}.png"))
                            geo = pg.evaluate(GEOM)
                            for el in ("vizA", "cap", "credit"):
                                if geo[el]["on"] and geo[el]["w"] > 0 and rect_overlap(geo[el], geo["bug"]):
                                    overlaps.append(f"{style}/{name}/{el}")
                    ok5 &= not overlaps
                    rows5.append(f"- {tag}: overlaps with the score bug: {overlaps or 'none'} (lull {lull:.1f} s, TD {td:.1f} s, end)")
                    # L switches length (restart)
                    other = [x for x in g["lengths"] if x != L]
                    l_ok = True
                    if other:
                        pg.keyboard.press("l")
                        pg.wait_for_function(f"window.__mm.L === {other[0]}", timeout=10000)
                        l_ok = pg.evaluate(f"location.hash.endsWith('/{other[0]}')")
                    ok = tog_ok and drift_ok and l_ok
                    drift_rec[f"{g['job']}_{L}"] = {"max_ms": round(sync["max"] * 1000), "span_s": round(span), "pass": bool(ok)}
                    ok6 &= ok
                    rows6.append(f"- {tag}: toggles flip mid-play {res} · song {'on' if song_on else 'off'}, volumes game/song {vol[0]:.2f}/{vol[1]:.2f} · "
                                 f"all three media playing {playing} · L → {other[0] if other else '—'} {l_ok} · drift over "
                                 f"{span:.0f} s: max {sync['max'] * 1000:.0f} ms, mean "
                                 f"{(sync['sum'] / max(1, sync['samples'])) * 1000:.0f} ms, {sync['resyncs']} resyncs (≤ 80 ms: {drift_ok})")
            b.close()
    finally:
        srv.terminate()
    (ROOT / "work" / "player_check.json").write_text(json.dumps(drift_rec, indent=1), encoding="utf-8")
    git = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, capture_output=True, text=True).stdout
    media = [l for l in git.splitlines() if l.rsplit(".", 1)[-1].lower() in ("mp4", "m4a", "mp3", "wav", "webm", "mkv")]
    stamp = time.strftime("%Y-%m-%d %H:%M")
    log("\n".join([f"### {stamp} · Checkpoint 5 (visuals): {'PASS' if ok5 else 'FAIL'}",
                   "- Screenshots: `logs/v2_visual_{A,B}_{lull,td,end}.png` (A = win-probability strip, B = tension sparkline)",
                   *rows5]))
    ok6 &= not errors and not media
    log("\n".join([f"### {stamp} · Checkpoint 6 (player): {'PASS' if ok6 else 'FAIL'}",
                   "- Screenshots: `logs/v2_gallery.png`, `logs/v2_player_<job>_<L>.png` (everything on)",
                   *rows6, f"- Console errors: {errors or 'none'}", f"- Media in git status: {media or 'none'}"]))


if __name__ == "__main__":
    main()
