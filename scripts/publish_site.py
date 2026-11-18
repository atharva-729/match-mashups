"""Rebuild everything the published site needs, on a machine that has the footage, then export the site.

  1. the Danse Macabre audio for the drama cut (library.ensure_audio: the Soundtrack Engine copy, else incompetech)
  2. the cuts made last: python run.py inputs/<job> --v2 --hq   (NEW_CUTS; their job.yaml must be in inputs/<job>/)
  3. every other cut's timeline, with the current timeline.py (the meter's song line = the matcher's song energy);
     if a rebuild fails, only that line is patched in its existing timeline
  4. the 1080p share videos of the highlights: scripts/capture_player.py (starts the local player server if needed)
  5. scripts/export_site.py -> docs/ (the site) + work/_release/ (the files for the GitHub release)
Then publish by hand (PUBLISH.md): the release upload, the push, GitHub Pages.
usage: python scripts/publish_site.py [--skip-capture]
"""
import json, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
NEW_CUTS = ["nfl_car_cin_excite", "nfl_car_cin_drama"]
HIGHLIGHT = "nfl_car_cin_excite"
SHARE = {60: "Panthers-Bengals_Neither-side-gives-in_60s_Winter_FINAL.mp4",
         120: "Panthers-Bengals_Neither-side-gives-in_120s_Winter_FINAL.mp4"}
EXCLUDE = {"nfl_car_cin_mkhits_build", "nfl_car_cin_mkhits_cresc"}


def run(*args):
    print("$", " ".join(map(str, args)), flush=True)
    subprocess.run([PY, *map(str, args)], cwd=ROOT, check=True)


def server_up(port=8765):
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{port}/player/", timeout=3).status == 200
    except Exception:
        return False


def patch_song_line(tl_path):
    tl = json.loads(tl_path.read_text(encoding="utf-8"))
    m, n, R = tl["match"], len(tl["samples"]), tl.get("rate", 10)
    import numpy as np
    tl["song_env"] = [round(float(x), 3) for x in np.interp(np.arange(n) / R, np.linspace(m["t0"], m["t1"], len(m["song"])), m["song"])]
    tl_path.write_text(json.dumps(tl, ensure_ascii=False), encoding="utf-8")


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    for job in NEW_CUTS:
        if not (ROOT / "inputs" / job / "job.yaml").exists():
            sys.exit(f"inputs/{job}/job.yaml is missing: copy it from the package (inputs/{job}/job.yaml)")
    dm = ROOT / "work" / "_library" / "audio" / "km_danse_macabre.mp3"
    if not dm.exists():
        try:
            import library
            library.ensure_audio(library.track("km_danse_macabre"))
        except Exception as e:                     # no Soundtrack Engine here: its public source (CC BY 4.0)
            print(f"  library lookup failed ({e}); downloading from incompetech", flush=True)
            dm.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request("https://incompetech.com/music/royalty-free/mp3-royaltyfree/Danse%20Macabre.mp3",
                                         headers={"User-Agent": "Mozilla/5.0"})
            dm.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    print("Danse Macabre audio:", dm, dm.exists(), flush=True)
    for job in NEW_CUTS:
        run("run.py", f"inputs/{job}", "--v2", "--hq")
    for d in sorted((ROOT / "outputs").iterdir()):
        if not d.is_dir() or d.name.startswith("_") or d.name in NEW_CUTS or d.name in EXCLUDE:
            continue
        for L in (60, 120):
            tl = d / str(L) / "timeline.json"
            if not tl.exists():
                continue
            try:
                run("scripts/timeline.py", f"inputs/{d.name}", "--len", L)
            except subprocess.CalledProcessError:
                print(f"  timeline rebuild failed for {d.name} {L} s: patching its song line only", flush=True)
                patch_song_line(tl)
    run("scripts/timeline.py", "--gallery")
    if "--skip-capture" not in sys.argv:
        srv = None
        if not server_up():
            srv = subprocess.Popen([PY, "run.py", "--serve"], cwd=ROOT)
            for _ in range(30):
                if server_up():
                    break
                time.sleep(1)
        try:
            for L, name in SHARE.items():
                run("scripts/capture_player.py", f"inputs/{HIGHLIGHT}", "--len", L)
                (ROOT / "outputs" / "_share").mkdir(exist_ok=True)
                shutil.copy2(ROOT / "outputs" / HIGHLIGHT / str(L) / "share_site.mp4", ROOT / "outputs" / "_share" / name)
        finally:
            if srv:
                srv.terminate()
    run("scripts/export_site.py")
    print("\nDone. Next: PUBLISH.md steps 4-6 (release upload, push, Pages).")


if __name__ == "__main__":
    main()
