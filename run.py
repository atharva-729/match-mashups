"""Run every phase for one job:
  python run.py inputs/<job> [--from align|fetch|box|tag|plan|render] [--reocr] [--refetch] [--debug]
v2 (NFL demo flow: 60 s + 120 s summaries, game sound / song stems, captions, timeline for the player):
  python run.py inputs/<job> --v2 [--from ...] [--debug]     the footage phases (fetch/box/tag) as above, then per length
                                                             align -> plan -> render stems -> caption check -> timeline
  python run.py --serve                                      the local player: http://127.0.0.1:8765/player/

fetch runs only for a job with `replay_url` in job.yaml (footage built from a full-game replay) and only once: it is
skipped while inputs/<job>/part1.mp4 and inputs/<job>/replay/reel.json exist, unless --refetch. --debug also renders
mashup_debug.mp4 and reads its labels back.

Phase 2 (the scoreboard box) needs a one-time human confirmation per video. If work/<job>/box.json is missing,
this proposes a box, writes work/<job>/box_preview.jpg and stops; confirm with
  python scripts/pick_box.py inputs/<job> --confirm "ok"      (or --manual to drag it)
then run this again.
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PHASES = ["align", "fetch", "box", "tag", "plan", "render"]
SCRIPT = {"align": "align.py", "fetch": "fetch_replay.py", "box": "pick_box.py", "tag": "tag_shots.py",
          "plan": "plan_edit.py", "render": "render.py"}


V2_PHASES = ["song", "align", "fetch", "box", "tag", "plan", "render", "captions", "timeline"]


def run_v2(job, name, cfg):
    """v2: the song + windows once, the footage once (shared), then everything per length."""
    import time, yaml
    py = sys.executable
    sc = lambda f: str(ROOT / "scripts" / f)
    t_all = time.time()
    dbg = (["--debug"] if "--debug" in sys.argv else []) + (["--hq"] if "--hq" in sys.argv else [])   # --hq: near-lossless render
    if cfg.get("footage_from"):               # a story cut: the footage job's song, footage and tags; its own plan
        for L in (60, 120):
            for ph, script, extra in (("align", "align.py", []), ("plan", "plan_edit.py", []),
                                      ("render", "render.py", dbg), ("captions", "story_captions.py", [])):
                print(f"=== {L} s · {ph} ===", flush=True)
                subprocess.run([py, sc(script), job, "--len", str(L), *extra], check=True)
            subprocess.run([py, sc("captions.py"), job, "--len", str(L), "--check"])
            subprocess.run([py, sc("timeline.py"), job, "--len", str(L)], check=True)
        print(f"=== story cut, all phases: {time.time() - t_all:.0f} s ===", flush=True)
        return
    subprocess.run([py, sc("pick_song.py"), job], check=True)
    for L in (60, 120):                       # alignment needs no footage: first, so the footage phases can read it
        subprocess.run([py, sc("align.py"), job, "--len", str(L)], check=True)
    if cfg.get("replay_url"):
        built = (ROOT / "inputs" / name / "part1.mp4").exists() and (ROOT / "inputs" / name / "replay/reel.json").exists()
        if not built or "--refetch" in sys.argv:
            subprocess.run([py, sc("fetch_replay.py"), job], check=True)
    if not (ROOT / "work" / name / "box.json").exists():
        subprocess.run([py, sc("pick_box.py"), job], check=True)
        sys.exit(f"Box proposed: look at work/{name}/box_preview.jpg, then\n"
                 f"  python scripts/pick_box.py {job} --confirm \"ok\"   and rerun with --v2")
    if not (ROOT / "work" / name / "clips.json").exists() or "--refetch" in sys.argv:
        subprocess.run([py, sc("tag_shots.py"), job], check=True)
    for L in (60, 120):
        for ph, script, extra in (("plan", "plan_edit.py", []), ("render", "render.py", dbg)):
            t0 = time.time()
            print(f"=== {L} s · {ph} ===", flush=True)
            subprocess.run([py, sc(script), job, "--len", str(L), *extra], check=True)
            print(f"=== {L} s · {ph}: {time.time() - t0:.0f} s ===", flush=True)
        if cfg.get("league", "").upper() != "NFL" and not (ROOT / "captions" / f"{name}_{L}.json").exists():
            print(f"  captions/{name}_{L}.json missing: Claude Code writes it from the race events (work/{name}/race.json)")
        elif (ROOT / "captions" / f"{name}_{L}.json").exists():
            subprocess.run([py, sc("captions.py"), job, "--len", str(L), "--check"])
        else:
            subprocess.run([py, sc("captions.py"), job, "--len", str(L), "--brief"], check=True)
            print(f"  captions/{name}_{L}.json missing: Claude Code writes it from work/{name}/{L}/caption_brief.md")
        subprocess.run([py, sc("timeline.py"), job, "--len", str(L)], check=True)
    print(f"=== v2, all phases: {time.time() - t_all:.0f} s ===", flush=True)


def main():
    if "--serve" in sys.argv:
        sys.argv = [sys.argv[0]]
        sys.path.insert(0, str(ROOT / "scripts"))
        import serve
        return serve.main()
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    job = sys.argv[1]
    start = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else "align"
    extra = ["--reocr"] if "--reocr" in sys.argv else []
    name = Path(job).name
    import time, yaml
    cfg = yaml.safe_load((ROOT / "inputs" / name / "job.yaml").read_text(encoding="utf-8"))
    if "--v2" in sys.argv:
        return run_v2(job, name, cfg)
    t_all = time.time()
    for ph in PHASES[PHASES.index(start):]:
        if ph == "fetch":
            built = (ROOT / "inputs" / name / "part1.mp4").exists() and (ROOT / "inputs" / name / "replay/reel.json").exists()
            if not cfg.get("replay_url") or (built and "--refetch" not in sys.argv):
                continue
            if built:                          # a new reel: the old reel's per-video caches no longer apply
                import shutil
                for f in ("crops_part1", "ocr_part1.json", "shots_part1.json", "deliv_part1.json"):
                    q = ROOT / "work" / name / f
                    shutil.rmtree(q, ignore_errors=True) if q.is_dir() else q.unlink(missing_ok=True)
        if ph == "box":
            if (ROOT / "work" / name / "box.json").exists():
                continue
            subprocess.run([sys.executable, str(ROOT / "scripts" / SCRIPT[ph]), job], check=True)
            sys.exit(f"Box proposed: look at work/{name}/box_preview.jpg, then\n"
                     f"  python scripts/pick_box.py {job} --confirm \"ok\"   and rerun with --from tag")
        print(f"=== {ph} ===", flush=True)
        t0 = time.time()
        args = extra if ph == "tag" else (["--debug"] if ph == "render" and "--debug" in sys.argv else [])
        subprocess.run([sys.executable, str(ROOT / "scripts" / SCRIPT[ph]), job, *args], check=True)
        print(f"=== {ph}: {time.time() - t0:.0f} s ===", flush=True)
    print(f"=== all phases: {time.time() - t_all:.0f} s ===", flush=True)


if __name__ == "__main__":
    main()
