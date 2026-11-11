"""Run every phase for one job:
  python run.py inputs/<job> [--from align|fetch|box|tag|plan|render] [--reocr] [--refetch] [--debug]

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


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    job = sys.argv[1]
    start = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else "align"
    extra = ["--reocr"] if "--reocr" in sys.argv else []
    name = Path(job).name
    import time, yaml
    cfg = yaml.safe_load((ROOT / "inputs" / name / "job.yaml").read_text(encoding="utf-8"))
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
