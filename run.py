"""Run every phase for one job:  python run.py inputs/<job> [--from align|box|tag|plan|render] [--reocr]

Phase 2 (the scoreboard box) needs a one-time human confirmation per video. If work/<job>/box.json is missing,
this proposes a box, writes work/<job>/box_preview.jpg and stops; confirm with
  python scripts/pick_box.py inputs/<job> --confirm "ok"      (or --manual to drag it)
then run this again.
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PHASES = ["align", "box", "tag", "plan", "render"]
SCRIPT = {"align": "align.py", "box": "pick_box.py", "tag": "tag_shots.py", "plan": "plan_edit.py", "render": "render.py"}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    job = sys.argv[1]
    start = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else "align"
    extra = ["--reocr"] if "--reocr" in sys.argv else []
    name = Path(job).name
    for ph in PHASES[PHASES.index(start):]:
        if ph == "box":
            if (ROOT / "work" / name / "box.json").exists():
                continue
            subprocess.run([sys.executable, str(ROOT / "scripts" / SCRIPT[ph]), job], check=True)
            sys.exit(f"Box proposed: look at work/{name}/box_preview.jpg, then\n"
                     f"  python scripts/pick_box.py {job} --confirm \"ok\"   and rerun with --from tag")
        print(f"=== {ph} ===", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / SCRIPT[ph]), job, *(extra if ph == "tag" else [])], check=True)


if __name__ == "__main__":
    main()
