"""Job paths and small helpers shared by every phase script."""
import json, subprocess, sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class Job:
    def __init__(self, arg):
        p = Path(arg)
        self.name = p.name if p.parts else str(arg)
        self.inp = ROOT / "inputs" / self.name
        self.work = ROOT / "work" / self.name
        self.out = ROOT / "outputs" / self.name
        self.work.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.cfg = yaml.safe_load((self.inp / "job.yaml").read_text(encoding="utf-8"))
        self.gid = self.cfg["match_id"]
        self.league = self.cfg["league"].upper()
        self.song = self.inp / "song.mp3"
        self.parts = sorted(self.inp.glob("part*.mp4"), key=lambda f: int(f.stem[4:] or 0))

    def wpath(self, name):
        return self.work / name

    def load(self, name):
        return json.loads(self.wpath(name).read_text(encoding="utf-8"))

    def save(self, name, obj):
        self.wpath(name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def job_from_argv():
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/<phase>.py inputs/<job>")
    return Job(sys.argv[1])


def log(text):
    """Append to logs/progress.md (the README's checkpoint log)."""
    f = ROOT / "logs/progress.md"
    f.parent.mkdir(exist_ok=True)
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(text.rstrip() + "\n\n")
    print(text.encode(sys.stdout.encoding or "utf-8", "replace").decode(sys.stdout.encoding or "utf-8"))


def stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def probe_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def mmss(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"
