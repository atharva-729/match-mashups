"""Job paths and small helpers shared by every phase script."""
import json, subprocess, sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


# files that differ per summary length (v2 --len): they live in work/<job>/<L>/ instead of work/<job>/
VARIANT = {"alignment.json", "edl.json", "plan_report.json", "audio_plan.json"}


class Job:
    def __init__(self, arg, L=None):
        p = Path(arg)
        self.name = p.name if p.parts else str(arg)
        self.inp = ROOT / "inputs" / self.name
        self.work = ROOT / "work" / self.name
        self.out = ROOT / "outputs" / self.name
        self.work.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.cfg = yaml.safe_load((self.inp / "job.yaml").read_text(encoding="utf-8"))
        # a story cut (job.yaml: footage_from: <job>, story: stories/<name>.yaml) reuses another job's footage, song and
        # tags; only its plan, captions and renders are its own
        self.foot_inp, self.foot_work = self.inp, self.work
        if self.cfg.get("footage_from"):
            base = self.cfg["footage_from"]
            self.foot_inp, self.foot_work = ROOT / "inputs" / base, ROOT / "work" / base
            cfg = yaml.safe_load((self.foot_inp / "job.yaml").read_text(encoding="utf-8"))
            cfg.update(self.cfg)
            self.cfg = cfg
        self.story = (yaml.safe_load((ROOT / self.cfg["story"]).read_text(encoding="utf-8"))
                      if self.cfg.get("story") else None)
        self.gid = self.cfg["match_id"]
        self.league = self.cfg["league"].upper()
        self.song = self.foot_inp / "song.mp3"
        self.parts = sorted(self.foot_inp.glob("part*.mp4"), key=lambda f: int(f.stem[4:] or 0))
        self.L = L                 # v2: the summary length (60 / 120 s); None = the v1 single mashup
        if L:                      # a length may have its own song: song_file_<L>, song_track_<L>, ... override
            for k in ("song_file", "song_track", "song_title", "song_artist", "song_credit_full"):
                if f"{k}_{L}" in self.cfg:
                    self.cfg[k] = self.cfg[f"{k}_{L}"]
        if L:
            (self.work / str(L)).mkdir(exist_ok=True)
            self.out_l = self.out / str(L)
            self.out_l.mkdir(exist_ok=True)

    @property
    def tag(self):
        return f"{self.name} · {self.L} s" if self.L else self.name

    def wpath(self, name):
        if self.L and name in VARIANT:
            return self.work / str(self.L) / name
        if name == "alignment.json":
            return self.work / name
        return self.foot_work / name      # footage files (clips, box, replay OCR): the footage job's

    def load(self, name):
        return json.loads(self.wpath(name).read_text(encoding="utf-8"))

    def alignment_any(self):
        """The v1 alignment, else the v2 120 s one (a v2-only job has no v1 song): for steps / key steps / song length."""
        for f in (self.work / "alignment.json", self.work / "120" / "alignment.json"):
            if f.exists():
                return json.loads(f.read_text(encoding="utf-8"))
        raise FileNotFoundError(f"no alignment for {self.name}: run align.py (v1) or align.py --len 120 (v2) first")

    def save(self, name, obj):
        self.wpath(name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def arg_len():
    return int(sys.argv[sys.argv.index("--len") + 1]) if "--len" in sys.argv else None


def job_from_argv():
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/<phase>.py inputs/<job> [--len 60|120]")
    return Job(sys.argv[1], arg_len())


def log(text):
    """Append to the checkpoint log: logs/progress.md (v1), or logs/progress_v2.md for a v2 run (--len)."""
    f = ROOT / ("logs/progress_v2.md" if arg_len() else "logs/progress.md")
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
