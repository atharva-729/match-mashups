"""v2: logs/results.md, one row per job x length, OVERWRITTEN on every run (one current set of numbers).
Reads what the checkpoints saved: work/<job>/<L>/check2.json (plan), check3.json (stems), captions/<job>_<L>.json,
work/player_check.json (drift in the browser), job.yaml (track + window + DTW).
usage: python scripts/results.py
"""
import json

import yaml

from common import ROOT, stamp


def main():
    rows = []
    pc = ROOT / "work" / "player_check.json"
    drift = json.loads(pc.read_text(encoding="utf-8")) if pc.exists() else {}
    for d in sorted((ROOT / "inputs").iterdir()):
        jf = d / "job.yaml"
        if not jf.exists():
            continue
        cfg = yaml.safe_load(jf.read_text(encoding="utf-8"))
        for L in (60, 120):
            w = ROOT / "work" / d.name / str(L)
            if not (w / "check2.json").exists() or not (ROOT / "outputs" / d.name / str(L) / "timeline.json").exists():
                continue
            c2 = json.loads((w / "check2.json").read_text(encoding="utf-8"))
            c3 = json.loads((w / "check3.json").read_text(encoding="utf-8")) if (w / "check3.json").exists() else {}
            cf = ROOT / "captions" / f"{d.name}_{L}.json"
            n_cap = len(json.loads(cf.read_text(encoding="utf-8"))) if cf.exists() else 0
            dr = drift.get(f"{d.name}_{L}", {})
            track = cfg.get("song_track", "?").replace("demo:", "")
            rows.append(
                f"| {d.name} | {L} s | {track} {cfg.get(f'song_start_{L}', 0):.1f}–{cfg.get(f'song_end_{L}', 0):.1f} s "
                f"| {cfg.get(f'song_dtw_{L}', '—')} | {c2['keys_shown']}/{c2['keys_in_footage']} "
                f"| {c2['speed_min']:.2f}–{c2['speed_max']:.2f}× | {n_cap} "
                f"| {str(dr.get('max_ms', '—')) + ' ms' if dr else '—'} "
                f"| {c3.get('click_ratio', '—')}× | {c3.get('tempo_max', '—')}× / {c3.get('drift_end_max', '—')} s |")
    out = [f"# Results (v2), {stamp()}",
           "", "Overwritten on every run. Key moments = TDs, turnovers, FGs, big swings and page moments that are in the "
           "footage; shown = on screen in the summary. Drift = the player's audio vs video in the browser (≤ 80 ms). "
           "Click = the worst sample jump at a game-sound cut vs the file's 99.9th percentile (≤ 1.5). "
           "Commentary = the game sound's fastest piece / its largest lag behind the picture.", "",
           "| Job | Length | Track + window | DTW | Key moments shown | Speeds | Captions | Drift max | Click | Commentary tempo / lag |",
           "|---|---|---|---|---|---|---|---|---|---|", *rows]
    (ROOT / "logs" / "results.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"logs/results.md: {len(rows)} rows")


if __name__ == "__main__":
    main()
