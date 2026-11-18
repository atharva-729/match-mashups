"""Captions for a story cut, written from its story file (stories/<name>.yaml) and timed on its edit plan.

Each beat's `say` appears when the beat's footage starts (its first shot on screen: a shown play or the anchor's
run-up); its `then` appears at the anchor's result (where the planner placed it). A caption stays up ~0.3 s per word
(2.2-4.5 s) and is cut short by the next one. Writes captions/<job>_<L>.json ([{start, end, step, text}]).
usage: python scripts/story_captions.py inputs/<job> --len 60|120
"""
import json

from common import ROOT, job_from_argv, mmss

MIN_D, MAX_D, PER_WORD, LEAD, GAP = 2.2, 4.5, 0.32, 0.15, 0.15


def main():
    job = job_from_argv()
    beats = job.story["cuts"][job.L]
    edl = job.load("edl.json")
    report = {int(k): v for k, v in job.load("plan_report.json").items()}
    end = edl[-1]["song_out"]
    raw, cursor = [], 0.0
    for b in beats:
        steps = {b["step"], *b.get("show", [])}
        first = next((e["song_in"] for e in edl if e["step"] in steps and e["song_out"] > cursor + 0.5), None)
        ev = (report.get(b["step"]) or {}).get("planned")
        if b.get("say") and first is not None:
            raw.append({"start": max(cursor, first) + LEAD, "step": b["step"], "text": b["say"]})
        if b.get("then") and ev is not None:
            raw.append({"start": ev + 0.3, "step": b["step"], "text": b["then"]})
        if ev is not None:
            cursor = max(cursor, ev)
        elif first is None:
            print(f"step {b['step']}: not on screen, captions skipped")
    raw.sort(key=lambda c: c["start"])
    caps = []
    for i, c in enumerate(raw):
        want = min(job.story.get("caption_max_s", MAX_D), max(MIN_D, PER_WORD * len(c["text"].split()) + 1.0))
        nxt = raw[i + 1]["start"] - GAP if i + 1 < len(raw) else end - 0.1
        caps.append({"start": round(c["start"], 2), "end": round(min(c["start"] + want, nxt), 2),
                     "step": c["step"], "text": c["text"]})
    f = ROOT / "captions" / f"{job.name}_{job.L}.json"
    f.write_text(json.dumps(caps, ensure_ascii=False, indent=0), encoding="utf-8")
    short = [c for c in caps if c["end"] - c["start"] < 1.8]
    print(f"{len(caps)} captions -> captions/{job.name}_{job.L}.json"
          + (f"; {len(short)} cut short by the next one: " + ", ".join(f"{mmss(c['start'])} '{c['text']}'" for c in short)
             if short else ""))


if __name__ == "__main__":
    main()
