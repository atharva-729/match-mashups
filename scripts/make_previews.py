"""Short muted preview loops + posters for the write-up page's game cards, and the hero loop (the featured cut's climax).

Writes work/_release/preview_<job>.mp4 and poster_<job>.jpg (uploaded to the release with the other media).
usage: python scripts/make_previews.py
"""
import json, subprocess

from common import ROOT

CARDS = ["nfl_winter_bal_sd", "nfl_tetris_bal_lac", "nfl_cancan_lac_was", "nfl_danse_macabre", "nascar_kansas_2024",
         "nfl_car_cin_excite", "nfl_car_cin_blink", "nfl_car_cin_kick", "nfl_car_cin_drama"]
SHARE120 = ROOT / "outputs" / "_share" / "Panthers-Bengals_Neither-side-gives-in_120s_Winter_FINAL.mp4"
OUT = ROOT / "work" / "_release"
SECS = 6


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-threads", "2", *map(str, args)], check=True)


def clip(src, start, dur, dst, width):
    ff("-ss", f"{start:.2f}", "-i", src, "-t", dur, "-an", "-vf", f"scale={width}:-2,fps=30", "-c:v", "libx264",
       "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for job in ([] if "--hero" in __import__("sys").argv else CARDS):
        src = ROOT / "outputs" / job / "120" / "video.mp4"
        dur = json.loads((src.parent / "timeline.json").read_text(encoding="utf-8"))["duration"]
        start = max(0.0, 0.62 * dur)
        clip(src, start, SECS, OUT / f"preview_{job}.mp4", 640)
        ff("-ss", f"{start + 1:.2f}", "-i", src, "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "4", OUT / f"poster_{job}.jpg")
        print("preview", job, flush=True)
    L = json.loads((ROOT / "outputs" / "nfl_car_cin_excite" / "120" / "timeline.json").read_text(encoding="utf-8"))["duration"]
    hero_start = 3.5 + L - 13                                         # title card 3.5 s, then the cut: its last 13 s
    crop = "crop=1524:857:18:99,"                                    # the video frame only (not the side panel)
    ff("-ss", f"{hero_start:.2f}", "-i", SHARE120, "-t", 10, "-an", "-vf", crop + "scale=1280:-2,fps=30", "-c:v", "libx264",
       "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT / "hero_loop.mp4")
    ff("-ss", f"{hero_start + 2:.2f}", "-i", SHARE120, "-frames:v", "1", "-vf", crop + "scale=1280:-2", "-q:v", "3", OUT / "hero_poster.jpg")
    print("hero done", flush=True)


if __name__ == "__main__":
    main()
