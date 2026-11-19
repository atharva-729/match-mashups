"""Export the published static site to docs/ (GitHub Pages serves that folder), with media on a GitHub Release.

  docs/index.html         the write-up, the Winter cuts on top as the highlights (scripts/build_page.py)
  docs/player/            the interactive player and gallery (a copy of player/)
  docs/outputs/           gallery.json + each cut's timeline.json; their media point at the release
  work/_release/          the files to upload to the release, under flat asset names (hard links, no extra space)

The highlights (HIGHLIGHT) come first in the gallery and are marked; cuts in EXCLUDE are left out. The YouTube links to
the full-game uploads are dropped from the public data (the timelines kept them private on purpose).
usage: python scripts/export_site.py [--media-base <url>]     (default: the release download URL; a local URL to test)
"""
import json, os, shutil, sys

from common import ROOT

REPO = "https://github.com/atharva-729/match-mashups"
TAG = "media-v1"
RELEASE = f"{REPO}/releases/download/{TAG}/"
HIGHLIGHT = ["nfl_car_cin_excite"]                                       # the Winter cuts
EXCLUDE = {"nfl_car_cin_mkhits_build", "nfl_car_cin_mkhits_cresc",     # Mountain King tries, dropped
           "ipl_mountain_king"}                                          # not on the public site
SHARE = {60: ("outputs/_share/Panthers-Bengals_Neither-side-gives-in_60s_Winter_FINAL.mp4", "Panthers-Bengals_Winter_60s_1080p.mp4"),
         120: ("outputs/_share/Panthers-Bengals_Neither-side-gives-in_120s_Winter_FINAL.mp4", "Panthers-Bengals_Winter_120s_1080p.mp4")}


def stage(src, name, out):
    dst = out / name
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    base = sys.argv[sys.argv.index("--media-base") + 1] if "--media-base" in sys.argv else RELEASE
    docs, rel = ROOT / "docs", ROOT / "work" / "_release"
    rel.mkdir(parents=True, exist_ok=True)
    # the player
    pdir = docs / "player"
    pdir.mkdir(parents=True, exist_ok=True)
    for f in ("player.css", "player.js"):
        shutil.copy2(ROOT / "player" / f, pdir / f)
    page = (ROOT / "player" / "index.html").read_text(encoding="utf-8").replace("Automatic game summaries · local demo",
                                                                                "Automatic game summaries").replace(
        " Open the original, then play the summary.", " The highlights are on top.")
    (pdir / "index.html").write_text(page, encoding="utf-8")
    # the gallery: highlights first, marked; no excluded cuts; no links to full-game uploads
    gallery = json.loads((ROOT / "outputs" / "gallery.json").read_text(encoding="utf-8"))
    cards = [g for g in gallery if g["job"] not in EXCLUDE]
    cards.sort(key=lambda g: (g["job"] not in HIGHLIGHT, HIGHLIGHT.index(g["job"]) if g["job"] in HIGHLIGHT else 0))
    n_files = 0
    for g in cards:
        g["highlight"] = g["job"] in HIGHLIGHT
        g["youtube"] = None
        for L in g["lengths"]:
            src = ROOT / "outputs" / g["job"] / str(L)
            tl = json.loads((src / "timeline.json").read_text(encoding="utf-8"))
            tl["youtube"] = None
            files = {}
            for kind, f in tl["files"].items():
                name = f"{g['job']}_{L}_{f}"
                stage(src / f, name, rel)
                files[kind] = base + name
                n_files += 1
            tl["files"] = files
            out = docs / "outputs" / g["job"] / str(L)
            out.mkdir(parents=True, exist_ok=True)
            (out / "timeline.json").write_text(json.dumps(tl, ensure_ascii=False), encoding="utf-8")
    (docs / "outputs").mkdir(parents=True, exist_ok=True)
    (docs / "outputs" / "gallery.json").write_text(json.dumps(cards, ensure_ascii=False, indent=1), encoding="utf-8")
    # the finished 1080p highlight videos, for the write-up page
    for L, (src, name) in SHARE.items():
        stage(ROOT / src, name, rel)
    import build_page
    build_page.VIDEO = {L: base + name for L, (_, name) in SHARE.items()}
    build_page.PLAYER_LINK = "player/"
    build_page.MEDIA_BASE = base
    build_page.main()
    size = sum(p.stat().st_size for p in rel.iterdir()) / 1e6
    print(f"site -> {docs}  ({len(cards)} cards, highlights: {HIGHLIGHT})")
    print(f"release files -> {rel}  ({n_files + len(SHARE)} files, {size:.0f} MB) for {REPO}/releases/tag/{TAG}")


if __name__ == "__main__":
    main()
