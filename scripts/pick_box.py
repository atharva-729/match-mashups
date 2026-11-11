"""Phase 2: find the on-screen scoreboard box, once per video (confirmed by Atharva).

Auto-propose: sample 200 frames, keep the top/bottom bands whose pixels stay mostly constant (the score bar),
OCR them on ~40 frames, find the score-like text that keeps reappearing at the same spot (IPL "139-4",
NFL "8:07"), and grow a box around it through the neighbouring text on the same bar.
Writes work/<job>/box_proposed.json and box_preview.jpg (box drawn on 4 frames).

  python scripts/pick_box.py inputs/<job>                 # propose + preview
  python scripts/pick_box.py inputs/<job> --manual        # drag the box yourself (OpenCV selectROI)
  python scripts/pick_box.py inputs/<job> --box x,y,w,h   # set it by numbers (pixels of the source video)
  python scripts/pick_box.py inputs/<job> --confirm "note" # accept the proposal -> box.json
"""
import re, sys
import cv2
import numpy as np

from common import job_from_argv, log, stamp

SCORE = {"IPL": re.compile(r"^\d{1,3}\s*[-/]\s*\d{1,2}$"), "NFL": re.compile(r"^\d{1,2}[:.]\d{2}$")}


def sample_frames(path, k):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for f in np.linspace(n * 0.03, n * 0.97, k).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, fr = cap.read()
        if ok:
            out.append(fr)
    cap.release()
    return out


def propose(path, league):
    import easyocr
    frames = sample_frames(path, 200)
    H, W = frames[0].shape[:2]
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.int16)
    med = np.median(gray, axis=0)
    stable = (np.abs(gray - med) < 14).mean(axis=0)           # fraction of frames each pixel sits at its median
    rows = stable.mean(axis=1)
    band_rows = [r for r in range(H) if (r < 0.25 * H or r > 0.70 * H) and rows[r] > 0.25]
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    hits = []                                                  # (x0, y0, x1, y1, text, frame)
    for fi, fr in enumerate(frames[::5]):
        for lo, hi in ((0, int(0.25 * H)), (int(0.70 * H), H)):
            if not any(lo <= r < hi for r in band_rows):
                continue
            for box, text, conf in reader.readtext(fr[lo:hi]):
                xs, ys = [p[0] for p in box], [p[1] + lo for p in box]
                hits.append((min(xs), min(ys), max(xs), max(ys), text.strip(), fi))
    anchors = [h for h in hits if SCORE[league].match(h[4].replace(" ", ""))]
    if not anchors:
        return None, frames
    # the score-like text that recurs at the same place in the most frames
    def same(a, b):
        return abs((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2) < 25 and abs((a[1] + a[3]) / 2 - (b[1] + b[3]) / 2) < 12
    best = max(anchors, key=lambda a: len({b[5] for b in anchors if same(a, b)}))
    group = [b for b in anchors if same(best, b)]
    x0, y0 = min(b[0] for b in group), min(b[1] for b in group)
    x1, y1 = max(b[2] for b in group), max(b[3] for b in group)
    h = y1 - y0
    # grow through recurring neighbours on the same bar (gap <= 1.6 text heights), up to 2 lines below
    frames_with = {b[5] for b in group}
    neigh = [b for b in hits if b[5] in frames_with]
    changed = True
    while changed:
        changed = False
        for b in neigh:
            vo = min(y1, b[3]) - max(y0, b[1])
            near_x = b[0] <= x1 + 1.6 * h and b[2] >= x0 - 1.6 * h
            below = 0 <= b[1] - y1 <= 0.8 * h and b[0] >= x0 - 0.5 * h and b[2] <= x1 + 2 * h
            if (vo > 0.3 * h and near_x) or below:
                nx0, ny0, nx1, ny1 = min(x0, b[0]), min(y0, b[1]), max(x1, b[2]), max(y1, b[3])
                if (nx0, ny0, nx1, ny1) != (x0, y0, x1, y1) and nx1 - nx0 < 0.6 * W and ny1 - ny0 < 0.12 * H:
                    x0, y0, x1, y1, changed = nx0, ny0, nx1, ny1, True
    pad = int(0.25 * h)
    box = [max(0, int(x0 - pad)), max(0, int(y0 - pad)), int(x1 - x0 + 2 * pad), int(y1 - y0 + 2 * pad)]
    return box, frames


def preview(frames, box, out):
    x, y, w, h = box
    picks = [frames[i] for i in np.linspace(len(frames) * 0.1, len(frames) * 0.9, 4).astype(int)]
    tiles = []
    for f in picks:
        f = f.copy()
        cv2.rectangle(f, (x, y), (x + w, y + h), (0, 255, 255), 3)
        tiles.append(cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2)))
    crop = cv2.resize(picks[1][y:y + h, x:x + w], None, fx=2, fy=2)
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    pad = np.zeros((crop.shape[0] + 20, grid.shape[1], 3), np.uint8)
    pad[10:10 + crop.shape[0], 10:10 + min(crop.shape[1], grid.shape[1] - 20)] = crop[:, :grid.shape[1] - 20]
    cv2.imwrite(str(out), np.vstack([grid, pad]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    job = job_from_argv()
    args = sys.argv[2:]
    boxes = job.load("box.json") if job.wpath("box.json").exists() else {}
    for part in job.parts:
        if "--manual" in args:
            cap = cv2.VideoCapture(str(part))
            at = float(args[args.index("--at") + 1]) if "--at" in args else 0.38 * cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            cap.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
            ok, fr = cap.read(); cap.release()
            title = "Drag the scoreboard box, then press Enter (c = cancel)"
            cv2.namedWindow(title, cv2.WINDOW_NORMAL); cv2.resizeWindow(title, fr.shape[1], fr.shape[0])
            cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
            box = [int(v) for v in cv2.selectROI(title, fr, showCrosshair=False)]
            cv2.destroyAllWindows()
            if box[2] < 5 or box[3] < 5:
                sys.exit("no box selected")
            preview(sample_frames(part, 40), box, job.wpath("box_preview.jpg"))
            boxes[part.name] = {"box": box, "confirmed": "manual selectROI", "when": stamp()}
        elif "--box" in args:
            box = [int(v) for v in args[args.index("--box") + 1].split(",")]
            preview(sample_frames(part, 40), box, job.wpath("box_preview.jpg"))
            job.save("box_proposed.json", {part.name: box})
            print("box set to", box, "-> preview", job.wpath("box_preview.jpg"))
            continue
        elif "--confirm" in args:
            prop = job.load("box_proposed.json")
            note = args[args.index("--confirm") + 1] if len(args) > args.index("--confirm") + 1 else "ok"
            boxes[part.name] = {"box": prop[part.name], "confirmed": note, "when": stamp()}
        else:
            box, frames = propose(part, job.league)
            if box is None:
                sys.exit("no score-like text found; run with --manual")
            preview(frames, box, job.wpath("box_preview.jpg"))
            prop = job.load("box_proposed.json") if job.wpath("box_proposed.json").exists() else {}
            prop[part.name] = box
            job.save("box_proposed.json", prop)
            print(part.name, "proposed box", box, "-> preview", job.wpath("box_preview.jpg"))
            continue
        job.save("box.json", boxes)
        log(f"### {stamp()} · Checkpoint 2 ({job.name}): PASS\n- Box for {part.name}: {boxes[part.name]['box']} "
            f"(x, y, w, h) · confirmed: {boxes[part.name]['confirmed']}")


if __name__ == "__main__":
    main()
