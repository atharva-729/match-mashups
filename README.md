# Match Mashups

**Your song, cut to your match.** Take a song and the game it was matched to (from *Which Match Is Your Song?*), plus that game's highlight footage, and automatically edit a 2–4 minute video. The game plays in order, sped up and trimmed to fit, with the song as the soundtrack and the big moments landing near the song's big moments.

> **To Claude Code:** work through the phases in order. Each checkpoint is a self-check: verify it, append the result to `logs/progress.md`, then continue. **Stop and ask Atharva only where a phase says "ask Atharva".** If a checkpoint fails twice after a real fix attempt, write what happened to `SUMMARY.md` and stop.

---

## How it works (read first)

```
song + match_id ──► alignment: which second of the song each play/ball belongs to   (reuse the matcher)
footage ─────────► shots ─► read the on-screen scoreboard ─► tag each shot with its play/ball
                                                   │
                     edit plan: fill each stretch of the song with that stretch's shots
                                                   │
                                        render with ffmpeg ─► mashup.mp4
```

- **The edit plan comes for free.** The main project's matcher already lines up every play (NFL) or ball (IPL) with a timestamp in the song. We reuse it.
- **The scoreboard is the label.** IPL's score bar shows e.g. `145/4  16.2`, which points to exactly one ball in Cricsheet. The NFL's shows quarter, clock, score and down-and-distance, which points to one play. Nobody tags clips by hand.
- **"Close enough" is the bar.** The guiding rule: the key moments don't need to be frame-perfect, "as long as it's not terrible".

### Reuse the sister repo first
This project sits next to **`which-match-is-your-song`** (the "sister repo", path in `config.yaml` as `MATCH_REPO`). That repo already did most of the data and matching work. **Never rebuild something it already has.** Import it, or read its files from disk.

| Need here | Take it from the sister repo (verify the exact names in Phase 0) |
|---|---|
| Song energy curve (Python) | The function used in its JS/Python parity test |
| Resample, z-score, banded DTW (±10) | `scripts/match.py` |
| Song-to-game alignment path / moment timestamps | Whatever produced "2:41 → punt-return TD" on the page (Python, or the JS logic to port 1:1) |
| Game tension and WP curves, full resolution | `data/curves/` |
| Game metadata, moments, titles | `site/games/{id}.json` |
| NFL play-by-play (desc, quarter, clock, score, down) | the nflverse parquet files already in its `data/raw/` |
| IPL ball-by-ball (innings, over.ball, runs, wickets) and its ball-index convention | its parsed Cricsheet data and parsing code in `data/raw/` / `scripts/` |
| Key-step definitions (wickets, sixes, TDs, swings) | its moments logic |

**Rules:**
- Import functions (add `MATCH_REPO` to `sys.path`) or read its data files. **Don't copy code across, don't re-download data it already holds, and don't modify the sister repo.**
- If something truly isn't there, write it here once, matching the sister repo's spec and conventions. Log it under "Gaps" in `progress.md`.
- The ball and play indices used here **must be the sister repo's**, so alignment and tagging agree.

### Hard rules
- **Footage never goes in git.** The repo is public. `inputs/`, `work/` and `outputs/` are gitignored. Only code, configs without paths to footage, and logs without frames go public. Finished videos are shared privately (Drive), never published to GitHub Pages.
- **No manual clip tagging.** The only manual step is confirming the scoreboard box once per video (Phase 2).
- **Credit the song** on the end card: track title, artist, and "via classicals.de" for those tracks.

---

## Repo layout

```
match-mashups/
  config.yaml            # MATCH_REPO: path to the local which-match-is-your-song clone
  run.py                 # python run.py inputs/<job>   → runs every phase for a job
  scripts/               # align.py, pick_box.py, tag_shots.py, plan_edit.py, render.py
  inputs/<job>/          # gitignored
    song.mp3
    part1.mp4 [part2.mp4 …]   # footage, in game order
    job.yaml             # match_id, league (IPL|NFL), song_title, song_artist, song_credit
  work/<job>/            # gitignored: alignment.json, shots.json, clips.json, edl.json, previews
  outputs/<job>/         # gitignored: mashup.mp4, mashup_debug.mp4
  logs/                  # progress.md + small plots (no video frames)
```

Example `job.yaml`:
```yaml
match_id: ipl_1359475        # as used in the main repo's site/games/{id}.json
league: IPL
song_title: "In the Hall of the Mountain King"
song_artist: "Edvard Grieg (performance credited per classicals.de)"
song_credit: "via classicals.de"
```

---

## Phase 0: Setup

**Do:**
- Create the layout above. `.gitignore` must contain `inputs/`, `work/`, `outputs/`, `*.mp4`, `*.mov`, `*.mkv`, `*.mp3`, `*.wav`.
- Install:
  - `ffmpeg`
  - `pip install numpy pandas pyyaml librosa opencv-python "scenedetect[opencv]" easyocr matplotlib`
  - Optionally Tesseract plus `pytesseract` as a second OCR engine.
- `config.yaml` → `MATCH_REPO: ../which-match-is-your-song` (adjust to the real local path).
- **Inventory the sister repo before writing any code.** Read its README, `SUMMARY.md`, `logs/progress.md`, `scripts/` and `data/` layout. Fill in `logs/reuse_map.md`: for every row of the "Reuse the sister repo first" table, give the exact module and function, or file path, you will use, plus a one-line note on its inputs and outputs. Mark any row it doesn't cover as a **gap**.
- Write `scripts/sister.py`, a thin adapter. It's the only place that imports from or reads paths in the sister repo. Every other script calls this adapter, so a path change is a one-line fix.
- **Only for gaps:** implement the missing piece here, following the sister repo's spec exactly. For example, if there is no Python song-energy function, follow its README: 2048/1024 frames, `0.6*loud + 0.4*flux`, 3 s smoothing, silence trimmed. Then check it against the sister repo's JS on one file.

**Don't:**
- Don't vendor or modify the main repo.
- Don't add a GUI app or web server. Everything is scripts.

**✅ Checkpoint 0:**
- `ffmpeg -version` works.
- `import easyocr, scenedetect, cv2, librosa` works.
- Through `scripts/sister.py`: the sister repo's matcher, one game JSON, one full-resolution curve, and the play-by-play or ball-by-ball rows for one NFL and one IPL game all load.
- `logs/reuse_map.md` is complete, with any gaps listed.
- `git status` shows no media files.

---

## Phase 1: Song-to-game alignment

**Do:**
- Write `scripts/align.py <job>`:
  1. Get the song's energy curve and the game's tension curve **from the sister repo's functions and data**, via `sister.py`. Run its **banded DTW** (band of ±10). If the sister repo already computes the alignment path or moment timestamps, call that and skip step 2's derivation, keeping only the per-step expansion.
  2. Convert the DTW path into a **mapping from game step to song time**: game step *i* (play or ball index, full resolution) → `t_i` seconds in the song. Map steps to the 200-point grid linearly, take the median song position of the DTW cells for that grid point, convert to seconds, then enforce monotonic non-decreasing and smooth lightly.
  3. Detect song beats and onsets with `librosa.beat.beat_track` and `librosa.onset.onset_detect`.
  4. Save `work/<job>/alignment.json`: `{steps:[{i, t, wp, tension, is_key, desc}], beats:[...], onsets:[...], song_len}`.
- **Key steps:** the 5 `moments` from the game JSON, plus every wicket and six (IPL) or touchdown, turnover and field goal (NFL). Read these events from the sister repo's existing play-by-play and ball-by-ball data. Don't download or re-parse anything.
- **Sanity check against the page:** the moment timestamps here must match what the sister repo's page shows for the same song and match (±1 s). If they don't, find out why before moving on.

**Don't:** don't re-rank or re-match. Use the `match_id` from `job.yaml` even if it isn't the song's #1 match today.

**✅ Checkpoint 1:**
- `t_i` is monotonic, starts within the first 5% of the song and ends within the last 5%.
- Save a plot of tension vs song energy with the alignment lines to `logs/<job>_alignment.png`.
- List each key step with its song timestamp in `progress.md`.

---

## Phase 2: Find the scoreboard box (ask Atharva once per video)

**Do:**
- Write `scripts/pick_box.py <job>`:
  1. **Auto-propose:** sample 200 frames across the video. Find the region that stays mostly constant in colour and layout but has small changing text, typically a horizontal band near the bottom or top. Save `work/<job>/box_preview.jpg` with the proposed box drawn on 4 sample frames.
  2. **Ask Atharva** to look at `box_preview.jpg` and say "ok", or run `python scripts/pick_box.py <job> --manual`. That opens one frame in an OpenCV `selectROI` window to drag the box. It takes about ten seconds.
- Save the box per video file in `work/<job>/box.json`.
- If the broadcast moves the score bar between parts (e.g. a different layout in the super over), allow one box per part file.

**Don't:** don't guess silently. The box must be confirmed once.

**✅ Checkpoint 2:** the box is saved and the confirmation is noted in `progress.md`.

---

## Phase 3: Cut into shots and tag each with its play or ball

**Do:**
- **Shots:** use PySceneDetect `ContentDetector` (start with threshold 27) to find shots across all parts, in order.
- **Read the scoreboard:** sample **2 frames a second**. Crop the box, upscale ×3, convert to grayscale, apply adaptive thresholding, then run OCR with easyocr (digits-and-letters allowlist). If easyocr's per-frame success rate is under 60% on a 100-frame sample, also try Tesseract (`--psm 7`) and keep the better engine.
- **Parse IPL readings:**
  - score `(\d{1,3})\s*[-/]\s*(\d{1,2})` → runs, wickets
  - overs `(\d{1,2})\.(\d)` → over.ball
  - innings 2 is detected from the score dropping back near 0, or from text like "TARGET" or "NEED"
  - Map (innings, runs, wickets, over.ball) to the ball index **using the sister repo's parsed ball-by-ball rows and its indexing**, by exact match first, then nearest. Don't re-derive the ball counting.
  - The scoreboard usually updates **after** a delivery. A shot whose reading changes from X to Y shows the ball that produced Y.
- **Parse NFL readings:**
  - quarter `1ST|2ND|3RD|4TH|OT`, clock `\d{1,2}:\d{2}`, scores, down `1ST|2ND|3RD|4TH & \d+|GOAL`
  - Map to the play, **in the sister repo's play indexing**, whose quarter matches and whose clock is the nearest at or above the reading, with the score as a tie-breaker.
- **Clean the sequence:**
  1. Take the median of a shot's readings.
  2. Enforce that step indices never go backwards across live shots, using a longest-increasing-subsequence or DP pass. Outliers get dropped.
  3. Classify each shot:
     - `live`: a valid reading consistent with its neighbours
     - `replay`: no scoreboard, or a reading that goes backwards, directly after a live shot → attach to that shot's step
     - `filler`: crowd or close-up shots between live shots → attach to the previous step
     - `junk`: studio, ads, intro or outro, or no valid reading within 20 s on either side → discard
- Save `work/<job>/clips.json`: `[{part, start, end, step, kind, conf}]`.
- Make a **contact sheet** at `work/<job>/contact_sheet.jpg`: 16 random tagged shots, each as one frame with its step, over.ball or clock, and the play description underneath, for a quick eyeball check.

**Don't:**
- Don't tag shots by hand or ask Atharva to.
- Don't use an LLM or vision API for OCR unless both OCR engines fail the checkpoint. Then you may try Gemini on the cropped box frames only, and log it.

**✅ Checkpoint 3:**
- At least 70% of non-junk video time is tagged `live` or `replay`.
- At least 80% of key steps have at least one live shot.
- Zero backwards steps among live shots.
- Report coverage (steps with footage / total steps) in `progress.md`.
- Look at the contact sheet yourself. If more than 3 of 16 are clearly wrong, fix the parsing before continuing.

---

## Phase 4: The edit plan

**Do:**
- Write `scripts/plan_edit.py <job>` to produce `work/<job>/edl.json`, a list of `{part, src_in, src_out, speed, song_in, song_out, step, why}` that exactly covers `[0, song_len]` with no gaps and no overlaps.
- Walk through the song stretch by stretch. Stretch *i* runs from `t_i` to `t_{i+1}`, the time the alignment gives step *i*. Fill it with that step's shots. Merge consecutive steps into one stretch if a stretch would be shorter than 0.6 s.
- **Rules:**
  - **Key moments:** include the live shot at speed ≤ 1.25×. Shift it so the event (about 40% into the shot for IPL deliveries, the shot start for NFL snaps) lands on the nearest beat or onset within ±0.5 s of its song time. One replay may follow if the song stretch has room.
  - **Normal steps:** use speed 1.25–2×, and trim to the most active part (for IPL, start at the shot start and keep the delivery; for NFL, start at the snap).
  - **Low-tension stretches** (tension below the 30th percentile): montage at up to 2.5× with short cuts (0.6–1.5 s) on beats.
  - **No footage for a step:** borrow from the nearest step with footage within ±3 steps. Otherwise extend the neighbouring shot at a lower speed (≥ 0.8×). **Never black frames.**
  - Snap cut points to the nearest beat when one is within 0.25 s.
  - Shots stay in game order, except a replay that directly follows its live shot.
- Write a short report: number of cuts, average shot length, and for each key moment its intended song time vs its planned song time.

**Don't:**
- Don't use slow motion below 0.8×, except a key moment's replay, which may go to 0.75×.
- Don't reorder the game for drama. It's a summary, not a remix.

**✅ Checkpoint 4:**
- The EDL covers the song exactly (sum of durations = `song_len` ± 0.05 s), with no gaps or overlaps.
- At least 80% of key moments are placed within ±1 s of their aligned song time.
- Log the report.

---

## Phase 5: Render

**Do:**
- Write `scripts/render.py <job>` using ffmpeg only:
  - Trim each EDL entry and set its speed with `setpts`. Drop the source audio.
  - Scale and pad to 1920×1080 (or 1280×720 if the source is 720p).
  - Concatenate. Add a 0.15 s crossfade only on cuts within the same step; everything else is a hard cut.
  - **Audio** is the song only: a 0.5 s fade-in and a 2 s fade-out.
  - **Opening title card (2.5 s, over the first clip):** song title and artist, then "matched to", then the match title.
  - **End card (3 s, over the last clip):** the final score, plus the credit line ("Music: <title>, <artist>, via classicals.de · Match data: Cricsheet / nflverse").
  - Draw cards with ffmpeg `drawtext` in a clean sans font with a subtle dark gradient band. No team or league logos.
- Also render `mashup_debug.mp4`, the same video with a small burned-in corner label: `step · over.ball/clock · desc`. It's for checking alignment.
- Output `outputs/<job>/mashup.mp4` (H.264, CRF 20, `-preset veryfast`, AAC 192k).

**Don't:** don't add any overlays beyond the title and end cards (commentary and the intensity meter come in the next round). Don't upload anything.

**✅ Checkpoint 5:**
- ffprobe duration = song length ± 0.1 s, with exactly one audio stream.
- Extract frames from `mashup_debug.mp4` at each key moment's song time. Check that the burned-in label shows that key step or a neighbour within ±2 steps. Save the frames to `work/<job>/keyframes/` (not `logs/`), and log pass or fail per moment.

---

## Phase 6: The second league

**Do:** run the whole pipeline on a second job in the other league (NFL if the first was IPL). Fix anything league-specific in the parsing, and keep both jobs working.

**✅ Checkpoint 6:** both jobs pass Checkpoints 1–5. `python run.py inputs/<job>` regenerates either one from scratch.

---

## Phase 7: Wrap up

**Do:**
- Commit the code, `config.yaml` (without personal absolute paths; use a `config.example.yaml`), and `logs/` (no frames, no media). Run `git status` and confirm no media is staged before every commit. End commit messages with the attribution lines Claude Code normally adds.
- Write `SUMMARY.md` for Atharva:
  1. Where the two videos are (local paths)
  2. Coverage and key-moment accuracy for each
  3. What was reused from the sister repo, and which gaps had to be written new
  4. What was hard
  5. Anything that deviated from this plan
  6. **Next round:** commentary captions and an intensity overlay (frequency bars or a pulsing glow plus a win-probability tug-of-war bar), from review feedback

**Don't:** don't push footage, frames or rendered videos. Don't create a Pages site for this repo.

**✅ Checkpoint 7:** `git ls-files | grep -Ei '\.(mp4|mov|mkv|mp3|wav|jpg|png)$'` returns only the alignment plots in `logs/` (and nothing from `work/` or `outputs/`), and `SUMMARY.md` exists.
---

## Beyond NFL and IPL (note, not a phase)

The pipeline isn't tied to football or cricket. Any sport with **(a) a step-by-step data feed** and **(b) an on-screen counter to OCR** can be a new league. Only three pieces are league-specific: the step data and its curve, the scoreboard parser, and which events count as key moments.

- **NASCAR** is the obvious next one. Steps = laps, from NASCAR Race Center's unofficial per-race `lap-times.json` (Sportradar's NASCAR API is the paid production route). The broadcast lap counter is easier to OCR than the IPL bar or the NFL bug. NASCAR has no win probability, so use the **story curve** (`scripts/story.py`), as with IPL: lead changes, restarts and late laps are key, and caution laps are quiet. Not started.
