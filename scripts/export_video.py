"""Export a finished, shareable MP4 of one cut: what the player shows, burned in.

The player draws its overlays live in HTML over a silent video. This draws the same ones into every frame:
  - captions: broadcast lower third, left (accent bar + dark box, Bahnschrift), fading in 150 ms after their start
  - the meter: bottom right, the song line (pink) and the game line (blue) over the last 15 s
  - the edge glow: the colour of the team gaining win probability over the last 8 s, stronger when the game is,
    pulsing on the music's hits
  - the song credit, top right, clear of the broadcast's score bug
plus a title card before the summary (match, story title, length, "why this song") and an end card after it (final
score, song credit), as the player shows before play and after the end. Audio = the song (the player's default).
Reads outputs/<job>/<L>/{video.mp4, song.m4a, timeline.json}; writes outputs/<job>/<L>/share.mp4.
usage: python scripts/export_video.py inputs/<job> --len 60|120
"""
import json, subprocess, sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import job_from_argv

# 1080p by default: the 720p cut is upscaled (lanczos + light sharpening) and the captions / graphics are drawn at
# 1080p, so text is crisp on a big screen; --720 keeps the cut's own size
W, H, FPS = (1280, 720, 30) if "--720" in sys.argv else (1920, 1080, 30)
UP = "scale=1920:1080:flags=lanczos,unsharp=5:5:0.45:5:5:0" if W == 1920 else "null"
CQ = W / 100.0                       # the player's 1cqw at this size
ACCENT, INK, INK2 = (242, 179, 61), (238, 241, 244), (154, 164, 174)
PINK, BLUE = (255, 79, 139), (58, 209, 255)
FONTS = "C:/Windows/Fonts/"
INTRO, OUTRO = 3.5, 4.0              # seconds of title card before, end card after
METER_S = 15


def font(name, px, variation=None):
    f = ImageFont.truetype(FONTS + name, int(round(px)))
    if variation:
        try:
            f.set_variation_by_name(variation)
        except Exception:
            pass
    return f


def wrap(draw, text, f, width):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=f) <= width or not cur:
            cur = t
        else:
            lines.append(cur); cur = w
    return lines + ([cur] if cur else [])


def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


class Overlay:
    def __init__(self, tl):
        self.tl = tl
        self.f_cap = font("bahnschrift.ttf", 2.05 * CQ, "Regular")
        self.f_lab = font("bahnschrift.ttf", 1.05 * CQ, "SemiBold")
        self.f_cred_b = font("bahnschrift.ttf", 0.95 * CQ * 1.15, "SemiBold")
        self.f_cred = font("segoeui.ttf", 0.82 * CQ * 1.15)
        # where the player puts the lower elements (placeLower) and the credit (placeCredit), given the score bug
        base, bug = 11.5, tl.get("bug")
        if bug:                          # the bug is in the cut's pixels: to this frame's
            k = W / tl.get("video_size", [1280, 720])[0]
            bug = [v * k for v in bug]
        if bug and (bug[1] + bug[3]) / H > 0.6:
            base = max(base, (1 - bug[1] / H) * 100 + 1.5)
        self.meter_bottom, self.cap_bottom = base, base + 5.5
        self.cred_top, self.cred_maxw = 2.0, 23.0
        if bug:
            room = 100 - (bug[0] + bug[2]) / W * 100 - 2.6
            if room >= 15:
                self.cred_maxw = min(23.0, room)
            else:
                self.cred_top = (bug[1] + bug[3]) / H * 100 + 1.5
        g = tl.get("game_env") or [s[2] for s in tl["samples"]]
        lo, hi = min(g), max(g)
        self.game = [(v - lo) / ((hi - lo) or 1) for v in g]
        self.song = tl["song_env"]
        # edge glow: distance of every pixel from the frame's edge
        yy, xx = np.mgrid[0:H, 0:W]
        self.dist = np.minimum(np.minimum(xx, W - 1 - xx), np.minimum(yy, H - 1 - yy)).astype(np.float32)
        self.glow_px, self.glow_col, self.glow_op, self.pulse = None, None, 0.0, 0.0
        self.credit = self._credit_layer()

    def _credit_layer(self):
        tl = self.tl
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        maxw = self.cred_maxw / 100 * W
        pad_x, pad_y = 0.7 * CQ, 0.45 * CQ
        title = "Music: " + tl["song"]["title"]
        t_lines = wrap(d, title, self.f_cred_b, maxw - 2 * pad_x)
        c_lines = wrap(d, tl["song"]["credit"], self.f_cred, maxw - 2 * pad_x)
        lh_b, lh = self.f_cred_b.size * 1.3, self.f_cred.size * 1.35
        bw = max([d.textlength(s, font=self.f_cred_b) for s in t_lines] + [d.textlength(s, font=self.f_cred) for s in c_lines]) + 2 * pad_x
        bh = len(t_lines) * lh_b + len(c_lines) * lh + 2 * pad_y
        x1 = W - 0.014 * W; x0 = x1 - bw; y0 = self.cred_top / 100 * H
        d.rounded_rectangle([x0, y0, x1, y0 + bh], radius=0.4 * CQ, fill=(8, 10, 13, 184))
        y = y0 + pad_y
        for s in t_lines:
            d.text((x0 + pad_x, y), s, font=self.f_cred_b, fill=INK + (255,)); y += lh_b
        for s in c_lines:
            d.text((x0 + pad_x, y), s, font=self.f_cred, fill=(255, 255, 255, 178)); y += lh
        return im

    def glow(self, t, frame):
        tl, R = self.tl, self.tl["rate"]
        s = tl["samples"]
        k = max(0, min(len(s) - 1, int(t * R)))
        k8 = max(0, min(len(s) - 1, int(max(0.0, t - 8) * R)))
        swing = s[k][1] - s[k8][1]
        col = np.array(hexrgb(tl["teams"]["home"]["colour"] if swing >= 0 else tl["teams"]["away"]["colour"]), np.float32)
        ex = self.game[min(k, len(self.game) - 1)]
        strength = min(1.0, 0.2 + 0.8 * ex)
        px = (3 + 8 * strength) * CQ
        base = 0.25 + 0.45 * strength
        hit = tl["song_hit"][min(len(tl["song_hit"]) - 1, k)] if tl.get("song_hit") else 0.0
        self.pulse = max(hit, self.pulse * 0.74)                  # the player's 0.86 per 60 fps frame, at 30 fps
        op = min(1.0, base + 0.55 * self.pulse)
        a = 1 - np.exp(-1 / (0.6 * FPS))                          # colour and size ease over ~0.6 s, opacity ~0.15 s
        self.glow_px = px if self.glow_px is None else self.glow_px + (px - self.glow_px) * a
        self.glow_col = col if self.glow_col is None else self.glow_col + (col - self.glow_col) * a
        self.glow_op += (op - self.glow_op) * (1 - np.exp(-1 / (0.15 * FPS)))
        spread = self.glow_px / 4
        alpha = np.clip(1 - (self.dist - spread) / max(1.0, self.glow_px), 0, 1) ** 2 * (0.85 * self.glow_op)
        alpha = alpha[..., None]
        return (frame.astype(np.float32) * (1 - alpha) + self.glow_col * alpha).astype(np.uint8)

    def layer(self, t):
        im = self.credit.copy()
        d = ImageDraw.Draw(im)
        self._meter(d, t)
        self._caption(im, t)
        return im

    def _meter(self, d, t):
        R = self.tl["rate"]
        mw, mh = 0.34 * W, 6.2 * CQ
        x1 = W - 0.03 * W; x0 = x1 - mw; y1 = H - self.meter_bottom / 100 * H; y0 = y1 - mh
        d.rounded_rectangle([x0, y0, x1, y1], radius=0.6 * CQ, fill=(8, 10, 13, 140))
        px, py = 0.8 * CQ, 0.5 * CQ
        lw = max(d.textlength("SONG", font=self.f_lab), d.textlength("GAME", font=self.f_lab))
        d.text((x0 + px, y0 + mh * 0.30), "SONG", font=self.f_lab, fill=PINK + (255,), anchor="lm")
        d.text((x0 + px, y0 + mh * 0.70), "GAME", font=self.f_lab, fill=BLUE + (255,), anchor="lm")
        cx0, cx1 = x0 + px + lw + 0.8 * CQ, x1 - px
        cy0, cy1 = y0 + py + 3, y1 - py - 3
        n = len(self.tl["samples"])
        k1 = min(n - 1, int(t * R)); k0 = max(0, k1 - METER_S * R)
        X = lambda k: cx1 - (k1 - k) / (METER_S * R) * (cx1 - cx0)
        Y = lambda v: cy1 - v * (cy1 - cy0)
        for vals, col in ((self.song, PINK), (self.game, BLUE)):
            pts = [(X(k), Y(vals[min(k, len(vals) - 1)])) for k in range(k0, k1 + 1)]
            if len(pts) > 1:
                d.line(pts, fill=col + (70,), width=round(6 * W / 1280), joint="curve")
                d.line(pts, fill=col + (255,), width=round(2 * W / 1280), joint="curve")
            ex, ey = pts[-1]; r = 3 * W / 1280
            d.ellipse([ex - r, ey - r, ex + r, ey + r], fill=col + (255,))

    def _caption(self, im, t):
        caps = self.tl.get("captions") or []
        c = next((c for c in caps if c["start"] <= t < c["end"]), None)
        if not c:
            return
        on = t - (c["start"] + 0.15)
        if on < 0:
            return
        op = min(1.0, on / 0.15, max(0.0, (c["end"] - t) / 0.15))
        dy = (1 - min(1.0, on / 0.15)) * 6
        cap = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(cap)
        bar, padx, pady = max(3, 0.45 * CQ), 1.3 * CQ, 0.7 * CQ
        maxw = 0.62 * W - bar - 2 * padx
        lines = wrap(d, c["text"], self.f_cap, maxw)
        lh = self.f_cap.size * 1.3
        tw = max(d.textlength(s, font=self.f_cap) for s in lines)
        bh = len(lines) * lh + pady + 0.75 * CQ
        x0 = 0.04 * W; y1 = H - self.cap_bottom / 100 * H + dy; y0 = y1 - bh
        d.rectangle([x0, y0, x0 + bar, y1], fill=ACCENT + (255,))
        d.rectangle([x0 + bar, y0, x0 + bar + tw + 2 * padx, y1], fill=(8, 10, 13, 209))
        y = y0 + pady
        for s in lines:
            d.text((x0 + bar + padx, y), s, font=self.f_cap, fill=(255, 255, 255, 255)); y += lh
        if op < 1:
            a = np.array(cap); a[..., 3] = (a[..., 3] * op).astype(np.uint8); cap = Image.fromarray(a)
        im.alpha_composite(cap)


def card(frame, kicker, title, metas, why=None, fade=1.0):
    """The player's title / end card over a darkened frame: kicker, big line, meta lines, optional why line."""
    base = Image.fromarray(frame).convert("RGBA")
    shade = np.zeros((H, W, 4), np.uint8)
    g = np.linspace(0.35, 0.92, H)[:, None]
    shade[..., 3] = (g * 255 * fade).astype(np.uint8); shade[..., :3] = (5, 7, 9)
    base.alpha_composite(Image.fromarray(shade))
    d = ImageDraw.Draw(base)
    fk = font("bahnschrift.ttf", 1.2 * CQ * 1.2, "SemiBold")
    ft = font("bahnschrift.ttf", 4.1 * CQ, "SemiBold")
    fm = font("segoeui.ttf", 1.35 * CQ * 1.15)
    fw = font("bahnschrift.ttf", 1.9 * CQ, "SemiBold")
    x, maxw = 0.06 * W, 0.88 * W
    blocks = [("k", [kicker.upper()], fk, ACCENT, fk.size * 1.6)]
    blocks.append(("t", wrap(d, title, ft, maxw), ft, INK, ft.size * 1.15))
    for m in metas:
        blocks.append(("m", wrap(d, m, fm, maxw), fm, INK2, fm.size * 1.45))
    if why:
        blocks.append(("w", wrap(d, why, fw, 0.70 * W - 1.6 * CQ), fw, INK, fw.size * 1.35))
    total = sum(len(ls) * lh + (0.9 * CQ if kind == "w" else 0.2 * CQ) for kind, ls, _, _, lh in blocks)
    y = H - 0.08 * H - total
    a = int(255 * fade)
    for kind, ls, f, col, lh in blocks:
        if kind == "w":
            y += 0.7 * CQ
            d.rectangle([x, y, x + 0.35 * CQ, y + len(ls) * lh], fill=ACCENT + (a,))
        for s in ls:
            xx = x + (1.6 * CQ if kind == "w" else 0)
            if kind == "k":
                for ch in s:                      # letter-spaced kicker
                    d.text((xx, y), ch, font=f, fill=col + (a,)); xx += d.textlength(ch, font=f) + 0.18 * f.size
            else:
                d.text((xx, y), s, font=f, fill=col + (a,))
            y += lh
        y += 0.2 * CQ
    return np.array(base.convert("RGB"))


def main():
    job = job_from_argv()
    out = job.out_l
    tl = json.loads((out / "timeline.json").read_text(encoding="utf-8"))
    L = tl["duration"]
    match, when = (tl["title"].split(" · ") + [""])[:2]
    story = (tl.get("story_title") or "").split(" · ")[0]
    head = f"{match}: {story}" if story else match
    meta = f"{when or tl['season']} · {job.L}-second summary · cut automatically from the full broadcast"
    src = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(out / "video.mp4"), "-vf", UP, "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", str(FPS), "-"], stdout=subprocess.PIPE)
    dst = out / "share.mp4"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-i", str(out / "song.m4a"),
                            "-filter_complex", f"[1:a]adelay={int(INTRO * 1000)}:all=1,apad=pad_dur={OUTRO}[a]",
                            "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-tune", "film",
                            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                            "-t", f"{INTRO + L + OUTRO:.3f}", str(dst)], stdin=subprocess.PIPE)
    ov = Overlay(tl)
    nbytes = W * H * 3
    first = last = None
    n_main = int(round(L * FPS))
    frames = []
    for i in range(n_main):
        buf = src.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        if first is None:                       # the title card sits on the first frame, then fades into it
            first = fr.copy()
            for j in range(int(INTRO * FPS)):
                fade = min(1.0, (INTRO - j / FPS) / 0.5)
                enc.stdin.write(card(first, "Summary of the full game", head, [meta], tl.get("why_song"), fade).tobytes())
        t = i / FPS
        img = Image.fromarray(ov.glow(t, fr)).convert("RGBA")
        img.alpha_composite(ov.layer(t))
        enc.stdin.write(np.array(img.convert("RGB")).tobytes())
        last = fr
        if i % 300 == 0:
            print(f"  {t:5.1f} s / {L:.1f} s", flush=True)
    src.stdout.close(); src.wait()
    end_meta = [f"Song: {tl['song']['credit']}"]
    if tl.get("youtube"):
        end_meta.append(f"Original: {tl['youtube']}")
    for j in range(int(OUTRO * FPS)):
        fade = min(1.0, (j / FPS) / 0.5)
        enc.stdin.write(card(last, "Final", tl["final"], end_meta, None, fade).tobytes())
    enc.stdin.close(); enc.wait()
    print(f"share video -> {dst}")


if __name__ == "__main__":
    main()
