"""Build the public write-up page: docs/index.html (one self-contained file; GitHub Pages serves docs/ at the site root).

Top: the featured cut (Panthers @ Bengals, 2014 Week 6, 37-37 OT, to Vivaldi's Winter) as a hero loop, then both lengths
side by side, the other games as preview cards, the result chart and what the numbers don't show. Below the line: the
full write-up (how it works, what was learned, credits). Every number and chart is computed here from the cuts' own data.
No footage is in the page or the repo: videos, previews and posters are release files, linked by URL (MEDIA_BASE).
usage: python scripts/build_page.py   (export_site.py sets VIDEO, PLAYER_LINK and MEDIA_BASE, then calls main())
"""
import html, json

import numpy as np

import excitement
import sister
from common import ROOT, Job

JOB, BASE = "nfl_car_cin_excite", "nfl_car_cin_blink"     # the final cuts; the first version, for comparison
GID = "nfl_2014_06_CAR_CIN"
VIDEO = {60: "", 120: ""}                                   # the hosted share videos (set by export_site.py)
PLAYER_LINK = "player/"                                     # the interactive player
MEDIA_BASE = ""                                             # where previews/posters live (the release)
TIES = {131, 150, 177, 208}
GAMES = ["nfl_winter_bal_sd", "nfl_tetris_bal_lac", "nfl_cancan_lac_was", "nfl_danse_macabre", "nascar_kansas_2024"]
SAME_GAME = ["nfl_car_cin_excite", "nfl_car_cin_blink", "nfl_car_cin_kick", "nfl_car_cin_drama"]
SONGS = [("Winter", "Vivaldi · Winter"), ("Mountain King", "Grieg · Mountain King"), ("Can-Can", "Offenbach · Can-Can"),
         ("Danse Macabre", "Danse Macabre"), ("Tetris", "Tetris theme"), ("William Tell", "Rossini · William Tell")]
e = html.escape


def cut_stats(name, L):
    job = Job(name, L)
    rep = {int(k): v for k, v in job.load("plan_report.json").items()}
    by = {b["step"]: b for b in job.story["cuts"][L]}
    steps = [s for s in by if (rep.get(s) or {}).get("planned") is not None]
    t_ = [rep[s]["planned"] for s in steps]
    tl = json.loads((job.out_l / "timeline.json").read_text(encoding="utf-8"))
    t, cc = excitement.cut_curve(GID, [by[s] for s in steps], t_, tl["duration"])
    m = tl["match"]
    song = np.interp(t, np.linspace(m["t0"], m["t1"], len(m["song"])), m["song"])
    ph = np.array(job.load("alignment.json")["hits"])
    off = [float(np.min(np.abs(ph[:, 0] - x))) for x in t_]
    speeds = [x["speed"] for x in job.load("edl.json")]
    detail = [(x, by[s].get("then") or by[s].get("say") or f"play {s}", o < 0.05, o, s in TIES)
              for s, x, o in sorted(zip(steps, t_, off), key=lambda z: z[1])]
    return {"t": t, "song": song, "game": (cc - cc.min()) / max(1e-9, np.ptp(cc)), "r": float(np.corrcoef(song, cc)[0, 1]),
            "beats": [(s, x) for s, x in zip(steps, t_)], "on": sum(1 for o in off if o < 0.05), "n": len(steps),
            "dur": tl["duration"], "speeds": (min(speeds), max(speeds)), "tl": tl, "detail": detail}


def path(xs, ys, x0, x1, y0, y1, w, h, pad=4):
    sx = lambda x: pad + (x - x0) / (x1 - x0) * (w - 2 * pad)
    sy = lambda y: pad + (1 - (y - y0) / (y1 - y0)) * (h - 2 * pad)
    return " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(zip(xs, ys)))


def chart_game():
    """The whole game: the old tension curve vs the new excitement curve, the plays that tied the game marked."""
    full = sister.curve(GID)
    old = np.asarray(full["tension"], float); old = (old - old.min()) / np.ptp(old)
    ex = excitement.curve(GID)["excitement"]; ex = (ex - ex.min()) / np.ptp(ex)
    ties = excitement.ties(GID)
    W, H = 900, 120
    x = np.arange(len(old))
    out = []
    for label, ys, col in (("Old: “tension” (how close the game is)", old, "#7f8ea3"), ("New: excitement (swings that fade, plus suspense)", ex, "#ff9f43")):
        marks = "".join(f'<line x1="{4 + s / (len(x) - 1) * (W - 8):.1f}" y1="0" x2="{4 + s / (len(x) - 1) * (W - 8):.1f}" y2="{H}" class="tie"/>' for s in ties)
        out.append(f'<div class="ch"><div class="cl" style="color:{col}">{e(label)}</div>'
                   f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none">{marks}'
                   f'<path d="{path(x, ys, 0, len(x) - 1, 0, 1, W, H)}" stroke="{col}" /></svg></div>')
    sb = full["sb"]
    lab = " · ".join(f'{sb["a"][s]}–{sb["h"][s]}' for s in ties)
    return "".join(out) + f'<div class="axis"><span>kickoff</span><span>green lines = the plays that tied the game ({lab})</span><span>end of overtime</span></div>'


def chart_cut(st, axis=True):
    W, H = 900, 150
    t = st["t"]
    ticks = "".join(f'<line x1="{4 + x / st["dur"] * (W - 8):.1f}" y1="0" x2="{4 + x / st["dur"] * (W - 8):.1f}" y2="{H}" '
                    f'class="{"tie" if s in TIES else "beat"}"/>' for s, x in st["beats"])
    svg = (f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none">{ticks}'
           f'<path d="{path(t, st["song"], 0, st["dur"], 0, 1, W, H)}" stroke="#ff4f8b"/>'
           f'<path d="{path(t, st["game"], 0, st["dur"], 0, 1, W, H)}" stroke="#3ad1ff"/></svg>')
    if not axis:
        return svg
    return svg + (f'<div class="axis"><span>0 s</span><span><b style="color:#ff4f8b">music energy</b> vs '
                  f'<b style="color:#3ad1ff">game excitement</b> · grey = a key play · green = a play that tied the game · '
                  f'match {st["r"]:.2f}</span><span>{st["dur"]:.0f} s</span></div>')


def drill(st, L):
    rows = "".join(f'<tr class="{"ok" if ok else "no"}"><td>{x:5.1f} s</td><td>{e(txt)}{" <i>tie</i>" if tie else ""}</td>'
                   f'<td>{"on the hit" if ok else f"{o:.2f} s off"}</td></tr>' for x, txt, ok, o, tie in st["detail"])
    return (f'<template id="drill{L}"><h3>{L}-second cut: {st["on"]} of {st["n"]} key plays on a hit</h3>'
            f'<p class="sub">Each row is a key play: when it lands, the caption, and how far it is from the nearest hit in the music.</p>'
            f'<table class="dt">{rows}</table></template>')


def match_svg(job, L=120, W=400, H=150):
    """The song ↔ game match, as in the player: song energy on top, the game below, grey lines pair matched moments."""
    m = json.loads((ROOT / "outputs" / job / str(L) / "timeline.json").read_text(encoding="utf-8"))["match"]
    N, pad = len(m["song"]), 4
    band = (H - 3 * pad - 24) / 2
    top, bot = pad + 12, pad + 12 + band + pad + 12
    X = lambda i: pad + i / (N - 1) * (W - 2 * pad)
    Y = lambda ys, y0: lambda i: y0 + band - ys[max(0, min(N - 1, int(round(i))))] * band
    ya, yb = Y(m["song"], top), Y(m["game"], bot)
    pairs = "".join(f'M{X(i):.1f},{ya(i):.1f}L{X(j):.1f},{yb(j):.1f}' for i, j in m["path"])
    line = lambda ys, y: "".join(f'{"M" if i == 0 else "L"}{X(i):.1f},{y(i):.1f}' for i in range(N))
    return (f'<svg class="mx" viewBox="0 0 {W} {H}" preserveAspectRatio="none" aria-label="song energy vs the game">'
            f'<path d="{pairs}" stroke="#ffffff22" stroke-width="1"/>'
            f'<path d="{line(m["song"], ya)}" stroke="#ff4f8b"/><path d="{line(m["game"], yb)}" stroke="#3ad1ff"/>'
            f'<text x="6" y="11">SONG</text><text x="6" y="{bot - 2:.0f}">GAME</text></svg>')


MX_KEY = ('<div class="mxk"><span><i style="background:#ff4f8b"></i>song energy</span><span><i style="background:#3ad1ff"></i>the game</span>'
          '<span><i style="background:#ffffff55"></i>matched moments</span></div>')


def song_short(s):
    return next((v for k, v in SONGS if k in s["title"]), s["title"])


def card(g, small=False):
    job, home, away = g["job"], g["teams"]["home"], g["teams"]["away"]
    c1, c2 = (away["colour"], home["colour"]) if g["sport"] == "NFL" else ("#ffd23f", "#ff3b3b")
    parts = g["title"].split(" · ")
    when = " · ".join(parts[1:]) if g["sport"] == "NFL" else " · ".join(parts[:2])
    head = (g.get("story_title") if small else None) or g["final"]
    play = f"{PLAYER_LINK}#/play/{job}/"
    return (f'<div class="card{" sm" if small else ""}" data-href="{play}120" style="--c1:{c1};--c2:{c2}">'
            f'<div class="media"><video muted loop playsinline preload="none" poster="{MEDIA_BASE}poster_{job}.jpg" '
            f'data-src="{MEDIA_BASE}preview_{job}.mp4"></video><span class="sport">{e(g["sport"])}</span><span class="pl">▶</span></div>'
            f'<div class="body"><div class="song">♪ {e(song_short(g["song"]))}</div><h3>{e(head)}</h3>'
            f'<div class="when">{e(g["final"] if small else when)}</div>{match_svg(job)}'
            f'<p><b class="exp">What to expect:</b> {e(g.get("story") or "")}</p>'
            f'<div class="lens"><a href="{play}60">60 s</a><a href="{play}120">120 s</a></div></div></div>')


CSS = """
:root{--bg:#07090d;--panel:#11151d;--panel2:#161b25;--line:#252c39;--fg:#eef1f6;--mut:#8f99a8;--acc:#f2b33d;--pink:#ff4f8b;--blue:#3ad1ff;--green:#3ecf8e;
 --grad:linear-gradient(90deg,#ff4f8b,#f2b33d 50%,#3ad1ff)}
*{box-sizing:border-box} html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 Inter,system-ui,-apple-system,"Segoe UI",sans-serif;overflow-x:hidden}
body:before{content:"";position:fixed;inset:-20%;z-index:-1;pointer-events:none;
 background:radial-gradient(40% 35% at 15% 10%,#ff4f8b22,transparent 70%),radial-gradient(40% 35% at 85% 20%,#3ad1ff1f,transparent 70%),radial-gradient(45% 40% at 50% 100%,#f2b33d14,transparent 70%);
 animation:drift 18s ease-in-out infinite alternate}
@keyframes drift{to{transform:translate(3%,-2%) scale(1.06)}}
a{color:inherit}
.wrap{max-width:1180px;margin:0 auto;padding:0 22px}
h1,h2,h3,.num{font-family:"Space Grotesk",Inter,system-ui,sans-serif}
.grad{background:var(--grad);background-size:200% auto;-webkit-background-clip:text;background-clip:text;color:transparent;animation:shine 6s linear infinite}
@keyframes shine{to{background-position:200% center}}
/* hero */
.hero{min-height:78vh;display:grid;grid-template-columns:1fr;max-width:760px;align-items:center;padding:36px 0}
.game{display:grid;grid-template-columns:1fr 1.15fr;gap:22px;align-items:center;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px 20px;margin:0 0 18px}
.game>*{min-width:0} .game .meta{margin:2px 0 8px} .game p{margin:10px 0 0;color:#c9d0da} .game .hint{font-size:13px;color:var(--mut)}
.exp{color:var(--acc)}
svg.mx{height:92px;margin:8px 0 2px;background:#0b0e14;border-radius:8px}
svg.mx path{fill:none;stroke-width:1.8;vector-effect:non-scaling-stroke} svg.mx text{fill:#8f99a8;font:600 9px Inter,sans-serif;letter-spacing:.1em}
.gc svg.mx{height:190px;margin:0}
.mxk{display:flex;flex-wrap:wrap;gap:6px 16px;color:var(--mut);font-size:12.5px;margin-top:8px} .mxk i{display:inline-block;width:16px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
.game .tm{font:700 20px "Space Grotesk"} .game .tm .a{color:#0085CA} .game .tm .h{color:#FB4F14} .game .meta{color:var(--mut);font-size:14px}
.game .song{--c2:var(--acc)}
.cut{display:grid;grid-template-columns:1.6fr 1fr;gap:22px;align-items:center;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px;margin:0 0 16px}
.cut.rec{border-color:transparent;background:linear-gradient(var(--panel),var(--panel)) padding-box,var(--grad) border-box;box-shadow:0 0 40px #ff4f8b1f}
.cut>*{min-width:0} .cut video,.cut .nov{display:block;width:100%;height:auto;aspect-ratio:16/9;object-fit:contain;background:#000;border-radius:10px}
.cut .len{font:700 34px "Space Grotesk";line-height:1} .cut .len small{font-size:15px;color:var(--mut);font-weight:500;margin-left:6px}
.badge{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#0b0d12;background:var(--grad);border-radius:99px;padding:3px 10px;margin-bottom:10px}
.badge.alt{background:var(--panel2);color:var(--mut);border:1px solid var(--line)}
.cut h3{margin:8px 0 6px;font-size:19px} .cut p{color:#c9d0da;margin:0 0 8px} .cut .fig{color:var(--mut);font-size:13px}
.tag{display:inline-flex;align-items:center;gap:10px;font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--acc);border:1px solid #5a4a22;background:#f2b33d0d;border-radius:99px;padding:5px 13px}
.eq{display:inline-flex;gap:2px;align-items:flex-end;height:12px}.eq i{width:3px;background:var(--acc);border-radius:2px;animation:eq 1s ease-in-out infinite}
.eq i:nth-child(2){animation-delay:-.3s}.eq i:nth-child(3){animation-delay:-.6s}.eq i:nth-child(4){animation-delay:-.15s}
@keyframes eq{0%,100%{height:3px}50%{height:12px}}
h1{font-size:clamp(28px,3.4vw,42px);line-height:1.12;margin:18px 0 12px;letter-spacing:-.02em}
.q{color:#c5ccd6;font-size:16.5px;margin:0 0 18px;max-width:540px}
ul.how{list-style:none;padding:0;margin:0 0 24px;display:grid;gap:8px}
ul.how li{display:flex;gap:12px;align-items:center;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:9px 14px;color:#d3d9e2}
ul.how li b{color:var(--fg)} ul.how li .n{font:600 12px "Space Grotesk";color:var(--mut);width:18px}
ul.how li.main{border-color:transparent;background:linear-gradient(var(--panel2),var(--panel2)) padding-box,var(--grad) border-box;box-shadow:0 0 28px #ff4f8b26}
ul.how li.main .n{color:var(--acc)}
.cta{display:flex;flex-wrap:wrap;gap:10px}
.btn{display:inline-flex;align-items:center;gap:8px;padding:11px 18px;border-radius:99px;font-weight:600;text-decoration:none;border:1px solid var(--line);background:var(--panel);cursor:pointer;color:var(--fg);font-size:14.5px;transition:transform .15s,box-shadow .2s}
.btn:hover{transform:translateY(-2px)} .btn.pri{background:var(--grad);border:0;color:#0b0d12;box-shadow:0 8px 30px #ff4f8b40}
.screen{position:relative;border-radius:16px;padding:2px;background:var(--grad);background-size:300% 300%;animation:glow 8s ease infinite;box-shadow:0 20px 80px #000a,0 0 60px #3ad1ff22}
@keyframes glow{50%{background-position:100% 50%}}
.screen video{display:block;width:100%;border-radius:14px;background:#000;aspect-ratio:16/9;object-fit:cover}
.chip{position:absolute;left:16px;bottom:16px;background:#07090dcc;backdrop-filter:blur(6px);border:1px solid #ffffff22;border-radius:99px;padding:6px 13px;font-size:13px;display:flex;gap:8px;align-items:center}
.chip .dot{width:8px;height:8px;border-radius:50%;background:var(--pink);box-shadow:0 0 10px var(--pink);animation:pulse 1.2s infinite}
@keyframes pulse{50%{opacity:.35}}
.scrollhint{text-align:center;color:var(--mut);font-size:12px;letter-spacing:.14em;text-transform:uppercase;margin-top:-40px;padding-bottom:20px}
/* stats */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:10px 0 0}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px 18px;position:relative;overflow:hidden}
.stat:after{content:"";position:absolute;left:0;top:0;height:3px;width:100%;background:var(--grad)}
.stat .num{font-size:32px;font-weight:700;line-height:1.1;font-variant-numeric:tabular-nums}
.stat span{color:var(--mut);font-size:13px} .stat.click{cursor:pointer} .stat.click:hover{border-color:#ff4f8b88}
.stat .more{position:absolute;right:12px;top:12px;font-size:11px;color:var(--acc)}
/* sections */
section{padding:70px 0 0}
.kicker{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--blue);margin:0 0 6px}
h2{font-size:clamp(22px,2.4vw,30px);margin:0 0 6px;letter-spacing:-.01em;line-height:1.2}
.sub{color:var(--mut);margin:0 0 22px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.vid{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden}
.vh{display:flex;justify-content:space-between;align-items:baseline;padding:11px 15px} .vh span{color:var(--mut);font-size:13px}
.vid video,.nov{display:block;width:100%;aspect-ratio:16/9;background:#000} .nov{display:grid;place-items:center;color:var(--mut)}
ul.pts{list-style:none;padding:0;margin:18px 0 0;display:grid;gap:8px;max-width:900px}
ul.pts li{padding-left:22px;position:relative;color:#c9d0da} ul.pts li:before{content:"";position:absolute;left:0;top:.62em;width:10px;height:10px;border-radius:3px;background:var(--grad)}
mark{background:linear-gradient(transparent 55%,#f2b33d55 55%);color:var(--fg);padding:0 2px;font-weight:600}
/* cards */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}
.grid.four{grid-template-columns:repeat(4,1fr)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden;cursor:pointer;position:relative;transition:transform .2s,box-shadow .25s,border-color .25s}
.card:before{content:"";position:absolute;inset:0 0 auto 0;height:3px;background:linear-gradient(90deg,var(--c1),var(--c2));z-index:2}
.card:hover{transform:translateY(-4px);border-color:var(--c2);box-shadow:0 14px 40px #0008,0 0 0 1px var(--c2),0 0 40px color-mix(in srgb,var(--c2) 25%,transparent)}
.media{position:relative;aspect-ratio:16/9;background:#000;overflow:hidden}
.media video{width:100%;height:100%;object-fit:cover;display:block;transition:transform .5s}
.card:hover .media video{transform:scale(1.05)}
.sport{position:absolute;left:10px;top:12px;font:700 11px "Space Grotesk";letter-spacing:.12em;background:#07090dcc;border:1px solid #ffffff22;padding:3px 8px;border-radius:6px}
.pl{position:absolute;right:12px;bottom:12px;width:38px;height:38px;border-radius:50%;display:grid;place-items:center;background:#fff;color:#000;font-size:14px;opacity:0;transform:scale(.7);transition:.2s}
.card:hover .pl{opacity:1;transform:none}
.body{padding:13px 16px 15px} .body h3{margin:4px 0 0;font-size:18px} .when{color:var(--mut);font-size:12.5px}
.body p{margin:8px 0 10px;color:#c9d0da;font-size:14px;min-height:2.6em}
.song{display:inline-block;font-size:12px;font-weight:600;color:var(--c2);background:color-mix(in srgb,var(--c2) 14%,transparent);border-radius:99px;padding:2px 10px}
.lens{display:flex;gap:8px} .lens a{font-size:12.5px;text-decoration:none;border:1px solid var(--line);border-radius:99px;padding:3px 11px;color:#d6dbe2}
.lens a:hover{border-color:var(--c2);color:var(--fg)}
.card.sm .body h3{font-size:16.5px} .card.sm .body p{font-size:13.5px}
/* results */
.res{display:grid;grid-template-columns:1fr 250px;gap:18px;align-items:start}
.legend{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px 16px;font-size:13.5px;color:#c9d0da}
.legend div{display:flex;gap:10px;align-items:center;margin:6px 0} .legend i{display:inline-block;width:22px;height:3px;border-radius:2px}
.legend .r{margin-top:12px;padding-top:10px;border-top:1px solid var(--line)} .legend .r b{font:700 24px "Space Grotesk";color:var(--fg)}
.cl{font-size:13px;font-weight:600;margin:12px 0 3px;color:#d6dbe2}
svg{display:block;width:100%;height:120px;background:var(--panel);border:1px solid var(--line);border-radius:10px}
svg path{fill:none;stroke-width:2;vector-effect:non-scaling-stroke} .chart svg,.res svg{height:150px}
svg line.tie{stroke:var(--green);stroke-width:1.5;opacity:.75;vector-effect:non-scaling-stroke} svg line.beat{stroke:#5d6878;stroke-width:1;opacity:.6;vector-effect:non-scaling-stroke}
.axis{display:flex;justify-content:space-between;gap:10px;color:var(--mut);font-size:12px;margin-top:4px}
.jump{display:inline-block;margin-top:14px;color:var(--acc);font-weight:600;text-decoration:none;font-size:14px} .jump:hover{text-decoration:underline}
/* divider + the how */
.divider{margin:90px 0 0;padding:26px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line);text-align:center;color:var(--mut);font-size:15px}
.divider b{color:var(--fg)}
.howto{max-width:900px;padding-bottom:80px}
.howto h2{font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:var(--blue);margin:44px 0 6px;font-family:Inter,system-ui,sans-serif}
.howto h3{font-size:16px;margin:24px 0 4px} .howto p{color:#c5ccd6} .howto b{color:var(--fg)}
.note{border-left:3px solid var(--acc);background:var(--panel);padding:12px 16px;border-radius:0 8px 8px 0;color:#d6dbe2}
ol.caps{color:#d6dbe2} ol.caps li{margin:2px 0}
.cols{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px} .cols .c{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px}
.cols .c h3{margin-top:0} .cols .c ul{padding-left:18px;margin:6px 0;color:#c5ccd6}
.src{font-size:13px;color:var(--mut)} .src a{color:var(--blue)}
/* drill box */
dialog{background:var(--panel2);color:var(--fg);border:1px solid var(--line);border-radius:16px;padding:22px 24px;max-width:760px;width:calc(100% - 32px);box-shadow:0 30px 90px #000c}
dialog::backdrop{background:#000a;backdrop-filter:blur(3px)} dialog h3{margin:0 0 4px;font-size:18px}
.x{position:absolute;right:14px;top:10px;background:none;border:0;color:var(--mut);font-size:22px;cursor:pointer}
table.dt{width:100%;border-collapse:collapse;font-size:13.5px} .dt td{padding:6px 8px;border-top:1px solid var(--line)} .dt td:first-child{color:var(--mut);white-space:nowrap;font-variant-numeric:tabular-nums}
.dt td:last-child{white-space:nowrap;text-align:right;font-weight:600} .dt tr.ok td:last-child{color:var(--green)} .dt tr.no td:last-child{color:var(--pink)}
.dt i{font-style:normal;font-size:11px;color:var(--green);border:1px solid #3ecf8e66;border-radius:5px;padding:0 5px;margin-left:6px}
footer{color:var(--mut);font-size:12.5px;text-align:center;padding:30px 0 50px}
.rv{opacity:0;transform:translateY(18px);transition:opacity .7s,transform .7s} .rv.in{opacity:1;transform:none}
@media (max-width:900px){.hero{min-height:auto;padding-top:28px}.cut,.game{grid-template-columns:1fr}.scrollhint{display:none}.pair,.res{grid-template-columns:1fr}
 .stats{grid-template-columns:1fr 1fr}.grid.four{grid-template-columns:1fr 1fr}.cols{grid-template-columns:1fr}}
@media (max-width:560px){.wrap{padding:0 16px}.grid.four{grid-template-columns:1fr}.stat .num{font-size:26px}}
@media (prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}.rv{opacity:1;transform:none}}
"""

JS = """
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
// cards: click opens the player; previews load lazily and play on hover (desktop) or when in view (touch)
const touch = matchMedia('(hover: none)').matches;
$$('.card').forEach(c => {
  const v = c.querySelector('video');
  const start = () => { if (!v.src) v.src = v.dataset.src; v.play().catch(() => {}); };
  const stop = () => v.pause();
  if (!touch) { c.addEventListener('mouseenter', start); c.addEventListener('mouseleave', stop); }
  c.addEventListener('click', ev => { if (!ev.target.closest('a')) location.href = c.dataset.href; });
});
if (touch) {
  const io = new IntersectionObserver(es => es.forEach(en => {
    const v = en.target.querySelector('video');
    if (en.isIntersecting) { if (!v.src) v.src = v.dataset.src; v.play().catch(() => {}); } else v.pause();
  }), {threshold: .6});
  $$('.card').forEach(c => io.observe(c));
}
// reveal on scroll + count-up numbers
const rv = new IntersectionObserver(es => es.forEach(en => { if (en.isIntersecting) { en.target.classList.add('in'); rv.unobserve(en.target); } }), {threshold: .12});
$$('.rv').forEach(el => rv.observe(el));
const cu = new IntersectionObserver(es => es.forEach(en => {
  if (!en.isIntersecting) return; cu.unobserve(en.target);
  const el = en.target, to = +el.dataset.to, suf = el.dataset.suf || '', pre = el.dataset.pre || '', t0 = performance.now();
  const tick = now => { const k = Math.min(1, (now - t0) / 1100), v = Math.round(to * (1 - Math.pow(1 - k, 3)));
    el.textContent = pre + v + suf; if (k < 1) requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
}), {threshold: .5});
$$('[data-to]').forEach(el => cu.observe(el));
// hero button: jump to the featured cut and play the 2-minute version with sound
document.getElementById('watch120').addEventListener('click', () => {
  document.getElementById('watch').scrollIntoView({behavior: 'smooth'});
  const v = document.getElementById('v120'); if (v) setTimeout(() => { v.muted = false; v.play().catch(() => {}); }, 500);
});
// drill-down boxes
const dlg = document.getElementById('dlg');
$$('[data-drill]').forEach(el => el.addEventListener('click', () => {
  document.getElementById('dlgBody').replaceChildren(document.getElementById(el.dataset.drill).content.cloneNode(true));
  dlg.showModal();
}));
dlg.addEventListener('click', ev => { if (ev.target === dlg || ev.target.classList.contains('x')) dlg.close(); });
"""


def main():
    s60, s120, base = cut_stats(JOB, 60), cut_stats(JOB, 120), cut_stats(BASE, 60)
    credit = s60["tl"]["song"]["credit"]
    lo = min(s60["speeds"][0], s120["speeds"][0]); hi = max(s60["speeds"][1], s120["speeds"][1])
    caps60 = "".join(f"<li>{e(c['text'])}</li>" for c in s60["tl"]["captions"])
    gallery = {g["job"]: g for g in json.loads((ROOT / "outputs" / "gallery.json").read_text(encoding="utf-8"))}
    games = [gallery[j] for j in GAMES if j in gallery]
    same = [gallery[j] for j in SAME_GAME if j in gallery]
    n_games = len(games) + 1
    n_cuts = 2 * (len(games) + len(same))
    sports = len({g["sport"] for g in games} | {"NFL"})
    song_credits = "".join(f"<li>{e(g['final'])}: {e(g['song']['credit'])}</li>" for g in games + same[1:])

    def vid(L, st):
        url = VIDEO.get(L) or ""
        player = (f'<video id="v{L}" controls preload="metadata" playsinline poster="{MEDIA_BASE}hero_poster.jpg" src="{e(url)}"></video>'
                  if url else '<div class="nov">Video link coming</div>')
        return player

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Match Mashups</title>
<meta name="description" content="A whole game, one song, every big play on the beat: sports broadcasts cut automatically to music.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>
<div class="wrap">

<header class="hero">
 <div>
  <span class="tag"><span class="eq"><i></i><i></i><i></i><i></i></span>Match Mashups</span>
  <h1>A three-hour game. <span class="grad">One song.</span><br>Every big play on the beat.</h1>
  <p class="q">Can a program cut a full broadcast into a minute or two that <b>moves with the music</b> and tells a story?</p>
  <ul class="how">
   <li><span class="n">01</span><span><b>Finds every play</b> by reading the on-screen scoreboard</span></li>
   <li><span class="n">02</span><span><b>Scores the excitement</b> of each moment: surprise + suspense</span></li>
   <li><span class="n">03</span><span><b>Picks the song</b> whose energy rises and falls with the game</span></li>
   <li class="main"><span class="n">04</span><span><b>Lands every big play on a hit</b> in the music</span></li>
  </ul>
  <div class="cta"><button class="btn pri" id="watch120">▶ Watch the 2-min cut</button>
   <a class="btn" href="#games">See all games ↓</a><a class="btn" href="{PLAYER_LINK}">Open the player →</a></div>
 </div>
</header>
<div class="scrollhint">scroll ↓</div>

<div class="stats rv">
 <div class="stat"><div class="num" data-to="{n_games}">{n_games}</div><span>games · {sports} sports</span></div>
 <div class="stat"><div class="num" data-to="{n_cuts}">{n_cuts}</div><span>summaries, 60 s and 120 s each</span></div>
 <div class="stat click" data-drill="drill120"><span class="more">see plays ↗</span><div class="num">{s120['on']}/{s120['n']}</div><span>key plays land exactly on a hit (120 s cut)</span></div>
 <div class="stat"><div class="num" data-to="600" data-pre="~">~600</div><span>songs searched for the best match</span></div>
</div>

<section id="watch" class="rv">
 <p class="kicker">The featured cut</p>
 <h2>Neither side gives in</h2>
 <div class="game"><div class="gi"><span class="tm"><span class="a">Carolina Panthers</span> @ <span class="h">Cincinnati Bengals</span></span>
  <div class="meta">NFL · 2014 season, Week 6 · Final <b>37–37</b> after overtime</div><span class="song">♪ Vivaldi · Winter</span>
  <p><b class="exp">What to expect:</b> a rollercoaster. Every score is answered, each with less time left. Then a 36-yard kick misses wide right with 2 s to go.</p>
  <p class="hint">The chart is the selling point: the <b style="color:#ff4f8b">song's energy</b> was matched to <b style="color:#3ad1ff">the game's excitement</b>, so the music rises when the game does.</p></div>
  <div class="gc">{match_svg(JOB, W=560, H=190)}{MX_KEY}</div></div>
 <p class="sub">Same game, same song, two lengths:</p>
 <div class="cut rec">{vid(120, s120)}<div><span class="badge">★ Watch this one</span>
  <div class="len">120 s<small>the full story</small></div>
  <h3>The story has room to build</h3>
  <p>The music drops after each tie and climbs again, so you feel the <mark>answer, wait, answer</mark> rhythm that made this game.</p>
  <div class="fig">{s120['on']}/{s120['n']} key plays on a hit · shape match {s120['r']:.2f}</div></div></div>
 <div class="cut">{vid(60, s60)}<div><span class="badge alt">Short on time</span>
  <div class="len">60 s<small>the quick hit</small></div>
  <h3>Every big moment, back to back</h3>
  <p>All the scores and the miss, at full pace. <mark>Pure highlights</mark>, with less room for the tension between them.</p>
  <div class="fig">{s60['on']}/{s60['n']} key plays on a hit · shape match {s60['r']:.2f}</div></div></div>
</section>

<section id="games" class="rv">
 <p class="kicker">More games, more songs</p>
 <h2>Every game has its own shape</h2>
 <p class="sub">Each chart shows the game's shape against its song. Hover to preview, click to watch.</p>
 {MX_KEY}<div style="height:14px"></div>
 <div class="grid">{"".join(card(g) for g in games)}</div>
</section>

<section class="rv">
 <p class="kicker">Same game, different stories</p>
 <h2>One game. Four ways to tell it.</h2>
 <p class="sub">Panthers @ Bengals, 2014. Same footage; the story, the captions and the song change.</p>
 <div class="grid four">{"".join(card(g, small=True) for g in same)}</div>
</section>

<section id="results" class="rv">
 <p class="kicker">The result</p>
 <h2>Does the music follow the game?</h2>
 <p class="sub">The featured cut, second by second.</p>
 <div class="res">
  <div><div class="cl">120 seconds</div>{chart_cut(s120, axis=False)}<div class="cl">60 seconds</div>{chart_cut(s60, axis=False)}
   <div class="axis"><span>start</span><span>end</span></div></div>
  <div class="legend">
   <div><i style="background:var(--pink)"></i>Music energy</div>
   <div><i style="background:var(--blue)"></i>Game excitement</div>
   <div><i style="background:#5d6878;width:2px;height:14px"></i>A key play</div>
   <div><i style="background:var(--green);width:2px;height:14px"></i>A play that tied the game</div>
   <div class="r"><b>{s120['r']:.2f}</b> / {s60['r']:.2f}<br>shape match, 120 s / 60 s (1 = identical; first version {base['r']:.2f})</div>
   <div class="r stat click" data-drill="drill60" style="padding:10px 0 0;border:0;background:none"><b>{s60['on']}/{s60['n']}</b> plays on a hit in the 60 s cut ↗<br>(first version {base['on']}/{base['n']})</div>
  </div>
 </div>
 <ul class="pts">
  <li><mark>The numbers match; the feeling doesn't always follow.</mark> Every change that made a cut land came from a human: the song, the story, what to leave out.</li>
  <li><mark>Shape alone picks the wrong song.</mark> Calm waltzes and horror cues fit the curve on paper. The final pick was made by ear from the shortlist.</li>
  <li><mark>The music replaces the crowd and the commentators</mark>, the strongest cues in sport. That is the hardest part to win back.</li>
 </ul>
 <a class="jump" href="#how">How the excitement curve is measured ↓</a>
</section>

<div class="divider"><b>That is the result.</b> Everything below is the how.</div>

<div class="howto" id="how">
<p>Both featured videos were cut automatically from the full broadcast of one game. A program reads the on-screen
scoreboard to know which play is showing, scores how much every moment of the game matters, matches that against the rises
and falls of a piece of music, and lands every big play on a hit in the music. The story (which plays to show, and the
captions) was written by hand, and the song was picked by ear from the program's shortlist.</p>
<p class="note"><b>Why two lengths.</b> In 60 seconds every play is a peak and there is no room for the stretches in
between, so nothing can build. The 120-second cut can let the music drop after a tie and climb again, and that rhythm of
answer, wait, answer is what made this game. The 60-second cut keeps the moments; the 120-second cut keeps the tension.</p>

<h2>The game</h2>
<p><b>Panthers @ Bengals, 2014 season, week 6. Final: 37–37 after overtime</b>, one of the rare ties in the NFL. Cincinnati
led early. Two Andy Dalton interceptions handed Carolina the lead. Then every score in the fourth quarter was answered, each
with less time left: 24–24, 31–31 after a 97-yard kickoff return, a Bengals field goal with two minutes to go, and a
Panthers kick to tie it with five seconds left. In overtime each side kicked a field goal, and with two seconds left the
Bengals missed a 36-yard kick wide right. <b>Almost all of the tension comes from the clock running out.</b></p>

<h2>How it works</h2>
<h3>1. Which play is on screen</h3>
<p>The broadcast's scoreboard (quarter, clock, score, down) is read every half second and matched to the exact play in the
NFL's public play-by-play data. No clip is tagged by hand.</p>

<h3>2. How the game feels: a curve based on research</h3>
<p>The first version measured <i>tension</i>: how close the game is. A tied game late on scored as high as it gets, the whole
time it stayed tied. But viewers react to <b>change</b>. Economists call the two parts <b>surprise</b> (how far the chances of
winning just moved) and <b>suspense</b> (how far they could move next) (Ely, Frankel and Kamenica, 2015). Minute-by-minute TV
audiences for tennis and football rise with both, and surprise matters most (Wimbledon: two to five times as much). Fans'
heart rate spikes on a goal and then settles. So the new curve treats every swing in win probability as a spike that fades,
on top of a smaller suspense level. A tie is now a spike and then a held breath.</p>
{chart_game()}

<h3>3. Choosing the music</h3>
<p>Each piece of music gets an energy curve (loudness and how fast the sound changes), and every window of about 600 pieces
is compared with the cut's own excitement curve, allowing for small shifts in timing (dynamic time warping). Matching the
shape alone picked calm waltzes and horror cues that fit on paper but felt wrong. The final pick was made by ear from
well-known, dramatic pieces: Vivaldi's <i>Winter</i>, which fits this cut better than <i>In the Hall of the Mountain King</i>
did in the first version.</p>

<h3>4. Landing on the hits</h3>
<p>The strong accents in the music are detected, and each key play is moved so that its result (the ball crossing the line,
the kick going through) lands exactly on one, with the biggest swings on the biggest hits. The footage in between is sped up
or slowed slightly ({lo:.2f}× to {hi:.2f}×) to arrive on time; no play is cut.</p>
<div class="chart"><div class="cl">60 seconds</div>{chart_cut(s60)}</div>
<div class="chart"><div class="cl">120 seconds</div>{chart_cut(s120)}</div>

<h3>5. Telling it as a story</h3>
<p>The captions read as one story, with the clock counting down through them. The 60-second cut:</p>
<ol class="caps">{caps60}</ol>

<h2>What was learned</h2>
<div class="cols">
 <div class="c"><h3>Works</h3><ul><li>Finding every play in a full broadcast, automatically</li><li>Timing: plays land on the music within a frame</li>
  <li>A research-based measure of how a game feels</li><li>A new cut in minutes once the story is written</li></ul></div>
 <div class="c"><h3>Not yet</h3><ul><li>The gut reaction. The numbers match; the feeling doesn't always follow</li>
  <li>Every change that made it land came from a human: the song, the story, what to leave out</li></ul></div>
 <div class="c"><h3>Probably why</h3><ul><li>Suspense needs a side to root for, and a neutral viewer has none</li>
  <li>The music replaces the crowd and the commentators, the strongest cues in sport</li><li>60 seconds leaves no room to build</li></ul></div>
</div>
<p><b>Tried and dropped:</b> <i>In the Hall of the Mountain King</i> and <i>Danse Macabre</i> versions of the featured cut; tracks chosen by shape
alone; slow motion and on-screen clock call-outs at the late plays (too distracting, so the clock went into the captions).</p>

<h2>Credits and sources</h2>
<p class="src">Footage: the original broadcasts and official highlights of each game, used here only for a non-commercial
experiment; the videos are hosted separately and not part of this page. Featured cut music: {e(credit)}. Play-by-play and win probability:
<a href="https://github.com/nflverse">nflverse</a>.<br>
Research: J. Ely, A. Frankel, E. Kamenica, “Suspense and Surprise”, <i>Journal of Political Economy</i> 123(1), 2015 ·
P. Bizzozero, R. Flepp, E. Franck, “The importance of suspense and surprise in entertainment demand: evidence from Wimbledon”,
<i>Journal of Economic Behavior &amp; Organization</i>, 2016 · B. Buraimo, D. Forrest, I. McHale, J. D. Tena, “Unscripted drama:
soccer audience response to suspense, surprise, and shock”, <i>Economic Inquiry</i> 58(2), 2020 · T. Richardson, G. Nalbantis,
T. Pawlowski, “Emotional cues and the demand for televised sports”, <i>Journal of Sports Economics</i>, 2023.</p>
<p class="src">Music for the other cuts:</p><ul class="src">{song_credits}</ul>
</div>
<footer>Match Mashups · a personal experiment · <a href="{PLAYER_LINK}">open the player</a></footer>
</div>
<dialog id="dlg"><button class="x" aria-label="Close">×</button><div id="dlgBody"></div></dialog>
{drill(s60, 60)}{drill(s120, 120)}
<script>{JS}</script>
</body></html>
"""
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"page -> {out}  (60 s: {s60['on']}/{s60['n']} on hits, match {s60['r']:.2f}; 120 s: {s120['on']}/{s120['n']}, "
          f"{s120['r']:.2f}; first version {base['on']}/{base['n']}, {base['r']:.2f}; {len(games)} game cards; videos set: {[L for L in VIDEO if VIDEO[L]]})")


if __name__ == "__main__":
    main()
