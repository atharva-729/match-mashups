"""Build the public write-up page: docs/index.html (one self-contained file; GitHub Pages can serve docs/).

Two cuts of one game (Panthers @ Bengals, 2014 Week 6, 37-37 OT) to Vivaldi's Winter, 60 s and 120 s, plus how they
were made and what was learned. Every number and chart is computed here from the cuts' own data (timelines, plans,
the excitement curve), not typed in. No footage is in the page: the finished videos are hosted elsewhere and linked
by URL (VIDEO below); until a URL is set, its player says the link is coming.
usage: python scripts/build_page.py
"""
import html, json, sys

import numpy as np

import excitement
import sister
from common import ROOT, Job

JOB, BASE = "nfl_car_cin_excite", "nfl_car_cin_blink"     # the final cuts; the first version, for comparison
GID = "nfl_2014_06_CAR_CIN"
VIDEO = {60: "", 120: ""}                                   # the hosted share videos (set when the links exist)
PLAYER_LINK = ""                                            # the interactive player (export_site.py sets it)
TIES = {131, 150, 177, 208}


def cut_stats(name, L):
    job = Job(name, L)
    rep = {int(k): v for k, v in job.load("plan_report.json").items()}
    by = {b["step"]: b for b in job.story["cuts"][L]}
    steps = [s for s in by if (rep.get(s) or {}).get("planned") is not None]
    e = [rep[s]["planned"] for s in steps]
    tl = json.loads((job.out_l / "timeline.json").read_text(encoding="utf-8"))
    t, cc = excitement.cut_curve(GID, [by[s] for s in steps], e, tl["duration"])
    m = tl["match"]
    song = np.interp(t, np.linspace(m["t0"], m["t1"], len(m["song"])), m["song"])
    ph = np.array(job.load("alignment.json")["hits"])
    on = sum(1 for x in e if np.min(np.abs(ph[:, 0] - x)) < 0.05)
    speeds = [x["speed"] for x in job.load("edl.json")]
    return {"t": t, "song": song, "game": (cc - cc.min()) / max(1e-9, np.ptp(cc)), "r": float(np.corrcoef(song, cc)[0, 1]),
            "beats": [(s, x) for s, x in zip(steps, e)], "on": on, "n": len(steps), "dur": tl["duration"],
            "speeds": (min(speeds), max(speeds)), "tl": tl}


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
        out.append(f'<div class="ch"><div class="cl" style="color:{col}">{html.escape(label)}</div>'
                   f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none">{marks}'
                   f'<path d="{path(x, ys, 0, len(x) - 1, 0, 1, W, H)}" stroke="{col}" /></svg></div>')
    sb = full["sb"]
    lab = " · ".join(f'{sb["a"][s]}–{sb["h"][s]}' for s in ties)
    return "".join(out) + f'<div class="axis"><span>kickoff</span><span>green lines = the plays that tied the game ({lab})</span><span>end of overtime</span></div>'


def chart_cut(st):
    W, H = 900, 150
    t = st["t"]
    ticks = "".join(f'<line x1="{4 + x / st["dur"] * (W - 8):.1f}" y1="0" x2="{4 + x / st["dur"] * (W - 8):.1f}" y2="{H}" '
                    f'class="{"tie" if s in TIES else "beat"}"/>' for s, x in st["beats"])
    return (f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none">{ticks}'
            f'<path d="{path(t, st["song"], 0, st["dur"], 0, 1, W, H)}" stroke="#ff4f8b"/>'
            f'<path d="{path(t, st["game"], 0, st["dur"], 0, 1, W, H)}" stroke="#3ad1ff"/></svg>'
            f'<div class="axis"><span>0 s</span><span><b style="color:#ff4f8b">music energy</b> vs '
            f'<b style="color:#3ad1ff">game excitement</b> · grey = a key play · green = a play that tied the game · '
            f'match {st["r"]:.2f}</span><span>{st["dur"]:.0f} s</span></div>')


def video_block(L, st):
    url = VIDEO.get(L) or ""
    player = (f'<video controls preload="metadata" playsinline src="{html.escape(url)}"></video>' if url
              else '<div class="nov">Video link coming</div>')
    return f'<div class="vid"><div class="vh"><b>{L} seconds</b><span>{st["on"]}/{st["n"]} key plays on a hit · match {st["r"]:.2f}</span></div>{player}</div>'


def main():
    s60, s120, base = cut_stats(JOB, 60), cut_stats(JOB, 120), cut_stats(BASE, 60)
    credit = s60["tl"]["song"]["credit"]
    lo = min(s60["speeds"][0], s120["speeds"][0]); hi = max(s60["speeds"][1], s120["speeds"][1])
    caps60 = "".join(f"<li>{html.escape(c['text'])}</li>" for c in s60["tl"]["captions"])
    player_link = (f'<p class="more"><a href="{PLAYER_LINK}">Open the interactive player: every cut, with the captions, meter '
                   f'and song-to-game chart live →</a></p>' if PLAYER_LINK else "")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Neither side gives in: a game cut to Vivaldi</title>
<style>
:root{{--bg:#0e1116;--panel:#171c24;--line:#2a323e;--fg:#e8ebf0;--mut:#8d97a6;--acc:#f2b33d;--blue:#5aa9ff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:980px;margin:0 auto;padding:40px 22px 70px}}
.tag{{display:inline-block;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--acc);border:1px solid #5a4a22;border-radius:99px;padding:2px 10px;margin-bottom:12px}}
h1{{font-size:30px;line-height:1.2;margin:0 0 12px;letter-spacing:-.01em}} h1 em{{font-style:normal;color:var(--acc)}}
.lede{{color:#c5ccd6;font-size:16px;max-width:860px;margin:0}}
h2{{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--blue);margin:44px 0 6px}}
h3{{font-size:16px;margin:24px 0 4px}} p{{max-width:860px;color:#c5ccd6}} b{{color:var(--fg)}}
.tiles{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:26px 0 0}}
.tile{{background:var(--panel);border:1px solid var(--line);border-top:3px solid var(--acc);border-radius:10px;padding:12px 14px}}
.tile b{{display:block;font-size:27px;line-height:1.15;font-variant-numeric:tabular-nums}} .tile span{{color:var(--mut);font-size:12.5px}}
.vids{{display:grid;grid-template-columns:1fr;gap:18px}}
.vid{{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}}
.vh{{display:flex;justify-content:space-between;align-items:baseline;padding:10px 14px}} .vh span{{color:var(--mut);font-size:13px}}
video,.nov{{display:block;width:100%;aspect-ratio:16/9;background:#000}} .nov{{display:grid;place-items:center;color:var(--mut)}}
.note{{border-left:3px solid var(--acc);background:var(--panel);padding:12px 16px;border-radius:0 8px 8px 0;max-width:860px;color:#d6dbe2}}
.ch{{margin:10px 0}} .cl{{font-size:13px;font-weight:600;margin-bottom:2px}}
svg{{display:block;width:100%;height:120px;background:var(--panel);border:1px solid var(--line);border-radius:8px}}
svg path{{fill:none;stroke-width:2;vector-effect:non-scaling-stroke}} .chart svg{{height:150px}}
svg line.tie{{stroke:#3ecf8e;stroke-width:1.5;opacity:.75;vector-effect:non-scaling-stroke}} svg line.beat{{stroke:#5d6878;stroke-width:1;opacity:.6;vector-effect:non-scaling-stroke}}
.axis{{display:flex;justify-content:space-between;gap:10px;color:var(--mut);font-size:12px;margin-top:4px}}
ol.caps{{color:#d6dbe2;max-width:860px}} ol.caps li{{margin:2px 0}}
.cols{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}} .card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px}}
.card h3{{margin-top:0}} .card ul{{padding-left:18px;margin:6px 0;color:#c5ccd6}}
h2.hl{{color:var(--acc);font-size:13px;margin-top:30px}} h2.hl i{{text-transform:none;font-style:italic}}
p.more a{{color:var(--acc);font-weight:600;text-decoration:none}} p.more a:hover{{text-decoration:underline}}
.src{{font-size:13px;color:var(--mut)}} .src a{{color:var(--blue)}}
@media (max-width:760px){{.tiles,.cols{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<div class="tag">Experiment · automatic sports summaries</div>
<h1>Can a program cut a three-hour NFL broadcast into a minute that <em>moves with the music</em>?</h1>
<p class="lede">Both videos below were cut automatically from the full broadcast of one game. A program reads the on-screen
scoreboard to know which play is showing, scores how much every moment of the game matters, matches that against the rises
and falls of a piece of music, and lands every big play on a hit in the music. The story (which plays to show, and the
captions) was written by hand, and the song was picked by ear from the program's shortlist.</p>
<h2 class="hl">Highlights · Vivaldi's <i>Winter</i>, 60 s and 120 s</h2>
<div class="vids">{video_block(60, s60)}{video_block(120, s120)}</div>
<p class="note"><b>Why two lengths.</b> In 60 seconds every play is a peak and there is no room for the stretches in
between, so nothing can build. The 120-second cut can let the music drop after a tie and climb again, and that rhythm of
answer, wait, answer is what made this game. The 60-second cut keeps the moments; the 120-second cut keeps the tension.</p>

{player_link}
<div class="tiles">
 <div class="tile"><b>{s60['on']}/{s60['n']}</b><span>key plays land exactly on a hit in the music (60 s cut; {base['on']}/{base['n']} in the first version)</span></div>
 <div class="tile"><b>{s60['r']:.2f}</b><span>how closely the music follows the game, 0 to 1 (first version: {base['r']:.2f})</span></div>
 <div class="tile"><b>~600</b><span>pieces of music searched; Vivaldi's <i>Winter</i> chosen from the shortlist</span></div>
 <div class="tile"><b>60 / 120 s</b><span>the same game, song and method at two lengths</span></div>
</div>

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
 <div class="card"><h3>Works</h3><ul><li>Finding every play in a full broadcast, automatically</li><li>Timing: plays land on the music within a frame</li>
  <li>A research-based measure of how a game feels</li><li>A new cut in minutes once the story is written</li></ul></div>
 <div class="card"><h3>Not yet</h3><ul><li>The gut reaction. The numbers match; the feeling doesn't always follow</li>
  <li>Every change that made it land came from a human: the song, the story, what to leave out</li></ul></div>
 <div class="card"><h3>Probably why</h3><ul><li>Suspense needs a side to root for, and a neutral viewer has none</li>
  <li>The music replaces the crowd and the commentators, the strongest cues in sport</li><li>60 seconds leaves no room to build</li></ul></div>
</div>
<p><b>Tried and dropped:</b> <i>In the Hall of the Mountain King</i> and <i>Danse Macabre</i> versions; tracks chosen by shape
alone; slow motion and on-screen clock call-outs at the late plays (too distracting, so the clock went into the captions).</p>

<h2>Credits and sources</h2>
<p class="src">Footage: the FOX broadcast of the game, used here only for a non-commercial experiment; the videos are hosted
separately and not part of this page. Music: {html.escape(credit)}. Play-by-play and win probability:
<a href="https://github.com/nflverse">nflverse</a>.<br>
Research: J. Ely, A. Frankel, E. Kamenica, “Suspense and Surprise”, <i>Journal of Political Economy</i> 123(1), 2015 ·
P. Bizzozero, R. Flepp, E. Franck, “The importance of suspense and surprise in entertainment demand: evidence from Wimbledon”,
<i>Journal of Economic Behavior &amp; Organization</i>, 2016 · B. Buraimo, D. Forrest, I. McHale, J. D. Tena, “Unscripted drama:
soccer audience response to suspense, surprise, and shock”, <i>Economic Inquiry</i> 58(2), 2020 · T. Richardson, G. Nalbantis,
T. Pawlowski, “Emotional cues and the demand for televised sports”, <i>Journal of Sports Economics</i>, 2023.</p>
</main></body></html>
"""
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"page -> {out}  (60 s: {s60['on']}/{s60['n']} on hits, match {s60['r']:.2f}; 120 s: {s120['on']}/{s120['n']}, "
          f"{s120['r']:.2f}; first version {base['on']}/{base['n']}, {base['r']:.2f}; videos set: {[L for L in VIDEO if VIDEO[L]]})")


if __name__ == "__main__":
    main()
