/* Match Mashups v2 player. No build step, no framework.
   The <video> is muted; the two audio stems (game sound, song) follow video.currentTime and are resynced when they
   drift more than 80 ms. Captions, the song credit and the intensity visual are lookups into timeline.json. */
(() => {
  const $ = (id) => document.getElementById(id);
  const OUT = "../outputs/";
  const DRIFT_MAX = 0.08, XFADE_MS = 300, TITLE_MS = 2000;

  const st = {
    games: [], tl: null, job: null, L: null,
    captions: false, song: false, visual: false, voiceOn: false, duck: 1,
    style: readPref("vizStyle", "A"),
    vol: { game: 1, song: 0 }, fade: null,
    capIdx: -1, ended: false, started: false,
    sync: { max: 0, resyncs: 0, samples: 0, since: 0, sum: 0 },
  };
  window.__mm = st;                 // for the automated checks (Checkpoint 6)

  function readPref(k, d) { try { return localStorage.getItem("mm." + k) || d; } catch (e) { return d; } }
  function writePref(k, v) { try { localStorage.setItem("mm." + k, v); } catch (e) { /* private mode */ } }
  const mmss = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  /* ---------------- gallery ---------------- */
  async function loadGallery() {
    if (st.games.length) return st.games;
    try {
      const r = await fetch(OUT + "gallery.json", { cache: "no-cache" });
      st.games = r.ok ? await r.json() : [];
    } catch (e) { st.games = []; }
    return st.games;
  }

  function renderGallery() {
    const cards = $("cards");
    cards.innerHTML = st.games.map((g) => {
      const [match, when] = g.title.split(" · ");
      const a = g.teams.away, h = g.teams.home;
      return `<article class="gcard">
        <div class="teams"><span class="chip" style="background:${a.colour}"></span>${esc(a.name)}
          <span class="vs">@</span><span class="chip" style="background:${h.colour}"></span>${esc(h.name)}</div>
        <div><span class="final">${esc(g.final)}</span> <span class="when">· ${esc(when || g.season)}</span></div>
        <div class="song">Song: <b>${esc(g.song.title)}</b> · ${esc(g.song.artist)}</div>
        <div class="row">
          ${g.lengths.map((L) => `<a class="btn" href="#/play/${g.job}/${L}">${L} s</a>`).join("")}
          ${g.youtube ? `<a class="yt" href="${esc(g.youtube)}" target="_blank" rel="noopener">Original game on YouTube ↗</a>` : ""}
        </div></article>`;
    }).join("");
    $("empty").hidden = st.games.length > 0;
  }

  /* ---------------- player ---------------- */
  const video = $("video"), game = $("game"), song = $("song");
  const audios = () => [game, song];

  async function openPlayer(job, L) {
    teardown();
    st.job = job; st.L = +L;
    const base = `${OUT}${job}/${L}/`;
    const r = await fetch(base + "timeline.json", { cache: "no-cache" });
    if (!r.ok) { location.hash = "#/"; return; }
    st.tl = await r.json();
    const tl = st.tl;
    video.src = base + tl.files.video;
    game.src = base + tl.files.game;
    song.src = base + tl.files.song;
    st.capIdx = -1; st.ended = false; st.started = false;
    st.sync = { max: 0, resyncs: 0, samples: 0, since: 0, sum: 0 };
    // the demo flow: raw first (game sound on, captions/song/visual off); switching length (L) keeps the toggles
    const carry = st.carry || { captions: false, song: false, visual: false, voiceOn: false };
    st.carry = null;
    st.captions = carry.captions; st.song = carry.song; st.visual = carry.visual;
    st.voiceOn = false; st.duck = 1;
    st.vol = { game: st.song ? 0 : 1, song: st.song ? 1 : 0 }; applyVolumes();
    const [match, when] = tl.title.split(" · ");
    $("tMatch").textContent = match;
    $("tMeta").textContent = `${when || tl.season} · ${L}-second summary · cut automatically from the full broadcast`;
    $("tLink").href = tl.youtube || "#"; $("tLink").hidden = !tl.youtube;
    $("ytLink").href = tl.youtube || "#"; $("ytLink").hidden = !tl.youtube;
    $("eFinal").textContent = tl.final;
    $("eSong").textContent = `Song: ${tl.song.credit}`;
    $("eLink").href = tl.youtube || "#"; $("eLink").textContent = tl.youtube || "";
    $("credit").innerHTML = `♪ <b>${esc(tl.song.title)}</b><br>${esc(tl.song.credit)}`;
    placeCredit(tl);
    placeLower(tl);
    $("vizAway").textContent = tl.teams.away.abbr;
    $("vizHome").textContent = tl.teams.home.abbr;
    $("fillAway").style.background = tl.teams.away.colour;
    $("fillHome").style.background = tl.teams.home.colour;
    const other = otherLength();
    $("tLlabel").textContent = `${L} s` + (other ? ` → ${other}` : "");
    $("tL").disabled = !other;
    $("endCard").hidden = true;
    $("titleCard").classList.remove("fade"); $("titleCard").hidden = false;
    $("bigPlay").hidden = true;
    paintToggles(); drawSpark(0);
    setTimeout(() => { $("titleCard").classList.add("fade"); start(); }, TITLE_MS);
  }

  /* the song credit sits top-right, beside the broadcast's score bug; with no room beside it, just below it */
  function placeCredit(tl) {
    const el = $("credit"), [W, H] = tl.video_size;
    el.style.top = "2%"; el.style.maxWidth = "23%";
    if (!tl.bug) return;
    const [bx, by, bw, bh] = tl.bug;
    const room = 100 - ((bx + bw) / W) * 100 - 1.4 - 1.2;       // right margin + a gap to the bug
    if (room >= 15) el.style.maxWidth = `${Math.min(23, room).toFixed(1)}%`;
    else el.style.top = `${(((by + bh) / H) * 100 + 1.5).toFixed(1)}%`;
  }

  /* the win-probability bar and the captions sit low; when the broadcast's own scoreboard is low too (CBS 2021,
     ESPN 2024), they move up to sit just above it */
  function placeLower(tl) {
    let base = 11.5;
    if (tl.bug) {
      const [, by, , bh] = tl.bug, H = tl.video_size[1];
      if ((by + bh) / H > 0.6) base = Math.max(base, (1 - by / H) * 100 + 1.5);
    }
    $("vizA").style.bottom = `${base.toFixed(1)}%`;
    $("cap").style.bottom = `${(base + 5.5).toFixed(1)}%`;
  }

  function otherLength() {
    const g = st.games.find((x) => x.job === st.job);
    if (!g) return null;
    return g.lengths.find((x) => x !== st.L) || null;
  }

  async function start() {
    st.started = true;
    video.currentTime = 0;
    for (const a of audios()) a.currentTime = 0;
    // only a NotAllowedError means "autoplay with sound needs a click first"; an AbortError just means a seek or a
    // buffering pause interrupted the play() call, and the 'playing' handler restarts the stems
    const blocked = (e) => {
      if (e && e.name === "NotAllowedError") { for (const m of [video, ...audios()]) m.pause(); $("bigPlay").hidden = false; }
    };
    try { await video.play(); } catch (e) { blocked(e); return; }
    for (const a of audios()) a.play().catch(blocked);
  }

  function teardown() {
    for (const m of [video, game, song]) { m.pause(); m.removeAttribute("src"); m.load(); }
    if (st.fade) cancelAnimationFrame(st.fade);
    st.fade = null; st.tl = null;
  }

  function playPause() {
    if (!st.tl) return;
    if (st.ended) { replay(); return; }
    if (video.paused) {
      $("bigPlay").hidden = true;
      if (!st.started) { $("titleCard").classList.add("fade"); }
      st.started = true;
      video.play(); for (const a of audios()) a.play().catch(() => {});
    } else {
      video.pause(); for (const a of audios()) a.pause();
    }
  }

  function replay() {
    st.ended = false; $("endCard").hidden = true; st.capIdx = -1;
    video.currentTime = 0; for (const a of audios()) a.currentTime = 0;
    video.play(); for (const a of audios()) a.play().catch(() => {});
  }

  function seek(t) {
    video.currentTime = t;
    for (const a of audios()) a.currentTime = t;
    st.sync.since = performance.now();
    if (st.ended && t < st.tl.duration - 0.5) { st.ended = false; $("endCard").hidden = true; }
  }

  /* audio follows the video */
  video.addEventListener("play", () => { for (const a of audios()) if (a.paused) { a.currentTime = video.currentTime; a.play().catch(() => {}); } st.sync.since = performance.now(); });
  video.addEventListener("pause", () => { for (const a of audios()) a.pause(); });
  video.addEventListener("seeking", () => { for (const a of audios()) a.currentTime = video.currentTime; st.sync.since = performance.now(); });
  video.addEventListener("waiting", () => { for (const a of audios()) a.pause(); });
  video.addEventListener("playing", () => { for (const a of audios()) { a.currentTime = video.currentTime; a.play().catch(() => {}); } st.sync.since = performance.now(); });
  video.addEventListener("ended", () => {
    st.ended = true; for (const a of audios()) a.pause();
    $("endCard").hidden = false; $("cap").classList.remove("on");
  });

  function syncAudio() {
    if (video.paused || video.seeking || !st.tl) return;
    const t = video.currentTime;
    const settled = performance.now() - st.sync.since > 1000;   // ignore the first second after play/seek
    for (const a of audios()) {
      if (a.paused || a.readyState < 2) continue;
      const d = Math.abs(a.currentTime - t);
      if (settled) { st.sync.max = Math.max(st.sync.max, d); st.sync.samples++; st.sync.sum += d; }
      // resync, then leave that track alone for 0.5 s (a seek needs a moment to settle; resyncing again
      // straight away makes it thrash)
      if (d > DRIFT_MAX && performance.now() - (a._resyncAt || 0) > 500) { a.currentTime = t; a._resyncAt = performance.now(); st.sync.resyncs++; }
    }
  }

  /* S: crossfade game sound <-> song */
  function applyVolumes() {
    const clamp = (v) => Math.max(0, Math.min(1, v));
    game.volume = clamp(st.vol.game * st.duck); song.volume = clamp(st.vol.song * st.duck);
  }
  /* O: while a voice-over line is spoken, the song (or the game sound) is turned down */
  let lastT = 0;
  function updateDuck(t) {
    const now = performance.now(), dt = Math.min(0.1, (now - lastT) / 1000); lastT = now;
    const spans = (st.tl && st.tl.voice) || [];
    const speaking = st.voiceOn && spans.some(([a, b]) => t >= a - 0.15 && t < b + 0.1);
    const target = speaking ? 0.3 : 1;
    const k = Math.min(1, dt / 0.12);
    const d = st.duck + (target - st.duck) * k;
    if (Math.abs(d - st.duck) > 1e-3 || (st.duck !== target && Math.abs(d - target) < 1e-3)) { st.duck = Math.abs(d - target) < 1e-3 ? target : d; applyVolumes(); }
  }
  function crossfade(toSong) {
    const from = { ...st.vol }, to = { game: toSong ? 0 : 1, song: toSong ? 1 : 0 };
    const t0 = performance.now();
    if (st.fade) cancelAnimationFrame(st.fade);
    const step = (now) => {
      const k = Math.min(1, (now - t0) / XFADE_MS);
      st.vol = { game: from.game + (to.game - from.game) * k, song: from.song + (to.song - from.song) * k };
      applyVolumes();
      st.fade = k < 1 ? requestAnimationFrame(step) : null;
    };
    st.fade = requestAnimationFrame(step);
  }

  /* toggles */
  function toggle(k) {
    if (!st.tl) return;
    if (k === "C") st.captions = !st.captions;
    if (k === "S") { st.song = !st.song; crossfade(st.song); }
    if (k === "V") st.visual = !st.visual;
    if (k === "F") {
      if (document.fullscreenElement) document.exitFullscreen();
      else $("stage").requestFullscreen().catch(() => {});
      return;
    }
    if (k === "L") {
      const o = otherLength();
      if (o) { st.carry = { captions: st.captions, song: st.song, visual: st.visual, voiceOn: st.voiceOn }; location.hash = `#/play/${st.job}/${o}`; }
      return;
    }
    st.capIdx = -1;
    paintToggles();
  }
  function paintToggles() {
    $("tC").classList.toggle("on", st.captions);
    $("tS").classList.toggle("on", st.song);
    $("tSlabel").textContent = st.song ? "Song" : "Song";
    $("tS").title = st.song ? "Playing the song (S for the game sound)" : "Playing the game sound (S for the song)";
    $("tV").classList.toggle("on", st.visual);
    $("credit").classList.toggle("on", st.song);
    $("vizA").classList.toggle("on", st.visual);
    $("glow").classList.toggle("on", st.visual);
    if (!st.captions) $("cap").classList.remove("on");
  }

  /* captions */
  let capTimer = null;
  function updateCaption(t) {
    if (!st.captions || !st.tl) return;
    const caps = st.tl.captions;
    const i = caps.findIndex((c) => c.start <= t && t < c.end);
    if (i === st.capIdx) return;
    st.capIdx = i;
    const el = $("cap");
    el.classList.remove("on");
    clearTimeout(capTimer);
    if (i < 0) return;
    capTimer = setTimeout(() => { $("capText").textContent = caps[i].text; el.classList.add("on"); }, 150);
  }

  /* visual: momentum glow (edges in the colour of the team gaining win probability over the last ~8 s, brighter when
     the game is tense) + a win-probability bar with live percentages */
  function sample(t) {
    const s = st.tl.samples;
    return s[Math.max(0, Math.min(s.length - 1, Math.floor(t * st.tl.rate)))];
  }
  let lastViz = -1;
  function updateViz(t) {
    if (!st.visual || !st.tl) return;
    const k = Math.floor(t * 4);
    if (k === lastViz) return;
    lastViz = k;
    const [, wp, ten] = sample(t), [, wp0] = sample(Math.max(0, t - 8));
    const tm = st.tl.teams, away = 1 - wp;
    $("fillAway").style.width = `${(away * 100).toFixed(1)}%`;
    $("fillHome").style.width = `${(wp * 100).toFixed(1)}%`;
    $("pAway").textContent = `${Math.round(away * 100)}%`;
    $("pHome").textContent = `${Math.round(wp * 100)}%`;
    const swing = wp - wp0;                                   // > 0: home gaining
    const col = swing >= 0 ? tm.home.colour : tm.away.colour;
    const strength = Math.min(1, 0.25 + 0.75 * ten) * Math.min(1, 0.35 + Math.abs(swing) * 6);
    const px = (3 + 9 * strength).toFixed(1), a = Math.round(90 + 165 * strength).toString(16).padStart(2, "0");
    $("glow").style.boxShadow = `inset 0 0 ${px}cqw ${(+px / 4).toFixed(1)}cqw ${col}${a}`;
  }
  function drawSpark() {}

  /* frame loop */
  function frame() {
    if (st.tl) {
      const t = video.currentTime || 0, d = st.tl.duration;
      syncAudio();
      updateDuck(t);
      updateCaption(t);
      updateViz(t);
      $("done").style.width = `${Math.min(100, (t / d) * 100)}%`;
      $("knob").style.left = `${Math.min(100, (t / d) * 100)}%`;
      $("time").textContent = `${mmss(t)} / ${mmss(d)}`;
      $("playBtn").textContent = video.paused ? "▶" : "❚❚";
      const avg = st.sync.samples ? (st.sync.sum / st.sync.samples) * 1000 : 0;
      $("drift").textContent = st.sync.samples ? `sync ${avg.toFixed(0)} ms avg · ${(st.sync.max * 1000).toFixed(0)} max` : "";
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  /* wiring */
  $("playBtn").onclick = playPause;
  $("bigPlay").onclick = playPause;
  $("again").onclick = replay;
  $("video").onclick = playPause;
  $("video").ondblclick = () => toggle("F");
  // seek bar: click or drag
  const scrubAt = (e) => {
    const r = $("scrub").getBoundingClientRect();
    return Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * st.tl.duration;
  };
  let dragging = false;
  $("scrub").addEventListener("pointerdown", (e) => {
    if (!st.tl) return;
    dragging = true; $("scrub").classList.add("drag"); $("scrub").setPointerCapture(e.pointerId);
    seek(scrubAt(e));
  });
  $("scrub").addEventListener("pointermove", (e) => { if (dragging) seek(scrubAt(e)); });
  const endDrag = () => { dragging = false; $("scrub").classList.remove("drag"); };
  $("scrub").addEventListener("pointerup", endDrag);
  $("scrub").addEventListener("pointercancel", endDrag);
  const skip = (d) => { if (st.tl) seek(Math.max(0, Math.min(st.tl.duration - 0.1, video.currentTime + d))); };
  $("back10").onclick = () => skip(-10);
  $("fwd10").onclick = () => skip(10);
  for (const id of ["tC", "tS", "tV", "tF", "tL"]) $(id).onclick = () => toggle($(id).dataset.key);
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toUpperCase();
    if ("CSVLF".includes(k) && k.length === 1) { toggle(k); e.preventDefault(); }
    if (e.key === " ") { playPause(); e.preventDefault(); }
    if (e.key === "ArrowLeft") { skip(-10); e.preventDefault(); }
    if (e.key === "ArrowRight") { skip(10); e.preventDefault(); }
  });

  async function route() {
    const m = location.hash.match(/^#\/play\/([^/]+)\/(\d+)/);
    await loadGallery();
    if (m) {
      $("gallery").hidden = true; $("player").hidden = false;
      openPlayer(m[1], m[2]);
    } else {
      teardown();
      $("player").hidden = true; $("gallery").hidden = false;
      renderGallery();
    }
  }
  window.addEventListener("hashchange", route);
  route();
})();
