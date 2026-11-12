/* Match Mashups v2 player. No build step, no framework.
   The <video> is muted; the two audio stems (game sound, song) follow video.currentTime and are resynced when they
   drift more than 80 ms. Captions, the song credit and the intensity visual are lookups into timeline.json. */
(() => {
  const $ = (id) => document.getElementById(id);
  const OUT = "../outputs/";
  const DRIFT_MAX = 0.06, XFADE_MS = 300, TITLE_MS = 2000;

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
  const video = $("video"), game = $("game"), song = $("song"), voice = $("voice");
  const audios = () => (st.tl && st.tl.files.voice ? [game, song, voice] : [game, song]);

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
    if (tl.files.voice) voice.src = base + tl.files.voice; else voice.removeAttribute("src");
    st.capIdx = -1; st.ended = false; st.started = false;
    st.sync = { max: 0, resyncs: 0, samples: 0, since: 0, sum: 0 };
    // the demo flow: raw first (game sound on, captions/song/visual off); switching length (L) keeps the toggles
    const carry = st.carry || { captions: false, song: false, visual: false, voiceOn: false };
    st.carry = null;
    st.captions = carry.captions; st.song = carry.song; st.visual = carry.visual;
    st.voiceOn = carry.voiceOn && !!tl.files.voice; st.duck = 1;
    $("tO").disabled = !tl.files.voice;
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
    $("vizAway").textContent = tl.teams.away.abbr;
    $("vizHome").textContent = tl.teams.home.abbr;
    $("fillAway").style.background = tl.teams.away.colour;
    $("fillHome").style.background = tl.teams.home.colour;
    $("vizStyle").value = st.style;
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
    for (const m of [video, game, song, voice]) { m.pause(); m.removeAttribute("src"); m.load(); }
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
      if (d > DRIFT_MAX) { a.currentTime = t; st.sync.resyncs++; }
    }
  }

  /* S: crossfade game sound <-> song */
  function applyVolumes() {
    game.volume = st.vol.game * st.duck; song.volume = st.vol.song * st.duck;
    voice.volume = st.voiceOn ? 1 : 0;
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
    if (k === "O" && st.tl.files.voice) { st.voiceOn = !st.voiceOn; applyVolumes(); }
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
    $("tO").classList.toggle("on", st.voiceOn);
    $("credit").classList.toggle("on", st.song);
    $("vizA").classList.toggle("on", st.visual && st.style === "A");
    $("vizB").classList.toggle("on", st.visual && st.style === "B");
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

  /* visual A: tug-of-war strip; B: tension sparkline */
  function sample(t) {
    const s = st.tl.samples;
    return s[Math.max(0, Math.min(s.length - 1, Math.floor(t * st.tl.rate)))];
  }
  let lastViz = -1;
  function updateViz(t) {
    if (!st.visual || !st.tl) return;
    const k = Math.floor(t * 4);                 // 4 updates a second: nothing moves faster than the game
    if (k === lastViz) return;
    lastViz = k;
    const [, wp, ten] = sample(t);
    if (st.style === "A") {
      const lead = Math.abs(wp - 0.5) * 2;       // 0 = even, 1 = decided
      const home = wp >= 0.5;
      const w = (lead * 50).toFixed(2) + "%";
      const fa = $("fillAway"), fh = $("fillHome");
      fa.style.left = home ? "50%" : `${50 - lead * 50}%`; fa.style.width = home ? "0%" : w;
      fh.style.right = home ? `${50 - lead * 50}%` : "50%"; fh.style.width = home ? w : "0%";
      fh.style.left = "auto";
      $("knot").style.left = `${50 + (wp - 0.5) * 100}%`;
      const glow = (2 + ten * 14).toFixed(1);
      const col = home ? st.tl.teams.home.colour : st.tl.teams.away.colour;
      for (const el of [fa, fh]) { el.style.opacity = (0.5 + 0.5 * ten).toFixed(2); el.style.boxShadow = `0 0 ${glow}px ${col}`; }
    } else {
      drawSpark(t);
    }
  }

  function drawSpark(t) {
    const cv = $("vizB"), ctx = cv.getContext("2d");
    const W = cv.width, H = cv.height, pad = 8;
    ctx.clearRect(0, 0, W, H);
    if (!st.tl) return;
    const s = st.tl.samples, n = s.length;
    ctx.fillStyle = "rgba(8,10,13,.62)";
    ctx.beginPath(); ctx.roundRect(0, 0, W, H, 10); ctx.fill();
    const x = (i) => pad + (i / (n - 1)) * (W - 2 * pad);
    const y = (v) => H - pad - v * (H - 2 * pad - 14);
    ctx.beginPath();
    for (let i = 0; i < n; i += 2) i ? ctx.lineTo(x(i), y(s[i][2])) : ctx.moveTo(x(i), y(s[i][2]));
    ctx.strokeStyle = "rgba(255,255,255,.75)"; ctx.lineWidth = 2; ctx.stroke();
    ctx.lineTo(x(n - 1), H - pad); ctx.lineTo(x(0), H - pad); ctx.closePath();
    ctx.fillStyle = "rgba(255,255,255,.08)"; ctx.fill();
    const i = Math.min(n - 1, Math.floor(t * st.tl.rate));
    ctx.fillStyle = "rgba(255,255,255,.5)"; ctx.fillRect(x(i), pad, 1, H - 2 * pad);
    ctx.beginPath(); ctx.arc(x(i), y(s[i][2]), 5, 0, Math.PI * 2);
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--accent"); ctx.fill();
    ctx.font = "600 16px Bahnschrift, Arial Narrow, sans-serif"; ctx.fillStyle = "rgba(255,255,255,.6)";
    ctx.fillText("TENSION", pad + 2, pad + 12);
  }

  /* frame loop */
  function frame() {
    if (st.tl) {
      const t = video.currentTime || 0, d = st.tl.duration;
      syncAudio();
      updateDuck(t);
      updateCaption(t);
      updateViz(t);
      $("done").style.width = `${Math.min(100, (t / d) * 100)}%`;
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
  $("scrub").onclick = (e) => {
    if (!st.tl) return;
    const r = e.currentTarget.getBoundingClientRect();
    seek(((e.clientX - r.left) / r.width) * st.tl.duration);
  };
  for (const id of ["tC", "tS", "tV", "tO", "tL"]) $(id).onclick = () => toggle($(id).dataset.key);
  $("vizStyle").onchange = (e) => { st.style = e.target.value; writePref("vizStyle", st.style); lastViz = -1; paintToggles(); drawSpark(video.currentTime || 0); };
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toUpperCase();
    if ("CSVOL".includes(k) && k.length === 1) { toggle(k); e.preventDefault(); }
    if (e.key === " ") { playPause(); e.preventDefault(); }
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
