// Moment timestamps exactly as the sister page computes them: core.js dtwPath + index.html buildMaps/timeOfGameFrac.
// usage: node scripts/page_moments.js <path to sister site/core.js>  < {"song": [...raw demo curve], "game": [...index curve], "t0", "t1", "n_steps", "steps": [...]}
const core = require(process.argv[2]);
const S = core.SongMatch || core, N = S.N;
let buf = "";
process.stdin.on("data", d => (buf += d)).on("end", () => {
  const q = JSON.parse(buf);
  const st = { t0: q.t0, t1: q.t1 };
  const { path } = S.dtwPath(S.normalise(q.song), q.game);
  // --- verbatim from site/index.html ---
  const sj = new Float64Array(N), cj = new Float64Array(N), si = new Float64Array(N), ci = new Float64Array(N);
  for (const [i, j] of path) { sj[i] += j; cj[i]++; si[j] += i; ci[j]++; }
  st.iOfJ = Array.from(si, (v, k) => v / ci[k]);
  const lerp = (arr, x) => { x = Math.min(N - 1, Math.max(0, x)); const a = Math.floor(x), b = Math.min(N - 1, a + 1); return arr[a] + (arr[b] - arr[a]) * (x - a); };
  const timeOfSongIdx = i => st.t0 + i / (N - 1) * (st.t1 - st.t0);
  const timeOfGameFrac = f => timeOfSongIdx(lerp(st.iOfJ, f * (N - 1)));
  // ---
  process.stdout.write(JSON.stringify(q.steps.map(s => timeOfGameFrac(s / (q.n_steps - 1)))));
});
