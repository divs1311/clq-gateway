// Economics simulator - JavaScript port of clq/economics.py (same algorithm, same PRNG).
// Parity with Python is enforced by tests/test_parity.py.
(function (root) {
  const TIERS = ["small", "mid", "frontier"];
  const DIFFS = ["easy", "medium", "hard"];
  const POLICIES = ["single:small", "single:mid", "single:frontier", "static_router", "cheap_first", "cascade"];
  const Z95 = 1.6448536269514722;

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = ((t + Math.imul(t ^ (t >>> 7), 61 | t)) >>> 0) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function normal(rnd) {
    const u1 = 1.0 - rnd();
    const u2 = rnd();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }
  function lognormalSample(rnd, median, p95) {
    const sigma = (Math.log(p95) - Math.log(median)) / Z95;
    return Math.exp(Math.log(median) + sigma * normal(rnd));
  }

  function enumeratePaths(p, policy) {
    const paths = [];
    const useRouter = policy === "static_router" || policy === "cascade";
    const escalate = policy === "cascade" || policy === "cheap_first";
    const cacheOn = !!p.cache_on;
    for (const d of DIFFS) {
      const md = p.mix[d];
      if (!(md > 0)) continue;
      const h = (cacheOn && d === "easy") ? p.cache_hit_easy : 0.0;
      if (h > 0) {
        paths.push({ prob: md * h * p.cache_precision, tiers: [], cache_hit: true, router: false, outcome: "pass" });
        paths.push({ prob: md * h * (1 - p.cache_precision), tiers: [], cache_hit: true, router: false, outcome: "silent" });
      }
      const miss = md * (1 - h);
      let start;
      if (policy.startsWith("single:")) {
        const only = policy.split(":")[1];
        start = { small: only === "small" ? 1 : 0, mid: only === "mid" ? 1 : 0, frontier: only === "frontier" ? 1 : 0 };
      } else if (policy === "cheap_first") {
        start = { small: 1, mid: 0, frontier: 0 };
      } else {
        start = { small: p.route[d][0], mid: p.route[d][1], frontier: p.route[d][2] };
      }
      const cutSet = new Set([0.0, 1.0]);
      for (const t of TIERS) cutSet.add(Math.min(Math.max(p.acc[t][d], 0.0), 1.0));
      const cuts = Array.from(cutSet).sort((x, y) => x - y);
      for (let si = 0; si < TIERS.length; si++) {
        const ps = start[TIERS[si]];
        if (!(ps > 0)) continue;
        for (let k = 0; k < cuts.length - 1; k++) {
          const lo = cuts[k], hi = cuts[k + 1];
          const w = hi - lo;
          if (!(w > 0)) continue;
          const midU = (lo + hi) / 2;
          let mass = miss * ps * w;
          const visited = [];
          for (let ti = si; ti < TIERS.length; ti++) {
            const t = TIERS[ti];
            visited.push(t);
            const correct = midU < p.acc[t][d];
            const f = correct ? p.false_flag : p.recall;
            const deliver = mass * (1 - f);
            if (deliver > 0) paths.push({ prob: deliver, tiers: visited.slice(), cache_hit: false, router: useRouter, outcome: correct ? "pass" : "silent" });
            mass *= f;
            if (!(mass > 0)) break;
            if (!(escalate && ti < TIERS.length - 1)) {
              paths.push({ prob: mass, tiers: visited.slice(), cache_hit: false, router: useRouter, outcome: "handoff" });
              break;
            }
          }
        }
      }
    }
    return paths;
  }

  function latencyPercentiles(p, paths) {
    const rnd = mulberry32(p.seed);
    const n = Math.floor(p.mc_samples);
    let total = 0;
    for (const x of paths) total += x.prob;
    const cum = [];
    let acc = 0;
    for (const x of paths) { acc += x.prob / total; cum.push(acc); }
    const samples = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const u = rnd();
      let k = 0;
      while (k < cum.length - 1 && u >= cum[k]) k++;
      const x = paths[k];
      let ms = 0.0;
      if (p.cache_on) ms += lognormalSample(rnd, p.cache_latency.median, p.cache_latency.p95);
      if (x.router) ms += p.router_ms;
      for (const t of x.tiers) ms += lognormalSample(rnd, p.latency[t].median, p.latency[t].p95) + p.validator_ms;
      samples[i] = ms;
    }
    samples.sort();
    const pct = (q) => samples[Math.max(1, Math.ceil(q / 100 * n)) - 1];
    return { p50: pct(50), p90: pct(90), p95: pct(95), p99: pct(99), samples };
  }

  function evaluatePolicy(p, policy, opts) {
    const paths = enumeratePaths(p, policy);
    const useRouter = policy === "static_router" || policy === "cascade";
    const rate = (pred) => paths.reduce((s, x) => s + (pred(x) ? x.prob : 0), 0);
    const out = { policy };
    out.prob_mass = paths.reduce((s, x) => s + x.prob, 0);
    out.pass_rate = rate((x) => x.outcome === "pass");
    out.silent_error_rate = rate((x) => x.outcome === "silent");
    out.handoff_rate = rate((x) => x.outcome === "handoff");
    out.escalation_rate = rate((x) => x.tiers.length > 1);
    out.cache_hit_rate = rate((x) => x.cache_hit);
    out.final_tier_share = {};
    for (const t of TIERS) out.final_tier_share[t] = rate((x) => x.tiers.length > 0 && x.tiers[x.tiers.length - 1] === t);
    const model = paths.reduce((s, x) => s + x.prob * x.tiers.reduce((c, t) => c + p.cost_per_call[t], 0), 0);
    const overhead = (p.cache_on ? p.cache_cost : 0.0) + (useRouter ? p.router_cost * (1 - out.cache_hit_rate) : 0.0);
    out.model_cost_per_1k = 1000 * (model + overhead);
    out.human_cost_per_1k = 1000 * out.handoff_rate * p.human_cost;
    out.silent_cost_per_1k = 1000 * out.silent_error_rate * p.silent_cost;
    out.fully_loaded_per_1k = out.model_cost_per_1k + out.human_cost_per_1k + out.silent_cost_per_1k;
    out.monthly_model_cost = out.model_cost_per_1k * p.volume / 1000;
    out.monthly_fully_loaded = out.fully_loaded_per_1k * p.volume / 1000;
    const lat = latencyPercentiles(p, paths);
    out.latency_ms = { p50: lat.p50, p90: lat.p90, p95: lat.p95, p99: lat.p99 };
    if (opts && opts.keepSamples) out.samples = lat.samples;
    return out;
  }

  function compare(p) {
    const r = {};
    for (const pol of POLICIES) r[pol] = evaluatePolicy(p, pol);
    return r;
  }

  const api = { TIERS, DIFFS, POLICIES, mulberry32, enumeratePaths, evaluatePolicy, compare };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.CLQSim = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
