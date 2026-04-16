/* Pure utilities (no side effects). */

const safeVal = (v, fb = "—") => (v != null && Number.isFinite(v)) ? v : fb;

const assert = (cond, message) => {
  if (!cond) throw new Error(message || "Assertion failed");
};

const debounce = (fn, ms) => {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
};

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

const formatMoney = (v) => {
  if (v == null || !Number.isFinite(v)) return "—";
  return `$${Number(v).toFixed(2)}`;
};

const formatPct = (v, digits = 1) => {
  if (v == null || !Number.isFinite(v)) return "—";
  const s = (v * 100).toFixed(digits);
  return `${v >= 0 ? "+" : ""}${s}%`;
};

const formatNum = (v, digits = 3) => {
  if (v == null || !Number.isFinite(v)) return "—";
  return Number(v).toFixed(digits);
};

const parseISODate = (s) => {
  if (!s) return null;
  const t = Date.parse(s);
  return Number.isFinite(t) ? t : null;
};

const deepMerge = (base, patch) => {
  if (patch === undefined) return base;
  if (patch === null) return null;
  if (typeof patch !== "object") return patch;
  if (base == null || typeof base !== "object") return patch;
  if (base instanceof Set || patch instanceof Set) return patch;
  if (Array.isArray(base) || Array.isArray(patch)) return patch;
  const out = { ...base };
  for (const [k, v] of Object.entries(patch)) {
    out[k] = deepMerge(base[k], v);
  }
  return out;
};

const parseRoute = (hash) => {
  const h = (hash || "").replace(/^#/, "");
  const parts = h.split("/").filter(Boolean);
  if (parts.length === 0) return { name: "overview", params: {} };
  if (parts[0] === "overview") return { name: "overview", params: {} };
  if (parts[0] === "compare") return { name: "compare", params: {} };
  if (parts[0] === "methodology") return { name: "methodology", params: {} };
  if (parts[0] === "ticker" && parts[1]) return { name: "ticker", params: { symbol: parts[1].toUpperCase() } };
  return { name: "overview", params: {} };
};

const uniq = (arr) => Array.from(new Set(arr));

const hueForAcc = (acc) => {
  if (acc == null || !Number.isFinite(acc)) return null;
  const pct = clamp((acc * 100 - 40) / 30, 0, 1);
  return pct * 120;
};

const pickBestModel = (metricsByModel) => {
  let best = null;
  for (const [model, m] of Object.entries(metricsByModel || {})) {
    const v = m?.rmse_mean_test;
    if (v == null || !Number.isFinite(v)) continue;
    if (!best || v < best.val) best = { model, val: v };
  }
  return best?.model ?? null;
};

const getQueryParam = (key) => {
  try {
    const u = new URL(window.location.href);
    return u.searchParams.get(key);
  } catch {
    return null;
  }
};

const stableHash32 = (s) => {
  // Deterministic, fast (FNV-1a 32-bit)
  const str = String(s ?? "");
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
};

const autoColorForKey = (key, { sat = 70, light = 48 } = {}) => {
  const h = stableHash32(key) % 360;
  return `hsl(${h} ${sat}% ${light}%)`;
};

