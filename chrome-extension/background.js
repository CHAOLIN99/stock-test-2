const DEFAULT_SERVER = "http://127.0.0.1:8080";
const CACHE_MS = 5 * 60 * 1000;
const FALLBACK_TICKERS = ["AMD", "META", "MSFT", "NVDA", "QCOM"];

const cache = new Map();

async function serverUrl() {
  const { serverUrl } = await chrome.storage.sync.get({ serverUrl: DEFAULT_SERVER });
  return serverUrl.replace(/\/+$/, "");
}

async function getJson(path) {
  const base = await serverUrl();
  let res;
  try {
    res = await fetch(`${base}${path}`, { cache: "no-store" });
  } catch {
    throw new Error(`Can't reach the prediction server at ${base}. Start it with: python server.py`);
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `Server returned HTTP ${res.status}`);
  return body;
}

async function predict(ticker, force) {
  const key = ticker.toUpperCase();
  const hit = cache.get(key);
  if (!force && hit && Date.now() - hit.at < CACHE_MS) return hit.data;
  const data = await getJson(`/api/predict/${encodeURIComponent(key)}`);
  cache.set(key, { at: Date.now(), data });
  return data;
}

async function tickers() {
  try {
    return (await getJson("/api/tickers")).tickers;
  } catch {
    return FALLBACK_TICKERS;
  }
}

chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  const run = {
    predict: () => predict(msg.ticker, msg.force),
    tickers: () => tickers(),
    health: () => getJson("/api/health"),
  }[msg.type];
  if (!run) return false;
  run()
    .then((data) => reply({ ok: true, data }))
    .catch((err) => reply({ ok: false, error: err.message }));
  return true;
});
