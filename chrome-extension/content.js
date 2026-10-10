(() => {
  if (window.__spInjected) return;
  window.__spInjected = true;

  const PANEL_CSS = `
    :host { all: initial; }
    .sp-wrap { position: fixed; right: 16px; bottom: 16px; z-index: 2147483647; width: 320px;
      background: #fff; border: 1px solid #d9dee3; border-radius: 12px; box-shadow: 0 8px 28px rgba(0,0,0,.18); }
    .sp-bar { display: flex; align-items: center; justify-content: space-between; padding: 8px 12px;
      font: 600 13px -apple-system, BlinkMacSystemFont, sans-serif; color: #1f2933; cursor: pointer;
      border-bottom: 1px solid #eef0f2; user-select: none; }
    .sp-body { padding: 10px 12px; max-height: 70vh; overflow: auto; }
    .sp-collapsed .sp-body { display: none; }
    .sp-collapsed .sp-bar { border-bottom: none; }
    .sp-detected { font-weight: 400; color: #616e7c; font-size: 11px; }
  `;

  const host = document.createElement("div");
  const shadow = host.attachShadow({ mode: "open" });
  const style = document.createElement("style");
  style.textContent = PANEL_CSS + SPRender.CSS;

  const wrap = document.createElement("div");
  wrap.className = "sp-wrap";
  const bar = document.createElement("div");
  bar.className = "sp-bar";
  const title = document.createElement("span");
  title.textContent = "\u{1F4C8} Stock Predictor";
  const detected = document.createElement("span");
  detected.className = "sp-detected";
  const toggle = document.createElement("span");
  bar.append(title, detected, toggle);
  const body = document.createElement("div");
  body.className = "sp-body";
  wrap.append(bar, body);
  shadow.append(style, wrap);
  document.documentElement.append(host);

  let app = null;
  let lastDetected = null;

  function setCollapsed(collapsed) {
    collapsed = Boolean(collapsed);
    wrap.classList.toggle("sp-collapsed", collapsed);
    toggle.textContent = collapsed ? "\u25B2" : "\u25BC";
    chrome.storage.local.set({ panelCollapsed: collapsed });
    if (!collapsed && !app) app = SPRender.mount(body, { initialTicker: lastDetected });
  }
  bar.addEventListener("click", () => setCollapsed(!wrap.classList.contains("sp-collapsed")));

  // Only the tickers the models were trained on are worth detecting.
  function detectTicker(tickers) {
    const re = new RegExp(`\\b(${tickers.join("|")})\\b`);
    const url = new URL(location.href);
    for (const key of ["symbol", "s", "ticker"]) {
      const v = (url.searchParams.get(key) || "").toUpperCase();
      if (tickers.includes(v)) return v;
    }
    const pathHit = location.pathname.toUpperCase().match(re);
    if (pathHit) return pathHit[1];

    for (const input of document.querySelectorAll("input")) {
      const v = (input.value || "").trim().toUpperCase();
      if (tickers.includes(v)) return v;
    }
    const selectors = ["h1", "h2", "h3", "[class*='symbol' i]", "[class*='ticker' i]", "[data-symbol]"];
    for (const node of document.querySelectorAll(selectors.join(","))) {
      if (host.contains(node)) continue;
      const text = `${node.getAttribute("data-symbol") || ""} ${node.textContent || ""}`.toUpperCase();
      const hit = text.match(re);
      if (hit) return hit[1];
    }
    return null;
  }

  let timer = null;
  let knownTickers = null;
  async function scan() {
    if (!knownTickers) {
      knownTickers = await new Promise((resolve) =>
        chrome.runtime.sendMessage({ type: "tickers" }, (res) => resolve(res && res.ok ? res.data : null)),
      );
    }
    const tickers = knownTickers || [];
    if (!tickers.length) return;
    const t = detectTicker(tickers);
    if (!t || t === lastDetected) return;
    lastDetected = t;
    detected.textContent = `on page: ${t}`;
    if (app) app.select(t);
  }
  const scheduleScan = () => {
    clearTimeout(timer);
    timer = setTimeout(scan, 600);
  };

  new MutationObserver(scheduleScan).observe(document.body, { childList: true, subtree: true, characterData: true });
  document.addEventListener("input", scheduleScan, true);
  window.addEventListener("popstate", scheduleScan);

  chrome.storage.local.get({ panelCollapsed: false }).then(({ panelCollapsed }) => {
    scan().finally(() => setCollapsed(panelCollapsed));
  });
})();
