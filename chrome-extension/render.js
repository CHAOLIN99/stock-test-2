// Shared by popup.js and content.js. Exposes globalThis.SPRender.
(() => {
  const CSS = `
    .sp-card { font: 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #1f2933; }
    .sp-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
    .sp-pill { border: 1px solid #cbd2d9; background: #fff; border-radius: 999px; padding: 3px 10px; cursor: pointer; font: inherit; color: inherit; }
    .sp-pill[aria-pressed="true"] { background: #1f2933; color: #fff; border-color: #1f2933; }
    .sp-head { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
    .sp-ticker { font-size: 18px; font-weight: 700; }
    .sp-signal { font-weight: 700; padding: 2px 10px; border-radius: 6px; color: #fff; }
    .sp-BUY { background: #16a34a; } .sp-SELL { background: #dc2626; } .sp-HOLD { background: #6b7280; }
    .sp-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; margin: 10px 0; }
    .sp-label { color: #616e7c; font-size: 11px; text-transform: uppercase; letter-spacing: .03em; }
    .sp-value { font-weight: 600; font-size: 15px; }
    .sp-up { color: #16a34a; } .sp-down { color: #dc2626; }
    .sp-models { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 6px; }
    .sp-models td, .sp-models th { padding: 3px 4px; text-align: right; border-bottom: 1px solid #eef0f2; }
    .sp-models td:first-child, .sp-models th:first-child { text-align: left; }
    .sp-models th { color: #616e7c; font-weight: 500; }
    .sp-weak { color: #9aa5b1; }
    details summary { cursor: pointer; color: #3e4c59; margin-top: 4px; }
    .sp-note { color: #616e7c; font-size: 11px; margin-top: 8px; }
    .sp-error { color: #b91c1c; background: #fef2f2; border-radius: 6px; padding: 8px; }
    .sp-muted { color: #616e7c; }
    .sp-actions { display: flex; justify-content: flex-end; margin-top: 6px; }
    .sp-link { background: none; border: none; color: #2563eb; cursor: pointer; font: inherit; padding: 0; }
  `;

  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };
  const pct = (x, digits = 2) => (x == null ? "—" : `${x >= 0 ? "+" : ""}${(x * 100).toFixed(digits)}%`);
  const usd = (x) => (x == null ? "—" : `$${x.toFixed(2)}`);
  const dirClass = (x) => (x > 0 ? "sp-up" : x < 0 ? "sp-down" : "");

  function stat(label, value, cls) {
    const box = el("div");
    box.append(el("div", "sp-label", label), el("div", `sp-value ${cls || ""}`, value));
    return box;
  }

  function pills(tickers, selected, onPick) {
    const wrap = el("div", "sp-pills");
    for (const t of tickers) {
      const b = el("button", "sp-pill", t);
      b.type = "button";
      b.setAttribute("aria-pressed", String(t === selected));
      b.addEventListener("click", () => onPick(t));
      wrap.append(b);
    }
    return wrap;
  }

  function prediction(d, onRefresh) {
    const card = el("div");
    const head = el("div", "sp-head");
    head.append(el("span", "sp-ticker", d.ticker), el("span", `sp-signal sp-${d.signal}`, d.signal));
    card.append(head, el("div", "sp-muted", `Forecast for the session after ${d.as_of}`));

    const grid = el("div", "sp-grid");
    const bt = d.backtest || {};
    const acc = bt.stacking && bt.stacking.dir_acc;
    grid.append(
      stat("Last close", usd(d.last_close)),
      stat("Predicted close", usd(d.predicted_close), dirClass(d.predicted_return)),
      stat("Expected move", pct(d.predicted_return), dirClass(d.predicted_return)),
      stat("Models bullish", `${Math.round(d.models_bullish * 8)}/8`),
      stat("Backtest hit-rate", acc == null ? "—" : `${(acc * 100).toFixed(1)}%`),
      stat(bt.regime ? `Regime (${bt.regime.as_of})` : "Regime", bt.regime ? bt.regime.label : "—"),
    );
    card.append(grid);

    const details = el("details");
    details.append(el("summary", null, "Per-model forecasts"));
    const table = el("table", "sp-models");
    const hr = el("tr");
    ["Model", "Move", "Backtest hit-rate"].forEach((h) => hr.append(el("th", null, h)));
    table.append(hr);
    const rows = Object.entries(d.models).sort((a, b) => (a[1].backtest_rmse ?? 9) - (b[1].backtest_rmse ?? 9));
    const bestRmse = Math.min(...rows.map(([, m]) => m.backtest_rmse ?? Infinity));
    for (const [name, m] of rows) {
      const tr = el("tr", m.backtest_rmse > bestRmse * 2 ? "sp-weak" : "");
      tr.append(
        el("td", null, name),
        el("td", dirClass(m.predicted_return), pct(m.predicted_return)),
        el("td", null, m.backtest_dir_acc == null ? "—" : `${(m.backtest_dir_acc * 100).toFixed(1)}%`),
      );
      table.append(tr);
    }
    details.append(table, el("div", "sp-note", "Greyed-out models had more than 2× the best model's backtest error."));
    card.append(details);

    if (bt.top_features && bt.top_features.length) {
      card.append(el("div", "sp-note", `Top drivers (LightGBM): ${bt.top_features.join(", ")}`));
    }
    card.append(
      el(
        "div",
        "sp-note",
        "Signal: BUY/SELL only when the ensemble expects a move over 0.25% and most models agree. " + d.disclaimer,
      ),
    );
    if (onRefresh) {
      const actions = el("div", "sp-actions");
      const btn = el("button", "sp-link", "Refresh");
      btn.type = "button";
      btn.addEventListener("click", onRefresh);
      actions.append(btn);
      card.append(actions);
    }
    return card;
  }

  function message(text, isError) {
    return el("div", isError ? "sp-error" : "sp-muted", text);
  }

  function request(msg) {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage(msg, (res) => {
        if (chrome.runtime.lastError) resolve({ ok: false, error: chrome.runtime.lastError.message });
        else resolve(res);
      });
    });
  }

  // Renders the full ticker picker + forecast UI into `root`.
  function mount(root, { initialTicker, onTickerChange } = {}) {
    root.classList.add("sp-card");
    const pillsSlot = el("div");
    const body = el("div");
    body.append(message("Connecting to the prediction server\u2026"));
    root.replaceChildren(pillsSlot, body);
    let tickers = [];
    let current = null;
    let seq = 0;

    async function show(ticker, force = false) {
      current = ticker;
      pillsSlot.replaceChildren(pills(tickers, ticker, (t) => show(t)));
      if (onTickerChange) onTickerChange(ticker);
      const mine = ++seq;
      body.replaceChildren(message(`Running models for ${ticker}…`));
      const res = await request({ type: "predict", ticker, force });
      if (mine !== seq) return;
      body.replaceChildren(
        res && res.ok ? prediction(res.data, () => show(ticker, true)) : message(res ? res.error : "No response", true),
      );
    }

    const ready = request({ type: "tickers" }).then((res) => {
      tickers = res && res.ok ? res.data : [];
      const pick = tickers.includes(initialTicker) ? initialTicker : tickers[0];
      if (pick) show(pick);
      else body.replaceChildren(message(res && res.error ? res.error : "No tickers available.", true));
    });

    return {
      select: (t) => ready.then(() => tickers.includes(t) && t !== current && show(t)),
      tickers: () => ready.then(() => tickers),
    };
  }

  globalThis.SPRender = { CSS, mount };
})();
