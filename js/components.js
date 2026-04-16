/* DOM components. Keep updates minimal; use textContent, not innerHTML. */

const createEl = (tag, attrs = {}, children = []) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") el.className = v;
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, "");
    else if (v != null) el.setAttribute(k, String(v));
  }
  for (const c of children) {
    if (c == null) continue;
    if (typeof c === "string") el.appendChild(document.createTextNode(c));
    else el.appendChild(c);
  }
  return el;
};

const clearNode = (node) => {
  while (node.firstChild) node.removeChild(node.firstChild);
};

const showBanner = ({ host, message, onRetry } = {}) => {
  clearNode(host);
  const msg = createEl("div", { class: "banner__msg" }, [message || "Unknown error"]);
  const actions = createEl("div", { class: "banner__actions" });
  if (typeof onRetry === "function") {
    const btn = createEl("button", { class: "btn btn--secondary", type: "button" }, ["Retry"]);
    btn.addEventListener("click", onRetry, { once: true });
    actions.appendChild(btn);
  }
  const box = createEl("div", { class: "banner", role: "region", "aria-label": "Error banner" }, [msg, actions]);
  host.appendChild(box);
};

const clearBanner = (host) => clearNode(host);

const renderTickerPills = ({ host, tickers, active, onSelect, abortSignal } = {}) => {
  clearNode(host);
  const buttons = [];
  const setActive = (t) => {
    for (const b of buttons) {
      const isOn = b.dataset.ticker === t;
      b.setAttribute("aria-selected", isOn ? "true" : "false");
      b.setAttribute("tabindex", isOn ? "0" : "-1");
    }
  };

  for (const t of tickers) {
    const b = createEl(
      "button",
      {
        class: "pill",
        role: "tab",
        type: "button",
        "aria-selected": t === active ? "true" : "false",
        "aria-label": `Select ticker ${t}`,
        tabindex: t === active ? "0" : "-1",
        dataset: { ticker: t },
      },
      [t]
    );
    b.addEventListener(
      "click",
      debounce(() => onSelect?.(t), 100),
      { signal: abortSignal }
    );
    buttons.push(b);
    host.appendChild(b);
  }

  setActive(active);

  host.addEventListener(
    "keydown",
    (e) => {
      const idx = buttons.findIndex((x) => x.getAttribute("aria-selected") === "true");
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        const next = e.key === "ArrowRight" ? idx + 1 : idx - 1;
        const j = (next + buttons.length) % buttons.length;
        buttons[j].focus();
        buttons[j].click();
      }
    },
    { signal: abortSignal }
  );

  return Object.freeze({ setActive });
};

const metricCard = ({ keyLabel, valueText, subText } = {}) => {
  const k = createEl("div", { class: "metric-card__k" }, [keyLabel || ""]);
  const v = createEl("div", { class: "metric-card__v" }, [valueText || "—"]);
  const s = createEl("div", { class: "metric-card__sub" }, [subText || ""]);
  return createEl("div", { dataset: { component: "metric-card" } }, [k, v, s]);
};

const renderMetricCards = ({ host, cards } = {}) => {
  clearNode(host);
  for (const c of cards || []) host.appendChild(metricCard(c));
};

const renderModelTogglePanel = ({ host, modelGroups, activeSet, onToggle, abortSignal } = {}) => {
  clearNode(host);
  for (const g of modelGroups) {
    const title = createEl("div", { class: "model-group__title" }, [
      createEl("div", { class: "h3" }, [g.title]),
      createEl("div", { class: "muted" }, [g.note || ""]),
    ]);
    const chips = createEl("div", { class: "model-group__chips" });
    for (const m of g.models) {
      const pressed = activeSet.has(m);
      const c =
        (typeof modelColor === "function" && modelColor(m)) ||
        ((typeof MODEL_COLORS === "object" && MODEL_COLORS?.[m]) ? MODEL_COLORS[m] : autoColorForKey(m));
      const sw = createEl("span", { class: "chip__swatch", style: `background:${c ?? "transparent"}` });
      const label = createEl("span", {}, [m]);
      const chip = createEl(
        "button",
        {
          class: "chip",
          type: "button",
          role: "button",
          "aria-pressed": pressed ? "true" : "false",
          "aria-label": `Toggle model ${m}`,
          dataset: { model: m },
        },
        [sw, label]
      );
      chip.addEventListener(
        "click",
        debounce(() => onToggle?.(m), 50),
        { signal: abortSignal }
      );
      chips.appendChild(chip);
    }
    const box = createEl("div", { class: "model-group" }, [title, chips]);
    host.appendChild(box);
  }
};

const renderDirectionalHeatmap = ({ host, tickers, models, metricsLookup } = {}) => {
  clearNode(host);
  const table = createEl("table", { "aria-label": "Directional accuracy heatmap" });
  const thead = createEl("thead");
  const trh = createEl("tr");
  trh.appendChild(createEl("th", {}, ["Ticker"]));
  for (const m of models) trh.appendChild(createEl("th", {}, [m]));
  thead.appendChild(trh);
  table.appendChild(thead);

  const tbody = createEl("tbody");
  for (const t of tickers) {
    const tr = createEl("tr");
    tr.appendChild(createEl("td", {}, [t]));
    for (const m of models) {
      const v = metricsLookup?.[t]?.[m]?.dir_acc;
      const hue = hueForAcc(v);
      const td = createEl("td", {}, [v != null && Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : "—"]);
      if (hue != null) {
        td.style.background = `hsl(${hue} 70% 50% / 0.28)`;
      }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  host.appendChild(table);
};

