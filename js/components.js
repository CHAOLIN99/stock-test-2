/* DOM components. Prefer diff-updates over full rebuilds for smooth UX. */

const createEl = (tag, attrs = {}, children = []) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") el.className = v;
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, "");
    else if (v === false || v == null) continue;
    else el.setAttribute(k, String(v));
  }
  for (const c of children) {
    if (c == null) continue;
    el.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return el;
};

const clearNode = (node) => {
  while (node.firstChild) node.removeChild(node.firstChild);
};

const setText = (node, text) => {
  if (node.textContent !== text) node.textContent = text;
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
  host.appendChild(createEl("div", { class: "banner", role: "alert" }, [msg, actions]));
};

const clearBanner = (host) => clearNode(host);

const renderTickerPills = ({ host, tickers, active, onSelect, abortSignal } = {}) => {
  clearNode(host);
  const byTicker = new Map();

  const setActive = (t) => {
    for (const [key, b] of byTicker) {
      const isOn = key === t;
      if (b.getAttribute("aria-selected") !== (isOn ? "true" : "false")) {
        b.setAttribute("aria-selected", isOn ? "true" : "false");
      }
      b.tabIndex = isOn ? 0 : -1;
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
    b.addEventListener("click", () => onSelect?.(t), { signal: abortSignal });
    byTicker.set(t, b);
    host.appendChild(b);
  }

  host.addEventListener(
    "keydown",
    (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft" && e.key !== "Home" && e.key !== "End") return;
      e.preventDefault();
      const keys = Array.from(byTicker.keys());
      const current = keys.findIndex((k) => byTicker.get(k).getAttribute("aria-selected") === "true");
      let j = current;
      if (e.key === "ArrowRight") j = (current + 1) % keys.length;
      else if (e.key === "ArrowLeft") j = (current - 1 + keys.length) % keys.length;
      else if (e.key === "Home") j = 0;
      else if (e.key === "End") j = keys.length - 1;
      const target = byTicker.get(keys[j]);
      target?.focus();
      onSelect?.(keys[j]);
    },
    { signal: abortSignal }
  );

  setActive(active);
  return Object.freeze({ setActive });
};

/** Diff-update metric cards so values animate and the DOM stays stable. */
const renderMetricCards = ({ host, cards } = {}) => {
  const list = cards || [];
  const existing = host.children;

  while (existing.length > list.length) host.removeChild(existing[existing.length - 1]);

  for (let i = 0; i < list.length; i++) {
    const c = list[i];
    let card = existing[i];
    if (!card) {
      card = createEl("div", { class: "metric-card", dataset: { component: "metric-card" } }, [
        createEl("div", { class: "metric-card__k" }),
        createEl("div", { class: "metric-card__v" }),
        createEl("div", { class: "metric-card__sub" }),
      ]);
      host.appendChild(card);
    }
    const [k, v, s] = card.children;
    setText(k, c.keyLabel || "");
    setText(v, c.valueText || "—");
    setText(s, c.subText || "");
  }
};

/** Rebuild chip structure only when models change; toggle aria-pressed otherwise. */
const renderModelTogglePanel = ({ host, modelGroups, activeSet, onToggle, abortSignal } = {}) => {
  const signature = modelGroups.map((g) => `${g.title}:${g.models.join(",")}`).join("|");

  if (host.dataset.signature !== signature) {
    clearNode(host);
    host.dataset.signature = signature;

    for (const g of modelGroups) {
      const title = createEl("div", { class: "model-group__title" }, [
        createEl("div", { class: "h3" }, [g.title]),
        createEl("div", { class: "muted" }, [g.note || ""]),
      ]);
      const chips = createEl("div", { class: "model-group__chips" });
      for (const m of g.models) {
        const color =
          (typeof modelColor === "function" && modelColor(m)) ||
          (typeof MODEL_COLORS === "object" && MODEL_COLORS?.[m]) ||
          autoColorForKey(m);
        const chip = createEl(
          "button",
          {
            class: "chip",
            type: "button",
            "aria-pressed": activeSet.has(m) ? "true" : "false",
            "aria-label": `Toggle model ${m}`,
            dataset: { model: m },
          },
          [
            createEl("span", { class: "chip__swatch", style: `background:${color ?? "transparent"}` }),
            createEl("span", { class: "chip__label" }, [m]),
          ]
        );
        chip.addEventListener("click", () => onToggle?.(m), { signal: abortSignal });
        chips.appendChild(chip);
      }
      host.appendChild(createEl("div", { class: "model-group" }, [title, chips]));
    }
  }

  for (const chip of host.querySelectorAll(".chip")) {
    const m = chip.dataset.model;
    const pressed = activeSet.has(m) ? "true" : "false";
    if (chip.getAttribute("aria-pressed") !== pressed) {
      chip.setAttribute("aria-pressed", pressed);
    }
  }
};

const renderDirectionalHeatmap = ({ host, tickers, models, metricsLookup } = {}) => {
  const signature = `${tickers.join(",")}|${models.join(",")}`;

  if (host.dataset.signature !== signature) {
    clearNode(host);
    host.dataset.signature = signature;

    const table = createEl("table", { "aria-label": "Directional accuracy heatmap" });
    const thead = createEl("thead");
    const trh = createEl("tr");
    trh.appendChild(createEl("th", { scope: "col" }, ["Ticker"]));
    for (const m of models) trh.appendChild(createEl("th", { scope: "col" }, [m]));
    thead.appendChild(trh);
    table.appendChild(thead);

    const tbody = createEl("tbody");
    for (const t of tickers) {
      const tr = createEl("tr", { dataset: { ticker: t } });
      tr.appendChild(createEl("td", { class: "heatmap__row-label" }, [t]));
      for (const m of models) {
        tr.appendChild(createEl("td", { dataset: { model: m } }, ["—"]));
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    host.appendChild(table);
  }

  const rows = host.querySelectorAll("tbody tr");
  for (const tr of rows) {
    const t = tr.dataset.ticker;
    const cells = tr.children;
    for (let i = 1; i < cells.length; i++) {
      const td = cells[i];
      const m = td.dataset.model;
      const v = metricsLookup?.[t]?.[m]?.dir_acc;
      const hue = hueForAcc(v);
      setText(td, v != null && Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : "—");
      td.style.background = hue != null ? `hsl(${hue} 70% 50% / 0.28)` : "";
    }
  }
};
