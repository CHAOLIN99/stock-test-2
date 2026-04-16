/* Chart rendering + lifecycle management. */

const MODEL_COLORS = Object.freeze({
  Naive: "#8a8a86",
  BuyHold: "#6b6967",
  SMA_Cross: "#888780",
  HoltWinters: "#BA7517",
  ARIMA: "#7c5cff",
  HMM_Regime: "#EF9F27",

  Ridge: "#185FA5",
  MARS: "#1D9E75",
  RandomForest: "#3a7bd5",
  SVR: "#a455d6",
  MLP: "#c04c4c",
  XGBoost: "#2aa198",
  LightGBM: "#6c71c4",

  RNN: "#d33682",
  LSTM_A: "#185FA5",
  LSTM_B: "#185FA5",
  BiLSTM: "#185FA5",

  VotingEnsemble: "#111827",
  StackingEnsemble: "#374151",
});

const MODEL_DASHES = Object.freeze({
  LSTM_A: [],
  LSTM_B: [6, 3],
  BiLSTM: [2, 2],
  VotingEnsemble: [10, 3, 3, 3],
  StackingEnsemble: [10, 3, 3, 3],
  Naive: [4, 3],
});

const TIER_COLORS = Object.freeze({
  baseline: "#8a8a86",
  classical: "#185FA5",
  ml: "#1D9E75",
  dl: "#d33682",
  ensemble: "#111827",
  hmm: "#EF9F27",
});

const REGIME_COLORS = Object.freeze({
  bull: "#639922",
  bear: "#E24B4A",
  volatile: "#EF9F27",
  sideways: "#888780",
});

const FEATURE_LABELS = Object.freeze({
  SMA_5: "SMA (5)",
  SMA_10: "SMA (10)",
  SMA_20: "SMA (20)",
  SMA_50: "SMA (50)",
  SMA_200: "SMA (200)",
  RSI_14: "RSI (14)",
  MACD: "MACD",
  VOL_20: "Volatility (20d)",
});

const chartInstances = new Map();
const chartCleanups = new Map(); // key -> cleanup fn

const chartKey = (id) => id;

const destroyChart = (key) => {
  const k = chartKey(key);
  const existing = chartInstances.get(k);
  if (existing) {
    existing.destroy();
    chartInstances.delete(k);
  }
  const cleanup = chartCleanups.get(k);
  if (cleanup) {
    cleanup();
    chartCleanups.delete(k);
  }
};

const destroyAllCharts = () => {
  for (const c of chartInstances.values()) c.destroy();
  chartInstances.clear();
  for (const fn of chartCleanups.values()) fn();
  chartCleanups.clear();
};

const getCssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const modelColor = (modelName) => {
  if (modelName === "VotingEnsemble") return getCssVar("--chart-color-voting") || MODEL_COLORS.VotingEnsemble;
  if (modelName === "StackingEnsemble") return getCssVar("--chart-color-stacking") || MODEL_COLORS.StackingEnsemble;
  return MODEL_COLORS[modelName] ?? null;
};

const tierColor = (tier) => {
  if (tier === "ensemble") return getCssVar("--chart-color-ensemble") || TIER_COLORS.ensemble;
  return TIER_COLORS[tier] ?? "#888780";
};

const baseChartOptions = (nPoints) => {
  const text = getCssVar("--color-text");
  const muted = getCssVar("--color-text-muted");
  const border = getCssVar("--color-border");
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: nPoints > 500 ? 0 : 280 },
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: {
        labels: { color: text, boxWidth: 10, boxHeight: 10, usePointStyle: true },
      },
      tooltip: {
        callbacks: {},
      },
      decimation: {
        enabled: true,
        algorithm: "lttb",
        samples: 500,
      },
      zoom: {
        pan: { enabled: true, mode: "x" },
        zoom: {
          wheel: { enabled: true },
          pinch: { enabled: true },
          mode: "x",
        },
        limits: { x: { min: "original", max: "original" } },
      },
    },
    scales: {
      x: {
        ticks: { color: muted, maxRotation: 0, autoSkip: true },
        grid: { color: border },
      },
      y: {
        ticks: { color: muted },
        grid: { color: border },
      },
    },
  };
};

const attachZoomReset = ({ chart, button, key }) => {
  if (!button) return;
  const updateBtn = () => {
    const zs = chart.getZoomLevel?.() ?? 1;
    button.hidden = !(zs && zs > 1.001);
  };
  const onClick = () => {
    chart.resetZoom?.();
    button.hidden = true;
  };
  button.addEventListener("click", onClick);
  const t = setInterval(updateBtn, 400);
  chartCleanups.set(chartKey(key), () => {
    clearInterval(t);
    button.removeEventListener("click", onClick);
  });
};

const buildRegimeAnnotations = ({ regime, regimeNames, labels }) => {
  if (!regime || !Array.isArray(regime) || !regimeNames || !Array.isArray(regimeNames)) return {};
  if (regime.length !== labels.length) return {};
  const ann = {};
  let start = 0;
  for (let i = 1; i <= regime.length; i++) {
    if (i === regime.length || regime[i] !== regime[start]) {
      const r = regime[start];
      const name = regimeNames[r] ?? `regime_${r}`;
      const c = REGIME_COLORS[name] ?? "#888780";
      ann[`reg_${start}`] = {
        type: "box",
        xMin: labels[start],
        xMax: labels[i - 1],
        backgroundColor: `${c}59`, // ~0.35 alpha
        borderWidth: 0,
      };
      start = i;
    }
  }
  return ann;
};

const renderPriceChart = ({ key = "price", canvas, tickerData, modelSet, regimeOverlay, resetButton } = {}) => {
  assert(canvas, "renderPriceChart: missing canvas");
  destroyChart(key);

  const labels = tickerData.dates;
  const n = labels.length;

  const actual = tickerData.actual.map((v) => (v != null && Number.isFinite(v) ? Number(v) : null));

  const datasets = [];
  datasets.push({
    label: "Actual",
    data: actual,
    borderColor: getCssVar("--color-text"),
    borderWidth: 3,
    pointRadius: 0,
    spanGaps: false,
  });

  const naive = tickerData.predictions?.Naive;
  if (Array.isArray(naive)) {
    datasets.push({
      label: "Naive",
      data: naive.map((v) => (v != null && Number.isFinite(v) ? Number(v) : null)),
      borderColor: modelColor("Naive") ?? "#888780",
      borderWidth: 1,
      borderDash: [6, 4],
      pointRadius: 0,
      spanGaps: false,
    });
  }

  const selected = Array.from(modelSet || []);
  for (const m of selected) {
    if (m === "Naive") continue;
    const arr = tickerData.predictions?.[m];
    if (!Array.isArray(arr)) continue;
    datasets.push({
      label: m,
      data: arr.map((v) => (v != null && Number.isFinite(v) ? Number(v) : null)),
      borderColor: modelColor(m) ?? autoColorForKey(m, { sat: 62, light: 46 }),
      borderWidth: m === "Naive" ? 1 : 2,
      borderDash: MODEL_DASHES[m] ?? [],
      pointRadius: 0,
      spanGaps: false,
    });
  }

  const opts = baseChartOptions(n);
  opts.scales.y.ticks.callback = (v) => `$${Number(v).toFixed(0)}`;
  opts.plugins.tooltip.callbacks = {
    title: (items) => items?.[0]?.label ?? "",
    label: (ctx) => {
      const y = ctx.parsed?.y;
      if (y == null || !Number.isFinite(y)) return `${ctx.dataset.label}: —`;
      const a = ctx.chart.data.datasets?.[0]?.data?.[ctx.dataIndex];
      const delta = (a != null && Number.isFinite(a)) ? (y - a) : null;
      const d = delta != null ? ` (Δ ${delta >= 0 ? "+" : ""}${delta.toFixed(2)})` : "";
      return `${ctx.dataset.label}: ${Number(y).toFixed(2)}${d}`;
    },
  };

  if (regimeOverlay) {
    const ann = buildRegimeAnnotations({
      regime: tickerData.regime,
      regimeNames: tickerData.regime_names,
      labels,
    });
    opts.plugins.annotation = { annotations: ann };
  }

  const chart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: { labels, datasets },
    options: opts,
  });
  chartInstances.set(chartKey(key), chart);
  attachZoomReset({ chart, button: resetButton, key });
  return chart;
};

const renderRmseChart = ({ canvas, rows, resetButton } = {}) => {
  assert(canvas, "renderRmseChart: missing canvas");
  destroyChart("rmse");
  const sorted = (rows || []).slice().sort((a, b) => (a.rmse - b.rmse));
  const labels = sorted.map((r) => r.label);
  const data = sorted.map((r) => r.rmse);
  const colors = sorted.map((r) => r.color);

  const opts = baseChartOptions(labels.length);
  opts.scales.y.title = { display: true, text: "RMSE/mean (test)", color: getCssVar("--color-text-muted") };
  opts.plugins.legend.display = false;
  // Intentionally avoid hardcoded reference lines; keep chart data-driven.
  const chart = new Chart(canvas.getContext("2d"), {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: "RMSE/mean (test)", data, backgroundColor: colors, borderColor: colors, borderWidth: 1 }],
    },
    options: opts,
  });
  chartInstances.set(chartKey("rmse"), chart);
  attachZoomReset({ chart, button: resetButton, key: "rmse" });
  return chart;
};

const renderEquityChart = ({ canvas, tickerData, modelSet } = {}) => {
  assert(canvas, "renderEquityChart: missing canvas");
  destroyChart("equity");

  const labels = tickerData.dates;
  const n = labels.length;
  const logRets = tickerData.strategy_log_returns || {};

  const datasets = [];
  const naive = logRets?.Naive;
  if (Array.isArray(naive)) {
    datasets.push({
      label: "Naive",
      data: cumulativeFromLogReturns(naive),
      borderColor: modelColor("Naive") ?? "#888780",
      borderWidth: 1,
      borderDash: [6, 4],
      pointRadius: 0,
    });
  }

  for (const m of Array.from(modelSet || [])) {
    if (m === "Naive") continue;
    const lr = logRets?.[m];
    if (!Array.isArray(lr)) continue;
    datasets.push({
      label: m,
      data: cumulativeFromLogReturns(lr),
      borderColor: modelColor(m) ?? autoColorForKey(m, { sat: 62, light: 46 }),
      borderWidth: 2,
      borderDash: MODEL_DASHES[m] ?? [],
      pointRadius: 0,
    });
  }

  const opts = baseChartOptions(n);
  opts.plugins.tooltip.callbacks = {
    label: (ctx) => {
      const y = ctx.parsed?.y;
      if (y == null || !Number.isFinite(y)) return `${ctx.dataset.label}: —`;
      return `${ctx.dataset.label}: ${formatPct(y, 1)}`;
    },
  };
  opts.scales.y.ticks.callback = (v) => `${(Number(v) * 100).toFixed(0)}%`;
  opts.scales.y.title = { display: true, text: "Cumulative return", color: getCssVar("--color-text-muted") };

  const chart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: { labels, datasets },
    options: opts,
  });
  chartInstances.set(chartKey("equity"), chart);
  return chart;
};

const cumulativeFromLogReturns = (logReturns) => {
  let s = 0;
  const out = [];
  for (const r of logReturns) {
    if (r == null || !Number.isFinite(r)) {
      out.push(out.length ? out[out.length - 1] : 0);
      continue;
    }
    s += Number(r);
    out.push(Math.exp(s) - 1);
  }
  return out;
};

const renderFeatureImportance = ({ canvas, tickerData, modelPreference } = {}) => {
  assert(canvas, "renderFeatureImportance: missing canvas");
  destroyChart("fi");
  const fi = tickerData.feature_importance || {};

  const chosen = modelPreference?.find((m) => Array.isArray(fi[m]) && fi[m].length) ?? null;
  const arr = chosen ? fi[chosen] : [];
  const top = (arr || []).slice(0, 20);
  const labels = top.map((x) => FEATURE_LABELS[x.feature] ?? x.feature);
  const data = top.map((x) => x.importance);

  const opts = baseChartOptions(labels.length);
  opts.indexAxis = "y";
  opts.plugins.legend.display = false;
  opts.scales.x.title = { display: true, text: chosen ? `Importance (${chosen})` : "Importance", color: getCssVar("--color-text-muted") };

  const accent = getCssVar("--color-accent");
  const chart = new Chart(canvas.getContext("2d"), {
    type: "bar",
    data: { labels, datasets: [{ label: "Importance", data, backgroundColor: `${accent}bf`, borderColor: accent }] },
    options: opts,
  });
  chartInstances.set(chartKey("fi"), chart);
  return chart;
};

const renderRegimeTimeline = ({ host, dates, regime, regimeNames } = {}) => {
  clearNode(host);
  if (!Array.isArray(dates) || !Array.isArray(regime) || dates.length !== regime.length) {
    host.appendChild(createEl("div", { class: "muted" }, ["Regime data unavailable."]));
    return;
  }
  const names = Array.isArray(regimeNames) ? regimeNames : [];

  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "40");
  svg.setAttribute("viewBox", "0 0 1000 40");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Regime timeline");
  const g = document.createElementNS(NS, "g");
  svg.appendChild(g);

  const draw = () => {
    while (g.firstChild) g.removeChild(g.firstChild);
    const w = host.getBoundingClientRect().width;
    const width = Math.max(320, w);
    svg.setAttribute("viewBox", `0 0 ${width} 40`);

    let start = 0;
    for (let i = 1; i <= regime.length; i++) {
      if (i === regime.length || regime[i] !== regime[start]) {
        const r = regime[start];
        const nm = names[r] ?? `regime_${r}`;
        const c = REGIME_COLORS[nm] ?? "#888780";
        const x0 = (start / regime.length) * width;
        const x1 = (i / regime.length) * width;
        const rect = document.createElementNS(NS, "rect");
        rect.setAttribute("x", String(x0));
        rect.setAttribute("y", "10");
        rect.setAttribute("width", String(Math.max(1, x1 - x0)));
        rect.setAttribute("height", "20");
        rect.setAttribute("fill", c);
        rect.setAttribute("opacity", "0.85");
        const title = document.createElementNS(NS, "title");
        title.textContent = `${nm} • ${dates[start]} → ${dates[i - 1]} • ${i - start} trading days`;
        rect.appendChild(title);
        g.appendChild(rect);
        start = i;
      }
    }
    const border = document.createElementNS(NS, "rect");
    border.setAttribute("x", "0");
    border.setAttribute("y", "10");
    border.setAttribute("width", String(width));
    border.setAttribute("height", "20");
    border.setAttribute("fill", "none");
    border.setAttribute("stroke", getCssVar("--color-border"));
    g.appendChild(border);
  };

  const ro = new ResizeObserver(debounce(draw, 200));
  ro.observe(host);
  draw();
  chartCleanups.set(chartKey("regimeTimeline"), () => ro.disconnect());
  host.appendChild(svg);
};

