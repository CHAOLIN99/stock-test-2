/* App bootstrap, routing, scoped state → scoped render dispatch. */

Chart.defaults.devicePixelRatio = Math.min(window.devicePixelRatio ?? 1, 2);
Chart.defaults.font.family =
  "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif";

/* -------------------- state -------------------- */

const DIRTY = Object.freeze({
  DATA: 1 << 0,
  ROUTE: 1 << 1,
  TICKER: 1 << 2,
  MODELS: 1 << 3,
  DATE: 1 << 4,
  REGIME: 1 << 5,
  THEME: 1 << 6,
});

let _state = {
  data: null,
  activeTicker: null,
  activeModels: new Set(),
  dateRange: [null, null],
  darkMode: false,
  regimeOverlay: false,
  _abortSignal: null,
  _modelHint: "",
};

let _dirty = 0;
let _paintQueued = false;

const getState = () => _state;

const markDirty = (flags) => {
  _dirty |= flags;
  if (!_paintQueued) {
    _paintQueued = true;
    requestAnimationFrame(paint);
  }
};

const setState = (patch = {}) => {
  let flags = 0;
  if ("data" in patch && patch.data !== _state.data) flags |= DIRTY.DATA;
  if ("activeTicker" in patch && patch.activeTicker !== _state.activeTicker) flags |= DIRTY.TICKER;
  if ("activeModels" in patch && !setEquals(patch.activeModels, _state.activeModels)) flags |= DIRTY.MODELS;
  if ("dateRange" in patch && !arrayEquals(patch.dateRange, _state.dateRange)) flags |= DIRTY.DATE;
  if ("darkMode" in patch && patch.darkMode !== _state.darkMode) flags |= DIRTY.THEME;
  if ("regimeOverlay" in patch && patch.regimeOverlay !== _state.regimeOverlay) flags |= DIRTY.REGIME;

  _state = { ..._state, ...patch };
  if (patch.activeModels instanceof Set) _state.activeModels = new Set(patch.activeModels);

  if (flags) markDirty(flags);
};

/* -------------------- model tiering -------------------- */

const tierForModel = (m) => {
  if (m === "HMM_Regime") return "hmm";
  if (m === "VotingEnsemble" || m === "StackingEnsemble") return "ensemble";
  if (m === "RNN" || m === "BiLSTM" || m.startsWith("LSTM")) return "dl";
  if (m === "Ridge" || m === "MARS" || m === "RandomForest" || m === "SVR" ||
      m === "MLP" || m === "XGBoost" || m === "LightGBM") return "ml";
  return "baseline";
};

const modelGroupMeta = Object.freeze({
  baseline: { title: "Baselines", note: "Tier 0" },
  ml: { title: "Multivariate ML", note: "Tier 3" },
  dl: { title: "Deep learning", note: "Tier 4" },
  ensemble: { title: "Ensembles", note: "Tier 5" },
  hmm: { title: "Regime", note: "HMM" },
});

const buildModelGroups = (models) => {
  const byTier = new Map();
  for (const m of models || []) {
    if (!m || m === "Naive") continue;
    const tier = tierForModel(m);
    if (!byTier.has(tier)) byTier.set(tier, []);
    byTier.get(tier).push(m);
  }
  const out = [];
  for (const tier of ["baseline", "ml", "dl", "ensemble", "hmm"]) {
    const ms = (byTier.get(tier) || []).slice().sort();
    if (!ms.length) continue;
    const meta = modelGroupMeta[tier];
    out.push({ title: meta.title, note: meta.note, models: ms });
  }
  return out;
};

/* -------------------- dom refs -------------------- */

const dom = {};
const tickerDom = new Map();

const initDom = () => {
  dom.bannerHost = document.getElementById("bannerHost");
  dom.skeleton = document.getElementById("loadingSkeleton");
  dom.appRoot = document.getElementById("appRoot");

  dom.pageOverview = document.getElementById("page-overview");
  dom.pageTicker = document.getElementById("page-ticker");
  dom.pageCompare = document.getElementById("page-compare");
  dom.pageMethod = document.getElementById("page-methodology");

  dom.generatedAt = document.getElementById("generatedAt");
  dom.tickerPills = document.getElementById("tickerPills");
  dom.dateStart = document.getElementById("dateStart");
  dom.dateEnd = document.getElementById("dateEnd");
  dom.dateApply = document.getElementById("dateApply");

  dom.darkToggle = document.getElementById("darkModeToggle");
  dom.regimeToggle = document.getElementById("regimeToggle");

  dom.overviewMetricGrid = document.getElementById("overviewMetricGrid");
  dom.rmseChart = document.getElementById("rmseChart");
  dom.resetRmseZoom = document.getElementById("resetRmseZoom");
  dom.exportRmsePng = document.getElementById("exportRmsePng");
  dom.dirHeatmap = document.getElementById("dirHeatmap");

  dom.smallMultiples = document.getElementById("smallMultiples");

  dom.tickerSectionsHost = document.getElementById("tickerSections");
  dom.tickerTemplate = document.getElementById("tickerSectionTemplate");
};

const ensureTickerSection = (t) => {
  if (tickerDom.has(t)) return tickerDom.get(t);
  const frag = dom.tickerTemplate.content.cloneNode(true);
  const section = frag.querySelector('[data-component="ticker-section"]');
  section.dataset.ticker = t;

  const refs = {
    section,
    title: section.querySelector('[data-role="tickerTitle"]'),
    subtitle: section.querySelector('[data-role="tickerSubtitle"]'),
    metricGrid: section.querySelector('[data-role="tickerMetricGrid"]'),
    modelPanel: section.querySelector('[data-role="modelTogglePanel"]'),
    modelHint: section.querySelector('[data-role="modelPanelHint"]'),
    priceCanvas: section.querySelector('[data-role="priceChart"]'),
    equityCanvas: section.querySelector('[data-role="equityChart"]'),
    fiCanvas: section.querySelector('[data-role="fiChart"]'),
    regimeHost: section.querySelector('[data-role="regimeTimelineHost"]'),
    exportPriceBtn: section.querySelector('[data-role="exportPricePng"]'),
    resetPriceZoomBtn: section.querySelector('[data-role="resetPriceZoom"]'),
  };

  refs.title.textContent = t;
  section.hidden = true;
  dom.tickerSectionsHost.appendChild(section);
  tickerDom.set(t, refs);

  refs.exportPriceBtn.addEventListener("click", () => exportChartPng("price", `${t}_price`));
  return refs;
};

const applyTickerVisibility = (activeTicker) => {
  for (const [t, refs] of tickerDom) {
    const shouldShow = t === activeTicker;
    if (refs.section.hidden !== !shouldShow) refs.section.hidden = !shouldShow;
  }
};

const syncDateInputsForTicker = (ticker) => {
  const td0 = _state.data?.data?.[ticker];
  if (!td0) return;
  if (td0._minDate) dom.dateStart.min = td0._minDate;
  if (td0._maxDate) dom.dateEnd.max = td0._maxDate;
};

/* -------------------- theme -------------------- */

const applyTheme = (dark) => {
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  invalidateCssVarCache();
};

/** Refresh chart colors when theme flips, without recreating charts. */
const refreshChartsForTheme = () => {
  const text = getCssVar("--color-text");
  const muted = getCssVar("--color-text-muted");
  const border = getCssVar("--color-border");
  for (const chart of chartInstances.values()) {
    const opts = chart.options;
    if (opts?.scales?.x?.ticks) opts.scales.x.ticks.color = muted;
    if (opts?.scales?.y?.ticks) opts.scales.y.ticks.color = muted;
    if (opts?.scales?.x?.grid) opts.scales.x.grid.color = border;
    if (opts?.scales?.y?.grid) opts.scales.y.grid.color = border;
    if (opts?.plugins?.legend?.labels) opts.plugins.legend.labels.color = text;
    for (const ds of chart.data?.datasets ?? []) {
      if (ds.label === "Actual") ds.borderColor = text;
      else if (ds.label) {
        const c = modelColor(ds.label);
        if (c) ds.borderColor = c;
      }
    }
    chart.update("none");
  }
};

/* -------------------- controls / events -------------------- */

const initControls = (abortSignal) => {
  const saved = localStorage.getItem("darkMode");
  let dark;
  if (saved === "1" || saved === "0") {
    dark = saved === "1";
  } else {
    dark = window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
  }
  dom.darkToggle.checked = dark;
  applyTheme(dark);
  setState({ darkMode: dark });

  dom.darkToggle.addEventListener("change", () => {
    const v = !!dom.darkToggle.checked;
    localStorage.setItem("darkMode", v ? "1" : "0");
    applyTheme(v);
    setState({ darkMode: v });
  }, { signal: abortSignal });

  dom.regimeToggle.addEventListener("change", () => {
    setState({ regimeOverlay: !!dom.regimeToggle.checked });
  }, { signal: abortSignal });

  const applyDate = debounce(() => {
    const a = dom.dateStart.value || null;
    const b = dom.dateEnd.value || null;
    if (a && b && a > b) {
      showBanner({ host: dom.bannerHost, message: "Date range invalid: start must be before end." });
      return;
    }
    clearBanner(dom.bannerHost);
    setState({ dateRange: [a, b] });
  }, 200);
  dom.dateApply.addEventListener("click", applyDate, { signal: abortSignal });

  // Enter key on date inputs applies range.
  const onDateEnter = (e) => { if (e.key === "Enter") applyDate(); };
  dom.dateStart.addEventListener("keydown", onDateEnter, { signal: abortSignal });
  dom.dateEnd.addEventListener("keydown", onDateEnter, { signal: abortSignal });

  dom.exportRmsePng.addEventListener("click", () => exportChartPng("rmse", "rmse"), { signal: abortSignal });

  // Sticky-header shadow on scroll.
  const onScroll = rafThrottle(() => {
    const scrolled = window.scrollY > 4;
    document.documentElement.classList.toggle("is-scrolled", scrolled);
  });
  window.addEventListener("scroll", onScroll, { passive: true, signal: abortSignal });

  // Keyboard: g then t/o/c/m for quick nav.
  let pendingG = false;
  let gTimer;
  window.addEventListener("keydown", (e) => {
    if (e.target?.matches?.("input, textarea, select")) return;
    if (e.key === "g") {
      pendingG = true;
      clearTimeout(gTimer);
      gTimer = setTimeout(() => (pendingG = false), 800);
      return;
    }
    if (!pendingG) return;
    pendingG = false;
    const map = { o: "overview", t: "ticker", c: "compare", m: "methodology" };
    const target = map[e.key];
    if (!target) return;
    if (target === "ticker") {
      const sym = _state.activeTicker || _state.data?.tickers?.[0] || "NVDA";
      location.hash = `#/ticker/${sym}`;
    } else {
      location.hash = `#/${target}`;
    }
  }, { signal: abortSignal });
};

const exportChartPng = (key, filename) => {
  const chart = chartInstances.get(key);
  if (!chart) return;
  const a = document.createElement("a");
  a.href = chart.toBase64Image();
  a.download = `${filename || key}.png`;
  a.click();
};

/* -------------------- routing -------------------- */

const showPage = (name) => {
  const map = {
    overview: dom.pageOverview,
    ticker: dom.pageTicker,
    compare: dom.pageCompare,
    methodology: dom.pageMethod,
  };
  for (const [n, el] of Object.entries(map)) {
    const show = n === name;
    if (el.hidden !== !show) el.hidden = !show;
  }
  for (const a of document.querySelectorAll(".top-nav__link")) {
    if (a.dataset.nav === name) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
};

const routeToHash = (name, params = {}) => {
  if (name === "ticker") return `#/ticker/${params.symbol || _state.activeTicker || "NVDA"}`;
  return `#/${name}`;
};

const currentRoute = () => parseRoute(location.hash);

const router = () => {
  const r = currentRoute();
  if (r.name === "ticker" && r.params.symbol) {
    const tickers = _state.data?._meta?.tickers || [];
    if (tickers.includes(r.params.symbol) && r.params.symbol !== _state.activeTicker) {
      setState({ activeTicker: r.params.symbol });
    }
  }
  markDirty(DIRTY.ROUTE);
};

/* -------------------- slicing (memoized on ticker + range) -------------------- */

const _sliceCache = new Map();

/** Binary search: leftmost index whose date is >= target (or length if none). */
const _lowerBound = (dates, target) => {
  let lo = 0, hi = dates.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (dates[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
};

const sliceByDateRange = (td, dateRange, cacheKey) => {
  const [a, b] = dateRange || [null, null];
  if (!a && !b) return td;
  if (!td?.dates?.length) return td;

  const key = `${cacheKey}|${a ?? ""}|${b ?? ""}`;
  const cached = _sliceCache.get(key);
  if (cached && cached.src === td) return cached.value;

  const nDates = td.dates.length;
  // Use the exact date index when present; otherwise fall forward to the nearest trading day.
  const resolveStart = (s) => (td._dateIndex?.[s] ?? _lowerBound(td.dates, s));
  const resolveEnd = (s) => {
    const exact = td._dateIndex?.[s];
    if (exact != null) return exact;
    const lb = _lowerBound(td.dates, s);
    return lb >= nDates ? nDates - 1 : (td.dates[lb] === s ? lb : Math.max(0, lb - 1));
  };
  const startIdx = a ? resolveStart(a) : 0;
  const endIdx = b ? resolveEnd(b) : (nDates - 1);
  const i0 = clamp(startIdx, 0, nDates - 1);
  const i1 = clamp(endIdx, i0, nDates - 1);

  const sub = {
    ...td,
    dates: td.dates.slice(i0, i1 + 1),
    actual: td.actual.slice(i0, i1 + 1),
    predictions: {},
    equity: {},
    regime: Array.isArray(td.regime) ? td.regime.slice(i0, i1 + 1) : td.regime,
  };
  for (const [m, arr] of Object.entries(td.predictions || {})) sub.predictions[m] = arr.slice(i0, i1 + 1);
  for (const [m, arr] of Object.entries(td.equity || {})) sub.equity[m] = arr.slice(i0, i1 + 1);

  _sliceCache.set(key, { src: td, value: sub });
  if (_sliceCache.size > 32) {
    const first = _sliceCache.keys().next().value;
    _sliceCache.delete(first);
  }
  return sub;
};

/* -------------------- render: overview -------------------- */

const renderOverview = () => {
  const { data } = _state;
  if (!data) return;

  const cards = [];
  for (const t of data.tickers) {
    const best = data._meta.perTicker[t].bestModel;
    const m = best ? data.data[t].metrics?.[best] : null;
    cards.push({
      keyLabel: `${t} best`,
      valueText: best ?? "—",
      subText: best ? `RMSE/mean ${formatNum(m?.rmse_mean_test, 4)}` : "",
    });
  }
  renderMetricCards({ host: dom.overviewMetricGrid, cards });

  const rmseRows = [];
  for (const t of data.tickers) {
    const best = data._meta.perTicker[t].bestModel;
    const v = best ? data.data[t].metrics?.[best]?.rmse_mean_test : null;
    if (v != null && Number.isFinite(v)) {
      rmseRows.push({ label: `${t} · ${best}`, rmse: Number(v), color: tierColor(tierForModel(best)) });
    }
  }
  renderRmseChart({ canvas: dom.rmseChart, rows: rmseRows, resetButton: dom.resetRmseZoom });

  const heatModels = (data._meta.allModels || []).filter((m) => m !== "Naive").slice(0, 14);
  const lookup = {};
  for (const t of data.tickers) lookup[t] = data.data[t].metrics || {};
  renderDirectionalHeatmap({ host: dom.dirHeatmap, tickers: data.tickers, models: heatModels, metricsLookup: lookup });
};

/* -------------------- render: ticker (scoped updates) -------------------- */

const renderTickerFull = (ticker) => {
  const { data } = _state;
  if (!data) return;
  const refs = ensureTickerSection(ticker);

  const td0 = data.data[ticker];
  if (!td0) {
    showBanner({ host: dom.bannerHost, message: `Ticker ${ticker} not found in loaded dataset.` });
    return;
  }
  const td = sliceByDateRange(td0, _state.dateRange, ticker);

  const best = data._meta.perTicker[ticker].bestModel;
  const bm = best ? td0.metrics?.[best] : null;

  renderMetricCards({
    host: refs.metricGrid,
    cards: [
      { keyLabel: "RMSE/mean", valueText: formatNum(bm?.rmse_mean_test, 4), subText: "Test" },
      { keyLabel: "Dir Acc", valueText: (bm?.dir_acc != null && Number.isFinite(bm.dir_acc)) ? `${(bm.dir_acc * 100).toFixed(1)}%` : "—", subText: "Test" },
      { keyLabel: "Sharpe", valueText: formatNum(bm?.sharpe, 2), subText: "Strategy" },
      { keyLabel: "R²", valueText: formatNum(bm?.r2, 2), subText: "Test" },
      { keyLabel: "Max DD", valueText: (bm?.max_dd != null && Number.isFinite(bm.max_dd)) ? formatPct(bm.max_dd, 1) : "—", subText: "Strategy" },
      { keyLabel: "Best model", valueText: best ?? "—", subText: "Lowest RMSE/mean" },
    ],
  });

  const availableModels = data._meta.perTicker?.[ticker]?.models || Object.keys(td0.predictions || {});
  const modelGroups = buildModelGroups(availableModels);
  renderModelTogglePanel({
    host: refs.modelPanel,
    modelGroups,
    activeSet: _state.activeModels,
    onToggle: (m) => {
      const cur = new Set(_state.activeModels);
      if (cur.has(m)) cur.delete(m);
      else cur.add(m);
      if (cur.size > 6) {
        refs.modelHint.textContent = "Max 6 models visible. Deselect one to add another.";
        _state._modelHint = refs.modelHint.textContent;
        return;
      }
      refs.modelHint.textContent = "";
      _state._modelHint = "";
      setState({ activeModels: cur });
    },
    abortSignal: _state._abortSignal,
  });
  refs.modelHint.textContent = _state._modelHint || "";

  renderPriceChart({
    key: "price",
    canvas: refs.priceCanvas,
    tickerData: td,
    modelSet: _state.activeModels,
    regimeOverlay: _state.regimeOverlay,
    resetButton: refs.resetPriceZoomBtn,
  });

  renderEquityChart({ canvas: refs.equityCanvas, tickerData: td, modelSet: _state.activeModels });

  const pref = [best, ...Array.from(_state.activeModels)].filter(Boolean);
  renderFeatureImportance({ canvas: refs.fiCanvas, tickerData: td0, modelPreference: pref });

  destroyChart("regimeTimeline");
  renderRegimeTimeline({ host: refs.regimeHost, dates: td0.dates, regime: td0.regime, regimeNames: td0.regime_names });
};

const renderTickerModels = (ticker) => {
  // Only datasets on price + equity charts change; toggle chip aria-pressed.
  const refs = tickerDom.get(ticker);
  const td0 = _state.data?.data?.[ticker];
  if (!refs || !td0) return;
  const td = sliceByDateRange(td0, _state.dateRange, ticker);

  renderModelTogglePanel({
    host: refs.modelPanel,
    modelGroups: buildModelGroups(_state.data._meta.perTicker[ticker].models),
    activeSet: _state.activeModels,
    onToggle: (m) => {
      const cur = new Set(_state.activeModels);
      if (cur.has(m)) cur.delete(m);
      else cur.add(m);
      if (cur.size > 6) {
        refs.modelHint.textContent = "Max 6 models visible. Deselect one to add another.";
        return;
      }
      refs.modelHint.textContent = "";
      setState({ activeModels: cur });
    },
    abortSignal: _state._abortSignal,
  });

  renderPriceChart({
    key: "price",
    canvas: refs.priceCanvas,
    tickerData: td,
    modelSet: _state.activeModels,
    regimeOverlay: _state.regimeOverlay,
    resetButton: refs.resetPriceZoomBtn,
  });
  renderEquityChart({ canvas: refs.equityCanvas, tickerData: td, modelSet: _state.activeModels });

  const best = _state.data._meta.perTicker[ticker].bestModel;
  const pref = [best, ...Array.from(_state.activeModels)].filter(Boolean);
  renderFeatureImportance({ canvas: refs.fiCanvas, tickerData: td0, modelPreference: pref });
};

const renderTickerRegime = (ticker) => {
  const refs = tickerDom.get(ticker);
  const td0 = _state.data?.data?.[ticker];
  if (!refs || !td0) return;
  const td = sliceByDateRange(td0, _state.dateRange, ticker);
  renderPriceChart({
    key: "price",
    canvas: refs.priceCanvas,
    tickerData: td,
    modelSet: _state.activeModels,
    regimeOverlay: _state.regimeOverlay,
    resetButton: refs.resetPriceZoomBtn,
  });
};

/* -------------------- render: compare -------------------- */

const renderCompare = () => {
  const { data } = _state;
  if (!data) return;

  // Ensure one panel per ticker; keep DOM stable.
  const existing = new Map();
  for (const panel of dom.smallMultiples.children) existing.set(panel.dataset.ticker, panel);

  for (const t of data.tickers) {
    let panel = existing.get(t);
    let canvas;
    if (!panel) {
      panel = createEl("div", { class: "panel panel--small", dataset: { ticker: t } }, [
        createEl("div", { class: "panel__header" }, [createEl("div", { class: "h2" }, [t])]),
        (() => {
          const frame = createEl("div", { class: "chart-frame chart-frame--mini" });
          canvas = createEl("canvas", { height: "160", "aria-label": `Mini price chart ${t}` });
          frame.appendChild(canvas);
          return frame;
        })(),
      ]);
      dom.smallMultiples.appendChild(panel);
    } else {
      canvas = panel.querySelector("canvas");
    }
    existing.delete(t);

    const td0 = data.data[t];
    const best = data._meta.perTicker[t].bestModel;
    const set = new Set(best ? [best] : []);
    renderPriceChart({
      key: `compare_${t}`,
      canvas,
      tickerData: sliceByDateRange(td0, _state.dateRange, t),
      modelSet: set,
      regimeOverlay: false,
    });
  }

  // Remove panels for removed tickers.
  for (const [t, panel] of existing) {
    destroyChart(`compare_${t}`);
    panel.remove();
  }
};

/* -------------------- paint: dispatch by dirty flags -------------------- */

const _pageCharts = Object.freeze({
  overview: ["rmse"],
  ticker: ["price", "equity", "fi", "regimeTimeline"],
  compare: null, // dynamic keys prefixed with "compare_"
  methodology: [],
});

let _currentPage = null;

const paint = () => {
  _paintQueued = false;
  const flags = _dirty;
  _dirty = 0;
  if (!_state.data) return;

  const r = currentRoute();
  const routeChanged = _currentPage !== r.name;

  if (routeChanged) {
    // Destroy only charts that belong to the page we're leaving.
    if (_currentPage === "overview") destroyChartsMatching((k) => k === "rmse");
    else if (_currentPage === "ticker") destroyChartsMatching((k) => ["price", "equity", "fi", "regimeTimeline"].includes(k));
    else if (_currentPage === "compare") destroyChartsMatching((k) => k.startsWith("compare_"));
    _currentPage = r.name;
    showPage(r.name);
  }

  if (flags & DIRTY.THEME) {
    refreshChartsForTheme();
  }

  if (r.name === "overview") {
    if (routeChanged || (flags & (DIRTY.DATA | DIRTY.ROUTE))) renderOverview();
    return;
  }

  if (r.name === "ticker") {
    const ticker = _state.activeTicker;
    if (!ticker) return;
    dom.tickerPillsApi?.setActive?.(ticker);

    // Render first so a newly created section exists in tickerDom before we toggle visibility.
    if (routeChanged || (flags & (DIRTY.DATA | DIRTY.TICKER | DIRTY.DATE))) {
      renderTickerFull(ticker);
      syncDateInputsForTicker(ticker);
    } else if (flags & DIRTY.MODELS) {
      renderTickerModels(ticker);
    } else if (flags & DIRTY.REGIME) {
      renderTickerRegime(ticker);
    }
    applyTickerVisibility(ticker);
    return;
  }

  if (r.name === "compare") {
    if (routeChanged || (flags & (DIRTY.DATA | DIRTY.DATE | DIRTY.ROUTE))) renderCompare();
    return;
  }
};

/* -------------------- boot -------------------- */

const boot = async () => {
  initDom();

  const ac = new AbortController();
  _state._abortSignal = ac.signal;

  initControls(ac.signal);

  const renderPills = (tickers, active) => {
    dom.tickerPillsApi = renderTickerPills({
      host: dom.tickerPills,
      tickers,
      active,
      onSelect: (t) => {
        const r = currentRoute();
        if (r.name !== "ticker") location.hash = routeToHash("ticker", { symbol: t });
        else setState({ activeTicker: t });
      },
      abortSignal: ac.signal,
    });
  };

  dom.skeleton.hidden = false;
  dom.appRoot.hidden = true;
  clearBanner(dom.bannerHost);

  const load = async () => {
    try {
      dom.skeleton.hidden = false;
      dom.appRoot.hidden = true;
      clearBanner(dom.bannerHost);

      const d = await loadChartsData({ signal: ac.signal });
      const tickers = d?._meta?.tickers || [];
      assert(Array.isArray(tickers) && tickers.length > 0, "Dataset has no tickers to display.");

      const current = _state.activeTicker;
      const initialTicker = tickers.includes(current) ? current : tickers[0];

      // Pre-create the initial ticker section only (others created on demand).
      ensureTickerSection(initialTicker);
      renderPills(tickers, initialTicker);

      const best = d._meta?.perTicker?.[initialTicker]?.bestModel ?? null;
      const modelsForTicker = d._meta?.perTicker?.[initialTicker]?.models || [];
      const fallback = modelsForTicker.find((m) => m && m !== "Naive" && m !== best) ?? null;
      const initialModels = new Set([best, fallback].filter(Boolean));

      setState({ data: d, activeTicker: initialTicker, activeModels: initialModels });

      dom.generatedAt.textContent = d.generated_at ? `Generated ${d.generated_at}` : "Offline data";
      dom.skeleton.hidden = true;
      dom.appRoot.hidden = false;
      if (!location.hash) location.hash = "#/overview";
      router();
    } catch (e) {
      if (e?.name === "AbortError") return;
      dom.skeleton.hidden = true;
      dom.appRoot.hidden = true;
      showBanner({ host: dom.bannerHost, message: e?.message ?? String(e), onRetry: load });
    }
  };

  window.addEventListener("hashchange", router, { signal: ac.signal });
  await load();
};

document.addEventListener("DOMContentLoaded", () => { void boot(); });
