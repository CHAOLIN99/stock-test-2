/* App bootstrap + routing + state. */

Chart.defaults.devicePixelRatio = Math.min(window.devicePixelRatio ?? 1, 2);

const AppState = Object.freeze({
  data: null,
  activeTicker: null,
  activeModels: new Set(),
  dateRange: [null, null],
  darkMode: false,
  regimeOverlay: false,
});

let _state = {
  data: null,
  activeTicker: AppState.activeTicker,
  activeModels: new Set(),
  dateRange: [null, null],
  darkMode: false,
  regimeOverlay: false,
};

const getState = () => _state;

const setState = (patch, { render: shouldRender = true } = {}) => {
  const prevTicker = _state.activeTicker;
  const next = deepMerge(_state, patch || {});
  if (patch?.activeModels instanceof Set) next.activeModels = new Set(patch.activeModels);
  _state = next;
  if (prevTicker !== _state.activeTicker) dom.tickerPillsApi?.setActive?.(_state.activeTicker);
  if (shouldRender) render(); // targeted inside render() based on route
};

const tierForModel = (m) => {
  if (m === "HMM_Regime") return "hmm";
  if (m === "VotingEnsemble" || m === "StackingEnsemble") return "ensemble";
  if (m === "RNN" || m.startsWith("LSTM") || m === "BiLSTM") return "dl";
  if (m === "Ridge" || m === "MARS" || m === "RandomForest" || m === "SVR" || m === "MLP" || m === "XGBoost" || m === "LightGBM") return "ml";
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
  const order = ["baseline", "ml", "dl", "ensemble", "hmm"];
  const out = [];
  for (const tier of order) {
    const ms = (byTier.get(tier) || []).slice().sort();
    if (!ms.length) continue;
    const meta = modelGroupMeta[tier] || { title: tier, note: "" };
    out.push({ title: meta.title, note: meta.note, models: ms });
  }
  return out;
};

const dom = {};
const tickerDom = new Map(); // ticker -> element refs

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

const initTickerSections = (tickers) => {
  clearNode(dom.tickerSectionsHost);
  tickerDom.clear();
  for (const t of tickers || []) {
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
    refs.subtitle.textContent = "Prediction performance and diagnostics.";
    section.hidden = true;

    dom.tickerSectionsHost.appendChild(section);
    tickerDom.set(t, refs);
  }
};

const applyTheme = (dark) => {
  document.documentElement.dataset.theme = dark ? "dark" : "light";
};

const initControls = (abortSignal) => {
  const saved = localStorage.getItem("darkMode");
  const dark = saved === "1";
  dom.darkToggle.checked = dark;
  setState({ darkMode: dark });
  applyTheme(dark);

  dom.darkToggle.addEventListener(
    "change",
    () => {
      const v = !!dom.darkToggle.checked;
      localStorage.setItem("darkMode", v ? "1" : "0");
      applyTheme(v);
      setState({ darkMode: v });
    },
    { signal: abortSignal }
  );

  dom.regimeToggle.addEventListener(
    "change",
    () => setState({ regimeOverlay: !!dom.regimeToggle.checked }),
    { signal: abortSignal }
  );

  dom.dateApply.addEventListener(
    "click",
    debounce(() => {
      const a = dom.dateStart.value || null;
      const b = dom.dateEnd.value || null;
      if (a && b && a > b) {
        showBanner({ host: dom.bannerHost, message: "Date range invalid: start must be before end." });
        return;
      }
      clearBanner(dom.bannerHost);
      setState({ dateRange: [a, b] });
    }, 300),
    { signal: abortSignal }
  );

  dom.exportRmsePng.addEventListener(
    "click",
    () => exportChartPng("rmse"),
    { signal: abortSignal }
  );
};

const exportChartPng = (key) => {
  const c = chartInstances.get(key);
  if (!c) return;
  const url = c.toBase64Image();
  const a = document.createElement("a");
  a.href = url;
  a.download = `${key}.png`;
  a.click();
};

const showPage = (name) => {
  dom.pageOverview.hidden = name !== "overview";
  dom.pageTicker.hidden = name !== "ticker";
  dom.pageCompare.hidden = name !== "compare";
  dom.pageMethod.hidden = name !== "methodology";

  for (const a of document.querySelectorAll(".top-nav__link")) {
    a.removeAttribute("aria-current");
  }
  const active = document.querySelector(`.top-nav__link[data-nav="${name}"]`);
  if (active) active.setAttribute("aria-current", "page");
};

const routeToHash = (name, params = {}) => {
  if (name === "ticker") return `#/ticker/${params.symbol || "NVDA"}`;
  return `#/${name}`;
};

const router = () => {
  destroyAllCharts();
  const r = parseRoute(location.hash);
  const name = r.name;
  showPage(name);

  if (name === "ticker") {
    const sym = r.params.symbol;
    const tickers = getState().data?._meta?.tickers || [];
    if (tickers.includes(sym) && sym !== getState().activeTicker) {
      setState({ activeTicker: sym }, { render: false });
    }
  }
  // Render exactly once per route change. If we updated state above, suppress the
  // extra render that setState() would normally trigger.
  render();
};

const applyTickerVisibility = (activeTicker) => {
  for (const [t, refs] of tickerDom.entries()) refs.section.hidden = t !== activeTicker;
};

const sliceByDateRange = (td, dateRange) => {
  const [a, b] = dateRange || [null, null];
  if (!a && !b) return td;
  const startIdx = a ? (td._dateIndex?.[a] ?? 0) : 0;
  const endIdx = b ? (td._dateIndex?.[b] ?? (td.dates.length - 1)) : (td.dates.length - 1);
  const i0 = clamp(startIdx, 0, td.dates.length - 1);
  const i1 = clamp(endIdx, i0, td.dates.length - 1);

  const sub = {
    ...td,
    dates: td.dates.slice(i0, i1 + 1),
    actual: td.actual.slice(i0, i1 + 1),
    predictions: {},
    regime: Array.isArray(td.regime) ? td.regime.slice(i0, i1 + 1) : td.regime,
  };
  for (const [m, arr] of Object.entries(td.predictions || {})) sub.predictions[m] = arr.slice(i0, i1 + 1);
  if (td.strategy_log_returns) {
    sub.strategy_log_returns = {};
    for (const [m, arr] of Object.entries(td.strategy_log_returns)) sub.strategy_log_returns[m] = arr.slice(i0, i1 + 1);
  }
  return sub;
};

const renderOverview = () => {
  const st = getState();
  const data = st.data;
  if (!data) return;

  const cards = [];
  for (const t of data.tickers) {
    const best = data._meta.perTicker[t].bestModel;
    const m = best ? data.data[t].metrics?.[best] : null;
    cards.push({
      keyLabel: `${t} best RMSE/mean`,
      valueText: best ? `${best}` : "—",
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

const renderTicker = (ticker) => {
  const st = getState();
  const data = st.data;
  if (!data) return;
  const refs = tickerDom.get(ticker);
  if (!refs) return;

  const td0 = data.data[ticker];
  if (!td0) {
    showBanner({ host: dom.bannerHost, message: `Ticker ${ticker} not found in loaded dataset.` });
    return;
  }
  const td = sliceByDateRange(td0, st.dateRange);

  const best = data._meta.perTicker[ticker].bestModel;
  const bm = best ? td0.metrics?.[best] : null;

  const cards = [
    { keyLabel: "RMSE/mean", valueText: formatNum(bm?.rmse_mean_test, 4), subText: "Test" },
    { keyLabel: "Dir Acc", valueText: safeVal(bm?.dir_acc, "—") !== "—" ? `${(bm.dir_acc * 100).toFixed(1)}%` : "—", subText: "Test" },
    { keyLabel: "Sharpe", valueText: formatNum(bm?.sharpe, 2), subText: "Strategy" },
    { keyLabel: "R²", valueText: formatNum(bm?.r2, 2), subText: "Test" },
    { keyLabel: "Max DD", valueText: safeVal(bm?.max_dd, "—") !== "—" ? formatPct(bm.max_dd, 1) : "—", subText: "Strategy" },
    { keyLabel: "Best model", valueText: best ?? "—", subText: "Lowest RMSE/mean (test)" },
  ];
  renderMetricCards({ host: refs.metricGrid, cards });

  // model toggles (shared selection across tickers)
  const availableModels = data._meta.perTicker?.[ticker]?.models || Object.keys(td0.predictions || {});
  const modelGroups = buildModelGroups(availableModels);
  renderModelTogglePanel({
    host: refs.modelPanel,
    modelGroups,
    activeSet: st.activeModels,
    onToggle: (m) => {
      const cur = new Set(st.activeModels);
      if (cur.has(m)) cur.delete(m);
      else cur.add(m);
      if (cur.size > 6) {
        refs.modelHint.textContent = "Max 6 models visible. Deselect one to add another.";
        return;
      }
      refs.modelHint.textContent = "";
      setState({ activeModels: cur });
    },
    abortSignal: st._abortSignal,
  });

  const onExport = () => exportChartPng("price");
  refs.exportPriceBtn.onclick = onExport;

  renderPriceChart({
    key: "price",
    canvas: refs.priceCanvas,
    tickerData: td,
    modelSet: st.activeModels,
    regimeOverlay: st.regimeOverlay,
    resetButton: refs.resetPriceZoomBtn,
  });

  renderEquityChart({ canvas: refs.equityCanvas, tickerData: td, modelSet: st.activeModels });

  const pref = [best, ...Array.from(st.activeModels)];
  renderFeatureImportance({ canvas: refs.fiCanvas, tickerData: td0, modelPreference: pref.filter(Boolean) });

  destroyChart("regimeTimeline");
  renderRegimeTimeline({ host: refs.regimeHost, dates: td0.dates, regime: td0.regime, regimeNames: td0.regime_names });
};

const renderCompare = () => {
  const st = getState();
  const data = st.data;
  if (!data) return;
  clearNode(dom.smallMultiples);

  for (const t of data.tickers) {
    const box = createEl("div", { class: "panel" });
    const header = createEl("div", { class: "panel__header" }, [createEl("div", { class: "h2" }, [t])]);
    const frame = createEl("div", { class: "chart-frame" });
    const canvas = createEl("canvas", { height: "160", "aria-label": `Mini price chart ${t}` });
    frame.appendChild(canvas);
    box.appendChild(header);
    box.appendChild(frame);
    dom.smallMultiples.appendChild(box);

    // Render mini chart (actual + best model)
    const td0 = data.data[t];
    const best = data._meta.perTicker[t].bestModel;
    const set = new Set(best ? [best] : []);
    renderPriceChart({ key: `compare_${t}`, canvas, tickerData: sliceByDateRange(td0, st.dateRange), modelSet: set, regimeOverlay: false });
  }
};

const render = () => {
  const st = getState();
  const data = st.data;
  if (!data) return;

  const r = parseRoute(location.hash);
  if (r.name === "overview") renderOverview();
  if (r.name === "ticker") {
    dom.tickerPillsApi?.setActive?.(st.activeTicker);
    applyTickerVisibility(st.activeTicker);
    renderTicker(st.activeTicker);
  }
  if (r.name === "compare") renderCompare();
};

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
        const r = parseRoute(location.hash);
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
      const tickers = d?._meta?.tickers || d?.tickers || [];
      assert(Array.isArray(tickers) && tickers.length > 0, "Dataset has no tickers to display.");

      const initialTicker = tickers.includes(getState().activeTicker) ? getState().activeTicker : tickers[0];
      initTickerSections(tickers);
      renderPills(tickers, initialTicker);

      // Pick sane defaults from the dataset: best model for first ticker (if available) + one extra.
      const best = d._meta?.perTicker?.[initialTicker]?.bestModel ?? null;
      const modelsForTicker = d._meta?.perTicker?.[initialTicker]?.models || Object.keys(d.data?.[initialTicker]?.predictions || {});
      const fallback = modelsForTicker.find((m) => m && m !== "Naive" && m !== best) ?? null;
      const initialModels = new Set([best, fallback].filter(Boolean));

      // Avoid rendering while appRoot is hidden; router() will render once after we show the UI.
      setState({ data: d, activeTicker: initialTicker, activeModels: initialModels }, { render: false });
      dom.generatedAt.textContent = d.generated_at ? `Generated ${d.generated_at}` : "Generated date unavailable";
      dom.skeleton.hidden = true;
      dom.appRoot.hidden = false;
      if (!location.hash) location.hash = "#/overview";
      router();
    } catch (e) {
      dom.skeleton.hidden = true;
      dom.appRoot.hidden = true;
      showBanner({ host: dom.bannerHost, message: e?.message ?? String(e), onRetry: load });
    }
  };

  window.addEventListener("hashchange", router, { signal: ac.signal });
  await load();
};

document.addEventListener("DOMContentLoaded", () => {
  void boot();
});

