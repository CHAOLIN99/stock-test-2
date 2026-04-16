/* Data loading + validation + one-time preprocessing. */

const defaultDataUrl = "./results/charts_data.json";

const resolveDataUrl = () => {
  const q = getQueryParam("data");
  return q ? String(q) : defaultDataUrl;
};

const validateSchema = (root) => {
  assert(root && typeof root === "object", "charts_data.json: root must be an object");
  assert(Array.isArray(root.tickers), "charts_data.json: tickers must be an array");
  if (root.generated_at != null) {
    assert(typeof root.generated_at === "string", "charts_data.json: generated_at must be an ISO string");
  }
  assert(root.data && typeof root.data === "object", "charts_data.json: data must be an object");

  for (const t of root.tickers) {
    assert(typeof t === "string", "charts_data.json: tickers must be strings");
    const td = root.data[t];
    assert(td && typeof td === "object", `missing data for ticker ${t}`);
    assert(Array.isArray(td.dates), `${t}: dates must be an array`);
    assert(Array.isArray(td.actual), `${t}: actual must be an array`);
    assert(td.predictions && typeof td.predictions === "object", `${t}: predictions must be an object`);
    assert(td.metrics && typeof td.metrics === "object", `${t}: metrics must be an object`);

    const n = td.dates.length;
    assert(td.actual.length === n, `${t}: actual length must match dates`);

    for (const [m, arr] of Object.entries(td.predictions)) {
      assert(Array.isArray(arr), `${t}: predictions.${m} must be an array`);
      assert(arr.length === n, `${t}: predictions.${m} length mismatch`);
    }
    if (td.regime != null) {
      assert(Array.isArray(td.regime), `${t}: regime must be an array`);
      assert(td.regime.length === n, `${t}: regime length mismatch`);
    }
    if (td.regime_names != null) {
      assert(Array.isArray(td.regime_names), `${t}: regime_names must be an array`);
    }
    if (td.feature_importance != null) {
      assert(typeof td.feature_importance === "object", `${t}: feature_importance must be an object`);
    }
  }
};

/** One-time normalization: coerce numbers, pre-compute equity curves, build indices. */
const preprocessData = (root) => {
  const out = {
    tickers: root.tickers.slice(),
    generated_at: root.generated_at ?? null,
    data: {},
    _meta: {
      tickers: root.tickers.slice(),
      generatedAtTs: parseISODate(root.generated_at),
      allModels: [],
      perTicker: {},
    },
  };

  const globalModels = new Set();

  for (const t of root.tickers) {
    const src = root.data[t];
    const dates = src.dates.slice();
    const actual = toNumericArray(src.actual);

    const predictions = {};
    for (const [m, arr] of Object.entries(src.predictions || {})) {
      predictions[m] = toNumericArray(arr);
      globalModels.add(m);
    }

    // Cumulative equity curves pre-computed once (rendered many times).
    const equity = {};
    const logRets = src.strategy_log_returns || {};
    for (const [m, arr] of Object.entries(logRets)) {
      equity[m] = cumulativeFromLogReturns(arr);
    }

    const dateIndex = Object.create(null);
    for (let i = 0; i < dates.length; i++) dateIndex[dates[i]] = i;

    out.data[t] = {
      dates,
      actual,
      predictions,
      equity, // derived
      strategy_log_returns: logRets,
      metrics: src.metrics || {},
      regime: Array.isArray(src.regime) ? src.regime.slice() : null,
      regime_names: Array.isArray(src.regime_names) ? src.regime_names.slice() : null,
      feature_importance: src.feature_importance || {},
      _dateIndex: dateIndex,
      _minDate: dates[0] ?? null,
      _maxDate: dates[dates.length - 1] ?? null,
    };

    const models = Object.keys(predictions);
    out._meta.perTicker[t] = {
      models,
      bestModel: pickBestModel(src.metrics),
    };
  }

  out._meta.allModels = Array.from(globalModels).sort();
  return out;
};

const loadChartsData = async ({ signal } = {}) => {
  const dataUrl = resolveDataUrl();
  let res;
  try {
    res = await fetch(`${dataUrl}?_=${Date.now()}`, { cache: "no-store", signal });
  } catch (e) {
    if (e?.name === "AbortError") throw e;
    throw new Error(`Data load error: ${e?.message ?? String(e)}`);
  }
  if (!res.ok) {
    if (res.status === 404) {
      throw new Error(
        `Dashboard data not found at ${dataUrl} (HTTP 404). ` +
          `Run "python run_experiment.py" to generate results/charts_data.json, then reload.`
      );
    }
    throw new Error(`Failed to fetch ${dataUrl}: HTTP ${res.status}`);
  }
  const json = await res.json();
  validateSchema(json);
  return preprocessData(json);
};
