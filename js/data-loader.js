/* Data loading + validation + one-time preprocessing. */

const defaultDataUrl = "./results/charts_data.json";

const resolveDataUrl = () => {
  // Allow swapping datasets without code changes:
  // - `dashboard.html?data=./results/charts_data.json`
  // - `dashboard.html?data=./results/charts_data_alt.json`
  const q = getQueryParam("data");
  return q ? String(q) : defaultDataUrl;
};

const validateSchema = (root) => {
  assert(root && typeof root === "object", "charts_data.json: root must be an object");
  assert(Array.isArray(root.tickers), "charts_data.json: tickers must be an array");
  if (root.generated_at != null) {
    assert(typeof root.generated_at === "string", "charts_data.json: generated_at must be an ISO string when present");
  }
  assert(root.data && typeof root.data === "object", "charts_data.json: data must be an object");

  for (const t of root.tickers) {
    assert(typeof t === "string", "charts_data.json: tickers must be strings");
    const td = root.data[t];
    assert(td && typeof td === "object", `charts_data.json: missing data for ticker ${t}`);
    assert(Array.isArray(td.dates), `${t}: dates must be an array`);
    assert(Array.isArray(td.actual), `${t}: actual must be an array`);
    assert(td.predictions && typeof td.predictions === "object", `${t}: predictions must be an object`);
    assert(td.metrics && typeof td.metrics === "object", `${t}: metrics must be an object`);
    if (td.regime != null) assert(Array.isArray(td.regime), `${t}: regime must be an array when present`);
    if (td.regime_names != null) assert(Array.isArray(td.regime_names), `${t}: regime_names must be an array when present`);
    if (td.feature_importance != null) assert(typeof td.feature_importance === "object", `${t}: feature_importance must be an object when present`);

    const n = td.dates.length;
    assert(td.actual.length === n, `${t}: actual length must match dates length`);
    for (const d of td.dates) assert(typeof d === "string", `${t}: dates must be ISO yyyy-mm-dd strings`);

    for (const [m, arr] of Object.entries(td.predictions)) {
      assert(Array.isArray(arr), `${t}: predictions.${m} must be an array`);
      assert(arr.length === n, `${t}: predictions.${m} length must match dates length`);
    }
    if (td.regime) assert(td.regime.length === n, `${t}: regime length must match dates length`);
  }
};

const preprocessData = (root) => {
  const out = structuredClone(root);

  out._meta = {
    tickers: out.tickers.slice(),
    generatedAtTs: parseISODate(out.generated_at),
    allModels: [],
    perTicker: {},
  };

  const globalModels = new Set();
  for (const t of out.tickers) {
    const td = out.data[t];
    const ts = td.dates.map(parseISODate);
    out.data[t]._ts = ts;
    out.data[t]._dateIndex = Object.freeze(Object.fromEntries(td.dates.map((d, i) => [d, i])));

    const models = Object.keys(td.predictions || {});
    models.forEach((m) => globalModels.add(m));
    out._meta.perTicker[t] = {
      models,
      bestModel: pickBestModel(td.metrics),
    };
  }
  out._meta.allModels = Object.freeze(Array.from(globalModels).sort());
  return out;
};

const loadChartsData = async ({ signal } = {}) => {
  try {
    const dataUrl = resolveDataUrl();
    const url = `${dataUrl}?_=${Date.now()}`;
    const res = await fetch(url, { cache: "no-store", signal });
    if (!res.ok) {
      if (res.status === 404) {
        throw new Error(
          `Dashboard data not found at ${dataUrl} (HTTP 404). ` +
            `Run "python run_experiment.py" to generate results/charts_data.json, then reload this page.`
        );
      }
      throw new Error(`Failed to fetch ${dataUrl}: HTTP ${res.status}`);
    }
    const json = await res.json();
    validateSchema(json);
    return preprocessData(json);
  } catch (e) {
    const msg = e?.message ?? String(e);
    throw new Error(`Data load error: ${msg}`);
  }
};

