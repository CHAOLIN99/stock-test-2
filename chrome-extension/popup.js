const style = document.createElement("style");
style.textContent = SPRender.CSS;
document.head.append(style);

const DEFAULT_SERVER = "http://127.0.0.1:8080";

async function start() {
  const { lastTicker } = await chrome.storage.local.get("lastTicker");
  SPRender.mount(document.getElementById("app"), {
    initialTicker: lastTicker,
    onTickerChange: (t) => chrome.storage.local.set({ lastTicker: t }),
  });

  const input = document.getElementById("server");
  const status = document.getElementById("status");
  const { serverUrl } = await chrome.storage.sync.get({ serverUrl: DEFAULT_SERVER });
  input.value = serverUrl;

  document.getElementById("save").addEventListener("click", async () => {
    await chrome.storage.sync.set({ serverUrl: input.value.trim() || DEFAULT_SERVER });
    status.textContent = "Testing…";
    const res = await chrome.runtime.sendMessage({ type: "health" });
    status.textContent = res && res.ok ? "Connected ✓" : `Failed: ${res ? res.error : "no response"}`;
  });
}

start();
