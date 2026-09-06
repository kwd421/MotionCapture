const status = document.getElementById("status"), details = document.getElementById("details");
const token = location.hash.slice(1);
history.replaceState(null, "", location.pathname);
if (!token) {
  status.textContent = "연결 토큰이 없습니다. 터미널에 표시된 주소를 여세요.";
} else {
  const client = crypto.randomUUID();
  const worker = new Worker("/app/worker.mjs");
  const headers = {"X-Mocap-Token": token, "X-Mocap-Client": client,
    "Content-Type": "application/json"};
  let connected = false, finished = false, heartbeat = null, sending = false, sequence = 0;
  let phase = {id: null, phase: "waiting"}, phaseAt = performance.now();
  const post = (path, body, keepalive = false) => fetch(path, {
    method: "POST", cache: "no-store", headers, keepalive, body: JSON.stringify(body),
  });
  // Fault strings are fixed codes; no exception text, URL, token or pixels leave the page.
  const fault = code => {
    if (finished) return;
    finished = true;
    clearInterval(heartbeat);
    status.textContent = "failed";
    details.textContent = code;
    if (connected) post("/rpc/fault", {code}, true).catch(() => {});
    // No automatic restart: retain the failing run and require an explicit new run.
    worker.terminate();
  };
  const progress = () => {
    if (!connected || finished || sending) return;
    sending = true;
    post("/rpc/progress", {sequence: sequence++, worker: phase,
      phase_age_ms: performance.now() - phaseAt,
      visible: document.visibilityState === "visible"})
      .then(r => { if (!r.ok) throw Error("progress_failed"); })
      .catch(() => fault("browser_fetch_failed"))
      .finally(() => { sending = false; });
  };
  const visibility = () => {
    if (connected && !finished) post("/rpc/visibility", {
      visible: document.visibilityState === "visible",
    }).catch(() => fault("browser_fetch_failed"));
  };
  worker.onmessage = ({data}) => {
    if (finished) return;
    if (data.phase) {
      phase = {id: data.id, phase: data.phase};
      phaseAt = performance.now();
      return;
    }
    status.textContent = data.status;
    if (data.status === "connected") {
      connected = true;
      visibility();
      progress();
      heartbeat = setInterval(progress, 1000);
    }
    if (data.status === "failed") fault(data.code || "browser_task_failed");
    if (data.renderer) details.textContent = data.renderer;
    if (data.status === "stopped") {
      finished = true;
      clearInterval(heartbeat);
    }
  };
  worker.onerror = () => fault("browser_worker_error");
  worker.onmessageerror = () => fault("browser_message_error");
  window.addEventListener("unhandledrejection", () => fault("browser_unhandled_rejection"));
  window.addEventListener("pagehide", () => fault("browser_page_closed"));
  document.addEventListener("visibilitychange", visibility);
  worker.postMessage({token, client});
}
