const status = document.getElementById("status"), details = document.getElementById("details");
const token = location.hash.slice(1);
history.replaceState(null, "", location.pathname);
if (!token) {
  status.textContent = "연결 토큰이 없습니다. 터미널에 표시된 주소를 여세요.";
} else {
  const client = crypto.randomUUID();
  const worker = new Worker("/app/worker.mjs");
  const visibility = () => fetch("/rpc/visibility", {method: "POST", cache: "no-store",
    headers: {"X-Mocap-Token": token, "X-Mocap-Client": client, "Content-Type": "application/json"},
    body: JSON.stringify({visible: document.visibilityState === "visible"})}).catch(() => {});
  worker.onmessage = ({data}) => {
    status.textContent = data.status;
    if (data.status === "connected") visibility();
    if (data.renderer || data.error) details.textContent = data.renderer || data.error;
  };
  worker.onerror = () => { status.textContent = "Worker 오류 — 터미널 결과를 확인하세요."; };
  document.addEventListener("visibilitychange", visibility);
  worker.postMessage({token, client});
}
