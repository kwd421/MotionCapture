// Classic worker: pinned MediaPipe 0.10.32 loads its WASM glue with importScripts.

let active = false;
self.onmessage = async ({data}) => {
  if (active || typeof data.token !== "string") return;
  active = true;
  const headers = {"X-Mocap-Token": data.token, "X-Mocap-Client": data.client};
  const post = async (path, body) => {
    const r = await fetch(path, {method: "POST", headers: {...headers, "Content-Type": "application/json"},
      body: JSON.stringify(body), cache: "no-store"});
    if (!r.ok) throw Error("protocol_post_failed");
  };
  let task = null, rgba = null, canvas = null, gl = null, lastTimestamp = -1;
  let HandLandmarker, FilesetResolver, vision;
  try {
    const {unpackMessage, rgbToRgba, checkGpuRenderer} = await import("./protocol.mjs");
    await post("/rpc/hello", {});
    self.postMessage({status: "connected"});
    while (true) {
      const response = await fetch("/rpc/next", {headers, cache: "no-store"});
      if (response.status === 204) continue;
      if (response.status === 410) break;
      if (!response.ok) throw Error("protocol_read_failed");
      const {meta, pixels} = unpackMessage(await response.arrayBuffer());
      try {
        let result;
        if (meta.op === "open") {
          if (task || !["CPU", "GPU"].includes(meta.delegate)) throw Error("invalid_open");
          if (!HandLandmarker) {
            ({HandLandmarker, FilesetResolver} = await import("/sdk/vision_bundle.mjs"));
            vision = await FilesetResolver.forVisionTasks("/sdk/wasm");
          }
          canvas = new OffscreenCanvas(1, 1);
          let renderer = null, vendor = null;
          if (meta.delegate === "GPU") {
            gl = canvas.getContext("webgl2", {failIfMajorPerformanceCaveat: true});
            if (!gl) throw Error("webgl2_unavailable");
            const info = gl.getExtension("WEBGL_debug_renderer_info");
            if (!info) throw Error("gpu_renderer_unavailable");
            renderer = checkGpuRenderer(gl.getParameter(info.UNMASKED_RENDERER_WEBGL));
            vendor = gl.getParameter(info.UNMASKED_VENDOR_WEBGL);
          }
          const begin = performance.now();
          task = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {modelAssetPath: "/assets/hand.task", delegate: meta.delegate},
            canvas, runningMode: "VIDEO", numHands: 2,
            minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
          });
          lastTimestamp = -1;
          result = {delegate: meta.delegate, renderer, vendor,
            sdk: "0.10.32", user_agent: navigator.userAgent,
            init_ms: performance.now() - begin, hardware_context_checked: meta.delegate === "GPU",
            per_operator_dispatch_verified: false};
          self.postMessage({status: "ready", delegate: meta.delegate, renderer});
        } else if (meta.op === "detect") {
          if (!task || meta.timestamp_ms <= lastTimestamp) throw Error("invalid_detect_state");
          lastTimestamp = meta.timestamp_ms;
          const begin = performance.now();
          rgba = rgbToRgba(pixels, meta.width, meta.height, rgba);
          const image = new ImageData(rgba, meta.width, meta.height);
          const prepared = performance.now();
          const hands = task.detectForVideo(image, meta.timestamp_ms);
          const finished = performance.now();
          result = {timestamp_ms: meta.timestamp_ms, pixels_ms: prepared - begin,
            detect_ms: finished - prepared,
            hands: {landmarks: hands.landmarks, worldLandmarks: hands.worldLandmarks,
                    handedness: hands.handedness}};
          // All GPU/CPU model results have returned before this buffer is reused.
        } else {
          if (!task) throw Error("invalid_close_state");
          task.close(); task = null; rgba = null;
          if (gl) gl.getExtension("WEBGL_lose_context")?.loseContext();
          gl = null; canvas = null;
          result = {closed: true};
          self.postMessage({status: "pass_closed"});
        }
        await post("/rpc/result", {id: meta.id, result});
      } catch (error) {
        self.postMessage({status: "failed", error: String(error.message)});
        const known = new Set(["webgl2_unavailable", "gpu_renderer_unavailable",
          "unverified_or_software_gpu", "invalid_open", "invalid_detect_state", "invalid_close_state"]);
        const code = known.has(error.message) ? error.message : "browser_task_failed";
        await post("/rpc/result", {id: meta.id, error: {code}});
        break; // Never retry a GPU operation on CPU.
      }
    }
    self.postMessage({status: "stopped"});
  } catch (error) {
    self.postMessage({status: "failed", error: String(error.message)});
  } finally {
    try { task?.close(); } catch { /* Python has already failed; not a success. */ }
    if (gl) gl.getExtension("WEBGL_lose_context")?.loseContext();
    task = rgba = canvas = gl = null;
  }
};
