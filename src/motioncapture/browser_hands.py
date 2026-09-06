"""Explicit loopback hand-task adapter; no CDN, camera access or CPU fallback."""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import mimetypes
import re
import secrets
import struct
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import numpy as np

SDK_VERSION = "0.10.32"
MAX_RESPONSE = 256 * 1024
MAX_PIXELS = 4096 * 4096
PROGRESS_PHASES = frozenset({"waiting", "received", "opening", "pixels", "detect",
                             "result_post", "result_acked", "closing", "stopped"})
FAULT_CODES = frozenset({"browser_worker_error", "browser_message_error", "browser_page_closed",
                         "browser_unhandled_rejection", "browser_fetch_failed",
                         "browser_task_failed", "browser_gpu_context_lost",
                         "protocol_post_failed", "protocol_read_failed"})


class BrowserHandError(RuntimeError):
    """Terminal bridge, browser or model error; never a missing detection."""


def _number(value, *, confidence=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise BrowserHandError("invalid_numeric_result")
    if confidence and not 0 <= value <= 1:
        raise BrowserHandError("invalid_confidence")
    return float(value)


def parse_hand_result(result: dict) -> SimpleNamespace:
    """Validate the web schema before the existing subject-relative adapter."""
    if not isinstance(result, dict):
        raise BrowserHandError("invalid_hand_result")
    groups = [result.get(k) for k in ("landmarks", "worldLandmarks", "handedness")]
    if (not all(isinstance(g, list) for g in groups)
            or not len(groups[0]) == len(groups[1]) == len(groups[2]) <= 2):
        raise BrowserHandError("invalid_hand_count")

    def points(group):
        out = []
        for hand in group:
            if not isinstance(hand, list) or len(hand) != 21:
                raise BrowserHandError("invalid_hand_topology")
            converted = []
            for point in hand:
                if not isinstance(point, dict):
                    raise BrowserHandError("invalid_hand_point")
                values = {k: _number(point.get(k)) for k in ("x", "y", "z")}
                for k in ("visibility", "presence"):
                    values[k] = (_number(point[k], confidence=True)
                                 if point.get(k) is not None else None)
                converted.append(SimpleNamespace(**values))
            out.append(converted)
        return out

    handedness = []
    for hand in groups[2]:
        if not isinstance(hand, list) or len(hand) != 1 or not isinstance(hand[0], dict):
            raise BrowserHandError("invalid_handedness")
        category = hand[0]
        if category.get("categoryName") not in {"Left", "Right"}:
            raise BrowserHandError("invalid_handedness_label")
        handedness.append([SimpleNamespace(
            category_name=category["categoryName"],
            score=_number(category.get("score"), confidence=True),
        )])
    return SimpleNamespace(hand_landmarks=points(groups[0]),
                           hand_world_landmarks=points(groups[1]), handedness=handedness)


def collect_assets(sdk_dir: Path, model: Path, web_dir: Path) -> tuple[dict, dict]:
    """Only known SDK/web files are served; never expose a repo directory."""
    sdk = sdk_dir.resolve()
    metadata = json.loads((sdk / "package.json").read_text())
    if metadata.get("name") != "@mediapipe/tasks-vision" or metadata.get("version") != SDK_VERSION:
        raise BrowserHandError("wrong_web_sdk_version")
    assets = {}
    for name in ("vision_bundle.mjs", "wasm/vision_wasm_internal.js",
                 "wasm/vision_wasm_internal.wasm", "wasm/vision_wasm_nosimd_internal.js",
                 "wasm/vision_wasm_nosimd_internal.wasm"):
        path = (sdk / name).resolve()
        if not path.is_relative_to(sdk) or not path.is_file():
            raise BrowserHandError("missing_web_sdk_asset")
        assets["/sdk/" + name] = path.read_bytes()
    for name in ("index.html", "page.mjs", "worker.mjs", "protocol.mjs", "style.css"):
        assets["/app/" + name] = (web_dir / name).read_bytes()
    assets["/assets/hand.task"] = model.read_bytes()
    assets["/"] = assets["/app/index.html"]
    manifest = {"sdk_version": SDK_VERSION, "sha256": {
        path: hashlib.sha256(data).hexdigest() for path, data in sorted(assets.items())
        if path != "/"
    }}
    return assets, manifest


@dataclass
class _Request:
    identifier: int
    operation: str
    metadata: dict
    pixels: bytes
    delivered: bool = False
    response: dict | None = None
    created_ns: int = field(default_factory=time.monotonic_ns)
    sent_ns: int | None = None


class BrowserHandLab:
    """Own one server, one browser client and at most one in-flight request."""

    def __init__(self, assets: dict[str, bytes], *, timeout=60.0):
        if not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError("Invalid browser timeout")
        self.assets = dict(assets)
        self.timeout = timeout
        self.token = secrets.token_urlsafe(32)
        self.condition = threading.Condition()
        self.pending: _Request | None = None
        self.client: str | None = None
        self.closed = False
        self.failure: str | None = None
        self.failure_snapshot: dict | None = None
        self.last_completed_request: dict | None = None
        self.browser_progress: dict | None = None
        self.progress_received_ns: int | None = None
        self.progress_sequence = -1
        self.heartbeat_count = 0
        self.next_id = 0
        self.hidden_events = 0
        self.visible = True
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.server.daemon_threads = True
        self.origin = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": .1}, daemon=True)

    @property
    def url(self):
        return self.origin + "/#" + self.token

    def snapshot(self):
        """Only protocol metadata; never pixels, token, URL or model outputs."""
        with self.condition:
            now = time.monotonic_ns()
            request = self.pending
            pending = None if request is None else {
                "id": request.identifier, "operation": request.operation,
                "timestamp_ms": request.metadata.get("timestamp_ms"),
                "age_ms": (now - request.created_ns) / 1e6,
                "claimed_by_browser_fetch": request.delivered,
                "http_write_completed": request.sent_ns is not None,
                "result_received": request.response is not None,
            }
            return {"schema_version": 1, "failure": self.failure,
                    "request_timeout_s": self.timeout, "pending": pending,
                    "last_completed_request": self.last_completed_request,
                    "browser_progress": self.browser_progress,
                    "heartbeat_count": self.heartbeat_count,
                    "heartbeat_age_ms": ((now - self.progress_received_ns) / 1e6
                                         if self.progress_received_ns is not None else None),
                    "tab_visible": self.visible, "tab_hidden_events": self.hidden_events,
                    "stage_is_sampled_not_a_causal_trace": True}

    def fail(self, code: str):
        with self.condition:
            if self.failure is None:
                self.failure = code
                self.failure_snapshot = self.snapshot()
            self.condition.notify_all()

    def _handler(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            disable_nagle_algorithm = True  # Single-frame request/response, no batching.

            def log_message(self, *args):
                pass  # Requests/tokens and private media never enter logs.

            def setup(self):
                super().setup()
                self.connection.settimeout(15)

            def reply(self, status, data=b"", content_type="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Content-Security-Policy", "default-src 'none'; "
                                 "script-src 'self' 'wasm-unsafe-eval'; "
                                 "worker-src 'self' blob:; connect-src 'self'; "
                                 "style-src 'self'; frame-ancestors 'none'")
                self.end_headers()
                if data:
                    self.wfile.write(data)

            def authorized(self, rpc=False):
                if self.headers.get("Host") != owner.origin.removeprefix("http://"):
                    self.reply(403)
                    return False
                origin = self.headers.get("Origin")
                if origin is not None and origin != owner.origin:
                    self.reply(403)
                    return False
                if rpc and not hmac.compare_digest(
                    self.headers.get("X-Mocap-Token", ""), owner.token
                ):
                    self.reply(403)
                    return False
                return True

            def do_GET(self):
                path = urlsplit(self.path).path
                if not self.authorized(rpc=path == "/rpc/next"):
                    return
                try:
                    if path == "/rpc/next":
                        client = self.headers.get("X-Mocap-Client")
                        with owner.condition:
                            if client is None or client != owner.client:
                                self.reply(403)
                                return
                            owner.condition.wait_for(
                                lambda: owner.closed or owner.failure is not None
                                or (owner.pending is not None and not owner.pending.delivered), 5,
                            )
                            if owner.closed or owner.failure is not None:
                                self.reply(410)
                                return
                            request = owner.pending
                            if request is None or request.delivered:
                                self.reply(204)
                                return
                            request.delivered = True
                            metadata = json.dumps(
                                {"id": request.identifier, "op": request.operation,
                                 **request.metadata}, allow_nan=False,
                            ).encode()
                            data = struct.pack("<I", len(metadata)) + metadata + request.pixels
                        self.reply(200, data, "application/octet-stream")
                        with owner.condition:
                            request.sent_ns = time.monotonic_ns()
                    elif path in owner.assets:
                        mime = "application/wasm" if path.endswith(".wasm") else (
                            "text/javascript" if path.endswith((".mjs", ".js")) else
                            mimetypes.guess_type(path)[0] or "application/octet-stream")
                        if path == "/":
                            mime = "text/html"
                        self.reply(200, owner.assets[path], mime)
                    else:
                        self.reply(404)
                except (BrokenPipeError, ConnectionError, TimeoutError):
                    if path == "/rpc/next":
                        owner.fail("browser_disconnected")

            def do_POST(self):
                if not self.authorized(rpc=True):
                    self.close_connection = True
                    return
                path = urlsplit(self.path).path
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= MAX_RESPONSE or self.headers.get("Transfer-Encoding"):
                        self.reply(413)
                        self.close_connection = True
                        return
                    raw = self.rfile.read(size)
                    if len(raw) != size:
                        raise ValueError("Truncated response")
                    data = json.loads(
                        raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
                    )
                    if not isinstance(data, dict):
                        raise ValueError("Object required")
                    client = self.headers.get("X-Mocap-Client", "")
                    with owner.condition:
                        if path == "/rpc/hello":
                            if (not 8 <= len(client) <= 100 or owner.closed
                                    or (owner.client is not None and owner.client != client)):
                                self.reply(409)
                                return
                            owner.client = client
                            owner.condition.notify_all()
                        elif client != owner.client or not client:
                            self.reply(403)
                            return
                        elif path == "/rpc/visibility":
                            if type(data.get("visible")) is not bool:
                                raise ValueError("Invalid visibility")
                            owner.visible = data["visible"]
                            owner.hidden_events += int(not owner.visible)
                        elif path == "/rpc/progress":
                            seq = data.get("sequence")
                            item = data.get("worker")
                            if (type(seq) is not int or seq < 0 or not isinstance(item, dict)
                                    or item.get("phase") not in PROGRESS_PHASES
                                    or (item.get("id") is not None
                                        and (type(item["id"]) is not int or item["id"] <= 0))
                                    or type(data.get("visible")) is not bool):
                                raise ValueError("Invalid progress")
                            age = _number(data.get("phase_age_ms"))
                            if not 0 <= age <= 86400000:
                                raise ValueError("Invalid progress age")
                            # Stale overlapping heartbeats cannot replace a newer observation.
                            if seq > owner.progress_sequence:
                                owner.progress_sequence = seq
                                owner.heartbeat_count += 1
                                owner.progress_received_ns = time.monotonic_ns()
                                owner.browser_progress = {"id": item.get("id"),
                                    "phase": item["phase"], "phase_age_ms": age}
                                if owner.visible and not data["visible"]:
                                    owner.hidden_events += 1
                                owner.visible = data["visible"]
                        elif path == "/rpc/fault":
                            code = data.get("code")
                            if code not in FAULT_CODES:
                                raise ValueError("Invalid fault code")
                            owner.fail(code)
                        elif path == "/rpc/result":
                            request = owner.pending
                            if (request is None or not request.delivered
                                    or request.response is not None
                                    or type(data.get("id")) is not int
                                    or data["id"] != request.identifier):
                                owner.fail("unexpected_or_duplicate_response")
                                self.reply(409)
                                return
                            request.response = data
                            owner.condition.notify_all()
                        else:
                            self.reply(404)
                            return
                    self.reply(200, b"{}")
                except (ValueError, TypeError, json.JSONDecodeError, TimeoutError, BrowserHandError):
                    owner.fail("invalid_browser_protocol")
                    self.reply(400)
                    self.close_connection = True
                except (BrokenPipeError, ConnectionError):
                    owner.fail("browser_disconnected")

        return Handler

    def __enter__(self):
        self.thread.start()
        return self

    def wait_ready(self):
        with self.condition:
            if not self.condition.wait_for(
                lambda: self.client is not None or self.failure or self.closed, self.timeout
            ):
                self.fail("browser_connect_timeout")
            self._check()

    def _check(self):
        if self.failure is not None:
            raise BrowserHandError(self.failure)
        if self.closed:
            raise BrowserHandError("browser_lab_closed")

    def call(self, operation, metadata=None, pixels=b""):
        if operation not in {"open", "detect", "close"}:
            raise ValueError("Unknown browser operation")
        with self.condition:
            self._check()
            if self.client is None or self.pending is not None:
                raise BrowserHandError("browser_not_ready_or_request_in_flight")
            self.next_id += 1
            request = _Request(self.next_id, operation, metadata or {}, pixels)
            self.pending = request
            self.condition.notify_all()
            try:
                done = self.condition.wait_for(
                    lambda: request.response is not None or self.failure or self.closed,
                    self.timeout,
                )
                if not done:
                    self.fail("browser_request_timeout")
                self._check()
                response = request.response
                if "error" in response:
                    known = {"webgl2_unavailable", "gpu_renderer_unavailable",
                             "unverified_or_software_gpu", "invalid_open",
                             "invalid_detect_state", "invalid_close_state", "browser_gpu_context_lost"}
                    detail = response["error"]
                    code = detail.get("code") if isinstance(detail, dict) else None
                    code = code if code in known else "browser_task_failed"
                    self.fail(code)
                    raise BrowserHandError(code)
                if not isinstance(response.get("result"), dict):
                    self.fail("invalid_browser_result")
                    raise BrowserHandError("invalid_browser_result")
                self.last_completed_request = {
                    "id": request.identifier, "operation": request.operation,
                    "timestamp_ms": request.metadata.get("timestamp_ms"),
                }
                return response["result"]
            finally:
                self.pending = None
                self.condition.notify_all()

    def __exit__(self, *_):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        if self.thread.is_alive():
            raise BrowserHandError("server_cleanup_incomplete")


class BrowserHandTask:
    """Drop-in task boundary, not a replacement MediaPipe module."""

    def __init__(self, lab: BrowserHandLab, delegate: str):
        if delegate not in {"CPU", "GPU"}:
            raise ValueError("Explicit CPU or GPU delegate required")
        self.lab, self.delegate = lab, delegate
        self.state = "active"
        self.last_timestamp = -1
        self.timings: list[tuple[float, float, float]] = []
        metadata = lab.call("open", {"delegate": delegate})
        self.metadata = {key: metadata.get(key) for key in (
            "delegate", "renderer", "vendor", "sdk", "user_agent", "init_ms",
            "hardware_context_checked", "per_operator_dispatch_verified")}
        if self.metadata["sdk"] != SDK_VERSION:
            lab.fail("wrong_browser_sdk")
            raise BrowserHandError("wrong_browser_sdk")
        if delegate == "GPU":
            renderer = self.metadata["renderer"]
            if (not isinstance(renderer, str) or not renderer.strip()
                    or self.metadata["hardware_context_checked"] is not True
                    or re.search(
                        "swiftshader|llvmpipe|softpipe|lavapipe|software|unknown", renderer, re.I,
                    )
                    or renderer.strip().lower() in {"webkit webgl", "webgl"}):
                lab.fail("unverified_or_software_gpu")
                raise BrowserHandError("unverified_or_software_gpu")
        if self.metadata.get("delegate") != delegate:
            lab.fail("wrong_delegate")
            raise BrowserHandError("wrong_delegate")
        self.hidden_at_open = lab.hidden_events

    @property
    def provider_name(self):
        return f"MediaPipe Web {SDK_VERSION}; {self.delegate} requested"

    def detect_for_video(self, image, timestamp_ms):
        if (self.state != "active" or type(timestamp_ms) is not int
                or timestamp_ms <= self.last_timestamp):
            raise BrowserHandError("invalid_hand_lifecycle_or_timestamp")
        rgb = image.numpy_view()
        if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
                or not 0 < rgb.shape[0] * rgb.shape[1] <= MAX_PIXELS):
            raise BrowserHandError("invalid_rgb_input")
        self.last_timestamp = timestamp_ms
        begin = time.perf_counter_ns()
        answer = self.lab.call("detect", {"timestamp_ms": timestamp_ms,
            "width": rgb.shape[1], "height": rgb.shape[0]}, rgb.tobytes(order="C"))
        rpc_ms = (time.perf_counter_ns() - begin) / 1e6
        if type(answer.get("timestamp_ms")) is not int or answer["timestamp_ms"] != timestamp_ms:
            raise BrowserHandError("wrong_result_timestamp")
        detect_ms, pixels_ms = (_number(answer.get(k)) for k in ("detect_ms", "pixels_ms"))
        if detect_ms < 0 or pixels_ms < 0:
            raise BrowserHandError("negative_timing")
        result = parse_hand_result(answer.get("hands"))
        self.timings.append((rpc_ms, detect_ms, pixels_ms))
        return result

    def close(self):
        if self.state == "closed":
            return
        self.state = "closing"
        try:
            if self.lab.call("close").get("closed") is not True:
                self.lab.fail("invalid_close_acknowledgement")
                raise BrowserHandError("invalid_close_acknowledgement")
        except BaseException:
            self.state = "failed"
            raise
        self.state = "closed"

    def summary(self, budget_ms):
        from motioncapture.recording_bench import Samples
        stages = {}
        for column, name in enumerate(
            ("rpc_roundtrip_ms", "browser_detect_ms", "browser_pixels_ms")
        ):
            samples = Samples()
            for row in self.timings:
                samples.add(row[column])
            stages[name] = samples.summary(budget_ms)
        return {"metadata": self.metadata, "stages": stages,
                "tab_hidden_events_in_pass": self.lab.hidden_events - self.hidden_at_open,
                "tab_visible_at_end": self.lab.visible,
                "per_operator_acceleration_verified": False,
                "ane_verified": False, "model_cleanup_acknowledged": self.state == "closed"}
