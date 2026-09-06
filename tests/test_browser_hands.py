"""Real loopback HTTP with explicit synthetic peers; NOT GPU/model benchmarks."""
from __future__ import annotations

import concurrent.futures
import json
import struct
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest

from motioncapture.browser_hands import (BrowserHandError, BrowserHandLab,
                                        BrowserHandTask, parse_hand_result)


def request(lab, path, data=None, *, token=None, client="fixture-client", origin=None):
    headers = {"X-Mocap-Token": lab.token if token is None else token,
               "X-Mocap-Client": client}
    if origin:
        headers["Origin"] = origin
    raw = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(lab.origin + path, data=raw, headers=headers)
    with urllib.request.urlopen(req, timeout=3) as response:
        return response.status, response.read()


def next_message(lab):
    _, raw = request(lab, "/rpc/next")
    size = struct.unpack("<I", raw[:4])[0]
    return json.loads(raw[4:4+size]), raw[4+size:]


def answer(lab, message, result):
    request(lab, "/rpc/result", {"id": message["id"], "result": result})


def empty_result():
    return {"landmarks": [], "worldLandmarks": [], "handedness": []}


def test_actual_http_transports_exact_pixels_and_one_request():
    with BrowserHandLab({"/": b"test"}) as lab:
        request(lab, "/rpc/hello", {})
        with concurrent.futures.ThreadPoolExecutor(1) as executor:
            future = executor.submit(lab.call, "detect", {"timestamp_ms": 16}, b"\x00\x17\xff")
            message, pixels = next_message(lab)
            assert pixels == b"\x00\x17\xff" and message["timestamp_ms"] == 16
            with pytest.raises(BrowserHandError, match="in_flight"):
                lab.call("detect")
            answer(lab, message, {"ok": True})
            assert future.result(timeout=2) == {"ok": True}
            assert lab.pending is None
    assert not lab.thread.is_alive()


def test_rejects_origin_token_unknown_paths_and_second_client():
    with BrowserHandLab({"/": b"test"}) as lab:
        for kwargs in ({"token": "wrong"}, {"origin": "https://evil.invalid"}):
            with pytest.raises(urllib.error.HTTPError) as err:
                request(lab, "/rpc/hello", {}, **kwargs)
            assert err.value.code == 403
        assert lab.client is None
        request(lab, "/rpc/hello", {})
        with pytest.raises(urllib.error.HTTPError) as err:
            request(lab, "/rpc/hello", {}, client="different-client")
        assert err.value.code == 409
        for path in ("/../../recording60.json", "/sdk/../../../models/private", "/rpc/unknown"):
            with pytest.raises(urllib.error.HTTPError) as err:
                request(lab, path)
            assert err.value.code == 404


def test_wrong_result_id_is_terminal_not_stale_success():
    with BrowserHandLab({}) as lab:
        request(lab, "/rpc/hello", {})
        with concurrent.futures.ThreadPoolExecutor(1) as executor:
            future = executor.submit(lab.call, "open")
            message, _ = next_message(lab)
            with pytest.raises(urllib.error.HTTPError):
                answer(lab, {"id": message["id"] + 1}, {})
            with pytest.raises(BrowserHandError, match="unexpected_or_duplicate"):
                future.result(timeout=2)
            assert lab.pending is None


def test_peer_error_does_not_initialize_cpu_fallback():
    with BrowserHandLab({}) as lab:
        request(lab, "/rpc/hello", {})
        with concurrent.futures.ThreadPoolExecutor(1) as executor:
            future = executor.submit(lab.call, "open", {"delegate": "GPU"})
            message, _ = next_message(lab)
            assert message["delegate"] == "GPU"
            request(lab, "/rpc/result", {"id": message["id"], "error": {"code": "failed"}})
            with pytest.raises(BrowserHandError, match="browser_task_failed"):
                future.result(timeout=2)
            with pytest.raises(BrowserHandError):
                lab.call("open", {"delegate": "CPU"})


def test_timeout_is_explicit_and_clears_owned_request():
    with BrowserHandLab({}, timeout=.03) as lab:
        request(lab, "/rpc/hello", {})
        with pytest.raises(BrowserHandError, match="timeout"):
            lab.call("open")
        assert lab.pending is None


class SyntheticLab:
    hidden_events = 0
    visible = True
    failure = None

    def __init__(self, renderer="Apple test GPU"):
        self.renderer = renderer
        self.calls = []

    def fail(self, code):
        self.failure = code

    def call(self, op, metadata=None, pixels=b""):
        self.calls.append((op, metadata, pixels))
        if op == "open":
            return {"delegate": metadata["delegate"], "sdk": "0.10.32", "renderer": self.renderer,
                    "hardware_context_checked": True}
        if op == "close":
            return {"closed": True}
        return {"timestamp_ms": metadata["timestamp_ms"], "detect_ms": 1, "pixels_ms": .1,
                "hands": empty_result()}


def test_task_retains_timestamp_pixels_and_lifecycle():
    lab = SyntheticLab()
    task = BrowserHandTask(lab, "GPU")
    pixels = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    saved = pixels.copy()
    output = task.detect_for_video(SimpleNamespace(numpy_view=lambda: pixels), 33)
    assert output.hand_landmarks == []
    assert lab.calls[-1][1] == {"timestamp_ms": 33, "width": 4, "height": 3}
    assert lab.calls[-1][2] == pixels.tobytes()
    np.testing.assert_array_equal(pixels, saved)
    with pytest.raises(BrowserHandError, match="timestamp"):
        task.detect_for_video(SimpleNamespace(numpy_view=lambda: pixels), 33)
    task.close()
    task.close()
    assert sum(c[0] == "close" for c in lab.calls) == 1
    assert task.summary(16.67)["model_cleanup_acknowledged"]


@pytest.mark.parametrize("renderer", [None, "", "SwiftShader", "llvmpipe", "Software Rasterizer"])
def test_gpu_rejects_unverified_or_software_renderer(renderer):
    with pytest.raises(BrowserHandError, match="software_gpu"):
        BrowserHandTask(SyntheticLab(renderer), "GPU")


def one_hand():
    p = {"x": .1, "y": .2, "z": -.3}
    return {"landmarks": [[dict(p) for _ in range(21)]],
            "worldLandmarks": [[dict(p) for _ in range(21)]],
            "handedness": [[{"categoryName": "Left", "score": .9}]]}


def test_native_shape_copy_does_not_invent_confidence():
    result = parse_hand_result(one_hand())
    assert len(result.hand_landmarks[0]) == 21
    assert result.hand_landmarks[0][0].presence is None
    assert result.handedness[0][0].category_name == "Left"


@pytest.mark.parametrize("change", ["nan", "count", "topology", "label", "confidence"])
def test_invalid_remote_data_is_error_not_empty(change):
    data = one_hand()
    if change == "nan":
        data["landmarks"][0][0]["x"] = float("nan")
    elif change == "count":
        data["handedness"] = []
    elif change == "topology":
        data["landmarks"][0].pop()
    elif change == "label":
        data["handedness"][0][0]["categoryName"] = "Unknown"
    else:
        data["landmarks"][0][0]["presence"] = -1
    with pytest.raises(BrowserHandError):
        parse_hand_result(data)
