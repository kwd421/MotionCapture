"""Real VFR decoding with injected model lifecycle failures; no native AI claims."""

import json

import numpy as np
import pytest
from test_recording import tiny_vfr as tiny_vfr

from motioncapture import wholebody_pose_batch_lab as lab
from motioncapture.wholebody_catalog import BenchmarkError


@pytest.mark.parametrize("primary_failure,cleanup_error,status,exit_code", [
    (False, RuntimeError, "failed", 2),
    (False, KeyboardInterrupt, "interrupted", 130),
    (True, RuntimeError, "failed", 2),
])
def test_cleanup_failure_is_terminal_and_all_models_are_closed(
    tiny_vfr, tmp_path, monkeypatch, primary_failure, cleanup_error, status, exit_code,
):
    closed = []

    class Model:
        metadata = {"synthetic_test_model": True}

        def __init__(self, path, shape, provider, **kwargs):
            self.name = "detector" if shape[2] == 416 else f"pose{shape[0]}"

        def run(self, tensor):
            if primary_failure:
                raise BenchmarkError("injected_inference_failure")
            return [np.zeros((1, 3549, 85), np.float32)]

        def close(self):
            closed.append(self.name)
            raise cleanup_error()

    monkeypatch.setattr(lab, "OrtModel", Model)
    monkeypatch.setattr(lab, "BatchOrtModel", Model)
    monkeypatch.setattr(lab, "verify_asset", lambda *a: (tmp_path / "unused.onnx", {}))
    args = lab.parser().parse_args([
        str(tiny_vfr), "--research-only", "--allow-cpu-partitions",
        "--output", str(tmp_path / "report.json"),
    ])
    assert lab.execute(args) == exit_code
    report = json.loads(args.output.read_text())
    assert report["status"] == status
    assert report["cleanup_errors"] == [cleanup_error.__name__] * 3
    assert closed == ["pose2", "pose1", "detector"]
    if primary_failure:
        assert report["error"]["code"] == "injected_inference_failure"
        assert report["error"]["phase"] == "scan_and_compare"
    else:
        assert report["result"]["frames_scanned"] == 6
        assert report["error"]["phase"] == "model_cleanup"
