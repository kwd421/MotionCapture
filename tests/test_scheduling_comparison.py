"""Actual file decoding plus explicitly synthetic predictions, not native AI."""
from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from fractions import Fraction
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from test_recording import tiny_vfr as tiny_vfr

from motioncapture import recording_bench as bench
from motioncapture.contracts import Blendshape, Landmark, LandmarkResult, LandmarkTimings
from motioncapture.recording import RecordedFrame, RecordedIdentity


def observation(x=0.25, visibility=None):
    points = (Landmark(x, 0.5, 0.1, visibility=visibility),) * 33
    return LandmarkResult(points, points, (), (), (), (), (), (Blendshape("jawOpen", 0.3, 1),))


def digest(result, *, pts=100, stream="s"):
    frame = RecordedFrame(RecordedIdentity("test", stream, 0, pts, Fraction(1, 1000)),
                          np.zeros((8, 8, 3), np.uint8), 0)
    value = hashlib.sha256()
    bench._digest_result(value, frame, result)
    return value.hexdigest()


def test_prediction_digest_includes_values_optional_scores_and_original_time():
    base = observation()
    assert digest(base) == digest(base, stream="fresh")
    assert digest(base) != digest(observation(0.251))
    assert digest(base) != digest(observation(visibility=0.0))
    assert digest(base) != digest(base, pts=101)
    assert digest(base) != digest(replace(base, face_blendshapes=(Blendshape("jawOpen", 0.4, 1),)))
    assert digest(base) != digest(replace(base, face_blendshapes=(Blendshape("jawOpen", 0.3),)))


def test_source_fingerprint_separates_code_change_from_unrelated_file(tmp_path):
    root = tmp_path / "src/motioncapture"
    root.mkdir(parents=True)
    code = root / "x.py"
    code.write_text("a = 1\n")
    first = bench._python_sources_digest(tmp_path)
    (tmp_path / "untracked-video.txt").write_text("not code")
    assert bench._python_sources_digest(tmp_path) == first
    code.write_text("a = 2\n")
    assert bench._python_sources_digest(tmp_path) != first


def test_abba_is_four_fresh_all_frame_passes_and_verifies_outputs(tiny_vfr, tmp_path, monkeypatch):
    import json

    owners = []
    assets = ModuleType("motioncapture.model_assets")
    assets.MODEL_ASSETS = ()  # Test only. No fake model exists on a production path.
    monkeypatch.setitem(sys.modules, "motioncapture.model_assets", assets)
    monkeypatch.setattr(bench, "_installed", lambda _: "0.10.31")

    class Tracker:
        def __init__(self, scheduling):
            self.scheduling = scheduling
            self.pts = []
            self.closed = False
            owners.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.closed = True

        def process_recorded(self, frame):
            self.pts.append(frame.identity.pts)
            return SimpleNamespace(identity=frame.identity, result=observation(),
                                   timings=LandmarkTimings(0, 1, 2, 1, 2, 0, 2))

    monkeypatch.setattr(bench, "_make_tracker", lambda _, schedule: Tracker(schedule))
    out = tmp_path / "comparison.json"
    monkeypatch.setattr(sys, "argv", ["bench", str(tiny_vfr), "--output", str(out),
                                    "--preview", "none", "--compare-scheduling",
                                    "--verify-results"])
    assert bench.main() == 0
    report = json.loads(out.read_text())
    plan = ["parallel", "staggered", "staggered", "parallel"]
    assert [t.scheduling for t in owners] == plan
    assert len({id(t) for t in owners}) == 4 and all(t.closed for t in owners)
    assert all(t.pts == owners[0].pts for t in owners)
    assert len(owners[0].pts) == 6
    assert report["configuration"]["repeats"] == 4
    assert report["configuration"]["scheduling_plan"] == plan
    assert [r["task_scheduling"] for r in report["runs"]] == plan
    assert report["prediction_equivalence"]["all_passes_equal"] is True
    assert all(r["all_frames"]["frames"] == 6 for r in report["runs"])
    assert all(r["result_hash_overhead_in_loop_fps"] for r in report["runs"])
    assert all(r["all_frames"]["stages"]["result_verification_ms"]["samples"] == 6
               for r in report["runs"])
    assert str(tiny_vfr) not in out.read_text()
    assert "jawOpen" not in out.read_text()


@pytest.mark.parametrize("extra", [
    ["--mode", "decode", "--compare-scheduling"],
    ["--compare-scheduling", "--repeats", "2"],
    ["--compare-scheduling", "--task-scheduling", "staggered"],
    ["--verify-results", "--mode", "decode"],
])
def test_invalid_comparison_never_opens_input(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["bench", "missing.mp4", "--output",
                                    str(tmp_path / "unused.json"), *extra])
    with pytest.raises(ValueError):
        bench.main()
    assert not (tmp_path / "unused.json").exists()
