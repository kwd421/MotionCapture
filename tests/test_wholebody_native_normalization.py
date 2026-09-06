"""Real OpenCV tensors/VFR clocks. Session fixtures are not neural performance."""
from __future__ import annotations

import json
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from motioncapture import wholebody_fast_input as fast
from motioncapture import wholebody_optimize_bench as bench
from motioncapture.recording import inspect_recording
from motioncapture.wholebody_cadence import OutputCadence
from motioncapture.wholebody_catalog import BenchmarkError
from motioncapture.wholebody_onnx import pose_tensor
from motioncapture.wholebody_stages import StagePipeline
from test_recording import tiny_vfr  # noqa: F401
from test_wholebody_handoff import Session, frames


@pytest.mark.parametrize("shape", [(1, 256), (256, 192), (288, 384)])
def test_native_lookup_matches_all_uint8_values_and_reference(shape):
    a = np.arange(np.prod(shape)*3, dtype=np.int64).astype(np.uint8).reshape(*shape, 3)
    a = a[:, ::-1]  # noncontiguous input is valid; source must remain unchanged
    before = a.tobytes()
    expected = ((a-fast.MEAN)/fast.STD).transpose(2, 0, 1)[None].astype(np.float32)
    out = fast.normalize(a, kernel="opencv")
    assert out.dtype == np.float32 and out.flags.c_contiguous
    assert out.tobytes() == expected.tobytes() == fast.normalize(a).tobytes()
    assert a.tobytes() == before and not np.shares_memory(out, a)
    saved = out.tobytes()
    fast.normalize(np.zeros_like(a), kernel="opencv")
    assert out.tobytes() == saved and not fast.TABLE.flags.writeable


@pytest.mark.parametrize("box", [[.01, -.001, 99.1234, 90.87], [-100, 80, 1900, 1000]])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_original_affine_and_tensor_bytes_are_preserved(box, dtype):
    image = np.random.default_rng(7).integers(0, 256, (1080, 1920, 3), np.uint8)
    before = fast.array_hash(image)
    box = np.array(box, dtype=dtype)
    for rect in [box, np.nextafter(box, np.full(4, np.inf, dtype=dtype))]:
        reference = pose_tensor(image, rect, (192, 256))
        actual = fast.fast_pose_tensor(image, rect, (192, 256), kernel="opencv")
        assert all(a.dtype == b.dtype and a.tobytes() == b.tobytes()
                   for a, b in zip(reference, actual, strict=True))
        fast.check_recipe(image, rect, (192, 256), kernel="opencv")
    assert fast.array_hash(image) == before


def test_selected_kernel_errors_are_not_fallbacks(monkeypatch):
    crop = np.zeros((3, 4, 3), np.uint8)
    with pytest.raises(BenchmarkError, match="unknown_normalization"):
        fast.normalize(crop, kernel="invalid")
    with pytest.raises(BenchmarkError, match="uint8"):
        fast.normalize(crop.astype(np.float32), kernel="opencv")
    with pytest.raises(BenchmarkError, match="nonempty"):
        fast.normalize(crop[:0], kernel="opencv")
    def fail(*args):
        raise RuntimeError("injected_opencv_error")
    monkeypatch.setattr(fast.cv2, "LUT", fail)
    with pytest.raises(RuntimeError, match="injected_opencv"):
        fast.normalize(crop, kernel="opencv")


def test_native_recipe_check_rejects_drift(monkeypatch):
    def bad(*args):
        values = list(pose_tensor(*args))
        values[0] += .1
        return values
    monkeypatch.setattr(fast, "pose_tensor", bad)
    with pytest.raises(BenchmarkError, match="recipe_mismatch"):
        fast.check_recipe(np.zeros((30, 30, 3), np.uint8), np.array([0., 0., 30., 30.]),
                          (192, 256), kernel="opencv")


def identity(i, pts):
    return SimpleNamespace(source_id="test", sequence=i, pts=pts, time_base=Fraction(1, 1000))


def test_cadence_uses_pts_bins_and_measured_host_span_not_nominal_fps():
    cadence = OutputCadence(5, 100)
    for i, (pts, ns, people) in enumerate([
        (100, 1_000_000, 1), (9900, 11_000_000, 1),
        (10100, 31_000_000, 2), (10200, 61_000_000, 1), (11100, 101_000_000, 1),
    ]):
        cadence.add(identity(i, pts), people, ns)
    result = cadence.summary()
    assert result["coverage"] == "selected_frames_observed"
    assert result["verified_intervals"]["samples"] == 4
    assert result["verified_intervals"]["mean_ms"] == 25
    assert result["verified_intervals"]["over_budget"] == 3
    assert result["longest_consecutive_over_60hz_interval_budget"] == 3
    left, right = result["source_windows"]
    assert left["frames"] == 2 and left["within_window_verified_rate_hz"] == 100
    assert left["closed_by_later_source_window"] is True
    assert right["frames"] == 3 and right["within_window_verified_rate_hz"] == pytest.approx(2/.07)
    assert right["within_window_intervals"]["samples"] == 2  # boundary pair NOT inside bin
    assert right["closed_by_later_source_window"] is False
    assert result["intervals_by_ending_person_count"]["2"]["samples"] == 1
    assert result["worst_intervals"][0]["sequence"] == 4
    assert result["live_60fps_verified"] is False


def test_cadence_empty_single_partial_and_bounded_worst():
    cadence = OutputCadence(25, 0)
    assert cadence.summary()["verified_intervals"]["mean_ms"] is None
    cadence.add(identity(0, 0), 0, 0)
    s = cadence.summary()
    assert s["source_windows"][0]["within_window_verified_rate_hz"] is None
    assert s["verified_intervals"]["samples"] == 0 and s["coverage"] == "partial"
    for i in range(1, 21):
        cadence.add(identity(i, i*17), 1, i*20_000_000)
    s = cadence.summary()
    assert len(s["worst_intervals"]) == 16
    assert s["longest_consecutive_over_60hz_interval_budget"] == 20
    assert s["coverage"] == "partial"
    assert s["interval_exceedances_are_dropped_frames"] is False


def test_bad_cadence_observation_never_mutates_completed_prefix():
    c = OutputCadence(2, 0)
    c.add(identity(0, 0), 1, 10)
    before = c.summary()
    for ident, people, ns in [(identity(2, 1), 1, 20), (identity(1, 0), 1, 20),
                              (identity(1, 1), 1, 10), (identity(1, 1), 9, 20)]:
        with pytest.raises(BenchmarkError):
            c.add(ident, people, ns)
        assert c.summary() == before


def test_native_pipeline_exact_n_no_mutation_and_owner_close():
    instances = []
    def factory(kind):
        s = Session(kind); instances.append(s); return s
    incoming = list(frames(4)); source_hashes = [fast.array_hash(f.image_bgr) for f in incoming]
    with StagePipeline(lambda: factory("detector"), lambda: factory("pose"),
                       normalization_kernel="opencv") as pipeline:
        output = list(pipeline.packets(incoming, 3, overlap=True, fast=True, advance_pose=True))
    assert len(output) == 3 and all(s.calls == 3 for s in instances)
    assert pipeline.snapshot()["read_frames"] == pipeline.snapshot()["emitted_frames"] == 3
    assert pipeline.snapshot()["normalization_kernel"] == "opencv"
    assert [fast.array_hash(f.image_bgr) for f in incoming] == source_hashes
    assert all(s.closed_on == s.created_on for s in instances)


def test_real_vfr_full_abba_hash_and_cadence_counts(tiny_vfr, tmp_path, monkeypatch):
    args = bench.parser().parse_args([str(tiny_vfr), "--research-only", "--max-frames", "0",
                                     "--suite", "native-normalize", "--output", str(tmp_path/"x.json")])
    monkeypatch.setattr(bench, "verify_asset", lambda root, key: (tmp_path/key, {"fixture": True}))
    seen = []
    def invoke(args, probe, mode, factories, reference):
        seen.append(mode)
        return bench.run_pass(args, probe, mode,
                              (lambda: Session("detector"), lambda: Session("pose")), reference)
    assert bench.execute(args, pass_runner=invoke) == 0
    result = json.loads(args.output.read_text())
    assert seen == ["overlap-ready-lut", "overlap-ready-cvlut",
                    "overlap-ready-cvlut", "overlap-ready-lut"]
    assert result["all_pass_prediction_hashes_equal"] is True
    assert result["all_pass_pixel_hashes_equal"] is True
    assert result["all_pass_detector_hashes_equal"] is True
    n = len(inspect_recording(tiny_vfr).pts)
    for row in result["runs"]:
        assert row["scope"] == "full_file" and row["all_frames"]["frames"] == n
        assert row["output_cadence"]["frames"] == n
        assert row["output_cadence"]["verified_intervals"]["samples"] == n-1
        assert row["all_frames"]["stages"]["cadence_observer_ms"]["samples"] == n
        assert row["live_60fps_verified"] is False
    assert result["runs"][1]["provider_disagreement"]["reference"] == (
        "same-provider first completed arm; NOT truth")
    with pytest.raises(FileExistsError):
        bench.execute(args, pass_runner=invoke)


def test_existing_handoff_plan_is_unchanged(tiny_vfr, tmp_path, monkeypatch):
    args = bench.parser().parse_args([str(tiny_vfr), "--research-only", "--suite", "handoff",
                                     "--output", str(tmp_path/"x")])
    monkeypatch.setattr(bench, "verify_asset", lambda root, key: (tmp_path/key, {}))
    def invoke(args, probe, mode, factories, reference):
        return bench.run_pass(args, probe, mode,
                              (lambda: Session("detector"), lambda: Session("pose")), reference)
    assert bench.execute(args, pass_runner=invoke) == 0
    d = json.loads(args.output.read_text())
    assert d["configuration"]["plan"] == ["overlap-lut", "overlap-ready-lut",
                                             "overlap-ready-lut", "overlap-lut"]
    assert all(r["normalization_kernel"] == "numpy" for r in d["runs"])
