"""Real pixel/decoder/HTTP-free asset tests; fake sessions are explicitly not ML."""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from motioncapture import wholebody_assets as assets
from motioncapture import wholebody_bench as bench
from motioncapture import wholebody_onnx as infer
from motioncapture.recording import inspect_recording


def simcc(hw=(256, 192), score=1.0):
    h, w = hw
    x, y = np.zeros((1, 133, w*2), np.float32), np.zeros((1, 133, h*2), np.float32)
    x[:, :, w], y[:, :, h] = score, score
    return [x, y]


def test_bgr_normalization_and_input_immutability():
    image = np.full((100, 100, 3), (11, 62, 233), np.uint8)
    saved = image.copy()
    x, center, scale = infer.pose_input(image, np.array([10, 10, 90, 90]), (256, 192))
    expected = (np.array([11, 62, 233])-np.array([123.675, 116.28, 103.53]))
    expected /= [58.395, 57.12, 57.375]
    np.testing.assert_allclose(x[0, :, 128, 96], expected, rtol=1e-6)
    np.testing.assert_array_equal(image, saved)
    assert x.shape == (1, 3, 256, 192) and x.dtype == np.float32
    np.testing.assert_array_equal(center, [50, 50])
    np.testing.assert_allclose(scale[0]/scale[1], 192/256)


@pytest.mark.parametrize("box", [[0, 0, 100, 400], [20, 50, 700, 200], [-20, -30, 10, 50]])
def test_simcc_center_maps_to_bbox_center_after_aspect_crop(box):
    image = np.zeros((480, 720, 3), np.uint8)
    _, center, scale = infer.pose_input(image, np.array(box), (256, 192))
    xy, scores = infer.decode_pose(simcc(), center, scale, (256, 192))
    np.testing.assert_allclose(xy, np.broadcast_to(center, (133, 2)))
    np.testing.assert_array_equal(scores, np.ones(133))


def test_affine_source_pixel_transform_matches_independent_center_scale_formula():
    rng = np.random.default_rng(6)
    image = rng.integers(0, 256, (600, 1000, 3), dtype=np.uint8)
    output, center, scale = infer.pose_input(image, np.array([49, 117, 802, 510]), (256, 192))
    # Algebraic zero-rotation center/scale matrix, not the adapter's 3-point construction.
    sx, sy = 192/scale[0], 256/scale[1]
    matrix = np.array([[sx, 0, 96-sx*center[0]], [0, sy, 128-sy*center[1]]])
    expected = cv2.warpAffine(image, matrix, (192, 256), flags=cv2.INTER_LINEAR)
    expected = (expected.astype(np.float32)-[123.675, 116.28, 103.53])/[58.395, 57.12, 57.375]
    np.testing.assert_allclose(output, infer.tensor(expected), atol=1e-5)


@pytest.mark.parametrize("defect", ["shape", "nan", "missing"])
def test_pose_schema_does_not_guess_or_zero_fill(defect):
    outputs = simcc()
    if defect == "shape":
        outputs[0] = outputs[0][:, :17]
    elif defect == "nan":
        outputs[0][0, 0, 0] = np.nan
    else:
        outputs.pop()
    with pytest.raises(ValueError):
        infer.decode_pose(outputs, np.array([0, 0]), np.array([100, 100]), (256, 192))


def test_empty_detector_is_empty_not_full_frame():
    empty = infer.decode_boxes([np.zeros((1, 10, 5), np.float32)], .5)
    assert empty.shape == (0, 4)
    boxes = infer.decode_boxes([np.array([[[10, 20, 50, 80, .8]]], np.float32)], .5)
    np.testing.assert_array_equal(boxes, [[20, 40, 100, 160]])


def test_detector_raw_one_class_schema_and_nms():
    output = np.zeros((1, 3549, 6), np.float32)
    output[0, 0] = [2, 2, 0, 0, .99, .99]
    boxes = infer.decode_boxes([output], 1)
    np.testing.assert_array_equal(boxes, [[12, 12, 20, 20]])


@pytest.mark.parametrize("defect", ["classes", "labels", "nan", "inverted", "layout"])
def test_unrecognized_detector_output_fails(defect):
    data = np.array([[[10, 20, 50, 80, .8]]], np.float32)
    outputs = [data]
    if defect == "classes":
        outputs = [np.zeros((1, 3549, 85), np.float32)]
    elif defect == "labels":
        outputs.append(np.ones((1, 1), np.int64))
    elif defect == "nan":
        data[0, 0, 0] = np.nan
    elif defect == "inverted":
        data[0, 0, 2] = 0
    else:
        outputs = [data[0]]
    with pytest.raises(ValueError):
        infer.decode_boxes(outputs, 1)


def test_letterbox_is_top_left_bgr_114_without_normalization():
    image = np.full((100, 200, 3), (3, 50, 240), np.uint8)
    result, ratio = infer.detector_input(image)
    assert ratio == 2.08
    np.testing.assert_array_equal(result[0, :, 0, 0], [3, 50, 240])
    np.testing.assert_array_equal(result[0, :, -1, -1], [114, 114, 114])


def fake_ort(providers=None, reported=None, observed=None):
    """Only tests session policy and profiling parsing, never model performance."""
    recorder = SimpleNamespace(options=None, providers=None, runs=0, retries_disabled=False)
    class Options:
        def __init__(self):
            self.entries = {}
        def add_session_config_entry(self, key, value):
            self.entries[key] = value
    class Session:
        def __init__(self, path, sess_options, providers):
            self.options = sess_options
            recorder.options, recorder.providers = sess_options, providers
        def disable_fallback(self):
            recorder.retries_disabled = True
        def get_providers(self):
            return reported or ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        def get_inputs(self):
            return [SimpleNamespace(name="input", type="tensor(float)", shape=[1, 3, 256, 192])]
        def get_outputs(self):
            return [SimpleNamespace(name="x"), SimpleNamespace(name="y")]
        def run(self, names, inputs):
            recorder.runs += 1
            return simcc()
        def end_profiling(self):
            path = Path(self.options.profile_file_prefix + ".json")
            path.write_text(json.dumps([{"cat": "Node", "args": {"provider": name}}
                                        for name in (observed or ["CoreMLExecutionProvider"])]))
            return str(path)
    ort = SimpleNamespace(__version__=infer.ORT_VERSION, SessionOptions=Options,
                          ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0), InferenceSession=Session,
                          get_available_providers=lambda: providers or ["CoreMLExecutionProvider"])
    return ort, recorder


def test_coreml_policy_has_no_ort_cpu_fallback_and_ends_profiling_before_measurement(tmp_path):
    ort, rec = fake_ort()
    session = infer.StrictSession(tmp_path/"explicit-fake-model", "coreml-ane", (256, 192),
                                  ort_module=ort)
    root = Path(session.profile.name)
    session.audit_probe(np.zeros((1, 3, 256, 192), np.float32))
    assert rec.options.entries["session.disable_cpu_ep_fallback"] == "1"
    assert rec.retries_disabled
    assert rec.providers[0][1]["MLComputeUnits"] == "CPUAndNeuralEngine"
    assert session.metadata()["ane_execution_verified"] is False
    assert session.metadata()["coreml_internal_cpu_is_permitted"] is True
    assert list(root.iterdir()) == []
    session.run(np.zeros((1, 3, 256, 192), np.float32))
    assert rec.runs == 2
    session.close()
    session.close()
    assert not root.exists()


@pytest.mark.parametrize("case", ["unavailable", "constructor_retried_cpu", "mixed_placement"])
def test_coreml_never_calls_cpu_success(case, tmp_path):
    ort, _ = fake_ort(
        providers=["CPUExecutionProvider"] if case == "unavailable" else None,
        reported=["CPUExecutionProvider"] if case == "constructor_retried_cpu" else None,
        observed=["CoreMLExecutionProvider", "CPUExecutionProvider"])
    with pytest.raises(infer.UnsupportedProvider):
        session = infer.StrictSession(tmp_path/"fake", "coreml-all", (256, 192), ort_module=ort)
        try:
            session.audit_probe(np.zeros((1, 3, 256, 192), np.float32))
        finally:
            session.close()


def make_archive(tmp_path, name="sdk/end2end.onnx", content=b"not-a-real-model"):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr(name, content)
    data = stream.getvalue()
    asset = replace(assets.ASSETS["dwpose-m"], archive_sha256=hashlib.sha256(data).hexdigest())
    (tmp_path/f"{asset.key}.zip").write_bytes(data)
    return asset


def test_explicit_extract_hashes_and_never_overwrites(tmp_path):
    asset = make_archive(tmp_path)
    info = assets.fetch(asset, tmp_path)
    assert info["onnx_sha256"] == hashlib.sha256(b"not-a-real-model").hexdigest()
    model = tmp_path/f"{asset.key}.onnx"
    model.write_bytes(b"user-owned-change")
    with pytest.raises(ValueError, match="checksum"):
        assets.fetch(asset, tmp_path)
    assert model.read_bytes() == b"user-owned-change"


@pytest.mark.parametrize("name", ["../bad.onnx", "/tmp/bad.onnx", "bad\\model.onnx"])
def test_archive_member_traversal_rejected(tmp_path, name):
    asset = make_archive(tmp_path, name)
    with pytest.raises(ValueError, match="member"):
        assets.fetch(asset, tmp_path)
    assert not (tmp_path/f"{asset.key}.onnx").exists()


def test_archive_checksum_mismatch_and_atomic_limit_preserve_destination(tmp_path):
    make_archive(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        assets.require_asset(assets.ASSETS["dwpose-m"], tmp_path)
    path = tmp_path/"preserve"
    path.write_bytes(b"before")
    with pytest.raises(FileExistsError):
        assets._atomic_stream(io.BytesIO(b"after"), path)
    assert path.read_bytes() == b"before"
    with pytest.raises(ValueError, match="limit"):
        assets._atomic_stream(io.BytesIO(b"abcdef"), tmp_path/"limit", 3)
    assert not (tmp_path/"limit").exists()


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "vfr.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=30:duration=0.2",
        "-vf", "setpts=(N+floor(N/2))/(30*TB)", "-fps_mode", "passthrough",
        "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(path),
    ], check=True, timeout=15)
    return path


def arguments(video, tmp_path):
    return bench.parser().parse_args([str(video), "--models", "dwpose-m", "--providers", "cpu",
                                    "coreml-all", "--max-frames", "0", "--warmup-frames", "1",
                                    "--decode-threads", "1",
                                    "--output-dir", str(tmp_path/"results")])


class FakeEngine:
    """Explicit fixture outputs; invocation counts are the test, not inference FPS."""
    fail_at = None
    fail_close = False
    def __init__(self, *args):
        self.calls = 0
        self.closed = False
    def preflight(self, image):
        return {"fixture": True}
    def process(self, image):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("test-model-failure")
        return (np.array([[0, 0, 64, 48]], np.float32), np.ones((1, 133, 2), np.float32),
                np.ones((1, 133), np.float32), {"pose_infer_ms": .25, "detector_infer_ms": .5})
    def close(self):
        self.closed = True
        if self.fail_close:
            raise RuntimeError("test-close-failure")


def asset_loader(asset, root):
    return root/f"fixture-{asset.key}", {"fixture": True}


def test_real_vfr_runner_full_and_exact_prefix_no_extra_inference(video, tmp_path):
    probe, args = inspect_recording(video), arguments(video, tmp_path)
    made = []
    def factory(*args):
        made.append(FakeEngine())
        return made[-1]
    for count in (2, len(probe.pts)):
        args.max_frames = count
        row, bank = bench.execute_candidate(args, probe, "dwpose-m", "cpu",
                                             engine_factory=factory, asset_loader=asset_loader)
        assert row["status"] == "completed" and row["completed_frames"] == count
        assert row["last_completed"]["pts"] == probe.pts[count-1]
        assert row["all_frames"]["person_instances"] == count
        assert bank.complete and tuple(bank.pts) == probe.pts[:count]
        assert made[-1].calls == count and made[-1].closed
        assert row["capabilities"]["world_xyz"] == "unsupported"
        assert row["cleanup"] == {"decoder": "complete", "engine": "complete"}


def test_failure_retains_prefix_and_primary_error_despite_cleanup_failure(video, tmp_path):
    class Failing(FakeEngine):
        fail_at, fail_close = 3, True
    args, probe = arguments(video, tmp_path), inspect_recording(video)
    row, bank = bench.execute_candidate(args, probe, "dwpose-m", "cpu",
                                         engine_factory=Failing, asset_loader=asset_loader)
    assert bank is None and row["status"] == "failed"
    assert row["completed_frames"] == 2 and row["unpaced_loop_fps"] is None
    assert row["error"]["phase"] == "inference" and row["cleanup"]["engine"] == "failed"
    assert row["current_frame"]["sequence"] == 2 and row["last_completed"]["sequence"] == 1
    assert row["prediction_hash_scope"] == "completed_prefix"
    assert row["cleanup_errors"] == ["RuntimeError"]
    assert str(video) not in json.dumps(row)


def test_reference_differences_are_source_pixels_no_missing_zero_error(video, tmp_path):
    args, probe = arguments(video, tmp_path), inspect_recording(video)
    _, bank = bench.execute_candidate(args, probe, "dwpose-m", "cpu",
                                       engine_factory=FakeEngine, asset_loader=asset_loader)
    class Shifted(FakeEngine):
        def process(self, image):
            boxes, xy, scores, stages = super().process(image)
            xy[:, :, 0] += 2
            scores[:, 112:] = 0
            return boxes, xy, scores, stages
    row, _ = bench.execute_candidate(args, probe, "dwpose-m", "coreml-all", bank,
                                      engine_factory=Shifted, asset_loader=asset_loader)
    comparison = row["cpu_comparison"]
    assert comparison["accuracy_verified"] is False
    assert comparison["parts"]["left_hand"]["frame_mean_px"]["mean"] == 2
    right = comparison["parts"]["right_hand"]
    assert right["frame_mean_px"]["mean"] is None
    assert right["reference_only_points"] == len(probe.pts)*21


def test_matrix_reverse_order_cell_failure_persisted_and_no_clobber(video, tmp_path):
    args = arguments(video, tmp_path)
    args.repeats = 2
    observed = []
    def cell(args, probe, model, mode, reference):
        observed.append((model, mode))
        if mode == "coreml-all":
            return {"status": "failed", "candidate": model, "pose_mode": mode,
                    "completed_frames": 0, "unpaced_loop_fps": None}, None
        return bench.execute_candidate(args, probe, model, mode, reference,
                                         engine_factory=FakeEngine, asset_loader=asset_loader)
    assert bench.run_matrix(args, candidate=cell) == 2
    assert [mode for _, mode in observed] == ["cpu", "coreml-all", "coreml-all", "cpu"]
    state = json.loads((args.output_dir/"summary.json").read_text())
    assert state["status"] == "completed_with_failures" and len(state["runs"]) == 4
    assert len(list(args.output_dir.glob("run-*.started.json"))) == 4
    assert json.loads((args.output_dir/"run-03.json").read_text())["status"] == "failed"
    last = json.loads((args.output_dir/"run-04.json").read_text())
    assert last["cpu_comparison"]["status"] == "observed"
    with pytest.raises(FileExistsError):
        bench.run_matrix(args, candidate=cell)


def test_source_change_stops_plan_and_does_not_become_success(video, tmp_path):
    args = arguments(video, tmp_path)
    def cell(*args):
        return {"status": "failed", "fatal_source_change": True, "error": {}}, None
    assert bench.run_matrix(args, candidate=cell) == 2
    summary = json.loads((args.output_dir/"summary.json").read_text())
    assert len(summary["runs"]) == 1 and summary["status"] != "completed"


def test_wholebody_adapter_no_detection_does_not_call_pose_and_multiple_calls_count(tmp_path):
    sessions = []
    class Session:
        def __init__(self, path, mode, hw, threads):
            self.kind = "detector" if hw == (416, 416) else "pose"
            self.runs = 0
            self.n = 0
            self.closed = False
            sessions.append(self)
        def audit_probe(self, x):
            return self.run(x)
        def metadata(self):
            return {"fixture": True}
        def run(self, x):
            self.runs += 1
            return ([np.array([[[1, 1, 20, 30, .9]]]*self.n, np.float32).reshape(1, self.n, 5)]
                    if self.kind == "detector" else simcc())
        def close(self):
            self.closed = True
    engine = infer.WholebodyONNX(tmp_path/"det", tmp_path/"pose", "cpu", session_factory=Session)
    engine.preflight(np.zeros((100, 100, 3), np.uint8))
    assert sessions[1].runs == 1  # labelled placement probe only
    boxes, xy, score, times = engine.process(np.zeros((100, 100, 3), np.uint8))
    assert xy.shape == (0, 133, 2) and sessions[1].runs == 1 and times["pose_infer_ms"] == 0
    sessions[0].n = 2
    boxes, xy, score, _ = engine.process(np.zeros((100, 100, 3), np.uint8))
    assert xy.shape == (2, 133, 2) and sessions[1].runs == 3
    engine.close()
    assert all(session.closed for session in sessions)


def test_constructor_failure_cleanup_stays_unknown_and_no_fake_fps(video, tmp_path):
    args, probe = arguments(video, tmp_path), inspect_recording(video)
    def fails(*args):
        raise infer.UnsupportedProvider("provider unavailable")
    row, bank = bench.execute_candidate(args, probe, "dwpose-m", "coreml-all",
                                         engine_factory=fails, asset_loader=asset_loader)
    assert row["completed_frames"] == 0 and bank is None
    assert row["cleanup"]["engine"] == "unknown_after_open_failure"
    assert row["error"]["code"] == "requested_provider_or_placement_unavailable"
    assert row["unpaced_loop_fps"] is None


def test_interrupt_stops_matrix_but_preserves_terminal_report(video, tmp_path):
    args = arguments(video, tmp_path)
    def cell(*args):
        return {"status": "interrupted", "completed_frames": 0}, None
    assert bench.run_matrix(args, candidate=cell) == 2
    summary = json.loads((args.output_dir/"summary.json").read_text())
    assert summary["status"] == "interrupted" and len(summary["runs"]) == 1


def test_native_controls_are_labelled_non_equivalent_and_do_not_replace_candidates(
        video, tmp_path, monkeypatch):
    args = arguments(video, tmp_path)
    args.include_mediapipe = True
    seen = []
    def native(args, probe, **kwargs):
        seen.append(args.preview)
        return {"status": "completed", "all_frames": {"frames": len(probe.pts)},
                "unpaced_loop_fps": 1.0}
    def cell(args, probe, model, mode, reference):
        return bench.execute_candidate(args, probe, model, mode, reference,
                                         engine_factory=FakeEngine, asset_loader=asset_loader)
    monkeypatch.setattr(bench, "run_pass", native)
    assert bench.run_matrix(args, candidate=cell) == 0
    first = json.loads((args.output_dir/"run-01.json").read_text())
    assert first["capability_equivalent_to_wholebody133"] is False
    assert seen == ["none", "none"]
    summary = json.loads((args.output_dir/"summary.json").read_text())
    assert [r["candidate"] for r in summary["runs"]] == [
        "mediapipe", "dwpose-m", "dwpose-m", "mediapipe"]
