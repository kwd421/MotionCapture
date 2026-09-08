"""Real decoder/renderer; scripted models and UI failure, no native performance claims."""
import hashlib
import json
import sys
import threading
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
from test_recording import tiny_vfr as tiny_vfr
from test_wholebody_optimization import Model

from motioncapture import wholebody_optimize_bench as bench
from motioncapture import wholebody_recorded_preview as app
from motioncapture.recording import inspect_recording
from motioncapture.wholebody_onnx import Person2D
from motioncapture.wholebody_recorded_preview import compose
from motioncapture.wholebody_stages import StagePipeline


@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt])
def test_real_replay_consumer_preserves_frames_and_cleanup(tiny_vfr, tmp_path, failure):
    args = bench.parser().parse_args([str(tiny_vfr), "--research-only", "--max-frames", "0",
                                     "--output", str(tmp_path / "report.json")])
    probe = inspect_recording(tiny_vfr)
    models = [Model("detector"), Model("pose")]
    seen = []

    def consume(packet):
        frame = packet.detected.frame
        before = hashlib.sha256(frame.image_bgr).digest()
        compose(packet, ["Recorded test"])
        assert hashlib.sha256(frame.image_bgr).digest() == before
        seen.append(frame.identity.pts)
        if failure is not None and len(seen) == 2:
            raise failure("injected consumer exit")

    row, _ = bench.run_pass(args, probe, "source-pts-ready-cvlut",
                            (lambda: models[0], lambda: models[1]), packet_consumer=consume)
    expected = "completed" if failure is None else (
        "interrupted" if failure is KeyboardInterrupt else "failed")
    assert row["status"] == expected
    assert all(m.closed for m in models)
    assert row["decoder_cleanup"] == "owner_released"
    assert seen == list(probe.pts[:len(seen)])
    assert row["all_frames"]["frames"] == len(seen)
    if failure is None:
        assert len(seen) == len(probe.pts)
    else:
        assert len(seen) == 2 and row["error"]["phase"] == "packet_consumer"
        assert row["paced_loop_fps"] is None


def test_render_ignores_invalid_and_out_of_frame_points_without_mutation():
    source = np.zeros((240, 320, 3), np.uint8)
    xy = np.full((133, 2), 80., np.float32)
    xy[1] = 1e20
    valid = np.zeros(133, dtype=bool)
    valid[1] = True
    person = Person2D(xy, np.ones(133, np.float32), valid)
    packet = SimpleNamespace(people=[person],
                             detected=SimpleNamespace(frame=SimpleNamespace(image_bgr=source)))
    image = compose(packet, [])
    packet.people = []
    assert np.array_equal(image, compose(packet, []))
    assert not source.any() and person.xy[1, 0] == np.float32(1e20)


@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt])
def test_bounded_ui_handoff_keeps_main_thread_order_and_joins(tiny_vfr, tmp_path, failure):
    args = bench.parser().parse_args([str(tiny_vfr), "--research-only", "--max-frames", "0",
                                     "--output", str(tmp_path / "report.json")])
    probe = inspect_recording(tiny_vfr)
    models = [Model("detector"), Model("pose")]
    main_thread = threading.get_ident()
    seen = []

    class View:
        def __call__(self, packet):
            assert threading.get_ident() == main_thread
            seen.append(packet.detected.frame.identity.sequence)
            if failure is not None:
                raise failure("display exit")

        def poll(self):
            assert threading.get_ident() == main_thread

    def runner(*a, **kw):
        assert threading.get_ident() != main_thread
        kw["pipeline_factory"] = StagePipeline
        return bench.run_pass(*a, **kw)

    row, _ = app.run_with_preview(args, probe, (lambda: models[0], lambda: models[1]),
                                  View(), runner=runner)
    assert all(m.closed for m in models)
    assert row["preview_handoff"]["inference_owner_joined"]
    assert row["preview_handoff"]["frames_replaced"] == 0
    if failure is None:
        assert seen == list(range(len(probe.pts))) and row["status"] == "completed"
        assert row["preview_handoff"]["undisplayed_queued_packets_at_exit"] == 0
    else:
        assert seen == [0]
        assert row["status"] == ("interrupted" if failure is KeyboardInterrupt else "failed")
        assert row["paced_loop_fps"] is None


@pytest.mark.parametrize("close_fails", [False, True])
def test_app_reports_once_and_preserves_ui_cleanup_failure(tiny_vfr, tmp_path,
                                                         monkeypatch, close_fails):
    args = SimpleNamespace(input=tiny_vfr, output=tmp_path / "preview.json",
                           asset_dir=tmp_path, max_frames=0, decode_threads=0,
                           provider="cpu", allow_cpu_partitions=False, snapshot_frame=[1])
    monkeypatch.setattr(app, "verify_asset", lambda directory, key: (directory / key, {}))
    monkeypatch.setattr(app, "OrtModel", lambda *a, **kw: Model("detector"))
    monkeypatch.setattr(app, "PoseBatchModels", lambda *a, **kw: Model("pose"))
    def runner(*a, **kw):
        kw["pipeline_factory"] = StagePipeline
        return bench.run_pass(*a, **kw)
    monkeypatch.setattr(app, "run_with_preview", partial(app.run_with_preview, runner=runner))
    for name in ("namedWindow", "imshow"):
        monkeypatch.setattr(app.cv2, name, lambda *a: None)
    monkeypatch.setattr(app.cv2, "waitKey", lambda *a: -1)
    monkeypatch.setattr(app.cv2, "getWindowProperty", lambda *a: 1)
    def close(*args):
        if close_fails:
            raise RuntimeError("window cleanup failed")
    monkeypatch.setattr(app.cv2, "destroyWindow", close)
    assert app.execute(args) == (2 if close_fails else 0)
    report = json.loads(args.output.read_text())
    assert report["status"] == ("failed" if close_fails else "completed")
    assert report["preview"]["submitted_frames"] == 6
    assert report["preview"]["snapshots_written"] == [1]
    assert (tmp_path / "preview.snapshots" / "frame-000001.png").is_file()
    before = args.output.read_bytes()
    with pytest.raises(FileExistsError):
        app.execute(args)
    assert args.output.read_bytes() == before


def test_requested_runtime_mismatch_fails_before_source_or_model_access(tmp_path, monkeypatch):
    args = SimpleNamespace(input=tmp_path / "unused.mp4", output=tmp_path / "mismatch.json",
                           snapshot_frame=[], expected_ort_version="1.29.0")
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(__version__="1.22.1"))
    monkeypatch.setattr(app, "inspect_recording",
                        lambda *a: pytest.fail("must reject before reading the source"))
    assert app.execute(args) == 2
    report = json.loads(args.output.read_text())
    assert report["status"] == "failed"
    assert report["error"]["code"] == "requested_onnxruntime_version_mismatch"
    assert report["runtime"]["onnxruntime_version"] == "1.22.1"
    assert "run" not in report and "preview" not in report
