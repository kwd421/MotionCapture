from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from motioncapture.wholebody_catalog import BenchmarkError
from motioncapture.wholebody_pose_batch import PoseBatchModels
from motioncapture.wholebody_pose_batch_pipeline_lab import pose_batched
from motioncapture.wholebody_stages import Detected


class RoutedModel:
    def __init__(self, shape, provider="coreml-all", threads=4):
        self.shape = tuple(shape)
        self.calls = []
        self.closed = False
        self.metadata = {
            "requested": provider,
            "input_shape": self.shape,
            "intra_op_threads": threads,
        }

    def run(self, tensor):
        self.calls.append(tuple(tensor.shape))
        n, _, h, w = tensor.shape
        x = np.zeros((n, 133, w * 2), np.float32)
        y = np.zeros((n, 133, h * 2), np.float32)
        for index in range(n):
            x[index, :, 20 + index] = 1
            y[index, :, 30 + index] = 1
        return [x, y]

    def close(self):
        self.closed = True


def test_pose_batch_models_route_only_requested_batch():
    created = []

    def factory(_path, shape, provider, *, allow_cpu, threads):
        model = RoutedModel(shape, provider, threads)
        created.append(model)
        return model

    models = PoseBatchModels(Path("unused.onnx"), (1, 3, 256, 192), "coreml-all",
                             allow_cpu=True, threads=4,
                             single_factory=factory, pair_factory=factory)
    one = np.zeros((1, 3, 256, 192), np.float32)
    two = np.zeros((2, 3, 256, 192), np.float32)
    assert [value.shape for value in models.run(one)] == [(1, 133, 384), (1, 133, 512)]
    assert [value.shape for value in models.run(two)] == [(2, 133, 384), (2, 133, 512)]
    assert created[0].calls == [(1, 3, 256, 192)]
    assert created[1].calls == [(2, 3, 256, 192)]
    assert models.metadata["same_frame_batching"]["future_frame_batching"] is False
    models.close()
    assert all(model.closed for model in created)


def test_pose_batch_models_do_not_retry_pair_as_single():
    class FailingPair(RoutedModel):
        def run(self, tensor):
            raise RuntimeError("pair failed")

    def single(_path, shape, provider, *, allow_cpu, threads):
        return RoutedModel(shape, provider, threads)

    def pair(_path, shape, provider, *, allow_cpu, threads):
        return FailingPair(shape, provider, threads)

    models = PoseBatchModels(Path("unused.onnx"), (1, 3, 256, 192), "coreml-all",
                             allow_cpu=True, threads=4,
                             single_factory=single, pair_factory=pair)
    with pytest.raises(RuntimeError, match="pair failed"):
        models.run(np.zeros((2, 3, 256, 192), np.float32))
    assert models.models[1].calls == []
    models.close()


def _detected(count):
    boxes = np.array([[2 + i, 3 + i, 55 + i, 60 + i] for i in range(count)], np.float32)
    frame = SimpleNamespace(image_bgr=np.zeros((80, 80, 3), np.uint8), decode_ms=.25,
                            identity=SimpleNamespace(sequence=0))
    return Detected(frame, boxes, {"detector_stage_ms": 1.0}, 1, 1)


def test_pose_batched_uses_pair_then_unpadded_remainder_and_preserves_count():
    session = RoutedModel((1, 3, 256, 192))
    session.metadata["same_frame_batching"] = {
        "maximum_batch_size": 2,
        "future_frame_batching": False,
        "parallel_model_calls": False,
    }
    packet = pose_batched(session, 1, _detected(3), (192, 256), .3, True, "opencv")
    assert session.calls == [(2, 3, 256, 192), (1, 3, 256, 192)]
    assert len(packet.people) == len(packet.per_person_ms) == 3
    assert sum(packet.per_person_ms) == pytest.approx(packet.times["pose_inference_ms"])
    assert packet.times["pose_stage_ms"] >= packet.times["pose_inference_ms"]


def test_pose_batched_never_calls_model_for_zero_people():
    session = RoutedModel((1, 3, 256, 192))
    session.metadata["same_frame_batching"] = {
        "maximum_batch_size": 2,
        "future_frame_batching": False,
        "parallel_model_calls": False,
    }
    packet = pose_batched(session, 1, _detected(0), (192, 256), .3, True, "opencv")
    assert session.calls == []
    assert packet.people == [] and packet.per_person_ms == ()


def test_pose_batched_rejects_unproven_session_contract():
    session = RoutedModel((1, 3, 256, 192))
    with pytest.raises(BenchmarkError, match="session_contract"):
        pose_batched(session, 1, _detected(2), (192, 256), .3, True, "opencv")
