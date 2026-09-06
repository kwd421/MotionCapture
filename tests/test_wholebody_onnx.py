"""Real OpenCV numeric tests; fake model calls are explicitly marked."""
from __future__ import annotations

import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from motioncapture import wholebody_onnx as w
from motioncapture.wholebody_catalog import BenchmarkError


def heads(x=192, y=256):
    a, b = np.zeros((1, 133, 384), np.float32), np.zeros((1, 133, 512), np.float32)
    a[:, :, x], b[:, :, y] = .8, .6
    return a, b


def test_real_affine_crop_color_and_inverse_geometry():
    image = np.zeros((120, 240, 3), np.uint8)
    image[:] = [20, 80, 170]  # Distinct B/G/R catches an accidental channel swap.
    original = image.copy()
    tensor, center, scale = w.pose_tensor(image, np.array([60, 20, 180, 100.]), (192, 256))
    assert tensor.shape == (1, 3, 256, 192) and tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[0, :, 128, 96],
                               (np.array([20, 80, 170]) - [123.675, 116.28, 103.53])
                               / [58.395, 57.12, 57.375], rtol=1e-6)
    result = w.decode_pose(list(reversed(heads())), (192, 256), center, scale, .3)
    np.testing.assert_allclose(result.xy, np.tile(center, (133, 1)))
    np.testing.assert_allclose(result.scores, .7)
    assert result.valid.all() and not result.xy.flags.writeable
    # A quarter-image displacement in SimCC must invert through the aspect-adjusted crop.
    shifted = w.decode_pose(list(heads(96, 128)), (192, 256), center, scale, .3)
    np.testing.assert_allclose(shifted.xy[0], center - scale / 4)
    np.testing.assert_array_equal(image, original)


def test_simcc_absence_not_probability_or_fake_3d():
    a, b = heads()
    a[:, 0], b[:, 0] = 0, 0
    a[:, 1, 0], b[:, 1, 0] = 3, 3
    result = w.decode_pose([a, b], (192, 256), np.array([100, 100]), np.array([150, 200]), .3)
    assert not result.valid[0] and result.scores[1] == 3
    assert not w.CAPABILITIES['face_blendshapes'] and not w.CAPABILITIES['metric_3d']
    assert sum(b - a for a, b in w.PARTS.values()) == 133
    assert w.PARTS['left_hand'] == (91, 112) and w.PARTS['right_hand'] == (112, 133)
    b[0, 4, 1] = np.nan
    with pytest.raises(BenchmarkError, match='nonfinite_simcc'):
        w.decode_pose([a, b], (192, 256), np.zeros(2), np.ones(2), .3)
    with pytest.raises(BenchmarkError, match='invalid_simcc_schema'):
        w.decode_pose([a[:, :17], b[:, :17]], (192, 256), np.zeros(2), np.ones(2), .3)


def test_real_detector_letterbox_person_filter_and_nms():
    image = np.full((100, 200, 3), 42, np.uint8)
    tensor, ratio = w.detector_tensor(image)
    assert ratio == 2.08 and tensor.shape == (1, 3, 416, 416)
    assert np.all(tensor[:, :, :208] == 42) and np.all(tensor[:, :, 208:] == 114)
    raw = np.zeros((1, 3549, 85), np.float32)
    # Two overlapping person boxes at the same decoded location.
    raw[0, 0, :6] = [10, 10, np.log(10), np.log(10), .9, .9]
    raw[0, 1, :6] = [9, 10, np.log(10), np.log(10), .8, .9]
    # High-confidence different class MUST NOT appear as a person.
    raw[0, 2, 4], raw[0, 2, 6] = 1, 1
    boxes = w.decode_people(raw, 1)
    assert boxes.shape == (1, 4)
    np.testing.assert_allclose(boxes[0], [40, 40, 120, 120], atol=1e-4)
    assert w.decode_people(np.zeros_like(raw), 1).shape == (0, 4)
    with pytest.raises(BenchmarkError, match='invalid_yolox_schema'):
        w.decode_people(raw[:, :, :6], 1)


def test_estimator_no_full_image_fallback_and_cleans_both_owners(monkeypatch, tmp_path):
    monkeypatch.setattr(w, 'verify_asset', lambda root, key: (tmp_path / key, {'test': True}))
    instances = []

    class FakeSession:
        def __init__(self, path, shape, provider, **kwargs):
            self.name, self.shape, self.calls = path.name, shape, 0
            self.metadata = {'fake': True, 'provider': provider}
            self.closed = False
            instances.append(self)

        def run(self, tensor):
            self.calls += 1
            assert tensor.shape == self.shape
            return [np.zeros((1, 3549, 85), np.float32)]

        def close(self):
            self.closed = True
            if self.name == 'dwpose-m':
                raise RuntimeError('injected close failure')

    estimator = w.WholebodyEstimator(tmp_path, 'dwpose-m', 'cpu', session_factory=FakeSession)
    people, timings = estimator.process(np.zeros((100, 100, 3), np.uint8))
    assert people == [] and instances[0].calls == 1 and instances[1].calls == 0
    assert timings['pose_inference_ms'] == 0  # No call is recorded as zero work, not a valid pose.
    with pytest.raises(BenchmarkError, match='estimator_cleanup_failed'):
        estimator.close()
    assert all(x.closed for x in instances)
    assert estimator.pose is None and estimator.detector is None


def test_coreml_selection_and_profile_reject_silent_cpu(monkeypatch):
    monkeypatch.setattr(w.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(w.platform, 'machine', lambda: 'arm64')
    available = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    providers, expected = w.provider_plan('coreml-ane', available, allow_cpu=False)
    assert len(providers) == 1
    assert providers[0][1]['MLComputeUnits'] == 'CPUAndNeuralEngine'
    assert providers[0][1]['ModelFormat'] == 'MLProgram'
    mixed, _ = w.provider_plan('coreml-gpu', available, allow_cpu=True)
    assert mixed[-1] == 'CPUExecutionProvider'
    events = [{'cat': 'Node', 'args': {'provider': 'CPUExecutionProvider'}}]
    with pytest.raises(BenchmarkError, match='no_profiled_nodes'):
        w.profile_placement(events, expected, True)
    events.append({'cat': 'Node', 'args': {'provider': expected}})
    with pytest.raises(BenchmarkError, match='forbidden_cpu'):
        w.profile_placement(events, expected, False)
    assert not w.profile_placement(events, expected, True)['ane_dispatch_verified']
    with pytest.raises(BenchmarkError, match='unavailable'):
        w.provider_plan('coreml-all', ['CPUExecutionProvider'], allow_cpu=False)
    monkeypatch.setattr(w.platform, 'system', lambda: 'Linux')
    with pytest.raises(BenchmarkError, match='requires_apple'):
        w.provider_plan('coreml-all', available, allow_cpu=False)


def test_session_initialization_rejects_substitution_before_any_inference(monkeypatch, tmp_path):
    """Synthetic ORT/protobuf boundary; not a native ONNX test."""
    order = []

    class TensorProto:
        FLOAT, EXTERNAL = 1, 1

    graph = NS(graph=NS(initializer=[], input=[NS(name='input', type=NS(tensor_type=NS(
        elem_type=1, shape=NS(dim=[NS(dim_param='', dim_value=v) for v in (1, 3, 256, 192)]))))]),
        ListFields=lambda: [])
    monkeypatch.setitem(sys.modules, 'onnx', NS(
        load=lambda *a, **k: graph, TensorProto=TensorProto))

    class Options:
        def add_session_config_entry(self, *a):
            order.append(a)

    class Session:
        def __init__(self, *a, **kw):
            order.append('construct')

        def disable_fallback(self):
            order.append('disable_fallback')

        def get_providers(self):
            return ['CPUExecutionProvider']  # Mimics a substituted CoreML initialization.

        def run(self, *a):
            pytest.fail('Substituted provider must never run inference')

    ort = NS(__version__='1.22.1', get_available_providers=lambda: ['CoreMLExecutionProvider'],
             SessionOptions=Options, ExecutionMode=NS(ORT_SEQUENTIAL=0), InferenceSession=Session)
    monkeypatch.setitem(sys.modules, 'onnxruntime', ort)
    monkeypatch.setattr(w.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(w.platform, 'machine', lambda: 'arm64')
    with pytest.raises(BenchmarkError, match='provider_substitution_rejected'):
        w.OrtModel(tmp_path / 'unused.onnx', (1, 3, 256, 192), 'coreml-all')
    assert ('session.disable_cpu_ep_fallback', '1') in order
    assert order[-1] == 'disable_fallback'


def test_real_codec_chain_with_scripted_onnx_outputs(monkeypatch, tmp_path):
    """Detector/crop/pose decoding is real; both network tensors are fixtures."""
    monkeypatch.setattr(w, 'verify_asset', lambda root, key: (tmp_path / key, {}))
    captured = []

    class Session:
        def __init__(self, path, shape, provider, **kw):
            self.shape, self.metadata = shape, {'fixture_only': True}

        def run(self, tensor):
            captured.append(tensor.shape)
            if self.shape == (1, 3, 416, 416):
                raw = np.zeros((1, 3549, 85), np.float32)
                raw[0, 0, :6] = [26, 26, np.log(20), np.log(20), .9, .9]
                return [raw]
            return list(heads())

        def close(self):
            pass

    engine = w.WholebodyEstimator(tmp_path, 'rtmw-m', 'cpu', session_factory=Session)
    source = np.full((416, 416, 3), 123, np.uint8)
    people, timings = engine.process(source)
    assert captured == [(1, 3, 416, 416), (1, 3, 256, 192)]
    assert len(people) == 1 and people[0].valid.all()
    np.testing.assert_allclose(people[0].xy, np.full((133, 2), 208.), atol=1e-5)
    assert all(v >= 0 for v in timings.values())
    assert engine.metadata['detector_cadence'] == 'every_source_frame'
    assert np.all(source == 123)
    engine.close()
