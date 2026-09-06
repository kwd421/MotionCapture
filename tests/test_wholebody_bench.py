"""Real VFR file path with explicitly synthetic estimators; not model benchmarks."""
from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace as NS

import numpy as np
import pytest

from motioncapture import wholebody_bench as b
from motioncapture.wholebody_catalog import BenchmarkError
from motioncapture.wholebody_onnx import Person2D


@pytest.fixture
def clip(tmp_path):
    video = tmp_path / 'controlled-vfr.mp4'
    subprocess.run([
        'ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
        'testsrc2=size=64x48:rate=30:duration=0.2', '-vf',
        'setpts=(N+floor(N/2))/(30*TB)', '-fps_mode', 'passthrough',
        '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p', str(video),
    ], check=True, timeout=15)
    probe = b.inspect_recording(video)
    assert len(probe.pts) == 6
    assert len(set(np.diff(probe.pts))) > 1
    return video, probe


def args(video, output, *extra):
    return b.parser().parse_args(['run', str(video), '--research-only',
                                 '--decode-threads', '1', '--warmup-frames', '2',
                                 '--output', str(output), *extra])


class FakeEstimator:
    instances = []
    fail_at = None
    fail_close = False

    def __init__(self, *args, **kwargs):
        self.calls, self.closed = 0, False
        self.metadata = {'synthetic_test_estimator': True}
        self.instances.append(self)

    def process(self, image):
        if self.calls == self.fail_at:
            raise BenchmarkError('injected_inference_failure')
        self.calls += 1
        person = Person2D(np.tile([self.calls, 20.], (133, 1)),
                          np.full(133, .8), np.ones(133, np.bool_))
        return [person], {'wholebody_service_ms': .2, 'pose_inference_ms': .1}

    def close(self):
        self.closed = True
        if self.fail_close:
            raise RuntimeError('injected cleanup failure')


@pytest.fixture(autouse=True)
def reset_fake():
    FakeEstimator.instances = []
    FakeEstimator.fail_at = None
    FakeEstimator.fail_close = False


@pytest.mark.parametrize('limit,expected', [(0, 6), (3, 3), (99, 6)])
def test_real_vfr_all_frames_and_prefix_without_n_plus_one(clip, tmp_path, limit, expected):
    video, probe = clip
    config = args(video, tmp_path / 'report.json', '--max-frames', str(limit))
    first, bank = b.candidate_pass(config, probe, 'dwpose-m', 'cpu',
                                    estimator_factory=FakeEstimator)
    assert first['status'] == 'completed' and first['all_frames']['frames'] == expected
    assert first['steady_after_initial_frames']['frames'] == expected - 2
    assert FakeEstimator.instances[-1].calls == expected and FakeEstimator.instances[-1].closed
    assert list(bank.pts) == list(probe.pts[:expected])
    second, _ = b.candidate_pass(config, probe, 'dwpose-m', 'cpu', bank,
                                 estimator_factory=FakeEstimator)
    assert first['predictions_sha256'] == second['predictions_sha256']
    assert second['cpu_reference_hash_equal'] is True
    assert second['provider_disagreement']['parts']['left_hand']['mean_pixels'] == 0
    assert first['live_60fps_verified'] is False and first['accuracy_verified'] is False
    json.dumps(second, allow_nan=False)


def test_failed_pass_keeps_completed_prefix_and_primary_error(clip, tmp_path):
    video, probe = clip
    FakeEstimator.fail_at, FakeEstimator.fail_close = 3, True
    record, bank = b.candidate_pass(args(video, tmp_path / 'unused.json'), probe,
                                    'dwpose-m', 'cpu', estimator_factory=FakeEstimator)
    assert record['status'] == 'failed' and bank is None
    assert record['all_frames']['frames'] == 3
    assert record['error']['code'] == 'injected_inference_failure'
    assert record['error']['current_frame']['sequence'] == 3
    assert record['error']['current_frame']['pts'] == probe.pts[3]
    assert record['error']['last_completed_frame']['sequence'] == 2
    assert record['unpaced_loop_fps'] is None and record['hash_scope'] == 'partial_prefix'
    assert record['decoder_cleanup'] == 'owner_released'
    assert record['estimator_cleanup'] == 'failed' and len(record['cleanup_errors']) == 1
    json.dumps(record, allow_nan=False)


def test_no_person_and_multi_person_are_not_fabricated_matches():
    frame = NS(identity=NS(sequence=0, pts=7))
    person = Person2D(np.ones((133, 2)), np.ones(133), np.ones(133, np.bool_))
    bank = b.ReferenceBank(1)
    bank.store(frame, [person])
    comparison = b.Disagreement()
    comparison.add(bank, frame, [])
    summary = comparison.summary()
    assert summary['parts']['left_hand']['mean_pixels'] is None
    assert summary['reference_only_point_observations']['left_hand'] == 21
    comparison.add(bank, frame, [person, person])
    assert comparison.summary()['ambiguous_multi_person_frames'] == 1
    wrong_time = NS(identity=NS(sequence=0, pts=8))
    with pytest.raises(BenchmarkError, match='reference_pts_mismatch'):
        comparison.add(bank, wrong_time, [person])


def test_control_plan_and_research_acknowledgement(tmp_path):
    plan = b.make_plan(['dwpose-m'], ['cpu', 'coreml-all'], True, True)
    assert plan == [('mediapipe-native', 'cpu'), ('dwpose-m', 'cpu'),
                    ('dwpose-m', 'coreml-all'), ('dwpose-m', 'coreml-all'),
                    ('dwpose-m', 'cpu'), ('mediapipe-native', 'cpu')]
    with pytest.raises(BenchmarkError, match='must_be_first'):
        b.make_plan(['rtmw-m'], ['coreml-all', 'cpu'], False, False)
    config = args(tmp_path / 'absent.mp4', tmp_path / 'r.json')
    config.research_only = False
    with pytest.raises(BenchmarkError, match='acknowledgement'):
        b.execute(config)
    assert not config.output.exists()


def test_execute_checkpoints_survive_later_failed_arm_and_no_clobber(clip, tmp_path, monkeypatch):
    video, probe = clip
    config = args(video, tmp_path / 'bench.json', '--models', 'dwpose-m',
                  '--providers', 'cpu', 'coreml-all', '--abba')
    recorded = []
    real_pass = b.candidate_pass

    def fake_pass(*a, **kw):
        FakeEstimator.fail_at = 3 if len(recorded) == 2 else None
        result = real_pass(*a, **kw, estimator_factory=FakeEstimator)
        recorded.append(result[0])
        return result

    monkeypatch.setattr(b, 'candidate_pass', fake_pass)
    assert b.execute(config) == 2
    assert [x['status'] for x in recorded] == ['completed', 'completed', 'failed', 'completed']
    report = json.loads(config.output.read_text())
    assert report['status'] == 'partial_failed'
    assert len(list(tmp_path.glob('bench.arm-*.json'))) == 4
    saved = config.output.read_bytes()
    with pytest.raises(BenchmarkError, match='already_exists'):
        b.execute(config)
    assert config.output.read_bytes() == saved
    # No capture images, raw landmarks or source paths in the report/checkpoints.
    assert str(video) not in config.output.read_text()
    assert report['privacy']['landmarks_written'] is False
    assert report['commercial_release_cleared'] is False


def test_native_control_retains_existing_observation_contract(monkeypatch, tmp_path):
    seen = []

    def native(native_args, probe, **kw):
        seen.append((native_args, kw))
        return {'status': 'completed', 'predictions_sha256': 'reference'}

    monkeypatch.setattr(b, 'run_pass', native)
    config = args(tmp_path / 'clip.mp4', tmp_path / 'r.json', '--max-frames', '900')
    record = b.native_control(config, None)
    assert seen[0][0].preview == 'none' and seen[0][0].verify_results is True
    assert seen[0][1]['frame_limit'] == 900
    assert record['capabilities']['face_blendshapes'] is True
    assert record['not_feature_equivalent_to_133point_candidates'] is True
