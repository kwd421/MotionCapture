import math
from dataclasses import replace
from fractions import Fraction

import pytest

from motioncapture import wholebody_pose_batch_60hz_lab as lab
from motioncapture.recording import RecordedIdentity
from motioncapture.wholebody_catalog import BenchmarkError
from motioncapture.wholebody_fixed_rate import FixedRatePacer
from motioncapture.wholebody_replay import ReplayAges, ReplayCancelled, SourcePacer


class Clock:
    def __init__(self):
        self.ns = 7_000_000_000
        self.waits = []

    def now(self):
        return self.ns

    def wait(self, seconds):
        self.waits.append(seconds)
        self.ns += math.ceil(seconds * 1e9)
        return False


def identity(i, pts, tb=Fraction(1, 90000)):
    return RecordedIdentity("fixture", "stream", i, pts, tb)


def test_fixed60_uses_absolute_sequence_schedule_without_rewriting_vfr_pts():
    clock = Clock()
    pacer = FixedRatePacer(4, rate_hz=60, clock=clock.now, waiter=clock.wait)
    epoch = clock.ns
    rows = [
        pacer.wait(identity(0, 0)),
        pacer.wait(identity(1, 759)),
        pacer.wait(identity(2, 2277)),
        pacer.wait(identity(3, 3036)),
    ]
    assert [row.due_ns for row in rows] == [
        epoch + math.ceil(Fraction(i, 60) * 1_000_000_000) for i in range(4)]
    summary = pacer.summary()
    assert summary["configured_rate_hz"] == 60.0
    assert summary["source_pts_preserved"] and not summary["source_pts_rewritten"]
    assert not summary["source_pts_drive_release_schedule"]
    assert summary["released_frames"] == 4 and summary["frames_skipped"] == 0
    assert summary["clock_rebases"] == 0 and summary["hardware_camera_emulated"] is False


def test_fixed60_late_work_never_rebases_or_sleeps_to_hide_backlog():
    clock = Clock()
    pacer = FixedRatePacer(3, clock=clock.now, waiter=clock.wait)
    pacer.wait(identity(0, 0))
    pacer.wait(identity(1, 1517))
    waits = len(clock.waits)
    clock.ns += 200_000_000
    late = pacer.wait(identity(2, 2276))
    assert late.released_ns > late.due_ns and late.wait_ms == 0
    assert len(clock.waits) == waits
    assert pacer.summary()["clock_rebases"] == 0


@pytest.mark.parametrize("bad", [
    identity(2, 1),
    identity(1, 0),
    identity(1, 1, Fraction(1, 1)),
    replace(identity(1, 1), stream_id="changed"),
])
def test_fixed60_rejects_source_identity_discontinuity(bad):
    clock = Clock()
    pacer = FixedRatePacer(2, clock=clock.now, waiter=clock.wait)
    pacer.wait(identity(0, 0))
    with pytest.raises(BenchmarkError, match="source_identity"):
        pacer.wait(bad)
    assert pacer.released == 1


def test_fixed60_cancel_does_not_release_pending_frame():
    clock = Clock()
    pacer = FixedRatePacer(2, clock=clock.now, waiter=clock.wait)
    pacer.wait(identity(0, 0))
    pacer.cancel()
    with pytest.raises(ReplayCancelled):
        pacer.wait(identity(1, 1517))
    assert pacer.released == 1 and pacer.summary()["cancel_requested"]


def test_fixed60_pipeline_factory_replaces_only_placeholder_pacer(monkeypatch):
    captured = {}

    class Pipeline:
        def __init__(self, *factories, **kwargs):
            captured["factories"] = factories
            captured["kwargs"] = kwargs

    monkeypatch.setattr(lab, "BatchStagePipeline", Pipeline)
    placeholder = SourcePacer(5)
    result = lab._fixed60_pipeline_factory("detector", "pose", pacer=placeholder,
                                           normalization_kernel="opencv")
    assert isinstance(result, Pipeline)
    fixed = captured["kwargs"]["pacer"]
    assert isinstance(fixed, FixedRatePacer) and fixed.limit == 5
    assert fixed.summary()["configured_rate_hz"] == 60.0
    assert placeholder.released == 0
    with pytest.raises(BenchmarkError, match="placeholder"):
        lab._fixed60_pipeline_factory("detector", "pose", pacer=fixed)


def test_strict_summary_requires_two_complete_lossless_60hz_runs():
    def row(max_age, last_age, over100):
        return {
            "status": "completed",
            "scope": "full_file",
            "execution_arm": {"release_schedule": "fixed_60hz"},
            "pipeline": {
                "read_frames": 10, "emitted_frames": 10, "pose_requests": 10,
                "intentional_frame_skips": 0, "unemitted_read_frames": 0,
                "unemitted_pose_requests": 0,
                "source_pacing": {
                    "frames_skipped": 0, "configured_rate_hz": 60.0,
                    "scheduled_source_rate_hz": 60.0,
                    "observed_release_rate_hz": 59.9,
                },
            },
            "replay_ages": {
                "source_age_ms": {
                    "mean_ms": 25.0, "p95_ms": 35.0, "p99_ms": 50.0,
                    "max_ms": max_age,
                    "threshold_exceedances": {"100.0": over100},
                },
                "last_source_age_ms": last_age,
            },
        }
    result = lab._strict_summary([row(80, 20, 0), row(90, 22, 0)])
    assert result["full_file_replayed_with_60hz_schedule"]
    assert result["all_frames_preserved"]
    assert result["configured_release_schedule_60hz"]
    assert result["observed_release_rate_hz"] == [59.9, 59.9]
    assert result["over_100ms_frames"] == [0, 0]
    assert result["live_camera_60fps_verified"] is False


def test_slow_actual_releases_are_not_reported_as_observed_60hz():
    clock = Clock()
    pacer = FixedRatePacer(61, clock=clock.now, waiter=clock.wait)
    ages = ReplayAges(61)
    epoch = clock.ns
    for i in range(61):
        clock.ns = epoch + i * 100_000_000  # Controlled 10Hz throughput.
        frame_id = identity(i, i, Fraction(1, 60))
        release = pacer.wait(frame_id)
        ages.add(frame_id, release, release.released_ns + 1_000_000)
    row = {
        "status": "completed", "scope": "full_file",
        "execution_arm": {"release_schedule": "fixed_60hz"},
        "pipeline": {
            "read_frames": 61, "emitted_frames": 61, "pose_requests": 61,
            "intentional_frame_skips": 0, "unemitted_read_frames": 0,
            "unemitted_pose_requests": 0, "source_pacing": pacer.summary(),
        },
        "replay_ages": ages.summary(),
    }
    result = lab._strict_summary([row, row])
    assert result["configured_release_schedule_60hz"]
    assert result["observed_release_rate_hz"] == [10.0, 10.0]
    assert result["last_source_age_ms"] == [5001.0, 5001.0]
    assert "configured_and_observed_release_rate_60hz" not in result
    assert "file_replay_exercised_at_fixed_60hz" not in result


def test_one_release_keeps_observed_rate_unknown_and_summary_valid():
    clock = Clock()
    pacer = FixedRatePacer(1, clock=clock.now, waiter=clock.wait)
    assert pacer.summary()["observed_release_rate_hz"] is None
    frame_id = identity(0, 0)
    release = pacer.wait(frame_id)
    ages = ReplayAges(1)
    ages.add(frame_id, release, release.released_ns + 1_000_000)
    row = {
        "status": "completed", "scope": "explicit_prefix",
        "execution_arm": {"release_schedule": "fixed_60hz"},
        "pipeline": {"source_pacing": pacer.summary()}, "replay_ages": ages.summary(),
    }
    result = lab._strict_summary([row, row])
    assert result["observed_release_rate_hz"] == [None, None]
    assert not result["configured_release_schedule_60hz"]
    assert not result["full_file_replayed_with_60hz_schedule"]
