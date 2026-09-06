"""Explicit real-time file release, not camera capture or a 60Hz frame generator."""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from fractions import Fraction

from motioncapture.recording import RecordedIdentity
from motioncapture.recording_bench import Samples
from motioncapture.wholebody_catalog import BenchmarkError


class ReplayCancelled(BenchmarkError):
    def __init__(self):
        super().__init__("source_replay_cancelled")


@dataclass(frozen=True, slots=True)
class Release:
    due_ns: int
    released_ns: int
    wait_ms: float


class SourcePacer:
    """Detector-owned schedule. Only cancel() crosses ownership; uses Event.

    Read summary only after the detector owner has joined. There is no new worker,
    buffer, timer thread or period accumulation. Late frames never reset the epoch.
    """
    def __init__(self, limit: int, *, clock=time.perf_counter_ns, waiter=None):
        if type(limit) is not int or not 1 <= limit <= 120000:
            raise BenchmarkError("invalid_replay_limit")
        self.limit = limit
        self._clock = clock
        self._stop = threading.Event()
        self._wait = self._stop.wait if waiter is None else waiter
        self._epoch = self._first_pts = self._last_pts = None
        self._identity_clock = None
        self._last_clock = None
        self._last_due = None
        self.released = 0
        self.pending_sequence = None

    @property
    def cancelled(self) -> bool:
        return self._stop.is_set()

    def cancel(self) -> None:
        self._stop.set()

    def _now(self) -> int:
        value = self._clock()
        if (type(value) is not int or value < 0
                or (self._last_clock is not None and value < self._last_clock)):
            raise BenchmarkError("replay_host_clock_regressed")
        self._last_clock = value
        return value

    def wait(self, identity: RecordedIdentity) -> Release:
        key = (identity.source_id, identity.stream_id, identity.time_base)
        if (type(identity.sequence) is not int or identity.sequence != self.released
                or self.released >= self.limit or type(identity.pts) is not int
                or not isinstance(identity.time_base, Fraction) or identity.time_base <= 0
                or (self._identity_clock is not None and key != self._identity_clock)
                or (self._last_pts is not None and identity.pts <= self._last_pts)):
            raise BenchmarkError("replay_source_clock_mismatch")
        if self.cancelled:
            raise ReplayCancelled()
        began = self._now()
        if self._epoch is None:
            self._epoch, self._first_pts, self._identity_clock = began, identity.pts, key
        offset = (identity.pts - self._first_pts) * identity.time_base * 1_000_000_000
        # Absolute rational offset, not repeated rounded frame periods.
        due = self._epoch + math.ceil(offset)
        self.pending_sequence = identity.sequence
        now = began
        while now < due:
            if self.cancelled or self._wait((due - now) / 1e9):
                raise ReplayCancelled()
            now = self._now()
        if self.cancelled:
            raise ReplayCancelled()
        self.released += 1
        self._last_pts, self._last_due = identity.pts, due
        self.pending_sequence = None
        return Release(due, now, (now - began) / 1e6)

    def summary(self) -> dict:
        return {
            "policy": "original_pts_fixed_epoch_no_skip_no_rebase",
            "epoch": "first_detector_worker_entry_after_setup_and_first_decode",
            "first_decode_in_source_age": False,
            "nanosecond_rounding": "ceil_of_absolute_rational_pts_offset",
            "expected_frames": self.limit, "released_frames": self.released,
            "pending_sequence": self.pending_sequence,
            "scheduled_source_span_s": ((self._last_due - self._epoch) / 1e9
                                         if self._last_due is not None else None),
            "scheduled_source_rate_hz": ((self.released - 1)*1e9 / (self._last_due-self._epoch)
                                         if self.released > 1 and self._last_due > self._epoch
                                         else None),
            "source_time_base": (str(self._identity_clock[2])
                                 if self._identity_clock is not None else None),
            "cancel_requested": self.cancelled,
            "clock_rebases": 0, "frames_skipped": 0,
            "wait_owner": "detector_worker_not_result_consumer",
            "hardware_camera_emulated": False,
        }


def _age_summary(samples: Samples) -> dict:
    result = samples.summary(0)
    result.pop("over_budget")  # Frame interval is NOT a source-age acceptance budget.
    result["threshold_exceedances"] = {
        str(limit): sum(value > limit for value in samples.data)
        for limit in (33.333333333333336, 50.0, 100.0)
    }
    return result


class ReplayAges:
    """Consumer-owned source-age statistics; no source images or joint values."""
    def __init__(self, limit: int):
        if type(limit) is not int or not 1 <= limit <= 120000:
            raise BenchmarkError("invalid_replay_limit")
        self.limit = limit
        self.frames = 0
        self._first = self._last_age = self._clock = None
        self._previous_due = self._previous_verified = self._last_pts = None
        self.ages, self.lateness = Samples(), Samples()
        self.windows = {}
        self.by_people, self.worst = {}, []

    def add(self, identity, release: Release, verified_ns: int, *, people=None,
            pose_stage_ms=None) -> None:
        clock = (identity.source_id, identity.stream_id, identity.time_base)
        if (identity.sequence != self.frames or self.frames >= self.limit
                or not isinstance(identity.time_base, Fraction) or identity.time_base <= 0
                or (self._last_pts is not None and identity.pts <= self._last_pts)
                or not isinstance(release, Release) or type(verified_ns) is not int
                or release.due_ns > release.released_ns or verified_ns < release.released_ns
                or (self._clock is not None and clock != self._clock)
                or (self._previous_due is not None and release.due_ns <= self._previous_due)
                or (self._previous_verified is not None
                    and verified_ns <= self._previous_verified)):
            raise BenchmarkError("invalid_replay_age_observation")
        if ((people is not None and (type(people) is not int or not 0 <= people <= 8))
                or (pose_stage_ms is not None and (not math.isfinite(pose_stage_ms)
                                                   or pose_stage_ms < 0))):
            raise BenchmarkError("invalid_replay_workload")
        if self._first is None:
            self._first, self._clock = identity.pts, clock
        age = (verified_ns - release.due_ns) / 1e6
        late = (release.released_ns - release.due_ns) / 1e6
        self.ages.add(age)
        self.lateness.add(late)
        if people is not None:
            self.by_people.setdefault(str(people), Samples()).add(age)
        if len(self.worst) < 16 or age > self.worst[-1]["source_age_ms"]:
            self.worst.append({"sequence": identity.sequence, "pts": identity.pts,
                               "people_at_output": people, "source_age_ms": age,
                               "release_lateness_ms": late, "pose_stage_ms": pose_stage_ms})
            self.worst.sort(key=lambda row: row["source_age_ms"], reverse=True)
            del self.worst[16:]
        bucket = int((identity.pts - self._first) * identity.time_base // 10)
        self.windows.setdefault(bucket, Samples()).add(age)
        self._last_age = age
        self._previous_due, self._previous_verified = release.due_ns, verified_ns
        self._last_pts = identity.pts
        self.frames += 1

    def summary(self) -> dict:
        return {
            "scope": "scheduled_file_frame_release_to_validation_complete; not photon latency",
            "frames": self.frames, "expected_frames": self.limit,
            "coverage": "complete" if self.frames == self.limit else "partial",
            "source_age_ms": _age_summary(self.ages),
            "detector_release_lateness_ms": _age_summary(self.lateness),
            "last_source_age_ms": self._last_age,
            "source_age_by_ending_person_count": {
                key: _age_summary(value) for key, value in self.by_people.items()},
            "worst_source_ages": self.worst,
            "workload_scope": "ending frame; backlog may originate in earlier frames",
            "source_windows": [{"start_s": key * 10, "end_s": (key + 1) * 10,
                                "observed_frames": len(value.data),
                                "source_age_ms": _age_summary(value)}
                               for key, value in self.windows.items()],
            "thresholds_are_diagnostics_not_acceptance_criteria": True,
            "live_60fps_verified": False,
        }
