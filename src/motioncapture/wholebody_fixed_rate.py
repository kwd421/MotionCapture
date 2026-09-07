"""Deterministic fixed-rate file release for capacity validation; not a camera emulator."""
from __future__ import annotations

import math
import threading
import time
from fractions import Fraction

from motioncapture.recording import RecordedIdentity
from motioncapture.wholebody_catalog import BenchmarkError
from motioncapture.wholebody_replay import Release, ReplayCancelled


class FixedRatePacer:
    """Release every original frame on an absolute fixed-rate host schedule.

    Source identity and PTS remain untouched and must stay monotonic. PTS are
    validation metadata here, not the release clock. Late work never rebases the
    epoch and no source frame is skipped.
    """

    def __init__(self, limit: int, *, rate_hz=60, clock=time.perf_counter_ns, waiter=None):
        if type(limit) is not int or not 1 <= limit <= 120000:
            raise BenchmarkError("invalid_fixed_rate_limit")
        if isinstance(rate_hz, bool) or not isinstance(rate_hz, (int, Fraction)):
            raise BenchmarkError("invalid_fixed_release_rate")
        rate = Fraction(rate_hz)
        if rate <= 0 or rate > 1000:
            raise BenchmarkError("invalid_fixed_release_rate")
        self.limit = limit
        self.rate_hz = rate
        self._clock = clock
        self._stop = threading.Event()
        self._wait = self._stop.wait if waiter is None else waiter
        self._epoch = None
        self._identity_clock = None
        self._first_pts = self._last_pts = None
        self._last_clock = self._last_due = None
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
            raise BenchmarkError("fixed_release_host_clock_regressed")
        self._last_clock = value
        return value

    def wait(self, identity: RecordedIdentity) -> Release:
        key = (identity.source_id, identity.stream_id, identity.time_base)
        if (type(identity.sequence) is not int or identity.sequence != self.released
                or self.released >= self.limit or type(identity.pts) is not int
                or not isinstance(identity.time_base, Fraction) or identity.time_base <= 0
                or (self._identity_clock is not None and key != self._identity_clock)
                or (self._last_pts is not None and identity.pts <= self._last_pts)):
            raise BenchmarkError("fixed_release_source_identity_mismatch")
        if self.cancelled:
            raise ReplayCancelled()
        began = self._now()
        if self._epoch is None:
            self._epoch = began
            self._identity_clock = key
            self._first_pts = identity.pts
        offset_ns = Fraction(identity.sequence, 1) / self.rate_hz * 1_000_000_000
        due = self._epoch + math.ceil(offset_ns)
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
        source_span = None
        if (self._first_pts is not None and self._last_pts is not None
                and self._identity_clock is not None):
            source_span = float((self._last_pts - self._first_pts) * self._identity_clock[2])
        return {
            "policy": "fixed_rate_fixed_epoch_no_skip_no_rebase",
            "configured_rate_hz": float(self.rate_hz),
            "epoch": "first_detector_worker_entry_after_setup_and_first_decode",
            "first_decode_in_source_age": False,
            "nanosecond_rounding": "ceil_of_absolute_rational_sequence_offset",
            "expected_frames": self.limit,
            "released_frames": self.released,
            "pending_sequence": self.pending_sequence,
            "scheduled_source_span_s": ((self._last_due - self._epoch) / 1e9
                                        if self._last_due is not None else None),
            "scheduled_source_rate_hz": (
                (self.released - 1) * 1e9 / (self._last_due - self._epoch)
                if self.released > 1 and self._last_due > self._epoch else None),
            "original_pts_span_s": source_span,
            "source_time_base": (str(self._identity_clock[2])
                                 if self._identity_clock is not None else None),
            "source_pts_preserved": True,
            "source_pts_rewritten": False,
            "source_pts_drive_release_schedule": False,
            "cancel_requested": self.cancelled,
            "clock_rebases": 0,
            "frames_skipped": 0,
            "wait_owner": "detector_worker_not_result_consumer",
            "hardware_camera_emulated": False,
            "sensor_exposure_timing_emulated": False,
        }
