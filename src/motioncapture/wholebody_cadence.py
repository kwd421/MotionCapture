"""Metadata-only cadence for unpaced file tests; never a camera/deadline verdict."""
from __future__ import annotations

from motioncapture.recording_bench import Samples
from motioncapture.wholebody_catalog import BenchmarkError


class OutputCadence:
    """One observer owns bounded scalar samples; no images or joint coordinates."""

    def __init__(self, limit: int, first_pts: int):
        if type(limit) is not int or not 1 <= limit <= 120000:
            raise BenchmarkError("invalid_cadence_limit")
        self.limit, self.first_pts = limit, first_pts
        self.frames = 0
        self.previous = None
        self.clock = None
        self.intervals = Samples()
        self.by_people, self.windows = {}, {}
        self.worst = []
        self.streak = self.longest_streak = 0

    def add(self, identity, people: int, verified_ns: int) -> None:
        clock = (identity.source_id, identity.time_base)
        if (identity.sequence != self.frames or self.frames >= self.limit
                or type(people) is not int or not 0 <= people <= 8
                or type(verified_ns) is not int or verified_ns < 0
                or identity.time_base <= 0 or identity.pts < self.first_pts
                or (self.clock is not None and clock != self.clock)):
            raise BenchmarkError("invalid_cadence_observation")
        if self.previous is not None:
            old_pts, old_ns = self.previous
            if identity.pts <= old_pts or verified_ns <= old_ns:
                raise BenchmarkError("nonmonotonic_cadence_clock")
        elif identity.pts != self.first_pts:
            raise BenchmarkError("cadence_source_start_mismatch")
        self.clock = clock
        elapsed = (identity.pts - self.first_pts) * identity.time_base
        bucket = int(elapsed // 10)
        if bucket not in self.windows:
            if self.windows:
                self.windows[next(reversed(self.windows))]["closed_by_later_source_window"] = True
            self.windows[bucket] = {
                "source_start_s": bucket * 10, "source_end_s": (bucket + 1) * 10,
                "first_sequence": identity.sequence, "first_pts": identity.pts,
                "first_ns": verified_ns, "frames": 0, "people": {}, "intervals": Samples(),
                "closed_by_later_source_window": False,
            }
        window = self.windows[bucket]
        if window["frames"]:
            window["intervals"].add((verified_ns - window["last_ns"]) / 1e6)
        window["frames"] += 1
        window["people"][str(people)] = window["people"].get(str(people), 0) + 1
        window.update(last_sequence=identity.sequence, last_pts=identity.pts, last_ns=verified_ns)
        if self.previous is not None:
            interval = (verified_ns - self.previous[1]) / 1e6
            self.intervals.add(interval)
            self.by_people.setdefault(str(people), Samples()).add(interval)
            self.streak = self.streak + 1 if interval > 1000 / 60 else 0
            self.longest_streak = max(self.longest_streak, self.streak)
            if len(self.worst) < 16 or interval > self.worst[-1]["interval_ms"]:
                self.worst.append({"sequence": identity.sequence, "pts": identity.pts,
                                   "people_at_interval_end": people, "interval_ms": interval})
                self.worst.sort(key=lambda r: r["interval_ms"], reverse=True)
                del self.worst[16:]
        self.previous = (identity.pts, verified_ns)
        self.frames += 1

    def summary(self) -> dict:
        rows = []
        for window in self.windows.values():
            span_ns = window["last_ns"] - window["first_ns"]
            rows.append({
                k: v for k, v in window.items()
                if k not in {"first_ns", "last_ns", "intervals"}
            } | {
                "observed_source_span_s": float(
                    (window["last_pts"] - window["first_pts"]) * self.clock[1]),
                "within_window_verified_rate_hz": (
                    (window["frames"] - 1) * 1e9 / span_ns if span_ns > 0 else None),
                "within_window_intervals": window["intervals"].summary(1000 / 60),
            })
        return {
            "frames": self.frames, "expected_frames": self.limit,
            "coverage": "selected_frames_observed" if self.frames == self.limit else "partial",
            "time_base": str(self.clock[1]) if self.clock else None,
            "scope": "unpaced host validation completions; not camera, display or deadlines",
            "source_windows": rows,
            "window_rate_definition": "(completions-1) / host first-to-last span in source bin",
            "window_edge_intervals": "excluded within bins; included in global intervals",
            "verified_intervals": self.intervals.summary(1000 / 60),
            "intervals_by_ending_person_count": {
                k: v.summary(1000 / 60) for k, v in self.by_people.items()},
            "person_count_attribution": "ending frame workload; not isolated inference cost",
            "longest_consecutive_over_60hz_interval_budget": self.longest_streak,
            "worst_intervals": self.worst,
            "interval_exceedances_are_dropped_frames": False,
            "live_60fps_verified": False,
        }
