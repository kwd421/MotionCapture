"""Bounded latency diagnostics; never stores frames, landmarks, or per-frame traces."""

from __future__ import annotations

import math
import subprocess
from array import array
from pathlib import Path
from typing import Any

from motioncapture.session import LatencySummary, SessionRecord


class ProfiledLatency(LatencySummary):
    """Quarter-ms histogram to 1000ms; overflow percentiles remain unknown.

    Mean/max share the original SessionRecord accumulator. Histogram counts are
    bounded in memory independently of session length. Percentiles are upper
    bucket bounds, not exact observations.
    """

    __slots__ = ("bins", "count", "overflow", "budget_ms", "over_budget")
    resolution_ms = 0.25
    limit_ms = 1000.0

    def __init__(self, budget_ms: float) -> None:
        super().__init__()
        if not math.isfinite(budget_ms) or budget_ms <= 0:
            raise ValueError("Frame budget must be positive and finite")
        self.bins = array("Q", [0]) * (int(self.limit_ms / self.resolution_ms) + 1)
        self.count = self.overflow = self.over_budget = 0
        self.budget_ms = budget_ms

    def observe(self, duration_ms: float) -> None:
        if not math.isfinite(duration_ms) or duration_ms < 0:
            raise ValueError("Latency must be finite and nonnegative")
        super().observe(duration_ms)
        self.count += 1
        self.over_budget += int(duration_ms > self.budget_ms)
        if duration_ms > self.limit_ms:
            self.overflow += 1
        else:
            self.bins[math.ceil(duration_ms / self.resolution_ms)] += 1

    def percentile_upper(self, fraction: float) -> float | None:
        if not 0 < fraction <= 1:
            raise ValueError("Percentile fraction must be in (0, 1]")
        if not self.count:
            return None
        rank = math.ceil(fraction * self.count)
        cumulative = 0
        for index, count in enumerate(self.bins):
            cumulative += count
            if cumulative >= rank:
                return index * self.resolution_ms
        return None  # The requested rank lies above the measured histogram range.

    def distribution(self) -> dict[str, object]:
        return {
            **super().payload(self.count),
            "samples": self.count,
            "p50_upper_ms": self.percentile_upper(0.50),
            "p95_upper_ms": self.percentile_upper(0.95),
            "p99_upper_ms": self.percentile_upper(0.99),
            "percentile_kind": "histogram_upper_bound",
            "resolution_ms": self.resolution_ms,
            "limit_ms": self.limit_ms,
            "overflow_samples": self.overflow,
            "frame_budget_ms": self.budget_ms,
            "over_budget_samples": self.over_budget,
        }


def revision_metadata(root: Path) -> dict[str, object]:
    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(root), *arguments], capture_output=True, text=True,
            timeout=3, check=False,
        )
    try:
        revision = run("rev-parse", "HEAD")
        dirty = run("status", "--porcelain", "--untracked-files=normal")
    except (OSError, subprocess.TimeoutExpired):
        return {"revision": None, "working_tree_dirty": None, "status": "unknown"}
    valid = revision.returncode == 0 and dirty.returncode == 0
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "working_tree_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "status": "measured" if valid else "unknown",
    }


class PerformanceSession(SessionRecord):
    """Single caller owns the record. The inference worker never mutates it."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        budget = 1000.0 / self.camera_request.fps
        self.latencies = {name: ProfiledLatency(budget) for name in self.latencies}
        self.capture_queue_latency = ProfiledLatency(budget)
        self.extra = {
            name: ProfiledLatency(budget) for name in (
                "capture_wait", "result_wait", "result_residence", "presentation_submit",
                "event_pump", "host_receive_to_gui_submit", "host_receive_to_event_pump",
                "iteration_wall",
            )
        }
        self.pipeline_snapshot: dict[str, object] | None = None
        self.pipeline_scheduling = "unknown"
        self.preview_mode = "unknown"
        self.headless = False
        self.source: dict[str, object] = {"status": "unknown"}
        self.first_completion_ns: int | None = None
        self.last_completion_ns: int | None = None

    def completed_at(self, timestamp_ns: int) -> None:
        if self.last_completion_ns is not None and timestamp_ns <= self.last_completion_ns:
            raise ValueError("Nonmonotonic completed-frame time")
        if self.first_completion_ns is None:
            self.first_completion_ns = timestamp_ns
        self.last_completion_ns = timestamp_ns

    def payload(self) -> dict[str, Any]:
        payload = super().payload()
        payload["mode"].update({
            "pipeline_scheduling": self.pipeline_scheduling,
            "preview_mode": self.preview_mode,
            "headless": self.headless,
        })
        payload["application"]["source"] = self.source
        # Additive extension keeps every schema-v2 field and timing definition.
        payload["performance"] = {
            "schema_version": 1,
            "bundle_version": "live-performance-v1",
            "pipeline": self.pipeline_snapshot,
            "stage_distributions": {
                name: latency.distribution() for name, latency in self.latencies.items()
            },
            "capture_queue": self.capture_queue_latency.distribution(),
            "additional_stages": {
                name: latency.distribution() for name, latency in self.extra.items()
            },
            "presentation_requested": not self.headless,
            "presentation_measured": (
                self.extra["presentation_submit"].count > 0
                and self.extra["event_pump"].count > 0
            ),
            "sensor_to_photon_measured": False,
            "gui_timing_scope": "host API call/event processing, not physical display",
            "delivered_completion_fps": (
                (self.frames - 1) * 1_000_000_000
                / (self.last_completion_ns - self.first_completion_ns)
                if self.frames >= 2 and self.first_completion_ns is not None
                and self.last_completion_ns is not None
                and self.last_completion_ns > self.first_completion_ns else None
            ),
        }
        return payload
