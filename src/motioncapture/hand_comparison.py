"""In-memory reference comparison. Aggregate disagreement is NOT accuracy."""
from __future__ import annotations

import numpy as np

FIELDS = (("left_hand_landmarks", "left_hand_world_landmarks"),
          ("right_hand_landmarks", "right_hand_world_landmarks"))


class HandReference:
    def __init__(self, pts, width, height):
        if len(pts) * 2 * 21 * 6 * 8 > 256 * 1024 * 1024:
            raise ValueError("Reference exceeds the explicit 256 MiB bound")
        self.pts = tuple(pts)
        self.width, self.height = width, height
        self.present = np.zeros((len(pts), 2), dtype=bool)
        self.values = np.full((len(pts), 2, 21, 6), np.nan, dtype=np.float64)
        self.count = 0

    @staticmethod
    def unpack(result):
        present = np.zeros(2, dtype=bool)
        values = np.full((2, 21, 6), np.nan, dtype=np.float64)
        for slot, (image_name, world_name) in enumerate(FIELDS):
            image, world = getattr(result, image_name), getattr(result, world_name)
            if bool(image) != bool(world) or len(image) not in (0, 21) or len(world) != len(image):
                raise ValueError("Incomplete image/world hand pair")
            if image:
                present[slot] = True
                values[slot] = [[p.x, p.y, p.z, w.x, w.y, w.z] for p, w in zip(image, world, strict=True)]
                if not np.isfinite(values[slot]).all():
                    raise ValueError("Nonfinite hand reference")
        return present, values

    def _check(self, frame, expected):
        i = frame.identity.sequence
        if i != expected or i >= len(self.pts) or frame.identity.pts != self.pts[i]:
            raise ValueError("Reference sequence/PTS mismatch")
        return i

    def record(self, frame, result):
        i = self._check(frame, self.count)
        self.present[i], self.values[i] = self.unpack(result)
        self.count += 1

    def comparator(self):
        if self.count != len(self.pts):
            raise ValueError("Reference is not complete")
        return HandComparison(self)


class HandComparison:
    def __init__(self, reference):
        self.reference = reference
        self.count = 0
        self.reference_only = np.zeros(2, dtype=np.int64)
        self.candidate_only = np.zeros(2, dtype=np.int64)
        self.matched = np.zeros(2, dtype=np.int64)
        n = len(reference.pts)
        self.pixel_errors = np.full((n, 2, 21), np.nan)
        self.world_errors = np.full((n, 2, 21), np.nan)

    def observe(self, frame, result):
        r = self.reference
        i = r._check(frame, self.count)
        present, values = r.unpack(result)
        before = r.present[i]
        self.reference_only += before & ~present
        self.candidate_only += present & ~before
        matched = present & before
        self.matched += matched
        if matched.any():
            diff = values[matched] - r.values[i, matched]
            self.pixel_errors[i, matched] = np.linalg.norm(
                diff[:, :, :2] * [r.width, r.height], axis=-1)
            self.world_errors[i, matched] = np.linalg.norm(diff[:, :, 3:] * 1000, axis=-1)
        self.count += 1

    def summary(self, *, allow_partial=False):
        if not allow_partial and self.count != len(self.reference.pts):
            raise ValueError("Incomplete comparison")

        def stats(values):
            a = values[np.isfinite(values)]
            return {"matched_points": len(a), "mean": float(a.mean()) if len(a) else None,
                    "p50": float(np.quantile(a, .5)) if len(a) else None,
                    "p95": float(np.quantile(a, .95)) if len(a) else None,
                    "p99": float(np.quantile(a, .99)) if len(a) else None,
                    "max": float(a.max()) if len(a) else None}
        # Store only locating metadata and aggregate displacement, never coordinates.
        errors = self.pixel_errors[:self.count]
        rows = np.flatnonzero(np.isfinite(errors).any(axis=(1, 2)))
        worst = []
        if len(rows):
            maxima = np.max(np.where(np.isfinite(errors[rows]), errors[rows], -np.inf), axis=(1, 2))
            for at in np.argsort(-maxima, kind="stable")[:16]:
                i = int(rows[at])
                worst.append({"sequence": i, "pts": self.reference.pts[i],
                              "max_xy_displacement_pixels": float(maxima[at])})
        return {"frames": self.count, "slot_order": ["left", "right"],
                "comparison_complete": self.count == len(self.reference.pts),
                "largest_disagreement_frames": worst,
                "outlier_locations_limit": 16,
                "reference_only_frames": self.reference_only.tolist(),
                "candidate_only_frames": self.candidate_only.tolist(),
                "matched_slot_frames": self.matched.tolist(),
                "image_xy_displacement_pixels": stats(self.pixel_errors),
                "hand_relative_world_displacement_mm": stats(self.world_errors),
                "accuracy_verified": False, "quality_verdict": "requires_review",
                "reference_is_ground_truth": False,
                "matching": "subject-relative handedness slots, no geometric relabeling"}
