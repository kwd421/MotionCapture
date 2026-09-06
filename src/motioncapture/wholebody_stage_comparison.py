"""Versioned comparison contract for the stage experiment, independent of the lab CLI.

This module does not import the user's uncommitted provider-comparison changes.
Only stable Person2D and recorded frame fields cross this boundary.
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np

from motioncapture.wholebody_catalog import BenchmarkError
from motioncapture.wholebody_onnx import PARTS


class StageReference:
    def __init__(self, count, *, frame_hashes=False):
        self.xy = np.full((count, 133, 2), np.nan, np.float64)
        self.valid = np.zeros((count, 133), np.bool_)
        self.people = np.full(count, -1, np.int16)
        self.pts = np.zeros(count, np.int64)
        self.clock = None
        self.predictions_sha256 = None
        self.frame_hashes = np.empty((count, 32), np.uint8) if frame_hashes else None
        self.box_hashes = np.empty((count, 32), np.uint8) if frame_hashes else None
        self.box_seen = np.zeros(count, np.bool_) if frame_hashes else None

    def store(self, frame, people, *, boxes=None):
        identity = frame.identity
        i = identity.sequence
        clock = (identity.source_id, identity.time_base)
        if not 0 <= i < len(self.people) or self.people[i] != -1:
            raise BenchmarkError("invalid_reference_sequence")
        if self.clock is not None and self.clock != clock:
            raise BenchmarkError("reference_source_changed")
        if self.frame_hashes is not None:
            digest = hashlib.sha256()
            update_digest(digest, frame, people)
            self.frame_hashes[i] = np.frombuffer(digest.digest(), np.uint8)
            if boxes is not None:
                self.box_hashes[i] = np.frombuffer(_box_digest(boxes, len(people)), np.uint8)
                self.box_seen[i] = True
        self.clock = clock
        self.people[i], self.pts[i] = len(people), identity.pts
        if len(people) == 1:
            self.xy[i], self.valid[i] = people[0].xy, people[0].valid


class StageDifference:
    def __init__(self):
        self.values = {name: [] for name in PARTS}
        self.only_reference = dict.fromkeys(PARTS, 0)
        self.only_candidate = dict.fromkeys(PARTS, 0)
        self.checked = self.unavailable = self.ambiguous = 0
        self.worst = []
        self.hash_checked = self.hash_changed = self.hash_multi_checked = 0
        self.hash_multi_changed = 0
        self.first_hash_changes = []
        self.box_checked = self.box_changed = self.box_count_changed = 0
        self.first_box_changes = []

    def add(self, reference, frame, people, *, boxes=None):
        i = frame.identity.sequence
        if reference is None or not 0 <= i < len(reference.people) or reference.people[i] < 0:
            self.unavailable += 1
            return
        if (reference.pts[i] != frame.identity.pts or reference.clock !=
                (frame.identity.source_id, frame.identity.time_base)):
            raise BenchmarkError("reference_identity_mismatch")
        self.checked += 1
        if reference.frame_hashes is not None:
            digest = hashlib.sha256()
            update_digest(digest, frame, people)
            changed = digest.digest() != reference.frame_hashes[i].tobytes()
            multi = reference.people[i] > 1 or len(people) > 1
            self.hash_checked += 1
            self.hash_changed += int(changed)
            self.hash_multi_checked += int(multi)
            self.hash_multi_changed += int(multi and changed)
            if changed and len(self.first_hash_changes) < 16:
                self.first_hash_changes.append({"sequence": i, "pts": frame.identity.pts,
                                               "reference_people": int(reference.people[i]),
                                               "candidate_people": len(people)})
        if boxes is not None and reference.box_hashes is not None and reference.box_seen[i]:
            changed = _box_digest(boxes, len(people)) != reference.box_hashes[i].tobytes()
            count_changed = reference.people[i] != len(people)
            self.box_checked += 1
            self.box_changed += int(changed)
            self.box_count_changed += int(count_changed)
            if changed and len(self.first_box_changes) < 16:
                self.first_box_changes.append({"sequence": i, "pts": frame.identity.pts,
                                               "count_changed": bool(count_changed)})
        if reference.people[i] > 1 or len(people) > 1:
            self.ambiguous += 1
            return
        a = reference.valid[i] if reference.people[i] else np.zeros(133, bool)
        b = people[0].valid if people else np.zeros(133, bool)
        common = a & b
        distances = np.full(133, np.nan, np.float64)
        if common.any():
            distances[common] = np.linalg.norm(reference.xy[i, common]-people[0].xy[common], axis=1)
            maximum = float(distances[common].max())
            if maximum > 0:
                self.worst.append({"sequence": i, "pts": frame.identity.pts,
                                   "max_pixel_disagreement": maximum})
                self.worst.sort(key=lambda row: row["max_pixel_disagreement"], reverse=True)
                del self.worst[16:]
        for name, (start, end) in PARTS.items():
            self.only_reference[name] += int((a[start:end] & ~b[start:end]).sum())
            self.only_candidate[name] += int((b[start:end] & ~a[start:end]).sum())
            if common[start:end].any():
                self.values[name].append(distances[start:end][common[start:end]].copy())

    def summary(self):
        parts = {}
        for name, arrays in self.values.items():
            value = np.concatenate(arrays) if arrays else np.empty(0)
            parts[name] = {"matched_points": len(value),
                "mean_pixels": float(value.mean()) if len(value) else None,
                "p95_pixels": float(np.quantile(value, .95)) if len(value) else None,
                "p99_pixels": float(np.quantile(value, .99)) if len(value) else None,
                "max_pixels": float(value.max()) if len(value) else None,
                "above_1px": int((value > 1).sum()), "above_10px": int((value > 10).sum())}
        return {"checked_frames": self.checked, "reference_unavailable_frames": self.unavailable,
                "ambiguous_multi_person_frames": self.ambiguous, "parts": parts,
                "reference_only_point_observations": self.only_reference,
                "candidate_only_point_observations": self.only_candidate,
                "worst_frames": self.worst, "coordinate_delta_lower_is_better": True,
                "accuracy_verified": False,
                "per_frame_detector_boxes": {
                    "checked_frames": self.box_checked, "changed_frames": self.box_changed,
                    "count_mismatch_frames": self.box_count_changed,
                    "first_changes": self.first_box_changes,
                    "scope": "ordered detector-slot float32 edges; not actor correspondence",
                },
                "per_frame_predictions": {
                    "status": "compared" if self.hash_checked else "not_requested_or_no_samples",
                    "checked_frames": self.hash_checked, "changed_frames": self.hash_changed,
                    "multi_person_frames_checked": self.hash_multi_checked,
                    "multi_person_frames_changed": self.hash_multi_changed,
                    "first_changes": self.first_hash_changes,
                    "scope": "all slots, scores, validity and PTS; bitwise, not actor accuracy"}}


def _box_digest(boxes, people_count):
    if (not isinstance(boxes, np.ndarray) or boxes.shape != (people_count, 4)
            or boxes.dtype != np.float32 or not np.isfinite(boxes).all()):
        raise BenchmarkError("invalid_detector_box_comparison")
    return hashlib.sha256(np.asarray(boxes, dtype="<f4").tobytes()).digest()


def update_digest(digest, frame, people):
    """Use the existing v1 framing, preserving original rational time and missingness."""
    identity = frame.identity
    digest.update(b"mocap-wholebody133-v1\0")
    digest.update(struct.pack("<QqqqI", identity.sequence, identity.pts,
                              identity.time_base.numerator, identity.time_base.denominator,
                              len(people)))
    for person in people:
        digest.update(np.asarray(person.xy, dtype="<f8").tobytes())
        digest.update(np.asarray(person.scores, dtype="<f8").tobytes())
        digest.update(person.valid.tobytes())
