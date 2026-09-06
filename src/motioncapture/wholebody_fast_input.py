"""Opt-in pixel-exact uint8 normalization, with unchanged affine crop geometry.

The recipe is the existing wholebody_onnx.pose_tensor / RTMLib contract.
Only normalization's implementation changes: 256 entries per channel replace
per-pixel float64 subtract/divide/cast. No geometry rounding or ROI stabilization.
"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np

from motioncapture.wholebody_catalog import BenchmarkError
from motioncapture.wholebody_onnx import pose_tensor

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float64)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float64)
TABLE = ((np.arange(256, dtype=np.float64)[:, None] - MEAN) / STD).astype(np.float32)
TABLE.flags.writeable = False
MEAN.flags.writeable = STD.flags.writeable = False


def array_hash(value: np.ndarray) -> str:
    """Frame-independent dtype/shape/content hash; no array values are serialized."""
    h = hashlib.sha256()
    h.update(value.dtype.str.encode() + b"\0" + str(value.shape).encode() + b"\0")
    h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def normalize(crop: np.ndarray) -> np.ndarray:
    if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3:
        raise BenchmarkError("normalization_requires_uint8_bgr")
    height, width = crop.shape[:2]
    result = np.empty((1, 3, height, width), np.float32)
    # np.take(out=...) avoids an intermediate HWC floating-point allocation.
    for channel in range(3):
        np.take(TABLE[:, channel], crop[:, :, channel], out=result[0, channel])
    return result


def crop_geometry(image: np.ndarray, bbox: np.ndarray, size: tuple[int, int]):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2):
        raise BenchmarkError("invalid_source_image")
    if (bbox.shape != (4,) or not np.isfinite(bbox).all()
            or np.any(bbox[2:] <= bbox[:2])):
        raise BenchmarkError("invalid_person_box")
    if len(size) != 2 or min(size) <= 0:
        raise BenchmarkError("invalid_pose_size")
    width, height = size
    center = (bbox[:2] + bbox[2:]) * .5
    scale = (bbox[2:] - bbox[:2]) * 1.25
    scale = np.array((max(scale[0], scale[1] * width / height),
                      max(scale[1], scale[0] * height / width)))
    src = np.float32([center, center + [0, -scale[0] / 2],
                      center + [-scale[0] / 2, -scale[0] / 2]])
    dst = np.float32([[width / 2, height / 2], [width / 2, (height - width) / 2],
                      [0, (height - width) / 2]])
    matrix = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(image, matrix, size, flags=cv2.INTER_LINEAR)
    return crop, center, scale, matrix


def fast_pose_tensor(image: np.ndarray, bbox: np.ndarray, size: tuple[int, int]):
    crop, center, scale, _ = crop_geometry(image, bbox, size)
    return normalize(crop), center, scale


def check_recipe(image: np.ndarray, bbox: np.ndarray, size: tuple[int, int]) -> None:
    """Fail on source-recipe drift. Run outside measured loops; not a fallback."""
    actual = fast_pose_tensor(image, bbox, size)
    expected = pose_tensor(image, bbox, size)
    if any(a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes()
           for a, b in zip(actual, expected, strict=True)):
        raise BenchmarkError("fast_preprocessing_recipe_mismatch")
