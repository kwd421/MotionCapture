"""Explicit, digest-pinned downloads for the WholeBody research slice.

No inference-time downloads, executable checkpoints, mirror retries or overwrites.
Archive hashes are publisher-mirror SHA256 values, not hashes measured on this host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

import certifi

HF_REVISION = "cd4d7095f5cfc9cfc4f46289bee91ea4a1e1d9fd"
MIRROR = f"https://huggingface.co/Tau-J/RTMPose/resolve/{HF_REVISION}/"
UPSTREAM = "https://download.openmmlab.com/mmpose/v1/projects/"
DEFAULT_DIR = Path("models/wholebody")
MAX_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class Asset:
    key: str
    remote_path: str
    archive_sha256: str
    input_hw: tuple[int, int]
    role: str
    training_sources: str

    def metadata(self) -> dict:
        return {**asdict(self), "download_url": MIRROR + self.remote_path,
                "original_url": UPSTREAM + self.remote_path,
                "publisher_mirror": "Tau-J/RTMPose", "revision": HF_REVISION,
                "published_license_label": "Apache-2.0",
                "commercial_distribution_clearance": "not_audited",
                "license_note": "Repository/model-card label is not a full rights opinion."}


ASSETS = {
    "yolox-tiny": Asset(
        "yolox-tiny", "rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
        "36e09ca555916253fa1b5d51bec01e48e98e1f0af966c51ce3e844d8c8fc4cfc",
        (416, 416), "person_detector", "HumanArt + COCO"),
    "dwpose-m": Asset(
        "dwpose-m", ("rtmposev1/onnx_sdk/rtmpose-m_simcc-ucoco_dw-ucoco_270e-"
                     "256x192-c8b76419_20230728.zip"),
        "a4e4d56e9dc043e2171c73f5918f334509a47fde60182c1961370c7160334ac0",
        (256, 192), "wholebody_133_2d", "COCO-WholeBody + UBody"),
    "rtmw-m": Asset(
        "rtmw-m", "rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip",
        "a6d08a575c9d8b8da7ce6c13e1c7341047fca35594a6d35de04b739eba51139c",
        (256, 192), "wholebody_133_2d", "Cocktail14"),
    "rtmw-l": Asset(
        "rtmw-l", "rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
        "1e3e77558dfc199129bfff1c583e51b4ee190914de6ae30688243c20163c148c",
        (256, 192), "wholebody_133_2d", "Cocktail14"),
}
POSE_MODELS = tuple(k for k, v in ASSETS.items() if v.role == "wholebody_133_2d")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_stream(source, destination: Path, limit: int = MAX_BYTES) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as f:
            temp = Path(f.name)
            total = 0
            for block in iter(lambda: source.read(1024 * 1024), b""):
                total += len(block)
                if total > limit:
                    raise ValueError("Asset exceeds download/extraction limit")
                f.write(block)
            f.flush()
            os.fsync(f.fileno())
        os.link(temp, destination)  # no replacement of user-owned files
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def require_asset(asset: Asset, root: Path) -> tuple[Path, dict]:
    """Verify BOTH archive and derived ONNX; preparation is explicit in fetch()."""
    archive = root / f"{asset.key}.zip"
    model = root / f"{asset.key}.onnx"
    if not archive.is_file() or digest_file(archive) != asset.archive_sha256:
        raise ValueError(f"Missing or checksum-invalid archive: {asset.key}; run assets fetch")
    with zipfile.ZipFile(archive) as z:
        members = [x for x in z.infolist() if x.filename.endswith(".onnx")]
        if len(members) != 1:
            raise ValueError("Require exactly one embedded ONNX graph in the pinned archive")
        member = members[0]
        path = PurePosixPath(member.filename)
        if (path.is_absolute() or ".." in path.parts or "\\" in member.filename
                or not 0 < member.file_size <= MAX_BYTES):
            raise ValueError("Invalid archive member")
        digest = hashlib.sha256()
        with z.open(member) as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        expected = digest.hexdigest()
    if not model.is_file() or digest_file(model) != expected:
        raise ValueError(f"Missing or checksum-invalid extracted model: {asset.key}")
    return model, {**asset.metadata(), "onnx_sha256": expected,
                   "archive_member": member.filename, "onnx_bytes": member.file_size,
                   "hash_verification": "archive_and_extracted_graph"}


def fetch(asset: Asset, root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    archive, model = root / f"{asset.key}.zip", root / f"{asset.key}.onnx"
    if not archive.exists():
        # A failed download/checksum never publishes a usable archive.
        with tempfile.TemporaryDirectory(dir=root) as directory:
            staged = Path(directory) / "download.zip"
            context = ssl.create_default_context(cafile=certifi.where())
            with urlopen(MIRROR + asset.remote_path, timeout=60, context=context) as response:
                if not response.url.startswith("https://"):
                    raise ValueError("Insecure model redirect")
                _atomic_stream(response, staged)
            if digest_file(staged) != asset.archive_sha256:
                raise ValueError(f"Downloaded archive checksum mismatch: {asset.key}")
            os.link(staged, archive)
    if digest_file(archive) != asset.archive_sha256:
        raise ValueError(f"Existing archive checksum mismatch: {asset.key}; not overwritten")
    if not model.exists():
        with zipfile.ZipFile(archive) as z:
            members = [x for x in z.infolist() if x.filename.endswith(".onnx")]
            if len(members) != 1 or not 0 < members[0].file_size <= MAX_BYTES:
                raise ValueError("Invalid ONNX archive structure")
            name = PurePosixPath(members[0].filename)
            if name.is_absolute() or ".." in name.parts or "\\" in members[0].filename:
                raise ValueError("Invalid archive member path")
            # Stream one file into our chosen name; never extractall or load pickle.
            with z.open(members[0]) as f:
                _atomic_stream(f, model)
    return require_asset(asset, root)[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "fetch", "verify"))
    parser.add_argument("--models", nargs="+", choices=POSE_MODELS, default=list(POSE_MODELS))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    keys = ["yolox-tiny", *dict.fromkeys(args.models)]
    try:
        rows = []
        for key in keys:
            asset = ASSETS[key]
            print(f"{args.action}: {key}", flush=True)
            rows.append(asset.metadata() if args.action == "list" else
                        fetch(asset, args.model_dir) if args.action == "fetch" else
                        require_asset(asset, args.model_dir)[1])
        print(json.dumps({"status": "completed", "assets": rows}, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
