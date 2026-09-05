"""Pinned model ownership, integrity verification, and explicit download CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import certifi

from motioncapture.errors import ModelAssetError


@dataclass(frozen=True, slots=True)
class ModelAsset:
    key: str
    name: str
    filename: str
    url: str
    sha256: str


MODEL_ASSETS = (
    ModelAsset(
        key="pose",
        name="pose_landmarker_full_float16_v1",
        filename="pose_landmarker_full.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/1/pose_landmarker_full.task"
        ),
        sha256="5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1",
    ),
    ModelAsset(
        key="hands",
        name="hand_landmarker_float16_v1",
        filename="hand_landmarker.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        ),
        sha256="fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    ),
    ModelAsset(
        key="face",
        name="face_landmarker_float16_v1",
        filename="face_landmarker.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task"
        ),
        sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
)

DEFAULT_MODEL_DIR = Path("models")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_model(asset: ModelAsset, path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ModelAssetError(
            f"Pinned {asset.key} model is missing: {resolved}. "
            "Run `uv run motioncapture-models fetch` explicitly."
        )
    actual = sha256_file(resolved)
    if actual != asset.sha256:
        raise ModelAssetError(
            f"{asset.key} model checksum mismatch: "
            f"expected={asset.sha256} actual={actual} path={resolved}"
        )
    return resolved


def require_models(model_dir: Path) -> dict[str, Path]:
    return {
        asset.key: require_model(asset, model_dir / asset.filename)
        for asset in MODEL_ASSETS
    }


def fetch_model(asset: ModelAsset, path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        return require_model(asset, resolved)

    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        tls_context = ssl.create_default_context(cafile=certifi.where())
        with urlopen(asset.url, timeout=60, context=tls_context) as response:  # noqa: S310
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f"{asset.key}-",
                suffix=".partial",
                dir=resolved.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                while chunk := response.read(1024 * 1024):
                    temporary_file.write(chunk)
        actual = sha256_file(temporary_path)
        if actual != asset.sha256:
            raise ModelAssetError(
                f"Downloaded {asset.key} model checksum mismatch: "
                f"expected={asset.sha256} actual={actual}"
            )
        os.replace(temporary_path, resolved)
        temporary_path = None
    except (OSError, URLError) as exc:
        raise ModelAssetError(f"Unable to download pinned {asset.key} model: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return require_model(asset, resolved)


def fetch_models(model_dir: Path) -> dict[str, Path]:
    return {
        asset.key: fetch_model(asset, model_dir / asset.filename)
        for asset in MODEL_ASSETS
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch", help="download and verify all pinned models")
    fetch.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    verify = subparsers.add_parser("verify", help="verify all existing models")
    verify.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        paths = (
            fetch_models(args.model_dir)
            if args.command == "fetch"
            else require_models(args.model_dir)
        )
    except ModelAssetError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "models": [
                    {
                        "key": asset.key,
                        "name": asset.name,
                        "path": str(paths[asset.key]),
                        "sha256": asset.sha256,
                        "source": asset.url,
                    }
                    for asset in MODEL_ASSETS
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
