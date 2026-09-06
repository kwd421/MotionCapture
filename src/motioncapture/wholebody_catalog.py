"""Explicit pretrained assets. First enrollment is TOFU, never license clearance."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import ssl
import stat
import tempfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import certifi


class BenchmarkError(RuntimeError):
    """A fixed public code; never store arbitrary exception messages in reports."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Asset:
    key: str
    url: str
    shape: tuple[int, int, int, int]
    kind: str
    evidence: str
    training: str
    code_license: str = "Apache-2.0"
    weights_and_data_clearance: str = "unreviewed"
    publisher_sha256: str | None = None

    def payload(self) -> dict:
        return asdict(self)


SDK = "https://download.openmmlab.com/mmpose/v1/projects/"
RTMLIB = "https://github.com/Tau-J/rtmlib/blob/03a1693e59e4f7cd84582c0fb30459b3bf18ad42/"
ASSETS = {
    "yolox-tiny": Asset(
        "yolox-tiny",
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx",
        (1, 3, 416, 416), "yolox_raw_coco80", RTMLIB + "README.md", "COCO"),
    "dwpose-m": Asset(
        "dwpose-m", SDK + "rtmposev1/onnx_sdk/"
        "rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.zip",
        (1, 3, 256, 192), "simcc133", RTMLIB + "README.md", "COCO-WholeBody + UBody"),
    "rtmw-m": Asset(
        "rtmw-m", SDK + "rtmw/onnx_sdk/"
        "rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip",
        (1, 3, 256, 192), "simcc133",
        RTMLIB + "rtmlib/tools/solution/wholebody.py", "Cocktail14; per-dataset review pending"),
    "rtmw-l": Asset(
        "rtmw-l", SDK + "rtmw/onnx_sdk/"
        "rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
        (1, 3, 256, 192), "simcc133", RTMLIB + "README.md",
        "Cocktail14; per-dataset review pending"),
}
CANDIDATES = ("dwpose-m", "rtmw-m", "rtmw-l")
MAX_BYTES = 512 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def catalog_digest() -> str:
    data = {k: v.payload() for k, v in ASSETS.items()}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _download(url: str, destination: Path) -> None:
    # No mirrors or HTTP downgrades; GitHub's official asset CDN is permitted.
    allowed = {"download.openmmlab.com", "github.com", "release-assets.githubusercontent.com",
               "objects.githubusercontent.com"}

    class Redirects(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            parsed = urlparse(newurl)
            if parsed.scheme != "https" or parsed.hostname not in allowed:
                raise BenchmarkError("unapproved_asset_redirect")
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    context = ssl.create_default_context(cafile=certifi.where())
    opener = urllib.request.build_opener(Redirects(), urllib.request.HTTPSHandler(context=context))
    request = urllib.request.Request(url, headers={"User-Agent": "MotionCapture-Research/1"})
    with opener.open(request, timeout=120) as response, destination.open("xb") as target:
        size = 0
        while block := response.read(1024 * 1024):
            size += len(block)
            if size > MAX_BYTES:
                raise BenchmarkError("asset_size_limit")
            target.write(block)
        if size == 0:
            raise BenchmarkError("empty_asset")


def _extract_onnx(archive: Path, target: Path) -> str | None:
    """Extract only the single ONNX payload, never paths, scripts or pickle files."""
    if not zipfile.is_zipfile(archive):
        shutil.copyfile(archive, target)
        return None
    with zipfile.ZipFile(archive) as packed:
        members = packed.infolist()
        if len(members) > 256 or sum(x.file_size for x in members) > MAX_BYTES:
            raise BenchmarkError("archive_size_limit")
        for member in members:
            p = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if (p.is_absolute() or ".." in p.parts or "\\" in member.filename
                    or stat.S_ISLNK(mode)):
                raise BenchmarkError("unsafe_asset_archive")
        models = [x for x in members if x.filename.lower().endswith(".onnx")]
        if len(models) != 1:
            raise BenchmarkError("expected_exactly_one_onnx")
        with packed.open(models[0]) as source, target.open("xb") as output:
            shutil.copyfileobj(source, output)
        return models[0].filename


def verify_asset(root: Path, key: str) -> tuple[Path, dict]:
    asset = ASSETS[key]
    directory = root / key
    try:
        receipt = json.loads((directory / "asset-lock.json").read_text())
        model = directory / "model.onnx"
        if (directory.is_symlink() or model.is_symlink()
                or (directory / "asset-lock.json").is_symlink()):
            raise BenchmarkError("asset_symlink_not_allowed")
        if (receipt["catalog_sha256"] != catalog_digest() or receipt["asset"] !=
                json.loads(json.dumps(asset.payload()))):
            raise BenchmarkError("asset_catalog_mismatch")
        if sha256(model) != receipt["model_sha256"]:
            raise BenchmarkError("asset_checksum_mismatch")
        if receipt.get("commercial_release_cleared") is not False:
            raise BenchmarkError("invalid_license_receipt")
        return model, receipt
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise BenchmarkError("asset_missing_or_invalid_lock") from exc


def fetch_assets(root: Path, models: list[str], *, research_only: bool) -> list[dict]:
    if not research_only:
        raise BenchmarkError("explicit_research_terms_acknowledgement_required")
    keys = list(dict.fromkeys(["yolox-tiny", *models]))
    if any(key not in ASSETS for key in keys):
        raise BenchmarkError("unknown_asset")
    root.mkdir(parents=True, exist_ok=True)
    receipts = []
    for key in keys:
        final = root / key
        if final.exists():
            receipts.append(verify_asset(root, key)[1])
            continue
        asset = ASSETS[key]
        temporary = Path(tempfile.mkdtemp(prefix=".enroll-", dir=root))
        try:
            archive = temporary / "download.bin"
            _download(asset.url, archive)
            archive_hash = sha256(archive)
            if asset.publisher_sha256 is not None and archive_hash != asset.publisher_sha256:
                raise BenchmarkError("publisher_checksum_mismatch")
            member = _extract_onnx(archive, temporary / "model.onnx")
            receipt = {
                "schema_version": 1, "asset": asset.payload(), "catalog_sha256": catalog_digest(),
                "download_sha256": archive_hash, "archive_member": member,
                "model_sha256": sha256(temporary / "model.onnx"),
                "hash_authority": "publisher" if asset.publisher_sha256 else "first_download_TOFU",
                "publisher_authentication_verified": asset.publisher_sha256 is not None,
                "research_only_acknowledged": True, "commercial_release_cleared": False,
            }
            (temporary / "asset-lock.json").write_text(json.dumps(receipt, indent=2) + "\n")
            archive.unlink()
            # Exclusive directory reservation prevents overwriting concurrent/user-owned assets.
            final.mkdir()
            try:
                for name in ("model.onnx", "asset-lock.json"):
                    os.replace(temporary / name, final / name)
            except BaseException:
                # Only our just-created directory is removed, never an existing asset directory.
                shutil.rmtree(final)
                raise
            receipts.append(receipt)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    return receipts
