"""Fetch and validate the pinned Live2D and VRM test avatars."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import ssl
import struct
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.error import URLError
from urllib.request import urlopen

import certifi

from motioncapture.errors import AvatarAssetError

DEFAULT_AVATAR_DIR = Path("assets/avatars")

LIVE2D_COMMIT = "b1de66b0b1f1cb881d95fb6158622aeb6a2827bd"
LIVE2D_ARCHIVE_URL = (
    "https://codeload.github.com/Live2D/CubismWebSamples/zip/" + LIVE2D_COMMIT
)
LIVE2D_ARCHIVE_SHA256 = "185facea379f29e19bcf99b3dac4e84c1674170b93258c3537d6577780156f1e"
LIVE2D_MODEL_RELATIVE_PATH = Path("live2d/haru/Haru.model3.json")

VRM_COMMIT = "821c11b250d8c70d5804ee13431e42bee56ea9c0"
VRM_URL = (
    "https://raw.githubusercontent.com/vrm-c/vrm-specification/"
    f"{VRM_COMMIT}/samples/Seed-san/vrm/Seed-san.vrm"
)
VRM_SHA256 = "624d0d554bc205bbdc33e22a68a2c3c20edebb3e573011ead8878a65e5329b23"
VRM_MODEL_RELATIVE_PATH = Path("vrm/seed-san/Seed-san.vrm")

REQUIRED_VRM_BONES = frozenset(
    {
        "hips",
        "spine",
        "chest",
        "neck",
        "head",
        "leftUpperArm",
        "leftLowerArm",
        "leftHand",
        "rightUpperArm",
        "rightLowerArm",
        "rightHand",
        "leftUpperLeg",
        "leftLowerLeg",
        "leftFoot",
        "leftToes",
        "rightUpperLeg",
        "rightLowerLeg",
        "rightFoot",
        "rightToes",
        *{
            f"{side}{finger}{segment}"
            for side in ("left", "right")
            for finger in ("Index", "Middle", "Ring", "Little")
            for segment in ("Proximal", "Intermediate", "Distal")
        },
        *{
            f"{side}Thumb{segment}"
            for side in ("left", "right")
            for segment in ("Metacarpal", "Proximal", "Distal")
        },
    }
)
REQUIRED_VRM_EXPRESSIONS = frozenset(
    {
        "neutral",
        "happy",
        "angry",
        "sad",
        "relaxed",
        "surprised",
        "blinkLeft",
        "blinkRight",
        "lookUp",
        "lookDown",
        "lookLeft",
        "lookRight",
        "aa",
        "ih",
        "ou",
        "ee",
        "oh",
    }
)


@dataclass(frozen=True, slots=True)
class AvatarStatus:
    key: str
    name: str
    format: str
    path: Path
    source: str
    source_revision: str
    sha256: str
    details: dict[str, object]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as asset_file:
        for chunk in iter(lambda: asset_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_verified(url: str, expected_sha256: str, directory: Path, prefix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        tls_context = ssl.create_default_context(cafile=certifi.where())
        with urlopen(url, timeout=120, context=tls_context) as response:  # noqa: S310
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=prefix,
                suffix=".partial",
                dir=directory,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                while chunk := response.read(1024 * 1024):
                    temporary_file.write(chunk)
        actual_sha256 = _sha256_file(temporary_path)
        if actual_sha256 != expected_sha256:
            raise AvatarAssetError(
                "Downloaded avatar checksum mismatch: "
                f"expected={expected_sha256} actual={actual_sha256} source={url}"
            )
        return temporary_path
    except AvatarAssetError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    except (OSError, URLError) as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise AvatarAssetError(f"Unable to download pinned avatar source {url}: {exc}") from exc


def _live2d_references(document: dict[str, object]) -> set[Path]:
    references = document.get("FileReferences")
    if not isinstance(references, dict):
        raise AvatarAssetError("Haru.model3.json has no FileReferences object")

    paths: set[Path] = set()
    for key in ("Moc", "Physics", "Pose", "DisplayInfo", "UserData"):
        value = references.get(key)
        if isinstance(value, str):
            paths.add(Path(value))

    textures = references.get("Textures", [])
    if not isinstance(textures, list):
        raise AvatarAssetError("Haru.model3.json Textures must be a list")
    paths.update(Path(value) for value in textures if isinstance(value, str))

    expressions = references.get("Expressions", [])
    if not isinstance(expressions, list):
        raise AvatarAssetError("Haru.model3.json Expressions must be a list")
    for expression in expressions:
        if isinstance(expression, dict) and isinstance(expression.get("File"), str):
            paths.add(Path(expression["File"]))

    motions = references.get("Motions", {})
    if not isinstance(motions, dict):
        raise AvatarAssetError("Haru.model3.json Motions must be an object")
    for motion_group in motions.values():
        if not isinstance(motion_group, list):
            continue
        for motion in motion_group:
            if not isinstance(motion, dict):
                continue
            for key in ("File", "Sound"):
                value = motion.get(key)
                if isinstance(value, str):
                    paths.add(Path(value))
    return paths


def validate_live2d(model_path: Path) -> AvatarStatus:
    resolved = model_path.expanduser().resolve()
    if not resolved.is_file():
        raise AvatarAssetError(
            f"Pinned Live2D avatar is missing: {resolved}. "
            "Run `uv run motioncapture-avatars fetch` explicitly."
        )
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AvatarAssetError(f"Invalid Live2D model manifest {resolved}: {exc}") from exc
    if document.get("Version") != 3:
        raise AvatarAssetError(
            f"Unsupported Live2D model manifest version: {document.get('Version')!r}"
        )

    references = _live2d_references(document)
    unsafe = sorted(str(path) for path in references if path.is_absolute() or ".." in path.parts)
    if unsafe:
        raise AvatarAssetError(f"Unsafe Live2D referenced paths: {unsafe}")
    missing = sorted(str(path) for path in references if not (resolved.parent / path).is_file())
    if missing:
        raise AvatarAssetError(f"Live2D avatar has missing referenced files: {missing}")

    groups = document.get("Groups", [])
    group_names = {
        group.get("Name")
        for group in groups
        if isinstance(group, dict) and isinstance(group.get("Name"), str)
    }
    required_groups = {"EyeBlink", "LipSync"}
    if missing_groups := sorted(required_groups - group_names):
        raise AvatarAssetError(f"Live2D avatar has missing parameter groups: {missing_groups}")

    return AvatarStatus(
        key="live2d_haru",
        name="Haru",
        format="Live2D Cubism model3",
        path=resolved,
        source=LIVE2D_ARCHIVE_URL,
        source_revision=LIVE2D_COMMIT,
        sha256=_sha256_file(resolved),
        details={
            "manifest_version": document["Version"],
            "referenced_files": len(references),
            "parameter_groups": sorted(group_names),
        },
    )


def _vrm_json_document(vrm_path: Path) -> dict[str, object]:
    data = vrm_path.read_bytes()
    if len(data) < 20:
        raise AvatarAssetError(f"VRM file is truncated: {vrm_path}")
    magic, gltf_version, declared_length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or gltf_version != 2 or declared_length != len(data):
        raise AvatarAssetError(
            "Invalid VRM/glTF header: "
            f"magic={magic!r} version={gltf_version} "
            f"declared_length={declared_length} actual_length={len(data)}"
        )

    offset = 12
    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk_end = offset + chunk_length
        if chunk_end > len(data):
            raise AvatarAssetError(f"VRM contains a truncated chunk: {vrm_path}")
        if chunk_type == 0x4E4F534A:
            try:
                return json.loads(data[offset:chunk_end].decode("utf-8").rstrip("\x00 "))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise AvatarAssetError(f"Invalid VRM JSON chunk: {exc}") from exc
        offset = chunk_end
    raise AvatarAssetError(f"VRM has no JSON chunk: {vrm_path}")


def validate_vrm(vrm_path: Path) -> AvatarStatus:
    resolved = vrm_path.expanduser().resolve()
    if not resolved.is_file():
        raise AvatarAssetError(
            f"Pinned VRM avatar is missing: {resolved}. "
            "Run `uv run motioncapture-avatars fetch` explicitly."
        )
    actual_sha256 = _sha256_file(resolved)
    if actual_sha256 != VRM_SHA256:
        raise AvatarAssetError(
            "Seed-san checksum mismatch: "
            f"expected={VRM_SHA256} actual={actual_sha256} path={resolved}"
        )

    document = _vrm_json_document(resolved)
    extensions = document.get("extensions")
    vrm = extensions.get("VRMC_vrm") if isinstance(extensions, dict) else None
    if not isinstance(vrm, dict) or vrm.get("specVersion") != "1.0":
        raise AvatarAssetError("Seed-san is not a supported VRM 1.0 avatar")

    humanoid = vrm.get("humanoid")
    bones = humanoid.get("humanBones") if isinstance(humanoid, dict) else None
    if not isinstance(bones, dict):
        raise AvatarAssetError("Seed-san has no VRM humanoid bone mapping")
    if missing_bones := sorted(REQUIRED_VRM_BONES - bones.keys()):
        raise AvatarAssetError(f"Seed-san is missing required humanoid bones: {missing_bones}")

    expressions = vrm.get("expressions")
    presets = expressions.get("preset") if isinstance(expressions, dict) else None
    if not isinstance(presets, dict):
        raise AvatarAssetError("Seed-san has no VRM expression presets")
    if missing_expressions := sorted(REQUIRED_VRM_EXPRESSIONS - presets.keys()):
        raise AvatarAssetError(
            f"Seed-san is missing required expression presets: {missing_expressions}"
        )

    meta = vrm.get("meta")
    if not isinstance(meta, dict):
        raise AvatarAssetError("Seed-san has no VRM license metadata")
    if meta.get("avatarPermission") != "everyone" or meta.get("allowRedistribution") is not True:
        raise AvatarAssetError("Seed-san metadata does not permit the intended avatar test setup")

    return AvatarStatus(
        key="vrm_seed_san",
        name=str(meta.get("name", "Seed-san")),
        format="VRM 1.0",
        path=resolved,
        source=VRM_URL,
        source_revision=VRM_COMMIT,
        sha256=actual_sha256,
        details={
            "humanoid_bones": len(bones),
            "expression_presets": sorted(presets),
            "avatar_permission": meta["avatarPermission"],
            "redistribution_allowed": meta["allowRedistribution"],
            "credit_required": meta.get("creditNotation") == "required",
            "authors": meta.get("authors", []),
            "license_url": meta.get("licenseUrl"),
        },
    )


def fetch_live2d(avatar_dir: Path) -> AvatarStatus:
    model_path = avatar_dir / LIVE2D_MODEL_RELATIVE_PATH
    if model_path.exists():
        return validate_live2d(model_path)
    target_directory = model_path.parent
    if target_directory.exists():
        raise AvatarAssetError(
            f"Incomplete Live2D target directory already exists: {target_directory}"
        )

    target_directory.parent.mkdir(parents=True, exist_ok=True)
    archive_path = _download_verified(
        LIVE2D_ARCHIVE_URL,
        LIVE2D_ARCHIVE_SHA256,
        target_directory.parent,
        "live2d-haru-",
    )
    staging_directory = Path(
        tempfile.mkdtemp(prefix="haru-staging-", dir=target_directory.parent)
    )
    prefix = PurePosixPath(
        f"CubismWebSamples-{LIVE2D_COMMIT}/Samples/Resources/Haru"
    )
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for entry in archive.infolist():
                entry_path = PurePosixPath(entry.filename)
                try:
                    relative = entry_path.relative_to(prefix)
                except ValueError:
                    continue
                if not relative.parts or entry.is_dir():
                    continue
                if entry_path.is_absolute() or ".." in relative.parts:
                    raise AvatarAssetError(
                        f"Unsafe path in Live2D source archive: {entry.filename}"
                    )
                destination = staging_directory.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
        validate_live2d(staging_directory / "Haru.model3.json")
        os.replace(staging_directory, target_directory)
    except (OSError, zipfile.BadZipFile) as exc:
        raise AvatarAssetError(f"Unable to install pinned Live2D avatar: {exc}") from exc
    finally:
        archive_path.unlink(missing_ok=True)
        if staging_directory.exists():
            shutil.rmtree(staging_directory)
    return validate_live2d(model_path)


def fetch_vrm(avatar_dir: Path) -> AvatarStatus:
    model_path = avatar_dir / VRM_MODEL_RELATIVE_PATH
    if model_path.exists():
        return validate_vrm(model_path)
    if model_path.parent.exists() and any(model_path.parent.iterdir()):
        raise AvatarAssetError(
            f"Incomplete VRM target directory already exists: {model_path.parent}"
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _download_verified(VRM_URL, VRM_SHA256, model_path.parent, "seed-san-")
    try:
        validate_vrm(temporary_path)
        os.replace(temporary_path, model_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return validate_vrm(model_path)


def require_avatars(avatar_dir: Path) -> tuple[AvatarStatus, AvatarStatus]:
    return (
        validate_live2d(avatar_dir / LIVE2D_MODEL_RELATIVE_PATH),
        validate_vrm(avatar_dir / VRM_MODEL_RELATIVE_PATH),
    )


def fetch_avatars(avatar_dir: Path) -> tuple[AvatarStatus, AvatarStatus]:
    return fetch_live2d(avatar_dir), fetch_vrm(avatar_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch", help="download and verify both pinned avatars")
    fetch.add_argument("--avatar-dir", type=Path, default=DEFAULT_AVATAR_DIR)
    verify = subparsers.add_parser("verify", help="verify both installed avatars")
    verify.add_argument("--avatar-dir", type=Path, default=DEFAULT_AVATAR_DIR)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        statuses = (
            fetch_avatars(args.avatar_dir)
            if args.command == "fetch"
            else require_avatars(args.avatar_dir)
        )
    except AvatarAssetError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "avatars": [
                    {
                        "key": status.key,
                        "name": status.name,
                        "format": status.format,
                        "path": str(status.path),
                        "source": status.source,
                        "source_revision": status.source_revision,
                        "sha256": status.sha256,
                        "details": status.details,
                    }
                    for status in statuses
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
