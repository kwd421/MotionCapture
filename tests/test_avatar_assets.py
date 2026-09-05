import json
import struct
from pathlib import Path

import pytest

from motioncapture.avatar_assets import validate_live2d, validate_vrm
from motioncapture.errors import AvatarAssetError


def test_live2d_validation_rejects_missing_references(tmp_path: Path) -> None:
    model_path = tmp_path / "Haru.model3.json"
    model_path.write_text(
        json.dumps(
            {
                "Version": 3,
                "FileReferences": {"Moc": "missing.moc3"},
                "Groups": [
                    {"Name": "EyeBlink", "Target": "Parameter", "Ids": []},
                    {"Name": "LipSync", "Target": "Parameter", "Ids": []},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AvatarAssetError, match="missing referenced files"):
        validate_live2d(model_path)


def test_vrm_validation_rejects_a_non_vrm_glb(tmp_path: Path) -> None:
    document = json.dumps({"asset": {"version": "2.0"}}, separators=(",", ":")).encode()
    document += b" " * ((4 - len(document) % 4) % 4)
    total_length = 12 + 8 + len(document)
    data = struct.pack("<4sII", b"glTF", 2, total_length)
    data += struct.pack("<II", len(document), 0x4E4F534A) + document
    model_path = tmp_path / "not-vrm.glb"
    model_path.write_bytes(data)
    with pytest.raises(AvatarAssetError, match="checksum mismatch"):
        validate_vrm(model_path)


def test_missing_avatar_is_a_terminal_error(tmp_path: Path) -> None:
    with pytest.raises(AvatarAssetError, match="missing"):
        validate_live2d(tmp_path / "Haru.model3.json")
