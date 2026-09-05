from pathlib import Path

import pytest

from motioncapture.errors import ModelAssetError
from motioncapture.model_assets import MODEL_ASSETS, require_model, require_models


def test_missing_model_is_a_terminal_error(tmp_path: Path) -> None:
    asset = MODEL_ASSETS[0]
    missing = tmp_path / asset.filename
    with pytest.raises(ModelAssetError, match="missing"):
        require_model(asset, missing)


def test_invalid_model_is_not_accepted(tmp_path: Path) -> None:
    asset = MODEL_ASSETS[0]
    invalid = tmp_path / asset.filename
    invalid.write_bytes(b"not a model")
    with pytest.raises(ModelAssetError, match="checksum mismatch"):
        require_model(asset, invalid)


def test_pinned_checksums_are_sha256() -> None:
    for asset in MODEL_ASSETS:
        assert len(asset.sha256) == 64
        int(asset.sha256, 16)


def test_all_three_models_are_required(tmp_path: Path) -> None:
    with pytest.raises(ModelAssetError, match="pose model is missing"):
        require_models(tmp_path)
