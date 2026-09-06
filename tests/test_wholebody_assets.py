"""Asset locks, explicit license uncertainty and safe container extraction."""
import json
import zipfile

import pytest

from motioncapture import wholebody_catalog as a


def test_first_download_is_tofu_not_publisher_or_commercial_clearance(monkeypatch, tmp_path):
    calls = []

    def download(url, destination):
        calls.append(url)
        with zipfile.ZipFile(destination, 'w') as z:
            z.writestr('sdk/end2end.onnx', b'explicit test bytes; not a real model')

    monkeypatch.setattr(a, '_download', download)
    with pytest.raises(a.BenchmarkError, match='acknowledgement'):
        a.fetch_assets(tmp_path, ['dwpose-m'], research_only=False)
    assert not calls
    receipts = a.fetch_assets(tmp_path, ['dwpose-m'], research_only=True)
    assert len(receipts) == len(calls) == 2
    for receipt in receipts:
        assert receipt['hash_authority'] == 'first_download_TOFU'
        assert receipt['publisher_authentication_verified'] is False
        assert receipt['commercial_release_cleared'] is False
        assert receipt['asset']['weights_and_data_clearance'] == 'unreviewed'
    a.fetch_assets(tmp_path, ['dwpose-m'], research_only=True)
    assert len(calls) == 2  # Existing bytes are verified, never redownloaded or changed.
    (tmp_path / 'dwpose-m/model.onnx').write_bytes(b'changed')
    with pytest.raises(a.BenchmarkError, match='checksum_mismatch'):
        a.fetch_assets(tmp_path, ['dwpose-m'], research_only=True)
    assert (tmp_path / 'dwpose-m/model.onnx').read_bytes() == b'changed'


@pytest.mark.parametrize('names', [['../escape.onnx'], ['a.onnx', 'b.onnx'], ['/a.onnx']])
def test_unsafe_or_ambiguous_archives_rejected(tmp_path, names):
    archive = tmp_path / 'a.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for name in names:
            z.writestr(name, b'fixture')
    with pytest.raises(a.BenchmarkError):
        a._extract_onnx(archive, tmp_path / 'output.onnx')
    assert not (tmp_path / 'output.onnx').exists()


def test_failed_download_leaves_existing_user_data_and_no_partial_asset(monkeypatch, tmp_path):
    (tmp_path / 'unrelated.txt').write_text('keep')

    def fail(url, destination):
        destination.write_bytes(b'partial')
        raise OSError('test offline')

    monkeypatch.setattr(a, '_download', fail)
    with pytest.raises(OSError):
        a.fetch_assets(tmp_path, ['rtmw-m'], research_only=True)
    assert sorted(x.name for x in tmp_path.iterdir()) == ['unrelated.txt']


def test_license_receipt_cannot_be_silently_promoted(monkeypatch, tmp_path):
    monkeypatch.setattr(a, '_download', lambda url, path: path.write_bytes(b'fake-onnx'))
    a.fetch_assets(tmp_path, [], research_only=True)
    receipt = tmp_path / 'yolox-tiny/asset-lock.json'
    data = json.loads(receipt.read_text())
    data['commercial_release_cleared'] = True
    receipt.write_text(json.dumps(data))
    with pytest.raises(a.BenchmarkError, match='invalid_license_receipt'):
        a.verify_asset(tmp_path, 'yolox-tiny')
