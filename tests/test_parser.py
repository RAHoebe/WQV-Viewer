import hashlib
import shutil
from pathlib import Path

import pytest

from wqv_viewer.parser import (
    PalmRecord,
    WQVImageKind,
    WQVLoadReport,
    delete_wqv_pdb_records,
    load_wqv_image,
    load_wqv_monochrome,
    load_wqv_pdb,
    _read_palm_database,
    _locate_monochrome_payload,
    _write_palm_database,
)


@pytest.fixture()
def sample_pdr(tmp_path) -> Path:
    width = height = 120
    nibble_count = (width * height + 1) // 2
    payload = bytes((index % 256 for index in range(nibble_count)))
    destination = tmp_path / "sample.pdr"
    destination.write_bytes(payload)
    return destination


@pytest.fixture()
def sample_pdb(tmp_path) -> Path:
    destination = tmp_path / "WQVLinkDB.PDB"
    shutil.copyfile(Path(__file__).parent / "data" / "WQVLinkDB.PDB", destination)
    return destination


@pytest.fixture()
def sample_color_pdb(tmp_path) -> Path:
    data_dir = Path(__file__).parent / "data"
    destination = tmp_path / "WQVColorDB.PDB"
    shutil.copyfile(data_dir / "WQVColorDB.PDB", destination)
    for companion in data_dir.glob("CASIJPG*.PDB"):
        shutil.copyfile(companion, tmp_path / companion.name)
    return destination


@pytest.fixture(params=sorted(Path(__file__).parent.glob("data/CASIJPG*.PDB")))
def sample_color_jpeg_pdb(request, tmp_path) -> Path:
    destination = tmp_path / request.param.name
    shutil.copyfile(request.param, destination)
    return destination


@pytest.fixture()
def sample_pdb_with_empty_record(tmp_path) -> Path:
    source = Path(__file__).parent / "data" / "WQVLinkDB.PDB"
    header, records = _read_palm_database(source)
    empty_record = PalmRecord(
        attr=records[0].attr,
        unique_id=1,
        payload=b"",
        index=len(records),
    )
    destination = tmp_path / "WQVLinkDB2.PDB"
    _write_palm_database(destination, header, list(records) + [empty_record])
    return destination


def _reference_pixels(path: Path) -> bytes:
    raw = path.read_bytes()
    width = height = 120
    nibble_count = (width * height + 1) // 2
    offset, nibble_stream = _locate_monochrome_payload(raw, nibble_count)
    pixels = bytearray(width * height)
    idx = 0
    for value in nibble_stream:
        for nib in ((value >> 4) & 0x0F, value & 0x0F):
            if idx >= len(pixels):
                break
            pixels[idx] = 255 - nib * 17
            idx += 1
        if idx >= len(pixels):
            break
    return bytes(pixels)


def test_monochrome_loader_decodes_pixels(sample_pdr: Path) -> None:
    wqv = load_wqv_monochrome(sample_pdr)
    assert wqv.kind == WQVImageKind.MONOCHROME
    assert wqv.image.size == (120, 120)
    assert wqv.metadata["raw_size"] == str(len(sample_pdr.read_bytes()))
    assert wqv.image.tobytes() == _reference_pixels(sample_pdr)


def test_autodetect_dispatches_to_monochrome(sample_pdr: Path) -> None:
    wqv = load_wqv_image(sample_pdr)
    assert wqv.kind == WQVImageKind.MONOCHROME
    assert wqv.image.size == (120, 120)


def test_load_wqv_pdb_extracts_records(sample_pdb: Path) -> None:
    images = load_wqv_pdb(sample_pdb)
    assert len(images) == 13

    expected_checksums = {
        "8151044": "1c5c218387d5be3c11f2d6ead2544aa4",
        "8151042": "d6d1b033e0e60b0c4d6d817ee45ae0eb",
        "8151055": "659e94fadfd659d71a3260150bc344db",
        "8151047": "c6a6be0b80460e11a6d820481d8f99d6",
        "8151057": "01d29e1f9d7f4db265e4e17732e9399a",
        "8151059": "a3a7f15d681b287755b8d7e9bf6c7c84",
        "8151061": "50820fc8ce787dfccd7cbf52bf17fe18",
        "8151063": "d7eb40a1011ea0ec6552dd81f5497999",
        "8151065": "194508daafd78675642901ea28223426",
        "8151067": "d0238efb3ef52be0dd79b1a7b7e40317",
        "8151051": "d72d508199a7bd50baa8b77a3de36f0b",
        "8151053": "648aa1654a10eb74ededd50b4d7f62d7",
        "8151069": "d72d508199a7bd50baa8b77a3de36f0b",
    }

    seen_names = set()
    for idx, image in enumerate(images):
        unique_id = image.metadata.get("record_unique_id")
        assert unique_id is not None
        assert image.metadata.get("record_index") == str(idx)
        assert Path(image.metadata.get("source_pdb", "")).name == "WQVLinkDB.PDB"
        seen_names.add(image.path.name)
        digest = hashlib.md5(image.image.tobytes()).hexdigest()
        assert digest == expected_checksums[unique_id]

    assert len(seen_names) == len(images)


def test_load_wqv_pdb_skips_empty_records(sample_pdb_with_empty_record: Path) -> None:
    images = load_wqv_pdb(sample_pdb_with_empty_record)
    assert len(images) == 13
    assert all(image.image.size == (120, 120) for image in images)
    header, records = _read_palm_database(sample_pdb_with_empty_record)
    assert len(records) == 13


def test_load_wqv_color_pdb_tiles_records(sample_color_pdb: Path) -> None:
    cas_digest_by_name = {}
    for cas_file in sample_color_pdb.parent.glob("CASIJPG*.PDB"):
        cas_image = load_wqv_pdb(cas_file)[0]
        title = cas_file.stem
        cas_digest_by_name[title] = hashlib.md5(cas_image.image.tobytes()).hexdigest()

    images = load_wqv_pdb(sample_color_pdb)
    assert len(images) == len(cas_digest_by_name)
    for image in images:
        assert image.kind == WQVImageKind.COLOR_JPEG
        title = image.title or image.metadata.get("casijpg_name")
        assert title in cas_digest_by_name
        assert hashlib.md5(image.image.tobytes()).hexdigest() == cas_digest_by_name[title]
        assert image.metadata.get("source_color_db") == str(sample_color_pdb)


def test_load_wqv_color_pdb_without_companions(tmp_path: Path) -> None:
    data_dir = Path(__file__).parent / "data"
    destination = tmp_path / "WQVColorDB.PDB"
    shutil.copyfile(data_dir / "WQVColorDB.PDB", destination)

    report = WQVLoadReport()
    images = load_wqv_pdb(destination, report=report)
    _, records = _read_palm_database(destination)
    assert len(images) == len(records)
    assert all(image.kind == WQVImageKind.MONOCHROME for image in images)
    assert all(image.image.size == (176, 144) for image in images)
    assert all(image.metadata.get("placeholder_reason") == "missing_companion_casijpg" for image in images)
    assert len(report.warnings) == len(images)
    assert any("placeholder" in warning for warning in report.warnings)


def test_load_casijpg_color_pdb(sample_color_jpeg_pdb: Path) -> None:
    images = load_wqv_pdb(sample_color_jpeg_pdb)
    assert len(images) == 1
    image = images[0]
    assert image.kind == WQVImageKind.COLOR_JPEG
    assert image.image.size == (176, 144)
    assert image.metadata.get("source_pdb") == str(sample_color_jpeg_pdb)


def test_delete_wqv_pdb_records(sample_pdb: Path) -> None:
    images = load_wqv_pdb(sample_pdb)
    first = images[0]
    unique_id = int(first.metadata["record_unique_id"]) if first.metadata.get("record_unique_id") else None
    index = int(first.metadata["record_index"]) if first.metadata.get("record_index") else None

    removed = delete_wqv_pdb_records(sample_pdb, [(unique_id, index)])
    assert removed == 1

    remaining = load_wqv_pdb(sample_pdb)
    assert len(remaining) == len(images) - 1
