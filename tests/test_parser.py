from pathlib import Path

import pytest

from wqv_viewer.parser import (
    WQVImageKind,
    load_wqv_image,
    load_wqv_monochrome,
    load_wqv_pdb,
    _locate_monochrome_payload,
)


@pytest.fixture(scope="module")
def sample_pdr() -> Path:
    return Path(__file__).parent / "data" / "000.40.8151044.pdr"


@pytest.fixture(scope="module")
def sample_pdb() -> Path:
    return Path(__file__).parent / "data" / "WQVLinkDB.PDB"


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
    assert len(images) == 3

    expected_lookup = {
        "8151044": Path(__file__).parent / "data" / "000.40.8151044.pdr",
        "8151041": Path(__file__).parent / "data" / "001.40.8151041.pdr",
        "8151042": Path(__file__).parent / "data" / "002.40.8151042.pdr",
    }

    seen_names = set()
    for idx, image in enumerate(images):
        unique_id = image.metadata.get("record_unique_id")
        assert unique_id is not None
        assert image.metadata.get("record_index") == str(idx)
        assert image.metadata.get("source_pdb", "").endswith("WQVLinkDB.PDB")
        seen_names.add(image.path.name)

        reference_path = expected_lookup[unique_id]
        reference = load_wqv_monochrome(reference_path)
        assert image.image.tobytes() == reference.image.tobytes()

    assert len(seen_names) == len(images)
