"""Utilities for decoding Casio WQV wrist camera image formats."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
import struct
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image


class WQVImageKind(str, Enum):
    """Enumerates the known WQV container variants."""

    MONOCHROME = "monochrome"
    COLOR_JPEG = "color-jpeg"


@dataclass()
class WQVImage:
    """Represents a decoded WQV image along with its metadata."""

    path: Path
    image: Image.Image
    kind: WQVImageKind
    title: Optional[str] = None
    captured_at: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_qimage(self):
        """Convert the backing :class:`PIL.Image.Image` into a QImage lazily.

        Importing Qt inside the module top-level would make it harder to use
        the parser in headless scripts, so the import is deferred here.
        """

        from PyQt6.QtGui import QImage

        if self.image.mode not in {"L", "RGB", "RGBA"}:
            converted = self.image.convert("RGBA")
        else:
            converted = self.image
        data = converted.tobytes("raw", converted.mode)
        fmt = {
            "L": QImage.Format.Format_Grayscale8,
            "RGB": QImage.Format.Format_RGB888,
            "RGBA": QImage.Format.Format_RGBA8888,
        }[converted.mode]
        qimage = QImage(data, converted.width, converted.height, fmt)
        # Pillow owns the buffer, so we need to keep a reference alive.
        qimage.ndarray = data  # type: ignore[attr-defined]
        return qimage


def load_wqv_image(path: Path | str) -> WQVImage:
    """Decode a WQV image from ``path``.

    The loader attempts to auto-detect the file flavour by extension.
    """

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".bin", ".pdr", ".wqv"}:
        return load_wqv_monochrome(path)
    if suffix in {".jpg", ".jpeg", ".jpe"}:
        return load_wqv_color(path)
    if suffix == ".pdb":  # pragma: no cover - guidance path
        raise ValueError(
            "Palm database archives can contain multiple images; use load_wqv_pdb()."
        )
    # Fall back to trying monochrome first – these are the most common assets.
    try:
        return load_wqv_monochrome(path)
    except Exception as exc_monochrome:  # pragma: no cover - rare path
        try:
            return load_wqv_color(path)
        except Exception as exc_color:  # pragma: no cover - rare path
            raise ValueError(
                f"Unrecognised WQV file format for {path}: {suffix}"
            ) from exc_color if exc_color else exc_monochrome


def load_wqv_monochrome(path: Path | str, *, width: int = 120, height: int = 120) -> WQVImage:
    """Load a monochrome WQV image (WQV-1 / WQV-2 models)."""

    path = Path(path)
    raw = path.read_bytes()
    return _load_wqv_monochrome_from_bytes(raw, path, width=width, height=height)


def load_wqv_color(path: Path | str) -> WQVImage:
    """Load a colour JPEG exported by the WQV-3 / WQV-10.

    The official software inserts a short ``DBLK`` block after a marker which
    many decoders choke on. The legacy tools simply strip that block; we do the
    same here before handing the bytes to Pillow.
    """

    path = Path(path)
    raw = bytearray(path.read_bytes())

    marker = b"DBLK"
    try:
        insertion_point = raw.index(marker) - 1
    except ValueError:
        cleaned = raw
    else:
        start = max(insertion_point, 0)
        end = min(start + 9, len(raw))
        del raw[start:end]
        cleaned = raw

    image = Image.open(BytesIO(cleaned)).convert("RGB")

    return WQVImage(
        path=path,
        image=image,
        kind=WQVImageKind.COLOR_JPEG,
    )


def load_wqv_pdb(path: Path | str, *, width: int = 120, height: int = 120) -> List[WQVImage]:
    """Extract all monochrome captures stored inside a ``WQVLinkDB.PDB`` archive."""

    path = Path(path)
    data = path.read_bytes()
    if len(data) < 78:
        raise ValueError(f"File {path} is too small to be a Palm database")

    num_records = struct.unpack_from(">H", data, 76)[0]
    record_table_size = 78 + num_records * 8
    if len(data) < record_table_size:
        raise ValueError(f"Palm database header truncated in {path}")

    records = [
        (
            struct.unpack_from(">I", data, 78 + i * 8)[0],
            data[78 + i * 8 + 4],
            int.from_bytes(data[78 + i * 8 + 5 : 78 + i * 8 + 8], "big"),
        )
        for i in range(num_records)
    ]

    # Append sentinel for easier slicing.
    records.append((len(data), 0, 0))

    images: List[WQVImage] = []
    for index in range(num_records):
        start, attr, unique_id = records[index]
        end = records[index + 1][0]
        if not (0 <= start <= end <= len(data)):
            raise ValueError(f"Palm database record {index} has invalid bounds")
        payload = data[start:end]
        record_metadata = {
            "record_index": str(index),
            "record_attr": str(attr),
            "record_unique_id": str(unique_id),
            "source_pdb": str(path),
        }
        synthetic_name = (
            f"{path.stem}_{unique_id:07d}.pdr" if unique_id else f"{path.stem}_record{index:03d}.pdr"
        )
        synthetic_path = path.with_name(synthetic_name)
        images.append(
            _load_wqv_monochrome_from_bytes(
                payload,
                synthetic_path,
                width=width,
                height=height,
                extra_metadata=record_metadata,
            )
        )

    return images


def _decode_four_bit_grayscale(data: Iterable[int], pixel_count: int) -> bytes:
    """Expand packed 4-bit grayscale samples into ``pixel_count`` bytes."""

    buffer = bytearray(pixel_count)
    index = 0
    for value in data:
        high = (value >> 4) & 0x0F
        low = value & 0x0F
        if index < pixel_count:
            buffer[index] = 255 - high * 17
            index += 1
        if index < pixel_count:
            buffer[index] = 255 - low * 17
            index += 1
        if index >= pixel_count:
            break
    return bytes(buffer)


def _load_wqv_monochrome_from_bytes(
    raw: bytes,
    path: Path,
    *,
    width: int,
    height: int,
    extra_metadata: Optional[Dict[str, str]] = None,
) -> WQVImage:
    """Decode a monochrome image from an in-memory byte stream."""

    nibble_count = (width * height + 1) // 2
    if len(raw) < nibble_count:
        raise ValueError(
            f"Payload {path} is too small to contain a {width}x{height} image"
        )

    data_offset, nibble_stream = _locate_monochrome_payload(raw, nibble_count)

    pixels = _decode_four_bit_grayscale(nibble_stream, width * height)
    image = Image.frombytes("L", (width, height), pixels)

    metadata = {
        "data_offset": str(data_offset),
        "raw_size": str(len(raw)),
    }
    padding_after = len(raw) - (data_offset + len(nibble_stream))
    if data_offset:
        metadata["padding_before"] = str(data_offset)
    if padding_after:
        metadata["padding_after"] = str(padding_after)
    if extra_metadata:
        metadata.update(extra_metadata)

    return WQVImage(
        path=path,
        image=image,
        kind=WQVImageKind.MONOCHROME,
        metadata=metadata,
    )


def _locate_monochrome_payload(raw: bytes, nibble_count: int) -> Tuple[int, bytes]:
    """Return the offset and payload containing packed 4-bit samples.

    Early WQV dumps embed the 4-bit image data between a small header and a
    trailing metadata block. The header advertises the image length in bytes;
    we prefer that marker when present, otherwise we fall back to a set of
    heuristics that keep existing edge-cases working.
    """

    if len(raw) == nibble_count:
        return 0, raw

    header_slice = raw[:128]
    marker = struct.pack(">H", nibble_count)
    marker_index = header_slice.find(marker)
    if marker_index != -1:
        candidate_offset = marker_index + 4  # length field (2 bytes) + stride field
        if candidate_offset + nibble_count <= len(raw):
            return candidate_offset, raw[candidate_offset : candidate_offset + nibble_count]

    canonical_offset = 36
    if len(raw) >= canonical_offset + nibble_count:
        return canonical_offset, raw[canonical_offset : canonical_offset + nibble_count]

    fallback_offset = max(len(raw) - nibble_count, 0)
    return fallback_offset, raw[fallback_offset : fallback_offset + nibble_count]
