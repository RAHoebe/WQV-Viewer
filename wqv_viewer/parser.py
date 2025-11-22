"""Utilities for decoding Casio WQV wrist camera image formats."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from io import BytesIO
from pathlib import Path
import logging
import math
import struct
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


logger = logging.getLogger(__name__)

PALM_EPOCH = datetime(1904, 1, 1, tzinfo=timezone.utc)


def _format_palm_timestamp(raw_value: int) -> Optional[str]:
    if raw_value <= 0:
        return None
    try:
        moment = PALM_EPOCH + timedelta(seconds=raw_value)
    except OverflowError:
        return None
    if moment.year < 1990:
        return None
    try:
        localized = moment.astimezone()
    except OSError:
        localized = moment
    if localized.tzinfo is not None:
        localized = localized.replace(tzinfo=None)
    return localized.strftime("%Y-%m-%d %H:%M:%S")


def _format_file_timestamp(raw_value: float) -> Optional[str]:
    try:
        moment = datetime.fromtimestamp(raw_value)
    except (OSError, OverflowError, ValueError):
        return None
    return moment.strftime("%Y-%m-%d %H:%M:%S")


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


@dataclass
class PalmRecord:
    """Represents a single Palm database record."""

    attr: int
    unique_id: int
    payload: bytes
    index: int


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
    """Extract captures stored inside a Palm OS backup archive."""

    path = Path(path)
    header, records = _read_palm_database(path)
    fallback_timestamp = _derive_pdb_timestamp(path, header)
    records = _strip_empty_palm_records(path, header, records)

    database_name = _decode_pdb_name(header)
    if database_name.startswith("WQVColorDB"):
        return _load_wqv_color_database(
            path,
            records,
            fallback_timestamp,
            companion_dir=path.parent,
        )
    if database_name.upper().startswith("CASIJPG"):
        return _load_wqv_color_jpeg_database(path, records, fallback_timestamp)
    return _load_wqv_monochrome_database(
        path,
        records,
        fallback_timestamp,
        width=width,
        height=height,
    )


def _decode_pdb_name(header: bytes) -> str:
    raw_name = header[:32]
    return raw_name.split(b"\x00", 1)[0].decode("ascii", "ignore")


def _derive_pdb_timestamp(path: Path, header: bytes) -> Optional[str]:
    creation_time = struct.unpack_from(">I", header, 32)[0]
    modification_time = struct.unpack_from(">I", header, 36)[0]
    backup_time = struct.unpack_from(">I", header, 40)[0]
    fallback_timestamp = next(
        (
            candidate
            for candidate in (
                _format_palm_timestamp(creation_time),
                _format_palm_timestamp(modification_time),
                _format_palm_timestamp(backup_time),
            )
            if candidate
        ),
        None,
    )

    if fallback_timestamp:
        return fallback_timestamp

    try:
        stat_result = path.stat()
    except OSError:
        return None
    return _format_file_timestamp(stat_result.st_ctime)


def _strip_empty_palm_records(
    path: Path, header: bytearray, records: List[PalmRecord]
) -> List[PalmRecord]:
    cleaned_records = [record for record in records if record.payload]
    removed = len(records) - len(cleaned_records)
    if removed:
        logger.info("Removing %s empty Palm database records from %s", removed, path)
        _write_palm_database(path, header, cleaned_records)
    return cleaned_records


def _load_wqv_monochrome_database(
    path: Path,
    records: List[PalmRecord],
    fallback_timestamp: Optional[str],
    *,
    width: int,
    height: int,
) -> List[WQVImage]:
    images: List[WQVImage] = []
    for new_index, record in enumerate(records):
        record_metadata = {
            "record_index": str(new_index),
            "record_attr": str(record.attr),
            "record_unique_id": str(record.unique_id),
            "source_pdb": str(path),
        }
        synthetic_name = (
            f"{path.stem}_{record.unique_id:07d}.pdr"
            if record.unique_id
            else f"{path.stem}_record{new_index:03d}.pdr"
        )
        synthetic_path = path.with_name(synthetic_name)

        images.append(
            _load_wqv_monochrome_from_bytes(
                record.payload,
                synthetic_path,
                width=width,
                height=height,
                extra_metadata=record_metadata,
            )
        )

    if fallback_timestamp:
        for image in images:
            if not image.captured_at:
                image.captured_at = fallback_timestamp
                image.metadata.setdefault("captured_at", fallback_timestamp)
    return images


def _load_wqv_color_database(
    path: Path,
    records: List[PalmRecord],
    fallback_timestamp: Optional[str],
    *,
    companion_dir: Optional[Path] = None,
) -> List[WQVImage]:
    if not records:
        return []

    images: List[WQVImage] = []
    for record in records:
        payload = record.payload
        if len(payload) < 60:
            raise ValueError(
                f"Palm color chunk header in record {record.index} of {path} is truncated"
            )

        width = struct.unpack_from(">H", payload, 2)[0]
        height = struct.unpack_from(">H", payload, 4)[0]
        rows_per_tile = struct.unpack_from(">H", payload, 58)[0]
        name_bytes = payload[36:52]
        record_name = name_bytes.split(b"\x00", 1)[0].decode("ascii", "ignore").strip()
        timestamp_raw = int.from_bytes(payload[32:36], "big")
        captured_at = _format_palm_timestamp(timestamp_raw) or fallback_timestamp

        metadata = {
            "source_color_db": str(path),
            "record_index": str(record.index),
            "record_attr": str(record.attr),
            "record_unique_id": str(record.unique_id),
        }
        if captured_at:
            metadata["captured_at"] = captured_at
        if record_name:
            metadata["casijpg_name"] = record_name

        casijpg_image: Optional[WQVImage] = None
        if companion_dir and record_name:
            candidate = companion_dir / f"{record_name}.PDB"
            if candidate.exists():
                try:
                    _, cas_records = _read_palm_database(candidate)
                    cas_images = _load_wqv_color_jpeg_database(candidate, cas_records, captured_at)
                except Exception as exc:  # pragma: no cover - corrupted companion
                    logger.warning("Failed to decode companion %s: %s", candidate, exc)
                else:
                    if cas_images:
                        casijpg_image = cas_images[0]

        if casijpg_image is not None:
            casijpg_image.metadata.update(metadata)
            if record_name and not casijpg_image.title:
                casijpg_image.title = record_name
            images.append(casijpg_image)
            continue

        if width <= 0 or height <= 0 or rows_per_tile <= 0:
            raise ValueError(f"Palm color chunk dimensions are invalid in {path}")

        pil_image = _render_missing_cas_placeholder(width, height)
        metadata["placeholder_reason"] = "missing_companion_casijpg"
        if record_name:
            metadata["expected_casijpg"] = f"{record_name}.PDB"
        synthetic_name = f"{path.stem}_{record.unique_id:07d}_thumb.pdb"
        synthetic_path = path.with_name(synthetic_name)
        image = WQVImage(
            path=synthetic_path,
            image=pil_image,
            kind=WQVImageKind.MONOCHROME,
            title=record_name or None,
            captured_at=captured_at,
            metadata=metadata,
        )
        images.append(image)

    return images


def _render_missing_cas_placeholder(width: int, height: int) -> Image.Image:
    width = max(64, min(width or 176, 512))
    height = max(64, min(height or 144, 512))
    placeholder = Image.new("L", (width, height), 24)
    draw = ImageDraw.Draw(placeholder)
    banner_height = max(24, height // 4)
    draw.rectangle((0, 0, width, banner_height), fill=80)

    message = "CAS image missing"
    font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), message, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:  # pragma: no cover - Pillow < 8 fallback
        text_width, text_height = draw.textsize(message, font=font)

    text_x = max((width - text_width) // 2, 0)
    text_y = max((banner_height - text_height) // 2, 0)
    draw.text((text_x, text_y), message, fill=255, font=font)

    footer_text = "Recorded thumbnail"
    footer_font = ImageFont.load_default()
    try:
        footer_bbox = draw.textbbox((0, 0), footer_text, font=footer_font)
        footer_height = footer_bbox[3] - footer_bbox[1]
    except AttributeError:  # pragma: no cover - Pillow < 8 fallback
        _, footer_height = draw.textsize(footer_text, font=footer_font)
    footer_y = height - footer_height - 6
    draw.text((6, footer_y), footer_text, fill=160, font=footer_font)
    draw.rectangle((0, banner_height, width - 1, height - 1), outline=64)

    return placeholder


def _load_wqv_color_jpeg_database(
    path: Path,
    records: List[PalmRecord],
    fallback_timestamp: Optional[str],
) -> List[WQVImage]:
    if not records:
        return []

    blob = bytearray()
    chunk_sizes: List[str] = []
    record_ids: List[str] = []

    for record in records:
        payload = record.payload
        if len(payload) < 8 or not payload.startswith(b"DBLK"):
            raise ValueError(
                f"Palm color JPEG chunk {record.index} in {path} is missing DBLK header"
            )
        chunk_length = struct.unpack_from(">H", payload, 6)[0]
        body = payload[8:]
        if chunk_length and chunk_length <= len(body):
            body = body[:chunk_length]
        blob.extend(body)
        chunk_sizes.append(str(len(body)))
        record_ids.append(str(record.unique_id))

    try:
        pil_image = Image.open(BytesIO(blob)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Unable to decode JPEG payload in {path}: {exc}") from exc

    metadata = {
        "source_pdb": str(path),
        "record_unique_ids": ",".join(record_ids),
        "chunk_sizes": ",".join(chunk_sizes),
    }

    title = path.stem
    image = WQVImage(
        path=path,
        image=pil_image,
        kind=WQVImageKind.COLOR_JPEG,
        title=title,
        captured_at=fallback_timestamp,
        metadata=metadata,
    )
    if fallback_timestamp:
        image.metadata.setdefault("captured_at", fallback_timestamp)
    return [image]


def load_wqv_backup(root: Path | str) -> List[WQVImage]:
    """Load every WQV-compatible image contained in a backup directory or file."""

    root_path = Path(root)
    if root_path.is_file():
        return _load_wqv_file(root_path)

    if not root_path.is_dir():
        raise FileNotFoundError(f"Path {root_path} does not exist")

    supported_suffixes = {
        ".pdb",
        ".bin",
        ".pdr",
        ".wqv",
        ".jpg",
        ".jpeg",
        ".jpe",
    }

    collected: List[WQVImage] = []
    files = sorted(
        (candidate for candidate in root_path.rglob("*") if candidate.is_file()),
        key=lambda candidate: (candidate.relative_to(root_path).as_posix().lower(), candidate.name.lower()),
    )

    for file_path in files:
        if file_path.suffix.lower() not in supported_suffixes:
            continue
        try:
            if file_path.suffix.lower() == ".pdb":
                collected.extend(load_wqv_pdb(file_path))
            else:
                collected.append(load_wqv_image(file_path))
        except Exception as exc:  # pragma: no cover - dataset variability
            logger.warning("Skipping %s: %s", file_path, exc)
            continue

    return collected


def _load_wqv_file(path: Path) -> List[WQVImage]:
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        return load_wqv_pdb(path)
    if suffix in {".bin", ".pdr", ".wqv", ".jpg", ".jpeg", ".jpe"}:
        return [load_wqv_image(path)]
    raise ValueError(f"Unsupported WQV file type: {path}")


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


def _read_palm_database(path: Path) -> Tuple[bytearray, List[PalmRecord]]:
    data = Path(path).read_bytes()
    if len(data) < 78:
        raise ValueError(f"File {path} is too small to be a Palm database")

    header = bytearray(data[:78])
    num_records = struct.unpack_from(">H", header, 76)[0]
    record_table_size = 78 + num_records * 8
    if len(data) < record_table_size:
        raise ValueError(f"Palm database header truncated in {path}")

    offsets: List[int] = []
    records: List[PalmRecord] = []
    for index in range(num_records):
        offset = struct.unpack_from(">I", data, 78 + index * 8)[0]
        attr = data[78 + index * 8 + 4]
        unique_id = int.from_bytes(data[78 + index * 8 + 5 : 78 + index * 8 + 8], "big")
        offsets.append(offset)
        records.append(PalmRecord(attr=attr, unique_id=unique_id, payload=b"", index=index))

    offsets.append(len(data))

    for index, record in enumerate(records):
        start = offsets[index]
        end = offsets[index + 1]
        if not (0 <= start <= end <= len(data)):
            raise ValueError(f"Palm database record {index} has invalid bounds")
        record.payload = data[start:end]

    return header, records


def _write_palm_database(path: Path, header: bytearray, records: Sequence[PalmRecord]) -> None:
    new_header = bytearray(header)
    struct.pack_into(">H", new_header, 76, len(records))

    record_table = bytearray()
    data_section = bytearray()
    offset = 78 + len(records) * 8

    for record in records:
        record_table.extend(struct.pack(">I", offset))
        record_table.append(record.attr & 0xFF)
        record_table.extend(int(record.unique_id).to_bytes(3, "big", signed=False))
        data_section.extend(record.payload)
        offset += len(record.payload)

    Path(path).write_bytes(bytes(new_header + record_table + data_section))


def delete_wqv_pdb_records(
    path: Path | str,
    selectors: Iterable[Tuple[Optional[int], Optional[int]]],
) -> int:
    """Delete Palm database records selected by ``selectors``.

    Each selector contains ``(unique_id, index)`` where ``unique_id`` may be ``None``.
    """

    selectors = [(uid, idx) for uid, idx in selectors]
    if not selectors:
        return 0

    path = Path(path)
    header, records = _read_palm_database(path)

    def _matches(record: PalmRecord) -> bool:
        for unique_id, index in selectors:
            if unique_id is not None and record.unique_id == unique_id:
                return True
            if unique_id is None and index is not None and record.index == index:
                return True
        return False

    remaining = [record for record in records if not _matches(record)]
    removed = len(records) - len(remaining)
    if removed:
        logger.info("Removed %s records from %s", removed, path)
        _write_palm_database(path, header, remaining)
    return removed
