"""Utilities for exporting upscaled images with rich metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image, PngImagePlugin

from .parser import WQVImage
from .pipeline import PipelineResult


def build_metadata(
    source: WQVImage,
    result: PipelineResult,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct a metadata payload describing the export."""

    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "application": _application_info(),
        "source": {
            "path": str(source.path),
            "kind": source.kind.value,
            "title": source.title,
            "captured_at": source.captured_at,
            "metadata": dict(source.metadata),
        },
        "pipeline": result.to_metadata(),
    }
    if extra:
        payload.update(extra)
    return payload


def save_image_with_metadata(image: Image.Image, destination: Path, metadata_payload: Dict[str, Any]) -> None:
    """Persist ``image`` to ``destination`` along with JSON metadata sidecar."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("wqv_viewer_metadata", json.dumps(metadata_payload, ensure_ascii=False))
    image_to_save = image.convert("RGB") if image.mode not in {"RGB", "RGBA", "L"} else image
    image_to_save.save(destination, format="PNG", pnginfo=png_info)

    sidecar_path = destination.with_suffix(".json")
    sidecar_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export_result(
    destination: Path,
    source: WQVImage,
    result: PipelineResult,
    *,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Export ``result`` to ``destination`` and persist metadata."""

    metadata_payload = build_metadata(source, result, extra=extra_metadata)
    save_image_with_metadata(result.image, destination, metadata_payload)


def _application_info() -> Dict[str, str]:
    try:
        version = metadata.version("wqv-viewer")
    except metadata.PackageNotFoundError:  # pragma: no cover - editable installs during dev
        version = "unknown"
    return {
        "name": "WQV Wristcam Viewer",
        "version": version,
    }
