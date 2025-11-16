"""Shared pipeline helpers for GUI and CLI flows."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from .upscaling import Upscaler, upscale_sequence

PipelineStage = Tuple[str, Upscaler, int]


logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Summary for a single pipeline stage."""

    label: str
    upscaler_id: str
    upscaler_label: str
    scale: int
    policy: Optional[str] = None
    backend_summary: Optional[str] = None


@dataclass
class PipelineResult:
    """Outcome of applying a pipeline to an image."""

    image: Image.Image
    summary: str
    fallback_used: bool
    original_policy: str
    applied_policy: str
    stages: List[StageResult] = field(default_factory=list)

    def to_metadata(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation."""

        return {
            "summary": self.summary,
            "fallback_used": self.fallback_used,
            "original_policy": self.original_policy,
            "applied_policy": self.applied_policy,
            "stages": [
                {
                    "label": stage.label,
                    "upscaler_id": stage.upscaler_id,
                    "upscaler_label": stage.upscaler_label,
                    "scale": stage.scale,
                    **({"policy": stage.policy} if stage.policy else {}),
                    **(
                        {"backend_summary": stage.backend_summary}
                        if stage.backend_summary
                        else {}
                    ),
                }
                for stage in self.stages
            ],
        }


@dataclass
class PipelineConfig:
    """Declarative pipeline description consumed by the GUI and CLI."""

    enable_conventional: bool = True
    conventional_id: Optional[str] = None
    conventional_scale: int = 2
    enable_ai: bool = False
    ai_id: Optional[str] = None
    ai_scale: Optional[int] = None
    ai_before_conventional: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "enable_conventional": self.enable_conventional,
            "conventional_id": self.conventional_id,
            "conventional_scale": self.conventional_scale,
            "enable_ai": self.enable_ai,
            "ai_id": self.ai_id,
            "ai_scale": self.ai_scale,
            "ai_before_conventional": self.ai_before_conventional,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object] | None) -> "PipelineConfig":
        if not data:
            return cls()
        return cls(
            enable_conventional=bool(data.get("enable_conventional", True)),
            conventional_id=_as_optional_str(data.get("conventional_id")),
            conventional_scale=int(data.get("conventional_scale", 2)),
            enable_ai=bool(data.get("enable_ai", False)),
            ai_id=_as_optional_str(data.get("ai_id")),
            ai_scale=_as_optional_int(data.get("ai_scale")),
            ai_before_conventional=bool(data.get("ai_before_conventional", False)),
        )


def _as_optional_str(value: object) -> Optional[str]:
    if isinstance(value, str):
        return value
    return None


def _as_optional_int(value: object) -> Optional[int]:
    if isinstance(value, int):
        return value
    try:
        if isinstance(value, str) and value.strip():
            return int(value)
    except ValueError:
        return None
    return None


def build_pipeline(
    config: PipelineConfig,
    *,
    conventional_map: Dict[str, Upscaler],
    ai_map: Dict[str, Upscaler],
) -> List[PipelineStage]:
    """Resolve ``config`` into executable pipeline stages."""

    conventional_stage: Optional[PipelineStage] = None
    ai_stage: Optional[PipelineStage] = None

    if config.enable_conventional:
        upscaler = _resolve_upscaler(config.conventional_id, conventional_map)
        if upscaler is not None:
            conventional_stage = ("Conventional", upscaler, config.conventional_scale)

    if config.enable_ai:
        upscaler = _resolve_upscaler(config.ai_id, ai_map)
        if upscaler is not None and config.ai_scale is not None:
            ai_stage = ("AI", upscaler, config.ai_scale)

    ordered: List[PipelineStage] = []
    stage_order = (ai_stage, conventional_stage) if config.ai_before_conventional else (conventional_stage, ai_stage)
    for stage in stage_order:
        if stage is not None:
            ordered.append(stage)

    return ordered


def _resolve_upscaler(identifier: Optional[str], registry: Dict[str, Upscaler]) -> Optional[Upscaler]:
    if identifier and identifier in registry:
        return registry[identifier]
    if registry:
        # Fall back to the first registered upscaler when unspecified.
        first_key = next(iter(registry))
        return registry[first_key]
    return None


def run_pipeline(
    pipeline: Sequence[PipelineStage],
    source_image: Image.Image,
    *,
    original_policy: str,
) -> PipelineResult:
    """Execute ``pipeline`` while applying the requested device policy."""

    if not pipeline:
        image = source_image.copy()
        summary = f"Original — {image.width}×{image.height}"
        return PipelineResult(
            image=image,
            summary=summary,
            fallback_used=False,
            original_policy=original_policy,
            applied_policy=original_policy,
            stages=[],
        )

    has_ai = any(stage[0] == "AI" for stage in pipeline)
    attempt_policies: List[str] = [original_policy]
    if has_ai and original_policy != "cpu":
        attempt_policies.append("cpu")

    last_error: Optional[Exception] = None
    for attempt_policy in attempt_policies:
        try:
            return _execute_pipeline_once(
                pipeline,
                source_image.copy(),
                attempt_policy=attempt_policy,
                original_policy=original_policy,
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            last_error = exc
            logger.debug(
                "Pipeline execution failed (policy=%s): %s", attempt_policy, exc, exc_info=True
            )
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError("Pipeline execution failed with no captured error.")


def _execute_pipeline_once(
    pipeline: Sequence[PipelineStage],
    source_image: Image.Image,
    *,
    attempt_policy: str,
    original_policy: str,
) -> PipelineResult:
    policy_overrides: List[Optional[str]] = []
    revert_stack: List[Optional[Tuple]] = []
    policy_error: Optional[Exception] = None

    # Apply device overrides to AI upscalers when supported.
    for label, upscaler, _ in pipeline:
        if label != "AI":
            policy_overrides.append(None)
            revert_stack.append(None)
            continue

        setter = getattr(upscaler, "set_device_policy", None)
        getter = getattr(upscaler, "device_policy", None)
        if not callable(setter):
            policy_overrides.append(None)
            revert_stack.append(None)
            continue

        original = None
        if callable(getter):
            try:
                original = getter()
            except Exception:  # pragma: no cover - defensive guard
                logger.debug("Unable to read current device policy", exc_info=True)

        try:
            setter(attempt_policy)
        except Exception as exc:  # pragma: no cover - device override failed
            policy_error = exc
            break

        policy_overrides.append(attempt_policy)
        if original is not None and original != attempt_policy:
            revert_stack.append((setter, original))
        else:
            revert_stack.append(None)

    if policy_error is not None:
        _restore_device_policies(revert_stack)
        raise policy_error

    upscaled_image = upscale_sequence(source_image, [(stage[1], stage[2]) for stage in pipeline])

    stages: List[StageResult] = []
    for index, (label, upscaler, scale) in enumerate(pipeline):
        summary: Optional[str] = None
        describe_backend = getattr(upscaler, "describe_backend", None)
        if callable(describe_backend) and label == "AI":
            try:
                summary = describe_backend()
            except Exception:  # pragma: no cover - defensive guard
                logger.debug("AI backend description failed", exc_info=True)
                summary = None
        policy = policy_overrides[index]
        stages.append(
            StageResult(
                label=label,
                upscaler_id=getattr(upscaler, "id", "unknown"),
                upscaler_label=getattr(upscaler, "label", label),
                scale=scale,
                policy=policy,
                backend_summary=summary,
            )
        )

    summary_text = _format_summary(upscaled_image, stages)

    _restore_device_policies(revert_stack)

    fallback_used = attempt_policy != original_policy and any(policy_overrides)
    result = PipelineResult(
        image=upscaled_image.copy(),
        summary=summary_text,
        fallback_used=fallback_used,
        original_policy=original_policy,
        applied_policy=attempt_policy,
        stages=stages,
    )
    return result


def _restore_device_policies(revert_stack: Iterable[Optional[Tuple]]) -> None:
    for entry in reversed(list(revert_stack)):
        if not entry:
            continue
        setter, previous = entry
        try:
            setter(previous)
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Failed to restore device policy", exc_info=True)


def _format_summary(image: Image.Image, stages: Sequence[StageResult]) -> str:
    stage_texts: List[str] = []
    for stage in stages:
        descriptor = stage.upscaler_label
        if stage.label == "AI" and stage.backend_summary:
            descriptor = f"{descriptor} [{stage.backend_summary}]"
        elif stage.policy:
            descriptor = f"{descriptor} [{stage.policy.upper()}]"
        stage_texts.append(f"{stage.label}: {descriptor} ×{stage.scale}")

    if not stage_texts:
        descriptor = "Original"
    else:
        descriptor = " → ".join(stage_texts)

    return f"{descriptor} — {image.width}×{image.height}"
