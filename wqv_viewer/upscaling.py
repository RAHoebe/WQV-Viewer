"""Utilities for upscaling WQV images."""

from __future__ import annotations

import logging
import contextlib
import io
import inspect
import shutil
import re
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Literal, Optional, Protocol, Sequence, Tuple, Union, cast

from PIL import Image

# ---------------------------------------------------------------------------
# Scale definitions

ALLOWED_CONVENTIONAL_SCALES: Tuple[int, ...] = (2, 3, 4, 5, 6)
ALLOWED_AI_SCALES: Tuple[int, ...] = (2, 4, 8)

_MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"
_REALESRGAN_MODEL_DIR = _MODELS_ROOT / "realesrgan"
_CUSTOM_MODEL_DIR = _MODELS_ROOT / "custom"


@dataclass(frozen=True)
class RealESRGANWeightSpec:
    model_name: str
    urls: Tuple[str, ...]
    local_path: Optional[Path] = None


@dataclass(frozen=True)
class RealESRGANModelSpec:
    weights: Tuple[RealESRGANWeightSpec, ...]
    arch: str
    arch_args: Dict[str, object]
    dni_weight: Optional[Tuple[float, ...]] = None


@dataclass(frozen=True)
class RealESRGANVariantSpec:
    id: str
    label: str
    models: Dict[int, RealESRGANModelSpec]


RealESRGANDevicePolicy = Literal["auto", "gpu", "cpu"]


def _ensure_torchvision_functional_tensor() -> None:
    """Provide a compatibility shim for newer torchvision releases.

    Torchvision 0.16+ renamed ``torchvision.transforms.functional_tensor`` to
    ``torchvision.transforms._functional_tensor`` while Real-ESRGAN (via
    BasicSR) still imports the old public path. Installing a slightly older
    torchvision build that matches the current PyTorch wheel is awkward, so we
    install a lightweight module alias on demand that forwards the attributes
    from the new location.
    """
    import importlib
    import sys
    import types

    target = "torchvision.transforms.functional_tensor"
    if target in sys.modules:
        return
    try:
        source = importlib.import_module("torchvision.transforms._functional_tensor")
    except ModuleNotFoundError:
        return

    shim = types.ModuleType(target)
    for name in dir(source):
        if name.startswith("__"):
            continue
        setattr(shim, name, getattr(source, name))
    sys.modules[target] = shim


class Upscaler(Protocol):
    """Protocol describing an upscaling implementation."""

    id: str
    label: str

    def upscale(self, image: Image.Image, scale: int) -> Image.Image:
        ...

    def supports_scale(self, scale: int) -> bool:
        ...

    def supported_scales(self) -> Sequence[int]:
        ...


@dataclass(frozen=True)
class PillowUpscaler:
    """Basic resampling upscaler powered by Pillow."""

    id: str
    label: str
    resample: int

    def upscale(self, image: Image.Image, scale: int) -> Image.Image:
        target_size = (image.width * scale, image.height * scale)
        return image.resize(target_size, resample=self.resample)

    def supports_scale(self, scale: int) -> bool:
        return scale in ALLOWED_CONVENTIONAL_SCALES

    def supported_scales(self) -> Sequence[int]:
        return ALLOWED_CONVENTIONAL_SCALES


_REALESRGAN_VARIANTS: Tuple[RealESRGANVariantSpec, ...] = (
    RealESRGANVariantSpec(
        id="realesrgan-plus",
        label="Real-ESRGAN Plus (2×/4×)",
        models={
            2: RealESRGANModelSpec(
                weights=(
                    RealESRGANWeightSpec(
                        "RealESRGAN_x2plus",
                        (
                            "https://github.com/xinntao/Real-ESRGAN/releases/latest/download/RealESRGAN_x2plus.pth",
                            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                            "https://huggingface.co/xinntao/Real-ESRGAN/resolve/main/RealESRGAN_x2plus.pth",
                        ),
                    ),
                ),
                arch="sber_rrdbnet",
                arch_args=dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
            ),
            4: RealESRGANModelSpec(
                weights=(
                    RealESRGANWeightSpec(
                        "RealESRGAN_x4plus",
                        (
                            "https://github.com/xinntao/Real-ESRGAN/releases/latest/download/RealESRGAN_x4plus.pth",
                            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                            "https://huggingface.co/xinntao/Real-ESRGAN/resolve/main/RealESRGAN_x4plus.pth",
                        ),
                    ),
                ),
                arch="rrdbnet",
                arch_args=dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            ),
        },
    ),
    RealESRGANVariantSpec(
        id="realesrgan-sber",
        label="Real-ESRGAN (Sber 2×/4×/8×)",
        models={
            2: RealESRGANModelSpec(
                weights=(
                    RealESRGANWeightSpec(
                        "RealESRGAN_sber_x2",
                        (
                            "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth",
                        ),
                    ),
                ),
                arch="rrdbnet",
                arch_args=dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
            ),
            4: RealESRGANModelSpec(
                weights=(
                    RealESRGANWeightSpec(
                        "RealESRGAN_sber_x4",
                        (
                            "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth",
                        ),
                    ),
                ),
                arch="sber_rrdbnet",
                arch_args=dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            ),
            8: RealESRGANModelSpec(
                weights=(
                    RealESRGANWeightSpec(
                        "RealESRGAN_sber_x8",
                        (
                            "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x8.pth",
                        ),
                    ),
                ),
                arch="sber_rrdbnet",
                arch_args=dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=8),
            ),
        },
    ),
    RealESRGANVariantSpec(
        id="realesrgan-general",
        label="Real-ESRGAN General x4v3",
        models={
            4: RealESRGANModelSpec(
                weights=(
                    RealESRGANWeightSpec(
                        "realesr-general-x4v3",
                        (
                            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                            "https://huggingface.co/xinntao/Real-ESRGAN/resolve/main/realesr-general-x4v3.pth",
                        ),
                    ),
                    RealESRGANWeightSpec(
                        "realesr-general-wdn-x4v3",
                        (
                            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                            "https://huggingface.co/xinntao/Real-ESRGAN/resolve/main/realesr-general-wdn-x4v3.pth",
                        ),
                    ),
                ),
                arch="srvgg",
                arch_args=dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"),
                dni_weight=(0.5, 0.5),
            ),
        },
    ),
    RealESRGANVariantSpec(
        id="realesrgan-general-wdn",
        label="Real-ESRGAN General x4v3 (denoise)",
        models={
            4: RealESRGANModelSpec(
                weights=(
                    RealESRGANWeightSpec(
                        "realesr-general-wdn-x4v3",
                        (
                            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                            "https://huggingface.co/xinntao/Real-ESRGAN/resolve/main/realesr-general-wdn-x4v3.pth",
                        ),
                    ),
                ),
                arch="srvgg",
                arch_args=dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"),
            ),
        },
    ),

)


_BUILTIN_WEIGHT_NAMES: set[str] = {
    weight.model_name
    for variant in _REALESRGAN_VARIANTS
    for spec in variant.models.values()
    for weight in spec.weights
}


def _infer_scale_from_name(name: str) -> Optional[int]:
    lowered = name.lower()
    for token, scale in (("x8", 8), ("x4", 4), ("x2", 2)):
        if re.search(rf"{token}(?!\d)", lowered):
            return scale
    return None


def _slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "custom-model"


def _cleanup_stale_custom_copies(existing_stems: set[str]) -> None:
    if not _REALESRGAN_MODEL_DIR.exists():
        return

    for path in _REALESRGAN_MODEL_DIR.glob("*.pth"):
        stem = path.stem
        if stem in existing_stems:
            continue
        if stem in _BUILTIN_WEIGHT_NAMES:
            continue
        try:
            path.unlink()
            logger.info("Removed stale custom Real-ESRGAN weight at %s", path)
        except OSError as exc:
            logger.warning("Failed to remove stale custom weight %s: %s", path, exc)


def _discover_custom_variants() -> Tuple[RealESRGANVariantSpec, ...]:
    if not _CUSTOM_MODEL_DIR.exists():
        _cleanup_stale_custom_copies(set())
        return ()

    variants: List[RealESRGANVariantSpec] = []
    seen_ids: set[str] = set()
    custom_paths = sorted(_CUSTOM_MODEL_DIR.rglob("*.pth"))
    existing_stems = {path.stem for path in custom_paths}

    for path in custom_paths:
        scale = _infer_scale_from_name(path.stem)
        if scale not in (2, 4, 8):
            logger.debug("Skipping custom model %s: unable to infer scale", path)
            continue

        slug_base = _slugify_name(path.stem)
        variant_id = f"custom-{slug_base}"
        unique_id = variant_id
        counter = 1
        while unique_id in seen_ids:
            counter += 1
            unique_id = f"{variant_id}-{counter}"
        seen_ids.add(unique_id)

        label = f"Custom: {path.stem} (×{scale})"
        weight_spec = RealESRGANWeightSpec(path.stem, (), local_path=path)
        arch = "sber_rrdbnet" if scale == 8 else "rrdbnet"
        model_spec = RealESRGANModelSpec(
            weights=(weight_spec,),
            arch=arch,
            arch_args=dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale),
        )
        variants.append(
            RealESRGANVariantSpec(
                id=unique_id,
                label=label,
                models={scale: model_spec},
            )
        )

    _cleanup_stale_custom_copies(existing_stems)

    return tuple(variants)


logger = logging.getLogger(__name__)


class RealESRGANUpscaler:
    """GAN-based upscaler powered by Real-ESRGAN."""

    def __init__(self, variant: RealESRGANVariantSpec) -> None:
        self.id = variant.id
        self.label = variant.label
        self._variant = variant
        self._model_specs = variant.models
        self._upsamplers: dict[tuple[int, RealESRGANDevicePolicy], object] = {}
        self._upsampler_details: dict[tuple[int, RealESRGANDevicePolicy], Dict[str, object]] = {}
        self._failed_attempts: dict[tuple[int, RealESRGANDevicePolicy], set[Tuple[str, bool, int]]] = defaultdict(set)
        self._device_policy: RealESRGANDevicePolicy = "auto"
        self._last_backend_details: Optional[Dict[str, object]] = None
        self._init_error: Optional[str] = None
        self._torch_supports_weights_only: bool = False
        self._model_dir = _REALESRGAN_MODEL_DIR
        self._model_dir.mkdir(parents=True, exist_ok=True)

    def supported_scales(self) -> Sequence[int]:
        return tuple(sorted(self._model_specs.keys()))

    def supports_scale(self, scale: int) -> bool:
        return scale in self._model_specs

    def set_device_policy(self, policy: RealESRGANDevicePolicy | str) -> None:
        if isinstance(policy, str):
            policy_lower = policy.lower()
        else:
            policy_lower = policy
        if policy_lower not in {"auto", "gpu", "cpu"}:
            raise ValueError(f"Unknown Real-ESRGAN device policy '{policy}'.")
        self._device_policy = cast(RealESRGANDevicePolicy, policy_lower)
        self._last_backend_details = None

    def device_policy(self) -> RealESRGANDevicePolicy:
        return self._device_policy

    def _ensure_backend(self):
        if self._init_error is not None:
            raise RuntimeError(self._init_error)
        if hasattr(self, "_backend"):
            return self._backend
        try:
            _ensure_torchvision_functional_tensor()
            from realesrgan import RealESRGANer  # type: ignore
            from basicsr.archs.rrdbnet_arch import RRDBNet, RRDB  # type: ignore
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact  # type: ignore
            from basicsr.archs import arch_util as basicsr_arch_util  # type: ignore
            from basicsr.utils.download_util import load_file_from_url  # type: ignore
            from torch import nn  # type: ignore
            from torch.nn import functional as F  # type: ignore
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency missing
            self._init_error = (
                "Real-ESRGAN dependencies are not available. Install the optional "
                "machine learning requirements listed in the README to enable AI upscaling."
            )
            raise RuntimeError(self._init_error) from exc

        try:
            signature = inspect.signature(torch.load)
            self._torch_supports_weights_only = "weights_only" in signature.parameters
        except (TypeError, ValueError):  # pragma: no cover - builtins without signature metadata
            self._torch_supports_weights_only = False

        if self._torch_supports_weights_only and not getattr(torch, "_wqv_weights_only_patch", False):
            original_torch_load = torch.load

            def _torch_load_weights_only(*args, **kwargs):
                kwargs.setdefault("weights_only", True)
                return original_torch_load(*args, **kwargs)

            torch.load = _torch_load_weights_only  # type: ignore[assignment]
            setattr(torch, "_wqv_weights_only_patch", True)

        default_init_weights = basicsr_arch_util.default_init_weights  # type: ignore[attr-defined]
        make_layer = basicsr_arch_util.make_layer  # type: ignore[attr-defined]
        pixel_unshuffle = basicsr_arch_util.pixel_unshuffle  # type: ignore[attr-defined]

        class SberRRDBNet(nn.Module):  # pragma: no cover - thin wrapper around upstream model
            def __init__(
                self,
                num_in_ch: int,
                num_out_ch: int,
                *,
                scale: int = 4,
                num_feat: int = 64,
                num_block: int = 23,
                num_grow_ch: int = 32,
            ) -> None:
                super().__init__()
                self.scale = scale
                in_channels = num_in_ch
                if scale == 2:
                    in_channels = num_in_ch * 4
                elif scale == 1:
                    in_channels = num_in_ch * 16

                self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1)
                self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
                self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                if scale == 8:
                    self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                else:
                    self.conv_up3 = None
                self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
                default_init_weights([self.conv_first, self.conv_body, self.conv_up1, self.conv_up2, self.conv_hr], 0.1)
                default_init_weights([self.conv_last], 0.1)
                if self.conv_up3 is not None:
                    default_init_weights([self.conv_up3], 0.1)
                self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

            def forward(self, x):
                if self.scale == 2:
                    feat = pixel_unshuffle(x, scale=2)
                elif self.scale == 1:
                    feat = pixel_unshuffle(x, scale=4)
                else:
                    feat = x

                feat = self.conv_first(feat)
                body_feat = self.conv_body(self.body(feat))
                feat = feat + body_feat

                feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
                feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
                if self.conv_up3 is not None:
                    feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode="nearest")))
                out = self.conv_last(self.lrelu(self.conv_hr(feat)))
                return out

        self._backend = {
            "RealESRGANer": RealESRGANer,
            "RRDBNet": RRDBNet,
            "SRVGGNetCompact": SRVGGNetCompact,
            "SberRRDBNet": SberRRDBNet,
            "load_file_from_url": load_file_from_url,
            "torch": torch,
        }
        return self._backend

    def _ensure_weights(
        self,
        weights: Sequence[RealESRGANWeightSpec],
        loader,
        *,
        torch_module,
    ) -> List[Path]:
        paths: List[Path] = []
        for weight in weights:
            destination = self._model_dir / f"{weight.model_name}.pth"
            local_source = Path(weight.local_path).expanduser() if weight.local_path else None

            if local_source is not None:
                if not local_source.exists():
                    raise FileNotFoundError(
                        f"Custom Real-ESRGAN weight '{weight.model_name}' not found at {local_source}"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                same_path = False
                if destination.exists():
                    try:
                        same_path = local_source.samefile(destination)
                    except (OSError, AttributeError):
                        same_path = False
                if not same_path:
                    needs_copy = True
                    if destination.exists():
                        try:
                            needs_copy = local_source.stat().st_mtime > destination.stat().st_mtime
                        except OSError:
                            needs_copy = True
                    if needs_copy:
                        shutil.copy2(local_source, destination)
            elif not destination.exists():
                if not weight.urls:
                    raise RuntimeError(
                        f"Local Real-ESRGAN weight '{weight.model_name}' not found at {destination}. "
                        "Place the file there manually or provide download URLs."
                    )
                errors: List[str] = []
                downloaded = False
                for url in weight.urls:
                    try:
                        logger.info("Downloading %s from %s", weight.model_name, url)
                        downloaded_path = loader(
                            url,
                            model_dir=str(self._model_dir),
                            progress=True,
                        )
                        candidate_paths: List[Path] = []
                        if downloaded_path:
                            candidate_paths.append(Path(downloaded_path))
                        candidate_paths.append(self._model_dir / Path(url).name)
                        for candidate in candidate_paths:
                            if candidate.exists():
                                if candidate.resolve() != destination.resolve():
                                    shutil.move(str(candidate), destination)
                                downloaded = True
                                break
                        if not downloaded and destination.exists():
                            downloaded = True
                        if downloaded:
                            break
                    except Exception as exc:  # pragma: no cover - network variability
                        errors.append(f"{url}: {exc}")
                        logger.warning(
                            "Failed to download %s from %s: %s", weight.model_name, url, exc
                        )
                if not downloaded:
                    raise RuntimeError(
                        "Unable to download Real-ESRGAN weights. Tried:\n" + "\n".join(errors)
                    )
            if destination.exists():
                try:  # Normalize plain state dicts into the expected wrapper.
                    load_kwargs = {"map_location": "cpu"}
                    if self._torch_supports_weights_only:
                        load_kwargs["weights_only"] = True
                    loadnet = torch_module.load(destination, **load_kwargs)
                except Exception:  # pragma: no cover - torch optional or corrupted file
                    pass
                else:
                    normalized = self._normalize_state_dict(loadnet)
                    if normalized is not None:
                        logger.debug(
                            "Normalised Real-ESRGAN weights at %s for compatibility.",
                            destination,
                        )
                        torch_module.save(normalized, destination)
            paths.append(destination)
        return paths

    @staticmethod
    def _extract_state_dict(payload: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        """Heuristic extraction of the actual model weights from mixed payloads."""

        if any(isinstance(key, str) and ("." in key or key.endswith(("weight", "bias"))) for key in payload):
            return payload

        for key in ("params_ema", "ema", "generator_ema", "params", "generator", "state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                extracted = RealESRGANUpscaler._extract_state_dict(candidate)
                if extracted is not None:
                    return extracted

        return None

    @staticmethod
    def _normalize_state_dict(loadnet: Any) -> Optional[Dict[str, Dict[str, Any]]]:
        """Normalise assorted checkpoint payloads into Real-ESRGAN's expected shape."""

        if not isinstance(loadnet, Mapping):
            return None

        candidate = RealESRGANUpscaler._extract_state_dict(loadnet)
        if candidate is None:
            return None

        return {"params": dict(candidate)}

    def _device_tile_attempts(self, policy: RealESRGANDevicePolicy, torch_module) -> List[Tuple[str, bool, int]]:
        attempts: List[Tuple[str, bool, int]] = []

        def add_gpu_attempts() -> None:
            tile_candidates: Tuple[int, ...] = (256, 192, 128, 96, 64, 0)
            for tile in tile_candidates:
                attempts.append(("cuda", False, tile))
                attempts.append(("cuda", True, tile))

        def add_cpu_attempts() -> None:
            tile_candidates_cpu: Tuple[int, ...] = (256, 0)
            attempts.extend(("cpu", False, tile) for tile in tile_candidates_cpu)

        cuda_available = bool(getattr(torch_module, "cuda", None) and torch_module.cuda.is_available())

        if policy == "gpu":
            if cuda_available:
                add_gpu_attempts()
            return attempts

        if policy == "cpu":
            add_cpu_attempts()
            return attempts

        if cuda_available:
            add_gpu_attempts()
        add_cpu_attempts()
        return attempts

    def _invalidate_cache(self, cache_key: tuple[int, RealESRGANDevicePolicy]) -> None:
        self._upsamplers.pop(cache_key, None)
        self._upsampler_details.pop(cache_key, None)
        self._last_backend_details = None

    @staticmethod
    def _extract_attempt(details: Optional[Dict[str, object]]) -> Optional[Tuple[str, bool, int]]:
        if not details:
            return None
        device = details.get("device")
        tile = details.get("tile")
        half = bool(details.get("half")) if "half" in details else False
        if isinstance(device, str) and isinstance(tile, int):
            return (device, half, tile)
        return None

    def _remaining_attempts_exist(
        self,
        cache_key: tuple[int, RealESRGANDevicePolicy],
        policy: RealESRGANDevicePolicy,
        torch_module,
    ) -> bool:
        attempts = self._device_tile_attempts(policy, torch_module)
        failed = self._failed_attempts.get(cache_key, set())
        return any(attempt not in failed for attempt in attempts)

    @staticmethod
    def _is_valid_output(sr, np_module) -> bool:
        if sr is None:
            return False
        if not hasattr(sr, "ndim") or sr.ndim != 3:
            return False
        if sr.shape[2] != 3:
            return False
        if not np_module.isfinite(sr).all():
            return False
        return True

    def _get_upsampler(self, scale: int, *, policy: RealESRGANDevicePolicy):
        cache_key = (scale, policy)
        if cache_key in self._upsamplers:
            self._last_backend_details = self._upsampler_details.get(cache_key)
            return self._upsamplers[cache_key]
        if scale not in self._model_specs:
            raise ValueError(f"Scale {scale} is not supported by {self.label}.")

        backend = self._ensure_backend()
        spec = self._model_specs[scale]
        weight_paths = self._ensure_weights(
            spec.weights,
            backend["load_file_from_url"],
            torch_module=backend["torch"],
        )

        arch_type = spec.arch.lower()
        torch = backend["torch"]
        attempts = self._device_tile_attempts(policy, torch)
        logger.debug("Real-ESRGAN attempt order (policy=%s): %s", policy, attempts)
        if policy == "gpu" and not attempts:
            raise RuntimeError(
                "GPU upscaling is not available because CUDA was not detected. "
                "Try switching to Auto or CPU mode."
            )
        if not attempts:
            raise RuntimeError("No device attempts configured for Real-ESRGAN.")

        model_path: Union[str, List[str]]
        if len(weight_paths) == 1:
            model_path = str(weight_paths[0])
        else:
            model_path = [str(path) for path in weight_paths]

        failed_attempts = self._failed_attempts[cache_key]
        last_error: Optional[Exception] = None

        for device, half, tile in attempts:
            if (device, half, tile) in failed_attempts:
                continue

            if arch_type == "rrdbnet":
                model = backend["RRDBNet"](**spec.arch_args)
            elif arch_type == "sber_rrdbnet":
                model = backend["SberRRDBNet"](**spec.arch_args)
            elif arch_type == "srvgg":
                model = backend["SRVGGNetCompact"](**spec.arch_args)
            else:  # pragma: no cover - defensive guard
                raise ValueError(f"Unsupported Real-ESRGAN architecture '{spec.arch}'.")

            try:
                upsampler = backend["RealESRGANer"](
                    scale=scale,
                    model_path=model_path,
                    model=model,
                    tile=tile,
                    tile_pad=10,
                    pre_pad=0,
                    half=half,
                    device=device,
                    dni_weight=spec.dni_weight,
                )
            except RuntimeError as exc:  # pragma: no cover - device/tile fallback path
                last_error = exc
                failed_attempts.add((device, half, tile))
                logger.warning(
                    "Real-ESRGAN backend failed on device %s (half=%s, tile=%s): %s",
                    device,
                    half,
                    tile,
                    exc,
                )
                continue

            details: Dict[str, object] = {
                "device": device,
                "half": half,
                "tile": tile,
                "policy": policy,
                "scale": scale,
            }
            self._upsamplers[cache_key] = upsampler
            self._upsampler_details[cache_key] = details
            self._last_backend_details = details
            logger.info(
                "Real-ESRGAN backend initialised (variant=%s, scale=%s, device=%s, half=%s, tile=%s)",
                self.id,
                scale,
                device,
                half,
                tile,
            )
            return upsampler

        self._last_backend_details = None
        if policy == "gpu":
            raise RuntimeError(
                "Unable to initialise Real-ESRGAN on the GPU. Try Auto or CPU mode."
            ) from last_error
        if policy == "cpu":
            raise RuntimeError(
                "Unable to initialise Real-ESRGAN on the CPU."
            ) from last_error
        raise RuntimeError(
            "Unable to initialise Real-ESRGAN backend on any device."
        ) from last_error

    def upscale(self, image: Image.Image, scale: int) -> Image.Image:
        if not self.supports_scale(scale):
            available = ", ".join(str(s) for s in self.supported_scales())
            raise ValueError(
                f"Real-ESRGAN variant '{self.label}' does not support {scale}× upscaling. "
                f"Available scales: {available}."
            )
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "NumPy is required for Real-ESRGAN upscaling. Reinstall the project dependencies."
            ) from exc

        policy = self._device_policy
        cache_key = (scale, policy)
        backend = self._ensure_backend()
        torch_module = backend["torch"]

        lr_rgb = image.convert("RGB")
        lr = np.ascontiguousarray(np.array(lr_rgb)[:, :, ::-1])  # RGB -> BGR for Real-ESRGAN

        while True:
            upsampler = self._get_upsampler(scale, policy=policy)
            details = self._last_backend_details or {}
            attempt = self._extract_attempt(details)

            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    sr, _ = upsampler.enhance(lr, outscale=scale)
            except Exception as exc:  # pragma: no cover - runtime fallback path
                if attempt is not None:
                    self._failed_attempts[cache_key].add(attempt)
                self._invalidate_cache(cache_key)
                logger.warning(
                    "Real-ESRGAN inference failed on device %s (half=%s, tile=%s): %s",
                    details.get("device"),
                    details.get("half"),
                    details.get("tile"),
                    exc,
                )
                if not self._remaining_attempts_exist(cache_key, policy, torch_module):
                    raise
                continue

            if not self._is_valid_output(sr, np):
                if attempt is not None:
                    self._failed_attempts[cache_key].add(attempt)
                self._invalidate_cache(cache_key)
                logger.warning(
                    "Real-ESRGAN inference produced invalid data on device %s (half=%s, tile=%s); retrying.",
                    details.get("device"),
                    details.get("half"),
                    details.get("tile"),
                )
                if not self._remaining_attempts_exist(cache_key, policy, torch_module):
                    raise RuntimeError(
                        "Real-ESRGAN produced invalid output on every available backend attempt."
                    )
                continue

            sr_rgb = np.ascontiguousarray(sr[:, :, ::-1])  # BGR -> RGB
            return Image.fromarray(np.clip(sr_rgb, 0, 255).astype("uint8"), mode="RGB")

    def describe_backend(self) -> str:
        details = self._last_backend_details or {}
        device = str(details.get("device") or "unknown").upper()
        precision = "FP16" if details.get("half") else "FP32"
        tile = details.get("tile")
        if not isinstance(tile, int):
            tile_repr = "unknown"
        else:
            tile_repr = str(tile)

        if device == "UNKNOWN":
            return "Real-ESRGAN backend not initialised"

        if device == "CUDA":
            device_label = "CUDA"
        elif device == "CPU":
            device_label = "CPU"
        else:
            device_label = device

        return f"{device_label} {precision}, tile {tile_repr}"


_CONVENTIONAL_UPSCALERS: Tuple[PillowUpscaler, ...] = (
    PillowUpscaler("nearest", "Nearest (fast)", Image.NEAREST),
    PillowUpscaler("bilinear", "Bilinear", Image.BILINEAR),
    PillowUpscaler("bicubic", "Bicubic", Image.BICUBIC),
    PillowUpscaler("lanczos", "Lanczos", Image.LANCZOS),
)


def conventional_upscalers() -> List[PillowUpscaler]:
    return list(_CONVENTIONAL_UPSCALERS)


def ai_upscalers() -> List[RealESRGANUpscaler]:
    variants = list(_REALESRGAN_VARIANTS)
    variants.extend(_discover_custom_variants())
    return [RealESRGANUpscaler(variant) for variant in variants]


def available_upscalers() -> List[Upscaler]:
    return conventional_upscalers() + ai_upscalers()


def upscale_image(method: Upscaler, image: Image.Image, scale: int) -> Image.Image:
    if not method.supports_scale(scale):
        raise ValueError(
            f"{method.label} does not support {scale}× upscaling. "
            f"Available scales: {', '.join(str(s) for s in method.supported_scales())}."
        )
    return method.upscale(image, scale)


def upscale_sequence(
    image: Image.Image,
    steps: Iterable[tuple[Upscaler, int]],
) -> Image.Image:
    result = image
    for upscaler, scale in steps:
        result = upscale_image(upscaler, result, scale)
    return result
