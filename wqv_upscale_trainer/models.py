"""Model builders for NeoSR-style super-resolution generators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


def _ensure_torchvision_functional_tensor() -> None:
    """Provide compatibility shim for torchvision >=0.16."""

    import importlib
    import sys
    import types

    target = "torchvision.transforms.functional_tensor"
    if target in sys.modules:
        return
    try:
        source = importlib.import_module("torchvision.transforms._functional_tensor")
    except ModuleNotFoundError:  # pragma: no cover - older torchvision
        return

    shim = types.ModuleType(target)
    for name in dir(source):
        if name.startswith("__"):
            continue
        setattr(shim, name, getattr(source, name))
    sys.modules[target] = shim


_ensure_torchvision_functional_tensor()

try:  # pragma: no cover - optional dependency
    from neosr.archs.rrdbnet_arch import RRDBNet as NeoSRRRDBNet  # type: ignore
except ImportError:  # pragma: no cover - optional dependency missing
    NeoSRRRDBNet = None

from basicsr.archs.rrdbnet_arch import RRDB, RRDBNet  # type: ignore
from basicsr.archs import arch_util as basicsr_arch_util  # type: ignore


@dataclass
class GeneratorSpec:
    num_in_ch: int = 3
    num_out_ch: int = 3
    num_feat: int = 64
    num_block: int = 23
    num_grow_ch: int = 32


@dataclass
class WarmStartState:
    state_dict: Dict[str, torch.Tensor]
    spec: GeneratorSpec
    scale: Optional[int] = None


class ExtendedRRDBNet(nn.Module):
    """RRDBNet variant that supports scale=8 with an extra upsample stage."""

    def __init__(self, *, scale: int, spec: GeneratorSpec) -> None:
        super().__init__()
        self.scale = scale
        default_init = basicsr_arch_util.default_init_weights  # type: ignore[attr-defined]
        make_layer = basicsr_arch_util.make_layer  # type: ignore[attr-defined]
        pixel_unshuffle = basicsr_arch_util.pixel_unshuffle  # type: ignore[attr-defined]

        self.pixel_unshuffle = pixel_unshuffle
        in_channels = spec.num_in_ch
        if scale == 2:
            in_channels = spec.num_in_ch * 4
        elif scale == 1:
            in_channels = spec.num_in_ch * 16

        self.conv_first = nn.Conv2d(in_channels, spec.num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, spec.num_block, num_feat=spec.num_feat, num_grow_ch=spec.num_grow_ch)
        self.conv_body = nn.Conv2d(spec.num_feat, spec.num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(spec.num_feat, spec.num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(spec.num_feat, spec.num_feat, 3, 1, 1)
        self.conv_up3: Optional[nn.Conv2d]
        if scale >= 8:
            self.conv_up3 = nn.Conv2d(spec.num_feat, spec.num_feat, 3, 1, 1)
        else:
            self.conv_up3 = None
        self.conv_hr = nn.Conv2d(spec.num_feat, spec.num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(spec.num_feat, spec.num_out_ch, 3, 1, 1)
        default_init(
            [
                self.conv_first,
                self.conv_body,
                self.conv_up1,
                self.conv_up2,
                self.conv_hr,
            ],
            0.1,
        )
        if self.conv_up3 is not None:
            default_init([self.conv_up3], 0.1)
        default_init([self.conv_last], 0.1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = self.pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = self.pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        if self.scale >= 4:
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        if self.conv_up3 is not None:
            feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def build_generator(scale: int, *, spec: GeneratorSpec | None = None) -> nn.Module:
    spec = spec or GeneratorSpec()
    if scale <= 4 and NeoSRRRDBNet is not None:
        return NeoSRRRDBNet(
            num_in_ch=spec.num_in_ch,
            num_out_ch=spec.num_out_ch,
            num_feat=spec.num_feat,
            num_block=spec.num_block,
            num_grow_ch=spec.num_grow_ch,
            upscale=scale,
        )
    if scale <= 4:
        return RRDBNet(
            num_in_ch=spec.num_in_ch,
            num_out_ch=spec.num_out_ch,
            num_feat=spec.num_feat,
            num_block=spec.num_block,
            num_grow_ch=spec.num_grow_ch,
            scale=scale,
        )
    return ExtendedRRDBNet(scale=scale, spec=spec)


def _extract_state_dict(payload: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    if any(isinstance(key, str) and ("." in key or key.endswith(("weight", "bias"))) for key in payload):
        return payload
    for candidate_key in ("params", "params_ema", "ema", "generator", "state_dict", "model"):
        candidate = payload.get(candidate_key)
        if isinstance(candidate, Mapping):
            extracted = _extract_state_dict(candidate)
            if extracted is not None:
                return extracted
    return None


def _infer_rrdb_spec(state: Mapping[str, torch.Tensor]) -> GeneratorSpec:
    spec = GeneratorSpec()
    conv_first = state.get("conv_first.weight")
    if isinstance(conv_first, torch.Tensor):
        spec.num_feat = int(conv_first.shape[0])
        in_ch = int(conv_first.shape[1])
        for factor in (16, 4, 1):
            if in_ch % factor == 0:
                candidate = in_ch // factor
                if candidate in (1, 3, 4):
                    spec.num_in_ch = candidate
                    break
        else:
            spec.num_in_ch = in_ch
    conv_last = state.get("conv_last.weight")
    if isinstance(conv_last, torch.Tensor):
        spec.num_out_ch = int(conv_last.shape[0])

    block_ids: set[int] = set()
    for key in state.keys():
        if not key.startswith("body."):
            continue
        parts = key.split(".")
        if len(parts) < 2:
            continue
        try:
            block_ids.add(int(parts[1]))
        except ValueError:
            continue
    if block_ids:
        spec.num_block = max(block_ids) + 1

    grow_tensor = state.get("body.0.rdb1.conv1.weight")
    if grow_tensor is None:
        for name, tensor in state.items():
            if name.endswith("rdb1.conv1.weight") and isinstance(tensor, torch.Tensor):
                grow_tensor = tensor
                break
    if isinstance(grow_tensor, torch.Tensor):
        spec.num_grow_ch = int(grow_tensor.shape[0])
    return spec


def load_warm_start_state(path: Path, *, arch_hint: str = "auto") -> WarmStartState:
    payload = torch.load(path, map_location="cpu")
    meta = {}
    if isinstance(payload, Mapping):
        meta_obj = payload.get("meta") or payload.get("metadata")
        if isinstance(meta_obj, Mapping):
            meta = dict(meta_obj)
    state_mapping: Optional[Mapping[str, Any]] = None
    if isinstance(payload, Mapping):
        params = payload.get("params")
        if isinstance(params, Mapping):
            state_mapping = params
    if state_mapping is None and isinstance(payload, Mapping):
        state_mapping = _extract_state_dict(payload)
    if state_mapping is None:
        raise ValueError(f"{path} does not contain a recognizable RRDB state dict.")

    arch = meta.get("arch") if isinstance(meta.get("arch"), str) else None
    if arch_hint != "auto":
        arch = arch_hint

    if arch is None:
        arch = "rrdb"
    if arch.lower() not in {"rrdb", "rrdbnet", "sber_rrdbnet"}:
        raise ValueError(
            f"Warm-start weights at {path} declare unsupported architecture '{arch}'."
        )

    spec = None
    spec_meta = meta.get("spec")
    if isinstance(spec_meta, Mapping):
        try:
            spec = GeneratorSpec(
                num_in_ch=int(spec_meta.get("num_in_ch", 3)),
                num_out_ch=int(spec_meta.get("num_out_ch", 3)),
                num_feat=int(spec_meta.get("num_feat", 64)),
                num_block=int(spec_meta.get("num_block", 23)),
                num_grow_ch=int(spec_meta.get("num_grow_ch", 32)),
            )
        except Exception as exc:  # pragma: no cover - metadata parsing guard
            raise ValueError(f"Invalid warm-start metadata in {path}: {exc}") from exc
    if spec is None:
        tensor_state = {k: v for k, v in state_mapping.items() if isinstance(v, torch.Tensor)}
        spec = _infer_rrdb_spec(tensor_state)

    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in state_mapping.items()
        if isinstance(tensor, torch.Tensor)
    }
    if not state_dict:
        raise ValueError(f"Warm-start payload at {path} does not contain tensor parameters.")
    scale = None
    scale_meta = meta.get("scale")
    if isinstance(scale_meta, int):
        scale = scale_meta
    return WarmStartState(state_dict=state_dict, spec=spec, scale=scale)


def save_generator(
    generator: nn.Module,
    path: Path,
    *,
    scale: int,
    spec: GeneratorSpec | None = None,
    arch: str = "rrdb",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = generator.state_dict()
    cpu_state = {name: tensor.detach().cpu() for name, tensor in state.items()}
    metadata = {
        "arch": arch,
        "scale": scale,
        "spec": asdict(spec) if spec is not None else None,
    }
    torch.save({"params": cpu_state, "meta": metadata}, path)
