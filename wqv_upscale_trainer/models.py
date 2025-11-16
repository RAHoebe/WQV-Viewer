"""Model builders for NeoSR-style super-resolution generators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


def save_generator(generator: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"params": generator.state_dict()}, path)
