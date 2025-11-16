"""Configuration objects for the WQV upscale trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence


ScaleFactor = Literal[2, 4, 8]


@dataclass
class TrainerConfig:
    """Holds training hyper-parameters and runtime options."""

    source_dir: Path
    workspace: Path
    scale: ScaleFactor
    steps: int = 200_000
    batch_size: int = 16
    learning_rate: float = 2e-4
    betas: Sequence[float] = field(default_factory=lambda: (0.9, 0.99))
    weight_decay: float = 0.0
    patches_per_image: int = 16
    train_split: float = 0.8
    val_split: float = 0.1
    seed: int = 1337
    device: Literal["auto", "cpu", "cuda"] = "auto"
    num_workers: int = 4
    amp: bool = True
    grad_accum_steps: int = 1
    tensorboard: bool = False
    val_interval: int = 5000
    checkpoint_interval: int = 10_000
    log_interval: int = 100
    image_log_interval: int = 2000
    image_log_max_samples: int = 4
    resume_from: Path | None = None
    perceptual_weight: float = 0.1
    l1_weight: float = 1.0
    ema_decay: float = 0.999
    gradient_clip: float = 0.5
    base_resolution: int = 120
    monochrome_style: bool = False
    monochrome_levels: int = 16
    monochrome_noise: float = 0.02

    def resolved_workspace(self) -> Path:
        return self.workspace.expanduser().resolve()

    def resolved_source(self) -> Path:
        return self.source_dir.expanduser().resolve()
