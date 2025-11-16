"""Training driver for WQV-targeted NeoSR models."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from torchvision import models as tv_models
from tqdm.auto import tqdm

from .config import TrainerConfig
from .data import SyntheticDegradationDataset, discover_images, split_datasets
from .models import build_generator, save_generator


logger = logging.getLogger(__name__)


def _configure_logging(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    log_path = workspace / "trainer.log"
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
    logger.info("Logging to %s", log_path)


def _select_device(preference: str) -> torch.device:
    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available.")
    if preference == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


class VGGPerceptualLoss(nn.Module):
    """Perceptual supervision using VGG-19 feature activations."""

    def __init__(self) -> None:
        super().__init__()
        weights = tv_models.VGG19_Weights.DEFAULT
        vgg = tv_models.vgg19(weights=weights).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.vgg = vgg

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        def _norm(x: torch.Tensor) -> torch.Tensor:
            return (x - self.mean) / self.std

        input_features = self.vgg(_norm(input))
        target_features = self.vgg(_norm(target))
        return torch.nn.functional.l1_loss(input_features, target_features)


class EMA:
    """Maintains an exponential moving average copy of model parameters."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup: Dict[str, torch.Tensor] = {}

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        self.backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data.copy_(self.shadow[name])
        try:
            yield
        finally:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                param.data.copy_(self.backup[name])
            self.backup = {}


@torch.no_grad()
def _evaluate_psnr(model: nn.Module, loader: Optional[DataLoader], device: torch.device) -> float:
    if loader is None:
        return 0.0
    total_psnr = 0.0
    total_images = 0
    model.eval()
    for batch in loader:
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        sr = model(lr)
        mse = torch.nn.functional.mse_loss(sr, hr, reduction="none")
        mse_val = mse.flatten(1).mean(dim=1)
        psnr = 10 * torch.log10(1.0 / (mse_val + 1e-8))
        total_psnr += psnr.sum().item()
        total_images += lr.size(0)
    model.train()
    if total_images == 0:
        return 0.0
    return total_psnr / total_images


def _make_dataloader(
    paths: Sequence[Path],
    config: TrainerConfig,
    device: torch.device,
    *,
    augment: bool,
    seed_offset: int,
    patches_per_image: int,
    shuffle: bool,
    drop_last: bool,
    workers: int,
) -> Optional[DataLoader]:
    if not paths:
        return None
    dataset = SyntheticDegradationDataset(
        list(paths),
        scale=config.scale,
        base_resolution=config.base_resolution,
        patches_per_image=patches_per_image,
        seed=config.seed + seed_offset,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=max(0, workers),
        pin_memory=(device.type == "cuda"),
        drop_last=drop_last,
    )


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: Optional[EMA],
) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["generator"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if ema and "ema" in checkpoint:
        ema.shadow = {k: v.to(next(model.parameters()).device) for k, v in checkpoint["ema"].items()}
    logger.info("Resumed from %s (step=%s)", path, checkpoint.get("step", 0))
    return int(checkpoint.get("step", 0))


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: Optional[EMA],
    step: int,
) -> None:
    payload = {
        "generator": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if ema is not None:
        payload["ema"] = {k: v.cpu() for k, v in ema.shadow.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    logger.info("Checkpoint written to %s", path)


def train_model(config: TrainerConfig) -> None:
    workspace = config.resolved_workspace()
    _configure_logging(workspace)
    _set_seed(config.seed)
    device = _select_device(config.device)
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    source_images = discover_images(config.resolved_source())
    if not source_images:
        raise ValueError(f"No training images found under {config.source_dir}")

    train_paths, val_paths, test_paths = split_datasets(
        source_images,
        train_split=config.train_split,
        val_split=config.val_split,
    )
    logger.info(
        "Dataset split -> train: %d, val: %d, test: %d",
        len(train_paths),
        len(val_paths),
        len(test_paths),
    )

    splits_path = workspace / "dataset_splits.json"
    splits_path.write_text(
        json.dumps(
            {
                "train": [str(p) for p in train_paths],
                "val": [str(p) for p in val_paths],
                "test": [str(p) for p in test_paths],
            },
            indent=2,
        )
    )

    train_loader = _make_dataloader(
        train_paths,
        config,
        device,
        augment=True,
        seed_offset=0,
        patches_per_image=config.patches_per_image,
        shuffle=True,
        drop_last=True,
        workers=config.num_workers,
    )
    if train_loader is None:
        raise ValueError("Training split is empty; cannot proceed.")

    eval_patches = max(2, config.patches_per_image // 2)
    val_loader = _make_dataloader(
        val_paths,
        config,
        device,
        augment=False,
        seed_offset=42,
        patches_per_image=eval_patches,
        shuffle=False,
        drop_last=False,
        workers=max(1, config.num_workers // 2),
    )
    test_loader = _make_dataloader(
        test_paths,
        config,
        device,
        augment=False,
        seed_offset=99,
        patches_per_image=eval_patches,
        shuffle=False,
        drop_last=False,
        workers=max(1, config.num_workers // 2),
    )

    generator = build_generator(config.scale).to(device)
    optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=config.learning_rate,
        betas=tuple(config.betas),
        weight_decay=config.weight_decay,
    )
    ema = EMA(generator, decay=config.ema_decay)

    perceptual_loss: Optional[VGGPerceptualLoss] = None
    if config.perceptual_weight > 0:
        perceptual_loss = VGGPerceptualLoss().to(device)

    scaler = amp.GradScaler(enabled=config.amp and device.type == "cuda")
    start_step = 0
    if config.resume_from is not None:
        start_step = _load_checkpoint(config.resume_from, generator, optimizer, ema)

    stats_path = workspace / "training_config.json"
    stats_path.write_text(json.dumps(asdict(config), indent=2, default=str))

    checkpoints_dir = workspace / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if config.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency missing
            raise RuntimeError(
                "TensorBoard logging requested but torch.utils.tensorboard is unavailable. Install the 'tensorboard' package."
            ) from exc
        tb_dir = workspace / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        writer.add_scalar("data/train_images", len(train_paths), start_step)
        if val_loader is not None:
            writer.add_scalar("data/val_images", len(val_paths), start_step)
        if test_loader is not None:
            writer.add_scalar("data/test_images", len(test_paths), start_step)

    generator.train()
    data_iter = iter(train_loader)
    grad_accum = max(1, config.grad_accum_steps)
    optimizer.zero_grad(set_to_none=True)
    accumulated_batches = 0
    progress = tqdm(
        range(start_step, config.steps),
        total=max(0, config.steps - start_step),
        desc="Training",
        dynamic_ncols=True,
    )

    for step in progress:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)

        with amp.autocast(enabled=scaler.is_enabled()):
            sr = generator(lr)
            loss_l1 = torch.nn.functional.l1_loss(sr, hr)
            loss = config.l1_weight * loss_l1
            loss_perc_value = 0.0
            if perceptual_loss is not None and config.perceptual_weight > 0:
                loss_perc = perceptual_loss(sr, hr)
                loss = loss + config.perceptual_weight * loss_perc
                loss_perc_value = float(loss_perc.detach().item())
        loss_value = float(loss.detach().item())
        loss_l1_value = float(loss_l1.detach().item())
        scaled_loss = loss / grad_accum
        scaler.scale(scaled_loss).backward()
        accumulated_batches += 1

        is_last_iteration = (step + 1) == config.steps
        should_step = accumulated_batches >= grad_accum or is_last_iteration
        if should_step:
            if config.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(generator)
            accumulated_batches = 0

        if config.log_interval > 0 and (step + 1) % config.log_interval == 0:
            logger.info(
                "step=%d lr=%.3e loss=%.4f l1=%.4f perc=%.4f",
                step + 1,
                optimizer.param_groups[0]["lr"],
                loss_value,
                loss_l1_value,
                loss_perc_value,
            )
        progress.set_postfix({"loss": loss_value})

        if writer is not None:
            global_step = step + 1
            writer.add_scalar("train/loss", loss_value, global_step)
            writer.add_scalar("train/l1", loss_l1_value, global_step)
            writer.add_scalar("train/perceptual", loss_perc_value, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        if config.val_interval > 0 and val_loader is not None and (step + 1) % config.val_interval == 0:
            with ema.average_parameters(generator):
                psnr = _evaluate_psnr(generator, val_loader, device)
            logger.info("validation step=%d psnr=%.3f dB", step + 1, psnr)
            if writer is not None:
                writer.add_scalar("val/psnr", psnr, step + 1)

        if config.checkpoint_interval > 0 and (step + 1) % config.checkpoint_interval == 0:
            ckpt_path = checkpoints_dir / f"scale{config.scale}_step{step + 1}.pth"
            _save_checkpoint(ckpt_path, generator, optimizer, ema, step + 1)

    final_ckpt = checkpoints_dir / f"scale{config.scale}_final.pth"
    _save_checkpoint(final_ckpt, generator, optimizer, ema, config.steps)

    best_model = workspace / "models" / f"wqv_neosr_x{config.scale}.pth"
    best_model.parent.mkdir(parents=True, exist_ok=True)
    with ema.average_parameters(generator):
        save_generator(generator, best_model)
    logger.info("Saved deployable generator to %s", best_model)

    if test_loader is not None:
        with ema.average_parameters(generator):
            psnr = _evaluate_psnr(generator, test_loader, device)
        logger.info("test psnr=%.3f dB", psnr)
        if writer is not None:
            writer.add_scalar("test/psnr", psnr, config.steps)

    if writer is not None:
        writer.flush()
        writer.close()
