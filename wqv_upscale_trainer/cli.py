"""Command line interface for `wqv-upscale-trainer`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import TrainerConfig
from .train import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wqv-upscale-trainer",
        description="Train NeoSR-style super-resolution models tailored to WQV imagery.",
    )
    parser.add_argument("source", type=Path, help="Directory containing high-resolution source images.")
    parser.add_argument("workspace", type=Path, help="Directory where datasets, checkpoints, and logs will be stored.")
    parser.add_argument("--scale", type=int, choices=[2, 4, 8], default=4, help="Upscale factor to train (default: 4).")
    parser.add_argument("--steps", type=int, default=200_000, help="Total optimisation steps to run (default: 200000).")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size (default: 16).")
    parser.add_argument("--patches-per-image", type=int, default=16, help="Synthetic patches drawn per source image (default: 16).")
    parser.add_argument("--train-split", type=float, default=0.8, help="Fraction of source images reserved for training (default: 0.8).")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of source images reserved for validation (default: 0.1).")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for Adam (default: 2e-4).")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.99), metavar=("B1", "B2"), help="Adam beta values (default: 0.9 0.99).")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay (default: 0).")
    parser.add_argument("--perceptual-weight", type=float, default=0.1, help="Weight for perceptual (VGG) loss (default: 0.1).")
    parser.add_argument("--l1-weight", type=float, default=1.0, help="Weight for pixel-wise L1 loss (default: 1.0).")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay for generator parameters (default: 0.999).")
    parser.add_argument("--gradient-clip", type=float, default=0.5, help="Gradient clipping norm (default: 0.5).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Training device preference (default: auto).")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes (default: 4).")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps (default: 1).")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed (default: 1337).")
    parser.add_argument("--val-interval", type=int, default=5000, help="Validation interval in steps (default: 5000).")
    parser.add_argument("--checkpoint-interval", type=int, default=10_000, help="Checkpoint interval in steps (default: 10000).")
    parser.add_argument("--log-interval", type=int, default=100, help="Logging interval in steps (default: 100).")
    parser.add_argument("--image-log-interval", type=int, default=2000, help="TensorBoard image logging interval in steps (default: 2000).")
    parser.add_argument("--image-log-max-samples", type=int, default=4, help="Maximum samples per TensorBoard image grid (default: 4).")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume training from.")
    parser.add_argument("--base-resolution", type=int, default=120, help="Target WQV base resolution (default: 120).")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging under the workspace directory.")
    parser.add_argument("--config-only", action="store_true", help="Emit resolved configuration as JSON and exit.")
    return parser


def parse_config(argv: list[str] | None = None) -> TrainerConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = TrainerConfig(
        source_dir=args.source,
        workspace=args.workspace,
        scale=int(args.scale),
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
        patches_per_image=args.patches_per_image,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        grad_accum_steps=max(1, args.grad_accum_steps),
        tensorboard=args.tensorboard,
        val_interval=args.val_interval,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        image_log_interval=max(0, args.image_log_interval),
        image_log_max_samples=max(1, args.image_log_max_samples),
        resume_from=args.resume,
        perceptual_weight=args.perceptual_weight,
        l1_weight=args.l1_weight,
        ema_decay=args.ema_decay,
        gradient_clip=args.gradient_clip,
        base_resolution=args.base_resolution,
    )
    if args.config_only:
        print(json.dumps(config.__dict__, default=str, indent=2))
        sys.exit(0)
    return config


def main(argv: list[str] | None = None) -> None:
    config = parse_config(argv)
    train_model(config)


if __name__ == "__main__":  # pragma: no cover
    main()
