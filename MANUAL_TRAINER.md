# WQV Upscale Trainer Manual

This guide explains how to prepare datasets, run `wqv_upscale_trainer`, monitor progress, and deploy newly trained models back into the viewer. The trainer is designed for NeoSR/Real-ESRGAN style RRDB architectures tailored to Casio WQV wristcam footage.

## 1. Prerequisites

- Python 3.9+ with CUDA-enabled PyTorch if you plan to train on a GPU.
- A collection of high-quality source images (PNG/JPEG/TIFF). Wristcam-like textures work best, but any photographic data can be degraded into synthetic WQV frames.
- At least 10 GB of disk space for datasets, checkpoints, and TensorBoard logs.
- Optional: TensorBoard (`pip install tensorboard`) for visualizing metrics.

## 2. Dataset Preparation

1. Gather high-resolution reference images in a directory, e.g., `data/highres`.
2. Images should have diverse lighting and textures; the trainer randomly crops and augments them.
3. Avoid extremely compressed JPEGs—artifacts will propagate into the “ground truth”.
4. (Optional) Pre-sort into `train/val/test` folders if you need deterministic membership. Otherwise the trainer handles splitting automatically.

## 3. Command Overview

Base syntax:
```bash
wqv-upscale-trainer <source_dir> <workspace_dir> [options]
```
- `<source_dir>`: folder with pristine images.
- `<workspace_dir>`: new or existing workspace; the trainer creates subdirectories inside it.

### Common options
| Option | Description |
| --- | --- |
| `--scale {2,4,8}` | Target upscale factor. Architecture adapts automatically. |
| `--steps N` | Total training steps (default 100k). |
| `--batch-size N` | Micro-batch per device (default 4). Combine with `--grad-accum-steps` for effective batch sizes. |
| `--grad-accum-steps N` | Accumulate gradients to emulate larger batches. |
| `--device {auto,gpu,cpu}` | Preferred device policy. `auto` tries CUDA first. |
| `--precision {amp,float32}` | `amp` enables mixed precision for speed. |
| `--monochrome-style` | Enable Casio-style monochrome simulation in the low-res path. |
| `--monochrome-levels N` | Limit grayscale levels (default 16). |
| `--monochrome-noise σ` | Adds Gaussian noise pre-quantization. |
| `--perceptual-weight λ` | Controls the LPIPS/feature-loss balance. |
| `--tensorboard` | Writes logs under `<workspace>/tensorboard`. |
| `--resume PATH` | Continue from a previous checkpoint. |

### Example
```bash
wqv-upscale-trainer data/highres runs/x4 --scale 4 --steps 150000 \
  --batch-size 8 --grad-accum-steps 2 --monochrome-style \
  --device auto --tensorboard
```

## 4. Workspace Layout

```
runs/x4/
├── config.yaml              # frozen CLI options & dataset stats
├── dataset_splits.json      # mapping of each source file to train/val/test
├── trainer.log              # rolling log with timestamps
├── tensorboard/             # (optional) event files for TB
├── checkpoints/             # periodic .pth checkpoints
├── models/
│   └── wqv_neosr_x4.pth     # best EMA/export-ready checkpoint
└── samples/                 # LR/SR/HR comparison grids (per eval interval)
```

Copy `models/wqv_neosr_x4.pth` into `models/custom/` inside the repo root to make it available in the viewer AI dropdown.

## 5. Monitoring & Evaluation

- **Console log**: shows current step, loss components, learning rate, and ETA.
- **TensorBoard** (if enabled): run `tensorboard --logdir runs/x4/tensorboard` to inspect scalars and image grids.
- **Samples**: PNG triplets demonstrating LR input, SR prediction, and HR ground truth.
- **Checkpoints**: the trainer writes both raw (`step_XYZ.pth`) and deployable (`*_deploy.pth`) versions. Deployables strip optimizer state for faster loading.

## 6. Resuming Training

1. Locate the desired checkpoint (raw or deploy) under `checkpoints/`.
2. Rerun the trainer with `--resume path/to/checkpoint.pth` and the *same* workspace argument.
3. All metadata (step count, EMA state, optimizer) is restored automatically.

## 7. Customizing Degradation Pipeline

- `--blur-kernel-size`, `--jpeg-quality`, and related options mirror Real-ESRGAN defaults; expose them if you need more/less aggressive degradations.
- Monochrome knobs (`--monochrome-style`, `--monochrome-levels`, `--monochrome-noise`) can replicate WQV-1/2 looks when training monochrome models.
- Use `--no-sharpen` or lower `--perceptual-weight` if over-sharpening occurs.

## 8. Deployment Checklist

1. Stop training when validation metrics plateau or the visuals meet your standards.
2. Grab the EMA export from `models/` (or convert a checkpoint via the provided CLI helper if needed).
3. Drop the `.pth` file in `models/custom/` (naming it `my_model_x4.pth` etc.).
4. Restart `wqv_viewer`—the model appears under the AI dropdown as `Custom: my_model_x4` with the inferred scale.
5. Optionally archive the entire workspace for reproducibility.

## 9. Troubleshooting

- **Out of memory**: lower `--batch-size`, increase `--grad-accum-steps`, or switch to `--precision float32` if AMP instability occurs.
- **Divergent loss**: reduce learning rate via `--lr` or increase EMA smoothing.
- **Poor generalization**: expand the source dataset, enable monochrome simulation, or add motion blur/noise if targeting video captures.
- **Slow CPU runs**: restrict transforms (disable heavy augmentations) and consider running fewer steps with frequent checkpoints.

Combine this trainer with the GUI’s preset system to iterate quickly: train, drop the weight into `models/custom`, save a preset referencing the new model, and compare results side-by-side.
