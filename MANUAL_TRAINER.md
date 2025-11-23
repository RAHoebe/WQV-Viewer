# WQV Upscale Trainer Manual

`wqv-upscale-trainer` fabricates WQV-style degradations, optimizes RRDB/NeoSR generators, and emits deployable `.pth` weights that the GUI discovers automatically. This rewrite covers the new warm-start workflow alongside the existing resume path.

## 1. Requirements

- Python 3.9+ environment with PyTorch installed (CUDA build recommended for speed).
- Source directory with high-quality reference images (PNG/JPEG/TIFF). Diverse lighting and textures help the synthetic degradations generalize.
- 10 GB+ of free disk for intermediate datasets, checkpoints, and TensorBoard runs.
- Optional: `tensorboard` package for live metrics, plus an NVIDIA GPU for mixed-precision training.

## 2. Preparing Data

1. Place clean source images under a single folder such as `data/highres`. Subdirectories are fine; the trainer recurses.
2. Avoid already upscaled or heavily compressed inputs—whatever artifacts live in the source will become the ground truth.
3. If you need deterministic splits, pre-stage `train/val/test` folders. Otherwise the trainer performs an 80/10/10 split and records it in `dataset_splits.json`.
4. For monochrome-focused models, enable `--monochrome-style` so the LR pathway mimics Casio quantisation and noise distribution.

## 3. CLI & Config Essentials

Base invocation:

```bash
wqv-upscale-trainer <source_dir> <workspace_dir> [options]
```

- `<source_dir>`: folder of pristine images.
- `<workspace_dir>`: location for logs, checkpoints, datasets, and deployable exports. Existing directories are reused.

`--config-only` prints the resolved configuration (including derived defaults and warm-start metadata) as JSON and exits, which is handy for scripted pipelines. The same structure is written to `<workspace>/training_config.json` every run, so reruns can omit repeated CLI flags.

### Frequently used options

| Option | Description |
| --- | --- |
| `--scale {2,4,8}` | Target upscale factor. The trainer builds the matching RRDB/NeoSR backbone automatically. |
| `--steps N` | Total optimisation steps (default `200000`). |
| `--batch-size N` | Mini-batch size per iteration (default `16`). Combine with `--grad-accum-steps` for larger effective batches. |
| `--grad-accum-steps N` | Gradient accumulation factor (default `1`). |
| `--lr LR` | Learning rate for Adam (default `2e-4`). |
| `--device {auto,cuda,cpu}` | Device preference. `auto` tries CUDA then CPU. |
| `--no-amp` | Disable automatic mixed precision (defaults to ON when CUDA is present). |
| `--monochrome-style` | Enable Casio-style grayscale quantisation (see Section 2). |
| `--tensorboard` | Enable TensorBoard logging inside `<workspace>/tensorboard`. |
| `--resume PATH` | Resume optimizer/EMA state from a checkpoint created by this trainer. |
| `--init-weights PATH` | **New** warm-start hook: load an inference/deployable RRDB checkpoint (Real-ESRGAN or custom) before training starts. Mutually exclusive with `--resume`. |
| `--init-arch {auto,rrdb}` | Warm-start architecture hint. Leave at `auto` so the trainer reads metadata; fall back to `rrdb` for legacy checkpoints that lack hints. |

Example:

```bash
wqv-upscale-trainer data/highres runs/x4 --scale 4 --steps 150000 \
  --batch-size 8 --grad-accum-steps 2 --monochrome-style \
  --init-weights models/realesrgan/RealESRGAN_x4plus.pth --tensorboard
```

The sample above fine-tunes the bundled Real-ESRGAN x4 checkpoint, with the warm-start metadata persisted into `training_config.json` so automated reruns can omit the flags.

## 4. Warm Starts vs. Resume

Use warm starts when you want to *seed* a new run from an existing deployable model. Use resume when you want to *continue the same run* with its optimizer state and EMA.

### Warm start (`--init-weights`)

1. Point to an RRDB `.pth` file (e.g., `models/realesrgan/RealESRGAN_x4plus.pth` or `models/custom/wqv_neosr_x8.pth`).
2. The trainer inspects embedded metadata or infers RRDB dimensions (channels, residual blocks, growth channels) before constructing the backbone.
3. If metadata is missing and inference fails, rerun with `--init-arch rrdb` to force RRDB loading; incompatible architectures (e.g., SRVGG) will raise a descriptive error instead of silently corrupting weights.
4. Warm starts populate as many matching parameters as possible and log any missing/unexpected tensors so you can confirm coverage.
5. The resulting workspace stores `init_weights` and `init_arch` inside `training_config.json`, ensuring scripted pipelines (or later `--resume` runs) know the original seed.

### Resume (`--resume`)

1. Choose any checkpoint from `<workspace>/checkpoints` (raw or `_deploy` variant).
2. Supply `--resume path/to/scale4_step80000.pth` alongside the *same* workspace directory you originally used.
3. The trainer restores generator parameters, EMA shadow weights, and optimizer state, then continues stepping from the saved counter. Warm-start flags are ignored in this mode.

Warm-start and resume are mutually exclusive at the CLI level; the parser will reject attempts to use both simultaneously to avoid ambiguous state.

## 5. Workspace Layout

```text
workspace/
├── training_config.json   # frozen config including init/resume paths
├── dataset_splits.json    # per-file train/val/test membership
├── trainer.log            # timestamped run log
├── checkpoints/           # raw + deploy checkpoints with metadata baked in
├── models/
│   └── wqv_neosr_x4.pth   # latest EMA export with embedded RRDB spec
└── tensorboard/           # optional TB event files
```

Copy `models/wqv_neosr_x*.pth` into `models/custom/` within the repo root to expose them inside the viewer’s AI dropdown.

## 6. Monitoring & QA

- **Logs**: `trainer.log` mirrors console output, logging loss components, learning rate, PSNR checkpoints, and warm-start summaries.
- **TensorBoard**: if enabled, run `tensorboard --logdir <workspace>/tensorboard` to inspect scalars and LR/SR/HR image grids.
- **Deployable mirrors**: every raw checkpoint emits a `_deploy` sibling that strips optimizer state but carries the same metadata, making it safe to feed back into `--init-weights` later.
- **Validation/test PSNR**: periodic PSNR calculations use EMA weights to approximate inference quality; both metrics are also written to TensorBoard when active.

## 7. Deployment Pipeline

1. Stop training when validation PSNR plateaus or the visual samples meet your quality bar.
2. Take the EMA export from `<workspace>/models/wqv_neosr_x{scale}.pth` (already metadata-rich).
3. Drop the file into `models/custom/` under the repo root. Include the scale suffix (`_x2/_x4/_x8`) so the viewer’s loader infers scaling correctly.
4. Restart `wqv_viewer` and select the new entry under the AI dropdown (`Custom: filename (×scale)`).
5. Archive the workspace (including `training_config.json`) if you need to re-run or audit the experiment later.

## 8. Troubleshooting

- **Out of memory (GPU)**: lower `--batch-size`, increase `--grad-accum-steps`, or run with `--no-amp` if mixed precision underflows. Warm starts do not change peak memory usage.
- **Warm-start mismatch errors**: ensure the supplied `.pth` is RRDB-based. The log lists unexpected/missing tensor names; persistent mismatches usually indicate SRVGG/other architectures, which are unsupported for training.
- **Resume rejected**: confirm the checkpoint lives under the workspace you passed and that it was generated by this trainer version (metadata fields are backwards compatible).
- **Slow CPU runs**: prefer CUDA when available; otherwise reduce `--steps`, turn off TensorBoard, and lower `--num-workers` so data loading keeps pace.

Combine warm starts with the viewer’s preset system to iterate quickly: fine-tune from a Real-ESRGAN baseline, export the EMA weight, drop it into `models/custom/`, and compare against prior presets inside the GUI.
