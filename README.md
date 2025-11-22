# WQV Wristcam Viewer & Trainer

WQV-Viewer is a PyQt6 desktop application and companion NeoSR trainer built for the Casio WQV wrist cameras. It decodes Palm backups, previews captures, upscales them with configurable pipelines, and ships a training toolchain for producing new Real-ESRGAN style weights tuned to wristcam footage.

## Highlights
- **Palm-first parser**: loads WQVLink/WQVColor databases, raw monochrome dumps, individual `CASIJPG*.PDB` files, and exported JPEGs while preserving capture metadata.
- **Color awareness**: when `WQVColorDB.PDB` is opened next to its companion `CASIJPG*.PDB` files the viewer stitches in the original colour JPEGs; missing companions trigger readable placeholder thumbnails plus diagnostics so you know what is absent.
- **Dual-pane GUI**: thumbnail grid, metadata pane, original/upscaled previews, zoom shortcuts, and asynchronous workers keep the UI responsive during heavy AI jobs.
- **Configurable pipelines**: combine Pillow resamplers with Real-ESRGAN/NeoSR checkpoints, choose execution order, device policy (auto GPU fallback), and save/load named pipeline presets directly from the toolbar.
- **Load diagnostics**: the parser collects warnings (missing CAS files, truncated chunks, etc.) and exposes them through **Help → View Load Diagnostics…** so issues are never silent.
- **Exports with provenance**: batch export runs the active pipeline, writes PNGs plus JSON sidecars, deduplicates filenames, and reports progress with cancel controls.
- **Trainer toolchain**: `wqv-upscale-trainer` fabricates WQV-style degradations, manages splits, trains RRDB/NeoSR models, logs to TensorBoard, and emits deployable checkpoints ready for the viewer.

## Interface Preview

![Main window with dual preview](resources/screenshots/viewer-overview.png "WQV-Viewer main window showing the thumbnail list, original preview, and upscaled preview")

![Color capture loaded from CASIJPG companion](resources/screenshots/WQV3-WQV10%20Color%20Viewer.png "WQVColor session paired with CASIJPG records")

## Requirements
- Python 3.9 or newer with pip available on your PATH.
- Desktop platform with Qt 6 support (Windows, macOS, or modern Linux).
- NVIDIA CUDA GPU recommended for Real-ESRGAN acceleration, but CPU mode is available.
- Sufficient disk space for model weights (`models/realesrgan` and `models/custom`).
- For WQV-3/WQV-10 colour sessions, place the matching `CASIJPG*.PDB` files next to `WQVColorDB.PDB`. The viewer can also open a `CASIJPG` archive directly if you only need a single JPEG.

## Installation
1. Clone the repository and open a shell inside `WQV-Viewer`.
2. Install the project in editable mode (this pulls in PyQt6, Torch, Real-ESRGAN, and pytest):
   ```bash
   python -m pip install -e .[dev]
   ```
3. (Optional) Verify that PyTorch detects your GPU: `python -c "import torch; print(torch.cuda.is_available())"`.

> Offline installs: download the Real-ESRGAN weights you need and place them under `models/realesrgan` before launching the viewer. The viewer will skip network downloads when the files already exist.

Prefer a containerized environment? Follow the noVNC-enabled workflow in [`readme_wqv_docker.md`](readme_wqv_docker.md). Warning!, this will a big 40GB+ container because of GPU support.

## Quick Start

Run the desktop application from the project root:

```bash
python -m wqv_viewer
```

1. **Open a Palm database** (`File → Open WQVLinkDB…` or drag & drop). Keep `CASIJPG*.PDB` files next to `WQVColorDB.PDB` so colour thumbnails light up.
2. **Navigate the thumbnails**: multi-select, inspect metadata, and rely on zoom shortcuts to inspect originals/upscaled previews.
3. **Configure the pipeline** using the conventional + AI panels, then save the combo as a preset using the toolbar icons for quick recall.
4. **Monitor diagnostics** in the status bar—warnings enable the **View Load Diagnostics…** action so you can review missing CAS companions or other anomalies.
5. **Export** with `File → Export Selected…` to generate PNG/JSON pairs or delete Palm records (monochrome archives only) using the context menu.

That’s the day-to-day workflow; everything else lives in `MANUAL_VIEWER.md`.

## Detailed Manuals

- [Viewer Manual](MANUAL_VIEWER.md): exhaustive coverage of loading, navigation, diagnostics, presets, exporting, and database hygiene.
- [Trainer Manual](MANUAL_TRAINER.md): dataset prep, CLI options, monitoring, and deployment of new checkpoints.

## Upscaling models

### Bundled Real-ESRGAN variants
- Real-ESRGAN Plus 2x and 4x (classic RRDBNet models).
- Real-ESRGAN Sber 2x, 4x, and 8x (ai-forever RRDBNet weights for aggressive upscaling).
- General x4v3 and General x4v3 WDN (SRVGG weights with optional denoise blend control).
- Anime x4plus 6B and AnimeVideo v3 (tailored for stylised footage but useful for crisp wristcam lines).
- RealESRGAN general-purpose weights live under `models/realesrgan` and can be refreshed manually if new upstream releases appear.

### Repository custom models
- `models/custom/wqv_neosr_x4.pth` is a trainer-produced RRDB variant tuned on WQV monochrome material for 4x enlargements.
- `models/custom/wqv_neosr_x8.pth` extends the same recipe to 8x upscaling by chaining an extra NeoSR upsample stage.
- Both models were trained with high-resolution imagery sourced from the [Flickr2K dataset](https://www.kaggle.com/datasets/daehoyang/flickr2k).
- Both weights are auto-discovered on startup and appear in the AI dropdown as `Custom: wqv_neosr_x4 (x4)` and `Custom: wqv_neosr_x8 (x8)`. Select them whenever you want a wristcam-biased baseline before or after conventional resampling.

### Adding your own weights
- Drop additional `.pth` files into `models/custom`. Include the target scale (`x2`, `x4`, or `x8`) in the filename so the loader can infer supported scales.
- Restart the viewer to pick up new weights. Each file is listed under a `Custom:` label based on its stem so you can tell variants apart.
- Stale copies in `models/realesrgan` are cleaned up automatically when their custom counterpart disappears, keeping the directory tidy.

## Testing

Run the automated tests from the project root:

```bash
python -m pytest
```

The suite covers the parser, pipeline helpers, and a headless Qt smoke test (`QT_QPA_PLATFORM=offscreen`) using the sample assets under `tests/data`.

## Troubleshooting
- **GPU fallback in status bar**: the viewer automatically retries on CPU when the GPU driver rejects a kernel launch; check your CUDA installation if this happens unexpectedly.
- **Missing Qt platform plugin**: ensure `python -m pip install -e .[dev]` completed successfully and that you are running within the same environment.
- **Model not listed in the AI dropdown**: confirm the `.pth` file name contains `x2`, `x4`, or `x8` and that it lives in `models/custom`.
- **Trainer cannot import TensorBoard**: install it explicitly with `python -m pip install tensorboard` when you pass `--tensorboard`.

## Credits

WQV-Viewer builds upon the research shared in the community project [WQV_PDB_Tools](https://github.com/nnnn2cat/WQV_PDB_Tools) and the outstanding Real-ESRGAN ecosystem maintained at [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN). All trademarks belong to their respective owners; please review upstream licenses when redistributing model weights.

## Additional Resources
- [Archived enthusiast write-up on WQV cameras (pages.zoom.co.uk/epayne)](https://web.archive.org/web/20041024174136/http://pages.zoom.co.uk/epayne/index.html)
- [Archived Casio WQV download portal](https://web.archive.org/web/20080430210835/http://world.casio.com/wat/download/en/)
- [Casio WQV Link software v1.1 for Palm OS](https://web.archive.org/web/20080430210835/http://world.casio.com/wat/download/en/)
- [Casio WQV Color for Palm OS v2.2](https://web.archive.org/web/20080421035139/http://world.casio.com/wat/download/en/wqv/3/dl_palm_link.html)

## License
- Distributed under the [Apache License 2.0](LICENSE).


