# WQV Wristcam Viewer (PyQt6)

A modern Python implementation of a Casio WQV wrist camera desktop viewer.
It can decode the monochrome ``.pdr``/``.bin`` dumps produced by WQV-1/2 models
and the colour PNG exports produced by later WQV cameras, presenting them in a
Qt 6 desktop interface.

This viewer takes inspiration from and pays tribute to [WQV_PDB_Tools](https://github.com/nnnn2cat/WQV_PDB_Tools), whose reverse-engineering work and tooling seeded much of the research behind WQV-Viewer.

## Features

- ✅ Pure-Python decoder for the packed 4-bit monochrome image stream
- ✅ Automatic clean-up of the corrupt ``DBLK`` chunk found in colour JPEG exports
- ✅ Understands ``WQVLinkDB.PDB`` archives and extracts every embedded frame
- ✅ Responsive PyQt6 UI with thumbnail list, dual-pane preview, metadata pane, and PNG export
- ✅ Built-in 2×/3×/4×/6× upscalers (nearest, bilinear, bicubic, Lanczos) plus an AI option powered by Real-ESRGAN (2×/4×/8×, including Sber's extended models) with manual GPU/CPU selection, automatic fallback, and a configurable stage order
- ✅ Custom toolbar and application icons inspired by the original wrist camera aesthetic
- ✅ Automated unit tests covering the parser and a GUI smoke test

The AI upscaling pipeline builds on the upstream [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) project and incorporates the extended 2×/4×/8× variants published by Sber AI-Forever on [Hugging Face](https://huggingface.co/ai-forever/Real-ESRGAN), bundling the necessary pretrained weights for convenience.

## Screenshot

![Main window with dual preview](resources/screenshots/viewer-overview.png "WQV-Viewer main window showing the thumb list, original preview, and upscaled preview")


## Project layout

```
WQV-Viewer/
├── README.md               # This document
├── pyproject.toml          # Project metadata & dependencies
├── resources/              # Custom toolbar & application icons
├── tests/                  # Pytest test-suite and sample assets
├── wqv_viewer/             # Runtime package (parser + GUI)
└── wqv_upscale_trainer/    # CLI trainer for custom NeoSR weights
```

## Getting started

Ensure you have Python 3.9+ with pip available. From the ``WQV-Viewer``
folder install the package in editable mode (this pulls in Pillow, PyQt6 and
pytest):

```bash
python -m pip install -e .[dev]
```

> ℹ️ The installation pulls in Real-ESRGAN and PyTorch for the AI upscaler.
> The first time you pick the AI method the pretrained weights (≈70 MB) are
> downloaded to ``WQV-Viewer/models/realesrgan`` and re-used afterwards. The
> downloader now prefers the official GitHub release assets and falls back to
> Hugging Face if required. If you are offline, grab the models you need from
> the release page (for example
> [`RealESRGAN_x2plus.pth`](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth),
> [`RealESRGAN_x4plus.pth`](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth),
> [`RealESRGAN_x2.pth`](https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth),
> [`RealESRGAN_x4.pth`](https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth),
> [`RealESRGAN_x8.pth`](https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x8.pth),
> [`realesr-general-x4v3.pth`](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth),
> [`realesr-general-wdn-x4v3.pth`](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth),
> [`RealESRGAN_x4plus_anime_6B.pth`](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth),
> [`realesr-animevideov3.pth`](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth))
> and drop them into that folder ahead of time. At runtime the viewer will try
> your GPU first (including half-precision) and automatically fall back to
> full-precision or CPU execution if the driver rejects the request, avoiding
> the hard crashes older CUDA stacks were triggering. Prefer the new **Device**
> selector in the GUI if you want to force GPU-only or CPU-only execution.

### Real-ESRGAN prerequisites

Real-ESRGAN depends on a small toolchain beyond ``torch`` and ``realesrgan`` itself.
The editable install already brings these in, but if you are composing your own
environment or trimming optional extras make sure the following packages remain
available (mirroring the upstream [`requirements.txt`](https://github.com/xinntao/Real-ESRGAN/blob/master/requirements.txt)):

- ``basicsr>=1.4.2``
- ``facexlib>=0.2.5``
- ``gfpgan>=1.3.5`` (needed for the optional face-enhancement toggle)
- ``opencv-python>=4.5``
- ``torchvision>=0.16``
- ``tqdm``

If you ever uninstall these libraries, reinstall with ``python -m pip install -e .[dev]``
to restore the full AI upscaler feature set.

Launch the viewer with:

```bash
python -m wqv_viewer
```

Use **File → Open WQVLinkDB…** to select a Palm ``.pdb`` archive and populate
the browser with every image inside. Select one or more frames in the list and
choose **File → Export Selected…** to save them as PNG files. The right-hand
pane shows the original frame alongside the selected upscaled preview and the
decoded metadata that is currently captured (raw size and data offset for
monochrome dumps).

### Moving photos from the watch to your PC (Palm OS bridge)

The recommended path is to offload images to a Palm handheld over infrared and
then sync the Palm backup database back to your computer. The outline below
combines the official Casio Palm app with the field notes in
[Sam Mortimer's deep dive](https://segamadebaddecisions.wordpress.com/2024/06/30/the-casio-wrist-camera-wqv-1-absolutely-unwilling-to-share-its-secrets-with-a-pc/).

1. **Prepare a Palm with IR** – Palm m100/m105/m125 or any Palm OS 3.3–4.x
  device with an infrared window works well. Install Palm Desktop/HotSync on
  your PC (Windows XP-era versions run fine inside a VM if you prefer).
2. **Install WQV Link on the Palm** – Download Casio's ``wqvlinkpalm11.lzh``
  archive (mirrors are linked in the blog post), extract the ``.prc``/
  ``.pdb`` payload, add them to the Palm Install Tool, then HotSync. A new
  ``WQV Link`` icon should appear on the handheld.
3. **Beam the photos from the watch** – On the Palm, open ``WQV Link`` and tap
  **Tools → Receive All** so it begins listening. On the watch, align the IR
  window with the Palm and choose ``DATA COMM → SEND → OTHER DEVICE`` (or the
  ``Send All`` entry on later models). Keep the devices steady until the Palm
  shows a completion dialog; a full 100-shot transfer takes a couple of minutes.

  ![Palm WQV Link receiving all images](resources/screenshots/wqv-wrist-camera-palm-recieve-all.png "Palm WQV Link app ready to receive all images over infrared")

4. **HotSync back to the desktop** – Run another HotSync. Palm Desktop writes a
  ``WQVLinkDB.PDB`` backup under
  ``%USERPROFILE%\Documents\Palm OS Desktop\<HotSyncName>\Backup`` (colour
  watches will emit per-image ``CASIJPG*.PDB`` files instead).
5. **Work from a copy of the backup** – Copy ``WQVLinkDB.PDB`` to a separate
  working folder before opening it so the Palm Desktop backup remains intact.
  Point WQV-Viewer at that copy via **File → Open WQVLinkDB…** to unpack and
  browse the images locally. When you prefer standalone bitmaps on disk, you
  can also drop the same ``.pdb`` into
  [WQV_PDB_Tools](https://github.com/nnnn2cat/WQV_PDB_Tools) to batch-export
  BMP/PNG files.

### Upscaling controls

Every image is shown twice: the original resolution on the left and an
upscaled preview on the right. Pick a method (nearest/bilinear/bicubic/Lanczos
or the AI-powered Real-ESRGAN model), a scale factor (2×, 3×, 4×, or 8× when available), a
 device preference (Auto/GPU/CPU), and the stage order using the new **Order**
 dropdown (``Conventional → AI`` or ``AI → Conventional``) from the toolbar above the
previews to refresh the upscale. The Real-ESRGAN weights are cached locally
after the first run under ``WQV-Viewer/models/realesrgan``.
The conventional stage runs first by default, so flipping the order is now a single click when you prefer to apply AI detail enhancement before a final resample.
The AI dropdown now exposes multiple Real-ESRGAN flavours so you can pick the
one that suits your footage:

- **Real-ESRGAN Plus (2×/4×)** – the classic high-quality RRDBNet models.
- **Real-ESRGAN (Sber 2×/4×/8×)** – the ai-forever variants with an extended
  8× RRDBNet stack for aggressive upscaling straight from the Hugging Face release.
- **General x4v3** – the lighter, denoise-tunable SRVGG model (defaults to a
  50/50 blend of the standard and WDN weights).
- **General x4v3 (denoise)** – the pure WDN weights for extra smoothing.
- **Anime x4plus (6B)** – the RRDBNet model tuned for crisp anime line art.
- **AnimeVideo v3** – the ultra-small SRVGG model optimised for anime videos.
- **WQV NeoSR (custom x4)** – loads a local ``wqv_neosr_x4.pth`` checkpoint produced by the trainer.

### Training your own NeoSR weights

The companion CLI, ``wqv-upscale-trainer``, packages everything you need to fine-tune a NeoSR-style generator on high resolution scans.

**1. Gather training material**
- Place lossless source images (PNG/TIFF) in a single directory. The trainer automatically derives synthetic low-resolution crops, so you only need HR references.
- Aim for at least a few hundred crops worth of material; mix in varied lighting to avoid overfitting.

**2. Launch a run**

```bash
wqv-upscale-trainer path/to/source-images run-workspace --scale 4 --steps 100000 \
  --batch-size 6 --grad-accum-steps 4 --tensorboard
```

- ``run-workspace`` is created automatically; logs, checkpoints, and TensorBoard events land there.
- Adjust ``--device`` (``auto``/``cuda``/``cpu``), ``--learning-rate``, or ``--perceptual-weight`` to experiment.

**3. Monitor progress**
- ``trainer.log`` captures per-step losses and validation PSNR.
- If ``--tensorboard`` is enabled, launch ``tensorboard --logdir run-workspace/tensorboard`` to visualise metrics. Alongside the scalar curves, the trainer now publishes a ``comparison/lr_sr_hr`` image grid at the configured interval so you can inspect low-resolution crops, current super-res outputs, and ground-truth targets side-by-side. Adjust ``--image-log-interval`` and ``--image-log-max-samples`` to tune how often and how many examples are emitted.
- Intermediate checkpoints live under ``run-workspace/checkpoints`` in case you want to resume with ``--resume-from``. Each checkpoint is accompanied by a ``*_deploy.pth`` sibling that already contains the slim ``{"params": ...}`` payload ready for the viewer.

**4. Install the trained model in the viewer**
- The final EMA export is written to ``run-workspace/models/wqv_neosr_x4.pth``.
- Copy that file into ``models/realesrgan`` (overwriting the previous version if present).
- The exported weights are already stored in the slim ``{"params": ...}`` format, so the viewer can load **WQV NeoSR (custom x4)** without further conversion.
- If you maintain multiple variants, store them beside the stock weights and swap them in/out or rename ``wqv_neosr_x4.pth`` before launching the viewer.
- Run ``wqv-upscale-trainer --help`` for the full list of configuration switches, including dataset splits, EMA decay, and gradient clipping.

## Running the tests

```bash
python -m pytest
```

The suite uses the sample ``.pdr`` file under ``tests/data`` and runs a headless
(``QT_QPA_PLATFORM=offscreen``) smoke test for the Qt main window.


