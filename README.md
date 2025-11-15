# WQV Wristcam Viewer (PyQt6)

A modern Python reimplementation of the Casio WQV wrist camera desktop viewer.
It can decode the monochrome ``.pdr``/``.bin`` dumps produced by WQV-1/2 models
and the colour JPEG exports produced by later WQV cameras, presenting them in a
Qt 6 desktop interface.

## Features

- ✅ Pure-Python decoder for the packed 4-bit monochrome image stream
- ✅ Automatic clean-up of the corrupt ``DBLK`` chunk found in colour JPEG exports
- ✅ Understands ``WQVLinkDB.PDB`` archives and extracts every embedded frame
- ✅ Responsive PyQt6 UI with thumbnail list, dual-pane preview, metadata pane, and PNG export
- ✅ Built-in 2×/3×/4×/6x upscalers (nearest, bilinear, bicubic, Lanczos) plus an AI option powered by Real-ESRGAN (2×/4×) with manual GPU/CPU selection and automatic fallback
- ✅ Custom toolbar and application icons inspired by the original wrist camera aesthetic
- ✅ Automated unit tests covering the parser and a GUI smoke test

The AI upscaling pipeline builds on the upstream [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) project and bundles its pretrained weights for convenience.

## Project layout

```
WQV-Viewer/
├── README.md               # This document
├── pyproject.toml          # Project metadata & dependencies
├── resources/              # Custom toolbar & application icons
├── tests/                  # Pytest test-suite and sample assets
└── wqv_viewer/             # Runtime package (parser + GUI)
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

### Upscaling controls

Every image is shown twice: the original resolution on the left and an
upscaled preview on the right. Pick a method (nearest/bilinear/bicubic/Lanczos
or the AI-powered Real-ESRGAN model), a scale factor (2×, 3×, or 4×), and now a
device preference (Auto/GPU/CPU) from the toolbar above the previews to refresh
the upscale. The Real-ESRGAN weights are cached locally after the first run
under ``WQV-Viewer/models/realesrgan``.
The AI dropdown now exposes multiple Real-ESRGAN flavours so you can pick the
one that suits your footage:

- **Real-ESRGAN Plus (2×/4×)** – the classic high-quality RRDBNet models.
- **General x4v3** – the lighter, denoise-tunable SRVGG model (defaults to a
  50/50 blend of the standard and WDN weights).
- **General x4v3 (denoise)** – the pure WDN weights for extra smoothing.
- **Anime x4plus (6B)** – the RRDBNet model tuned for crisp anime line art.
- **AnimeVideo v3** – the ultra-small SRVGG model optimised for anime videos.

## Running the tests

```bash
python -m pytest
```

The suite uses the sample ``.pdr`` file under ``tests/data`` and runs a headless
(``QT_QPA_PLATFORM=offscreen``) smoke test for the Qt main window.

## Next steps

- Flesh out metadata decoding (timestamps, titles) by reverse-engineering the
  header segment.
- Add a live watch-communication stub if a serial protocol capture becomes
  available.
