# WQV Viewer Manual

This manual covers every workflow exposed by `wqv_viewer`, from installing dependencies to exporting upscaled captures and managing Palm databases. Pair it with `MANUAL_TRAINER.md` if you also plan to train custom NeoSR/Real-ESRGAN checkpoints.

## 1. Installation & First Launch

1. **Clone the repo** and switch into the project directory.
2. **Install dependencies**:
   ```bash
   python -m pip install -e .[dev]
   ```
   This pulls in PyQt6, Torch, Real-ESRGAN, Pillow, and pytest. Verify CUDA availability with `python -c "import torch; print(torch.cuda.is_available())"` if you plan to use the GPU.
3. **Optional:** populate `models/realesrgan` and `models/custom` ahead of time when working offline.
4. **Run the viewer**:
   ```bash
   python -m wqv_viewer
   ```
   The window remembers its last geometry and pipeline settings after the first successful exit.

## 2. Loading Wristcam Archives

### 2.1 Supported assets

- Palm backups: `WQVLinkDB.PDB`, `WQVColorDB.PDB`, and `CASIJPG*.PDB` files.
- Raw dumps: `.bin`, `.pdr`, `.wqv`.
- Exported JPEGs: `.jpg`, `.jpeg`, `.jpe` (Casio viewer output).
- Standalone captures: ordinary `.jpg/.jpeg` and `.png` files from any source.

> **Color requirement:** for WQV-3/WQV-10 colour captures, keep the corresponding `CASIJPG*.PDB` files next to `WQVColorDB.PDB`. The viewer auto-detects and uses them to show real colour thumbnails; otherwise it generates a placeholder and records a warning in the diagnostics panel. You can also open a `CASIJPG` PDB directly—each contains one JPEG.

### 2.2 Loading methods

- **File → Open WQVLinkDB…** browses for `.pdb` archives.
- **Drag & drop** a `.pdb` file onto the main window. Invalid drops are ignored with a status explanation.
- **Open Recent** holds the last ten databases; missing files are pruned the moment you try to open them.
- **File → Clear** closes the session but keeps the history so you can reopen quickly.

### 2.3 Standalone PNG/JPEG workflow

- **File → Open Image…** accepts any mixture of PNG and JPEG files. The viewer remembers the last folder you used for image imports separately from Palm databases.
- **Drag & drop** PNG/JPEG files directly from Explorer/Finder—the same handler that processes PDB drops now ingests image files. When you drop both PDBs and images at once, PDBs take priority and loose images are queued right after.
- Each imported file becomes a `WQVImage` entry with metadata fields describing its source path and decoder. PNGs always go through Pillow; JPEGs try Pillow first and fall back to the Palm decoder if the file contains the Casio `DBLK` shim. When a fallback happens you’ll see a status-bar suffix such as “Palm decoder used for test.jpg”.
- Thumbnails for large standalone images are clamped to 200×200 so dropping DSLR shots does not stall the UI. The original pixel resolution remains untouched in the preview pane.
- Once loaded, standalone images behave just like Palm records: they participate in multi-selection, can be exported or upscaled, and show filename/resolution metadata in the side panel. Because they are not part of a Palm archive the **Delete Selected** action stays disabled.

During load the parser removes empty Palm records, reconstructs JPEGs (stripping the proprietary `DBLK` blocks), captures all metadata, and emits diagnostics if anything looked suspicious.

## 3. User Interface Tour

1. **Thumbnail grid (left)**: icon-mode list with multi-selection and context actions (delete etc.).
2. **Metadata panel**: filename, capture timestamp, resolution, and Palm record references update as you change the selection.
3. **Preview panes**: Original (left) and upscaled (right) views share zoom controls. Toggle fit/actual, use CTRL + mouse wheel, or the toolbar/shortcut actions.
4. **Upscaling controls**: conventional stage (resampler + scale), AI stage (Real-ESRGAN/NeoSR selection, scale, ordering), device policy, and a progress widget with cancel button.
5. **Status bar**: shows pipeline summaries, diagnostics, export progress, and GPU fallbacks.
6. **Toolbar shortcuts**: open, export, delete, save/load pipeline presets, clear view, zoom modes, and exit.

## 4. Advanced Loading Diagnostics

- The parser collects warnings (e.g., *“WQVColorDB.PDB: using placeholder for record 3 (missing CASIJPG0025701A.PDB)”*).
- When warnings exist, the status bar surfaces the first summary and **Help → View Load Diagnostics…** becomes active.
- Diagnostics persist until you clear the viewer or load a different database. Use them to identify which CAS files are missing.

## 5. Navigation & Selection Tips

- Click a thumbnail to update both previews.
- CTRL/SHIFT multi-select affects exports and deletions.
- `Home/End` jump to the start/end of the grid; `Ctrl+F` (standard Qt search) doesn’t apply, so rely on the scrollbar or mouse wheel.
- The context menu inherits the same actions as the main toolbar—right-click a thumbnail to delete without moving the cursor to the toolbar.

## 6. Upscaling Pipelines in Detail

### 6.1 Conventional stage

- Choose between nearest, bilinear, bicubic, Lanczos, or custom Pillow resamplers.
- Supported scales appear in the combo (2× through 6×). This stage can be disabled entirely.

### 6.2 AI stage

- Pick from bundled Real-ESRGAN weights (`models/realesrgan`) or custom `.pth` files in `models/custom`.
- Scales depend on the model (2×, 4×, 8×). Unsupported scales are hidden.
- Enable/disable the stage via the checkbox.
- Ordering combo decides whether AI runs before conventional (AI → Conventional) or after.

### 6.3 Device policy

- **Auto**: try GPU first, fall back to CPU transparently.
- **GPU only** / **CPU only**: restricts execution for reproducibility or when CUDA isn’t available.

### 6.4 Pipeline presets

- Click the toolbar icons or use **Pipeline → Save/Load Pipeline Preset…**.
- Saving suggests a name like `AI_NeoSR_8x__Conv_Bicubic_2x` based on the pipeline state. You can adjust before confirming.
- Presets store both stage configurations and the device/order choices. They live in `QSettings` and sync instantly across sessions.

### 6.5 Async execution

- Starting an upscale spawns a worker thread. The progress widget shows a busy indicator plus a Cancel button.
- When the job finishes or fails, the status message reflects the pipeline summary (`AI: RealESRGAN_x4plus GPU→CPU fallback`).

## 7. Exporting Images

1. Select one or many thumbnails.
2. Choose **File → Export Selected…**.
3. Pick a destination directory. The viewer runs the pipeline for each selection and writes:
   - `name.png`: upscaled result.
   - `name.json`: metadata sidecar (capture info, pipeline configuration).
4. Filename collisions get `_001`, `_002`, … suffixes.
5. The last export folder is remembered per session.

## 8. Managing Palm Databases

- Deletion is intentionally disabled when you load `WQVColorDB.PDB` entries with CAS companions, because removing entries without touching the CAS files would orphan data.
- For monochrome (`WQVLinkDB.PDB`) archives, select thumbnails then **Delete Selected**. The viewer confirms the count, deletes the Palm records in-place, reloads the database, and updates the UI.
- If selections span multiple PDB files, the delete action is rejected to avoid cross-file edits.

## 9. Session Persistence

The viewer stores in `QSettings`:

- Last opened database and selection signature.
- Recent file list (max 10 entries).
- Pipeline configuration and last used preset.
- Window geometry, splitter positions, and zoom mode.
- Saved pipeline presets (up to 15 entries).

Everything restores automatically the next time you run `python -m wqv_viewer`.

## 10. Tips & Troubleshooting

- **Missing CAS thumbnails:** copy the `CASIJPG*.PDB` files from your Palm backup into the same directory as `WQVColorDB.PDB` and reopen the database.
- **GPU fallback messages:** check CUDA drivers; the viewer already reran the pipeline on CPU.
- **Presets not saving:** ensure the config directory is writable (Windows: `HKEY_CURRENT_USER\Software\RAHoebe\WQVViewer`).
- **Offscreen testing:** set `QT_QPA_PLATFORM=offscreen` to run the GUI tests headlessly like the CI suite does.

Happy exploring!
