# Quickstart (Windows)

## One-command setup

Olaf includes a bootstrap script that:
- creates a local virtual environment (`.venv`)
- installs dependencies (including torch/torchaudio, demucs, whisper, and visualization packages)
- launches the GUI

This is implemented in `run_olaf.py`. :contentReference[oaicite:0]{index=0}

Typical entry points:
- `olaf.bat` (Windows launcher)
- `run_olaf.py` (bootstrap + run)

## First run checklist

- Make sure you have enough disk space (installation may exceed **12 GB**).
- On first run, installation can take a while (downloads + wheels + models).

## Minimal workflow (validate fast)

1. **Projects**: create a project, import the main audio (+ optional cover)
2. **2D visualizations**: add one simple effect to the chain (optional)
3. **Export**: render a short range (10–20 seconds)

Note: exports can be **slow**, especially for 2D cover effects. Start with a short test export.
