<table>
  <tr>
    <td style="vertical-align: middle; padding-right: 12px;">
      <img src="OLAF.png" alt="OLAF logo" width="50" />
    </td>
    <td style="vertical-align: middle;">
      <h1 style="margin: 0;">Olaf</h1>
    </td>
  </tr>
</table>

Olaf is a Windows-first desktop application that helps automate a **video-ready visuals pipeline** for AI-generated songs (e.g. Suno).

The goal is simple: start from a song (audio + optional cover + lyrics), then build and export a visual composition using:
- **stem separation** (to route visuals to drums/bass/vocals)
- **vocal/lyrics alignment** (word-level timings)
- **3D audio-reactive visual plugins**
- **2D cover effects chain** (audio-reactive post-processing / stylization)
- **lyrics visuals** (karaoke / scroll / overlays)
- **final export** to a rendered video

> Olaf is a personal/experimental tool. Expect rough edges and hardware-dependent performance.

---

## Demos

- **Audio-Reactive 3D Visuals (Mecha-Choir Constellations)**  
  https://www.youtube.com/watch?v=bvN-NeYG6i4
- **Audio-Reactive 2D Cover**  
  https://www.youtube.com/watch?v=wsodUrkk3sQ
- **Audio-Reactive 3D Visuals (SDF Pulsing Glyph Rings)**  
  https://www.youtube.com/watch?v=IMLHwsDYGwA

---

## Platform support

- **Primary target:** Windows
- Other operating systems are **not thoroughly tested**.

---

## Installation (Windows, automatic)

Olaf ships with a bootstrap launcher that:
- creates a local virtual environment (`.venv`)
- installs dependencies
- tries to install **torch + torchaudio** with CUDA wheels first (then falls back to CPU)
- installs optional packages used by some visuals
- installs **Demucs** (stems) and **openai-whisper** (alignment)
- launches the GUI

This behavior is implemented in `run_olaf.py`.

### Recommended way

- Run `olaf.bat` (it activates `.venv` and starts Olaf)

### Alternative

- Run:
  - `python run_olaf.py`

On first run, installation can take a while (downloads + wheels + models).

---

## Disk space (important)

Plan for a **large installation (often > 12 GB)** due to ML/audio models and heavy dependencies.

---

## Performance expectations (important)

Rendering/export can be **very slow**, especially for **2D cover effect chains** (depending on resolution, FPS, effects, and hardware).

**Strong recommendation:** export a short segment first (10–20 seconds) before committing to a full render.

---

## What the app does (workflow)

A typical workflow inside Olaf looks like:

1. **Projects**
   - create/select a project
   - import the main audio track (+ optional cover image)

2. **Stems (optional but recommended)**
   - generate stems (e.g. drums/bass/vocals)
   - use stems for better routing (cleaner beat/pulse triggers, etc.)

3. **Vocal alignment**
   - align lyrics to audio (word-level timings)
   - review and manually adjust phrase/word timing when needed

4. **Visuals**
   - **3D visualizations:** configure audio-reactive 3D plugins and routing
   - **2D cover visualizations:** build an ordered chain of cover effects and routing
   - **Lyrics visuals:** choose a lyrics visualization plugin + style

5. **Export**
   - render the final video composition

---

## Notes & tips

- If you only need beat/pulse, routing to the **drums** stem often produces the most stable trigger.
- If export is too slow, temporarily reduce:
  - export resolution
  - export FPS
  - number of 2D effects (2D is frequently the slowest part)

---

## Troubleshooting (high level)

- If an effect/plugin does not render, verify that its optional dependencies installed successfully.
- If stems/alignment features are missing, check installation logs: Demucs / Whisper are installed explicitly (without pulling torch again).

---

## Documentation

- In-repo documentation lives in `docs/` (and can be published via GitHub Pages if enabled).

---

## Disclaimer

This project is provided as-is, without warranty. Use at your own risk.
=======
# Olaf – Audio‑Reactive Visual Experiments

Olaf is a collection of audio‑reactive visual experiments built with Python and various asset pipelines.  
The repository is organized into standalone “logs”, each focusing on a different style of visualization.

> ⚠️ Work in progress – APIs, folder structure and assets may change frequently.

---

## Repository Structure

- `standalones/Log 1 (3d visualizers)/`  
  Audio‑reactive **3D visualizers** written in Python.
- `standalones/Log 3 (puppies)/`  
  Audio‑reactive **sprite‑based animations** using cute chibi dogs.

More logs and tools may be added over time as experiments evolve.

---

## Log 1 – 3D Audio‑Reactive Visualizers

**Path:** `standalones/Log 1 (3d visualizers)/`  

This log contains Python scripts that generate 3D visualizations driven by audio input (e.g. waveforms, FFT, beat detection).  
The goal is to explore spatial, abstract scenes that respond in real‑time or in offline rendering to music or sound design.

### Demo Video

You can watch a demo of Log 1 here:  
👉 https://youtu.be/5c6w_tLqjCs




## Log 3 – Puppies (Sprite‑Based Audio‑Reactive Animations)

**Path:** `standalones/Log 3 (puppies)/`  

This log contains all the scripts and assets needed to create audio‑reactive animations based on **cute chibi dog sprites**:

- Sprite sheets / animation frames
- Timing / sequencing logic
- Audio analysis driving the animation (intensity, beat, etc.)
- Simple pipelines to render sequences or real‑time previews

You can watch a demo of Log 3 here:  
👉 https://youtu.be/adwtR8TwF6s

---

## Goals & Scope

- Explore different visual languages for audio‑reactive content:
  - 3D abstract scenes
  - 2D sprite‑based character animations
- Keep the code relatively modular so it can be reused in:
  - Music videos
  - VJ setups
  - Experimental art / prototypes

---

## Contributing / Forking

This repo is primarily experimental, but you are welcome to:

- Fork the project and adapt scripts to your own workflow.
- Reuse parts of the pipelines for your own audio‑reactive tools.
- Open issues if you spot bugs or have questions about the structure.

---

## License

See the `LICENSE` file at the root of the repository for licensing details.

Please check asset‑specific licensing notes (if any) before reusing sprites, textures or models in your own projects.