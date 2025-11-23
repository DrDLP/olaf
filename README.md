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

> ℹ️ Note: GitHub Markdown does **not** support embedding a YouTube player directly with `<iframe>`.  
> The usual pattern is to link the video (as above) or to use a thumbnail image that links to YouTube.

Example thumbnail pattern (optional, if you add an image to the repo):

```markdown
[![Olaf Log 1 Demo](path/to/thumbnail.png)](https://youtu.be/5c6w_tLqjCs)
```

---

## Log 3 – Puppies (Sprite‑Based Audio‑Reactive Animations)

**Path:** `standalones/Log 3 (puppies)/`  

This log contains all the scripts and assets needed to create audio‑reactive animations based on **cute chibi dog sprites**:

- Sprite sheets / animation frames
- Timing / sequencing logic
- Audio analysis driving the animation (intensity, beat, etc.)
- Simple pipelines to render sequences or real‑time previews

A showcase video for this log is currently in editing and will be linked here once it is published.

---

## Getting Started

> The exact setup may depend on your environment; the steps below are indicative.

1. **Clone the repository**
   ```bash
   git clone https://github.com/DrDLP/olaf.git
   cd olaf
   ```

2. **Navigate to the log you want to explore**
   ```bash
   cd "standalones/Log 1 (3d visualizers)"
   # or
   cd "standalones/Log 3 (puppies)"
   ```

3. **Install dependencies**  
   Check the log‑specific documentation or script headers for required Python packages and tools  
   (e.g. `pip install -r requirements.txt` if provided).

4. **Run a demo script**  
   Typical pattern (example, actual script names may differ):
   ```bash
   python main.py
   ```
   or
   ```bash
   python render_sequence.py --audio path/to/audio.wav --output out/
   ```

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
