# Performance & disk usage

## Disk usage

Olaf can require **> 12 GB** of disk space due to ML/audio models and dependencies.

If you are tight on space:
- keep only the models you actually use
- archive old projects (especially stems and intermediate exports)

## Rendering is slow (especially 2D cover)

This is expected in many configurations.

Common factors:
- high resolution (e.g. 4K)
- high FPS
- long duration
- heavy 2D cover chains (multiple effects, glow/bloom/post-processing)
- CPU-only mode or limited GPU acceleration

Recommended iteration loop:
1. Export **10–20 seconds** at your target settings
2. Reduce chain complexity while tuning
3. Only then export full duration

## Practical speed tips

- Lower export FPS temporarily while designing
- Reduce preview/export resolution during iteration
- Disable expensive effects while tuning (re-enable for final)
- Avoid stacking too many 2D effects unless needed
