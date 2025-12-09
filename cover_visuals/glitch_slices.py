# cover_visuals/glitch_slices.py
from __future__ import annotations

from typing import Dict, Any

import cv2
import numpy as np

from olaf_app.cover_visualization_api import BaseCoverEffect, FrameFeatures
from olaf_app.visualization_api import PluginParameter


class CoverGlitchSlicesEffect(BaseCoverEffect):
    """
    Audio-reactive glitch effect for cover images.

    This effect applies horizontal or vertical band glitches to an RGB frame.
    Glitches are triggered by the per-frame amplitude (features.amp) of the
    routed stem.

    Typical usage in the host:
        effect = CoverGlitchSlicesEffect(config)
        effect.on_sequence_start(duration, fps)
        for each frame:
            frame = effect.apply_to_frame(frame, t, features)

    Notes
    -----
    * The host is responsible for computing `features.amp` in [0, 1] from
      the chosen stem (RMS / energy).
    * The effect is purely 2D (no Qt / widgets), suitable for offline export.
    """

    # ----------------------- Metadata --------------------------------- #

    effect_id: str = "olaf_cover_glitch_slices"
    effect_name: str = "Glitch slices"
    effect_description: str = (
        "Audio-reactive horizontal/vertical glitch bands, "
        "with random offsets and color shifts."
    )
    effect_author: str = "Olaf"
    effect_version: str = "0.1.0"
    effect_max_inputs: int = 1  # one stem typically drives this effect

    # -------------------- Parameter schema ---------------------------- #

    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        """
        Describe user-editable parameters for this effect.

        These parameters are safe defaults and can be mapped to sliders /
        combos in the CoverVisualizationsTab later.
        """
        return {
            "threshold": PluginParameter(
                name="threshold",
                label="Amplitude threshold",
                type="float",
                default=0.6,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="Minimal amplitude required to allow a glitch event.",
            ),
            "probability": PluginParameter(
                name="probability",
                label="Glitch probability",
                type="float",
                default=0.25,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description=(
                    "Probability of triggering a glitch when amplitude exceeds "
                    "the threshold."
                ),
            ),
            "intensity": PluginParameter(
                name="intensity",
                label="Glitch intensity",
                type="float",
                default=0.7,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description=(
                    "Controls slice count, slice size and color shift strength. "
                    "Higher = more aggressive glitches."
                ),
            ),
            "orientation": PluginParameter(
                name="orientation",
                label="Orientation",
                type="enum",
                default="horizontal",
                choices=["horizontal", "vertical", "both"],
                description="Orientation of the glitch bands.",
            ),
            "max_slice_height_frac": PluginParameter(
                name="max_slice_height_frac",
                label="Max slice fraction",
                type="float",
                default=0.18,
                minimum=0.01,
                maximum=0.5,
                step=0.01,
                description=(
                    "Maximum relative height/width of a glitch slice "
                    "(fraction of image height/width)."
                ),
            ),
            "random_seed": PluginParameter(
                name="random_seed",
                label="Random seed",
                type="int",
                default=2024,
                minimum=0,
                maximum=2**31 - 1,
                step=1,
                description="Seed for deterministic glitch patterns.",
            ),
        }

    # -------------------- Lifecycle overrides ------------------------- #

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config=config)

        # We keep a separate RNG so we can re-seed when a new sequence starts.
        seed = int(self.config.get("random_seed", 2024))
        self._base_seed = seed
        self._rng = np.random.default_rng(seed=seed)

    def on_sequence_start(self, duration: float, fps: int) -> None:
        """
        Re-seed RNG at the start of a new render so that the same project
        produces repeatable glitches for identical settings.
        """
        seed = int(self.config.get("random_seed", self._base_seed))
        # Mix base seed with duration/fps to avoid trivial collisions if wanted
        mixed_seed = seed ^ int(duration * 1000) ^ int(fps)
        self._rng = np.random.default_rng(seed=mixed_seed)

    # -------------------- Core effect --------------------------------- #

    def apply_to_frame(
        self,
        frame_rgb: np.ndarray,
        t: float,
        features: FrameFeatures,
    ) -> np.ndarray:
        """
        Apply audio-driven glitch slices to the given frame.

        The algorithm:
        1) Check if current amplitude exceeds threshold.
        2) If not, return frame unchanged.
        3) If yes, use a Bernoulli draw with `probability` to decide if this
           frame should be glitched.
        4) If glitched, create a copy of the frame and randomly select
           a number of slices, shifting them and applying small color offsets.
        """
        if frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("CoverGlitchSlicesEffect expects uint8 RGB frame (H, W, 3).")

        amp = float(getattr(features, "amp", 0.0))
        threshold = float(self.config.get("threshold", 0.6))
        probability = float(self.config.get("probability", 0.25))
        intensity = float(self.config.get("intensity", 0.7))
        orientation = str(self.config.get("orientation", "horizontal")).lower()
        max_slice_frac = float(self.config.get("max_slice_height_frac", 0.18))

        # No glitch if amplitude is below threshold
        if amp < threshold:
            return frame_rgb

        # Randomly decide whether to apply glitch on this frame
        if self._rng.random() > probability:
            return frame_rgb

        # Clamp and shape intensity
        intensity = float(np.clip(intensity, 0.0, 1.0))
        if intensity <= 0.0:
            return frame_rgb

        # Decide which orientation(s) to apply
        if orientation not in ("horizontal", "vertical", "both"):
            orientation = "horizontal"

        out = frame_rgb.copy()

        if orientation in ("horizontal", "both"):
            out = self._apply_glitch_slices(
                out,
                intensity=intensity,
                max_slice_frac=max_slice_frac,
                horizontal=True,
            )

        if orientation in ("vertical", "both"):
            out = self._apply_glitch_slices(
                out,
                intensity=intensity,
                max_slice_frac=max_slice_frac,
                horizontal=False,
            )

        return out

    # -------------------- Internal helpers ---------------------------- #

    def _apply_glitch_slices(
        self,
        frame_rgb: np.ndarray,
        intensity: float,
        max_slice_frac: float,
        horizontal: bool = True,
    ) -> np.ndarray:
        """
        Core slice-based glitch routine.

        If horizontal=True:
            * Choose several horizontal bands (y ranges) and shift them
              along the X axis, with small color shifts.
        If horizontal=False:
            * Same logic but on vertical bands (x ranges), shifting along Y.
        """
        h, w, _ = frame_rgb.shape
        out = frame_rgb.copy()

        # Number of slices scales with intensity
        min_slices = 2
        max_slices = 8
        n_slices = int(round(min_slices + (max_slices - min_slices) * intensity))
        n_slices = max(1, n_slices)

        # Slice height/width fraction scales with intensity
        max_slice_frac = float(np.clip(max_slice_frac, 0.01, 0.5))
        max_slice_pixels = int(max_slice_frac * (h if horizontal else w))
        max_slice_pixels = max(1, max_slice_pixels)

        # Maximum offset for shifting (in pixels)
        max_offset = int((0.02 + 0.08 * intensity) * (w if horizontal else h))
        max_offset = max(0, max_offset)

        # Color shift strength (in 0–255 range)
        color_shift_strength = 0.15 + 0.5 * intensity

        for _ in range(n_slices):
            # Choose slice size and position
            size = int(self._rng.integers(1, max_slice_pixels + 1))

            if horizontal:
                if size >= h:
                    y0 = 0
                    y1 = h
                else:
                    y0 = int(self._rng.integers(0, h - size))
                    y1 = y0 + size
                x0, x1 = 0, w
            else:
                if size >= w:
                    x0 = 0
                    x1 = w
                else:
                    x0 = int(self._rng.integers(0, w - size))
                    x1 = x0 + size
                y0, y1 = 0, h

            # Random shift direction & magnitude
            shift = int(self._rng.integers(-max_offset, max_offset + 1)) if max_offset > 0 else 0

            # Extract region
            if horizontal:
                region = out[y0:y1, x0:x1]
            else:
                region = out[y0:y1, x0:x1]

            if region.size == 0:
                continue

            # Apply spatial shift
            if horizontal:
                shifted = np.roll(region, shift=shift, axis=1)
            else:
                shifted = np.roll(region, shift=shift, axis=0)

            # Apply random color shift
            c_shift = self._rng.integers(-30, 31, size=3) * color_shift_strength
            region_f = shifted.astype(np.float32)

            for c in range(3):
                region_f[..., c] = np.clip(region_f[..., c] + c_shift[c], 0.0, 255.0)

            region_out = region_f.astype(np.uint8)

            # Write back into the frame
            out[y0:y1, x0:x1] = region_out

        return out
