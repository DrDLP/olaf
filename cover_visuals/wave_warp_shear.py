# cover_visuals/wave_warp_shear.py
from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

import cv2
import numpy as np

from olaf_app.cover_visualization_api import BaseCoverEffect, FrameFeatures
from olaf_app.visualization_api import PluginParameter


class CoverWaveWarpShearEffect(BaseCoverEffect):
    """
    Audio-reactive wave warp / shear effect for cover images.

    Concept
    -------
    We horizontally (or vertically) shift image scanlines following a sine wave.
    The strength is driven by the current audio amplitude.

    This effect is designed to be chain-friendly:
      - It keeps the image readable at low amplitudes.
      - It can be blended with the original frame for subtle motion.

    Notes
    -----
    - Input and output are RGB uint8 (H, W, 3).
    - We use cv2.remap for proper border handling and smooth warping.
    """

    effect_id: str = "cover_wave_warp_shear"
    effect_name: str = "Wave warp / shear"
    effect_description: str = "Sine-based scanline shear, driven by audio amplitude."
    effect_author: str = "DrDLP"
    effect_version: str = "0.1.0"
    effect_max_inputs: int = 1

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        # Cache base coordinate grids (recomputed only when resolution changes).
        self._grid_shape: Tuple[int, int] | None = None
        self._x_grid: np.ndarray | None = None
        self._y_grid: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Parameter schema
    # ------------------------------------------------------------------ #

    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        return {
            "orientation": PluginParameter(
                name="orientation",
                label="Orientation",
                type="enum",
                default="horizontal",
                choices=["horizontal", "vertical", "both"],
                description="Direction of the shear (horizontal, vertical, or both).",
            ),
            "max_shift_px": PluginParameter(
                name="max_shift_px",
                label="Max shift (px)",
                type="float",
                default=24.0,
                minimum=0.0,
                maximum=200.0,
                step=1.0,
                description="Maximum scanline displacement at full amplitude.",
            ),
            "amp_curve": PluginParameter(
                name="amp_curve",
                label="Amp curve",
                type="float",
                default=0.9,
                minimum=0.1,
                maximum=3.0,
                step=0.05,
                description="Exponent shaping: <1 boosts low levels, >1 focuses peaks.",
            ),
            "wave_freq": PluginParameter(
                name="wave_freq",
                label="Wave frequency",
                type="float",
                default=2.0,
                minimum=0.0,
                maximum=20.0,
                step=0.1,
                description="Number of sine periods across the image axis.",
            ),
            "speed": PluginParameter(
                name="speed",
                label="Speed",
                type="float",
                default=0.6,
                minimum=0.0,
                maximum=10.0,
                step=0.05,
                description="Wave phase speed (cycles per second).",
            ),
            "phase_offset": PluginParameter(
                name="phase_offset",
                label="Phase offset",
                type="float",
                default=0.0,
                minimum=-1.0,
                maximum=1.0,
                step=0.01,
                description="Additional phase offset (in cycles). Useful for syncing.",
            ),
            "blend_original": PluginParameter(
                name="blend_original",
                label="Blend original",
                type="float",
                default=0.0,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="Mix original frame back in (0=no mix, 1=only original).",
            ),
            "border_mode": PluginParameter(
                name="border_mode",
                label="Border mode",
                type="enum",
                default="reflect",
                choices=["reflect", "replicate", "wrap", "constant"],
                description="How to fill areas that warp outside the image bounds.",
            ),
            "border_value": PluginParameter(
                name="border_value",
                label="Border value (0-255)",
                type="int",
                default=0,
                minimum=0,
                maximum=255,
                step=1,
                description="Constant border fill value (only used when border_mode=constant).",
            ),
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _ensure_grids(self, h: int, w: int) -> None:
        """Build and cache base pixel coordinate grids for cv2.remap."""
        if self._grid_shape == (h, w) and self._x_grid is not None and self._y_grid is not None:
            return

        # float32 grids: (H, W)
        x = np.arange(w, dtype=np.float32)
        y = np.arange(h, dtype=np.float32)
        self._x_grid = np.tile(x[None, :], (h, 1))
        self._y_grid = np.tile(y[:, None], (1, w))
        self._grid_shape = (h, w)

    @staticmethod
    def _cv2_border_mode(mode: str) -> int:
        return {
            "reflect": cv2.BORDER_REFLECT_101,
            "replicate": cv2.BORDER_REPLICATE,
            "wrap": cv2.BORDER_WRAP,
            "constant": cv2.BORDER_CONSTANT,
        }.get(mode, cv2.BORDER_REFLECT_101)

    # ------------------------------------------------------------------ #
    # Main effect
    # ------------------------------------------------------------------ #

    def apply_to_frame(self, frame_rgb: np.ndarray, t: float, features: FrameFeatures) -> np.ndarray:
        if frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("CoverWaveWarpShearEffect expects uint8 RGB frame (H, W, 3).")

        h, w, _ = frame_rgb.shape
        self._ensure_grids(h=h, w=w)
        assert self._x_grid is not None
        assert self._y_grid is not None

        # ---------------- Read parameters ------------------------------ #
        orientation = str(self.config.get("orientation", "horizontal")).lower()
        max_shift_px = float(self.config.get("max_shift_px", 24.0))
        amp_curve = float(self.config.get("amp_curve", 0.9))
        wave_freq = float(self.config.get("wave_freq", 2.0))
        speed = float(self.config.get("speed", 0.6))
        phase_offset = float(self.config.get("phase_offset", 0.0))
        blend_original = float(self.config.get("blend_original", 0.0))
        border_mode = str(self.config.get("border_mode", "reflect")).lower()
        border_value = int(self.config.get("border_value", 0))

        # Clamp to safe ranges
        max_shift_px = float(np.clip(max_shift_px, 0.0, 500.0))
        amp_curve = float(np.clip(amp_curve, 0.1, 5.0))
        wave_freq = float(np.clip(wave_freq, 0.0, 100.0))
        speed = float(np.clip(speed, 0.0, 50.0))
        phase_offset = float(np.clip(phase_offset, -10.0, 10.0))
        blend_original = float(np.clip(blend_original, 0.0, 1.0))
        border_value = int(np.clip(border_value, 0, 255))

        # ---------------- Amplitude mapping ---------------------------- #
        amp = float(np.clip(float(getattr(features, "amp", 0.0)), 0.0, 1.0))
        amp = amp ** amp_curve

        # Early out for silence / no shift
        shift_px = max_shift_px * amp
        if shift_px <= 0.01 or wave_freq <= 0.0:
            return frame_rgb

        # ---------------- Build displacement field --------------------- #
        # Phase expressed in cycles -> convert to radians: 2*pi*(...)
        phase_cycles = (t * speed) + phase_offset

        map_x = self._x_grid
        map_y = self._y_grid

        # Allocate maps (float32) only when needed to avoid modifying cached grids.
        out_map_x = map_x.copy()
        out_map_y = map_y.copy()

        if orientation in ("horizontal", "both"):
            # Shift X based on Y position.
            y_norm = map_y / max(1.0, float(h))
            offset_x = shift_px * np.sin(2.0 * np.pi * (y_norm * wave_freq + phase_cycles))
            out_map_x = out_map_x + offset_x.astype(np.float32)

        if orientation in ("vertical", "both"):
            # Shift Y based on X position.
            x_norm = map_x / max(1.0, float(w))
            # Use a slight phase lead for vertical component to avoid identical motion.
            offset_y = shift_px * np.sin(2.0 * np.pi * (x_norm * wave_freq + phase_cycles + 0.25))
            out_map_y = out_map_y + offset_y.astype(np.float32)

        # ---------------- Warp ---------------------------------------- #
        warped = cv2.remap(
            frame_rgb,
            out_map_x,
            out_map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=self._cv2_border_mode(border_mode),
            borderValue=(border_value, border_value, border_value),
        )

        if blend_original > 0.0:
            # blend_original = 1 -> original only
            alpha = float(1.0 - blend_original)
            warped = cv2.addWeighted(warped, alpha, frame_rgb, 1.0 - alpha, 0.0)

        if warped.dtype != np.uint8:
            warped = np.clip(warped, 0, 255).astype(np.uint8)

        return warped
