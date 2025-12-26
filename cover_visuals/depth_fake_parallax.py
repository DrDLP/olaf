# cover_visuals/depth_fake_parallax.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from olaf_app.cover_visualization_api import BaseCoverEffect, FrameFeatures
from olaf_app.visualization_api import PluginParameter


class CoverDepthFakeParallaxEffect(BaseCoverEffect):
    """
    Depth Fake Parallax (cover effect)

    Overview
    --------
    Creates a cheap "depth map" from the current frame (mainly luminance + a bit of edge energy),
    optionally smooths it, and uses it to remap pixels to simulate a pseudo-3D parallax shift.

    Audio mapping
    -------------
    - amp -> parallax amplitude (in pixels). Quiet = subtle, loud = stronger parallax.

    Parameters
    ----------
    - depth_gain (float):
        Depth contrast. 1.0 = neutral. >1 increases separation between "near" and "far".
        Values around 1.2–2.0 usually feel good.
    - parallax_px (float):
        Max parallax in pixels at amp=1.0 (scaled linearly by amp).
    - direction (enum):
        Direction of the parallax shift (camera pan feel).
    - blur_depth (float):
        Gaussian blur sigma applied to the depth map (smooths noisy depth estimation).

    Notes
    -----
    - Input/Output: uint8 RGB (H, W, 3)
    - Border handling uses REFLECT to avoid black seams.
    """

    effect_id: str = "cover_depth_fake_parallax"
    effect_name: str = "Depth fake parallax"
    effect_description: str = (
        "Pseudo-3D parallax from a cheap depth map (luminance + edges). "
        "Audio amp drives parallax amplitude. Params: depth_gain, parallax_px, direction, blur_depth."
    )
    effect_author: str = "DrDLP"
    effect_version: str = "0.1.0"
    effect_max_inputs: int = 1

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)

        # Cached coordinate grids for cv2.remap
        self._grid_shape: Tuple[int, int] | None = None
        self._x_grid: np.ndarray | None = None
        self._y_grid: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Parameter schema
    # ------------------------------------------------------------------ #

    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        # Non-zero defaults for all parameters.
        return {
            "depth_gain": PluginParameter(
                name="depth_gain",
                label="Depth gain",
                type="float",
                default=1.35,
                minimum=0.1,
                maximum=5.0,
                step=0.05,
                description="Depth contrast. >1 increases near/far separation.",
            ),
            "parallax_px": PluginParameter(
                name="parallax_px",
                label="Parallax (px)",
                type="float",
                default=18.0,
                minimum=1.0,
                maximum=200.0,
                step=1.0,
                description="Maximum shift at amp=1.0, scaled by depth map and amp.",
            ),
            "direction": PluginParameter(
                name="direction",
                label="Direction",
                type="enum",
                default="right",
                choices=[
                    "right", "left", "up", "down",
                    "up_right", "up_left", "down_right", "down_left",
                ],
                description="Parallax shift direction.",
            ),
            "blur_depth": PluginParameter(
                name="blur_depth",
                label="Blur depth (sigma)",
                type="float",
                default=2.2,
                minimum=0.1,
                maximum=30.0,
                step=0.1,
                description="Gaussian blur sigma for smoothing the depth map.",
            ),
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _ensure_grids(self, h: int, w: int) -> None:
        """Build and cache base pixel coordinate grids for cv2.remap."""
        if self._grid_shape == (h, w) and self._x_grid is not None and self._y_grid is not None:
            return

        x = np.arange(w, dtype=np.float32)
        y = np.arange(h, dtype=np.float32)
        self._x_grid = np.tile(x[None, :], (h, 1))
        self._y_grid = np.tile(y[:, None], (1, w))
        self._grid_shape = (h, w)

    @staticmethod
    def _dir_to_unit(direction: str) -> Tuple[float, float]:
        """Map enum direction to a unit vector (dx, dy)."""
        d = str(direction).lower()
        mapping = {
            "right": (1.0, 0.0),
            "left": (-1.0, 0.0),
            "up": (0.0, -1.0),
            "down": (0.0, 1.0),
            "up_right": (1.0, -1.0),
            "up_left": (-1.0, -1.0),
            "down_right": (1.0, 1.0),
            "down_left": (-1.0, 1.0),
        }
        dx, dy = mapping.get(d, (1.0, 0.0))
        # Normalize diagonals
        norm = float(np.hypot(dx, dy))
        if norm <= 1e-6:
            return 1.0, 0.0
        return dx / norm, dy / norm

    # ------------------------------------------------------------------ #
    # Main effect
    # ------------------------------------------------------------------ #

    def apply_to_frame(self, frame_rgb: np.ndarray, t: float, features: FrameFeatures) -> np.ndarray:
        if frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("CoverDepthFakeParallaxEffect expects uint8 RGB frame (H, W, 3).")

        h, w, _ = frame_rgb.shape
        self._ensure_grids(h=h, w=w)
        assert self._x_grid is not None
        assert self._y_grid is not None

        # Read params
        depth_gain = float(self.config.get("depth_gain", 1.35))
        parallax_px = float(self.config.get("parallax_px", 18.0))
        direction = str(self.config.get("direction", "right"))
        blur_depth = float(self.config.get("blur_depth", 2.2))

        depth_gain = float(np.clip(depth_gain, 0.05, 10.0))
        parallax_px = float(np.clip(parallax_px, 0.0, 500.0))
        blur_depth = float(np.clip(blur_depth, 0.0, 100.0))

        # Audio amp -> amplitude
        amp = float(np.clip(float(getattr(features, "amp", 0.0)), 0.0, 1.0))
        amplitude_px = parallax_px * amp
        if amplitude_px <= 0.01:
            return frame_rgb

        # 1) Build a cheap depth map: luminance + edge energy
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)

        # Normalize edge magnitude to 0..1 (robust-ish)
        mag = mag / (np.percentile(mag, 99) + 1e-6)
        mag = np.clip(mag, 0.0, 1.0)

        # Depth hypothesis (simple): bright + structured areas are "near"
        depth = (0.75 * gray + 0.25 * mag).astype(np.float32)

        # Apply depth contrast around mid-gray
        depth = 0.5 + (depth - 0.5) * depth_gain
        depth = np.clip(depth, 0.0, 1.0)

        # Smooth depth (optional)
        if blur_depth > 0.0:
            depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=blur_depth, sigmaY=blur_depth)

        # 2) Parallax remap: far (depth~0) moves little, near (depth~1) moves more
        dx_u, dy_u = self._dir_to_unit(direction)
        shift_x = np.float32(dx_u * amplitude_px)
        shift_y = np.float32(dy_u * amplitude_px)

        # Inverse mapping: output samples from (x - shift*depth, y - shift*depth)
        map_x = (self._x_grid - shift_x * depth).astype(np.float32)
        map_y = (self._y_grid - shift_y * depth).astype(np.float32)

        warped = cv2.remap(
            frame_rgb,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        return warped
