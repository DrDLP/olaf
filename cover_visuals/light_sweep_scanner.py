# cover_visuals/light_sweep_scanner.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from olaf_app.cover_visualization_api import BaseCoverEffect, FrameFeatures
from olaf_app.visualization_api import PluginParameter


class CoverLightSweepScannerEffect(BaseCoverEffect):
    """
    Light Sweep / Scanner (cover effect)

    Overview
    --------
    A bright band (scanner) traverses the cover at a given angle. The band position
    is driven by time (t) and speed, while audio amplitude (amp) modulates band width
    and intensity.

    The band uses a smooth falloff ("softness") so it looks like a light sweep rather
    than a hard stripe.

    Audio mapping
    -------------
    - amp -> increases band_width and band_gain
    - t   -> drives band position (via speed)

    Parameters
    ----------
    - angle (deg):
        Band orientation. 0 = vertical band moving left->right.
        90 = horizontal band moving top->bottom.
        45 = diagonal.
    - speed (cycles/s):
        How many sweeps per second. 1.0 = one full pass per second.
    - band_width (px):
        Base band half-width (in pixels). Effective width increases with amp.
    - band_gain:
        Base brightness multiplier for the sweep. Effective gain increases with amp.
    - softness:
        Controls falloff of the band edges. Higher = softer and wider falloff.

    Notes
    -----
    - Input/Output: uint8 RGB (H, W, 3)
    - Compositing uses a screen-like blend for a luminous feel.
    """

    effect_id: str = "cover_light_sweep_scanner"
    effect_name: str = "Light sweep (scanner)"
    effect_description: str = (
        "A moving luminous band sweeps across the cover. "
        "t controls position (speed), amp controls width & intensity. "
        "Params: angle, speed, band_width, band_gain, softness."
    )
    effect_author: str = "DrDLP"
    effect_version: str = "0.1.0"
    effect_max_inputs: int = 1

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)

        # Cached grids (H,W) of normalized coordinates centered around 0
        self._grid_shape: Tuple[int, int] | None = None
        self._u: np.ndarray | None = None  # x in [-0.5..0.5]
        self._v: np.ndarray | None = None  # y in [-0.5..0.5]

    # ------------------------------------------------------------------ #
    # Parameters
    # ------------------------------------------------------------------ #

    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        # Non-zero defaults for all parameters.
        return {
            "angle": PluginParameter(
                name="angle",
                label="Angle (deg)",
                type="float",
                default=25.0,
                minimum=-180.0,
                maximum=180.0,
                step=1.0,
                description="Band angle in degrees. 0=vertical band, 90=horizontal band.",
            ),
            "speed": PluginParameter(
                name="speed",
                label="Speed (cycles/s)",
                type="float",
                default=0.35,
                minimum=0.01,
                maximum=5.0,
                step=0.01,
                description="How many full sweeps per second.",
            ),
            "band_width": PluginParameter(
                name="band_width",
                label="Band width (px)",
                type="float",
                default=70.0,
                minimum=1.0,
                maximum=800.0,
                step=1.0,
                description="Base band half-width in pixels (increases with amp).",
            ),
            "band_gain": PluginParameter(
                name="band_gain",
                label="Band gain",
                type="float",
                default=1.15,
                minimum=0.1,
                maximum=10.0,
                step=0.05,
                description="Base brightness gain (increases with amp).",
            ),
            "softness": PluginParameter(
                name="softness",
                label="Softness",
                type="float",
                default=1.8,
                minimum=0.1,
                maximum=10.0,
                step=0.1,
                description="Falloff softness. Higher = smoother edges.",
            ),
            "amp_curve": PluginParameter(
                name="amp_curve",
                label="Amp curve",
                type="float",
                default=1.1,
                minimum=0.1,
                maximum=3.0,
                step=0.05,
                description="Shapes audio response: shaped_amp = amp ** amp_curve.",
            ),
            "width_amp_scale": PluginParameter(
                name="width_amp_scale",
                label="Width amp scale",
                type="float",
                default=0.8,
                minimum=0.0,
                maximum=5.0,
                step=0.05,
                description="Extra width from amp: width *= 1 + shaped_amp * scale.",
            ),
            "gain_amp_scale": PluginParameter(
                name="gain_amp_scale",
                label="Gain amp scale",
                type="float",
                default=1.0,
                minimum=0.0,
                maximum=5.0,
                step=0.05,
                description="Extra gain from amp: gain *= 1 + shaped_amp * scale.",
            ),
            "tint_color": PluginParameter(
                name="tint_color",
                label="Tint color (R,G,B)",
                type="string",
                default="255,255,255",
                description="Optional band tint color (RGB). White keeps original colors.",
            ),
            "blend_original": PluginParameter(
                name="blend_original",
                label="Blend original",
                type="float",
                default=0.06,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="Mix original back in (0=full effect, 1=original only).",
            ),
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _ensure_uv(self, h: int, w: int) -> None:
        """Cache centered normalized coordinates in [-0.5..0.5]."""
        if self._grid_shape == (h, w) and self._u is not None and self._v is not None:
            return

        # normalized coordinates
        x = (np.arange(w, dtype=np.float32) + 0.5) / float(w) - 0.5
        y = (np.arange(h, dtype=np.float32) + 0.5) / float(h) - 0.5
        self._u = np.tile(x[None, :], (h, 1))
        self._v = np.tile(y[:, None], (1, w))
        self._grid_shape = (h, w)

    @staticmethod
    def _parse_rgb(s: str) -> Tuple[int, int, int]:
        parts = [p.strip() for p in str(s).split(",")]
        if len(parts) != 3:
            return 255, 255, 255
        try:
            r = int(np.clip(int(float(parts[0])), 0, 255))
            g = int(np.clip(int(float(parts[1])), 0, 255))
            b = int(np.clip(int(float(parts[2])), 0, 255))
            return r, g, b
        except Exception:
            return 255, 255, 255

    @staticmethod
    def _screen_blend(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        """Screen blend for luminous look."""
        a = base.astype(np.float32)
        b = overlay.astype(np.float32)
        out = 255.0 - (255.0 - a) * (255.0 - b) / 255.0
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _smooth_band(dist: np.ndarray, width_n: float, softness: float) -> np.ndarray:
        """
        Build a soft band mask from distance to band center line.

        dist: signed/unsigned distance in normalized units.
        width_n: half-width in normalized units.
        softness: controls edge roll-off.
        """
        # Convert distance to [0..1] mask: 1 at center, 0 outside.
        # Use a soft exponential falloff around the width boundary.
        d = np.abs(dist)
        # edge = d/width, clamp; then apply softness
        x = np.clip(d / max(width_n, 1e-6), 0.0, 10.0)
        # Smoothstep-ish: exp falloff gives nice "light" feel
        mask = np.exp(-(x ** (1.0 + softness)))
        return mask.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Main
    # ------------------------------------------------------------------ #

    def apply_to_frame(self, frame_rgb: np.ndarray, t: float, features: FrameFeatures) -> np.ndarray:
        if frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("CoverLightSweepScannerEffect expects uint8 RGB frame (H, W, 3).")

        h, w, _ = frame_rgb.shape
        self._ensure_uv(h=h, w=w)
        assert self._u is not None and self._v is not None

        # Read params
        angle = float(self.config.get("angle", 25.0))
        speed = float(self.config.get("speed", 0.35))
        band_width_px = float(self.config.get("band_width", 70.0))
        band_gain = float(self.config.get("band_gain", 1.15))
        softness = float(self.config.get("softness", 1.8))
        amp_curve = float(self.config.get("amp_curve", 1.1))
        width_amp_scale = float(self.config.get("width_amp_scale", 0.8))
        gain_amp_scale = float(self.config.get("gain_amp_scale", 1.0))
        tint_color = self._parse_rgb(self.config.get("tint_color", "255,255,255"))
        blend_original = float(self.config.get("blend_original", 0.06))

        # Clamp
        speed = float(np.clip(speed, 0.001, 50.0))
        band_width_px = float(np.clip(band_width_px, 1.0, 5000.0))
        band_gain = float(np.clip(band_gain, 0.0, 50.0))
        softness = float(np.clip(softness, 0.05, 20.0))
        amp_curve = float(np.clip(amp_curve, 0.1, 5.0))
        width_amp_scale = float(np.clip(width_amp_scale, 0.0, 20.0))
        gain_amp_scale = float(np.clip(gain_amp_scale, 0.0, 20.0))
        blend_original = float(np.clip(blend_original, 0.0, 1.0))

        # Audio
        amp = float(np.clip(float(getattr(features, "amp", 0.0)), 0.0, 1.0))
        shaped = amp ** amp_curve

        # Effective width/gain (keep baseline non-zero)
        width_eff_px = band_width_px * (1.0 + shaped * width_amp_scale)
        gain_eff = band_gain * (1.0 + shaped * gain_amp_scale)

        # Convert width from pixels to normalized units (~ fraction of min dimension)
        min_dim = float(min(h, w))
        width_n = float(width_eff_px / max(min_dim, 1.0))  # half width in normalized coords

        # Band motion: position along the normal direction goes from -0.7..+0.7
        # Use a saw wave based on t and speed (cycles/s).
        phase = (t * speed) % 1.0
        pos = -0.7 + 1.4 * phase  # covers beyond edges

        # Orientation: angle defines the band direction; we need its normal.
        # With u=x, v=y, a line has normal n=(cos, sin). We place band by dot([u,v],n)=pos.
        rad = np.deg2rad(angle)
        nx = float(np.cos(rad))
        ny = float(np.sin(rad))

        # Signed distance along normal
        dist = (self._u * nx + self._v * ny) - np.float32(pos)

        # Band mask (0..1)
        band = self._smooth_band(dist=dist, width_n=width_n, softness=softness)

        # Build overlay: band tinted and scaled by gain
        tr, tg, tb = tint_color
        overlay = np.zeros_like(frame_rgb, dtype=np.float32)
        overlay[..., 0] = band * float(tr)
        overlay[..., 1] = band * float(tg)
        overlay[..., 2] = band * float(tb)

        overlay *= float(gain_eff)
        overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8)

        # Composite (screen for luminous sweep)
        out = self._screen_blend(frame_rgb, overlay_u8)

        # Mix original back in
        if blend_original > 0.0:
            alpha_eff = float(1.0 - blend_original)
            out = cv2.addWeighted(out, alpha_eff, frame_rgb, 1.0 - alpha_eff, 0.0)

        return out
