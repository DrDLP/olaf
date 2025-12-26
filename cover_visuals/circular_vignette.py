# cover_visuals/circular_vignette.py
from __future__ import annotations

from typing import Dict, Any

import numpy as np

from olaf_app.cover_visualization_api import BaseCoverEffect, FrameFeatures
from olaf_app.visualization_api import PluginParameter


class CoverCircularVignetteEffect(BaseCoverEffect):
    """
    Audio-reactive circular vignette for cover images.

    This effect:
      - expects an uint8 RGB cover (H, W, 3),
      - uses features.amp in [0, 1] to modulate the radius,
      - keeps a bright spot in the center and darkens the borders.

    Configurable parameters:
      - center_x / center_y: position of the vignette center (relative),
      - min_radius_frac / max_radius_frac: radius range as fractions of the
        shortest side,
      - outer_dim_factor / inner_gain: brightness outside / inside,
      - feather_frac: soft edge width as a fraction of the shortest side,
      - amp_curve: exponent for mapping amplitude to radius.
    """

    effect_id: str = "olaf_cover_circular_vignette"
    effect_name: str = "Circular vignette"
    effect_description: str = (
        "Audio-reactive circular spotlight on the cover, with adjustable "
        "center, radius range, brightness, feather and amplitude curve."
    )
    effect_author: str = "DrDLP"
    effect_version: str = "1.3.0"
    effect_max_inputs: int = 1

    # ------------------------------------------------------------------ #
    # Parameter schema                                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        """
        Describe user-editable parameters for this effect.

        Defaults requested:
          - center_x / center_y = 0.5,
          - min_radius_frac = 0.01,
          - max_radius_frac = 0.3,
          - outer_dim_factor = 0.3,
          - inner_gain = 0.7,
          - feather_frac = 0.05,
          - amp_curve in [0, 1], default 0.5.
        """
        return {
            "center_x": PluginParameter(
                name="center_x",
                label="Center X (relative)",
                type="float",
                default=0.5,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description=(
                    "Horizontal position of the vignette center as a fraction "
                    "of image width (0 = left, 1 = right)."
                ),
            ),
            "center_y": PluginParameter(
                name="center_y",
                label="Center Y (relative)",
                type="float",
                default=0.5,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description=(
                    "Vertical position of the vignette center as a fraction "
                    "of image height (0 = top, 1 = bottom)."
                ),
            ),
            "min_radius_frac": PluginParameter(
                name="min_radius_frac",
                label="Min radius (fraction of min side)",
                type="float",
                default=0.01,
                minimum=0.00,
                maximum=0.50,  # max 0.5 pour garder un effet visible
                step=0.01,
                description=(
                    "Base radius as a fraction of the shortest image side "
                    "when amplitude is low."
                ),
            ),
            "max_radius_frac": PluginParameter(
                name="max_radius_frac",
                label="Max radius (fraction of min side)",
                type="float",
                default=0.30,
                minimum=0.01,
                maximum=0.50,  # max 0.5 pour éviter de tout remplir
                step=0.01,
                description=(
                    "Maximum radius as a fraction of the shortest image side "
                    "when amplitude is at its maximum."
                ),
            ),
            "outer_dim_factor": PluginParameter(
                name="outer_dim_factor",
                label="Outer brightness",
                type="float",
                default=0.30,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description=(
                    "Brightness factor applied to pixels outside the vignette "
                    "(0 = fully black, 1 = unchanged)."
                ),
            ),
            "inner_gain": PluginParameter(
                name="inner_gain",
                label="Inner brightness gain",
                type="float",
                default=0.70,
                minimum=0.0,
                maximum=2.0,
                step=0.01,
                description=(
                    "Brightness multiplier inside the vignette. "
                    "Values <1 darken the center compared to the original, "
                    "values >1 brighten it."
                ),
            ),
            "feather_frac": PluginParameter(
                name="feather_frac",
                label="Feather width (fraction of min side)",
                type="float",
                default=0.05,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description=(
                    "Soft edge width as a fraction of the shortest image side. "
                    "0 = hard edge, higher = smoother fade."
                ),
            ),
            "amp_curve": PluginParameter(
                name="amp_curve",
                label="Amplitude curve exponent",
                type="float",
                default=0.50,
                minimum=0.0,   # 0 autorisé en UI, clampé en interne
                maximum=1.0,
                step=0.01,
                description=(
                    "Exponent for mapping amplitude to radius. "
                    "0 = almost always max radius, 1 = linear. "
                    "Lower values make the effect more reactive at low "
                    "amplitudes."
                ),
            ),
        }

    # ------------------------------------------------------------------ #

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config=config)

        self._last_shape: tuple[int, int] | None = None
        self._coord_x: np.ndarray | None = None
        self._coord_y: np.ndarray | None = None

    # ------------------------------------------------------------------ #

    def apply_to_frame(
        self,
        frame_rgb: np.ndarray,
        t: float,              # noqa: ARG002
        features: FrameFeatures,
    ) -> np.ndarray:
        """
        Apply an audio-driven circular vignette to the given frame.

        Parameters used:
          - center_x / center_y: vignette center (relative in [0, 1]),
          - min_radius_frac / max_radius_frac: radius range as fractions of
            the shortest side,
          - outer_dim_factor / inner_gain: brightness outside / inside,
          - feather_frac: soft edge width as a fraction of the shortest side,
          - amp_curve: exponent for mapping amplitude to [0..1] radius blend.

        There is no amplitude floor: amp = 0 -> min radius, amp = 1 -> max radius.
        """
        # Expect uint8 RGB, same contract as glitch_slices.
        if (
            frame_rgb.dtype != np.uint8
            or frame_rgb.ndim != 3
            or frame_rgb.shape[2] != 3
        ):
            raise ValueError(
                "CoverCircularVignetteEffect expects uint8 RGB frame (H, W, 3)."
            )

        h, w, _ = frame_rgb.shape

        # Build / update coordinate grids
        if (
            self._last_shape != (h, w)
            or self._coord_x is None
            or self._coord_y is None
        ):
            y = np.arange(h, dtype=np.float32)
            x = np.arange(w, dtype=np.float32)
            self._coord_y, self._coord_x = np.meshgrid(y, x, indexing="ij")
            self._last_shape = (h, w)

        # ---------------- Read and clamp parameters -------------------- #

        center_x = float(self.config.get("center_x", 0.5))
        center_y = float(self.config.get("center_y", 0.5))
        min_radius_frac = float(self.config.get("min_radius_frac", 0.01))
        max_radius_frac = float(self.config.get("max_radius_frac", 0.30))
        outer_dim_factor = float(self.config.get("outer_dim_factor", 0.30))
        inner_gain = float(self.config.get("inner_gain", 0.70))
        feather_frac = float(self.config.get("feather_frac", 0.05))
        amp_curve = float(self.config.get("amp_curve", 0.50))

        # Center in [0, 1]
        center_x = float(np.clip(center_x, 0.0, 1.0))
        center_y = float(np.clip(center_y, 0.0, 1.0))

        # Radius fractions and ordering (fractions of min(H, W))
        min_radius_frac = float(np.clip(min_radius_frac, 0.01, 0.50))
        max_radius_frac = float(np.clip(max_radius_frac, 0.01, 0.50))
        if max_radius_frac < min_radius_frac:
            max_radius_frac = min_radius_frac

        # Brightness
        outer_dim_factor = float(np.clip(outer_dim_factor, 0.0, 2.0))
        inner_gain = float(np.clip(inner_gain, 0.0, 4.0))

        # Feather and amplitude curve
        feather_frac = float(max(0.0, feather_frac))
        amp_curve = float(np.clip(amp_curve, 0.0, 1.0))
        # Avoid exponent 0 (0^0 mal défini) : 0 UI -> 0.01 interne
        amp_curve_eff = max(0.01, amp_curve)

        # ---------------- Amplitude mapping ---------------------------- #

        amp_raw = float(getattr(features, "amp", 0.0))
        amp_raw = float(np.clip(amp_raw, 0.0, 1.0))

        # Simple exponent mapping: 0 -> min radius, 1 -> max radius
        amp_shaped = amp_raw**amp_curve_eff

        # ---------------- Geometry ------------------------------------ #

        short_side = float(min(h, w))

        min_radius = min_radius_frac * short_side
        max_radius = max_radius_frac * short_side
        radius = min_radius + (max_radius - min_radius) * amp_shaped

        feather = feather_frac * short_side

        cx = center_x * (w - 1.0)
        cy = center_y * (h - 1.0)

        dx = self._coord_x - cx
        dy = self._coord_y - cy
        dist = np.sqrt(dx * dx + dy * dy)

        # ---------------- Mask computation ----------------------------- #

        # Base mask = outer brightness
        mask = np.full_like(dist, fill_value=outer_dim_factor, dtype=np.float32)

        inner_region = dist <= radius
        outer_region = dist >= (radius + feather)
        mid_region = ~(inner_region | outer_region)

        mask[inner_region] = inner_gain
        mask[outer_region] = outer_dim_factor

        if np.any(mid_region) and feather > 1e-6:
            d_mid = dist[mid_region]
            alpha = (d_mid - radius) / max(feather, 1e-6)
            alpha = np.clip(alpha, 0.0, 1.0)
            mask[mid_region] = (
                (1.0 - alpha) * inner_gain + alpha * outer_dim_factor
            )

        # Safety: if the mask is completely broken, return original frame
        if not np.isfinite(mask).all() or float(mask.max()) <= 0.0:
            return frame_rgb

        frame_f = frame_rgb.astype(np.float32)
        frame_f *= mask[..., np.newaxis]

        out = np.clip(frame_f, 0.0, 255.0).astype(np.uint8)
        return out
