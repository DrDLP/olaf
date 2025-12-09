# cover_visuals/pixelate_pulse.py
from __future__ import annotations

from typing import Dict, Any

import cv2
import numpy as np

from olaf_app.cover_visualization_api import BaseCoverEffect, FrameFeatures
from olaf_app.visualization_api import PluginParameter


class CoverPixelatePulseEffect(BaseCoverEffect):
    """
    Audio-reactive pixelation / mosaic effect for cover images.

    Concept
    -------
    The idea is to dynamically reduce the effective resolution of the cover
    based on audio amplitude:

        - When the sound is calm (low amplitude):
            * scale ~ 1.0  -> almost original image.
        - When the sound is strong (high amplitude):
            * scale -> min_scale (e.g. 0.1 .. 0.3)
            * the image becomes a coarse pixelated mosaic.

    Implementation
    --------------
    For each frame:
        1) Compute a scale factor in (min_scale .. 1.0) from features.amp.
        2) Downscale the image by this factor using nearest-neighbor.
        3) Upscale back to the original size (also nearest-neighbor).
        4) Return the pixelated frame.
    """

    # ----------------------- Metadata --------------------------------- #

    effect_id: str = "olaf_cover_pixelate_pulse"
    effect_name: str = "Pixelate / Mosaic pulse"
    effect_description: str = (
        "Audio-reactive pixelation: the image becomes a coarse mosaic on loud "
        "sections and returns to full resolution on quiet parts."
    )
    effect_author: str = "Olaf"
    effect_version: str = "0.1.0"
    effect_max_inputs: int = 1  # typically one stem drives the amplitude

    # -------------------- Parameter schema ---------------------------- #

    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        """
        Describe user-editable parameters for this effect.

        Parameters
        ----------
        min_scale : float
            Minimum spatial scale when amplitude is at maximum.
            A value of 0.1 means the cover is shrunk to 10% of its size
            and then blown back up, producing large blocky pixels.
        amp_curve : float
            Exponent for mapping features.amp to scale:
                - < 1.0: more reactive at low levels,
                - = 1.0: linear,
                - > 1.0: more reactive on peaks.
        """
        return {
            "min_scale": PluginParameter(
                name="min_scale",
                label="Min scale",
                type="float",
                default=0.15,
                minimum=0.02,
                maximum=0.5,
                step=0.01,
                description=(
                    "Smallest resolution scale at maximum amplitude. "
                    "Lower values produce bigger pixel blocks."
                ),
            ),
            "amp_curve": PluginParameter(
                name="amp_curve",
                label="Amplitude curve exponent",
                type="float",
                default=0.8,
                minimum=0.1,
                maximum=3.0,
                step=0.05,
                description=(
                    "Exponent for mapping amplitude to pixelation strength. "
                    "<1.0 = more reactive at low levels, >1.0 = peaks only."
                ),
            ),
        }

    # ------------------------------------------------------------------ #

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        # No persistent state required for this effect.

    # ------------------------------------------------------------------ #

    def apply_to_frame(
        self,
        frame_rgb: np.ndarray,
        t: float,              # noqa: ARG002 (time reserved for future use)
        features: FrameFeatures,
    ) -> np.ndarray:
        """
        Apply audio-driven pixelation to the given frame.

        The algorithm:
          1) Map features.amp in [0, 1] to a scale factor in [min_scale, 1].
          2) Resize the frame down by this scale using INTER_NEAREST.
          3) Resize back up to the original size using INTER_NEAREST.
        """
        # ---------------- Input validation ---------------------------- #

        if (
            frame_rgb.dtype != np.uint8
            or frame_rgb.ndim != 3
            or frame_rgb.shape[2] != 3
        ):
            raise ValueError(
                "CoverPixelatePulseEffect expects uint8 RGB frame (H, W, 3)."
            )

        h, w, _ = frame_rgb.shape

        # ---------------- Read parameters ------------------------------ #

        min_scale = float(self.config.get("min_scale", 0.15))
        amp_curve = float(self.config.get("amp_curve", 0.8))

        # Clamp to safe ranges
        min_scale = float(np.clip(min_scale, 0.02, 0.5))
        amp_curve = float(np.clip(amp_curve, 0.1, 3.0))

        # ---------------- Amplitude mapping ---------------------------- #

        amp_raw = float(getattr(features, "amp", 0.0))
        amp_raw = float(np.clip(amp_raw, 0.0, 1.0))

        # Shape amplitude using a power curve
        amp_shaped = amp_raw**amp_curve

        # Map amplitude to scale:
        #   amp = 0 -> scale = 1.0 (full resolution)
        #   amp = 1 -> scale = min_scale (max pixelation)
        scale = 1.0 - (1.0 - min_scale) * amp_shaped
        scale = float(np.clip(scale, min_scale, 1.0))

        # If scale is effectively 1, skip processing
        if scale >= 0.999:
            return frame_rgb

        # ---------------- Pixelation logic ----------------------------- #

        # Compute target size (at least 1x1)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # Downscale with nearest-neighbor
        small = cv2.resize(
            frame_rgb,
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST,
        )

        # Upscale back to original size with nearest-neighbor
        out = cv2.resize(
            small,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

        # Ensure uint8
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)

        return out
