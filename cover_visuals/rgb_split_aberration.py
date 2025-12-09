# cover_visuals/rgb_split_aberration.py
from __future__ import annotations

from typing import Dict, Any

import numpy as np

from olaf_app.cover_visualization_api import BaseCoverEffect, FrameFeatures
from olaf_app.visualization_api import PluginParameter


class CoverRGBSplitAberrationEffect(BaseCoverEffect):
    """
    Audio-reactive chromatic aberration (RGB split) for cover images.

    Concept
    -------
    The idea is to separate the Red, Green and Blue channels of the cover
    image. When the audio amplitude increases, the Red channel is shifted
    to the left, and the Blue channel to the right, while the Green channel
    stays in place.

    This produces a "chromatic aberration" / "RGB split" look, similar to
    analog glitch or violent impact on strong beats.

    Expected pipeline in the host:
        effect = CoverRGBSplitAberrationEffect(config)
        effect.on_sequence_start(duration, fps)   # optional, no-op here
        for each frame:
            frame = effect.apply_to_frame(frame, t, features)

    Notes
    -----
    * Input frames are RGB uint8 (H, W, 3), as in all cover effects.
    * The host must provide features.amp in [0, 1].
    """

    # ----------------------- Metadata --------------------------------- #

    effect_id: str = "olaf_cover_rgb_split_aberration"
    effect_name: str = "Chromatic aberration (RGB split)"
    effect_description: str = (
        "Audio-reactive RGB channel split: red shifts left, blue shifts right "
        "as amplitude increases, creating a chromatic aberration / glitch look."
    )
    effect_author: str = "Olaf"
    effect_version: str = "0.1.0"
    effect_max_inputs: int = 1

    # -------------------- Parameter schema ---------------------------- #

    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        """
        Describe user-editable parameters for this effect.

        Parameters
        ----------
        max_offset_frac : float
            Maximum horizontal shift (in pixels) expressed as a fraction
            of the shortest image side. For example:
                - 0.02 -> up to 2% of min(H, W),
                - 0.05 -> up to 5% of min(H, W).
        amp_curve : float
            Exponent applied to features.amp when mapping amplitude to
            offset. Lower values (< 1) make the effect more sensitive
            to low amplitudes, higher values (> 1) emphasize peaks.
        """
        return {
            "max_offset_frac": PluginParameter(
                name="max_offset_frac",
                label="Max offset (fraction of min side)",
                type="float",
                default=0.03,
                minimum=0.0,
                maximum=0.25,
                step=0.001,
                description=(
                    "Maximum horizontal split distance as a fraction of the "
                    "shortest image side (red left, blue right)."
                ),
            ),
            "amp_curve": PluginParameter(
                name="amp_curve",
                label="Amplitude curve exponent",
                type="float",
                default=0.7,
                minimum=0.1,
                maximum=3.0,
                step=0.05,
                description=(
                    "Exponent for mapping amplitude to split strength. "
                    "<1.0 = more reactive at low levels, >1.0 = more reactive "
                    "on peaks."
                ),
            ),
        }

    # ------------------------------------------------------------------ #

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        # No persistent state needed for this effect.
        # We keep the RNG from BaseCoverEffect available if we want to
        # add random jitter in future versions.
        return

    # ------------------------------------------------------------------ #

    def apply_to_frame(
        self,
        frame_rgb: np.ndarray,
        t: float,              # noqa: ARG002  (not used yet, reserved)
        features: FrameFeatures,
    ) -> np.ndarray:
        """
        Apply audio-driven RGB splitting to the given frame.

        Algorithm
        ---------
        1. Read features.amp in [0, 1], clamp to [0, 1].
        2. Apply a power curve (amp_curve) to shape the response.
        3. Compute an integer pixel offset proportional to:
               offset_px = max_offset_frac * min(H, W) * amp_shaped
        4. Split RGB channels:
               R: shift LEFT  by offset_px
               G: unchanged
               B: shift RIGHT by offset_px
           using np.roll.
        5. To avoid wrap-around artifacts, zero-out the newly exposed bands
           on the sides for each shifted channel.
        6. Recombine channels and return uint8 RGB frame.
        """
        # ---------------- Input validation ---------------------------- #

        if (
            frame_rgb.dtype != np.uint8
            or frame_rgb.ndim != 3
            or frame_rgb.shape[2] != 3
        ):
            raise ValueError(
                "CoverRGBSplitAberrationEffect expects uint8 RGB frame (H, W, 3)."
            )

        h, w, _ = frame_rgb.shape

        # ---------------- Read parameters ------------------------------ #

        max_offset_frac = float(self.config.get("max_offset_frac", 0.03))
        amp_curve = float(self.config.get("amp_curve", 0.7))

        # Clamp to safe ranges
        max_offset_frac = float(np.clip(max_offset_frac, 0.0, 0.25))
        amp_curve = float(np.clip(amp_curve, 0.1, 3.0))

        # ---------------- Amplitude mapping ---------------------------- #

        amp_raw = float(getattr(features, "amp", 0.0))
        amp_raw = float(np.clip(amp_raw, 0.0, 1.0))

        # Shape amplitude using a power curve
        amp_shaped = amp_raw**amp_curve

        # Compute pixel offset (integer)
        base_pixels = max(1.0, min(h, w) * max_offset_frac)
        offset_px = int(round(base_pixels * amp_shaped))

        # If offset is negligible, keep the original frame
        if offset_px <= 0:
            return frame_rgb

        # ---------------- Channel split logic -------------------------- #

        # Explicit RGB split (frame_rgb is already RGB)
        r = frame_rgb[..., 0]
        g = frame_rgb[..., 1]
        b = frame_rgb[..., 2]

        # Shift red to the LEFT (negative axis-1 roll)
        r_shifted = np.roll(r, -offset_px, axis=1)
        # Shift blue to the RIGHT (positive axis-1 roll)
        b_shifted = np.roll(b, offset_px, axis=1)

        # Remove wrap-around artifacts by zeroing the bands where new pixels
        # "enter" the frame due to np.roll:
        # - Red shifting left: right band becomes newly exposed.
        r_shifted[:, -offset_px:] = 0
        # - Blue shifting right: left band becomes newly exposed.
        b_shifted[:, :offset_px] = 0

        # Green channel stays in place (optional vertical jitter could be
        # added in future versions).
        g_shifted = g

        # Recombine into RGB
        out = np.stack([r_shifted, g_shifted, b_shifted], axis=2)

        # Ensure proper dtype / clipping
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)

        return out
