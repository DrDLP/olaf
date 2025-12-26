# cover_visuals/ink_outline_adaptive.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from olaf_app.cover_visualization_api import BaseCoverEffect, FrameFeatures
from olaf_app.visualization_api import PluginParameter


class CoverInkOutlineAdaptiveEffect(BaseCoverEffect):
    """
    Ink Outline (comic) with adaptive threshold + optional neon overlay.

    Overview
    --------
    1) Build a high-contrast "ink" mask using adaptive thresholding (local contrast).
    2) Extract edges from that mask, then thicken them to get bold comic outlines.
    3) Composite the outlines onto the original image.
    4) Optionally add a neon glow/overlay on top of the outlines.

    Audio mapping
    -------------
    - amp -> increases outline thickness and overall outline intensity.

    Parameters
    ----------
    - outline_thickness (px):
        Base thickness of outlines. Effective thickness increases with amp.
    - adaptive_blocksize:
        Neighborhood size for adaptive threshold (must be odd, >=3).
        Larger -> more global threshold, smaller -> more local detail.
    - adaptive_C:
        Constant subtracted from mean/gaussian to tune threshold aggressiveness.
        Higher -> fewer white pixels (more "ink"/more aggressive).
    - outline_color (R,G,B):
        Outline color, usually black for comic ink ("0,0,0") or dark colors.
    - neon_enable:
        If enabled, adds a neon glow on outlines.
    - neon_color (R,G,B):
        Neon overlay color.
    - neon_gain:
        Neon intensity multiplier (also scaled by amp).
    - amp_curve:
        Shapes amp response: shaped_amp = amp ** amp_curve.
    - blend_original:
        Mix original back in (0 = full effect, 1 = original only).

    Notes
    -----
    - Input/Output: uint8 RGB (H, W, 3)
    """

    effect_id: str = "cover_ink_outline_adaptive"
    effect_name: str = "Ink outline (adaptive)"
    effect_description: str = (
        "Comic ink outlines via adaptive threshold + edge extraction, with optional neon overlay. "
        "Audio amp drives outline thickness and intensity."
    )
    effect_author: str = "DrDLP"
    effect_version: str = "0.1.0"
    effect_max_inputs: int = 1

    # ------------------------------------------------------------------ #
    # Parameters
    # ------------------------------------------------------------------ #

    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        # Non-zero defaults for all parameters.
        return {
            "outline_thickness": PluginParameter(
                name="outline_thickness",
                label="Outline thickness (px)",
                type="int",
                default=2,
                minimum=1,
                maximum=20,
                step=1,
                description="Base outline thickness (scaled up with amp).",
            ),
            "adaptive_blocksize": PluginParameter(
                name="adaptive_blocksize",
                label="Adaptive block size",
                type="int",
                default=19,
                minimum=3,
                maximum=101,
                step=2,
                description="Odd neighborhood size for adaptive thresholding.",
            ),
            "adaptive_C": PluginParameter(
                name="adaptive_C",
                label="Adaptive C",
                type="int",
                default=5,
                minimum=-30,
                maximum=30,
                step=1,
                description="Constant subtracted from mean/gaussian (controls aggressiveness).",
            ),
            "outline_color": PluginParameter(
                name="outline_color",
                label="Outline color (R,G,B)",
                type="string",
                default="8,8,8",
                description="Outline color in RGB. Use '0,0,0' for pure black ink.",
            ),
            "outline_gain": PluginParameter(
                name="outline_gain",
                label="Outline gain",
                type="float",
                default=1.25,
                minimum=0.1,
                maximum=10.0,
                step=0.05,
                description="Outline opacity/intensity multiplier (scaled by amp).",
            ),
            "amp_curve": PluginParameter(
                name="amp_curve",
                label="Amp curve",
                type="float",
                default=1.15,
                minimum=0.1,
                maximum=3.0,
                step=0.05,
                description="Shapes audio response: shaped_amp = amp ** amp_curve.",
            ),
            "neon_enable": PluginParameter(
                name="neon_enable",
                label="Neon overlay",
                type="enum",
                default="on",
                choices=["on", "off"],
                description="Enable/disable neon overlay on outlines.",
            ),
            "neon_color": PluginParameter(
                name="neon_color",
                label="Neon color (R,G,B)",
                type="string",
                default="0,255,255",
                description="Neon overlay color (RGB).",
            ),
            "neon_gain": PluginParameter(
                name="neon_gain",
                label="Neon gain",
                type="float",
                default=1.10,
                minimum=0.1,
                maximum=10.0,
                step=0.05,
                description="Neon brightness multiplier (scaled by amp).",
            ),
            "neon_blur": PluginParameter(
                name="neon_blur",
                label="Neon blur (sigma)",
                type="float",
                default=3.2,
                minimum=0.1,
                maximum=30.0,
                step=0.1,
                description="Glow blur sigma for neon overlay.",
            ),
            "blend_original": PluginParameter(
                name="blend_original",
                label="Blend original",
                type="float",
                default=0.10,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="Mix original back in (0=full effect, 1=original only).",
            ),
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_rgb(s: str, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
        parts = [p.strip() for p in str(s).split(",")]
        if len(parts) != 3:
            return fallback
        try:
            r = int(np.clip(int(float(parts[0])), 0, 255))
            g = int(np.clip(int(float(parts[1])), 0, 255))
            b = int(np.clip(int(float(parts[2])), 0, 255))
            return r, g, b
        except Exception:
            return fallback

    # ------------------------------------------------------------------ #
    # Main
    # ------------------------------------------------------------------ #

    def apply_to_frame(self, frame_rgb: np.ndarray, t: float, features: FrameFeatures) -> np.ndarray:
        if frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("CoverInkOutlineAdaptiveEffect expects uint8 RGB frame (H, W, 3).")

        # Read params
        outline_thickness = int(self.config.get("outline_thickness", 2))
        block = int(self.config.get("adaptive_blocksize", 19))
        c_val = int(self.config.get("adaptive_C", 5))
        outline_gain = float(self.config.get("outline_gain", 1.25))
        amp_curve = float(self.config.get("amp_curve", 1.15))
        neon_enable = str(self.config.get("neon_enable", "on")).lower()
        neon_gain = float(self.config.get("neon_gain", 1.10))
        neon_blur = float(self.config.get("neon_blur", 3.2))
        blend_original = float(self.config.get("blend_original", 0.10))

        outline_color = self._parse_rgb(self.config.get("outline_color", "8,8,8"), (8, 8, 8))
        neon_color = self._parse_rgb(self.config.get("neon_color", "0,255,255"), (0, 255, 255))

        # Clamp
        outline_thickness = int(np.clip(outline_thickness, 1, 50))
        block = int(np.clip(block, 3, 201))
        if block % 2 == 0:
            block += 1
        outline_gain = float(np.clip(outline_gain, 0.0, 50.0))
        amp_curve = float(np.clip(amp_curve, 0.1, 5.0))
        neon_gain = float(np.clip(neon_gain, 0.0, 50.0))
        neon_blur = float(np.clip(neon_blur, 0.0, 100.0))
        blend_original = float(np.clip(blend_original, 0.0, 1.0))

        # Audio
        amp = float(np.clip(float(getattr(features, "amp", 0.0)), 0.0, 1.0))
        shaped = amp ** amp_curve

        # Thickness and intensity scale with amp (with a baseline so it never becomes 0)
        thickness_eff = int(np.clip(round(outline_thickness * (1.0 + 1.2 * shaped)), 1, 80))
        intensity = float(outline_gain * (0.35 + 0.65 * shaped))

        # Grayscale + denoise a bit for nicer thresholding
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0, sigmaY=1.0)

        # Adaptive threshold (binary)
        thr = cv2.adaptiveThreshold(
            gray_blur,
            255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=block,
            C=c_val,
        )

        # Extract edges from the thresholded image
        # (Canny on binary is stable and gives crisp outlines)
        edges = cv2.Canny(thr, 50, 150)

        # Thicken outlines
        k = 2 * thickness_eff + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        outlines = cv2.dilate(edges, kernel, iterations=1)

        # Build colored outline layer
        mask_f = outlines.astype(np.float32) / 255.0
        out_layer = np.zeros_like(frame_rgb, dtype=np.float32)

        r, g, b = outline_color
        out_layer[..., 0] = mask_f * float(r)
        out_layer[..., 1] = mask_f * float(g)
        out_layer[..., 2] = mask_f * float(b)
        out_layer *= intensity

        out_layer_u8 = np.clip(out_layer, 0, 255).astype(np.uint8)

        # Composite outlines: for "ink" look, we prefer darkening.
        # We approximate by taking the per-pixel minimum with the outline layer (dark ink).
        base = frame_rgb.copy()
        inked = np.minimum(base, out_layer_u8)

        # Optional neon overlay on outlines
        if neon_enable == "on" and neon_gain > 0.0:
            glow = outlines
            if neon_blur > 0.0:
                glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=neon_blur, sigmaY=neon_blur)

            glow_f = glow.astype(np.float32) / 255.0
            nr, ng, nb = neon_color
            neon = np.zeros_like(frame_rgb, dtype=np.float32)
            neon[..., 0] = glow_f * float(nr)
            neon[..., 1] = glow_f * float(ng)
            neon[..., 2] = glow_f * float(nb)

            neon_intensity = float(neon_gain * (0.25 + 0.75 * shaped))
            neon *= neon_intensity
            neon_u8 = np.clip(neon, 0, 255).astype(np.uint8)

            # Add neon on top (saturating)
            inked = cv2.add(inked, neon_u8)

        # Mix original back in
        if blend_original > 0.0:
            alpha_eff = float(1.0 - blend_original)  # 1 -> inked only
            inked = cv2.addWeighted(inked, alpha_eff, frame_rgb, 1.0 - alpha_eff, 0.0)

        return inked
