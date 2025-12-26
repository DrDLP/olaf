# cover_visuals/neon_edges.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from olaf_app.cover_visualization_api import BaseCoverEffect, FrameFeatures
from olaf_app.visualization_api import PluginParameter


class CoverNeonEdgesEffect(BaseCoverEffect):
    """
    Neon edges (cover effect)

    Overview
    --------
    Extract edges (Canny or Sobel), thicken them (dilate), add a soft glow (blur),
    and composite them back onto the original image using an additive/screen-like blend.

    Audio mapping
    -------------
    - amp -> edge intensity (edge_gain is multiplied by a shaped amp)

    Parameters
    ----------
    - method:
        "canny" (sharp binary edges) or "sobel" (gradient-based edges).
    - edge_thresh1, edge_thresh2:
        Canny thresholds (only used when method="canny").
    - sobel_ksize:
        Sobel kernel size (only used when method="sobel"). Must be odd: 3/5/7.
    - edge_thickness:
        Dilation size in pixels (thicker neon strokes).
    - glow_radius:
        Blur radius controlling the glow size (bloom-ish).
    - edge_color:
        Neon color as "R,G,B" (0..255). Applied to the edge mask.
    - edge_gain:
        Global edge brightness multiplier.
    - amp_curve:
        Shapes amplitude response: shaped_amp = amp ** amp_curve.
    - blend_mode:
        "add" (cv2.addWeighted-like + saturate) or "screen" (lighter blend).
    - blend_original:
        Mix original back in (0 = fully effect, 1 = original only).

    Notes
    -----
    - Input/Output: uint8 RGB (H, W, 3)
    """

    effect_id: str = "cover_neon_edges"
    effect_name: str = "Neon edges"
    effect_description: str = (
        "Extract edges (Canny/Sobel), thicken (dilate), add glow (blur), then composite "
        "as neon outlines (add/screen). Audio amp drives intensity."
    )
    effect_author: str = "DrDLP"
    effect_version: str = "0.1.0"
    effect_max_inputs: int = 1

    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        # Non-zero defaults for all parameters.
        return {
            "method": PluginParameter(
                name="method",
                label="Edge method",
                type="enum",
                default="canny",
                choices=["canny", "sobel"],
                description="Edge extraction method (Canny or Sobel).",
            ),
            "edge_thresh1": PluginParameter(
                name="edge_thresh1",
                label="Canny threshold 1",
                type="int",
                default=70,
                minimum=0,
                maximum=255,
                step=1,
                description="Lower Canny threshold (method=canny).",
            ),
            "edge_thresh2": PluginParameter(
                name="edge_thresh2",
                label="Canny threshold 2",
                type="int",
                default=160,
                minimum=0,
                maximum=255,
                step=1,
                description="Upper Canny threshold (method=canny).",
            ),
            "sobel_ksize": PluginParameter(
                name="sobel_ksize",
                label="Sobel kernel size",
                type="int",
                default=3,
                minimum=3,
                maximum=7,
                step=2,
                description="Odd Sobel kernel size: 3/5/7 (method=sobel).",
            ),
            "edge_thickness": PluginParameter(
                name="edge_thickness",
                label="Edge thickness (px)",
                type="int",
                default=2,
                minimum=1,
                maximum=20,
                step=1,
                description="Dilation size to thicken the edge strokes.",
            ),
            "glow_radius": PluginParameter(
                name="glow_radius",
                label="Glow radius (px)",
                type="int",
                default=7,
                minimum=1,
                maximum=50,
                step=1,
                description="Blur radius controlling the glow size.",
            ),
            "edge_color": PluginParameter(
                name="edge_color",
                label="Edge color (R,G,B)",
                type="string",
                default="0,255,255",
                description="Neon color in RGB, e.g. '255,0,255'.",
            ),
            "edge_gain": PluginParameter(
                name="edge_gain",
                label="Edge gain",
                type="float",
                default=1.6,
                minimum=0.1,
                maximum=10.0,
                step=0.05,
                description="Global brightness multiplier for edges/glow.",
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
            "blend_mode": PluginParameter(
                name="blend_mode",
                label="Blend mode",
                type="enum",
                default="add",
                choices=["add", "screen"],
                description="How neon is composited over the original.",
            ),
            "blend_original": PluginParameter(
                name="blend_original",
                label="Blend original",
                type="float",
                default=0.08,
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
    def _parse_rgb(s: str) -> Tuple[int, int, int]:
        """
        Parse a "R,G,B" string into ints in [0..255].
        Accepts spaces.
        """
        parts = [p.strip() for p in str(s).split(",")]
        if len(parts) != 3:
            return 0, 255, 255
        try:
            r = int(np.clip(int(float(parts[0])), 0, 255))
            g = int(np.clip(int(float(parts[1])), 0, 255))
            b = int(np.clip(int(float(parts[2])), 0, 255))
            return r, g, b
        except Exception:
            return 0, 255, 255

    @staticmethod
    def _screen_blend(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        """
        Screen blend in uint8 space:
        screen(a,b) = 255 - ((255-a)*(255-b))/255
        """
        a = base.astype(np.float32)
        b = overlay.astype(np.float32)
        out = 255.0 - (255.0 - a) * (255.0 - b) / 255.0
        return np.clip(out, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # Main
    # ------------------------------------------------------------------ #

    def apply_to_frame(self, frame_rgb: np.ndarray, t: float, features: FrameFeatures) -> np.ndarray:
        if frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("CoverNeonEdgesEffect expects uint8 RGB frame (H, W, 3).")

        # Read params
        method = str(self.config.get("method", "canny")).lower()
        th1 = int(self.config.get("edge_thresh1", 70))
        th2 = int(self.config.get("edge_thresh2", 160))
        sobel_ksize = int(self.config.get("sobel_ksize", 3))
        edge_thickness = int(self.config.get("edge_thickness", 2))
        glow_radius = int(self.config.get("glow_radius", 7))
        edge_color = self._parse_rgb(self.config.get("edge_color", "0,255,255"))
        edge_gain = float(self.config.get("edge_gain", 1.6))
        amp_curve = float(self.config.get("amp_curve", 1.1))
        blend_mode = str(self.config.get("blend_mode", "add")).lower()
        blend_original = float(self.config.get("blend_original", 0.08))

        # Clamp
        th1 = int(np.clip(th1, 0, 255))
        th2 = int(np.clip(th2, 0, 255))
        if th2 < th1:
            th2 = th1 + 1 if th1 < 255 else 255

        sobel_ksize = int(np.clip(sobel_ksize, 3, 7))
        if sobel_ksize % 2 == 0:
            sobel_ksize += 1
        edge_thickness = int(np.clip(edge_thickness, 1, 50))
        glow_radius = int(np.clip(glow_radius, 1, 100))
        edge_gain = float(np.clip(edge_gain, 0.0, 50.0))
        amp_curve = float(np.clip(amp_curve, 0.1, 5.0))
        blend_original = float(np.clip(blend_original, 0.0, 1.0))

        # Audio
        amp = float(np.clip(float(getattr(features, "amp", 0.0)), 0.0, 1.0))
        shaped = amp ** amp_curve

        # Edge intensity: keep a baseline so it is never fully off (non-zero defaults request)
        # This prevents the effect from "disappearing" on quiet sections.
        intensity = (0.25 + 0.75 * shaped) * edge_gain

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        # Extract edges
        if method == "sobel":
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_ksize)
            mag = cv2.magnitude(gx, gy)
            # Normalize to 0..255
            mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            edges = mag.astype(np.uint8)
            # Threshold to focus on real contours (adaptive-ish via percentiles)
            lo = np.percentile(edges, 75)
            edges = cv2.threshold(edges, float(lo), 255, cv2.THRESH_BINARY)[1]
        else:
            # Default: canny
            edges = cv2.Canny(gray, threshold1=th1, threshold2=th2)

        # Thicken edges (dilate)
        k = 2 * edge_thickness + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edges_thick = cv2.dilate(edges, kernel, iterations=1)

        # Glow (blur)
        # Convert mask to float and blur to create bloom-like halo
        glow = cv2.GaussianBlur(edges_thick, (0, 0), sigmaX=float(glow_radius), sigmaY=float(glow_radius))
        glow_f = glow.astype(np.float32) / 255.0

        # Colorize neon
        r, g, b = edge_color
        neon = np.zeros_like(frame_rgb, dtype=np.float32)
        neon[..., 0] = glow_f * float(r)
        neon[..., 1] = glow_f * float(g)
        neon[..., 2] = glow_f * float(b)

        # Apply intensity
        neon *= float(intensity)

        neon_u8 = np.clip(neon, 0, 255).astype(np.uint8)

        # Composite
        if blend_mode == "screen":
            out = self._screen_blend(frame_rgb, neon_u8)
        else:
            # "add" style: saturating addition
            out = cv2.add(frame_rgb, neon_u8)

        # Optional: mix original back in
        if blend_original > 0.0:
            alpha_eff = float(1.0 - blend_original)  # 1 -> effect only
            out = cv2.addWeighted(out, alpha_eff, frame_rgb, 1.0 - alpha_eff, 0.0)

        return out
