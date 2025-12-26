# cover_visuals/shockwave_ripple_radial.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from olaf_app.cover_visualization_api import BaseCoverEffect, FrameFeatures
from olaf_app.visualization_api import PluginParameter


class CoverShockwaveRippleRadialEffect(BaseCoverEffect):
    """
    Shockwave / ripple radial (cover effect)

    Overview
    --------
    This effect spawns expanding circular "shockwave rings" based on audio amplitude.
    Each ring lens-warps the image radially around (center_x, center_y).

    Trigger logic (important)
    -------------------------
    Many pipelines produce an amplitude that stays relatively high most of the time.
    A simple "cross-above-threshold" trigger can then fire once at the beginning only.

    To make spawning reliable, we use:
    - trigger_threshold: crossing above this threshold can spawn a wave
    - rearm_threshold: the trigger is re-armed only after amp falls below this value
    - min_rise: minimum positive delta (amp - prev_amp) required to count as an "attack"
    - cooldown_s: minimum delay between spawns

    Audio mapping
    -------------
    - amp: wave spawning (threshold + rearm + attack detection)
    - amp: wave intensity/width scaling via amp_curve

    Parameters (what they do)
    -------------------------
    - center_x, center_y (0..1):
        Ring origin (normalized). 0.5/0.5 is image center.
    - wave_speed (px/s):
        How fast the ring expands outward.
    - wave_width (px):
        Base thickness of the ring. Higher = thicker, softer band.
    - strength (px):
        Base maximum radial displacement produced by a wave (before amp scaling).
    - amp_curve:
        Exponent applied to amp (amp ** amp_curve). >1 emphasizes peaks, <1 boosts quieter parts.
    - width_amp_scale:
        Extra thickness added from amp:
        effective_width = wave_width * (1 + shaped_amp * width_amp_scale)
    - trigger_threshold (0..1):
        A wave can spawn when amp rises above this value.
    - rearm_threshold (0..1):
        Trigger re-arms when amp falls below this lower value (prevents "one-shot" behavior).
    - min_rise (0..1):
        Minimum required amp increase between frames to qualify as a new attack/beat.
    - cooldown_s:
        Minimum time between wave spawns (prevents spamming).
    - max_waves:
        Maximum simultaneous waves; older ones are dropped first.
    - blend_original (0..1):
        Mix original image back in. 0 = fully warped, 1 = original only.
    - border_mode:
        How pixels outside bounds are handled during warping: reflect/replicate/wrap/constant.
    - border_value (0..255):
        Fill value if border_mode = constant.

    Notes
    -----
    - Input/Output: uint8 RGB (H, W, 3)
    - Uses cv2.remap with cached coordinate grids for performance.
    """

    effect_id: str = "cover_shockwave_ripple_radial"
    effect_name: str = "Shockwave / ripple radial"
    effect_description: str = (
        "Audio-reactive expanding radial shockwaves that lens-warp the cover. "
        "Robust beat spawning with trigger + rearm + attack detection. "
        "Key params: center_x/y, wave_speed, wave_width, strength, amp_curve, "
        "trigger_threshold, rearm_threshold, min_rise, cooldown_s, max_waves, "
        "blend_original, border_mode, border_value."
    )
    effect_author: str = "DrDLP"
    effect_version: str = "0.3.0"
    effect_max_inputs: int = 1

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)

        # Cached base grids for remap
        self._grid_shape: Tuple[int, int] | None = None
        self._x_grid: np.ndarray | None = None
        self._y_grid: np.ndarray | None = None

        # Peak detection / wave spawning state
        self._prev_amp: float = 0.0
        self._last_trigger_t: float = -1e9
        self._waves: List[Dict[str, float]] = []  # each: {"t0":..., "strength":..., "width":...}

        # Rearm latch: once triggered, we wait until amp < rearm_threshold
        self._armed: bool = True

    # ------------------------------------------------------------------ #
    # Parameter schema
    # ------------------------------------------------------------------ #

    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        # NOTE: Non-zero defaults for *all* parameters (as requested previously).
        return {
            "center_x": PluginParameter(
                name="center_x",
                label="Center X",
                type="float",
                default=0.52,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="Wave center X (normalized 0..1). 0.5 = center.",
            ),
            "center_y": PluginParameter(
                name="center_y",
                label="Center Y",
                type="float",
                default=0.48,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="Wave center Y (normalized 0..1). 0.5 = center.",
            ),
            "wave_speed": PluginParameter(
                name="wave_speed",
                label="Wave speed (px/s)",
                type="float",
                default=850.0,
                minimum=0.0,
                maximum=5000.0,
                step=10.0,
                description="Expansion speed of the ring (pixels per second).",
            ),
            "wave_width": PluginParameter(
                name="wave_width",
                label="Wave width (px)",
                type="float",
                default=24.0,
                minimum=1.0,
                maximum=200.0,
                step=1.0,
                description="Base thickness of the ring (higher = thicker/softer).",
            ),
            "strength": PluginParameter(
                name="strength",
                label="Strength (px)",
                type="float",
                default=32.0,
                minimum=0.0,
                maximum=200.0,
                step=1.0,
                description="Base maximum radial displacement (pixels).",
            ),
            "amp_curve": PluginParameter(
                name="amp_curve",
                label="Amp curve",
                type="float",
                default=1.35,
                minimum=0.1,
                maximum=3.0,
                step=0.05,
                description="Amp shaping exponent (amp ** curve). >1 emphasizes peaks.",
            ),
            "width_amp_scale": PluginParameter(
                name="width_amp_scale",
                label="Width amp scale",
                type="float",
                default=0.9,
                minimum=0.0,
                maximum=3.0,
                step=0.05,
                description="Extra width from amp: width *= 1 + shaped_amp * scale.",
            ),
            "trigger_threshold": PluginParameter(
                name="trigger_threshold",
                label="Trigger threshold",
                type="float",
                default=0.32,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="Wave can spawn when amp rises above this threshold.",
            ),
            "rearm_threshold": PluginParameter(
                name="rearm_threshold",
                label="Rearm threshold",
                type="float",
                default=0.22,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="Trigger re-arms only when amp falls below this lower threshold.",
            ),
            "min_rise": PluginParameter(
                name="min_rise",
                label="Min rise",
                type="float",
                default=0.08,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="Minimum (amp - prev_amp) required to count as a new attack/beat.",
            ),
            "cooldown_s": PluginParameter(
                name="cooldown_s",
                label="Cooldown (s)",
                type="float",
                default=0.10,
                minimum=0.0,
                maximum=2.0,
                step=0.01,
                description="Minimum time between two wave spawns.",
            ),
            "max_waves": PluginParameter(
                name="max_waves",
                label="Max waves",
                type="int",
                default=3,
                minimum=1,
                maximum=10,
                step=1,
                description="Maximum simultaneous rings (older ones are dropped).",
            ),
            "blend_original": PluginParameter(
                name="blend_original",
                label="Blend original",
                type="float",
                default=0.15,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="Mix original back in (0=full warp, 1=original only).",
            ),
            "border_mode": PluginParameter(
                name="border_mode",
                label="Border mode",
                type="enum",
                default="reflect",
                choices=["reflect", "replicate", "wrap", "constant"],
                description="Border handling for remap outside bounds.",
            ),
            "border_value": PluginParameter(
                name="border_value",
                label="Border value (0-255)",
                type="int",
                default=16,
                minimum=0,
                maximum=255,
                step=1,
                description="Constant border fill value (used only if border_mode=constant).",
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
    def _cv2_border_mode(mode: str) -> int:
        return {
            "reflect": cv2.BORDER_REFLECT_101,
            "replicate": cv2.BORDER_REPLICATE,
            "wrap": cv2.BORDER_WRAP,
            "constant": cv2.BORDER_CONSTANT,
        }.get(mode, cv2.BORDER_REFLECT_101)

    @staticmethod
    def _clamp01(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    def _spawn_wave_if_needed(self, t: float, amp: float) -> None:
        """
        Robust spawning:
        - Rearm latch: after a spawn, we require amp < rearm_threshold to arm again.
        - Attack detection: require amp to be rising by at least min_rise to avoid flat plateaus.
        - Cooldown: limits spawn rate.
        """
        trigger_threshold = float(self.config.get("trigger_threshold", 0.32))
        rearm_threshold = float(self.config.get("rearm_threshold", 0.22))
        min_rise = float(self.config.get("min_rise", 0.08))
        cooldown_s = float(self.config.get("cooldown_s", 0.10))
        max_waves = int(self.config.get("max_waves", 3))

        trigger_threshold = float(np.clip(trigger_threshold, 0.0, 1.0))
        rearm_threshold = float(np.clip(rearm_threshold, 0.0, 1.0))
        min_rise = float(np.clip(min_rise, 0.0, 1.0))
        cooldown_s = float(np.clip(cooldown_s, 0.0, 10.0))
        max_waves = int(np.clip(max_waves, 1, 20))

        # Ensure sensible ordering
        if rearm_threshold > trigger_threshold:
            rearm_threshold = trigger_threshold

        # Update arming state
        if amp < rearm_threshold:
            self._armed = True

        cooled = (t - self._last_trigger_t) >= cooldown_s
        rising_fast_enough = (amp - self._prev_amp) >= min_rise
        above_trigger = amp >= trigger_threshold

        if self._armed and cooled and above_trigger and rising_fast_enough:
            base_strength = float(self.config.get("strength", 32.0))
            base_width = float(self.config.get("wave_width", 24.0))
            amp_curve = float(self.config.get("amp_curve", 1.35))
            width_amp_scale = float(self.config.get("width_amp_scale", 0.9))

            base_strength = float(np.clip(base_strength, 0.0, 500.0))
            base_width = float(np.clip(base_width, 1.0, 500.0))
            amp_curve = float(np.clip(amp_curve, 0.1, 5.0))
            width_amp_scale = float(np.clip(width_amp_scale, 0.0, 10.0))

            shaped = amp ** amp_curve
            strength = base_strength * shaped
            width = base_width * (1.0 + shaped * width_amp_scale)

            self._waves.append({"t0": float(t), "strength": float(strength), "width": float(width)})
            self._last_trigger_t = float(t)
            self._armed = False  # disarm until amp goes below rearm_threshold

            if len(self._waves) > max_waves:
                self._waves = self._waves[-max_waves:]

        self._prev_amp = amp

    # ------------------------------------------------------------------ #
    # Main effect
    # ------------------------------------------------------------------ #

    def apply_to_frame(self, frame_rgb: np.ndarray, t: float, features: FrameFeatures) -> np.ndarray:
        if frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("CoverShockwaveRippleRadialEffect expects uint8 RGB frame (H, W, 3).")

        h, w, _ = frame_rgb.shape
        self._ensure_grids(h=h, w=w)
        assert self._x_grid is not None
        assert self._y_grid is not None

        # Read params
        cx_n = float(self.config.get("center_x", 0.52))
        cy_n = float(self.config.get("center_y", 0.48))
        wave_speed = float(self.config.get("wave_speed", 850.0))
        blend_original = float(self.config.get("blend_original", 0.15))
        border_mode = str(self.config.get("border_mode", "reflect")).lower()
        border_value = int(self.config.get("border_value", 16))

        cx_n = self._clamp01(cx_n)
        cy_n = self._clamp01(cy_n)
        wave_speed = float(np.clip(wave_speed, 0.0, 20000.0))
        blend_original = float(np.clip(blend_original, 0.0, 1.0))
        border_value = int(np.clip(border_value, 0, 255))

        # Audio amp (0..1)
        amp = float(np.clip(float(getattr(features, "amp", 0.0)), 0.0, 1.0))

        # Spawn waves
        self._spawn_wave_if_needed(t=t, amp=amp)

        # No waves -> no effect
        if not self._waves or wave_speed <= 0.0:
            return frame_rgb

        # Center in pixels
        cx = cx_n * (w - 1)
        cy = cy_n * (h - 1)

        # Polar components
        dx = self._x_grid - np.float32(cx)
        dy = self._y_grid - np.float32(cy)
        r = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        r_safe = np.maximum(r, 1e-4)

        # Max radius used to cull finished waves
        max_r = float(np.hypot(max(cx, (w - 1) - cx), max(cy, (h - 1) - cy)))

        total_disp = np.zeros((h, w), dtype=np.float32)
        alive: List[Dict[str, float]] = []

        for wave in self._waves:
            t0 = float(wave["t0"])
            age = float(t - t0)
            if age < 0.0:
                continue

            R = float(age * wave_speed)
            width = float(wave["width"])
            strength = float(wave["strength"])

            # Cull when ring is far beyond the image
            if (R - 3.0 * width) > (max_r + 2.0 * width):
                continue

            dr = (r - np.float32(R))
            sigma = max(1.0, width)
            ring = np.exp(-(dr * dr) / (2.0 * sigma * sigma)).astype(np.float32)

            # Lens-like push (single strong band)
            total_disp += (np.float32(strength) * ring).astype(np.float32)
            alive.append(wave)

        self._waves = alive
        if not self._waves:
            return frame_rgb

        # Unit vectors from center
        ux = (dx / r_safe).astype(np.float32)
        uy = (dy / r_safe).astype(np.float32)

        # Inverse mapping: output pixel samples from (x - u*disp, y - v*disp)
        map_x = (self._x_grid - ux * total_disp).astype(np.float32)
        map_y = (self._y_grid - uy * total_disp).astype(np.float32)

        warped = cv2.remap(
            frame_rgb,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=self._cv2_border_mode(border_mode),
            borderValue=(border_value, border_value, border_value),
        )

        if blend_original > 0.0:
            # blend_original=1 -> original only
            alpha_warp = float(1.0 - blend_original)
            warped = cv2.addWeighted(warped, alpha_warp, frame_rgb, 1.0 - alpha_warp, 0.0)

        return warped
