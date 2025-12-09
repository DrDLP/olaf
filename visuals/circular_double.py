"""
Circular outer audio histogram around a reactive black circle.

- Inner black disc pulsation depends on stem 2 (input_2).
- Outer amorphous ring depends on stem 1 (input_1).
- The ring is a filled polygon with a moving color gradient.
- Thin black radial strokes give a subtle "histogram" feel.

Designed for Olaf's visualization plugin system.
"""

from __future__ import annotations

import math
import colorsys
from typing import Any, Dict, Optional, List

from PyQt6.QtCore import Qt, QTimer, QPointF, QSize
from PyQt6.QtGui import QPainter, QColor, QPen, QPolygonF, QConicalGradient
from PyQt6.QtWidgets import QWidget

# Try both import paths so the plugin can work whether the app is installed
# as a package (olaf_app) or run from a flat layout during development.
try:
    from olaf_app.visualization_api import BaseVisualization, PluginParameter
except ImportError:
    from visualization_api import BaseVisualization, PluginParameter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _hsv_to_qcolor(h: float, s: float, v: float) -> QColor:
    """Convert HSV floats in [0, 1] to a QColor."""
    h = h % 1.0
    s = _clamp(s, 0.0, 1.0)
    v = _clamp(v, 0.0, 1.0)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return QColor(int(r * 255), int(g * 255), int(b * 255))


# ---------------------------------------------------------------------------
# Preview widget
# ---------------------------------------------------------------------------


class _CircularOuterWidget(QWidget):
    """
    Preview widget for the circular outer histogram visualization.

    Inner disc energy  <- stem 2 (input_2)
    Outer ring energy  <- stem 1 (input_1)

    This widget is used both:
      - as a small real-time preview in the Visualizations tab,
      - and as an off-screen, high-resolution renderer during export.

    For small previews, we keep the configured angular resolution.
    For large off-screen widgets (e.g. 1920x1080, 2560x1440),
    we automatically increase num_bins so the outline stays smooth.
    """

    def __init__(self, config: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._time_s: float = 0.0
        self._inner_energy: float = 0.0  # stem 2
        self._outer_energy: float = 0.0  # stem 1

        # Reasonable minimum size for docked preview
        self.setMinimumSize(320, 180)

        # Lightweight timer: only drives repaint, not time itself.
        # Time and energies are provided by the host via update_audio_state().
        self._timer = QTimer(self)
        self._timer.setInterval(50)  # default ~20 FPS
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    def sizeHint(self) -> QSize:  # type: ignore[override]
        """
        Hint used by layouts when the visualization is shown in a tab.

        The actual size for export is controlled by the host via
        apply_preview_settings(width, height, fps).
        """
        return QSize(640, 360)

    def set_timer_interval(self, interval_ms: int) -> None:
        """Update the internal timer interval used to repaint the preview."""
        interval_ms = max(10, int(interval_ms))
        self._timer.setInterval(interval_ms)

    def update_audio_state(self, time_ms: int, inner_energy: float, outer_energy: float) -> None:
        """Called by the plugin with host-provided audio features."""
        self._time_s = max(0.0, float(time_ms) / 1000.0)
        self._inner_energy = _clamp(float(inner_energy), 0.0, 1.0)
        self._outer_energy = _clamp(float(outer_energy), 0.0, 1.0)

    def _on_tick(self) -> None:
        # Only triggers a repaint; state is updated via update_audio_state().
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        width = self.width()
        height = self.height()
        if width <= 0 or height <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Background
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        cfg = self._config

        # ------------------------------------------------------------------
        # Angular resolution (num_bins) with quality scaling
        #
        # - For small widgets (UI preview), use the configured value.
        # - For large widgets (off-screen export, e.g. >= 1280x720),
        #   automatically increase num_bins in proportion to the size
        #   to keep the outline smooth at 1080p / 1440p / 4K.
        # ------------------------------------------------------------------
        num_bins = int(cfg.get("num_bins", 160))
        num_bins = max(16, min(512, num_bins))

        if width >= 1280 and height >= 720:
            # Scale angular resolution with respect to a 640x360 "base" preview.
            scale_w = width / 640.0
            scale_h = height / 360.0
            upscale_factor = max(scale_w, scale_h)
            upscale_factor = max(1.0, min(upscale_factor, 3.0))  # cap to x3

            num_bins = int(num_bins * upscale_factor)
            num_bins = max(32, min(1024, num_bins))

        inner_radius_factor = float(cfg.get("inner_radius_factor", 0.55))
        inner_radius_factor = _clamp(inner_radius_factor, 0.2, 0.9)

        # Allow much stronger pulsation than before.
        # We clamp the *effective* pulse so the disc never fully collapses.
        inner_pulse_amount = float(cfg.get("inner_pulse_amount", 0.35))
        inner_pulse_amount = _clamp(inner_pulse_amount, 0.0, 2.0)


        bar_length_factor = float(cfg.get("bar_length_factor", 0.45))
        bar_length_factor = _clamp(bar_length_factor, 0.1, 2.0)  # max 2.0

        bar_energy_gain = float(cfg.get("bar_energy_gain", 1.0))
        bar_energy_gain = _clamp(bar_energy_gain, 0.0, 3.0)

        rotation_speed_deg = float(cfg.get("rotation_speed_deg", 60.0))
        rotation_speed_deg = _clamp(rotation_speed_deg, 0.0, 720.0)

        rotation_energy_gain = float(cfg.get("rotation_energy_gain", 1.5))
        rotation_energy_gain = _clamp(rotation_energy_gain, 0.0, 4.0)

        hue_cycle_speed_deg = float(cfg.get("hue_cycle_speed_deg", 40.0))
        hue_cycle_speed_deg = _clamp(hue_cycle_speed_deg, 0.0, 360.0)  # 0..360

        saturation = float(cfg.get("saturation", 0.9))
        saturation = _clamp(saturation, 0.0, 1.0)

        brightness_base = float(cfg.get("brightness_base", 0.60))
        brightness_base = _clamp(brightness_base, 0.1, 1.0)

        brightness_energy_gain = float(cfg.get("brightness_energy_gain", 0.8))
        brightness_energy_gain = _clamp(brightness_energy_gain, 0.0, 2.0)

        chaos_speed = float(cfg.get("chaos_speed", 2.5))

        inner_energy = self._inner_energy
        outer_energy = self._outer_energy
        t = self._time_s

        # Geometry
        cx = width * 0.5
        cy = height * 0.5
        min_dim = float(min(width, height))

        # Inner disc radius (breathes with stem 2)
        base_inner_radius = inner_radius_factor * (min_dim * 0.5)

        # Effective pulse: higher inner_pulse_amount + strong energy
        # => radius can shrink a lot, but never more than 90%.
        pulse = inner_pulse_amount * inner_energy
        pulse = _clamp(pulse, 0.0, 0.9)

        inner_radius = base_inner_radius * (1.0 - pulse)
        inner_radius = max(5.0, inner_radius)

        # Base outward span from the inner disc
        base_bar_len = bar_length_factor * (min_dim * 0.5)

        # Global rotation, mainly driven by outer_energy (stem 1)
        rot_deg = rotation_speed_deg * t * (1.0 + rotation_energy_gain * outer_energy)
        rot_rad = math.radians(rot_deg)

        # Hue shift for color cycling
        hue_shift = (hue_cycle_speed_deg * t) / 360.0

        # --------- Build angular noise + radii ---------
        angles: List[float] = []
        radii_outer: List[float] = []
        bin_energies: List[float] = []

        for i in range(num_bins):
            theta = (i / float(num_bins)) * (2.0 * math.pi)  # base angle

            # Multi-frequency, angle-based noise to avoid "fleur" pattern
            x = theta
            base_shape = 0.5 * (math.sin(x * 11.0) + 1.0)
            dyn1 = 0.5 * (math.sin(x * 5.3 + t * chaos_speed) + 1.0)
            dyn2 = 0.5 * (math.sin(x * 17.1 + t * chaos_speed * 1.7 + 1.3) + 1.0)
            chaos = (base_shape + dyn1 + dyn2) / 3.0  # [0, 1]

            # Local energy from stem 1, but with some randomness even if outer_energy is flat
            bin_energy = outer_energy * chaos

            # Outward length from the inner disc (breathes outward)
            min_len = base_bar_len * 0.05
            effective_energy = 0.3 * outer_energy + 0.7 * bin_energy
            bar_len = base_bar_len * (0.15 + bar_energy_gain * effective_energy)
            bar_len = _clamp(bar_len, min_len, base_bar_len * (1.0 + bar_energy_gain))

            r_outer = inner_radius + bar_len

            angles.append(theta)
            radii_outer.append(r_outer)
            bin_energies.append(bin_energy)

        # --------- Discrete radial bars forming the outer structure ---------
        if angles:
            # Bar thickness depends a bit on resolution so it stays visible in 4K.
            base_bar_thickness = 0.004 * min_dim  # ~4 px at 1080p
            base_bar_thickness = _clamp(base_bar_thickness, 1.5, 8.0)

            for i, (theta, r_outer, bin_energy) in enumerate(
                zip(angles, radii_outer, bin_energies)
            ):
                angle = theta + rot_rad
                ca = math.cos(angle)
                sa = math.sin(angle)

                # Start radius: just outside the inner disc, plus a small gap
                gap = 0.02 * min_dim
                r_inner_bar = inner_radius + gap

                # End radius: where the polygon would have been
                r_outer_bar = r_outer

                x1 = cx + r_inner_bar * ca
                y1 = cy + r_inner_bar * sa
                x2 = cx + r_outer_bar * ca
                y2 = cy + r_outer_bar * sa

                # Color per bar, based on angle + time + local energy
                pos = i / float(num_bins)
                h = pos + hue_shift
                v = brightness_base + brightness_energy_gain * (
                    0.4 * outer_energy + 0.6 * bin_energy
                )
                color = _hsv_to_qcolor(h, saturation, v)

                pen = QPen(color)
                pen.setWidthF(base_bar_thickness)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))


        # --------- Inner black disc on top ---------
        painter.setBrush(Qt.GlobalColor.black)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(
            int(cx - inner_radius),
            int(cy - inner_radius),
            int(inner_radius * 2.0),
            int(inner_radius * 2.0),
        )

        painter.end()

# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------


class CircularOuterNoBorderVisualization(BaseVisualization):
    """
    Circular outer histogram visualization plugin.

    - Outer amorphous ring reacts to stem 1 (input_1).
    - Inner black disc pulsation reacts to stem 2 (input_2).
    """

    plugin_id = "dr_dlp_circular_double"
    plugin_name = "Double circular"
    plugin_description = (
        "Amorphous circular ring around a pulsing inner disc. "
        "Outer shape follows stem 1, inner disc follows stem 2."
    )
    plugin_author = "Dr DLP"
    plugin_version = "1.3.0"
    plugin_max_inputs = 2

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._widget: Optional[_CircularOuterWidget] = None

        defaults = {
            "num_bins": 256,
            "inner_radius_factor": 0.55,
            "inner_pulse_amount": 0.35,
            "bar_length_factor": 0.45,       # up to 2.0
            "bar_energy_gain": 1.0,
            "rotation_speed_deg": 60.0,
            "rotation_energy_gain": 1.5,
            "hue_cycle_speed_deg": 40.0,     # 0..360
            "saturation": 0.9,
            "brightness_base": 0.60,
            "brightness_energy_gain": 0.8,
            "chaos_speed": 2.5,
        }
        for key, value in defaults.items():
            self.config.setdefault(key, value)

    # ----- plugin parameters metadata -----

    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        return {
            "num_bins": PluginParameter(
                name="num_bins",
                label="Angular resolution",
                type="int",
                default=256,
                minimum=32,
                maximum=1024,
                step=16,
                description="Number of samples around the ring (higher = smoother outline).",
            ),
            "inner_radius_factor": PluginParameter(
                name="inner_radius_factor",
                label="Inner circle size",
                type="float",
                default=0.55,
                minimum=0.2,
                maximum=0.9,
                step=0.02,
                description="Relative size of the inner black disc.",
            ),
            "inner_pulse_amount": PluginParameter(
                name="inner_pulse_amount",
                label="Inner pulse amount",
                type="float",
                default=0.35,
                minimum=0.0,
                maximum=100.0,   
                step=1,
                description="How much the inner disc shrinks on strong beats (stem 2).",
            ),

            "bar_length_factor": PluginParameter(
                name="bar_length_factor",
                label="Ring thickness",
                type="float",
                default=0.45,
                minimum=0.1,
                maximum=2.0,
                step=0.05,
                description="Base thickness of the ring relative to the preview size.",
            ),
            "bar_energy_gain": PluginParameter(
                name="bar_energy_gain",
                label="Energy gain",
                type="float",
                default=1.0,
                minimum=0.0,
                maximum=3.0,
                step=0.1,
                description="How strongly the ring thickness reacts to stem 1 energy.",
            ),
            "rotation_speed_deg": PluginParameter(
                name="rotation_speed_deg",
                label="Rotation speed (deg/s)",
                type="float",
                default=60.0,
                minimum=0.0,
                maximum=720.0,
                step=5.0,
                description="Base angular speed of the whole pattern.",
            ),
            "rotation_energy_gain": PluginParameter(
                name="rotation_energy_gain",
                label="Rotation energy gain",
                type="float",
                default=1.5,
                minimum=0.0,
                maximum=4.0,
                step=0.1,
                description="Extra rotation speed added on loud sections (stem 1).",
            ),
            "hue_cycle_speed_deg": PluginParameter(
                name="hue_cycle_speed_deg",
                label="Color cycle speed (deg/s)",
                type="float",
                default=40.0,
                minimum=0.0,
                maximum=360.0,
                step=5.0,
                description="How fast colors rotate around the ring (0 = static colors).",
            ),
            "saturation": PluginParameter(
                name="saturation",
                label="Color saturation",
                type="float",
                default=0.9,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                description="Saturation of the ring colors.",
            ),
            "brightness_base": PluginParameter(
                name="brightness_base",
                label="Base brightness",
                type="float",
                default=0.60,
                minimum=0.1,
                maximum=1.0,
                step=0.05,
                description="Base brightness of the ring at low energy.",
            ),
            "brightness_energy_gain": PluginParameter(
                name="brightness_energy_gain",
                label="Brightness energy gain",
                type="float",
                default=0.8,
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                description="How much brightness increases with stem 1 energy.",
            ),
            "chaos_speed": PluginParameter(
                name="chaos_speed",
                label="Chaos speed",
                type="float",
                default=2.5,
                minimum=0.5,
                maximum=10.0,
                step=0.1,
                description="How fast the ring surface wiggles over time.",
            ),
        }

    # ----- BaseVisualization hooks -----

    def create_widget(self, parent: Optional[QWidget] = None) -> QWidget:
        self._widget = _CircularOuterWidget(self.config, parent)
        return self._widget

    def apply_preview_settings(self, width: int, height: int, fps: int) -> None:
        """Called by the host when preview size/FPS are changed."""
        if self._widget is None:
            return

        fps = max(5, min(60, int(fps)))
        interval_ms = int(1000 / fps)

        self._widget.setMinimumSize(width, height)
        self._widget.setMaximumSize(width, height)
        self._widget.set_timer_interval(interval_ms)

    def on_audio_features(self, features: Dict[str, Any]) -> None:
        """
        Called periodically with host-provided features.

        Expected structure:
          features = {
              "time_ms": <int>,
              "inputs": {
                  "input_1": {"rms": float, ...},
                  "input_2": {"rms": float, ...},
              }
          }
        """
        if self._widget is None:
            return

        time_ms = int(features.get("time_ms", 0))
        inputs = features.get("inputs", {}) or {}

        input_1 = inputs.get("input_1", {})
        input_2 = inputs.get("input_2", {})

        level_1 = float(input_1.get("rms", 0.0))  # outer ring
        level_2 = float(input_2.get("rms", 0.0))  # inner disc

        outer_energy = _clamp(level_1, 0.0, 1.0)
        inner_energy = _clamp(level_2, 0.0, 1.0)

        self._widget.update_audio_state(time_ms, inner_energy, outer_energy)

    def on_activate(self) -> None:
        """Called when the plugin becomes the active visualization."""
        pass

    def on_deactivate(self) -> None:
        """Called when the plugin is no longer active."""
        pass
