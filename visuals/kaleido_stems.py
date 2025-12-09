"""
Kaleido Stems visualization plugin for Olaf.

Audio-reactive radial kaleidoscope:

- A base pattern is drawn in a single wedge sector.
- That wedge is then replicated by rotation + mirroring around the center.
- The whole structure slowly rotates, and its shapes/palette react to
  multiple audio stems.

Recommended stem routing:
- input_1 (drums)  -> flashes, line thickness, contrast
- input_2 (bass)   -> radial amplitude (depth of the patterns)
- input_3 (vocals) -> palette shift (warm/cold hue, saturation)
- input_4 (other)  -> global rotation, angular jitter/glitch
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt, QTimer, QSize, QRectF
from PyQt6.QtGui import (
    QPainter,
    QColor,
    QRadialGradient,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import QWidget

try:
    from olaf_app.visualization_api import BaseVisualization, PluginParameter
except ImportError:
    from visualization_api import BaseVisualization, PluginParameter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smoothstep(x: float) -> float:
    x = _clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _hsv_to_qcolor(h: float, s: float, v: float) -> QColor:
    """
    Convert HSV (h in [0,1], s in [0,1], v in [0,1]) to a QColor.
    """
    h = h % 1.0
    s = _clamp(s, 0.0, 1.0)
    v = _clamp(v, 0.0, 1.0)
    c = QColor()
    c.setHsvF(h, s, v)
    return c


# ---------------------------------------------------------------------------
# Preview / export widget
# ---------------------------------------------------------------------------


class _KaleidoStemsWidget(QWidget):
    """
    Preview / export widget for the Kaleido Stems visualization.

    Used both as:
      - a small real-time preview in the Visualizations tab,
      - a large off-screen widget during video export (high resolution).
    """

    def __init__(self, config: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config

        # Time (seconds) as provided by the host
        self._time_s: float = 0.0

        # Audio levels in [0, 1] for each input
        self._level_1: float = 0.0  # drums
        self._level_2: float = 0.0  # bass
        self._level_3: float = 0.0  # vocals
        self._level_4: float = 0.0  # other

        self._timer = QTimer(self)
        self._timer.setInterval(40)  # ~25 FPS
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

        self.setMinimumSize(320, 180)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        """
        Hint used by layouts when the visualization is shown in a tab.

        The actual size for export is controlled by the host via
        apply_preview_settings(width, height, fps).
        """
        return QSize(640, 360)

    def set_timer_interval(self, interval_ms: int) -> None:
        """
        Adjust internal timer interval for preview usage.
        """
        interval_ms = max(15, int(interval_ms))
        self._timer.setInterval(interval_ms)

    def update_audio_state(
        self,
        time_ms: int,
        level_1: float,
        level_2: float,
        level_3: float,
        level_4: float,
    ) -> None:
        """
        Receive updated audio features from the plugin.

        All levels are expected in [0, 1].
        """
        self._time_s = max(0.0, float(time_ms) / 1000.0)
        self._level_1 = _clamp(level_1, 0.0, 1.0)
        self._level_2 = _clamp(level_2, 0.0, 1.0)
        self._level_3 = _clamp(level_3, 0.0, 1.0)
        self._level_4 = _clamp(level_4, 0.0, 1.0)
        self.update()

    def _on_tick(self) -> None:
        """Timer callback used in interactive preview."""
        self.update()

    # ------------------------------------------------------------------
    # Main painting logic
    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # type: ignore[override]
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        cfg = self._config
        t = self._time_s

        cx = w * 0.5
        cy = h * 0.5
        min_dim = float(min(w, h))

        # ------------------------------------------------------------------
        # Audio levels
        # ------------------------------------------------------------------
        level_drums = self._level_1
        level_bass = self._level_2
        level_vocals = self._level_3
        level_other = self._level_4

        # Global energy (used for background + rotation)
        global_energy = _clamp(
            0.35 * level_drums + 0.35 * level_bass + 0.2 * level_vocals + 0.1 * level_other,
            0.0,
            1.0,
        )

        # ------------------------------------------------------------------
        # Background gradient
        # ------------------------------------------------------------------
        bg_radius = min_dim * 0.75
        bg_grad = QRadialGradient(cx, cy, bg_radius, cx, cy)
        bg_center = QColor(5, 5, 12)
        bg_edge = QColor(15, 10, 30)

        e_factor = 0.4 + 0.6 * global_energy
        bg_edge = QColor(
            int(bg_edge.red() * e_factor),
            int(bg_edge.green() * (0.8 + 0.2 * e_factor)),
            int(bg_edge.blue() * (0.8 + 0.2 * e_factor)),
        )

        bg_grad.setColorAt(0.0, bg_center)
        bg_grad.setColorAt(1.0, bg_edge)
        painter.fillRect(self.rect(), bg_grad)

        # ------------------------------------------------------------------
        # Kaleidoscope geometry parameters
        # ------------------------------------------------------------------
        num_slices = int(cfg.get("num_slices", 12))
        num_slices = max(4, min(64, num_slices))

        wedge_angle = 2.0 * math.pi / float(num_slices)

        # Base radius and inner radius (for a central empty disk)
        outer_radius_factor = float(cfg.get("outer_radius_factor", 0.9))
        outer_radius_factor = _clamp(outer_radius_factor, 0.3, 1.2)
        inner_radius_factor = float(cfg.get("inner_radius_factor", 0.08))
        inner_radius_factor = _clamp(inner_radius_factor, 0.0, 0.4)

        base_radius = 0.5 * min_dim * outer_radius_factor
        inner_radius = base_radius * inner_radius_factor

        # Number of radial rings (pattern repetitions along the radius)
        num_rings = int(cfg.get("num_rings", 7))
        num_rings = max(1, min(32, num_rings))

        # Subdivisions within the wedge (angular resolution)
        wedge_resolution = int(cfg.get("wedge_resolution", 64))
        wedge_resolution = max(16, min(256, wedge_resolution))

        # Bass drives shape “depth”
        bass_shape_gain = float(cfg.get("bass_shape_gain", 1.3))
        bass_shape_gain = _clamp(bass_shape_gain, 0.0, 4.0)

        # Drums drive sharpness / flashes
        drums_flash_gain = float(cfg.get("drums_flash_gain", 1.6))
        drums_flash_gain = _clamp(drums_flash_gain, 0.0, 4.0)

        # Vocals drive palette
        vocal_palette_shift = float(cfg.get("vocal_palette_shift", 0.35))
        vocal_palette_shift = _clamp(vocal_palette_shift, 0.0, 1.0)

        # Other drives rotation / jitter
        other_rotation_gain = float(cfg.get("other_rotation_gain", 1.4))
        other_rotation_gain = _clamp(other_rotation_gain, 0.0, 5.0)

        # ------------------------------------------------------------------
        # Rotation and jitter
        # ------------------------------------------------------------------
        base_rotation_speed_deg = float(cfg.get("base_rotation_speed_deg", 25.0))
        base_rotation_speed_deg = _clamp(base_rotation_speed_deg, -360.0, 360.0)
        rotation_audio_gain = float(cfg.get("rotation_audio_gain", 80.0))
        rotation_audio_gain = _clamp(rotation_audio_gain, 0.0, 360.0)

        rotation_speed_deg = base_rotation_speed_deg + rotation_audio_gain * global_energy
        rotation_deg = rotation_speed_deg * t
        base_rotation_rad = math.radians(rotation_deg)

        # Glitch: small angular wobble from input_4
        glitch_amount_deg = float(cfg.get("glitch_amount_deg", 5.0))
        glitch_amount_deg = _clamp(glitch_amount_deg, 0.0, 40.0)
        glitch_phase = t * (1.0 + 2.0 * level_other)
        glitch_offset_rad = math.radians(glitch_amount_deg * math.sin(glitch_phase)) * level_other

        # ------------------------------------------------------------------
        # Colors and line thickness
        # ------------------------------------------------------------------
        base_hue = float(cfg.get("base_hue", 0.65))  # default bluish / purple
        hue_range = float(cfg.get("hue_range", 0.4))
        hue_range = _clamp(hue_range, 0.0, 1.0)

        # Vocals shift hue towards warm/cold
        base_hue += vocal_palette_shift * (level_vocals - 0.5)

        saturation = float(cfg.get("saturation", 0.9))
        saturation = _clamp(saturation, 0.0, 1.0)

        brightness_base = float(cfg.get("brightness_base", 0.6))
        brightness_base = _clamp(brightness_base, 0.1, 1.0)

        brightness_gain = float(cfg.get("brightness_gain", 0.9))
        brightness_gain = _clamp(brightness_gain, 0.0, 2.0)

        # Drums flash -> brightness modulation
        drums_flash_level = _smoothstep(level_drums * drums_flash_gain)

        line_width_factor = float(cfg.get("line_width_factor", 0.008))
        line_width_factor = _clamp(line_width_factor, 0.001, 0.05)

        # ------------------------------------------------------------------
        # Build base wedge path (in polar coordinates)
        # ------------------------------------------------------------------
        # We work in centered coordinates, then replicate by rotations.
        painter.translate(cx, cy)

        # Precompute some shape parameters
        radial_noise_speed = float(cfg.get("radial_noise_speed", 0.7))
        radial_noise_speed = _clamp(radial_noise_speed, 0.0, 5.0)

        angular_warp_amount = float(cfg.get("angular_warp_amount", 0.25))
        angular_warp_amount = _clamp(angular_warp_amount, 0.0, 1.0)

        # For each ring, we build one path in the base wedge (centered around angle=0)
        # and we will draw it in each slice via rotation + mirror.
        base_paths = []

        for ring_idx in range(num_rings):
            ring_t = (ring_idx + 0.5) / float(num_rings)
            ring_radius = inner_radius + ring_t * (base_radius - inner_radius)

            # Bass controls radial deformation amplitude for each ring
            ring_amp = ring_radius * 0.15 * (1.0 + bass_shape_gain * level_bass * (0.3 + 0.7 * ring_t))

            # Extra frequency with ring index (inner rings more detailed)
            base_freq = 1.5 + 4.0 * ring_t

            path = QPainterPath()
            first = True

            for i in range(wedge_resolution + 1):
                u = i / float(wedge_resolution)
                # Angle within wedge (centered around 0)
                theta = (u - 0.5) * wedge_angle

                # Apply angular warp based on audio, gives a floral / glitchy aspect
                warp = angular_warp_amount * math.sin(
                    theta * (4.0 + 6.0 * ring_t) + t * radial_noise_speed * (1.0 + 0.5 * level_other)
                ) * (0.4 + 0.6 * global_energy)
                theta_warped = theta + warp

                # Radial modulation (petal-like)
                wave = math.sin(
                    theta * base_freq + t * (0.8 + 0.6 * level_bass) + ring_idx * 1.3
                )
                wave2 = math.sin(
                    theta * (base_freq * 2.0 + 1.0) - t * 0.9 + ring_idx * 2.1
                )
                combined = 0.6 * wave + 0.4 * wave2

                radial_offset = ring_amp * combined
                r = ring_radius + radial_offset

                if r < inner_radius * 0.7:
                    r = inner_radius * 0.7

                x = r * math.cos(theta_warped)
                y = r * math.sin(theta_warped)

                if first:
                    path.moveTo(x, y)
                    first = False
                else:
                    path.lineTo(x, y)

            base_paths.append((ring_t, path))

        # ------------------------------------------------------------------
        # Draw the kaleidoscope: replicate wedge paths into slices
        # ------------------------------------------------------------------
        # Global line width based on resolution and settings
        line_width_px = line_width_factor * min_dim * (0.7 + 0.6 * drums_flash_level)
        line_width_px = _clamp(line_width_px, 0.3, min_dim * 0.025)

        for slice_idx in range(num_slices):
            angle_slice = slice_idx * wedge_angle + base_rotation_rad
            # Extra rotation from input_4
            angle_slice += other_rotation_gain * level_other * 0.01 * math.sin(
                t * 1.5 + slice_idx * 0.8
            )

            # Draw normal + mirrored for each slice
            for mirror_sign in (1, -1):
                painter.save()
                # Rotate to slice, then mirror (for kaleidoscope symmetry)
                painter.rotate(math.degrees(angle_slice))
                painter.scale(mirror_sign, 1.0)

                for (ring_t, path) in base_paths:
                    # Hue shifts slightly with ring index
                    h = (base_hue + hue_range * (ring_t - 0.5)) % 1.0
                    # Vocals push towards warmer palette
                    h += 0.2 * (level_vocals - 0.5)
                    s = saturation
                    v = brightness_base + brightness_gain * (global_energy * (0.5 + 0.5 * ring_t))

                    # Drums flash brighten rings
                    v *= (1.0 + drums_flash_level * 0.5 * (0.3 + 0.7 * ring_t))
                    v = _clamp(v, 0.0, 1.0)

                    color = _hsv_to_qcolor(h, s, v)
                    alpha = int(130 + 100 * global_energy * (0.2 + 0.8 * ring_t))
                    alpha = _clamp(alpha, 40, 255)
                    color.setAlpha(alpha)

                    pen = QPen(color)
                    pen.setWidthF(line_width_px * (0.8 + 0.4 * ring_t))
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    painter.setPen(pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawPath(path)

                painter.restore()

        # ------------------------------------------------------------------
        # Central disk (soft glow)
        # ------------------------------------------------------------------
        core_radius_factor = float(cfg.get("core_radius_factor", 0.12))
        core_radius_factor = _clamp(core_radius_factor, 0.02, 0.4)
        core_radius = base_radius * core_radius_factor

        if core_radius > 0.0:
            core_grad = QRadialGradient(0.0, 0.0, core_radius, 0.0, 0.0)
            core_h = (base_hue + 0.1 * level_vocals) % 1.0
            core_v_inner = brightness_base + brightness_gain * (0.7 * global_energy + 0.3 * level_vocals)
            core_v_outer = core_v_inner * 0.5

            col_inner = _hsv_to_qcolor(core_h, saturation, core_v_inner)
            col_outer = _hsv_to_qcolor(core_h, saturation * 0.5, core_v_outer)
            col_inner.setAlpha(255)
            col_outer.setAlpha(60 + int(120 * global_energy))

            core_grad.setColorAt(0.0, col_inner)
            core_grad.setColorAt(1.0, col_outer)

            painter.setBrush(core_grad)
            painter.setPen(Qt.PenStyle.NoPen)

            core_rect = QRectF(
                -core_radius,
                -core_radius,
                core_radius * 2.0,
                core_radius * 2.0,
            )
            painter.drawEllipse(core_rect)

        painter.end()


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------


class KaleidoStemsVisualization(BaseVisualization):
    """
    Audio-reactive kaleidoscope visualization.

    - A base radial pattern is drawn in one wedge sector.
    - This wedge is replicated by rotation + mirroring around the center.
    - Shapes, rotation and colors are driven by multiple stems.

    Recommended stem routing:
      - input_1: drums  -> flashes, line thickness, contrast
      - input_2: bass   -> radial amplitude (depth of the patterns)
      - input_3: vocals -> palette shift (warm/cold, saturation)
      - input_4: other  -> global rotation speed and glitchy angular wobble
    """

    plugin_id = "dr_dlp_kaleido_stems"
    plugin_name = "Kaleido Stems"
    plugin_author = "Dr DLP"
    plugin_version = "1.0.0"
    plugin_max_inputs = 4

    plugin_description = (
        "Radial audio-reactive kaleidoscope built from a single wedge sector.\n"
        "\n"
        "Recommended stem routing:\n"
        "  - input_1: drums  -> flashes, line thickness, contrast\n"
        "  - input_2: bass   -> radial amplitude (depth of the patterns)\n"
        "  - input_3: vocals -> palette shift (warm/cold, saturation)\n"
        "  - input_4: other  -> global rotation speed and glitchy wobble\n"
        "\n"
        "You can reroute stems freely, but this mapping usually gives a\n"
        "clear and musical kaleidoscope."
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._widget: Optional[_KaleidoStemsWidget] = None

        defaults = {
            "num_slices": 12,
            "outer_radius_factor": 0.9,
            "inner_radius_factor": 0.08,
            "num_rings": 7,
            "wedge_resolution": 64,
            "bass_shape_gain": 1.3,
            "drums_flash_gain": 1.6,
            "vocal_palette_shift": 0.35,
            "other_rotation_gain": 1.4,
            "base_rotation_speed_deg": 25.0,
            "rotation_audio_gain": 80.0,
            "glitch_amount_deg": 5.0,
            "base_hue": 0.65,
            "hue_range": 0.4,
            "saturation": 0.9,
            "brightness_base": 0.6,
            "brightness_gain": 0.9,
            "line_width_factor": 0.008,
            "radial_noise_speed": 0.7,
            "angular_warp_amount": 0.25,
            "core_radius_factor": 0.12,
        }
        for k, v in defaults.items():
            self.config.setdefault(k, v)

    # ------------------------------------------------------------------
    # Parameters exposed to the UI
    # ------------------------------------------------------------------
    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        """
        Expose configurable parameters to the host UI.
        """
        return {
            "num_slices": PluginParameter(
                name="num_slices",
                label="Number of slices",
                type="int",
                default=12,
                minimum=4,
                maximum=64,
                step=1,
                description="Number of kaleidoscope sectors (symmetry order).",
            ),
            "outer_radius_factor": PluginParameter(
                name="outer_radius_factor",
                label="Outer radius factor",
                type="float",
                default=0.9,
                minimum=0.3,
                maximum=1.2,
                step=0.02,
                description="Radius of the kaleidoscope relative to half the frame size.",
            ),
            "inner_radius_factor": PluginParameter(
                name="inner_radius_factor",
                label="Inner radius factor",
                type="float",
                default=0.08,
                minimum=0.0,
                maximum=0.4,
                step=0.01,
                description="Radius of the central empty disk as fraction of outer radius.",
            ),
            "num_rings": PluginParameter(
                name="num_rings",
                label="Number of radial rings",
                type="int",
                default=7,
                minimum=1,
                maximum=32,
                step=1,
                description="How many radial bands of patterns are drawn in the wedge.",
            ),
            "wedge_resolution": PluginParameter(
                name="wedge_resolution",
                label="Wedge resolution",
                type="int",
                default=64,
                minimum=16,
                maximum=256,
                step=4,
                description="Angular resolution used to sample the base wedge pattern.",
            ),
            "bass_shape_gain": PluginParameter(
                name="bass_shape_gain",
                label="Bass shape gain",
                type="float",
                default=1.3,
                minimum=0.0,
                maximum=4.0,
                step=0.1,
                description="How strongly bass inflates radial deformations.",
            ),
            "drums_flash_gain": PluginParameter(
                name="drums_flash_gain",
                label="Drums flash gain",
                type="float",
                default=1.6,
                minimum=0.0,
                maximum=4.0,
                step=0.1,
                description="How strongly drums affect flashes and contrast.",
            ),
            "vocal_palette_shift": PluginParameter(
                name="vocal_palette_shift",
                label="Vocal palette shift",
                type="float",
                default=0.35,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                description="How much vocals shift the hue (warm/cold bias).",
            ),
            "other_rotation_gain": PluginParameter(
                name="other_rotation_gain",
                label="Other rotation gain",
                type="float",
                default=1.4,
                minimum=0.0,
                maximum=5.0,
                step=0.1,
                description="How strongly input_4 adds extra rotation/jitter per slice.",
            ),
            "base_rotation_speed_deg": PluginParameter(
                name="base_rotation_speed_deg",
                label="Base rotation speed (deg/s)",
                type="float",
                default=25.0,
                minimum=-360.0,
                maximum=360.0,
                step=5.0,
                description="Base rotation speed of the kaleidoscope.",
            ),
            "rotation_audio_gain": PluginParameter(
                name="rotation_audio_gain",
                label="Rotation audio gain (deg/s)",
                type="float",
                default=80.0,
                minimum=0.0,
                maximum=360.0,
                step=5.0,
                description="Additional rotation speed per unit of global energy.",
            ),
            "glitch_amount_deg": PluginParameter(
                name="glitch_amount_deg",
                label="Glitch amount (deg)",
                type="float",
                default=5.0,
                minimum=0.0,
                maximum=40.0,
                step=0.5,
                description="Maximum angular jitter applied from input_4.",
            ),
            "base_hue": PluginParameter(
                name="base_hue",
                label="Base hue",
                type="float",
                default=0.65,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="Base hue of the palette (0..1).",
            ),
            "hue_range": PluginParameter(
                name="hue_range",
                label="Hue range",
                type="float",
                default=0.4,
                minimum=0.0,
                maximum=1.0,
                step=0.02,
                description="Hue variation between inner and outer rings.",
            ),
            "saturation": PluginParameter(
                name="saturation",
                label="Saturation",
                type="float",
                default=0.9,
                minimum=0.0,
                maximum=1.0,
                step=0.02,
                description="Base saturation of the pattern colors.",
            ),
            "brightness_base": PluginParameter(
                name="brightness_base",
                label="Base brightness",
                type="float",
                default=0.6,
                minimum=0.1,
                maximum=1.0,
                step=0.02,
                description="Base brightness of the rings.",
            ),
            "brightness_gain": PluginParameter(
                name="brightness_gain",
                label="Brightness gain",
                type="float",
                default=0.9,
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                description="How strongly energy increases brightness.",
            ),
            "line_width_factor": PluginParameter(
                name="line_width_factor",
                label="Line width factor",
                type="float",
                default=0.008,
                minimum=0.001,
                maximum=0.05,
                step=0.001,
                description="Base width of the lines relative to frame size.",
            ),
            "radial_noise_speed": PluginParameter(
                name="radial_noise_speed",
                label="Radial noise speed",
                type="float",
                default=0.7,
                minimum=0.0,
                maximum=5.0,
                step=0.1,
                description="Speed of radial modulation oscillations.",
            ),
            "angular_warp_amount": PluginParameter(
                name="angular_warp_amount",
                label="Angular warp amount",
                type="float",
                default=0.25,
                minimum=0.0,
                maximum=1.0,
                step=0.02,
                description="How strongly the wedge pattern is warped in angle.",
            ),
            "core_radius_factor": PluginParameter(
                name="core_radius_factor",
                label="Core radius factor",
                type="float",
                default=0.12,
                minimum=0.02,
                maximum=0.4,
                step=0.01,
                description="Size of the central glowing disk as fraction of outer radius.",
            ),
        }

    # ------------------------------------------------------------------
    # Widget creation / host integration
    # ------------------------------------------------------------------
    def create_widget(self, parent: Optional[QWidget] = None) -> QWidget:
        """
        Create the Qt widget used for both preview and off-screen rendering.
        """
        self._widget = _KaleidoStemsWidget(self.config, parent=parent)
        return self._widget

    def apply_preview_settings(self, width: int, height: int, fps: int) -> None:
        """
        Let the plugin know about the intended preview size and FPS.
        """
        if self._widget is None:
            return

        if fps <= 0:
            fps = 25
        interval_ms = int(1000.0 / float(fps))
        self._widget.set_timer_interval(interval_ms)

    def on_audio_features(self, features: Dict[str, Any]) -> None:
        """
        Entry point called by the host with current audio features.

        Features are expected in the form:
            {
                "time_ms": <int>,
                "inputs": {
                    "input_1": {"rms": ...},  # drums
                    "input_2": {"rms": ...},  # bass
                    "input_3": {"rms": ...},  # vocals
                    "input_4": {"rms": ...},  # other
                }
            }
        """
        if self._widget is None:
            return

        time_ms = int(features.get("time_ms", 0))
        inputs = features.get("inputs", {}) or {}

        i1 = inputs.get("input_1", {})
        i2 = inputs.get("input_2", {})
        i3 = inputs.get("input_3", {})
        i4 = inputs.get("input_4", {})

        level_1 = float(i1.get("rms", 0.0))  # drums
        level_2 = float(i2.get("rms", 0.0))  # bass
        level_3 = float(i3.get("rms", 0.0))  # vocals
        level_4 = float(i4.get("rms", 0.0))  # other

        level_1 = _clamp(level_1, 0.0, 1.0)
        level_2 = _clamp(level_2, 0.0, 1.0)
        level_3 = _clamp(level_3, 0.0, 1.0)
        level_4 = _clamp(level_4, 0.0, 1.0)

        self._widget.update_audio_state(time_ms, level_1, level_2, level_3, level_4)

    def on_activate(self) -> None:
        """Called when the plugin becomes the active visualization."""
        return

    def on_deactivate(self) -> None:
        """Called when the plugin is no longer active."""
        return
