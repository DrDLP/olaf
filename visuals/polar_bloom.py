"""
Polar Bloom visualization plugin for Olaf.

A multi-stem, radial "flower" visualization:
- input_1 (bass)   -> outer ring petals (large, slow breathing)
- input_2 (drums)  -> middle ring petals (more jagged / high frequency)
- input_3 (vocals) -> inner flower (small, vivid, color-rich)
- input_4 (other)  -> decorative arcs / halos around the bloom

Fully QPainter-based, so it scales nicely to high resolutions (1080p, 1440p, 4K).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt, QTimer, QSize, QRectF
from PyQt6.QtGui import QPainter, QColor, QRadialGradient, QPainterPath, QPen
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


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smoothstep(x: float) -> float:
    """Classic smoothstep in [0, 1]."""
    x = _clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _hsv_to_qcolor(h: float, s: float, v: float) -> QColor:
    """
    Convert HSV (h in [0,1], s in [0,1], v in [0,1]) to a QColor.
    Hue is wrapped around 1.0.
    """
    h = h % 1.0
    s = _clamp(s, 0.0, 1.0)
    v = _clamp(v, 0.0, 1.0)
    color = QColor()
    color.setHsvF(h, s, v)
    return color


# ---------------------------------------------------------------------------
# Preview / export widget
# ---------------------------------------------------------------------------


class _PolarBloomWidget(QWidget):
    """
    Preview widget for the Polar Bloom visualization.

    Used both as:
      - a small real-time preview in the Visualizations tab,
      - a large off-screen widget during video export (high resolution).
    """

    def __init__(self, config: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config

        # Time in seconds (driven by the host via update_audio_state())
        self._time_s: float = 0.0

        # Per-stem normalized energies in [0, 1]
        # We follow the recommended mapping:
        #   - level_1: bass   (outer ring)
        #   - level_2: drums  (middle ring)
        #   - level_3: vocals (inner flower)
        #   - level_4: other  (decorative arcs)
        self._level_1: float = 0.0
        self._level_2: float = 0.0
        self._level_3: float = 0.0
        self._level_4: float = 0.0

        # Simple repaint timer for interactive preview; for off-screen export
        # the host drives time and calls update() as needed.
        self._timer = QTimer(self)
        self._timer.setInterval(40)  # ~25 FPS by default
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
        Adjust the internal timer interval.

        The host can call this via apply_preview_settings() to align the
        preview refresh rate with the nominal FPS.
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

        All levels are expected to be in [0, 1].
        """
        self._time_s = max(0.0, float(time_ms) / 1000.0)
        self._level_1 = _clamp(float(level_1), 0.0, 1.0)
        self._level_2 = _clamp(float(level_2), 0.0, 1.0)
        self._level_3 = _clamp(float(level_3), 0.0, 1.0)
        self._level_4 = _clamp(float(level_4), 0.0, 1.0)
        self.update()

    def _on_tick(self) -> None:
        """Internal timer callback for interactive preview."""
        self.update()

    # ------------------------------------------------------------------
    # Main painting logic
    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # type: ignore[override]
        width = self.width()
        height = self.height()
        if width <= 0 or height <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        cfg = self._config
        t = self._time_s

        # ------------------------------------------------------------------
        # Determine angular resolution (number of "petals" / segments).
        # Higher resolution for large off-screen renders.
        # ------------------------------------------------------------------
        base_petals = int(cfg.get("base_petals", 120))
        base_petals = max(24, min(360, base_petals))

        if width >= 1280 and height >= 720:
            scale_w = width / 640.0
            scale_h = height / 360.0
            upscale = max(scale_w, scale_h)
            upscale = _clamp(upscale, 1.0, 3.0)
            num_petals = int(base_petals * upscale)
            num_petals = max(48, min(720, num_petals))
        else:
            num_petals = base_petals

        # ------------------------------------------------------------------
        # Global parameters and colors
        # ------------------------------------------------------------------
        level_bass = self._level_1
        level_drums = self._level_2
        level_vocals = self._level_3
        level_other = self._level_4

        # A rough global energy to modulate background / bloom
        global_energy = _clamp(
            0.4 * level_bass + 0.3 * level_drums + 0.2 * level_vocals + 0.1 * level_other,
            0.0,
            1.0,
        )

        # Background radial gradient (dark center, slightly tinted edges)
        cx = width * 0.5
        cy = height * 0.5
        min_dim = float(min(width, height))
        bg_radius = min_dim * 0.65

        bg_grad = QRadialGradient(cx, cy, bg_radius, cx, cy)
        bg_inner = QColor(5, 5, 15)
        bg_outer = QColor(20, 10, 40)

        # Slightly brighten with energy
        e_factor = 0.3 + 0.7 * global_energy
        bg_outer = QColor(
            int(bg_outer.red() * e_factor),
            int(bg_outer.green() * e_factor),
            int(bg_outer.blue() * e_factor),
        )

        bg_grad.setColorAt(0.0, bg_inner)
        bg_grad.setColorAt(1.0, bg_outer)
        painter.fillRect(self.rect(), bg_grad)

        # ------------------------------------------------------------------
        # Polar bloom parameters for each ring
        # ------------------------------------------------------------------
        outer_radius_factor = float(cfg.get("outer_radius_factor", 0.78))
        outer_radius_factor = _clamp(outer_radius_factor, 0.3, 1.1)

        mid_radius_factor = float(cfg.get("mid_radius_factor", 0.52))
        mid_radius_factor = _clamp(mid_radius_factor, 0.2, 0.9)

        inner_radius_factor = float(cfg.get("inner_radius_factor", 0.28))
        inner_radius_factor = _clamp(inner_radius_factor, 0.1, 0.7)

        outer_petal_amp = float(cfg.get("outer_petal_amplitude", 0.22))
        outer_petal_amp = _clamp(outer_petal_amp, 0.0, 0.6)

        mid_petal_amp = float(cfg.get("mid_petal_amplitude", 0.18))
        mid_petal_amp = _clamp(mid_petal_amp, 0.0, 0.6)

        inner_petal_amp = float(cfg.get("inner_petal_amplitude", 0.16))
        inner_petal_amp = _clamp(inner_petal_amp, 0.0, 0.6)

        outer_petal_freq = float(cfg.get("outer_petal_frequency", 6.0))
        outer_petal_freq = _clamp(outer_petal_freq, 1.0, 24.0)

        mid_petal_freq = float(cfg.get("mid_petal_frequency", 10.0))
        mid_petal_freq = _clamp(mid_petal_freq, 2.0, 40.0)

        inner_petal_freq = float(cfg.get("inner_petal_frequency", 14.0))
        inner_petal_freq = _clamp(inner_petal_freq, 2.0, 48.0)

        global_rotation_deg = float(cfg.get("global_rotation_speed_deg", 40.0))
        global_rotation_deg = _clamp(global_rotation_deg, -360.0, 360.0)

        hue_cycle_speed_deg = float(cfg.get("hue_cycle_speed_deg", 60.0))
        hue_cycle_speed_deg = _clamp(hue_cycle_speed_deg, 0.0, 360.0)

        saturation = float(cfg.get("saturation", 0.9))
        saturation = _clamp(saturation, 0.0, 1.0)

        brightness_base = float(cfg.get("brightness_base", 0.65))
        brightness_base = _clamp(brightness_base, 0.1, 1.0)

        brightness_energy_gain = float(cfg.get("brightness_energy_gain", 0.85))
        brightness_energy_gain = _clamp(brightness_energy_gain, 0.0, 2.0)

        deco_strength = float(cfg.get("decoration_strength", 0.9))
        deco_strength = _clamp(deco_strength, 0.0, 2.0)

        deco_thickness = float(cfg.get("decoration_thickness", 0.02))
        deco_thickness = _clamp(deco_thickness, 0.001, 0.08)

        # Derived radii (in pixels)
        base_radius = min_dim * 0.5
        outer_radius = outer_radius_factor * base_radius
        mid_radius = mid_radius_factor * base_radius
        inner_radius = inner_radius_factor * base_radius

        # Global rotation (in radians)
        rot_deg = global_rotation_deg * t
        rot_rad = math.radians(rot_deg)

        # Hue shift over time
        hue_shift = (hue_cycle_speed_deg * t) / 360.0

        painter.setPen(Qt.PenStyle.NoPen)

        # ------------------------------------------------------------------
        # Helper to build and draw a single polar "flower" ring
        # ------------------------------------------------------------------
        def draw_ring(
            base_r: float,
            petal_amp: float,
            petal_freq: float,
            energy: float,
            base_hue: float,
            ring_alpha: int,
            jaggedness: float,
        ) -> None:
            """
            Draw one radial "flower" ring as a filled polygon.

            - base_r: base radius
            - petal_amp: relative radial modulation (0..1)
            - petal_freq: frequency of petal lobes
            - energy: stem energy in [0, 1]
            - base_hue: base hue in [0, 1]
            - ring_alpha: global alpha (0..255)
            - jaggedness: adds extra angular modulation / noise.
            """
            if base_r <= 0.0 or ring_alpha <= 0:
                return

            path = QPainterPath()
            first = True

            # Energy-smoothed amplitude
            amp = petal_amp * (0.2 + 0.8 * _smoothstep(energy))
            # Slight time-based phase to make petals twist
            phase = 2.0 * math.pi * t * (0.15 + 0.35 * energy)

            for i in range(num_petals + 1):
                u = i / float(num_petals)
                theta = u * (2.0 * math.pi)

                # Petal modulation
                base_wave = 0.5 * (1.0 + math.sin(theta * petal_freq + phase))

                # Extra jaggedness from a higher-frequency component
                jag_wave = 0.5 * (1.0 + math.sin(theta * petal_freq * 2.3 - t * 1.7))
                jag_mix = _lerp(base_wave, jag_wave, jaggedness)

                local_amp = amp * jag_mix
                r = base_r * (1.0 + local_amp)

                # Global rotation
                theta_rot = theta + rot_rad

                x = cx + r * math.cos(theta_rot)
                y = cy + r * math.sin(theta_rot)

                if first:
                    path.moveTo(x, y)
                    first = False
                else:
                    path.lineTo(x, y)

            # Close the shape at the center to make a filled ring-like flower
            path.closeSubpath()

            # Build color based on base hue + energy + global hue shift
            h = (base_hue + hue_shift) % 1.0
            v = brightness_base + brightness_energy_gain * energy
            col = _hsv_to_qcolor(h, saturation, v)
            col.setAlpha(ring_alpha)

            painter.setBrush(col)
            painter.drawPath(path)

        # ------------------------------------------------------------------
        # Draw outer, middle, inner rings
        # ------------------------------------------------------------------
        # Outer (bass) – larger, smoother petals
        draw_ring(
            base_r=outer_radius,
            petal_amp=outer_petal_amp,
            petal_freq=outer_petal_freq,
            energy=level_bass,
            base_hue=0.62,  # blue / violet
            ring_alpha=int(120 + 100 * level_bass),
            jaggedness=0.25,
        )

        # Middle (drums) – sharper, more jagged
        draw_ring(
            base_r=mid_radius,
            petal_amp=mid_petal_amp,
            petal_freq=mid_petal_freq,
            energy=level_drums,
            base_hue=0.90,  # magenta / pink
            ring_alpha=int(140 + 100 * level_drums),
            jaggedness=0.6,
        )

        # Inner (vocals) – small, very vivid
        draw_ring(
            base_r=inner_radius,
            petal_amp=inner_petal_amp,
            petal_freq=inner_petal_freq,
            energy=level_vocals,
            base_hue=0.08,  # warm orange/yellow
            ring_alpha=int(160 + 80 * level_vocals),
            jaggedness=0.4,
        )

        # ------------------------------------------------------------------
        # Central core disc for vocals (glow)
        # ------------------------------------------------------------------
        core_radius = inner_radius * (0.35 + 0.45 * _smoothstep(level_vocals))
        if core_radius > 0.0:
            core_grad = QRadialGradient(cx, cy, core_radius, cx, cy)
            core_h = (0.08 + hue_shift * 0.5) % 1.0
            core_v_outer = brightness_base * 0.6
            core_v_inner = brightness_base + brightness_energy_gain * level_vocals

            col_inner = _hsv_to_qcolor(core_h, saturation, core_v_inner)
            col_outer = _hsv_to_qcolor(core_h, saturation * 0.6, core_v_outer)
            col_inner.setAlpha(255)
            col_outer.setAlpha(int(80 + 100 * level_vocals))

            core_grad.setColorAt(0.0, col_inner)
            core_grad.setColorAt(1.0, col_outer)

            painter.setBrush(core_grad)
            painter.setPen(Qt.PenStyle.NoPen)

            # Use a QRectF so drawEllipse receives a single rect argument
            core_rect = QRectF(
                cx - core_radius,
                cy - core_radius,
                core_radius * 2.0,
                core_radius * 2.0,
            )
            painter.drawEllipse(core_rect)


        # ------------------------------------------------------------------
        # Decorative arcs / halos (input_4)
        # ------------------------------------------------------------------
        if deco_strength > 0.0 and level_other > 0.01:
            painter.setBrush(Qt.BrushStyle.NoBrush)

            # Radius for decorative arcs: just outside the outer ring
            deco_radius = outer_radius * (1.05 + 0.35 * deco_strength)
            arc_thick_px = deco_thickness * min_dim
            arc_thick_px = _clamp(arc_thick_px, 1.0, min_dim * 0.05)

            # Number of arcs scales with energy
            base_arcs = 6
            max_arcs = 20
            nb_arcs = int(_lerp(base_arcs, max_arcs, _smoothstep(level_other)))
            nb_arcs = max(1, nb_arcs)

            for k in range(nb_arcs):
                # Each arc is a partial circle with a varying span
                u = k / float(nb_arcs)
                start_angle_deg = (u * 360.0) + (t * 50.0 * (0.5 + level_other))
                span_deg = 20.0 + 90.0 * (0.3 + 0.7 * level_other)

                # Color per arc
                h = (0.55 + 0.25 * u + hue_shift * 0.3) % 1.0
                v = brightness_base + 0.7 * brightness_energy_gain * level_other
                col = _hsv_to_qcolor(h, saturation * 0.8, v)
                alpha = int(80 + 140 * level_other * deco_strength)
                col.setAlpha(alpha)

                pen = QPen(col)
                pen.setWidthF(arc_thick_px)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)

                # QRectF for the arc
                rect = QRectF(
                    cx - deco_radius,
                    cy - deco_radius,
                    deco_radius * 2.0,
                    deco_radius * 2.0,
                )

                # QPainter.drawArc expects angles in 1/16 degrees
                start_angle_16 = int(start_angle_deg * 16.0)
                span_angle_16 = int(span_deg * 16.0)
                painter.drawArc(rect, start_angle_16, span_angle_16)

        painter.end()


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------


class PolarBloomVisualization(BaseVisualization):
    """
    Audio-reactive polar flower visualization.

    - Outer ring (input_1, bass): wide petals on a large radius.
    - Middle ring (input_2, drums): sharper petals with more jaggedness.
    - Inner flower (input_3, vocals): small, vivid core bloom.
    - Decorative arcs (input_4, other): halos and arcs around the flower.
    """

    plugin_id = "dr_dlp_polar_bloom"
    plugin_name = "Polar Bloom"
    plugin_author = "Dr DLP"
    plugin_version = "1.0.0"
    plugin_max_inputs = 4

    plugin_description = (
        "Multi-layer polar \"flower\" driven by multiple stems.\n"
        "\n"
        "Recommended stem routing:\n"
        "  - input_1: bass   -> outer ring petals (large, slow breathing)\n"
        "  - input_2: drums  -> middle ring petals (more jagged / rhythmic)\n"
        "  - input_3: vocals -> inner flower core (small, very colorful)\n"
        "  - input_4: other  -> decorative arcs / halos around the bloom\n"
        "\n"
        "You can still route any stem to any input, but this mapping usually\n"
        "gives the most readable and musical polar bloom."
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._widget: Optional[_PolarBloomWidget] = None

        defaults = {
            "base_petals": 120,
            "outer_radius_factor": 0.78,
            "mid_radius_factor": 0.52,
            "inner_radius_factor": 0.28,
            "outer_petal_amplitude": 0.22,
            "mid_petal_amplitude": 0.18,
            "inner_petal_amplitude": 0.16,
            "outer_petal_frequency": 6.0,
            "mid_petal_frequency": 10.0,
            "inner_petal_frequency": 14.0,
            "global_rotation_speed_deg": 40.0,
            "hue_cycle_speed_deg": 60.0,
            "saturation": 0.9,
            "brightness_base": 0.65,
            "brightness_energy_gain": 0.85,
            "decoration_strength": 0.9,
            "decoration_thickness": 0.02,
        }
        for key, value in defaults.items():
            self.config.setdefault(key, value)

    # ------------------------------------------------------------------
    # Plugin parameter definitions (UI schema)
    # ------------------------------------------------------------------
    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        """
        Expose configurable parameters to the host UI.
        """
        return {
            "base_petals": PluginParameter(
                name="base_petals",
                label="Base petal resolution",
                type="int",
                default=120,
                minimum=24,
                maximum=360,
                step=4,
                description="Base angular resolution of the rings (scaled up at high resolution).",
            ),
            "outer_radius_factor": PluginParameter(
                name="outer_radius_factor",
                label="Outer ring radius",
                type="float",
                default=0.78,
                minimum=0.3,
                maximum=1.1,
                step=0.02,
                description="Radius of the outer bass ring as a fraction of half the frame size.",
            ),
            "mid_radius_factor": PluginParameter(
                name="mid_radius_factor",
                label="Middle ring radius",
                type="float",
                default=0.52,
                minimum=0.2,
                maximum=0.9,
                step=0.02,
                description="Radius of the middle drums ring.",
            ),
            "inner_radius_factor": PluginParameter(
                name="inner_radius_factor",
                label="Inner ring radius",
                type="float",
                default=0.28,
                minimum=0.1,
                maximum=0.7,
                step=0.02,
                description="Radius of the inner vocals ring / core bloom.",
            ),
            "outer_petal_amplitude": PluginParameter(
                name="outer_petal_amplitude",
                label="Outer petal amplitude",
                type="float",
                default=0.22,
                minimum=0.0,
                maximum=0.6,
                step=0.02,
                description="Radial modulation amplitude for outer bass petals.",
            ),
            "mid_petal_amplitude": PluginParameter(
                name="mid_petal_amplitude",
                label="Middle petal amplitude",
                type="float",
                default=0.18,
                minimum=0.0,
                maximum=0.6,
                step=0.02,
                description="Radial modulation amplitude for middle drums petals.",
            ),
            "inner_petal_amplitude": PluginParameter(
                name="inner_petal_amplitude",
                label="Inner petal amplitude",
                type="float",
                default=0.16,
                minimum=0.0,
                maximum=0.6,
                step=0.02,
                description="Radial modulation amplitude for inner vocals petals.",
            ),
            "outer_petal_frequency": PluginParameter(
                name="outer_petal_frequency",
                label="Outer petal frequency",
                type="float",
                default=6.0,
                minimum=1.0,
                maximum=24.0,
                step=0.5,
                description="Number of lobes around the circle for the outer bass ring.",
            ),
            "mid_petal_frequency": PluginParameter(
                name="mid_petal_frequency",
                label="Middle petal frequency",
                type="float",
                default=10.0,
                minimum=2.0,
                maximum=40.0,
                step=0.5,
                description="Number of lobes for the middle drums ring.",
            ),
            "inner_petal_frequency": PluginParameter(
                name="inner_petal_frequency",
                label="Inner petal frequency",
                type="float",
                default=14.0,
                minimum=2.0,
                maximum=48.0,
                step=0.5,
                description="Number of lobes for the inner vocals ring.",
            ),
            "global_rotation_speed_deg": PluginParameter(
                name="global_rotation_speed_deg",
                label="Global rotation speed (deg/s)",
                type="float",
                default=40.0,
                minimum=-360.0,
                maximum=360.0,
                step=5.0,
                description="Rotation speed of the whole bloom (can be negative for opposite rotation).",
            ),
            "hue_cycle_speed_deg": PluginParameter(
                name="hue_cycle_speed_deg",
                label="Hue cycle speed (deg/s)",
                type="float",
                default=60.0,
                minimum=0.0,
                maximum=360.0,
                step=5.0,
                description="How fast the colors cycle around the hue wheel.",
            ),
            "saturation": PluginParameter(
                name="saturation",
                label="Saturation",
                type="float",
                default=0.9,
                minimum=0.0,
                maximum=1.0,
                step=0.02,
                description="Global saturation of the bloom colors.",
            ),
            "brightness_base": PluginParameter(
                name="brightness_base",
                label="Base brightness",
                type="float",
                default=0.65,
                minimum=0.1,
                maximum=1.0,
                step=0.02,
                description="Base brightness of the rings.",
            ),
            "brightness_energy_gain": PluginParameter(
                name="brightness_energy_gain",
                label="Brightness energy gain",
                type="float",
                default=0.85,
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                description="How strongly audio energy boosts the brightness.",
            ),
            "decoration_strength": PluginParameter(
                name="decoration_strength",
                label="Decoration strength",
                type="float",
                default=0.9,
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                description="How intense the outer decorative arcs (input_4) are.",
            ),
            "decoration_thickness": PluginParameter(
                name="decoration_thickness",
                label="Decoration thickness",
                type="float",
                default=0.02,
                minimum=0.001,
                maximum=0.08,
                step=0.005,
                description="Thickness of decorative arcs as a fraction of the frame size.",
            ),
        }

    # ------------------------------------------------------------------
    # Widget creation / host integration
    # ------------------------------------------------------------------
    def create_widget(self, parent: Optional[QWidget] = None) -> QWidget:
        """
        Create the Qt widget used for both preview and off-screen rendering.
        """
        self._widget = _PolarBloomWidget(self.config, parent=parent)
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
                    "input_1": {"rms": ...},  # bass
                    "input_2": {"rms": ...},  # drums
                    "input_3": {"rms": ...},  # vocals
                    "input_4": {"rms": ...},  # other
                }
            }
        """
        if self._widget is None:
            return

        time_ms = int(features.get("time_ms", 0))
        inputs = features.get("inputs", {}) or {}

        input_1 = inputs.get("input_1", {})
        input_2 = inputs.get("input_2", {})
        input_3 = inputs.get("input_3", {})
        input_4 = inputs.get("input_4", {})

        level_1 = float(input_1.get("rms", 0.0))  # bass
        level_2 = float(input_2.get("rms", 0.0))  # drums
        level_3 = float(input_3.get("rms", 0.0))  # vocals
        level_4 = float(input_4.get("rms", 0.0))  # other

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
