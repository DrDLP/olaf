"""
Stem Orbits visualization plugin for Olaf.

A system of concentric, audio-reactive orbits:
- input_1 (drums)  -> outer ring, fast rotation and strong flicker
- input_2 (bass)   -> middle ring, slower, heavier points
- input_3 (vocals) -> inner ring, thin, warm-colored orbit
- input_4 (other)  -> small decorative satellites / broken outer ring

Fully QPainter-based so it scales cleanly to high resolutions.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt, QTimer, QSize, QRectF
from PyQt6.QtGui import QPainter, QColor, QRadialGradient, QPen
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
    h = h % 1.0
    s = _clamp(s, 0.0, 1.0)
    v = _clamp(v, 0.0, 1.0)
    c = QColor()
    c.setHsvF(h, s, v)
    return c


# ---------------------------------------------------------------------------
# Preview / export widget
# ---------------------------------------------------------------------------


class _StemOrbitsWidget(QWidget):
    """
    Preview / export widget for the Stem Orbits visualization.

    Used both as:
      - a small real-time preview in the Visualizations tab,
      - a large off-screen widget during video export (high resolution).
    """

    def __init__(self, config: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config

        self._time_s: float = 0.0
        self._level_1: float = 0.0  # drums -> outer ring
        self._level_2: float = 0.0  # bass  -> middle ring
        self._level_3: float = 0.0  # vocals-> inner ring
        self._level_4: float = 0.0  # other -> satellites

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
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        cfg = self._config
        t = self._time_s

        # ------------------------------------------------------------------
        # Resolution scaling: base number of particles per ring
        # ------------------------------------------------------------------
        base_particles = int(cfg.get("base_particles", 80))
        base_particles = max(8, min(200, base_particles))

        if w >= 1280 and h >= 720:
            scale_w = w / 640.0
            scale_h = h / 360.0
            upscale = max(scale_w, scale_h)
            upscale = _clamp(upscale, 1.0, 3.0)
            particles_base = int(base_particles * upscale)
            particles_base = max(16, min(360, particles_base))
        else:
            particles_base = base_particles

        # ------------------------------------------------------------------
        # Audio levels
        # ------------------------------------------------------------------
        level_drums = self._level_1
        level_bass = self._level_2
        level_vocals = self._level_3
        level_other = self._level_4

        global_energy = _clamp(
            0.3 * level_drums + 0.3 * level_bass + 0.25 * level_vocals + 0.15 * level_other,
            0.0,
            1.0,
        )

        cx = w * 0.5
        cy = h * 0.5
        min_dim = float(min(w, h))

        # ------------------------------------------------------------------
        # Background radial gradient
        # ------------------------------------------------------------------
        bg_radius = min_dim * 0.75
        bg_grad = QRadialGradient(cx, cy, bg_radius, cx, cy)
        bg_center = QColor(5, 5, 12)
        bg_edge = QColor(15, 20, 45)
        e_factor = 0.3 + 0.7 * global_energy
        bg_edge = QColor(
            int(bg_edge.red() * e_factor),
            int(bg_edge.green() * e_factor),
            int(bg_edge.blue() * e_factor),
        )
        bg_grad.setColorAt(0.0, bg_center)
        bg_grad.setColorAt(1.0, bg_edge)
        painter.fillRect(self.rect(), bg_grad)

        # ------------------------------------------------------------------
        # Global zoom of the orbital system
        # ------------------------------------------------------------------
        orbit_zoom_amount = float(cfg.get("orbit_zoom_amount", 0.20))
        orbit_zoom_amount = _clamp(orbit_zoom_amount, 0.0, 0.6)
        zoom = 0.85 + orbit_zoom_amount * global_energy

        base_radius = min_dim * 0.5 * zoom

        # ------------------------------------------------------------------
        # Radii factors
        # ------------------------------------------------------------------
        outer_radius_factor = float(cfg.get("outer_radius_factor", 0.85))
        outer_radius_factor = _clamp(outer_radius_factor, 0.3, 1.2)

        mid_radius_factor = float(cfg.get("mid_radius_factor", 0.60))
        mid_radius_factor = _clamp(mid_radius_factor, 0.2, 1.0)

        inner_radius_factor = float(cfg.get("inner_radius_factor", 0.35))
        inner_radius_factor = _clamp(inner_radius_factor, 0.1, 0.8)

        # ------------------------------------------------------------------
        # Particle sizes (fractions of min_dim)
        # ------------------------------------------------------------------
        outer_particle_size = float(cfg.get("outer_particle_size", 0.018))
        outer_particle_size = _clamp(outer_particle_size, 0.004, 0.08)

        mid_particle_size = float(cfg.get("mid_particle_size", 0.022))
        mid_particle_size = _clamp(mid_particle_size, 0.004, 0.10)

        inner_particle_size = float(cfg.get("inner_particle_size", 0.018))
        inner_particle_size = _clamp(inner_particle_size, 0.004, 0.08)

        # ------------------------------------------------------------------
        # Rotation speeds in degrees per second
        # ------------------------------------------------------------------
        outer_rot_speed_deg = float(cfg.get("outer_rotation_speed_deg", 90.0))
        outer_rot_speed_deg = _clamp(outer_rot_speed_deg, -360.0, 360.0)

        mid_rot_speed_deg = float(cfg.get("mid_rotation_speed_deg", 45.0))
        mid_rot_speed_deg = _clamp(mid_rot_speed_deg, -360.0, 360.0)

        inner_rot_speed_deg = float(cfg.get("inner_rotation_speed_deg", 25.0))
        inner_rot_speed_deg = _clamp(inner_rot_speed_deg, -360.0, 360.0)

        hue_cycle_speed_deg = float(cfg.get("hue_cycle_speed_deg", 50.0))
        hue_cycle_speed_deg = _clamp(hue_cycle_speed_deg, 0.0, 360.0)

        saturation = float(cfg.get("saturation", 0.9))
        saturation = _clamp(saturation, 0.0, 1.0)

        brightness_base = float(cfg.get("brightness_base", 0.70))
        brightness_base = _clamp(brightness_base, 0.1, 1.0)

        brightness_energy_gain = float(cfg.get("brightness_energy_gain", 0.9))
        brightness_energy_gain = _clamp(brightness_energy_gain, 0.0, 2.0)

        # ------------------------------------------------------------------
        # Decorative arcs (satellites) parameters
        # ------------------------------------------------------------------
        deco_strength = float(cfg.get("decoration_strength", 1.0))
        deco_strength = _clamp(deco_strength, 0.0, 2.0)

        deco_thickness = float(cfg.get("decoration_thickness", 0.015))
        deco_thickness = _clamp(deco_thickness, 0.001, 0.05)

        deco_radius_factor = float(cfg.get("decoration_radius_factor", 1.05))
        deco_radius_factor = _clamp(deco_radius_factor, 0.8, 1.4)

        deco_rot_speed_deg = float(cfg.get("decoration_rotation_speed_deg", 35.0))
        deco_rot_speed_deg = _clamp(deco_rot_speed_deg, -360.0, 360.0)

        # Derived radii
        outer_radius = outer_radius_factor * base_radius
        mid_radius = mid_radius_factor * base_radius
        inner_radius = inner_radius_factor * base_radius

        # Hue shift over time
        hue_shift = (hue_cycle_speed_deg * t) / 360.0

        painter.setPen(Qt.PenStyle.NoPen)

        # ------------------------------------------------------------------
        # Helper to draw a single orbit ring of particles
        # ------------------------------------------------------------------
        def draw_orbit(
            radius: float,
            level: float,
            particle_size_factor: float,
            rot_speed_deg: float,
            base_hue: float,
            energy_weight: float,
        ) -> None:
            if radius <= 0.0 or level <= 0.001:
                return

            # Number of particles scales with level and resolution
            count = int(particles_base * (0.3 + 0.7 * level))
            count = max(3, min(5 * particles_base, count))

            # Rotation for this ring (faster when level is high)
            rot_deg = rot_speed_deg * t * (0.4 + 0.6 * level)
            rot_rad = math.radians(rot_deg)

            # Pixel radius of each particle
            r_px = particle_size_factor * min_dim * (0.5 + level)
            r_px = _clamp(r_px, 1.0, min_dim * 0.08)

            for i in range(count):
                u = i / float(count)
                theta = 2.0 * math.pi * u + rot_rad

                # Slight radial jitter to avoid a perfect circle
                jitter = 0.03 * radius * level * math.sin(2.0 * math.pi * u * 3.0 + t * 2.5)
                r = radius + jitter

                x = cx + r * math.cos(theta)
                y = cy + r * math.sin(theta)

                # Color for this particle
                h = (base_hue + hue_shift + 0.15 * u) % 1.0
                v = brightness_base + brightness_energy_gain * (energy_weight * level)
                col = _hsv_to_qcolor(h, saturation, v)
                alpha = int(120 + 135 * level)
                col.setAlpha(alpha)

                painter.setBrush(col)

                rect = QRectF(
                    x - r_px,
                    y - r_px,
                    2.0 * r_px,
                    2.0 * r_px,
                )
                painter.drawEllipse(rect)

        # ------------------------------------------------------------------
        # Outer, middle, inner orbits
        # ------------------------------------------------------------------
        # Outer orbit (drums)
        draw_orbit(
            radius=outer_radius,
            level=level_drums,
            particle_size_factor=outer_particle_size,
            rot_speed_deg=outer_rot_speed_deg,
            base_hue=0.56,  # cyan / teal
            energy_weight=1.0,
        )

        # Middle orbit (bass)
        draw_orbit(
            radius=mid_radius,
            level=level_bass,
            particle_size_factor=mid_particle_size,
            rot_speed_deg=mid_rot_speed_deg,
            base_hue=0.08,  # warm yellow/orange
            energy_weight=0.9,
        )

        # Inner orbit (vocals)
        draw_orbit(
            radius=inner_radius,
            level=level_vocals,
            particle_size_factor=inner_particle_size,
            rot_speed_deg=inner_rot_speed_deg,
            base_hue=0.85,  # magenta / pink
            energy_weight=1.0,
        )

        # ------------------------------------------------------------------
        # Decorative satellites / broken ring (input_4)
        # ------------------------------------------------------------------
        if deco_strength > 0.0 and level_other > 0.01:
            painter.setBrush(Qt.BrushStyle.NoBrush)

            deco_radius = deco_radius_factor * outer_radius
            arc_thick = deco_thickness * min_dim * (0.5 + 0.5 * deco_strength)
            arc_thick = _clamp(arc_thick, 1.0, min_dim * 0.04)

            base_arcs = 4
            max_arcs = 16
            nb_arcs = int(_lerp(base_arcs, max_arcs, _smoothstep(level_other)))
            nb_arcs = max(1, nb_arcs)

            rot_deg = deco_rot_speed_deg * t * (0.4 + 0.6 * level_other)

            for k in range(nb_arcs):
                u = k / float(nb_arcs)
                start_deg = u * 360.0 + rot_deg
                span_deg = 20.0 + 70.0 * (0.2 + 0.8 * level_other)

                h = (0.6 + 0.25 * u + hue_shift * 0.4) % 1.0
                v = brightness_base + 0.7 * brightness_energy_gain * level_other
                col = _hsv_to_qcolor(h, saturation * 0.8, v)
                alpha = int(70 + 150 * level_other * deco_strength)
                col.setAlpha(alpha)

                pen = QPen(col)
                pen.setWidthF(arc_thick)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)

                rect = QRectF(
                    cx - deco_radius,
                    cy - deco_radius,
                    deco_radius * 2.0,
                    deco_radius * 2.0,
                )
                start_16 = int(start_deg * 16.0)
                span_16 = int(span_deg * 16.0)
                painter.drawArc(rect, start_16, span_16)

        painter.end()


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------


class StemOrbitsVisualization(BaseVisualization):
    """
    Audio-reactive concentric orbits visualization.

    - Outer ring (input_1, drums): fast rotation, strong flicker.
    - Middle ring (input_2, bass): heavier, slower points.
    - Inner ring (input_3, vocals): thin, colorful orbit.
    - Satellites (input_4, other): broken arcs / small satellites.
    """

    plugin_id = "dr_dlp_stem_orbits"
    plugin_name = "Stem Orbits"
    plugin_author = "Dr DLP"
    plugin_version = "1.0.0"
    plugin_max_inputs = 4

    plugin_description = (
        "Concentric orbital rings driven by multiple stems.\n"
        "\n"
        "Recommended stem routing:\n"
        "  - input_1: drums  -> outer ring (fast rotation, strong flicker)\n"
        "  - input_2: bass   -> middle ring (heavier, slower points)\n"
        "  - input_3: vocals -> inner ring (thin, warm orbit)\n"
        "  - input_4: other  -> small decorative satellites / broken outer ring\n"
        "\n"
        "You can still route any stem to any input, but this mapping usually\n"
        "gives a clean and readable orbital system."
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._widget: Optional[_StemOrbitsWidget] = None

        # Default configuration values (also used by the UI as defaults).
        defaults = {
            "base_particles": 80,
            "outer_radius_factor": 0.85,
            "mid_radius_factor": 0.60,
            "inner_radius_factor": 0.35,
            "outer_particle_size": 0.018,
            "mid_particle_size": 0.022,
            "inner_particle_size": 0.018,
            "outer_rotation_speed_deg": 90.0,
            "mid_rotation_speed_deg": 45.0,
            "inner_rotation_speed_deg": 25.0,
            "orbit_zoom_amount": 0.20,
            "hue_cycle_speed_deg": 50.0,
            "saturation": 0.9,
            "brightness_base": 0.70,
            "brightness_energy_gain": 0.9,
            "decoration_strength": 1.0,
            "decoration_thickness": 0.015,
            "decoration_radius_factor": 1.05,
            "decoration_rotation_speed_deg": 35.0,
        }
        for k, v in defaults.items():
            self.config.setdefault(k, v)

    # ------------------------------------------------------------------
    # Plugin parameter definitions (UI schema)
    # ------------------------------------------------------------------
    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        """
        Expose configurable parameters to the host UI.
        """
        return {
            "base_particles": PluginParameter(
                name="base_particles",
                label="Base particle count",
                type="int",
                default=80,
                minimum=8,
                maximum=200,
                step=4,
                description="Base number of particles per ring (scaled up at high resolution).",
            ),
            "outer_radius_factor": PluginParameter(
                name="outer_radius_factor",
                label="Outer ring radius",
                type="float",
                default=0.85,
                minimum=0.3,
                maximum=1.2,
                step=0.02,
                description="Radius of the outer drums ring as fraction of half the frame size.",
            ),
            "mid_radius_factor": PluginParameter(
                name="mid_radius_factor",
                label="Middle ring radius",
                type="float",
                default=0.60,
                minimum=0.2,
                maximum=1.0,
                step=0.02,
                description="Radius of the middle bass ring.",
            ),
            "inner_radius_factor": PluginParameter(
                name="inner_radius_factor",
                label="Inner ring radius",
                type="float",
                default=0.35,
                minimum=0.1,
                maximum=0.8,
                step=0.02,
                description="Radius of the inner vocals ring.",
            ),
            "outer_particle_size": PluginParameter(
                name="outer_particle_size",
                label="Outer particle size",
                type="float",
                default=0.018,
                minimum=0.004,
                maximum=0.08,
                step=0.002,
                description="Size of outer ring particles as fraction of the frame size.",
            ),
            "mid_particle_size": PluginParameter(
                name="mid_particle_size",
                label="Middle particle size",
                type="float",
                default=0.022,
                minimum=0.004,
                maximum=0.10,
                step=0.002,
                description="Size of middle ring particles.",
            ),
            "inner_particle_size": PluginParameter(
                name="inner_particle_size",
                label="Inner particle size",
                type="float",
                default=0.018,
                minimum=0.004,
                maximum=0.08,
                step=0.002,
                description="Size of inner ring particles.",
            ),
            "outer_rotation_speed_deg": PluginParameter(
                name="outer_rotation_speed_deg",
                label="Outer rotation speed (deg/s)",
                type="float",
                default=90.0,
                minimum=-360.0,
                maximum=360.0,
                step=5.0,
                description="Rotation speed of the outer drums ring.",
            ),
            "mid_rotation_speed_deg": PluginParameter(
                name="mid_rotation_speed_deg",
                label="Middle rotation speed (deg/s)",
                type="float",
                default=45.0,
                minimum=-360.0,
                maximum=360.0,
                step=5.0,
                description="Rotation speed of the middle bass ring.",
            ),
            "inner_rotation_speed_deg": PluginParameter(
                name="inner_rotation_speed_deg",
                label="Inner rotation speed (deg/s)",
                type="float",
                default=25.0,
                minimum=-360.0,
                maximum=360.0,
                step=5.0,
                description="Rotation speed of the inner vocals ring.",
            ),
            "orbit_zoom_amount": PluginParameter(
                name="orbit_zoom_amount",
                label="Orbit zoom amount",
                type="float",
                default=0.20,
                minimum=0.0,
                maximum=0.6,
                step=0.02,
                description="How much the whole orbital system zooms with the mix energy.",
            ),
            "hue_cycle_speed_deg": PluginParameter(
                name="hue_cycle_speed_deg",
                label="Hue cycle speed (deg/s)",
                type="float",
                default=50.0,
                minimum=0.0,
                maximum=360.0,
                step=5.0,
                description="How fast colors move around the hue wheel.",
            ),
            "saturation": PluginParameter(
                name="saturation",
                label="Saturation",
                type="float",
                default=0.9,
                minimum=0.0,
                maximum=1.0,
                step=0.02,
                description="Global saturation of the ring colors.",
            ),
            "brightness_base": PluginParameter(
                name="brightness_base",
                label="Base brightness",
                type="float",
                default=0.70,
                minimum=0.1,
                maximum=1.0,
                step=0.02,
                description="Base brightness of all rings.",
            ),
            "brightness_energy_gain": PluginParameter(
                name="brightness_energy_gain",
                label="Brightness energy gain",
                type="float",
                default=0.9,
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                description="How strongly audio energy boosts brightness.",
            ),
            "decoration_strength": PluginParameter(
                name="decoration_strength",
                label="Decoration strength",
                type="float",
                default=1.0,
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                description="Intensity of the outer decorative arcs from input_4.",
            ),
            "decoration_thickness": PluginParameter(
                name="decoration_thickness",
                label="Decoration thickness",
                type="float",
                default=0.015,
                minimum=0.001,
                maximum=0.05,
                step=0.002,
                description="Thickness of decorative arcs as fraction of the frame size.",
            ),
            "decoration_radius_factor": PluginParameter(
                name="decoration_radius_factor",
                label="Decoration radius factor",
                type="float",
                default=1.05,
                minimum=0.8,
                maximum=1.4,
                step=0.02,
                description="Radius of outer decorative arcs relative to outer ring.",
            ),
            "decoration_rotation_speed_deg": PluginParameter(
                name="decoration_rotation_speed_deg",
                label="Decoration rotation speed (deg/s)",
                type="float",
                default=35.0,
                minimum=-360.0,
                maximum=360.0,
                step=5.0,
                description="Rotation speed of the decorative arcs (input_4).",
            ),
        }

    # ------------------------------------------------------------------
    # Widget creation / host integration
    # ------------------------------------------------------------------
    def create_widget(self, parent: Optional[QWidget] = None) -> QWidget:
        """
        Create the Qt widget used for both preview and off-screen rendering.
        """
        self._widget = _StemOrbitsWidget(self.config, parent=parent)
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
