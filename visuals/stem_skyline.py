"""
Stem Skyline visualization plugin for Olaf.

This plugin renders an audio-reactive city skyline. Each building's height
and lights respond to several audio stems:

- input_1: typically "drums"  -> fast flicker and sharp movement
- input_2: typically "bass"   -> large, heavy buildings
- input_3: typically "vocals" -> sky glow and accent lights
- input_4: typically "other"  -> secondary details

It is fully vector-based via QPainter, so it scales nicely to high
resolutions (1080p, 1440p, 4K) when used by the export tab.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt, QTimer, QPointF, QSize
from PyQt6.QtGui import QPainter, QColor, QLinearGradient
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


def _mix_color(a: QColor, b: QColor, t: float) -> QColor:
    """Linear blend between two QColor objects."""
    t = _clamp(t, 0.0, 1.0)
    r = int(_lerp(a.red(), b.red(), t))
    g = int(_lerp(a.green(), b.green(), t))
    b_ = int(_lerp(a.blue(), b.blue(), t))
    return QColor(r, g, b_)


# ---------------------------------------------------------------------------
# Preview / export widget
# ---------------------------------------------------------------------------


class _StemSkylineWidget(QWidget):
    """
    Preview widget for the Stem Skyline visualization.

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
        self._level_1: float = 0.0  # typically drums
        self._level_2: float = 0.0  # typically bass
        self._level_3: float = 0.0  # typically vocals
        self._level_4: float = 0.0  # other

        # Simple repaint timer for interactive preview; for off-screen export
        # the host drives time and calls update() as needed.
        self._timer = QTimer(self)
        self._timer.setInterval(40)  # ~25 FPS by default
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

        self.setMinimumSize(320, 180)

        # Base colors for the skyline; blended according to height + energy.
        self._color_building_low = QColor(10, 25, 60)
        self._color_building_high = QColor(120, 200, 255)

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
        # Compute number of buildings (scaled with resolution).
        # ------------------------------------------------------------------
        base_buildings = int(cfg.get("num_buildings", 48))
        base_buildings = max(8, min(256, base_buildings))

        if width >= 1280 and height >= 720:
            scale_w = width / 640.0
            scale_h = height / 360.0
            upscale = max(scale_w, scale_h)
            upscale = _clamp(upscale, 1.0, 3.0)
            num_buildings = int(base_buildings * upscale)
            num_buildings = max(16, min(512, num_buildings))
        else:
            num_buildings = base_buildings

        # ------------------------------------------------------------------
        # Global energy metrics:
        #   - bass  -> large masses
        #   - drums -> flicker / jitter
        #   - vocals / other -> sky / accents
        # ------------------------------------------------------------------
        level_drums = self._level_1
        level_bass = self._level_2
        level_vocals = self._level_3
        level_other = self._level_4

        global_energy = _clamp(
            0.4 * level_bass + 0.3 * level_drums + 0.2 * level_vocals + 0.1 * level_other,
            0.0,
            1.0,
        )

        # ------------------------------------------------------------------
        # Background sky gradient
        # ------------------------------------------------------------------
        bg_bottom = QColor(5, 5, 20)
        sky_base = QColor(20, 40, 90)
        sky_peak = QColor(120, 60, 160)
        vocals_mix = _smoothstep(level_vocals)
        sky_top = _mix_color(sky_base, sky_peak, vocals_mix)

        energy_factor = 0.2 + 0.8 * global_energy
        sky_top = QColor(
            int(sky_top.red() * energy_factor),
            int(sky_top.green() * energy_factor),
            int(sky_top.blue() * energy_factor),
        )

        grad = QLinearGradient(0, 0, 0, height)
        grad.setColorAt(0.0, sky_top)
        grad.setColorAt(1.0, bg_bottom)
        painter.fillRect(self.rect(), grad)

        horizon_y = int(height * 0.6)
        horizon_thickness = max(2, int(height * 0.01))
        glow_color = QColor(220, 180, 120, int(60 + 150 * global_energy))
        painter.fillRect(0, horizon_y, width, horizon_thickness, glow_color)

        # ------------------------------------------------------------------
        # Skyline parameters with user control
        # ------------------------------------------------------------------
        legacy_max = float(cfg.get("skyline_height_factor", 0.55))

        max_height_factor = float(cfg.get("skyline_max_height_factor", legacy_max))
        max_height_factor = _clamp(max_height_factor, 0.1, 1.2)

        min_height_rel = float(cfg.get("skyline_min_height_factor", 0.25))
        min_height_rel = _clamp(min_height_rel, 0.0, 0.9)

        ground_level_factor = float(cfg.get("ground_level_factor", 0.88))
        ground_level_factor = _clamp(ground_level_factor, 0.5, 0.98)

        center_emphasis = float(cfg.get("center_emphasis", 0.35))
        center_emphasis = _clamp(center_emphasis, 0.0, 1.0)

        bass_gain = float(cfg.get("bass_gain", 1.8))
        bass_gain = _clamp(bass_gain, 0.0, 4.0)

        drums_gain = float(cfg.get("drums_gain", 1.4))
        drums_gain = _clamp(drums_gain, 0.0, 4.0)

        vocals_glow_gain = float(cfg.get("vocals_glow_gain", 1.0))
        vocals_glow_gain = _clamp(vocals_glow_gain, 0.0, 3.0)

        sway_amount = float(cfg.get("camera_sway_amount", 0.05))
        sway_amount = _clamp(sway_amount, 0.0, 0.5)

        window_density = float(cfg.get("window_density", 1.0))
        window_density = _clamp(window_density, 0.3, 2.0)

        gap_factor = _clamp(float(cfg.get("gap_factor", 0.15)), 0.0, 0.6)

        sway_x = sway_amount * math.sin(t * 0.3) * (0.3 + 0.7 * global_energy)
        sway_y = sway_amount * math.cos(t * 0.25) * (0.3 + 0.7 * global_energy)

        building_slot = width / float(num_buildings)
        gap_px = gap_factor * building_slot

        max_skyline_h = max_height_factor * height
        max_skyline_h = max(10.0, max_skyline_h)

        min_skyline_h = max_skyline_h * min_height_rel
        min_skyline_h = max(5.0, min_skyline_h)

        ground_y = height * (ground_level_factor + sway_y * 0.2)

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        flicker_phase = t * 15.0

        for i in range(num_buildings):
            u = (i + 0.5) / float(num_buildings)

            rnd_seed = (i * 13) & 0xFFFF
            rng = random.Random(rnd_seed)

            bass_component = bass_gain * level_bass
            bass_component *= (0.4 + 0.6 * abs(math.sin(u * 4.3 + t * 0.8)))

            drums_component = drums_gain * level_drums
            drums_component *= (0.3 + 0.7 * abs(math.sin(u * 11.1 + flicker_phase)))

            other_component = 0.6 * level_other * (0.5 + 0.5 * math.sin(u * 6.7 - t))

            h_factor = bass_component + drums_component + other_component
            h_factor = _clamp(h_factor, 0.0, 2.0)
            h_factor = _smoothstep(h_factor * 0.5)

            if center_emphasis > 0.0:
                pos_from_center = 1.0 - abs(2.0 * (u - 0.5))
                center_boost = 1.0 + center_emphasis * pos_from_center
                h_factor *= center_boost
                h_factor = _clamp(h_factor, 0.0, 2.0)

            building_h = _lerp(min_skyline_h, max_skyline_h, h_factor)
            building_h *= _lerp(0.8, 1.2, rng.random())

            x_center = (i + 0.5 + sway_x) * building_slot
            w_building = max(4.0, building_slot - gap_px)
            x0 = x_center - w_building * 0.5
            x1 = x_center + w_building * 0.5

            y0 = ground_y - building_h
            y1 = ground_y

            if x1 < 0 or x0 > width:
                continue

            height_norm = (building_h - min_skyline_h) / max(
                1.0, (max_skyline_h - min_skyline_h)
            )
            energy_norm = _smoothstep(global_energy)
            color_mix = _clamp(0.3 * height_norm + 0.7 * energy_norm, 0.0, 1.0)

            base_color = _mix_color(self._color_building_low, self._color_building_high, color_mix)
            tint = rng.uniform(-0.15, 0.15)
            base_color = QColor(
                int(_clamp(base_color.red() * (1.0 + tint), 0, 255)),
                int(_clamp(base_color.green() * (1.0 + tint * 0.5), 0, 255)),
                int(_clamp(base_color.blue() * (1.0 - tint * 0.3), 0, 255)),
            )

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(base_color)
            painter.drawRect(int(x0), int(y0), int(w_building), int(y1 - y0))

            # ---------------- Windows / light strips ----------------
            base_rows = building_h / max(16.0, height * 0.04)
            base_cols = w_building / max(10.0, width * 0.03)

            window_rows = int(max(3, base_rows * window_density))
            window_cols = int(max(2, base_cols * window_density))

            window_intensity = _clamp(
                0.3 + 0.7 * (level_drums * 0.7 + level_vocals * 0.5),
                0.0,
                1.0,
            )
            window_intensity *= _clamp(height_norm * 1.2, 0.0, 1.0)

            if window_intensity > 0.01 and window_rows > 0 and window_cols > 0:
                window_color_on = QColor(255, 220, 160)
                window_color_off = QColor(20, 20, 30)

                for r in range(window_rows):
                    row_t = (r + 0.5) / float(window_rows)
                    y_row = y0 + row_t * (y1 - y0)

                    for c in range(window_cols):
                        col_t = (c + 0.5) / float(window_cols)
                        x_col = x0 + col_t * (w_building)

                        rnd_val = rng.random()
                        drums_pulse = abs(
                            math.sin(flicker_phase + u * 5.0 + r * 0.7 + c * 1.3)
                        )
                        vocals_pulse = abs(math.sin(t * 1.2 + u * 2.5))

                        on_prob = (
                            0.15
                            + 0.5 * window_intensity * drums_pulse
                            + 0.35 * level_vocals * vocals_glow_gain * vocals_pulse
                        )
                        on_prob = _clamp(on_prob, 0.0, 1.0)

                        if rnd_val < on_prob:
                            col = window_color_on
                        else:
                            col = window_color_off

                        w_win = max(2.0, w_building / (window_cols * 1.4))
                        h_win = max(2.0, (y1 - y0) / (window_rows * 1.6))
                        rect_x = x_col - w_win * 0.5
                        rect_y = y_row - h_win * 0.5

                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.setBrush(col)
                        painter.drawRect(int(rect_x), int(rect_y), int(w_win), int(h_win))

        # ------------------------------------------------------------------
        # Foreground haze / subtle ground reflection, driven by bass
        # ------------------------------------------------------------------
        haze_strength = float(cfg.get("foreground_haze_strength", 0.6))
        haze_strength = _clamp(haze_strength, 0.0, 2.0)

        haze_alpha = int(60 * haze_strength * (0.3 + 0.7 * level_bass))
        if haze_alpha > 0:
            haze_color = QColor(10, 10, 20, haze_alpha)
            painter.fillRect(0, int(ground_y), width, height - int(ground_y), haze_color)

        painter.end()


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------


class StemSkylineVisualization(BaseVisualization):
    """
    Audio-reactive city skyline visualization.

    - Buildings heights are driven mainly by bass and drums.
    - Windows flicker with drums and vocals.
    - Sky glow is influenced by vocals and the overall mix.
    """

    plugin_id = "dr_dlp_stem_skyline"
    plugin_name = "Stem Skyline"
    plugin_description = (
        "Audio-reactive skyline made of buildings driven by multiple stems.\n"
        "\n"
        "Recommended stem routing:\n"
        " \n - input_1: drums  -> fast flicker and windows activity\n"
        " \n - input_2: bass   -> large, heavy building heights\n"
        " \n - input_3: vocals -> sky glow and extra window brightness\n"
        " \n - input_4: other  -> secondary animation details\n"
        "\n"
        "You can still route any stem to any input, but this mapping usually "
        "gives the most readable and musical skyline."
    )
    plugin_author = "Dr DLP"
    plugin_version = "1.1.0"
    plugin_max_inputs = 4

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._widget: Optional[_StemSkylineWidget] = None

        # Default configuration values (also used by the UI as defaults).
        defaults = {
            "num_buildings": 48,
            "skyline_max_height_factor": 0.55,
            "skyline_min_height_factor": 0.25,
            "ground_level_factor": 0.88,
            "center_emphasis": 0.35,
            "bass_gain": 1.8,
            "drums_gain": 1.4,
            "vocals_glow_gain": 1.0,
            "camera_sway_amount": 0.05,
            "gap_factor": 0.15,
            "window_density": 1.0,
            "foreground_haze_strength": 0.6,
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

        All names must be stable and JSON-serializable.
        """
        return {
            "num_buildings": PluginParameter(
                name="num_buildings",
                label="Number of buildings",
                type="int",
                default=48,
                minimum=8,
                maximum=256,
                step=4,
                description="Base number of skyline buildings (scaled up at high resolution).",
            ),
            "skyline_max_height_factor": PluginParameter(
                name="skyline_max_height_factor",
                label="Max skyline height factor",
                type="float",
                default=0.55,
                minimum=0.1,
                maximum=1.2,
                step=0.02,
                description="Maximum height of buildings relative to frame height.",
            ),
            "skyline_min_height_factor": PluginParameter(
                name="skyline_min_height_factor",
                label="Min skyline height factor",
                type="float",
                default=0.25,
                minimum=0.0,
                maximum=0.9,
                step=0.02,
                description="Minimum height of buildings as a fraction of max height.",
            ),
            "ground_level_factor": PluginParameter(
                name="ground_level_factor",
                label="Ground level (vertical position)",
                type="float",
                default=0.88,
                minimum=0.5,
                maximum=0.98,
                step=0.01,
                description="Vertical position of the skyline base (0=top, 1=bottom).",
            ),
            "center_emphasis": PluginParameter(
                name="center_emphasis",
                label="Center emphasis",
                type="float",
                default=0.35,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                description="How much buildings near the center are taller than those at the edges.",
            ),
            "bass_gain": PluginParameter(
                name="bass_gain",
                label="Bass gain",
                type="float",
                default=1.8,
                minimum=0.0,
                maximum=4.0,
                step=0.1,
                description="How strongly bass (input_2) inflates building heights.",
            ),
            "drums_gain": PluginParameter(
                name="drums_gain",
                label="Drums gain",
                type="float",
                default=1.4,
                minimum=0.0,
                maximum=4.0,
                step=0.1,
                description="How strongly drums (input_1) add spikes and flicker.",
            ),
            "vocals_glow_gain": PluginParameter(
                name="vocals_glow_gain",
                label="Vocals glow gain",
                type="float",
                default=1.0,
                minimum=0.0,
                maximum=3.0,
                step=0.1,
                description="How much vocals (input_3) brighten the sky glow and windows.",
            ),
            "camera_sway_amount": PluginParameter(
                name="camera_sway_amount",
                label="Camera sway amount",
                type="float",
                default=0.05,
                minimum=0.0,
                maximum=0.5,
                step=0.01,
                description="How much the skyline gently sways in X/Y with the music.",
            ),
            "gap_factor": PluginParameter(
                name="gap_factor",
                label="Gap between buildings",
                type="float",
                default=0.15,
                minimum=0.0,
                maximum=0.6,
                step=0.02,
                description="Fraction of each building slot left as horizontal gap.",
            ),
            "window_density": PluginParameter(
                name="window_density",
                label="Window density",
                type="float",
                default=1.0,
                minimum=0.3,
                maximum=2.0,
                step=0.05,
                description="Multiplier for the number of window rows and columns.",
            ),
            "foreground_haze_strength": PluginParameter(
                name="foreground_haze_strength",
                label="Foreground haze strength",
                type="float",
                default=0.6,
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                description="Opacity of the haze at the bottom of the skyline.",
            ),
        }

    # ------------------------------------------------------------------
    # Widget creation / host integration
    # ------------------------------------------------------------------
    def create_widget(self, parent: Optional[QWidget] = None) -> QWidget:
        """
        Create the Qt widget used for both preview and off-screen rendering.
        """
        self._widget = _StemSkylineWidget(self.config, parent=parent)
        return self._widget

    def apply_preview_settings(self, width: int, height: int, fps: int) -> None:
        """
        Let the plugin know about the intended preview size and FPS.

        We use this to roughly match the animation refresh rate to the
        nominal FPS, which keeps motion smoother when the preview is
        driven by an internal timer.
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
                    "input_1": {"rms": ...},
                    "input_2": {"rms": ...},
                    ...
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

        level_1 = float(input_1.get("rms", 0.0))  # drums
        level_2 = float(input_2.get("rms", 0.0))  # bass
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
