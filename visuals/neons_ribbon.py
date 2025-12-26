"""
Neon ribbons tunnel visualization plugin for Olaf.

This plugin is a real-time, audio-reactive adaptation of a standalone
neon ribbons script. It renders a neon tunnel made of depth-stacked
ribbons and reacts to the RMS level of up to two audio inputs.

Requirements (optional):
    - numpy (required)
    - skia-python (strongly recommended)
    - opencv-python (optional, only for glow effect)

If skia is not available, the plugin will still load but the preview
widget will only display a warning message instead of the effect.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import math
import numpy as np

# Optional dependencies
try:
    import skia  # type: ignore
    HAVE_SKIA = True
except Exception:
    skia = None  # type: ignore
    HAVE_SKIA = False

try:
    import cv2  # type: ignore
    HAVE_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    HAVE_CV2 = False

from PyQt6.QtCore import Qt, QTimer, QUrl, QSettings, QSize
from PyQt6.QtGui import QImage, QPainter, QColor
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout

# Try both import paths so the plugin can work whether the app is installed
# as a package (olaf_app) or run from a flat layout during development.
try:
    from olaf_app.visualization_api import BaseVisualization, PluginParameter
except ImportError:
    from visualization_api import BaseVisualization, PluginParameter


# ---------------------------------------------------------------------------
# 3D PROJECTION / CORE MATH
# ---------------------------------------------------------------------------

def project_3d_point(
    x: float,
    y: float,
    z: float,
    width: int,
    height: int,
) -> tuple[float, float, float]:
    """
    Simple 3D perspective projection.
    """
    fov = 300.0
    if z <= 0.1:
        z = 0.1

    scale = fov / z
    x_proj = (x * scale) + (width / 2.0)
    y_proj = (y * scale) + (height / 2.0)

    return x_proj, y_proj, scale


def make_neon_ribbons_frame(
    t: float,
    energy: float,
    width: int,
    height: int,
    config: Dict[str, Any],
) -> np.ndarray:
    """
    Generate a single RGB frame of the neon ribbons effect at time t.

    If skia is not available, this function returns a simple fallback
    gradient frame so that the plugin still works.
    """
    # If Skia is not available, return a simple gradient as a fallback
    if not HAVE_SKIA:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            v = int(255 * y / max(1, height - 1))
            img[y, :, :] = (v // 2, v, 255 - v // 2)
        return img

    # Configuration with safe defaults
    num_lines = int(config.get("num_lines", 200))
    points_per_line = int(config.get("points_per_line", 100))
    grid_depth = float(config.get("grid_depth", 30.0))
    speed = float(config.get("speed", 4.0))
    glow_sigma = float(config.get("glow_sigma", 15.0))
    glow_strength = float(config.get("glow_strength", 0.8))
    energy_boost = float(config.get("energy_boost", 1.5))

    num_lines = max(10, min(num_lines, 400))
    points_per_line = max(16, min(points_per_line, 400))
    grid_depth = max(5.0, min(grid_depth, 80.0))
    speed = max(0.1, min(speed, 20.0))
    glow_sigma = max(0.0, min(glow_sigma, 40.0))
    glow_strength = max(0.0, min(glow_strength, 3.0))
    energy_boost = max(0.0, min(energy_boost, 4.0))

    # Colors (BGR for later OpenCV, but Skia wants RGBA)
    color_far = np.array([255, 0, 128], dtype=np.float32)   # BGR
    color_near = np.array([0, 255, 255], dtype=np.float32)  # BGR

    # Create a Skia surface and clear to black
    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()
    canvas.clear(skia.ColorBLACK)

    # Audio energy
    energy = float(energy)
    energy = max(0.0, min(1.0, energy))
    energy = energy * energy_boost + 0.1

    # Depth-stacked ribbons, far to near
    for i in range(num_lines, 0, -1):
        z_norm = i / float(num_lines)
        z_world = 1.0 + (z_norm * grid_depth)

        col_mix = color_near * (1.0 - z_norm) + color_far * z_norm
        alpha = int(255 * (1.0 - (z_norm * 0.6)))
        color = skia.Color(
            int(col_mix[2]),  # R
            int(col_mix[1]),  # G
            int(col_mix[0]),  # B
            alpha,
        )

        path = skia.Path()
        first_point = True

        for j in range(points_per_line):
            x_norm = (j / (points_per_line - 1)) - 0.5
            x_world = x_norm * (grid_depth * 1.5)

            offset_x = x_world * 2.0
            offset_z = z_world * 0.5
            phase = t * speed

            y_base = math.sin(offset_x + phase + offset_z)
            y_detail = math.sin(offset_x * 3.0 - phase) * 0.3

            fade_edges = 1.0 - (2.0 * abs(x_norm)) ** 2

            amplitude = energy * 1.2 * fade_edges
            y_world = (y_base + y_detail) * amplitude * 0.8
            y_world += 0.5

            px, py, _ = project_3d_point(x_world, y_world, z_world, width, height)

            if first_point:
                path.moveTo(px, py)
                first_point = False
            else:
                path.lineTo(px, py)

        paint = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style)
        paint.setStrokeWidth(4.0 * (2.0 / z_world))
        paint.setColor(color)
        canvas.drawPath(path, paint)

    # Convert Skia surface to RGBA bytes -> numpy
    img_array = np.frombuffer(surface.makeImageSnapshot().tobytes(), dtype=np.uint8)
    img_array = img_array.reshape((height, width, 4))  # RGBA

    # Convert RGBA -> BGR
    img_bgr = img_array[:, :, [2, 1, 0]]

    # Optional glow with OpenCV
    if HAVE_CV2 and glow_sigma > 0.0 and glow_strength > 0.0:
        glow = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=glow_sigma, sigmaY=glow_sigma)
        final_bgr = cv2.addWeighted(img_bgr, 1.0, glow, glow_strength, 0.0)
    else:
        final_bgr = img_bgr

    # Return RGB for Qt
    return final_bgr[:, :, ::-1]


# ---------------------------------------------------------------------------
# PREVIEW WIDGET
# ---------------------------------------------------------------------------

class _NeonRibbonsWidget(QWidget):
    """
    Lightweight preview widget for the neon ribbons visualization.
    """
    def __init__(self, config: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._time_s: float = 0.0
        self._energy: float = 0.0

        if not HAVE_SKIA:
            # If Skia is missing, show a simple warning label
            layout = QVBoxLayout(self)
            label = QLabel(
                "Skia is not available.\n"
                "Neon Ribbons preview is disabled.\n"
                "Please install 'skia-python' to enable this visualization.",
                self,
            )
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(640, 360)

    def update_audio_state(self, time_ms: int, energy: float) -> None:
        self._time_s = time_ms / 1000.0
        self._energy = float(energy)
        self.update()

    def set_timer_interval(self, interval_ms: int) -> None:
        """
        Compatibility stub for the host preview API.

        The host (VisualizationsTab / ExportTab) drives animation timing
        via on_audio_features(time_ms, ...), so we do not need an internal
        QTimer here. This method exists only to satisfy the expected API.
        """
        _ = interval_ms  # no-op

    def paintEvent(self, event) -> None:  # type: ignore[override]
        widget_w = self.width()
        widget_h = self.height()
        if widget_w <= 0 or widget_h <= 0:
            return

        # ------------------------------------------------------------------
        # Internal render resolution
        #
        # - For "small" widgets (UI preview), keep a reduced resolution so
        #   the effect stays real-time.
        # - For "large" widgets (off-screen export, typically >= 1280x720),
        #   render 1:1 at the widget size so the final video is truly
        #   high-res and not just an upscaled 480p buffer.
        # ------------------------------------------------------------------
        if widget_w >= 1280 and widget_h >= 720:
            # High-quality / export mode: full-res render
            render_w = widget_w
            render_h = widget_h
        else:
            # Interactive preview: lower internal resolution
            max_render_width = 480
            max_render_height = 270

            # Preserve aspect ratio of the widget
            aspect = widget_w / float(widget_h)
            render_w = min(widget_w, max_render_width)
            render_h = int(render_w / aspect)

            if render_h > max_render_height:
                render_h = max_render_height
                render_w = int(render_h * aspect)

            render_w = max(160, render_w)
            render_h = max(90, render_h)

        frame_rgb = make_neon_ribbons_frame(
            t=self._time_s,
            energy=self._energy,
            width=render_w,
            height=render_h,
            config=self._config,
        )

        h, w, _ = frame_rgb.shape
        bytes_per_line = 3 * w
        qimg = QImage(
            frame_rgb.tobytes(),
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(self.rect(), qimg)


# ---------------------------------------------------------------------------
# PLUGIN CLASS
# ---------------------------------------------------------------------------

class NeonRibbonsVisualization(BaseVisualization):
    """
    Neon ribbons tunnel visualization plugin.

    This plugin supports up to two audio inputs (stems).
    """

    plugin_id = "dr_dlp_neon_ribbons"
    plugin_name = "Neon Ribbons Tunnel"
    plugin_description = (
        "Depth-stacked neon ribbons forming a 3D tunnel, reacting to the "
        "energy of the audio stem."
    )
    plugin_author = "DrDLP"
    plugin_version = "1.0.1"
    plugin_max_inputs = 1

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._widget: Optional[_NeonRibbonsWidget] = None

        defaults = {
            "speed": 4.0,
            "grid_depth": 30.0,
            "num_lines": 120,
            "points_per_line": 80,
            "glow_sigma": 15.0,
            "glow_strength": 0.8,
            "energy_boost": 1.5,
        }
        for key, value in defaults.items():
            self.config.setdefault(key, value)

    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        return {
            "speed": PluginParameter(
                name="speed",
                label="Wave speed",
                type="float",
                default=4.0,
                minimum=0.5,
                maximum=20.0,
                step=0.5,                
                description="Controls how fast the tunnel waves move forward."
            ),
            "grid_depth": PluginParameter(
                name="grid_depth",
                label="Tunnel depth",
                type="float",
                default=30.0,
                minimum=5.0,
                maximum=80.0,
                step=1,                
                description="Controls how deep the tunnel extends into the distance."
            ),
            "num_lines": PluginParameter(
                name="num_lines",
                label="Number of ribbons",
                type="int",
                default=200,
                minimum=20,
                maximum=400,
                step=2,                
                description="Number of depth layers (ribbons) in the tunnel."
            ),
            "points_per_line": PluginParameter(
                name="points_per_line",
                label="Horizontal resolution",
                type="int",
                default=100,
                minimum=32,
                maximum=400,
                step=2,                
                description="Horizontal resolution of each ribbon polyline."
            ),
            "glow_sigma": PluginParameter(
                name="glow_sigma",
                label="Glow blur radius",
                type="float",
                default=15.0,
                minimum=0.0,
                maximum=40.0,
                step=1,               
                description="Gaussian blur radius used for the neon glow."
            ),
            "glow_strength": PluginParameter(
                name="glow_strength",
                label="Glow intensity",
                type="float",
                default=0.8,
                minimum=0.0,
                maximum=3.0,
                step=0.2,               
                description="Amount of glow mixed into the final image."
            ),
            "energy_boost": PluginParameter(
                name="energy_boost",
                label="Energy boost",
                type="float",
                default=1.5,
                minimum=0.0,
                maximum=4.0,
                step=0.2,                
                description="Scales the incoming audio energy before applying it."
            ),
        }

    def create_widget(self, parent: Optional[QWidget] = None) -> QWidget:
        self._widget = _NeonRibbonsWidget(self.config, parent)
        return self._widget
        
    def apply_preview_settings(self, width: int, height: int, fps: int) -> None:
        """
        Adapt the preview widget size and timer to host-provided settings.
        """
        if self._widget is None:
            return

        fps = max(5, min(60, int(fps)))
        interval_ms = int(1000 / fps)

        self._widget.setMinimumSize(width, height)
        self._widget.setMaximumSize(width, height)
        self._widget.set_timer_interval(interval_ms)
        

    def on_audio_features(self, features: Dict[str, Any]) -> None:
        if self._widget is None:
            return

        time_ms = int(features.get("time_ms", 0))
        inputs = features.get("inputs", {}) or {}

        input_1 = inputs.get("input_1", {})
        input_2 = inputs.get("input_2", {})

        level_1 = float(input_1.get("rms", 0.0))
        level_2 = float(input_2.get("rms", 0.0))

        energy = max(level_1, level_2)
        energy = max(0.0, min(1.0, energy))

        self._widget.update_audio_state(time_ms, energy)

    def on_activate(self) -> None:
        pass

    def on_deactivate(self) -> None:
        pass
