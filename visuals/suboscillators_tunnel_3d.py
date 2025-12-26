"""
Suboscillators Oscilloscope Tunnel (3D) for Olaf
------------------------------------------------
Concept
- A depth-stacked "oscilloscope" tunnel: each layer is a closed ring whose radius is
  modulated by a stack of sub-oscillators (frequency ratios + detune), producing
  interference / moiré-like patterns.
- Pure CPU rendering using Skia (no OpenGL / VisPy), so the Qt preview/export capture
  path is the same as the existing `neons_ribbon.py` plugin.

Color presets
- Includes the same preset list style as `sdf_neon_tunnel.py` (enum parameter shown in UI).

Audio mapping (single stem)
- input_1 (MAIN / full mix): drives waveform amplitude, forward speed and glow intensity.

Author: Olaf community plugin (based on Dr DLP's Skia tunnel patterns)
License: same as project
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

try:
    # When Olaf is installed as a package
    from olaf_app.visualization_api import BaseVisualization, PluginParameter
except Exception:
    # When running from source / single-file testing
    from visualization_api import BaseVisualization, PluginParameter

try:
    import numpy as np  # type: ignore

    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False

try:
    import skia  # type: ignore

    HAVE_SKIA = True
except Exception:
    HAVE_SKIA = False

try:
    import cv2  # type: ignore

    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False


# ---------------------------------------------------------------------------
# Color presets (same list style as sdf_neon_tunnel.py)
# ---------------------------------------------------------------------------

_COLOR_PRESETS = [
    "Cyberpunk (Cyan/Magenta)",
    "Synthwave (Pink/Orange)",
    "Matrix (Green)",
    "Electric (Blue/Purple)",
    "Acid (Yellow/Cyan)",
    "Infrared (Red/Pink)",
    "Rainbow",
]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _clamp01(x: float) -> float:
    return _clamp(float(x), 0.0, 1.0)


def _energy_map(x: float, gamma: float) -> float:
    """Nonlinear mapping to make small RMS changes more visible."""
    return float(_clamp01(x) ** max(0.15, float(gamma)))


def project_3d_point(x: float, y: float, z: float, width: int, height: int) -> Tuple[float, float, float]:
    """
    Simple 3D perspective projection.
    Matches the pattern used by `neons_ribbon.py`.
    """
    fov = 300.0
    if z <= 0.1:
        z = 0.1

    scale = fov / z
    x_proj = (x * scale) + (width / 2.0)
    y_proj = (y * scale) + (height / 2.0)
    return x_proj, y_proj, scale


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """h,s,v in [0..1] -> RGB 0..255"""
    h = h % 1.0
    s = _clamp01(s)
    v = _clamp01(v)

    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6

    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q

    return int(r * 255.0), int(g * 255.0), int(b * 255.0)


def _palette_rgb(preset: str, k: float) -> Tuple[int, int, int]:
    """
    Map a scalar k to an RGB color based on a named preset.

    k: typically increases with depth / angle to get gradients.
    """
    k = k % 1.0

    # A small helper to do 2-color mixes with some variation.
    def mix_rgb(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
        t = _clamp01(t)
        return (
            int(a[0] + (b[0] - a[0]) * t),
            int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t),
        )

    # Phase for palette motion (kept deterministic from k).
    s1 = 0.5 + 0.5 * math.sin(2.0 * math.pi * (k + 0.00))
    s2 = 0.5 + 0.5 * math.sin(2.0 * math.pi * (k + 0.33))
    s3 = 0.5 + 0.5 * math.sin(2.0 * math.pi * (k + 0.66))

    if preset == _COLOR_PRESETS[0]:
        # Cyberpunk (Cyan/Magenta)
        c = mix_rgb((0, 255, 255), (255, 0, 255), s1)
        c = mix_rgb(c, (40, 130, 255), 0.35 * s2)
        return c

    if preset == _COLOR_PRESETS[1]:
        # Synthwave (Pink/Orange)
        c = mix_rgb((255, 30, 200), (255, 140, 30), s1)
        c = mix_rgb(c, (80, 240, 255), 0.25 * s3)
        return c

    if preset == _COLOR_PRESETS[2]:
        # Matrix (Green)
        return mix_rgb((10, 60, 20), (30, 255, 80), s1)

    if preset == _COLOR_PRESETS[3]:
        # Electric (Blue/Purple)
        c = mix_rgb((30, 90, 255), (200, 40, 255), s1)
        c = mix_rgb(c, (0, 255, 220), 0.20 * s2)
        return c

    if preset == _COLOR_PRESETS[4]:
        # Acid (Yellow/Cyan)
        return mix_rgb((255, 240, 40), (0, 255, 240), s1)

    if preset == _COLOR_PRESETS[5]:
        # Infrared (Red/Pink)
        return mix_rgb((255, 20, 50), (255, 50, 230), s1)

    # Rainbow (default)
    return _hsv_to_rgb(k, 0.85, 1.0)


def _skia_color(preset: str, k: float, alpha: float) -> int:
    r, g, b = _palette_rgb(preset, k)
    a = int(_clamp01(alpha) * 255.0)
    return skia.ColorSetARGB(a, r, g, b)


def _suboscillator_wave(theta: float, t: float, z: float, *, base_freq: float, sub_count: int, sub_ratio: float,
                        sub_detune: float, sub_gain: float) -> float:
    """
    Stack of sinusoids (carrier + sub-oscillators).
    theta in radians around the ring.
    z: depth slice for subtle de-correlation.
    """
    w = 0.0
    # Carrier
    w += math.sin(base_freq * theta + 0.35 * t + 0.15 * z)

    # Sub-oscillators (geometric frequency scaling)
    # Using a deterministic phase offset per index.
    for i in range(max(0, int(sub_count))):
        f = base_freq * (float(sub_ratio) ** (i + 1))
        f = f * (1.0 + float(sub_detune) * 0.015 * math.sin(0.7 * t + 1.9 * i))
        gain = float(sub_gain) * (0.85 ** i)
        phase = (0.9 + 0.31 * i) * t + 0.11 * z * (i + 1)
        w += gain * math.sin(f * theta - phase)

    # Normalize roughly (not exact, but keeps ranges predictable)
    return w / (1.0 + max(0, int(sub_count)) * max(0.2, float(sub_gain)))


def make_subosc_oscilloscope_tunnel_frame(
    *,
    t: float,
    energy: float,
    width: int,
    height: int,
    config: Dict[str, Any],
) -> "np.ndarray":
    """
    Render one RGB frame (numpy array HxWx3).
    """

    if not HAVE_NUMPY or not HAVE_SKIA:
        # Host will show a missing dependency label in the widget.
        raise RuntimeError("Missing numpy or skia")

    # Parameters (defaults kept in plugin __init__)
    speed = float(config.get("speed", 2.5))
    grid_depth = float(config.get("grid_depth", 40.0))
    num_rings = int(config.get("num_rings", 220))
    points_per_ring = int(config.get("points_per_ring", 220))

    base_radius = float(config.get("base_radius", 2.4))
    radius_gain = float(config.get("radius_gain", 2.1))
    waveform_amp = float(config.get("waveform_amp", 1.0))

    twist = float(config.get("twist", 1.4))
    spiral = float(config.get("spiral", 0.35))
    z_spacing = float(config.get("z_spacing", 0.22))

    line_width = float(config.get("line_width", 2.4))
    glow_passes = int(config.get("glow_passes", 3))
    glow_width_mult = float(config.get("glow_width_mult", 2.0))

    energy_boost = float(config.get("energy_boost", 1.6))
    reactivity_gamma = float(config.get("reactivity_gamma", 0.55))
    fog_strength = float(config.get("fog_strength", 1.15))

    base_freq = float(config.get("base_freq", 2.8))
    sub_count = int(config.get("sub_count", 5))
    sub_ratio = float(config.get("sub_ratio", 1.62))
    sub_detune = float(config.get("sub_detune", 0.55))
    sub_gain = float(config.get("sub_gain", 0.55))

    color_preset = str(config.get("color_preset", _COLOR_PRESETS[0]))
    hue_shift = float(config.get("hue_shift", 0.0))  # applied only to Rainbow preset

    # Energy mapping
    e = _energy_map(float(energy) * energy_boost, reactivity_gamma)

    # Render target
    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()
    canvas.clear(skia.ColorBLACK)

    # A little global "camera wobble" that reacts to audio
    cam_wobble = (0.25 + 1.1 * e) * math.sin(0.65 * t)

    # Draw from far -> near for better overlap
    for i in range(num_rings, 0, -1):
        # z grows towards the viewer: keep z positive
        z_world = 2.0 + i * z_spacing
        if z_world > grid_depth:
            # Keep a soft cap
            z_world = 2.0 + (i % int(max(1.0, grid_depth / z_spacing))) * z_spacing

        # Fog: far rings are dimmer and thinner
        fog = math.exp(-fog_strength * (z_world / max(1.0, grid_depth)))
        fog = _clamp01(fog)

        # Forward motion (scroll): phase depends on time and depth
        phase_t = t * speed
        theta_twist = twist * (0.08 * z_world + 0.25 * phase_t)

        # Base radius increases with depth a bit => tunnel feel
        r0 = base_radius + radius_gain * (1.0 - (z_world / max(1.0, grid_depth)))
        r0 *= (0.92 + 0.18 * math.sin(0.22 * phase_t + 0.10 * z_world))

        # Audio-driven amplitude
        amp = waveform_amp * (0.15 + 1.75 * e) * fog

        # Each ring is a closed polyline in 3D: (x,y) on a perturbed circle.
        path = skia.Path()
        first = True

        for j in range(points_per_ring + 1):
            u = j / float(points_per_ring)
            theta = 2.0 * math.pi * u

            # Oscilloscope "signal" around the ring
            w = _suboscillator_wave(
                theta=theta,
                t=phase_t,
                z=z_world,
                base_freq=base_freq,
                sub_count=sub_count,
                sub_ratio=sub_ratio,
                sub_detune=sub_detune,
                sub_gain=sub_gain,
            )

            # A second, higher-frequency "sizzle" helps the moiré look (still deterministic)
            sizzle = 0.22 * math.sin((base_freq * 7.0) * theta + 1.1 * phase_t + 0.15 * z_world)

            r = r0 + amp * (w + sizzle)

            # Spiral / helix offset so rings don't perfectly align
            theta2 = theta + theta_twist + spiral * z_world

            x = r * math.cos(theta2) + cam_wobble
            y = r * math.sin(theta2) + 0.35 * cam_wobble

            px, py, scale = project_3d_point(x, y, z_world, width, height)

            if first:
                path.moveTo(px, py)
                first = False
            else:
                path.lineTo(px, py)

        # Color gradient along depth (k) + slight time shift
        k = (i / max(1.0, float(num_rings))) + 0.06 * math.sin(0.4 * phase_t)
        k = (k + hue_shift) % 1.0 if color_preset == "Rainbow" else k % 1.0

        # Stroke width scales with perspective (scale is fov/z).
        # We clamp to avoid giant near-camera lines.
        w_px = _clamp(line_width * (scale / 120.0), 0.6, 6.0)

        # Multi-pass glow: draw wider + lower alpha behind the main line.
        # This is a lightweight glow even without OpenCV.
        for p in range(max(0, glow_passes)):
            kk = (p + 1) / max(1, glow_passes)
            a_glow = (0.18 + 0.35 * e) * fog * (1.0 - 0.55 * kk)
            paint = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style)
            paint.setStrokeWidth(w_px * (1.0 + glow_width_mult * kk))
            paint.setColor(_skia_color(color_preset, k + 0.07 * kk, a_glow))
            canvas.drawPath(path, paint)

        # Main line
        paint = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style)
        paint.setStrokeWidth(w_px)
        paint.setColor(_skia_color(color_preset, k, (0.55 + 0.45 * e) * fog))
        canvas.drawPath(path, paint)

    # Convert Skia surface to RGBA -> numpy
    img_rgba = np.frombuffer(surface.makeImageSnapshot().tobytes(), dtype=np.uint8)
    img_rgba = img_rgba.reshape((height, width, 4))  # RGBA

    # Convert RGBA -> BGR (OpenCV uses BGR)
    img_bgr = img_rgba[:, :, [2, 1, 0]]

    # Optional glow with OpenCV (stronger bloom-like look)
    glow_sigma = float(config.get("glow_sigma", 0.0))
    glow_strength = float(config.get("glow_strength", 0.0))
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

class _SubOscOscilloscopeTunnelWidget(QWidget):
    """
    Lightweight preview widget for the suboscillators oscilloscope tunnel visualization.

    Pattern intentionally matches `neons_ribbon.py`:
      - render at lower internal resolution for small previews
      - render 1:1 for export (large widget sizes)
    """

    def __init__(self, config: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._time_s: float = 0.0
        self._energy: float = 0.0

        if not HAVE_SKIA or not HAVE_NUMPY:
            layout = QVBoxLayout(self)
            label = QLabel(
                "Skia or NumPy is not available.\n"
                "SubOscillators Oscilloscope Tunnel preview is disabled.\n"
                "Please install 'skia-python' and 'numpy' to enable this visualization.",
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
        # Host-driven animation; no internal QTimer needed.
        _ = interval_ms

    def paintEvent(self, event) -> None:  # type: ignore[override]
        if not HAVE_SKIA or not HAVE_NUMPY:
            return

        widget_w = self.width()
        widget_h = self.height()
        if widget_w <= 0 or widget_h <= 0:
            return

        # Keep realtime in UI, full-res in export
        if widget_w >= 1280 and widget_h >= 720:
            render_w, render_h = widget_w, widget_h
        else:
            max_render_width = int(self._config.get("preview_max_width", 520))
            max_render_height = int(self._config.get("preview_max_height", 300))

            aspect = widget_w / float(widget_h)
            render_w = min(widget_w, max_render_width)
            render_h = int(render_w / aspect)

            if render_h > max_render_height:
                render_h = max_render_height
                render_w = int(render_h * aspect)

            render_w = max(160, int(render_w))
            render_h = max(90, int(render_h))

        frame_rgb = make_subosc_oscilloscope_tunnel_frame(
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

class SubOscillatorsOscilloscopeTunnelVisualization(BaseVisualization):
    """
    Depth-stacked oscilloscope rings forming a 3D tunnel (Skia CPU renderer).
    """

    plugin_id = "suboscillators_oscilloscope_tunnel_3d"
    plugin_name = "SubOscillators Oscilloscope Tunnel"
    plugin_description = (
        "3D tunnel made of stacked oscilloscope-like rings, driven by sub-oscillator interference. "
        "Pure Skia CPU rendering for reliable preview/export."
    )
    plugin_author = "DrDLP"
    plugin_version = "1.0.0"
    plugin_max_inputs = 1

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._widget: Optional[_SubOscOscilloscopeTunnelWidget] = None

        defaults = {
            # Motion / depth
            "speed": 2.5,
            "grid_depth": 40.0,
            "num_rings": 220,
            "points_per_ring": 220,
            "z_spacing": 0.22,

            # Geometry
            "base_radius": 2.4,
            "radius_gain": 2.1,
            "waveform_amp": 1.0,
            "twist": 1.4,
            "spiral": 0.35,

            # Sub-oscillators
            "base_freq": 2.8,
            "sub_count": 5,
            "sub_ratio": 1.62,
            "sub_detune": 0.55,
            "sub_gain": 0.55,

            # Look
            "color_preset": _COLOR_PRESETS[0],
            "hue_shift": 0.0,
            "line_width": 2.4,
            "glow_passes": 3,
            "glow_width_mult": 2.0,

            # Post (optional OpenCV bloom)
            "glow_sigma": 0.0,
            "glow_strength": 0.0,

            # Reactivity
            "energy_boost": 1.6,
            "reactivity_gamma": 0.55,
            "fog_strength": 1.15,

            # Preview perf caps
            "preview_max_width": 520,
            "preview_max_height": 300,
        }
        for k, v in defaults.items():
            self.config.setdefault(k, v)

    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        return {
            # Depth/motion
            "speed": PluginParameter(
                name="speed",
                label="Forward speed",
                type="float",
                default=2.5,
                minimum=0.2,
                maximum=12.0,
                step=0.1,
                description="Speed of the forward motion through the tunnel.",
            ),
            "grid_depth": PluginParameter(
                name="grid_depth",
                label="Tunnel depth",
                type="float",
                default=40.0,
                minimum=10.0,
                maximum=120.0,
                step=1.0,
                description="How deep the tunnel extends (used for fog and spacing).",
            ),
            "num_rings": PluginParameter(
                name="num_rings",
                label="Number of rings",
                type="int",
                default=220,
                minimum=40,
                maximum=520,
                step=5,
                description="How many stacked oscilloscope rings are drawn per frame.",
            ),
            "points_per_ring": PluginParameter(
                name="points_per_ring",
                label="Ring resolution",
                type="int",
                default=220,
                minimum=64,
                maximum=720,
                step=8,
                description="Polyline resolution for each ring (higher = smoother, heavier).",
            ),
            "z_spacing": PluginParameter(
                name="z_spacing",
                label="Z spacing",
                type="float",
                default=0.22,
                minimum=0.05,
                maximum=1.0,
                step=0.01,
                description="Distance between successive rings along depth.",
            ),

            # Geometry
            "base_radius": PluginParameter(
                name="base_radius",
                label="Base radius",
                type="float",
                default=2.4,
                minimum=0.3,
                maximum=6.0,
                step=0.05,
                description="Base ring radius near the camera.",
            ),
            "radius_gain": PluginParameter(
                name="radius_gain",
                label="Radius gain (depth)",
                type="float",
                default=2.1,
                minimum=0.0,
                maximum=6.0,
                step=0.05,
                description="How much the radius changes across depth (tunnel perspective).",
            ),
            "waveform_amp": PluginParameter(
                name="waveform_amp",
                label="Waveform amplitude",
                type="float",
                default=1.0,
                minimum=0.0,
                maximum=3.0,
                step=0.05,
                description="Amplitude of the oscilloscope modulation (before audio).",
            ),
            "twist": PluginParameter(
                name="twist",
                label="Twist",
                type="float",
                default=1.4,
                minimum=-6.0,
                maximum=6.0,
                step=0.05,
                description="How much the rings rotate as depth increases.",
            ),
            "spiral": PluginParameter(
                name="spiral",
                label="Spiral",
                type="float",
                default=0.35,
                minimum=-2.0,
                maximum=2.0,
                step=0.01,
                description="Additional helix/spiral term to break perfect alignment.",
            ),

            # Sub-oscillators
            "base_freq": PluginParameter(
                name="base_freq",
                label="Base frequency",
                type="float",
                default=2.8,
                minimum=0.5,
                maximum=12.0,
                step=0.1,
                description="Base oscillator frequency around the ring.",
            ),
            "sub_count": PluginParameter(
                name="sub_count",
                label="Sub-oscillators",
                type="int",
                default=5,
                minimum=0,
                maximum=10,
                step=1,
                description="How many sub-oscillators are stacked on top of the carrier.",
            ),
            "sub_ratio": PluginParameter(
                name="sub_ratio",
                label="Sub ratio",
                type="float",
                default=1.62,
                minimum=1.05,
                maximum=3.0,
                step=0.01,
                description="Geometric frequency ratio between successive sub-oscillators.",
            ),
            "sub_detune": PluginParameter(
                name="sub_detune",
                label="Sub detune",
                type="float",
                default=0.55,
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                description="Adds time-varying detune to create richer interference patterns.",
            ),
            "sub_gain": PluginParameter(
                name="sub_gain",
                label="Sub gain",
                type="float",
                default=0.55,
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                description="Gain of the sub-oscillators relative to the carrier.",
            ),

            # Look / palette
            "color_preset": PluginParameter(
                name="color_preset",
                label="Color preset",
                type="enum",
                default=_COLOR_PRESETS[0],
                choices=_COLOR_PRESETS,
                description="Select a neon palette preset (same style as sdf_neon_tunnel).",
            ),
            "hue_shift": PluginParameter(
                name="hue_shift",
                label="Hue shift (Rainbow only)",
                type="float",
                default=0.0,
                minimum=-1.0,
                maximum=1.0,
                step=0.01,
                description="Hue shift applied only when the preset is Rainbow.",
            ),
            "line_width": PluginParameter(
                name="line_width",
                label="Line width",
                type="float",
                default=2.4,
                minimum=0.5,
                maximum=8.0,
                step=0.1,
                description="Base stroke width (scaled by perspective).",
            ),
            "glow_passes": PluginParameter(
                name="glow_passes",
                label="Glow passes",
                type="int",
                default=3,
                minimum=0,
                maximum=8,
                step=1,
                description="Number of cheap glow passes (draw wider lines behind).",
            ),
            "glow_width_mult": PluginParameter(
                name="glow_width_mult",
                label="Glow width multiplier",
                type="float",
                default=2.0,
                minimum=0.5,
                maximum=5.0,
                step=0.1,
                description="How much wider the glow strokes are.",
            ),

            # Optional OpenCV bloom
            "glow_sigma": PluginParameter(
                name="glow_sigma",
                label="Bloom sigma (OpenCV)",
                type="float",
                default=0.0,
                minimum=0.0,
                maximum=30.0,
                step=0.5,
                description="Gaussian blur sigma for bloom (requires opencv-python).",
            ),
            "glow_strength": PluginParameter(
                name="glow_strength",
                label="Bloom strength (OpenCV)",
                type="float",
                default=0.0,
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                description="Bloom intensity added on top of the base render.",
            ),

            # Reactivity / fog
            "energy_boost": PluginParameter(
                name="energy_boost",
                label="Energy boost",
                type="float",
                default=1.6,
                minimum=0.0,
                maximum=4.0,
                step=0.05,
                description="Boost the RMS before mapping (stronger reaction).",
            ),
            "reactivity_gamma": PluginParameter(
                name="reactivity_gamma",
                label="Reactivity gamma",
                type="float",
                default=0.55,
                minimum=0.15,
                maximum=2.0,
                step=0.05,
                description="Lower values make small RMS changes more visible.",
            ),
            "fog_strength": PluginParameter(
                name="fog_strength",
                label="Fog strength",
                type="float",
                default=1.15,
                minimum=0.0,
                maximum=4.0,
                step=0.05,
                description="Higher = far rings are dimmer faster.",
            ),
        }

    def create_widget(self, parent: Optional[QWidget] = None) -> QWidget:
        self._widget = _SubOscOscilloscopeTunnelWidget(self.config, parent)
        return self._widget

    def apply_preview_settings(self, width: int, height: int, fps: int) -> None:
        # Olaf will set the widget size; we only expose a timer-interval API stub.
        if self._widget is None:
            return
        self._widget.setMinimumSize(width, height)
        self._widget.setMaximumSize(width, height)
        fps = max(5, min(60, int(fps)))
        self._widget.set_timer_interval(int(1000 / fps))

    def on_audio_features(self, features: Dict[str, Any]) -> None:
        """
        Olaf provides:
          features = {"time_ms": int, "inputs": {"input_1": {"rms":..}, ...}}
        """
        if self._widget is None:
            return

        time_ms = int(features.get("time_ms", 0))
        inputs = features.get("inputs", {}) or {}
        energy = float((inputs.get("input_1", {}) or {}).get("rms", 0.0))

        self._widget.update_audio_state(time_ms=time_ms, energy=energy)
