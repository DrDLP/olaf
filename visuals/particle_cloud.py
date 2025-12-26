"""
Particle Cloud visualization plugin for Olaf.

Adapted from a standalone dense particle cloud concept (librosa + pygfx)
to a lightweight QPainter-based visualization. The idea is:

- A static 3D cloud of particles around the origin.
- A virtual camera orbits inside / around the cloud with a perspective projection.
- Audio energy drives:
    * a "flash" in the center (brighter particles near the core),
    * the angular speed of the camera,
    * overall brightness.

Stem routing suggestion:
- input_1 (drums)  -> flash / flicker (fast transients)
- input_2 (bass)   -> camera rotation speed + depth
- input_3 (vocals) -> color warmth and central emphasis
- input_4 (other)  -> subtle extra brightness / turbulence
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt, QTimer, QSize, QRectF
from PyQt6.QtGui import QPainter, QColor, QRadialGradient
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
    """Classic smoothstep in [0, 1]."""
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


class _ParticleCloudWidget(QWidget):
    """
    Preview / export widget for the Particle Cloud visualization.

    Used both as:
      - a small real-time preview in the Visualizations tab,
      - a large off-screen widget during video export (high resolution).
    """

    def __init__(self, config: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config

        # Time in seconds (driven by the host via update_audio_state()).
        self._time_s: float = 0.0

        # Normalized stem levels in [0, 1].
        self._level_drums: float = 0.0    # input_1
        self._level_bass: float = 0.0     # input_2
        self._level_vocals: float = 0.0   # input_3
        self._level_other: float = 0.0    # input_4

        # Particle data: list of tuples (x, y, z, r_norm, center_weight).
        self._particles = []
        self._last_particle_count: int = 0

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
        self._level_drums = _clamp(level_1, 0.0, 1.0)
        self._level_bass = _clamp(level_2, 0.0, 1.0)
        self._level_vocals = _clamp(level_3, 0.0, 1.0)
        self._level_other = _clamp(level_4, 0.0, 1.0)
        self.update()

    def _on_tick(self) -> None:
        """Internal timer callback for interactive preview."""
        self.update()

    # ------------------------------------------------------------------
    # Particle initialization
    # ------------------------------------------------------------------
    def _ensure_particles(self) -> None:
        """
        Initialize or re-initialize the 3D particle cloud if the number
        of particles in the config changed.
        """
        n = int(self._config.get("num_particles", 4000))
        n = max(500, min(20000, n))

        if n == self._last_particle_count and self._particles:
            return

        self._particles.clear()
        self._last_particle_count = n

        cloud_radius = float(self._config.get("cloud_radius", 10.0))
        cloud_radius = max(1.0, cloud_radius)
        center_sigma = float(self._config.get("center_sigma", 0.35))
        center_sigma = max(0.05, center_sigma)

        # Sampling strategy:
        # - sample random directions on the unit sphere,
        # - sample radius with r ~ U(0,1)^(1/3) to bias towards center,
        # - scale to [0, cloud_radius].
        for _ in range(n):
            u1 = random.random()
            u2 = random.random()
            theta = 2.0 * math.pi * u1
            cosphi = 2.0 * u2 - 1.0
            sinphi = math.sqrt(max(0.0, 1.0 - cosphi * cosphi))

            # Random direction.
            dx = sinphi * math.cos(theta)
            dy = sinphi * math.sin(theta)
            dz = cosphi

            # Radius biased towards the center.
            r_unit = random.random() ** (1.0 / 3.0)
            r = cloud_radius * r_unit

            x = r * dx
            y = r * dy
            z = r * dz

            r_norm = _clamp(r / cloud_radius, 0.0, 1.0)
            center_weight = math.exp(-((r_norm / center_sigma) ** 2))

            self._particles.append((x, y, z, r_norm, center_weight))

    # ------------------------------------------------------------------
    # Main painting logic
    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # type: ignore[override]
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return

        self._ensure_particles()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        cfg = self._config
        t = self._time_s

        cx = w * 0.5
        cy = h * 0.5
        min_dim = float(min(w, h))

        # ------------------------------------------------------------------
        # Audio / energy mapping
        # ------------------------------------------------------------------
        level_drums = self._level_drums
        level_bass = self._level_bass
        level_vocals = self._level_vocals
        level_other = self._level_other

        global_energy = _clamp(
            0.35 * level_bass + 0.35 * level_drums + 0.2 * level_vocals + 0.1 * level_other,
            0.0,
            1.0,
        )

        # Flash level: non-linear mapping + smoothing by design (smoothstep).
        flash_power = float(cfg.get("flash_power", 2.5))
        flash_power = max(0.5, flash_power)
        flash_level = _smoothstep(global_energy) ** flash_power

        # ------------------------------------------------------------------
        # Background
        # ------------------------------------------------------------------
        bg_inner = QColor(3, 3, 10)
        bg_outer = QColor(10, 10, 24)

        e_factor = 0.3 + 0.7 * global_energy
        bg_outer = QColor(
            int(bg_outer.red() * e_factor),
            int(bg_outer.green() * e_factor),
            int(bg_outer.blue() * e_factor),
        )

        bg_radius = min_dim * 0.8
        bg_grad = QRadialGradient(cx, cy, bg_radius, cx, cy)
        bg_grad.setColorAt(0.0, bg_inner)
        bg_grad.setColorAt(1.0, bg_outer)
        painter.fillRect(self.rect(), bg_grad)

        # ------------------------------------------------------------------
        # Camera setup (simplified adaptation of the original script):
        # - orbit around the origin with radius depending on audio,
        # - elevation oscillating slowly over time,
        # - look-at fixed at origin (for stability).
        # ------------------------------------------------------------------
        cloud_radius = float(cfg.get("cloud_radius", 10.0))
        base_cam_radius = float(cfg.get("cam_base_radius", 7.0))
        base_cam_radius = max(1.0, base_cam_radius)

        cam_depth_amplitude = float(cfg.get("cam_depth_amplitude", 2.0))
        cam_depth_amplitude = max(0.0, cam_depth_amplitude)

        cam_depth_speed = float(cfg.get("cam_depth_speed", 0.4))
        cam_depth_speed = max(0.0, cam_depth_speed)

        # Breathing radius
        depth_osc = math.sin(t * cam_depth_speed * 2.0 * math.pi)
        cam_radius = base_cam_radius + cam_depth_amplitude * depth_osc
        cam_radius = _clamp(cam_radius, 0.4 * cloud_radius, 1.5 * cloud_radius)

        # Elevation oscillation
        elev_base_deg = float(cfg.get("cam_elev_base_deg", 18.0))
        elev_var_deg = float(cfg.get("cam_elev_variation_deg", 20.0))
        elev_speed = float(cfg.get("cam_elev_speed", 0.15))

        elev = math.radians(
            elev_base_deg + elev_var_deg * math.sin(t * elev_speed * 2.0 * math.pi)
        )

        # Rotation speed base + gain from audio
        base_rot_speed_deg = float(cfg.get("base_rotation_speed_deg", 50.0))
        base_rot_speed_deg = max(-360.0, min(360.0, base_rot_speed_deg))
        rot_gain = float(cfg.get("rotation_gain", 2.0))
        rot_gain = max(0.0, rot_gain)

        angle_deg = base_rot_speed_deg * t * (1.0 + rot_gain * global_energy)
        angle = math.radians(angle_deg)

        # Camera position in 3D
        cam_x = cam_radius * math.cos(elev) * math.cos(angle)
        cam_y = cam_radius * math.sin(elev)
        cam_z = cam_radius * math.cos(elev) * math.sin(angle)

        cam_pos = (cam_x, cam_y, cam_z)

        # Camera looks at origin
        target = (0.0, 0.0, 0.0)

        # ------------------------------------------------------------------
        # Build camera basis (right, up, forward) for projection
        # ------------------------------------------------------------------
        def v_sub(a, b):
            return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

        def v_dot(a, b):
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

        def v_cross(a, b):
            return (
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            )

        def v_norm(a):
            n = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
            if n < 1e-6:
                return (0.0, 0.0, 0.0)
            return (a[0] / n, a[1] / n, a[2] / n)

        forward = v_norm(v_sub(target, cam_pos))  # from camera to target
        up_world = (0.0, 1.0, 0.0)
        right = v_norm(v_cross(forward, up_world))
        if right == (0.0, 0.0, 0.0):
            # Fallback if forward is colinear with up_world
            right = (1.0, 0.0, 0.0)
        up = v_norm(v_cross(right, forward))

        # Perspective projection
        fov_deg = float(cfg.get("fov_degrees", 60.0))
        fov_deg = _clamp(fov_deg, 20.0, 120.0)
        fov_rad = math.radians(fov_deg)
        focal = 0.5 * min_dim / math.tan(fov_rad / 2.0)
        near_plane = float(cfg.get("near_plane", 0.5))
        near_plane = max(0.01, near_plane)

        # ------------------------------------------------------------------
        # Color mapping parameters (adapted from the original script)
        # ------------------------------------------------------------------
        hue_base = float(cfg.get("hue_base", 0.02))
        hue_range = float(cfg.get("hue_range", 0.88))
        sat_max = float(cfg.get("sat_max", 0.85))
        sat_drop = float(cfg.get("sat_drop", 0.35))
        base_v_min = float(cfg.get("base_v_min", 0.25))
        base_v_range = float(cfg.get("base_v_range", 0.45))

        flash_center_exp = float(cfg.get("flash_center_exp", 1.3))
        flash_strength = float(cfg.get("flash_strength", 3.5))
        flash_strength = max(0.0, flash_strength)

        saturation_global = float(cfg.get("saturation_global", 1.0))
        saturation_global = _clamp(saturation_global, 0.2, 1.5)

        point_size_factor = float(cfg.get("point_size_factor", 0.01))
        point_size_factor = _clamp(point_size_factor, 0.002, 0.05)

        # Slight hue drift with time
        hue_time_shift = (t * float(cfg.get("hue_shift_speed_deg", 30.0)) / 360.0)

        painter.setPen(Qt.PenStyle.NoPen)

        # ------------------------------------------------------------------
        # Draw all particles
        # ------------------------------------------------------------------
        for (px, py, pz, r_norm, center_weight) in self._particles:
            # Transform into camera space
            vx = px - cam_pos[0]
            vy = py - cam_pos[1]
            vz = pz - cam_pos[2]

            # Camera space coordinates
            x_cam = v_dot((vx, vy, vz), right)
            y_cam = v_dot((vx, vy, vz), up)
            z_cam = v_dot((vx, vy, vz), forward)

            if z_cam <= near_plane:
                continue

            # Perspective projection
            x_ndc = (focal * x_cam) / z_cam
            y_ndc = (focal * y_cam) / z_cam

            sx = cx + x_ndc
            sy = cy - y_ndc

            # Skip particles completely off screen (small optimization)
            if sx < -50 or sx > w + 50 or sy < -50 or sy > h + 50:
                continue

            # Color mapping: radial gradient + flash from center
            h_loc = hue_base + hue_range * r_norm + hue_time_shift * 0.5
            # Vocals warm things up slightly
            h_loc += 0.1 * (level_vocals - 0.5)
            s_loc = sat_max - sat_drop * r_norm
            s_loc *= saturation_global

            base_v = base_v_min + base_v_range * (1.0 - r_norm)

            center_boost = center_weight ** flash_center_exp
            flash_factor = 1.0 + flash_strength * flash_level * center_boost
            v_loc = base_v * flash_factor
            v_loc = _clamp(v_loc, 0.0, 1.0)

            col = _hsv_to_qcolor(h_loc, _clamp(s_loc, 0.0, 1.0), v_loc)
            # Alpha slightly decreases with radius
            alpha = int(80 + 150 * (1.0 - r_norm) * (0.3 + 0.7 * global_energy))
            alpha = _clamp(alpha, 20, 255)
            col.setAlpha(alpha)

            painter.setBrush(col)

            # Particle size in pixels
            size_px = point_size_factor * min_dim * (0.5 + 0.5 * (1.0 - r_norm))
            size_px = _clamp(size_px, 1.0, min_dim * 0.04)

            rect = QRectF(
                sx - size_px,
                sy - size_px,
                2.0 * size_px,
                2.0 * size_px,
            )
            painter.drawEllipse(rect)

        painter.end()


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------


class ParticleCloudVisualization(BaseVisualization):
    """
    Audio-driven 3D particle cloud projected in 2D.

    - A static 3D cloud of particles centered at the origin.
    - A virtual camera orbits inside/around the cloud with perspective projection.
    - Audio controls flashes in the core and camera rotation speed.
    """

    plugin_id = "dr_dlp_particle_cloud"
    plugin_name = "Particle Cloud"
    plugin_author = "DrDLP"
    plugin_version = "1.0.0"
    plugin_max_inputs = 4

    plugin_description = (
        "Dense particle cloud with a virtual camera orbiting in 3D space.\n"
        "\n"
        "Recommended stem routing:\n"
        "  - input_1: drums  -> flash / flicker intensity in the core\n"
        "  - input_2: bass   -> camera rotation speed and depth breathing\n"
        "  - input_3: vocals -> color warmth and central emphasis\n"
        "  - input_4: other  -> subtle extra brightness / turbulence\n"
        "\n"
        "You can reroute stems freely, but this mapping usually gives a\n"
        "readable and immersive particle cloud."
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._widget: Optional[_ParticleCloudWidget] = None

        defaults = {
            # Particle / cloud
            "num_particles": 4000,
            "cloud_radius": 10.0,
            "center_sigma": 0.35,
            # Camera / motion
            "cam_base_radius": 7.0,
            "cam_depth_amplitude": 2.0,
            "cam_depth_speed": 0.4,
            "cam_elev_base_deg": 18.0,
            "cam_elev_variation_deg": 20.0,
            "cam_elev_speed": 0.15,
            "base_rotation_speed_deg": 50.0,
            "rotation_gain": 2.0,
            "fov_degrees": 60.0,
            "near_plane": 0.5,
            # Color / flashes
            "hue_base": 0.02,
            "hue_range": 0.88,
            "sat_max": 0.85,
            "sat_drop": 0.35,
            "base_v_min": 0.25,
            "base_v_range": 0.45,
            "flash_power": 2.5,
            "flash_center_exp": 1.3,
            "flash_strength": 3.5,
            "saturation_global": 1.0,
            "point_size_factor": 0.01,
            "hue_shift_speed_deg": 30.0,
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
            "num_particles": PluginParameter(
                name="num_particles",
                label="Number of particles",
                type="int",
                default=4000,
                minimum=500,
                maximum=20000,
                step=500,
                description="Total number of particles in the cloud (performance-sensitive).",
            ),
            "cloud_radius": PluginParameter(
                name="cloud_radius",
                label="Cloud radius",
                type="float",
                default=10.0,
                minimum=3.0,
                maximum=30.0,
                step=0.5,
                description="Overall radius of the 3D particle cloud.",
            ),
            "center_sigma": PluginParameter(
                name="center_sigma",
                label="Center flash width",
                type="float",
                default=0.35,
                minimum=0.1,
                maximum=1.0,
                step=0.05,
                description="Width of the central hot core used for flashes.",
            ),
            "cam_base_radius": PluginParameter(
                name="cam_base_radius",
                label="Camera base radius",
                type="float",
                default=7.0,
                minimum=2.0,
                maximum=20.0,
                step=0.5,
                description="Average distance of the camera from the origin.",
            ),
            "cam_depth_amplitude": PluginParameter(
                name="cam_depth_amplitude",
                label="Camera depth amplitude",
                type="float",
                default=2.0,
                minimum=0.0,
                maximum=10.0,
                step=0.2,
                description="Amplitude of in/out breathing of the camera radius.",
            ),
            "cam_depth_speed": PluginParameter(
                name="cam_depth_speed",
                label="Camera depth speed",
                type="float",
                default=0.4,
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                description="Speed of the camera breathing motion.",
            ),
            "cam_elev_base_deg": PluginParameter(
                name="cam_elev_base_deg",
                label="Base elevation (deg)",
                type="float",
                default=18.0,
                minimum=0.0,
                maximum=80.0,
                step=1.0,
                description="Base vertical angle of the camera in degrees.",
            ),
            "cam_elev_variation_deg": PluginParameter(
                name="cam_elev_variation_deg",
                label="Elevation variation (deg)",
                type="float",
                default=20.0,
                minimum=0.0,
                maximum=60.0,
                step=1.0,
                description="Amplitude of elevation oscillation.",
            ),
            "cam_elev_speed": PluginParameter(
                name="cam_elev_speed",
                label="Elevation speed",
                type="float",
                default=0.15,
                minimum=0.0,
                maximum=1.0,
                step=0.02,
                description="Speed of the elevation oscillation.",
            ),
            "base_rotation_speed_deg": PluginParameter(
                name="base_rotation_speed_deg",
                label="Base rotation speed (deg/s)",
                type="float",
                default=50.0,
                minimum=-360.0,
                maximum=360.0,
                step=5.0,
                description="Base angular speed of the camera around the cloud.",
            ),
            "rotation_gain": PluginParameter(
                name="rotation_gain",
                label="Rotation gain (audio)",
                type="float",
                default=2.0,
                minimum=0.0,
                maximum=5.0,
                step=0.1,
                description="How strongly audio energy speeds up the rotation.",
            ),
            "fov_degrees": PluginParameter(
                name="fov_degrees",
                label="Field of view (deg)",
                type="float",
                default=60.0,
                minimum=20.0,
                maximum=120.0,
                step=1.0,
                description="Perspective field of view used for projection.",
            ),
            "near_plane": PluginParameter(
                name="near_plane",
                label="Near plane",
                type="float",
                default=0.5,
                minimum=0.01,
                maximum=5.0,
                step=0.05,
                description="Near clipping plane in camera space.",
            ),
            "hue_base": PluginParameter(
                name="hue_base",
                label="Hue base",
                type="float",
                default=0.02,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="Hue at the very center of the cloud (0..1).",
            ),
            "hue_range": PluginParameter(
                name="hue_range",
                label="Hue range",
                type="float",
                default=0.88,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="Hue variation from center to edge.",
            ),
            "sat_max": PluginParameter(
                name="sat_max",
                label="Max saturation",
                type="float",
                default=0.85,
                minimum=0.0,
                maximum=1.0,
                step=0.02,
                description="Saturation at the center.",
            ),
            "sat_drop": PluginParameter(
                name="sat_drop",
                label="Saturation drop",
                type="float",
                default=0.35,
                minimum=0.0,
                maximum=1.0,
                step=0.02,
                description="Decrease in saturation towards the edge.",
            ),
            "base_v_min": PluginParameter(
                name="base_v_min",
                label="Base brightness min",
                type="float",
                default=0.25,
                minimum=0.0,
                maximum=1.0,
                step=0.02,
                description="Minimal base brightness at the cloud edge.",
            ),
            "base_v_range": PluginParameter(
                name="base_v_range",
                label="Base brightness range",
                type="float",
                default=0.45,
                minimum=0.0,
                maximum=1.0,
                step=0.02,
                description="Additional brightness towards the center.",
            ),
            "flash_power": PluginParameter(
                name="flash_power",
                label="Flash nonlinearity",
                type="float",
                default=2.5,
                minimum=0.5,
                maximum=5.0,
                step=0.1,
                description="Exponent applied to global energy for flash shaping.",
            ),
            "flash_center_exp": PluginParameter(
                name="flash_center_exp",
                label="Flash center exponent",
                type="float",
                default=1.3,
                minimum=0.5,
                maximum=4.0,
                step=0.1,
                description="Emphasis of the core vs outer shell during flashes.",
            ),
            "flash_strength": PluginParameter(
                name="flash_strength",
                label="Flash strength",
                type="float",
                default=3.5,
                minimum=0.0,
                maximum=8.0,
                step=0.2,
                description="How strongly flashes amplify brightness.",
            ),
            "saturation_global": PluginParameter(
                name="saturation_global",
                label="Global saturation",
                type="float",
                default=1.0,
                minimum=0.2,
                maximum=1.5,
                step=0.05,
                description="Overall saturation multiplier for all colors.",
            ),
            "point_size_factor": PluginParameter(
                name="point_size_factor",
                label="Point size factor",
                type="float",
                default=0.01,
                minimum=0.002,
                maximum=0.05,
                step=0.001,
                description="Base size of particles relative to frame size.",
            ),
            "hue_shift_speed_deg": PluginParameter(
                name="hue_shift_speed_deg",
                label="Hue shift speed (deg/s)",
                type="float",
                default=30.0,
                minimum=0.0,
                maximum=360.0,
                step=5.0,
                description="Speed of slow hue drift over time.",
            ),
        }

    # ------------------------------------------------------------------
    # Widget creation / host integration
    # ------------------------------------------------------------------
    def create_widget(self, parent: Optional[QWidget] = None) -> QWidget:
        """
        Create the Qt widget used for both preview and off-screen rendering.
        """
        self._widget = _ParticleCloudWidget(self.config, parent=parent)
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

        level_1 = float(i1.get("rms", 0.0))
        level_2 = float(i2.get("rms", 0.0))
        level_3 = float(i3.get("rms", 0.0))
        level_4 = float(i4.get("rms", 0.0))

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
