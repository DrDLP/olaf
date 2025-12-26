"""
VisPy organic double neon vortex tunnel visualization plugin for Olaf.

This version renders TWO bright, glowing tunnels seen from the front:
- an inner vortex (smaller radius),
- an outer vortex (larger radius), slightly offset and farther away.

Geometry:
    - Two deformed cylindrical "tubes" in front of the camera.
    - We render the inside of the tubes and look straight into them.

Audio:
    - Global RMS energy inflates the tunnel and boosts the glow.
    - Input 1 RMS modulates the petal amplitude (good candidate for drums).
    - Input 2 RMS modulates the twist strength (good candidate for leads / pads).

Color:
    - Inner and outer tunnels can use different neon templates.
    - Inner: bright but with some vignette.
    - Outer: full neon (no dark edges), strong glow, minimum brightness.
    - The inner vortex opacity is adjustable to reveal the outer one.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import math
import numpy as np

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

# Optional VisPy dependency
try:
    from vispy import app, scene  # type: ignore
    HAVE_VISPY = True
except Exception:
    app = None  # type: ignore
    scene = None  # type: ignore
    HAVE_VISPY = False

# Support both flat-layout and package install for imports
try:
    from olaf_app.visualization_api import BaseVisualization, PluginParameter
except ImportError:
    from visualization_api import BaseVisualization, PluginParameter


# ---------------------------------------------------------------------------
# PREVIEW WIDGET
# ---------------------------------------------------------------------------

class _VispyVortexWidget(QWidget):
    """
    QWidget embedding a VisPy SceneCanvas that renders a double neon tunnel.

    The tunnels are cylindrical surfaces whose radius and twist vary along
    the depth to create a soft, blooming vortex. The center glows more as
    the audio energy increases.
    """

    def __init__(self, config: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._time_s: float = 0.0
        self._energy: float = 0.0          # global normalized energy [0, 1]
        self._petal_mod: float = 0.0       # from input_1 RMS in [0, 1]
        self._twist_mod: float = 0.0       # from input_2 RMS in [0, 1]

        self._canvas: Optional["scene.SceneCanvas"] = None
        self._view = None
        self._surface_inner: Optional["scene.visuals.SurfacePlot"] = None
        self._surface_outer: Optional["scene.visuals.SurfacePlot"] = None

        # Shared grids
        self._theta_mesh: Optional[np.ndarray] = None
        self._depth_mesh: Optional[np.ndarray] = None

        if not HAVE_VISPY:
            layout = QVBoxLayout(self)
            label = QLabel(
                "VisPy is not available.\n"
                "VisPy Neon Tunnel preview is disabled.\n"
                "Please install 'vispy' to enable this visualization.",
                self,
            )
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            return

        # Make sure VisPy uses the PyQt6 backend
        app.use_app("pyqt6")  # type: ignore[arg-type]

        self._canvas = scene.SceneCanvas(
            keys=None,
            size=(640, 360),
            show=False,
            bgcolor="black",
        )
        self._canvas.create_native()
        self._canvas.native.setParent(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas.native)

        # 3D view: camera parameters will be refined once we know tunnel_length
        self._view = self._canvas.central_widget.add_view()
        self._view.camera = "turntable"
        self._view.camera.fov = 60

        self._init_vortex_surfaces()

    def get_background_target(self):
        return {
            "canvas": getattr(self, "canvas", None),
            "view": getattr(self, "view", None),
            "scene": getattr(getattr(self, "view", None), "scene", None),
            "camera": getattr(getattr(self, "view", None), "camera", None),
        }

    # ------------------------------------------------------------------ #
    # Geometry initialization                                            #
    # ------------------------------------------------------------------ #

    def _init_vortex_surfaces(self) -> None:
        """
        Build the parametric grid (theta, depth) and initialize the two surfaces.
        """
        if not HAVE_VISPY or self._canvas is None or self._view is None:
            return

        # Fixed resolution: we do not expose these as sliders in the UI.
        angle_steps = 256
        depth_steps = 256

        # Angle around the tunnel axis
        theta_grid = np.linspace(0.0, 2.0 * math.pi, angle_steps, dtype=np.float32)
        # Depth from near (front) to far
        depth_grid = np.linspace(0.0, 1.0, depth_steps, dtype=np.float32)

        self._theta_mesh, self._depth_mesh = np.meshgrid(theta_grid, depth_grid)

        # Placeholders (flat tube before first update)
        zeros = np.zeros_like(self._theta_mesh, dtype=np.float32)

        # Important: create OUTER first, then INNER, so inner is drawn above.
        self._surface_outer = scene.visuals.SurfacePlot(
            x=zeros,
            y=zeros,
            z=zeros,
            color=(0.4, 0.4, 0.8, 1.0),
            shading="smooth",
            parent=self._view.scene,
        )
        self._surface_inner = scene.visuals.SurfacePlot(
            x=zeros,
            y=zeros,
            z=zeros,
            color=(1.0, 1.0, 1.0, 1.0),
            shading="smooth",
            parent=self._view.scene,
        )

        # First computation of geometry + colors
        self._update_vortex_surface()

        # Camera INSIDE the tunnel, on the central axis, looking forward.
        tunnel_length = float(self._config.get("tunnel_length", 6.0))
        tunnel_length = max(2.0, min(tunnel_length, 20.0))

        center_z = -0.6 * tunnel_length   # look-at point (deep inside)
        distance = 0.4 * tunnel_length    # camera sits between 0 and center_z

        # Turntable camera with elevation 90° means: camera lies on the Z axis.
        self._view.camera.center = (0.0, 0.0, center_z)
        self._view.camera.distance = distance
        self._view.camera.elevation = 90
        self._view.camera.azimuth = 0

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(640, 360)

    # ------------------------------------------------------------------ #
    # Host interaction                                                   #
    # ------------------------------------------------------------------ #

    def set_timer_interval(self, interval_ms: int) -> None:
        """
        Stub for compatibility with other plugins (no internal timer needed).
        """
        _ = interval_ms  # no-op

    def update_audio_state(
        self,
        time_ms: int,
        energy: float,
        petal_mod: float = 0.0,
        twist_mod: float = 0.0,
    ) -> None:
        """
        Update internal time and audio state, then recompute the tunnels.
        """
        self._time_s = time_ms / 1000.0
        self._energy = float(max(0.0, min(1.0, energy)))
        self._petal_mod = float(max(0.0, min(1.0, petal_mod)))
        self._twist_mod = float(max(0.0, min(1.0, twist_mod)))
        self._update_vortex_surface()
        if HAVE_VISPY and self._canvas is not None:
            self._canvas.update()

    # ------------------------------------------------------------------ #
    # Color palettes                                                     #
    # ------------------------------------------------------------------ #

    def _get_palette_for_theme(self, theme: str) -> Dict[str, np.ndarray]:
        """
        Return (near_color, far_color, glow_color) for a given theme.
        All colors are RGBA arrays in [0, 1], intentionally bright.
        """
        if theme == "cyan_blue":
            near_color = np.array([0.95, 1.00, 1.00, 1.00], dtype=np.float32)  # almost white cyan
            far_color = np.array([0.25, 0.45, 0.90, 1.00], dtype=np.float32)   # bright blue
            glow_color = np.array([0.80, 1.00, 1.00, 0.00], dtype=np.float32)
        elif theme == "gold_magenta":
            near_color = np.array([1.00, 0.96, 0.75, 1.00], dtype=np.float32)  # light gold
            far_color = np.array([0.75, 0.30, 0.80, 1.00], dtype=np.float32)   # bright magenta
            glow_color = np.array([1.00, 0.80, 0.95, 0.00], dtype=np.float32)
        elif theme == "acid_green":
            near_color = np.array([0.95, 1.00, 0.85, 1.00], dtype=np.float32)  # light lime
            far_color = np.array([0.40, 0.95, 0.65, 1.00], dtype=np.float32)   # bright green/teal
            glow_color = np.array([0.90, 1.00, 0.90, 0.00], dtype=np.float32)
        elif theme == "ice_white":
            near_color = np.array([1.00, 1.00, 1.00, 1.00], dtype=np.float32)  # pure white
            far_color = np.array([0.65, 0.80, 0.98, 1.00], dtype=np.float32)   # light icy blue
            glow_color = np.array([0.95, 0.98, 1.00, 0.00], dtype=np.float32)
        else:
            # Default: bright pink / violet
            near_color = np.array([1.00, 0.90, 1.00, 1.00], dtype=np.float32)  # very light pink
            far_color = np.array([0.70, 0.35, 0.90, 1.00], dtype=np.float32)   # bright violet
            glow_color = np.array([1.00, 0.85, 1.00, 0.00], dtype=np.float32)

        return {
            "near": near_color,
            "far": far_color,
            "glow": glow_color,
        }

    # ------------------------------------------------------------------ #
    # Vortex computation                                                 #
    # ------------------------------------------------------------------ #

    def _update_vortex_surface(self) -> None:
        """
        Recompute geometry and per-vertex neon colors for both tunnels.
        """
        if not HAVE_VISPY:
            return
        if (
            self._surface_inner is None
            or self._surface_outer is None
            or self._theta_mesh is None
            or self._depth_mesh is None
        ):
            return

        # ------------------------------------------------------------------
        # 1) Read and clamp configuration parameters
        # ------------------------------------------------------------------
        base_radius = float(self._config.get("base_radius", 1.5))
        bloom_factor = float(self._config.get("bloom_factor", 0.8))
        petal_amplitude = float(self._config.get("petal_amplitude", 0.35))
        petal_count = float(self._config.get("petal_count", 9.0))
        twist_strength = float(self._config.get("twist_strength", 2.8))
        twist_speed = float(self._config.get("twist_speed", 0.8))
        tunnel_length = float(self._config.get("tunnel_length", 6.0))
        noise_depth_factor = float(self._config.get("noise_depth_factor", 3.5))

        outer_radius_factor = float(self._config.get("outer_radius_factor", 1.5))
        inner_opacity = float(self._config.get("inner_opacity", 0.7))
        outer_brightness = float(self._config.get("outer_brightness", 0.9))
        outer_depth_offset = float(self._config.get("outer_depth_offset", 1.0))
        outer_offset_x = float(self._config.get("outer_offset_x", 0.6))

        inner_theme = str(self._config.get("color_theme", "pink_violet"))
        outer_theme = str(self._config.get("outer_color_theme", "cyan_blue"))

        base_radius = max(0.3, min(base_radius, 5.0))
        bloom_factor = max(0.0, min(bloom_factor, 2.0))
        petal_amplitude = max(0.0, min(petal_amplitude, 1.0))
        petal_count = max(3.0, min(petal_count, 32.0))
        twist_strength = max(0.0, min(twist_strength, 6.0))
        twist_speed = max(0.0, min(twist_speed, 5.0))
        tunnel_length = max(2.0, min(tunnel_length, 20.0))
        noise_depth_factor = max(0.0, min(noise_depth_factor, 10.0))
        outer_radius_factor = max(1.05, min(outer_radius_factor, 3.0))
        inner_opacity = max(0.0, min(inner_opacity, 1.0))
        outer_brightness = max(0.1, min(outer_brightness, 2.0))
        outer_depth_offset = max(0.0, min(outer_depth_offset, 10.0))
        outer_offset_x = max(-10.0, min(outer_offset_x, 10.0))

        # ------------------------------------------------------------------
        # 2) Normalized depth and angle grids
        # ------------------------------------------------------------------
        theta = np.asarray(self._theta_mesh, dtype=np.float32)
        depth_raw = np.asarray(self._depth_mesh, dtype=np.float32)

        d_min = float(depth_raw.min())
        d_max = float(depth_raw.max())
        span = max(1e-6, d_max - d_min)
        depth = (depth_raw - d_min) / span  # [0, 1]

        # ------------------------------------------------------------------
        # 3) Radial profile, energy term and petal modulation
        # ------------------------------------------------------------------
        energy_term = bloom_factor * float(self._energy)

        # Slightly smaller radius towards the back of the tunnel
        radial_profile = (0.4 + 0.6 * (1.0 - depth)).astype(np.float32)

        # Effective petal amplitude boosted by input_1 RMS
        petal_amp_eff = petal_amplitude * (0.4 + 1.6 * self._petal_mod)

        petal_wave = np.cos(
            petal_count * theta + noise_depth_factor * depth + 2.0 * self._time_s
        ).astype(np.float32)

        radius_local_inner = (
            base_radius * radial_profile
            + energy_term * (1.2 - 0.7 * depth)
            + petal_amp_eff * (0.4 + 0.6 * (1.0 - depth)) * petal_wave
        ).astype(np.float32)

        # Outer radius: scaled version of the inner one
        radius_local_outer = (outer_radius_factor * radius_local_inner).astype(np.float32)

        # ------------------------------------------------------------------
        # 4) Twisting and Cartesian coordinates
        # ------------------------------------------------------------------
        twist_strength_eff = twist_strength * (0.5 + 1.5 * self._twist_mod)
        twist_speed_eff = twist_speed * (0.4 + 0.6 * self._energy)

        twist = twist_strength_eff * depth + twist_speed_eff * self._time_s
        theta_twisted = theta + twist

        # Inner vortex geometry
        x_inner = radius_local_inner * np.cos(theta_twisted)
        y_inner = radius_local_inner * np.sin(theta_twisted)
        z_inner = -depth * tunnel_length

        # Outer vortex geometry:
        #  - same depth profile but pushed further back
        #  - offset along X so it's not concentric with the inner tunnel
        depth_offset = outer_depth_offset * (tunnel_length / 6.0)
        x_outer = radius_local_outer * np.cos(theta_twisted) + outer_offset_x
        y_outer = radius_local_outer * np.sin(theta_twisted)
        z_outer = -(depth * tunnel_length + depth_offset)

        # Clean NaN / inf
        x_inner = np.nan_to_num(x_inner, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_inner = np.nan_to_num(y_inner, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        z_inner = np.nan_to_num(z_inner, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        x_outer = np.nan_to_num(x_outer, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_outer = np.nan_to_num(y_outer, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        z_outer = np.nan_to_num(z_outer, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # ------------------------------------------------------------------
        # 5) Per-vertex neon colors for inner and outer tunnels
        # ------------------------------------------------------------------
        depth_flat = depth.astype(np.float32)                 # (H, W)
        t_depth = depth_flat[..., None]                       # (H, W, 1)

        # Inner palette
        inner_palette = self._get_palette_for_theme(inner_theme)
        near_inner = inner_palette["near"]  # (4,)
        far_inner = inner_palette["far"]
        glow_inner = inner_palette["glow"]

        # Outer palette (can be completely different)
        outer_palette = self._get_palette_for_theme(outer_theme)
        near_outer = outer_palette["near"]
        far_outer = outer_palette["far"]
        glow_outer = outer_palette["glow"]

        # ---- Inner: gradient + glow + vignette ----
        base_inner = near_inner * (1.0 - t_depth) + far_inner * t_depth
        glow_strength_inner = 0.8 + 1.2 * float(self._energy)
        center_falloff = np.exp(-3.0 * depth_flat)[..., None]
        glow_inner_field = glow_inner * center_falloff * glow_strength_inner

        # Radial vignette based on inner geometry
        r_inner = np.sqrt(x_inner ** 2 + y_inner ** 2)
        r_norm_inner = r_inner / (np.max(r_inner) + 1e-6)
        vignette_inner = (0.85 + 0.15 * (1.0 - r_norm_inner)).astype(np.float32)[..., None]

        colors_inner = (base_inner + glow_inner_field) * vignette_inner
        colors_inner = np.clip(colors_inner, 0.0, 1.0)
        colors_inner[..., 3] *= (0.85 + 0.15 * (1.0 - depth_flat))
        colors_inner[..., 3] *= inner_opacity  # reveal outer tunnel
        colors_inner = np.nan_to_num(colors_inner, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- Outer: full neon, no dark edges ----
        base_outer = near_outer * (1.0 - t_depth) + far_outer * t_depth
        glow_strength_outer = 0.9 + 1.3 * float(self._energy)
        glow_outer_field = glow_outer * center_falloff * glow_strength_outer

        colors_outer = base_outer + glow_outer_field
        colors_outer[..., :3] *= outer_brightness

        # Important: ensure a minimum brightness so it never fades to black.
        min_brightness = 0.25
        colors_outer[..., :3] = np.maximum(colors_outer[..., :3], min_brightness)

        colors_outer = np.clip(colors_outer, 0.0, 1.0)
        colors_outer[..., 3] *= (0.75 + 0.25 * (1.0 - depth_flat))
        colors_outer = np.nan_to_num(colors_outer, nan=0.0, posinf=0.0, neginf=0.0)

        # ------------------------------------------------------------------
        # 6) Push geometry + colors to VisPy
        # ------------------------------------------------------------------
        self._surface_outer.set_data(
            x=x_outer,
            y=y_outer,
            z=z_outer,
            colors=colors_outer,
        )
        self._surface_inner.set_data(
            x=x_inner,
            y=y_inner,
            z=z_inner,
            colors=colors_inner,
        )


# ---------------------------------------------------------------------------
# PLUGIN CLASS
# ---------------------------------------------------------------------------

class VispyVortexTunnelVisualization(BaseVisualization):
    """
    Double VisPy-based neon tunnel visualization plugin.

    Audio routing:
        - Global energy        = max RMS over inputs  -> tunnel inflation + glow.
        - Input 1 RMS (input_1) -> petal amplitude modulation.
        - Input 2 RMS (input_2) -> twist strength modulation.
    """

    plugin_id = "dr_dlp_vispy_vortex_tunnel"
    plugin_name = "VisPy Neon Double Tunnel"
    plugin_description = (
        "Double bright neon tunnel rendered in 3D with VisPy, with inner and "
        "outer tunnels using independent color templates and stem-driven "
        "geometry (petals + twist). The outer tunnel is offset and farther "
        "to add depth and parallax."
    )
    plugin_author = "DrDLP"
    plugin_version = "0.10.0"
    plugin_max_inputs = 2

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._widget: Optional[_VispyVortexWidget] = None

        defaults = {
            "base_radius": 1.5,
            "bloom_factor": 0.8,
            "petal_amplitude": 0.35,
            "petal_count": 24.0,
            "twist_strength": 4.8,
            "twist_speed": 0.8,
            "tunnel_length": 6.0,
            "noise_depth_factor": 3.5,
            "color_theme": "pink_violet",     # inner tunnel
            "outer_color_theme": "cyan_blue", # outer tunnel
            "inner_opacity": 0.5,
            "outer_radius_factor": 1.15,
            "outer_brightness": 0.55,
            "outer_depth_offset": 0.0,
            "outer_offset_x": 0.3,
        }
        for key, value in defaults.items():
            self.config.setdefault(key, value)

    # ------------------------------------------------------------------ #
    # Parameters exposed to the UI                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        return {
            "base_radius": PluginParameter(
                name="base_radius",
                label="Base radius",
                type="float",
                default=1.5,
                minimum=0.3,
                maximum=5.0,
                step=0.1,
                description="Base radius of the inner tunnel when the audio is silent.",
            ),
            "bloom_factor": PluginParameter(
                name="bloom_factor",
                label="Energy bloom",
                type="float",
                default=0.8,
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                description="How much the audio inflates the front of the tunnels.",
            ),
            "petal_amplitude": PluginParameter(
                name="petal_amplitude",
                label="Petal amplitude (base)",
                type="float",
                default=0.35,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                description="Base strength of floral / petal-like distortions.",
            ),
            "petal_count": PluginParameter(
                name="petal_count",
                label="Petal count",
                type="float",
                default=24.0,
                minimum=3.0,
                maximum=32.0,
                step=1.0,
                description="Number of lobes around the tunnel.",
            ),
            "twist_strength": PluginParameter(
                name="twist_strength",
                label="Twist strength (base)",
                type="float",
                default=4.8,
                minimum=0.0,
                maximum=6.0,
                step=0.1,
                description="Base amount of twist over the tunnel depth.",
            ),
            "twist_speed": PluginParameter(
                name="twist_speed",
                label="Base twist speed",
                type="float",
                default=0.8,
                minimum=0.0,
                maximum=5.0,
                step=0.1,
                description="Base rotation speed (scaled by global energy).",
            ),
            "tunnel_length": PluginParameter(
                name="tunnel_length",
                label="Tunnel length",
                type="float",
                default=6.0,
                minimum=2.0,
                maximum=20.0,
                step=0.5,
                description="Depth of the tunnels in scene units.",
            ),
            "noise_depth_factor": PluginParameter(
                name="noise_depth_factor",
                label="Depth noise factor",
                type="float",
                default=3.5,
                minimum=0.0,
                maximum=10.0,
                step=0.1,
                description="How fast the petal pattern changes along the depth.",
            ),
            "inner_opacity": PluginParameter(
                name="inner_opacity",
                label="Inner vortex opacity",
                type="float",
                default=0.5,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                description="Opacity of the inner tunnel (0 = fully transparent, 1 = fully opaque).",
            ),
            "outer_radius_factor": PluginParameter(
                name="outer_radius_factor",
                label="Outer radius factor",
                type="float",
                default=1.15,
                minimum=1.05,
                maximum=3.0,
                step=0.05,
                description="Scale factor between inner and outer tunnel radii.",
            ),
            "outer_brightness": PluginParameter(
                name="outer_brightness",
                label="Outer brightness",
                type="float",
                default=0.5,
                minimum=0.1,
                maximum=2.0,
                step=0.05,
                description="Brightness multiplier applied to the outer tunnel colors.",
            ),
            "outer_depth_offset": PluginParameter(
                name="outer_depth_offset",
                label="Outer depth offset",
                type="float",
                default=1.0,
                minimum=0.0,
                maximum=10.0,
                step=0.1,
                description="How much farther the outer tunnel is along the Z axis.",
            ),
            "outer_offset_x": PluginParameter(
                name="outer_offset_x",
                label="Outer X offset",
                type="float",
                default=0.0,
                minimum=-10.0,
                maximum=10.0,
                step=0.1,
                description="Horizontal offset of the outer tunnel (parallax effect).",
            ),
            "color_theme": PluginParameter(
                name="color_theme",
                label="Inner color template",
                type="enum",
                default="pink_violet",
                choices=[
                    "pink_violet",
                    "cyan_blue",
                    "gold_magenta",
                    "acid_green",
                    "ice_white",
                ],
                description="Neon palette for the INNER tunnel.",
            ),
            "outer_color_theme": PluginParameter(
                name="outer_color_theme",
                label="Outer color template",
                type="enum",
                default="cyan_blue",
                choices=[
                    "pink_violet",
                    "cyan_blue",
                    "gold_magenta",
                    "acid_green",
                    "ice_white",
                ],
                description="Neon palette for the OUTER tunnel.",
            ),
        }

    # ------------------------------------------------------------------ #
    # Widget / host hooks                                                #
    # ------------------------------------------------------------------ #

    def create_widget(self, parent: Optional[QWidget] = None) -> QWidget:
        self._widget = _VispyVortexWidget(self.config, parent)
        return self._widget

    def apply_preview_settings(self, width: int, height: int, fps: int) -> None:
        if self._widget is None:
            return

        fps = max(5, min(60, int(fps)))
        interval_ms = int(1000 / fps)

        self._widget.setMinimumSize(width, height)
        self._widget.setMaximumSize(width, height)
        self._widget.set_timer_interval(interval_ms)

    def on_audio_features(self, features: Dict[str, Any]) -> None:
        """
        Host callback: routes audio features into the visualization.

        - Global energy  = max RMS of available inputs.
        - Petal amplitude modulation = input_1 RMS.
        - Twist strength modulation  = input_2 RMS.
        """
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

        petal_mod = max(0.0, min(1.0, level_1))
        twist_mod = max(0.0, min(1.0, level_2))

        self._widget.update_audio_state(
            time_ms=time_ms,
            energy=energy,
            petal_mod=petal_mod,
            twist_mod=twist_mod,
        )

    def on_activate(self) -> None:
        pass

    def on_deactivate(self) -> None:
        pass
