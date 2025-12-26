"""sdf_neon_tunnel_aggressive.py

Olaf Visualization Plugin (VisPy / Gloo) - SDF Raymarched Neon Tunnel (Aggressive)

Concept
------
- Infinite neon tunnel rendered with raymarching (signed distance fields).
- Repeating gates/rings along the tunnel.
- Aggressive look compared to the baseline tunnel:
  * Multi-pass bloom (prefilter + separable blur + combine)
  * Optional scanlines + vignette + extra contrast
  * Post-process chromatic aberration (cheaper than triple raymarch)

Audio mapping (recommended routing)
---------------------------------
- input_1 (MAIN / full mix): speed + overall intensity (also pushes exposure a bit)
- input_2 (DRUMS): gates opening/closing (constriction + ring punch)
- input_3 (BASS): radial wobble (tunnel center offset / breathing)

Implementation notes
--------------------
- Designed for Olaf's export pipeline (off-screen capture of the Qt widget).
- UI changes are read from the shared `config` dict every frame.
- Bloom is implemented using two downsampled framebuffers for performance.

Author: DrDLP
License: same as project
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

try:
    # When Olaf is installed as a package
    from olaf_app.visualization_api import BaseVisualization, PluginParameter
except Exception:
    # When running from source / single-file testing
    from visualization_api import BaseVisualization, PluginParameter


# ---------------------------------------------------------------------
# Optional VisPy dependency
# ---------------------------------------------------------------------
try:
    from vispy import app, gloo  # type: ignore

    HAVE_VISPY = True
except Exception:
    HAVE_VISPY = False

try:
    import numpy as np  # type: ignore

    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class _AudioState:
    """Smoothed audio state used as shader uniforms."""

    t: float = 0.0
    energy: float = 0.0


# ---------------------------------------------------------------------
# Color presets
# ---------------------------------------------------------------------

# Kept as strings so the UI can display them directly (enum parameter).
_COLOR_PRESETS = [
    "Cyberpunk (Cyan/Magenta)",
    "Synthwave (Pink/Orange)",
    "Matrix (Green)",
    "Electric (Blue/Purple)",
    "Acid (Yellow/Cyan)",
    "Infrared (Red/Pink)",
    "Rainbow",
]

# Shader uses a float id (0..N-1) for GLSL ES 1.0 compatibility.
_COLOR_PRESET_TO_ID = {name: float(i) for i, name in enumerate(_COLOR_PRESETS)}

# ---------------------------------------------------------------------
# GLSL shaders
# ---------------------------------------------------------------------

_VERTEX = r"""
attribute vec2 a_position;
varying vec2 v_uv;

void main()
{
    v_uv = (a_position + vec2(1.0)) * 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

# Scene (raymarch) fragment shader: outputs *linear* color.
# Post processing (bloom/chroma/gamma/scanlines) happens in later passes.
_FRAGMENT_SCENE = r"""
precision highp float;

varying vec2 v_uv;

uniform vec2  u_resolution;
uniform float u_time;

uniform float u_main;
uniform float u_palette_id;

uniform float u_tunnel_radius;
uniform float u_ring_frequency;
uniform float u_twist;
uniform float u_gate_strength;
uniform float u_fog;
uniform float u_detail_steps;   // 32..160 (used as an early-exit cap)
uniform float u_glow_gain;

#define MAX_STEPS 160
#define MAX_DIST  90.0
#define EPS       0.0014

mat2 rot(float a)
{
    float s = sin(a);
    float c = cos(a);
    return mat2(c, -s, s, c);
}

vec3 _sat(vec3 c, float s)
{
    float luma = dot(c, vec3(0.299, 0.587, 0.114));
    return mix(vec3(luma), c, s);
}

vec3 neon_palette(float k, float preset)
{
    float t = 6.2831 * k;
    float s1 = 0.5 + 0.5 * sin(t);
    float s2 = 0.5 + 0.5 * sin(t + 2.0944);
    float s3 = 0.5 + 0.5 * sin(t + 4.1888);

    vec3 c = vec3(1.0);

    if (preset < 0.5)
    {
        // Cyberpunk (Cyan / Magenta)
        vec3 a = vec3(0.00, 1.00, 1.00);
        vec3 b = vec3(1.00, 0.00, 1.00);
        vec3 d = mix(a, b, s1);
        c = mix(d, vec3(0.15, 0.50, 1.00), 0.35 * s2);
    }
    else if (preset < 1.5)
    {
        // Synthwave (Pink / Orange)
        vec3 a = vec3(1.00, 0.10, 0.80);
        vec3 b = vec3(1.00, 0.55, 0.10);
        c = mix(a, b, s1);
        c = mix(c, vec3(0.30, 0.95, 1.00), 0.25 * s3);
    }
    else if (preset < 2.5)
    {
        // Matrix (Neon Green)
        vec3 a = vec3(0.05, 0.35, 0.08);
        vec3 b = vec3(0.10, 1.00, 0.25);
        c = mix(a, b, s1);
    }
    else if (preset < 3.5)
    {
        // Electric (Blue / Purple)
        vec3 a = vec3(0.10, 0.35, 1.00);
        vec3 b = vec3(0.75, 0.15, 1.00);
        c = mix(a, b, s1);
        c = mix(c, vec3(0.00, 1.00, 0.85), 0.20 * s2);
    }
    else if (preset < 4.5)
    {
        // Acid (Yellow / Cyan)
        vec3 a = vec3(1.00, 0.95, 0.10);
        vec3 b = vec3(0.00, 1.00, 0.95);
        c = mix(a, b, s1);
    }
    else if (preset < 5.5)
    {
        // Infrared (Red / Pink)
        vec3 a = vec3(1.00, 0.05, 0.15);
        vec3 b = vec3(1.00, 0.15, 0.90);
        c = mix(a, b, s1);
    }
    else
    {
        // Rainbow (original sinus palette)
        c = 0.55 + 0.45 * vec3(
            sin(6.2831 * (k + 0.00)),
            sin(6.2831 * (k + 0.33)),
            sin(6.2831 * (k + 0.66))
        );
    }

    return _sat(c, 1.35);
}

float sd_torus(vec3 p, vec2 t)
{
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

// Scene SDF: inside-tunnel + repeating ring gates.
float map_sdf(vec3 p)
{
    // Twist the cross-section.
    float tw = u_twist;
    p.xy = rot(tw * p.z * 0.14 + 0.22 * u_time) * p.xy;

    // Repetition along z.
    float freq = max(0.001, u_ring_frequency);
    float period = 6.2831 / freq;
    float zrep = mod(p.z + 0.5 * period, period) - 0.5 * period;

    // Gates: constriction + ring accent (static; no audio modulation).
    float gate_amp = u_gate_strength;
    float gate = 1.0 - gate_amp * exp(-10.0 * zrep * zrep);

    float R = max(0.10, u_tunnel_radius * gate);

    // Tunnel wall (distance to cylinder surface).
    float d_wall = abs(length(p.xy) - R);

    // Ring gate: a torus-like band around the wall.
    float ring_thickness = 0.14 + 0.42 * gate_amp;
    float ring_radius = R * (0.88 + 0.06 * sin(1.2 * u_time + p.z * 0.15));
    float d_ring = length(vec2(length(p.xy) - ring_radius, zrep)) - ring_thickness;

    // Add thin inner tracers for more "busy" detail.
    float tracer_r = R * (0.55 + 0.15 * sin(0.7 * u_time + p.z * 0.35));
    float d_tracer = abs(length(p.xy) - tracer_r) - 0.012;

    return min(min(d_wall, d_ring), d_tracer);
}

vec3 estimate_normal(vec3 p)
{
    const vec2 e = vec2(0.001, 0.0);
    float d = map_sdf(p);
    vec3 n = vec3(
        map_sdf(p + vec3(e.x, e.y, e.y)) - d,
        map_sdf(p + vec3(e.y, e.x, e.y)) - d,
        map_sdf(p + vec3(e.y, e.y, e.x)) - d
    );
    return normalize(n);
}

// Returns (hit_t, glow_accum).
vec2 raymarch(vec3 ro, vec3 rd)
{
    float t = 0.0;
    float glow = 0.0;

    int steps_cap = int(clamp(u_detail_steps, 32.0, float(MAX_STEPS)));

    for (int i = 0; i < MAX_STEPS; i++)
    {
        if (i >= steps_cap) break;

        vec3 p = ro + rd * t;
        float d = map_sdf(p);

        // Volumetric-ish glow accumulation.
        glow += u_glow_gain * (0.012 + 0.030 * u_main) * exp(-12.0 * d);

        if (d < EPS || t > MAX_DIST)
            break;

        // Slightly smaller steps => more detail (more aggressive).
        t += d * 0.78;
    }

    return vec2(t, glow);
}

vec3 shade(vec3 p, vec3 rd, float glow_accum)
{
    vec3 n = estimate_normal(p);

    vec3 light_dir = normalize(vec3(0.35, 0.70, -0.65));
    float diff = clamp(dot(n, light_dir) * 0.5 + 0.5, 0.0, 1.0);

    float ang = atan(p.y, p.x) / 6.2831;
    float k = 0.12 * p.z + ang + 0.04 * u_time;
    vec3 base = neon_palette(k, u_palette_id);

    // Rim lighting (very neon).
    float rim = pow(1.0 - clamp(dot(n, -rd), 0.0, 1.0), 2.2);

    // Stronger main-driven intensity.
    float intensity = 0.70 + 1.75 * u_main;

    vec3 col = intensity * base * (0.20 + 1.05 * diff) + 1.35 * rim * base;

    // Add accumulated glow.
    col += glow_accum * (0.80 + 0.80 * u_main) * base;

    return col;
}

void main()
{
    vec2 fc = v_uv * u_resolution;

    // Normalized coords (-1..1), aspect corrected.
    vec2 uv = (fc / u_resolution) * 2.0 - 1.0;
    uv.x *= u_resolution.x / max(1.0, u_resolution.y);

    // Camera
    vec3 ro = vec3(0.0, 0.0, 0.0);
    float base_speed = 1.25;
    float speed = base_speed * (0.75 + 1.65 * u_main);
    ro.z = u_time * speed;

    // Wider FOV than baseline.
    vec3 rd = normalize(vec3(uv, 1.25));

    vec2 rm = raymarch(ro, rd);
    float t = rm.x;
    float glow_accum = rm.y;

    if (t > MAX_DIST)
    {
        // Background: faint volumetric glow.
        float fog = exp(-u_fog * 0.055 * MAX_DIST);
        vec3 bg = glow_accum * fog * vec3(0.85, 0.92, 1.05);
        gl_FragColor = vec4(max(bg, 0.0), 1.0);
        return;
    }

    vec3 p = ro + rd * t;
    vec3 col = shade(p, rd, glow_accum);

    // Fog with distance.
    float fog = exp(-u_fog * 0.055 * t);
    col *= fog;

    gl_FragColor = vec4(max(col, 0.0), 1.0);
}
"""

# Bloom prefilter (threshold + soft knee). Downsamples into bloom buffer.
_FRAGMENT_PREFILTER = r"""
precision highp float;

varying vec2 v_uv;

uniform sampler2D u_scene;
uniform float u_threshold;
uniform float u_soft_knee;

vec3 prefilter(vec3 c)
{
    // Soft knee thresholding (common bloom trick).
    float br = max(max(c.r, c.g), c.b);
    float knee = u_threshold * u_soft_knee + 1e-5;
    float soft = clamp((br - u_threshold + knee) / (2.0 * knee), 0.0, 1.0);
    float contrib = max(br - u_threshold, 0.0) + soft * soft * knee;
    return c * (contrib / max(br, 1e-5));
}

void main()
{
    vec3 c = texture2D(u_scene, v_uv).rgb;
    gl_FragColor = vec4(prefilter(c), 1.0);
}
"""

# Separable gaussian blur.
_FRAGMENT_BLUR = r"""
precision highp float;

varying vec2 v_uv;

uniform sampler2D u_tex;
uniform vec2 u_texel;     // 1/texture_size
uniform vec2 u_dir;       // (1,0) or (0,1)
uniform float u_radius;   // 0.5..6.0

void main()
{
    // 9-tap gaussian-ish blur. Radius scales offsets.
    vec2 d = u_dir * u_texel * u_radius;

    vec3 sum = vec3(0.0);
    sum += texture2D(u_tex, v_uv - 4.0 * d).rgb * 0.051;
    sum += texture2D(u_tex, v_uv - 3.0 * d).rgb * 0.091;
    sum += texture2D(u_tex, v_uv - 2.0 * d).rgb * 0.122;
    sum += texture2D(u_tex, v_uv - 1.0 * d).rgb * 0.153;
    sum += texture2D(u_tex, v_uv).rgb          * 0.163;
    sum += texture2D(u_tex, v_uv + 1.0 * d).rgb * 0.153;
    sum += texture2D(u_tex, v_uv + 2.0 * d).rgb * 0.122;
    sum += texture2D(u_tex, v_uv + 3.0 * d).rgb * 0.091;
    sum += texture2D(u_tex, v_uv + 4.0 * d).rgb * 0.051;

    gl_FragColor = vec4(sum, 1.0);
}
"""

# Final combine: chromatic aberration + bloom + tone mapping + scanlines/vignette.
_FRAGMENT_COMBINE = r"""
precision highp float;

varying vec2 v_uv;

uniform sampler2D u_scene;
uniform sampler2D u_bloom;

uniform vec2  u_resolution;
uniform float u_exposure;
uniform float u_contrast;
uniform float u_audio_exposure_gain;
uniform float u_audio_contrast_gain;
uniform float u_bloom_intensity;
uniform float u_chroma;
uniform float u_scanlines;
uniform float u_vignette;
uniform float u_main;

vec3 tonemap_aces(vec3 x)
{
    // ACES-like approximation.
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

float rand(vec2 p)
{
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main()
{
    // Post chromatic aberration (texture coord warp for R/B).
    vec2 uv = v_uv;
    vec2 centered = uv * 2.0 - 1.0;
    centered.x *= u_resolution.x / max(1.0, u_resolution.y);

    // Single-input: chroma can be pushed slightly by the same energy.
    float chroma = clamp(u_chroma * (1.0 + 0.25 * u_main), 0.0, 0.03);
    vec2 off = chroma * vec2(centered.y, -centered.x);

    vec3 scene_r = texture2D(u_scene, uv + off).rgb;
    vec3 scene_g = texture2D(u_scene, uv).rgb;
    vec3 scene_b = texture2D(u_scene, uv - off).rgb;
    vec3 scene = vec3(scene_r.r, scene_g.g, scene_b.b);

    vec3 bloom = texture2D(u_bloom, uv).rgb;

    // Aggressive bloom: intensity also slightly driven by main.
    float bloom_i = u_bloom_intensity * (0.85 + 0.35 * u_main);
    vec3 col = scene + bloom_i * bloom;

    // Scanlines (subtle)
    if (u_scanlines > 0.001)
    {
        float s = sin(uv.y * u_resolution.y * 3.14159);
        float scan = mix(1.0, 0.92 + 0.08 * s, clamp(u_scanlines, 0.0, 1.0));
        col *= scan;
    }

    // Vignette
    if (u_vignette > 0.001)
    {
        float r = dot(centered, centered);
        float vig = smoothstep(1.15, 0.15, r);
        col *= mix(1.0, vig, clamp(u_vignette, 0.0, 1.0));
    }

    // Add a tiny animated grain for energy (main-driven).
    float g = (rand(uv + vec2(u_main, u_main * 1.37)) - 0.5) * 0.012 * (0.5 + u_main);
    col += g;

    // Exposure + contrast (optional audio modulation)
    float expo = max(0.05, u_exposure * (1.0 + u_audio_exposure_gain * u_main));
    float cont = max(0.65, u_contrast * (1.0 + u_audio_contrast_gain * u_main));
    col *= expo;
    col = tonemap_aces(col);
    col = pow(col, vec3(0.90 / cont));

    // Gamma
    col = pow(max(col, 0.0), vec3(0.4545));

    gl_FragColor = vec4(col, 1.0);
}
"""


# ---------------------------------------------------------------------
# VisPy canvas
# ---------------------------------------------------------------------
if HAVE_VISPY:

    class _AggressiveSDFTunnelCanvas(app.Canvas):
        """VisPy canvas using a multi-pass pipeline (scene -> bloom -> combine)."""

        def __init__(self, config: Dict[str, Any]) -> None:
            self._config = config
            self._audio = _AudioState()

            super().__init__(keys=None, size=(640, 360), show=False)

            # Programs
            self._prog_scene = gloo.Program(_VERTEX, _FRAGMENT_SCENE)
            self._prog_prefilter = gloo.Program(_VERTEX, _FRAGMENT_PREFILTER)
            self._prog_blur = gloo.Program(_VERTEX, _FRAGMENT_BLUR)
            self._prog_combine = gloo.Program(_VERTEX, _FRAGMENT_COMBINE)

            verts = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
            if HAVE_NUMPY:
                v = np.array(verts, dtype="float32")
            else:
                v = verts

            vb = gloo.VertexBuffer(v)
            self._prog_scene["a_position"] = vb
            self._prog_prefilter["a_position"] = vb
            self._prog_blur["a_position"] = vb
            self._prog_combine["a_position"] = vb

            gloo.set_state(blend=False, depth_test=False, cull_face=False)

            # Framebuffer resources (created lazily on first draw/resize)
            self._scene_tex = None
            self._scene_fbo = None
            self._bloom_tex0 = None
            self._bloom_tex1 = None
            self._bloom_fbo0 = None
            self._bloom_fbo1 = None

            self._last_sizes = (-1, -1, -1)  # (w, h, downsample)

        # -----------------------------
        # Resource management
        # -----------------------------
        def _ensure_fbos(self) -> None:
            w, h = self.physical_size
            down = int(self._config.get("bloom_downsample", 2))
            down = int(_clamp(float(down), 1.0, 4.0))

            if (w, h, down) == self._last_sizes and self._scene_tex is not None:
                return

            self._last_sizes = (w, h, down)

            # Full-res scene
            self._scene_tex = self._make_texture(w, h)
            self._scene_fbo = gloo.FrameBuffer(color=self._scene_tex)

            # Downsampled bloom buffers
            bw = max(16, w // down)
            bh = max(16, h // down)
            self._bloom_tex0 = self._make_texture(bw, bh)
            self._bloom_tex1 = self._make_texture(bw, bh)
            self._bloom_fbo0 = gloo.FrameBuffer(color=self._bloom_tex0)
            self._bloom_fbo1 = gloo.FrameBuffer(color=self._bloom_tex1)

        @staticmethod
        def _make_texture(w: int, h: int):
            """Create a float texture when possible, fallback to uint8."""

            if HAVE_NUMPY:
                try:
                    data = np.zeros((h, w, 4), dtype=np.float32)
                    tex = gloo.Texture2D(data, interpolation="linear")
                    return tex
                except Exception:
                    data = np.zeros((h, w, 4), dtype=np.uint8)
                    tex = gloo.Texture2D(data, interpolation="linear")
                    return tex
            # No numpy: use VisPy default path.
            tex = gloo.Texture2D((h, w, 4), interpolation="linear")
            return tex

        # -----------------------------
        # Uniform updates
        # -----------------------------
        def _update_uniforms(self) -> None:
            w, h = self.physical_size

            # Shared uniforms
            for p in (self._prog_scene, self._prog_combine):
                p["u_resolution"] = (float(w), float(h))

            # Scene
            self._prog_scene["u_time"] = float(self._audio.t)
            self._prog_scene["u_main"] = float(self._audio.energy)

            preset = str(self._config.get("color_preset", _COLOR_PRESETS[0]))
            self._prog_scene["u_palette_id"] = float(_COLOR_PRESET_TO_ID.get(preset, 0.0))

            radius = float(self._config.get("tunnel_radius", 1.55))
            ring_freq = float(self._config.get("ring_frequency", 6.0))
            twist = float(self._config.get("twist", 1.0))
            gate = float(self._config.get("gate_strength", 1.15))
            fog = float(self._config.get("fog", 1.0))
            steps = float(self._config.get("detail_steps", 120))
            glow_gain = float(self._config.get("glow_gain", 1.35))

            self._prog_scene["u_tunnel_radius"] = float(_clamp(radius, 0.2, 6.0))
            self._prog_scene["u_ring_frequency"] = float(_clamp(ring_freq, 0.1, 24.0))
            self._prog_scene["u_twist"] = float(_clamp(twist, -8.0, 8.0))
            self._prog_scene["u_gate_strength"] = float(_clamp(gate, 0.0, 2.5))
            self._prog_scene["u_fog"] = float(_clamp(fog, 0.0, 3.0))
            self._prog_scene["u_detail_steps"] = float(_clamp(steps, 32.0, 160.0))
            self._prog_scene["u_glow_gain"] = float(_clamp(glow_gain, 0.2, 3.0))

            # Prefilter
            thr = float(self._config.get("bloom_threshold", 0.65))
            knee = float(self._config.get("bloom_soft_knee", 0.6))
            self._prog_prefilter["u_threshold"] = float(_clamp(thr, 0.0, 2.0))
            self._prog_prefilter["u_soft_knee"] = float(_clamp(knee, 0.0, 1.0))

            # Blur
            blur_radius = float(self._config.get("bloom_radius", 2.6))
            self._prog_blur["u_radius"] = float(_clamp(blur_radius, 0.5, 6.0))

            # Combine
            exp_ = float(self._config.get("exposure", 1.35))
            contrast = float(self._config.get("contrast", 1.10))
            bloom_i = float(self._config.get("bloom", 1.40))
            chroma = float(self._config.get("chromatic_aberration", 0.012))
            scan = float(self._config.get("scanlines", 0.25))
            vig = float(self._config.get("vignette", 0.35))

            self._prog_combine["u_exposure"] = float(_clamp(exp_, 0.1, 3.0))
            self._prog_combine["u_contrast"] = float(_clamp(contrast, 0.6, 1.8))
            self._prog_combine["u_bloom_intensity"] = float(_clamp(bloom_i, 0.0, 3.0))
            self._prog_combine["u_chroma"] = float(_clamp(chroma, 0.0, 0.03))
            self._prog_combine["u_scanlines"] = float(_clamp(scan, 0.0, 1.0))
            self._prog_combine["u_vignette"] = float(_clamp(vig, 0.0, 1.0))
            self._prog_combine["u_main"] = float(self._audio.energy)

            # Optional audio -> exposure/contrast modulation (disabled by default).
            audio_expo_gain = float(self._config.get("audio_exposure_gain", 0.0))
            audio_contrast_gain = float(self._config.get("audio_contrast_gain", 0.0))
            self._prog_combine["u_audio_exposure_gain"] = float(_clamp(audio_expo_gain, 0.0, 2.0))
            self._prog_combine["u_audio_contrast_gain"] = float(_clamp(audio_contrast_gain, 0.0, 2.0))

        # -----------------------------
        # Public hooks
        # -----------------------------
        def set_audio_state(self, *, time_s: float, energy: float) -> None:
            """Update smoothed audio state and request a redraw."""

            # Slightly faster smoothing than the baseline plugin.
            a = 0.33
            self._audio.t = float(time_s)
            self._audio.energy = (1.0 - a) * self._audio.energy + a * float(_clamp(energy, 0.0, 2.0))
            self.update()

        # -----------------------------
        # VisPy callbacks
        # -----------------------------
        def on_resize(self, event) -> None:  # type: ignore[override]
            gloo.set_viewport(0, 0, *event.physical_size)
            # Resources are rebuilt lazily.
            self.update()

        def on_draw(self, event) -> None:  # type: ignore[override]
            self._ensure_fbos()
            assert self._scene_fbo is not None
            assert self._bloom_fbo0 is not None and self._bloom_fbo1 is not None
            assert self._scene_tex is not None
            assert self._bloom_tex0 is not None and self._bloom_tex1 is not None

            self._update_uniforms()

            w, h = self.physical_size

            # 1) Render scene to full-res texture
            with self._scene_fbo:
                gloo.set_viewport(0, 0, w, h)
                gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                self._prog_scene.draw("triangle_strip")

            # 2) Prefilter + downsample into bloom_tex0
            bw, bh = self._bloom_tex0.shape[1], self._bloom_tex0.shape[0]
            self._prog_prefilter["u_scene"] = self._scene_tex
            with self._bloom_fbo0:
                gloo.set_viewport(0, 0, bw, bh)
                gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                self._prog_prefilter.draw("triangle_strip")

            # 3) Blur horizontal: bloom0 -> bloom1
            self._prog_blur["u_tex"] = self._bloom_tex0
            self._prog_blur["u_texel"] = (1.0 / float(bw), 1.0 / float(bh))
            self._prog_blur["u_dir"] = (1.0, 0.0)
            with self._bloom_fbo1:
                gloo.set_viewport(0, 0, bw, bh)
                gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                self._prog_blur.draw("triangle_strip")

            # 4) Blur vertical: bloom1 -> bloom0
            self._prog_blur["u_tex"] = self._bloom_tex1
            self._prog_blur["u_dir"] = (0.0, 1.0)
            with self._bloom_fbo0:
                gloo.set_viewport(0, 0, bw, bh)
                gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                self._prog_blur.draw("triangle_strip")

            # 5) Combine to screen
            gloo.set_viewport(0, 0, w, h)
            gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
            self._prog_combine["u_scene"] = self._scene_tex
            self._prog_combine["u_bloom"] = self._bloom_tex0
            self._prog_combine.draw("triangle_strip")


# ---------------------------------------------------------------------
# Widget wrapper (Qt)
# ---------------------------------------------------------------------
class _AggressiveSDFTunnelWidget(QWidget):
    def __init__(self, config: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._canvas = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not HAVE_VISPY:
            label = QLabel(
                "VisPy is not available.\n"
                "SDF Neon Tunnel (Aggressive) is disabled.\n"
                "Please install 'vispy' to enable this visualization.",
                self,
            )
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            return

        app.use_app("pyqt6")  # type: ignore[arg-type]

        self._canvas = _AggressiveSDFTunnelCanvas(config=self._config)  # type: ignore[name-defined]
        self._canvas.native.setParent(self)
        layout.addWidget(self._canvas.native)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(640, 360)

    def update_audio_state(self, *, time_ms: int, energy: float) -> None:
        if self._canvas is None:
            return

        # Single-input mode: we only drive the forward motion (and optionally exposure/contrast)
        # from one energy scalar.
        self._canvas.set_audio_state(
            time_s=float(time_ms) / 1000.0,
            energy=float(energy),
        )


# ---------------------------------------------------------------------
# Olaf plugin class
# ---------------------------------------------------------------------
class SDFRaymarchedNeonTunnelAggressiveVisualization(BaseVisualization):
    plugin_id = "sdf_neon_tunnel_aggressive"
    plugin_name = "SDF Raymarched Neon Tunnel (Aggressive)"
    plugin_description = (
        "Aggressive SDF raymarched neon tunnel with multi-pass bloom, post chromatic aberration, "
        "optional scanlines/vignette, and stronger audio reactivity."
    )
    plugin_author = "DrDLP"
    plugin_version = "1.1.2"
    plugin_max_inputs = 1

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._widget: Optional[_AggressiveSDFTunnelWidget] = None

        # Defaults (only if missing so saved project state can override)
        self.config.setdefault("tunnel_radius", 4.05)
        self.config.setdefault("ring_frequency", 96.0)
        self.config.setdefault("twist", 8)
        self.config.setdefault("gate_strength", 0.15)
        self.config.setdefault("fog", 3.0)
        self.config.setdefault("color_preset", _COLOR_PRESETS[0])

        # Aggressive post-process defaults
        self.config.setdefault("bloom", 1.95)
        self.config.setdefault("bloom_radius", 2.6)
        self.config.setdefault("bloom_threshold", 1.15)
        self.config.setdefault("bloom_soft_knee", 0.6)
        self.config.setdefault("bloom_downsample", 2)

        self.config.setdefault("chromatic_aberration", 0.003)
        self.config.setdefault("exposure", 1.35)
        self.config.setdefault("contrast", 1.10)
        self.config.setdefault("audio_exposure_gain", 0.0)
        self.config.setdefault("audio_contrast_gain", 0.0)
        self.config.setdefault("scanlines",2)
        self.config.setdefault("vignette", 0.35)
        self.config.setdefault("detail_steps", 160)
        self.config.setdefault("glow_gain", 2)

    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        return {
            # Core geometry
            "tunnel_radius": PluginParameter(
                name="tunnel_radius",
                label="Tunnel radius",
                type="float",
                default=4.05,
                minimum=0.2,
                maximum=6.0,
                step=0.01,
                description="Base radius of the tunnel (before drum gates).",
            ),
            "ring_frequency": PluginParameter(
                name="ring_frequency",
                label="Ring frequency",
                type="float",
                default=96.0,
                minimum=0.1,
                maximum=96.0,
                step=0.5,
                description="How often gates/rings repeat along the tunnel (higher = more rings).",
            ),
            "twist": PluginParameter(
                name="twist",
                label="Twist",
                type="float",
                default=8,
                minimum=-16.0,
                maximum=16.0,
                step=0.05,
                description="Rotational twist of the tunnel as depth increases.",
            ),
            "gate_strength": PluginParameter(
                name="gate_strength",
                label="Gate strength",
                type="float",
                default=0.15,
                minimum=0.0,
                maximum=1.5,
                step=0.05,
                description="How strongly the tunnel constricts at the repeating gates/rings.",
            ),
            "fog": PluginParameter(
                name="fog",
                label="Fog",
                type="float",
                default=3.0,
                minimum=0.0,
                maximum=3.0,
                step=0.05,
                description="Distance fog in the tunnel (higher = fades faster).",
            ),

            # Bloom
            "bloom": PluginParameter(
                name="bloom",
                label="Bloom intensity",
                type="float",
                default=1.95,
                minimum=0.0,
                maximum=3.0,
                step=0.05,
                description="Bloom contribution in the final combine (multi-pass).",
            ),
            "bloom_radius": PluginParameter(
                name="bloom_radius",
                label="Bloom radius",
                type="float",
                default=2.6,
                minimum=0.5,
                maximum=6.0,
                step=0.05,
                description="Blur radius for bloom (higher = softer and more aggressive glow).",
            ),
            "bloom_threshold": PluginParameter(
                name="bloom_threshold",
                label="Bloom threshold",
                type="float",
                default=1.15,
                minimum=0.0,
                maximum=2.0,
                step=0.01,
                description="Only bright parts above this threshold contribute to bloom.",
            ),
            "bloom_soft_knee": PluginParameter(
                name="bloom_soft_knee",
                label="Bloom soft knee",
                type="float",
                default=0.6,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="Soft transition around the bloom threshold.",
            ),
            "bloom_downsample": PluginParameter(
                name="bloom_downsample",
                label="Bloom downsample",
                type="int",
                default=2,
                minimum=1,
                maximum=4,
                step=1,
                description="Downsampling factor for bloom buffers (2 is a good default).",
            ),

            # Post-process look
            "color_preset": PluginParameter(
                name="color_preset",
                label="Color preset",
                type="enum",
                default="Rainbow",
                choices=_COLOR_PRESETS,
                description="Select a neon palette preset used by the tunnel shader.",
            ),

            "chromatic_aberration": PluginParameter(
                name="chromatic_aberration",
                label="Chromatic aberration",
                type="float",
                default=0.3,
                minimum=0.0,
                maximum=0.3,
                step=0.005,
                description="RGB split amount applied as a post-process warp.",
            ),
            "exposure": PluginParameter(
                name="exposure",
                label="Exposure",
                type="float",
                default=1.35,
                minimum=0.1,
                maximum=3.0,
                step=0.01,
                description="Overall brightness before tone mapping.",
            ),
            "contrast": PluginParameter(
                name="contrast",
                label="Contrast",
                type="float",
                default=1.10,
                minimum=0.6,
                maximum=1.8,
                step=0.01,
                description="Contrast shaping in the final pass.",
            ),
            "audio_exposure_gain": PluginParameter(
                name="audio_exposure_gain",
                label="Audio exposure gain",
                type="float",
                default=0.3,
                minimum=0.0,
                maximum=4.0,
                step=0.1,
                description="How much audio energy increases exposure (0 = disabled).",
            ),
            "audio_contrast_gain": PluginParameter(
                name="audio_contrast_gain",
                label="Audio contrast gain",
                type="float",
                default=0.3,
                minimum=0.0,
                maximum=4.0,
                step=0.1,
                description="How much audio energy increases contrast (0 = disabled).",
            ),
            "scanlines": PluginParameter(
                name="scanlines",
                label="Scanlines",
                type="float",
                default=2,
                minimum=0.0,
                maximum=2.0,
                step=0.01,
                description="CRT-like scanlines (subtle by default).",
            ),
            "vignette": PluginParameter(
                name="vignette",
                label="Vignette",
                type="float",
                default=0.35,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="Darken edges to focus the center.",
            ),

            # Raymarch quality
            "detail_steps": PluginParameter(
                name="detail_steps",
                label="Detail steps",
                type="int",
                default=160,
                minimum=32,
                maximum=320,
                step=1,
                description="Maximum raymarch steps (higher = more detail, slower).",
            ),
            "glow_gain": PluginParameter(
                name="glow_gain",
                label="Glow gain",
                type="float",
                default=2,
                minimum=0.2,
                maximum=4.0,
                step=0.01,
                description="Volumetric glow accumulation gain inside the raymarch.",
            ),
        }

    def create_widget(self, parent: Optional[QWidget] = None) -> QWidget:
        self._widget = _AggressiveSDFTunnelWidget(config=self.config, parent=parent)
        return self._widget

    def on_audio_features(self, features: Dict[str, Any]) -> None:
        if self._widget is None:
            return

        time_ms = int(features.get("time_ms", 0))
        inputs = features.get("inputs", {}) or {}

        # Routing:
        #   input_1 = SINGLE INPUT (full mix or selected stem)
        energy = float((inputs.get("input_1", {}) or {}).get("rms", 0.0))

        self._widget.update_audio_state(time_ms=time_ms, energy=energy)
