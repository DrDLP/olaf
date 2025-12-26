
"""sdf_galaxy_system.py

Olaf Visualization Plugin (VisPy / Gloo) - SDF / Volumetric Galaxy System

Spec
----
- Entire galaxy rotates in sync with bass frequencies.
- Spiral warp effect intensifies with overall energy.
- Over time, spiral arms stretch into a vertical helix along the Y-axis.
- Treble spikes trigger accretion beam effects bursting out from the core.
- Camera orbit subtly shifts the color hue across the scene.
- No background starfield.

Implementation
--------------
- Procedural volumetric field (spiral arms + core + beams) rendered by raymarching.
- Multi-pass bloom + tone mapping + optional scanlines/vignette (same pipeline style as sdf_neon_tunnel).
- Single audio input: input_1 (full mix or chosen stem). Bass/treble are extracted if available; otherwise fall back to RMS.

License: same as project
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

try:
    from olaf_app.visualization_api import BaseVisualization, PluginParameter
except Exception:
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


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _extract_band_energy(input_dict: Dict[str, Any], keys: tuple[str, ...]) -> Optional[float]:
    """
    Tries common shapes:
      - input_1["bass"] or input_1["low"] etc.
      - input_1["bands"] / ["band_energies"] dict, where keys may be "bass"/"low"/"high"/"treble"/...
    """
    # Direct keys
    for k in keys:
        if k in input_dict:
            v = _safe_float(input_dict.get(k, 0.0), 0.0)
            return v

    # Nested dicts
    for container_key in ("bands", "band_energies", "eq", "spectrum_bands"):
        bd = input_dict.get(container_key)
        if isinstance(bd, dict):
            for k in keys:
                if k in bd:
                    return _safe_float(bd.get(k, 0.0), 0.0)

            # Some hosts use "low"/"mid"/"high" only
            # (handled by providing those keys in 'keys' tuple)
    return None


@dataclass
class _AudioState:
    """Smoothed audio state used as shader uniforms."""
    t: float = 0.0
    energy: float = 0.0
    bass: float = 0.0
    treble: float = 0.0
    treble_spike: float = 0.0
    _last_spike_t: float = -999.0
    _prev_treble: float = 0.0


# ---------------------------------------------------------------------
# Color presets (copied to match the tunnel plugin style)
# ---------------------------------------------------------------------
_COLOR_PRESETS = [
    "Cyberpunk (Cyan/Magenta)",
    "Synthwave (Pink/Orange)",
    "Matrix (Green)",
    "Electric (Blue/Purple)",
    "Acid (Yellow/Cyan)",
    "Infrared (Red/Pink)",
    "Rainbow",
]
_COLOR_PRESET_TO_ID = {name: float(i) for i, name in enumerate(_COLOR_PRESETS)}


# ---------------------------------------------------------------------
# Scene presets (multi-parameter "looks")
# ---------------------------------------------------------------------
# These are meant to reduce manual tweaking: choose one, then fine-tune.
# Presets are applied ONCE when the selection changes (so you can tweak afterward).
SCENE_PRESETS: Dict[str, Dict[str, Any]] = {
    # Note: "Custom" intentionally does nothing.
    "Balanced": {
        # Keep close to defaults; slightly smoother + readable arms.
        "arms": 3,
        "disk_radius": 3.6,
        "disk_thickness": 0.50,
        "core_radius": 0.55,
        "spiral_tightness": 1.35,
        "arm_width": 7.2,
        "arm_power": 4.6,
        "base_rotation": 0.55,
        "bass_rotation_gain": 3.2,
        "warp_strength": 0.55,
        "warp_speed": 1.75,
        "warp_energy_gain": 1.35,
        "helix_growth": 0.018,
        "helix_height": 1.55,
        "helix_pitch": 1.10,
        "helix_twist_speed": 0.65,
        "beam_strength": 2.25,
        "beam_length": 6.0,
        "beam_radius": 0.25,
        "treble_spike_threshold": 0.22,
        "treble_spike_min": 0.18,
        "treble_spike_cooldown_s": 0.12,
        "treble_spike_decay": 0.88,
        "camera_yaw": 0.85,
        "camera_pitch": 0.62,
        "camera_distance": 4.1,
        "fov": 1.55,
        "camera_orbit_amount": 0.14,
        "camera_orbit_speed": 0.18,
        "hue_orbit_amount": 0.10,
        "fog": 1.15,
        "detail_steps": 120,
        "glow_gain": 1.35,
        "bloom": 1.55,
        "bloom_radius": 2.4,
        "bloom_threshold": 0.55,
        "bloom_soft_knee": 0.35,
        "bloom_downsample": 2,
        "chromatic_aberration": 0.010,
        "exposure": 1.00,
        "contrast": 1.08,
        "scanlines": 0.0,
        "vignette": 0.38,
    },
    "Cinematic Wide": {
        "disk_radius": 3.9,
        "disk_thickness": 0.46,
        "arm_width": 6.8,
        "arm_power": 4.3,
        "warp_strength": 0.48,
        "warp_speed": 1.45,
        "warp_energy_gain": 1.15,
        "camera_yaw": 0.75,
        "camera_pitch": 0.58,
        "camera_distance": 5.0,
        "fov": 1.35,
        "camera_orbit_amount": 0.12,
        "camera_orbit_speed": 0.10,
        "hue_orbit_amount": 0.08,
        "fog": 0.92,
        "detail_steps": 130,
        "glow_gain": 1.22,
        "bloom": 1.25,
        "bloom_radius": 2.2,
        "bloom_threshold": 0.58,
        "chromatic_aberration": 0.006,
        "exposure": 1.05,
        "contrast": 1.10,
        "vignette": 0.45,
        "scanlines": 0.0,
    },
    "Bass Spin": {
        "base_rotation": 0.30,
        "bass_rotation_gain": 6.8,
        "spiral_tightness": 1.55,
        "arm_width": 6.4,
        "arm_power": 4.9,
        "warp_strength": 0.70,
        "warp_speed": 2.05,
        "warp_energy_gain": 2.10,
        "helix_growth": 0.016,
        "camera_orbit_amount": 0.18,
        "camera_orbit_speed": 0.24,
        "hue_orbit_amount": 0.11,
        # Fewer beams so bass stays the "driver"
        "treble_spike_threshold": 0.28,
        "treble_spike_min": 0.22,
        "beam_strength": 1.75,
        "beam_length": 5.4,
        "beam_radius": 0.22,
        "fog": 1.10,
        "glow_gain": 1.35,
        "bloom": 1.55,
        "chromatic_aberration": 0.010,
        "contrast": 1.12,
    },
    "Warp Storm": {
        "warp_strength": 1.25,
        "warp_speed": 2.90,
        "warp_energy_gain": 3.00,
        "spiral_tightness": 1.15,
        "arm_width": 7.6,
        "arm_power": 4.8,
        "fog": 1.55,
        "detail_steps": 120,
        "glow_gain": 1.60,
        "bloom": 1.95,
        "bloom_radius": 3.20,
        "bloom_threshold": 0.50,
        "bloom_soft_knee": 0.42,
        "chromatic_aberration": 0.012,
        "exposure": 1.10,
        "contrast": 1.16,
        "camera_orbit_amount": 0.16,
        "camera_orbit_speed": 0.20,
        "hue_orbit_amount": 0.12,
        "vignette": 0.40,
    },
    "Helix Lift": {
        "helix_growth": 0.030,
        "helix_height": 2.40,
        "helix_pitch": 1.55,
        "helix_twist_speed": 1.00,
        "disk_thickness": 0.44,
        "disk_radius": 3.5,
        "warp_strength": 0.60,
        "warp_speed": 1.65,
        "warp_energy_gain": 1.45,
        "camera_orbit_amount": 0.10,
        "camera_orbit_speed": 0.12,
        "hue_orbit_amount": 0.10,
        "fog": 1.05,
        "glow_gain": 1.35,
        "bloom": 1.45,
        "chromatic_aberration": 0.009,
        "contrast": 1.10,
    },
    "Treble Beams": {
        "beam_strength": 3.80,
        "beam_length": 9.50,
        "beam_radius": 0.22,
        "treble_spike_threshold": 0.16,
        "treble_spike_min": 0.14,
        "treble_spike_cooldown_s": 0.08,
        "treble_spike_decay": 0.86,
        "warp_strength": 0.58,
        "warp_speed": 1.85,
        "warp_energy_gain": 1.55,
        "glow_gain": 1.55,
        "bloom": 1.85,
        "bloom_radius": 2.85,
        "bloom_threshold": 0.50,
        "chromatic_aberration": 0.011,
        "exposure": 1.05,
        "contrast": 1.12,
        "fog": 1.10,
        "camera_orbit_amount": 0.14,
        "camera_orbit_speed": 0.18,
        "hue_orbit_amount": 0.13,
    },
    "Clean Minimal": {
        "warp_strength": 0.45,
        "warp_speed": 1.55,
        "warp_energy_gain": 1.10,
        "arm_width": 7.0,
        "arm_power": 4.2,
        "fog": 0.75,
        "detail_steps": 140,
        "glow_gain": 1.10,
        "bloom": 0.75,
        "bloom_radius": 1.60,
        "bloom_threshold": 0.62,
        "chromatic_aberration": 0.003,
        "exposure": 1.00,
        "contrast": 1.20,
        "scanlines": 0.0,
        "vignette": 0.25,
        "camera_orbit_amount": 0.08,
        "camera_orbit_speed": 0.10,
        "hue_orbit_amount": 0.05,
    },
}

SCENE_PRESET_NAMES = ["Custom"] + list(SCENE_PRESETS.keys())



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

# Scene fragment shader (linear output).
_FRAGMENT_SCENE = r"""
precision highp float;

varying vec2 v_uv;

uniform vec2  u_resolution;
uniform float u_time;

uniform float u_energy;
uniform float u_bass;
uniform float u_treble;
uniform float u_treble_spike;

uniform float u_palette_id;

uniform float u_arms;
uniform float u_disk_radius;
uniform float u_disk_thickness;
uniform float u_core_radius;

uniform float u_base_rot;
uniform float u_bass_rot_gain;

uniform float u_spiral_tightness;
uniform float u_arm_width;
uniform float u_arm_power;

uniform float u_warp_strength;
uniform float u_warp_speed;
uniform float u_warp_energy_gain;

uniform float u_helix_amount;
uniform float u_helix_height;
uniform float u_helix_pitch;
uniform float u_helix_twist_speed;

uniform float u_beam_strength;
uniform float u_beam_length;
uniform float u_beam_radius;

uniform float u_cam_yaw;
uniform float u_cam_pitch;
uniform float u_cam_dist;
uniform float u_fov;

uniform float u_orbit_amount;
uniform float u_orbit_speed;
uniform float u_hue_orbit_amount;

uniform float u_fog;
uniform float u_detail_steps;
uniform float u_glow_gain;

#define MAX_STEPS 140
#define MAX_DIST  24.0

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
        vec3 a = vec3(0.00, 1.00, 1.00);
        vec3 b = vec3(1.00, 0.00, 1.00);
        vec3 d = mix(a, b, s1);
        c = mix(d, vec3(0.15, 0.50, 1.00), 0.35 * s2);
    }
    else if (preset < 1.5)
    {
        vec3 a = vec3(1.00, 0.10, 0.80);
        vec3 b = vec3(1.00, 0.55, 0.10);
        c = mix(a, b, s1);
        c = mix(c, vec3(0.30, 0.95, 1.00), 0.25 * s3);
    }
    else if (preset < 2.5)
    {
        vec3 a = vec3(0.05, 0.35, 0.08);
        vec3 b = vec3(0.10, 1.00, 0.25);
        c = mix(a, b, s1);
    }
    else if (preset < 3.5)
    {
        vec3 a = vec3(0.10, 0.35, 1.00);
        vec3 b = vec3(0.75, 0.15, 1.00);
        c = mix(a, b, s1);
        c = mix(c, vec3(0.00, 1.00, 0.85), 0.20 * s2);
    }
    else if (preset < 4.5)
    {
        vec3 a = vec3(1.00, 0.95, 0.10);
        vec3 b = vec3(0.00, 1.00, 0.95);
        c = mix(a, b, s1);
    }
    else if (preset < 5.5)
    {
        vec3 a = vec3(1.00, 0.05, 0.15);
        vec3 b = vec3(1.00, 0.15, 0.90);
        c = mix(a, b, s1);
    }
    else
    {
        c = 0.55 + 0.45 * vec3(
            sin(6.2831 * (k + 0.00)),
            sin(6.2831 * (k + 0.33)),
            sin(6.2831 * (k + 0.66))
        );
    }

    return _sat(c, 1.35);
}

void camera_basis(in vec3 ro, in vec3 ta, out vec3 ww, out vec3 uu, out vec3 vv)
{
    ww = normalize(ta - ro);
    uu = normalize(cross(ww, vec3(0.0, 1.0, 0.0)));
    vv = cross(uu, ww);
}

float arms_density(vec3 p, float rot_angle)
{
    // Basic disk coordinates
    float r = length(p.xz);
    float y = p.y;

    // Disk envelope (no background stars)
    float disk = exp(- (r*r) / max(0.0001, u_disk_radius * u_disk_radius));
    float thick = exp(- (y*y) / max(0.0001, u_disk_thickness * u_disk_thickness));

    // Spiral arms
    float theta = atan(p.z, p.x); // -pi..pi
    float spiral = theta + u_spiral_tightness * log(r + 0.12) - rot_angle;

    float a = 0.5 + 0.5 * cos(u_arms * spiral);
    a = pow(max(a, 0.0), u_arm_power);

    // Arm width shaping
    float w = exp(-u_arm_width * (1.0 - a) * (1.0 - a));

    return disk * thick * w;
}

float core_density(vec3 p)
{
    float r = length(p);
    float c = exp(- (r*r) / max(0.0001, u_core_radius * u_core_radius));
    return c;
}

float beam_density(vec3 p)
{
    // Vertical accretion beams from core (driven by treble spikes)
    float r = length(p.xz);
    float core = exp(- (r*r) / max(0.0001, u_beam_radius * u_beam_radius));
    float len = exp(-abs(p.y) / max(0.0001, u_beam_length));
    return core * len;
}

vec3 shade_field(vec3 p, float orbit_phase, float rot_angle)
{
    // Warp (energy driven): swirl the sample space slightly.
    float warp = u_warp_strength * (1.0 + u_warp_energy_gain * u_energy);
    float w = warp * sin(u_warp_speed * u_time + 1.7 * length(p.xz) + 0.8 * p.y);
    p.xz = rot(w) * p.xz;

    // Helix progression: lift arms into Y over time.
    float theta = atan(p.z, p.x);
    float r = length(p.xz);
    float hel = u_helix_amount;
    float hel_y = hel * u_helix_height * sin(u_arms * theta + u_helix_pitch * r + u_helix_twist_speed * u_time);
    p.y += hel_y;

    // Densities
    float d_arm = arms_density(p, rot_angle);
    float d_core = core_density(p);
    float d_beam = beam_density(p) * u_treble_spike;

    // Composite density
    float density = 1.15 * d_arm + 1.45 * d_core + u_beam_strength * d_beam;

    // Color key: orbit hue shift
    float hue_shift = u_hue_orbit_amount * sin(orbit_phase);

    // Palette coordinate
    float k = 0.12 * r + 0.18 * (theta / 6.2831) + 0.06 * u_time + hue_shift;

    vec3 base = neon_palette(k, u_palette_id);

    // Core warms slightly with treble presence
    vec3 core_tint = mix(base, vec3(1.0, 0.85, 0.95), clamp(u_treble * 0.85, 0.0, 1.0));

    // Beams are brighter / whiter
    vec3 beam_tint = mix(core_tint, vec3(1.0), 0.55);

    vec3 col = core_tint * (0.65 * d_arm + 1.35 * d_core) + beam_tint * (1.75 * d_beam);

    // Energy-driven gain
    col *= (0.70 + 1.30 * u_energy);

    // Density -> emissive scaling
    col *= density;

    // Extra glow accumulator input
    col += u_glow_gain * 0.08 * density * core_tint;

    return col;
}

void main()
{
    vec2 fc = v_uv * u_resolution;

    vec2 uv = (fc / u_resolution) * 2.0 - 1.0;
    uv.x *= u_resolution.x / max(1.0, u_resolution.y);

    // Camera (static params + subtle orbit)
    float orbit_phase = u_orbit_speed * u_time;
    float yaw = u_cam_yaw + u_orbit_amount * sin(orbit_phase);
    float pitch = u_cam_pitch + 0.45 * u_orbit_amount * cos(orbit_phase);

    // Spherical camera position
    vec3 ro = vec3(
        u_cam_dist * cos(pitch) * sin(yaw),
        u_cam_dist * sin(pitch),
        u_cam_dist * cos(pitch) * cos(yaw)
    );

    vec3 ta = vec3(0.0, 0.0, 0.0);

    vec3 ww, uu, vv;
    camera_basis(ro, ta, ww, uu, vv);

    float fov = max(0.35, u_fov);
    vec3 rd = normalize(uu * uv.x + vv * uv.y + ww * fov);

    // Rotation (bass-synced)
    float rot_speed = u_base_rot + u_bass_rot_gain * u_bass;
    float rot_angle = rot_speed * u_time;

    // Raymarch volumetric field
    float t = 0.0;
    vec3 accum = vec3(0.0);
    float alpha = 0.0;

    int steps_cap = int(clamp(u_detail_steps, 32.0, float(MAX_STEPS)));

    // Step length adapts a bit with energy (more energy -> more "thickness")
    float dt = 0.12 + 0.08 * clamp(u_energy, 0.0, 1.5);

    for (int i = 0; i < MAX_STEPS; i++)
    {
        if (i >= steps_cap) break;

        vec3 p = ro + rd * t;

        // Stop if too far
        if (t > MAX_DIST) break;

        vec3 col = shade_field(p, orbit_phase, rot_angle);

        // Convert emissive to alpha via luminance-ish proxy
        float a = clamp(dot(col, vec3(0.2126, 0.7152, 0.0722)) * 0.55, 0.0, 1.0);

        // Front-to-back composition
        float w = (1.0 - alpha) * a;
        accum += w * col;
        alpha += w;

        // March forward
        t += dt;

        // Early exit
        if (alpha > 0.98) break;
    }

    // Fog (keeps edges clean; background stays black)
    float fog = exp(-u_fog * 0.06 * t);
    accum *= fog;

    gl_FragColor = vec4(max(accum, 0.0), 1.0);
}
"""

# Bloom prefilter / blur / combine are re-used (same style as tunnel plugin)
_FRAGMENT_PREFILTER = r"""
precision highp float;

varying vec2 v_uv;

uniform sampler2D u_scene;
uniform float u_threshold;
uniform float u_soft_knee;

vec3 prefilter(vec3 c)
{
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

_FRAGMENT_BLUR = r"""
precision highp float;

varying vec2 v_uv;

uniform sampler2D u_tex;
uniform vec2 u_texel;
uniform vec2 u_dir;
uniform float u_radius;

void main()
{
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

_FRAGMENT_COMBINE = r"""
precision highp float;

varying vec2 v_uv;

uniform sampler2D u_scene;
uniform sampler2D u_bloom;

uniform vec2  u_resolution;
uniform float u_exposure;
uniform float u_contrast;
uniform float u_bloom_intensity;
uniform float u_chroma;
uniform float u_scanlines;
uniform float u_vignette;

uniform float u_energy;

vec3 tonemap_aces(vec3 x)
{
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
    vec2 uv = v_uv;
    vec2 centered = uv * 2.0 - 1.0;
    centered.x *= u_resolution.x / max(1.0, u_resolution.y);

    float chroma = clamp(u_chroma * (1.0 + 0.15 * u_energy), 0.0, 0.03);
    vec2 off = chroma * vec2(centered.y, -centered.x);

    vec3 scene_r = texture2D(u_scene, uv + off).rgb;
    vec3 scene_g = texture2D(u_scene, uv).rgb;
    vec3 scene_b = texture2D(u_scene, uv - off).rgb;
    vec3 scene = vec3(scene_r.r, scene_g.g, scene_b.b);

    vec3 bloom = texture2D(u_bloom, uv).rgb;

    float bloom_i = u_bloom_intensity * (0.85 + 0.35 * u_energy);
    vec3 col = scene + bloom_i * bloom;

    if (u_scanlines > 0.001)
    {
        float s = sin(uv.y * u_resolution.y * 3.14159);
        float scan = mix(1.0, 0.92 + 0.08 * s, clamp(u_scanlines, 0.0, 1.0));
        col *= scan;
    }

    if (u_vignette > 0.001)
    {
        float r = dot(centered, centered);
        float vig = smoothstep(1.15, 0.15, r);
        col *= mix(1.0, vig, clamp(u_vignette, 0.0, 1.0));
    }

    // Tiny grain to avoid banding in dark areas
    float g = (rand(uv + vec2(u_energy, u_energy * 1.37)) - 0.5) * 0.010 * (0.35 + u_energy);
    col += g;

    float expo = max(0.05, u_exposure);
    float cont = max(0.65, u_contrast);

    col *= expo;
    col = tonemap_aces(col);
    col = pow(col, vec3(0.90 / cont));

    col = pow(max(col, 0.0), vec3(0.4545));

    gl_FragColor = vec4(col, 1.0);
}
"""


# ---------------------------------------------------------------------
# VisPy canvas
# ---------------------------------------------------------------------
if HAVE_VISPY:

    class _SDFGalaxyCanvas(app.Canvas):
        """Multi-pass pipeline (scene -> bloom -> combine)."""

        def __init__(self, config: Dict[str, Any]) -> None:
            self._config = config
            self._last_scene_preset: str = ""
            self._audio = _AudioState()

            super().__init__(keys=None, size=(640, 360), show=False)

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
            for p in (self._prog_scene, self._prog_prefilter, self._prog_blur, self._prog_combine):
                p["a_position"] = vb

            gloo.set_state(blend=False, depth_test=False, cull_face=False)

            self._scene_tex = None
            self._scene_fbo = None
            self._bloom_tex0 = None
            self._bloom_tex1 = None
            self._bloom_fbo0 = None
            self._bloom_fbo1 = None

            self._last_sizes = (-1, -1, -1)

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

            self._scene_tex = self._make_texture(w, h)
            self._scene_fbo = gloo.FrameBuffer(color=self._scene_tex)

            bw = max(16, w // down)
            bh = max(16, h // down)
            self._bloom_tex0 = self._make_texture(bw, bh)
            self._bloom_tex1 = self._make_texture(bw, bh)
            self._bloom_fbo0 = gloo.FrameBuffer(color=self._bloom_tex0)
            self._bloom_fbo1 = gloo.FrameBuffer(color=self._bloom_tex1)

        @staticmethod
        def _make_texture(w: int, h: int):
            if HAVE_NUMPY:
                try:
                    data = np.zeros((h, w, 4), dtype=np.float32)
                    return gloo.Texture2D(data, interpolation="linear")
                except Exception:
                    data = np.zeros((h, w, 4), dtype=np.uint8)
                    return gloo.Texture2D(data, interpolation="linear")
            return gloo.Texture2D((h, w, 4), interpolation="linear")

        # -----------------------------
        # Uniform updates
        # -----------------------------
        def _maybe_apply_scene_preset(self) -> None:
            preset = str(self._config.get("scene_preset", "Custom"))
            if not preset or preset == "Custom":
                self._last_scene_preset = preset
                return

            # Apply only once per selection change.
            if preset == self._last_scene_preset:
                return

            values = SCENE_PRESETS.get(preset)
            if isinstance(values, dict):
                # Update config in-place so uniforms pick it up immediately.
                for k, v in values.items():
                    self._config[k] = v

            self._last_scene_preset = preset

        def _update_uniforms(self) -> None:
            w, h = self.physical_size
            self._maybe_apply_scene_preset()

            # Shared
            self._prog_scene["u_resolution"] = (float(w), float(h))
            self._prog_scene["u_time"] = float(self._audio.t)

            self._prog_scene["u_energy"] = float(self._audio.energy)
            self._prog_scene["u_bass"] = float(self._audio.bass)
            self._prog_scene["u_treble"] = float(self._audio.treble)
            self._prog_scene["u_treble_spike"] = float(self._audio.treble_spike)

            preset = str(self._config.get("color_preset", _COLOR_PRESETS[0]))
            self._prog_scene["u_palette_id"] = float(_COLOR_PRESET_TO_ID.get(preset, 0.0))

            # Galaxy shape
            self._prog_scene["u_arms"] = float(_clamp(_safe_float(self._config.get("arms", 3), 3), 1.0, 8.0))
            self._prog_scene["u_disk_radius"] = float(_clamp(_safe_float(self._config.get("disk_radius", 3.6), 3.6), 0.6, 8.0))
            self._prog_scene["u_disk_thickness"] = float(_clamp(_safe_float(self._config.get("disk_thickness", 0.55), 0.55), 0.08, 2.5))
            self._prog_scene["u_core_radius"] = float(_clamp(_safe_float(self._config.get("core_radius", 0.55), 0.55), 0.08, 2.0))

            # Rotation / arms look
            self._prog_scene["u_base_rot"] = float(_clamp(_safe_float(self._config.get("base_rotation", 0.55), 0.55), -6.0, 6.0))
            self._prog_scene["u_bass_rot_gain"] = float(_clamp(_safe_float(self._config.get("bass_rotation_gain", 3.2), 3.2), 0.0, 12.0))

            self._prog_scene["u_spiral_tightness"] = float(_clamp(_safe_float(self._config.get("spiral_tightness", 1.35), 1.35), 0.0, 4.0))
            self._prog_scene["u_arm_width"] = float(_clamp(_safe_float(self._config.get("arm_width", 7.0), 7.0), 0.5, 20.0))
            self._prog_scene["u_arm_power"] = float(_clamp(_safe_float(self._config.get("arm_power", 4.5), 4.5), 1.0, 10.0))

            # Warp
            self._prog_scene["u_warp_strength"] = float(_clamp(_safe_float(self._config.get("warp_strength", 0.55), 0.55), 0.0, 3.0))
            self._prog_scene["u_warp_speed"] = float(_clamp(_safe_float(self._config.get("warp_speed", 1.75), 1.75), 0.0, 6.0))
            self._prog_scene["u_warp_energy_gain"] = float(_clamp(_safe_float(self._config.get("warp_energy_gain", 1.35), 1.35), 0.0, 6.0))

            # Helix
            self._prog_scene["u_helix_amount"] = float(_clamp(self._compute_helix_amount(), 0.0, 1.0))
            self._prog_scene["u_helix_height"] = float(_clamp(_safe_float(self._config.get("helix_height", 1.55), 1.55), 0.0, 6.0))
            self._prog_scene["u_helix_pitch"] = float(_clamp(_safe_float(self._config.get("helix_pitch", 1.10), 1.10), 0.0, 6.0))
            self._prog_scene["u_helix_twist_speed"] = float(_clamp(_safe_float(self._config.get("helix_twist_speed", 0.65), 0.65), 0.0, 6.0))

            # Beams
            self._prog_scene["u_beam_strength"] = float(_clamp(_safe_float(self._config.get("beam_strength", 2.25), 2.25), 0.0, 8.0))
            self._prog_scene["u_beam_length"] = float(_clamp(_safe_float(self._config.get("beam_length", 6.0), 6.0), 0.5, 18.0))
            self._prog_scene["u_beam_radius"] = float(_clamp(_safe_float(self._config.get("beam_radius", 0.25), 0.25), 0.05, 2.0))

            # Camera
            self._prog_scene["u_cam_yaw"] = float(_safe_float(self._config.get("camera_yaw", 0.85), 0.85))
            self._prog_scene["u_cam_pitch"] = float(_safe_float(self._config.get("camera_pitch", 0.62), 0.62))
            self._prog_scene["u_cam_dist"] = float(_clamp(_safe_float(self._config.get("camera_distance", 4.1), 4.1), 2.0, 20.0))
            self._prog_scene["u_fov"] = float(_clamp(_safe_float(self._config.get("fov", 1.55), 1.55), 0.35, 2.0))

            self._prog_scene["u_orbit_amount"] = float(_clamp(_safe_float(self._config.get("camera_orbit_amount", 0.14), 0.14), 0.0, 1.2))
            self._prog_scene["u_orbit_speed"] = float(_clamp(_safe_float(self._config.get("camera_orbit_speed", 0.18), 0.18), 0.0, 2.0))
            self._prog_scene["u_hue_orbit_amount"] = float(_clamp(_safe_float(self._config.get("hue_orbit_amount", 0.10), 0.10), 0.0, 0.6))

            # Quality / fog
            self._prog_scene["u_fog"] = float(_clamp(_safe_float(self._config.get("fog", 1.15), 1.15), 0.0, 3.0))
            self._prog_scene["u_detail_steps"] = float(_clamp(_safe_float(self._config.get("detail_steps", 120), 120), 32.0, 140.0))
            self._prog_scene["u_glow_gain"] = float(_clamp(_safe_float(self._config.get("glow_gain", 1.35), 1.35), 0.0, 4.0))

            # Prefilter / blur / combine
            thr = float(_clamp(_safe_float(self._config.get("bloom_threshold", 0.55), 0.55), 0.0, 2.0))
            knee = float(_clamp(_safe_float(self._config.get("bloom_soft_knee", 0.65), 0.65), 0.0, 1.0))
            self._prog_prefilter["u_threshold"] = thr
            self._prog_prefilter["u_soft_knee"] = knee

            blur_radius = float(_clamp(_safe_float(self._config.get("bloom_radius", 2.4), 2.4), 0.5, 6.0))
            self._prog_blur["u_radius"] = blur_radius

            exp_ = float(_clamp(_safe_float(self._config.get("exposure", 1.25), 1.25), 0.1, 3.0))
            contrast = float(_clamp(_safe_float(self._config.get("contrast", 1.05), 1.05), 0.6, 1.8))
            bloom_i = float(_clamp(_safe_float(self._config.get("bloom", 1.55), 1.55), 0.0, 3.0))
            chroma = float(_clamp(_safe_float(self._config.get("chromatic_aberration", 0.010), 0.010), 0.0, 0.03))
            scan = float(_clamp(_safe_float(self._config.get("scanlines", 0.10), 0.10), 0.0, 1.0))
            vig = float(_clamp(_safe_float(self._config.get("vignette", 0.22), 0.22), 0.0, 1.0))

            self._prog_combine["u_resolution"] = (float(w), float(h))
            self._prog_combine["u_exposure"] = exp_
            self._prog_combine["u_contrast"] = contrast
            self._prog_combine["u_bloom_intensity"] = bloom_i
            self._prog_combine["u_chroma"] = chroma
            self._prog_combine["u_scanlines"] = scan
            self._prog_combine["u_vignette"] = vig
            self._prog_combine["u_energy"] = float(self._audio.energy)

        def _compute_helix_amount(self) -> float:
            # Time-driven transition (seconds) -> [0..1]
            growth = float(_clamp(_safe_float(self._config.get("helix_growth", 0.018), 0.018), 0.0, 1.0))
            # growth is "per second" roughly; keep deterministic
            return _clamp(self._audio.t * growth, 0.0, 1.0)

        # -----------------------------
        # Public hooks
        # -----------------------------
        def set_audio_features(
            self,
            *,
            time_s: float,
            energy: float,
            bass: float,
            treble: float,
            treble_spike: float,
        ) -> None:
            # EMA smoothing
            self._audio.t = float(time_s)

            a_e = 0.28
            a_b = 0.22
            a_t = 0.26

            self._audio.energy = (1.0 - a_e) * self._audio.energy + a_e * float(_clamp(energy, 0.0, 2.0))
            self._audio.bass = (1.0 - a_b) * self._audio.bass + a_b * float(_clamp(bass, 0.0, 2.0))
            self._audio.treble = (1.0 - a_t) * self._audio.treble + a_t * float(_clamp(treble, 0.0, 2.0))

            # Spike envelope (short decay)
            decay = float(_clamp(_safe_float(self._config.get("treble_spike_decay", 0.88), 0.88), 0.5, 0.98))
            self._audio.treble_spike = max(float(treble_spike), self._audio.treble_spike * decay)

            self.update()

        # -----------------------------
        # VisPy callbacks
        # -----------------------------
        def on_resize(self, event) -> None:  # type: ignore[override]
            gloo.set_viewport(0, 0, *event.physical_size)
            self.update()

        def on_draw(self, event) -> None:  # type: ignore[override]
            self._ensure_fbos()
            assert self._scene_fbo is not None
            assert self._bloom_fbo0 is not None and self._bloom_fbo1 is not None
            assert self._scene_tex is not None
            assert self._bloom_tex0 is not None and self._bloom_tex1 is not None

            self._update_uniforms()

            w, h = self.physical_size

            # 1) Scene
            with self._scene_fbo:
                gloo.set_viewport(0, 0, w, h)
                gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                self._prog_scene.draw("triangle_strip")

            # 2) Prefilter + downsample -> bloom0
            bw, bh = self._bloom_tex0.shape[1], self._bloom_tex0.shape[0]
            self._prog_prefilter["u_scene"] = self._scene_tex
            with self._bloom_fbo0:
                gloo.set_viewport(0, 0, bw, bh)
                gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                self._prog_prefilter.draw("triangle_strip")

            # 3) Blur horizontal bloom0 -> bloom1
            self._prog_blur["u_tex"] = self._bloom_tex0
            self._prog_blur["u_texel"] = (1.0 / float(bw), 1.0 / float(bh))
            self._prog_blur["u_dir"] = (1.0, 0.0)
            with self._bloom_fbo1:
                gloo.set_viewport(0, 0, bw, bh)
                gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                self._prog_blur.draw("triangle_strip")

            # 4) Blur vertical bloom1 -> bloom0
            self._prog_blur["u_tex"] = self._bloom_tex1
            self._prog_blur["u_dir"] = (0.0, 1.0)
            with self._bloom_fbo0:
                gloo.set_viewport(0, 0, bw, bh)
                gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                self._prog_blur.draw("triangle_strip")

            # 5) Combine
            gloo.set_viewport(0, 0, w, h)
            gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
            self._prog_combine["u_scene"] = self._scene_tex
            self._prog_combine["u_bloom"] = self._bloom_tex0
            self._prog_combine.draw("triangle_strip")


# ---------------------------------------------------------------------
# Widget wrapper (Qt)
# ---------------------------------------------------------------------
class _SDFGalaxyWidget(QWidget):
    def __init__(self, config: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._canvas = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not HAVE_VISPY:
            label = QLabel(
                "VisPy is not available.\n"
                "SDF Galaxy System is disabled.\n"
                "Please install 'vispy' to enable this visualization.",
                self,
            )
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            return

        app.use_app("pyqt6")  # type: ignore[arg-type]

        self._canvas = _SDFGalaxyCanvas(config=self._config)  # type: ignore[name-defined]
        self._canvas.native.setParent(self)
        layout.addWidget(self._canvas.native)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(640, 360)

    def update_audio_features(self, *, time_ms: int, energy: float, bass: float, treble: float, treble_spike: float) -> None:
        if self._canvas is None:
            return

        self._canvas.set_audio_features(
            time_s=float(time_ms) / 1000.0,
            energy=float(energy),
            bass=float(bass),
            treble=float(treble),
            treble_spike=float(treble_spike),
        )


# ---------------------------------------------------------------------
# Olaf plugin class
# ---------------------------------------------------------------------
class SDFGalaxySystemVisualization(BaseVisualization):
    plugin_id = "sdf_galaxy_system"
    plugin_name = "SDF Galaxy System"
    plugin_description = (
        "Procedural volumetric galaxy: bass-synced rotation, energy-driven spiral warp, "
        "time-evolving helix, treble-spike accretion beams, orbit-driven hue shift. No starfield."
    )
    plugin_author = "DrDLP"
    plugin_version = "0.2.0"
    plugin_max_inputs = 1

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._widget: Optional[_SDFGalaxyWidget] = None

        # Defaults (only if missing, so saved project overrides win)
        self.config.setdefault("scene_preset", "Custom")

        self.config.setdefault("color_preset", _COLOR_PRESETS[0])

        self.config.setdefault("arms", 3)
        self.config.setdefault("disk_radius", 3.6)
        self.config.setdefault("disk_thickness", 0.55)
        self.config.setdefault("core_radius", 0.55)

        self.config.setdefault("base_rotation", 0.55)
        self.config.setdefault("bass_rotation_gain", 3.2)

        self.config.setdefault("spiral_tightness", 1.35)
        self.config.setdefault("arm_width", 7.0)
        self.config.setdefault("arm_power", 4.5)

        self.config.setdefault("warp_strength", 0.55)
        self.config.setdefault("warp_speed", 1.75)
        self.config.setdefault("warp_energy_gain", 1.35)

        self.config.setdefault("helix_growth", 0.018)
        self.config.setdefault("helix_height", 1.55)
        self.config.setdefault("helix_pitch", 1.10)
        self.config.setdefault("helix_twist_speed", 0.65)

        self.config.setdefault("beam_strength", 2.25)
        self.config.setdefault("beam_length", 6.0)
        self.config.setdefault("beam_radius", 0.25)

        self.config.setdefault("treble_spike_threshold", 0.22)
        self.config.setdefault("treble_spike_min", 0.18)
        self.config.setdefault("treble_spike_cooldown_s", 0.12)
        self.config.setdefault("treble_spike_decay", 0.88)

        self.config.setdefault("camera_yaw", 0.85)
        self.config.setdefault("camera_pitch", 0.62)
        self.config.setdefault("camera_distance", 4.1)
        self.config.setdefault("fov", 1.55)

        self.config.setdefault("camera_orbit_amount", 0.14)
        self.config.setdefault("camera_orbit_speed", 0.18)
        self.config.setdefault("hue_orbit_amount", 0.10)

        self.config.setdefault("fog", 1.15)
        self.config.setdefault("detail_steps", 120)
        self.config.setdefault("glow_gain", 1.35)

        # Post-process defaults
        self.config.setdefault("bloom", 1.55)
        self.config.setdefault("bloom_radius", 2.4)
        self.config.setdefault("bloom_threshold", 0.55)
        self.config.setdefault("bloom_soft_knee", 0.65)
        self.config.setdefault("bloom_downsample", 2)

        self.config.setdefault("chromatic_aberration", 0.010)
        self.config.setdefault("exposure", 1.25)
        self.config.setdefault("contrast", 1.05)
        self.config.setdefault("scanlines", 0.10)
        self.config.setdefault("vignette", 0.22)

        # Internal spike state
        self._last_spike_t: float = -999.0
        self._prev_treble: float = 0.0

    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        return {
            "scene_preset": PluginParameter(
                name="scene_preset",
                label="Scene preset",
                type="enum",
                default="Custom",
                choices=SCENE_PRESET_NAMES,
                description="Multi-parameter look. Applied once when changed; then you can fine-tune.",
            ),

            "color_preset": PluginParameter(
                name="color_preset",
                label="Color preset",
                type="enum",
                default=_COLOR_PRESETS[0],
                choices=_COLOR_PRESETS,
                description="Select a neon palette preset (same style as the tunnel plugin).",
            ),

            # Galaxy shape
            "arms": PluginParameter(
                name="arms",
                label="Spiral arms",
                type="int",
                default=3,
                minimum=1,
                maximum=8,
                step=1,
                description="Number of spiral arms.",
            ),
            "disk_radius": PluginParameter(
                name="disk_radius",
                label="Disk radius",
                type="float",
                default=3.6,
                minimum=0.6,
                maximum=8.0,
                step=0.01,
                description="Overall disk size.",
            ),
            "disk_thickness": PluginParameter(
                name="disk_thickness",
                label="Disk thickness",
                type="float",
                default=0.55,
                minimum=0.08,
                maximum=2.5,
                step=0.01,
                description="Base thickness around Y (before helix stretch).",
            ),
            "core_radius": PluginParameter(
                name="core_radius",
                label="Core radius",
                type="float",
                default=0.55,
                minimum=0.08,
                maximum=2.0,
                step=0.01,
                description="Size of the central core glow.",
            ),

            # Rotation
            "base_rotation": PluginParameter(
                name="base_rotation",
                label="Base rotation speed",
                type="float",
                default=0.55,
                minimum=-6.0,
                maximum=6.0,
                step=0.01,
                description="Base angular speed (radians/sec).",
            ),
            "bass_rotation_gain": PluginParameter(
                name="bass_rotation_gain",
                label="Bass rotation gain",
                type="float",
                default=3.2,
                minimum=0.0,
                maximum=12.0,
                step=0.05,
                description="Adds to rotation speed proportionally to bass energy.",
            ),

            # Arms / spiral
            "spiral_tightness": PluginParameter(
                name="spiral_tightness",
                label="Spiral tightness",
                type="float",
                default=1.35,
                minimum=0.0,
                maximum=4.0,
                step=0.01,
                description="Higher = tighter winding spiral.",
            ),
            "arm_width": PluginParameter(
                name="arm_width",
                label="Arm width",
                type="float",
                default=7.0,
                minimum=0.5,
                maximum=20.0,
                step=0.05,
                description="Higher = thinner arms (sharper).",
            ),
            "arm_power": PluginParameter(
                name="arm_power",
                label="Arm contrast",
                type="float",
                default=4.5,
                minimum=1.0,
                maximum=10.0,
                step=0.05,
                description="Higher = arms stand out more from the disk.",
            ),

            # Warp (overall energy)
            "warp_strength": PluginParameter(
                name="warp_strength",
                label="Warp strength",
                type="float",
                default=0.55,
                minimum=0.0,
                maximum=3.0,
                step=0.01,
                description="Spiral warp strength.",
            ),
            "warp_speed": PluginParameter(
                name="warp_speed",
                label="Warp speed",
                type="float",
                default=1.75,
                minimum=0.0,
                maximum=6.0,
                step=0.01,
                description="Temporal speed of the warp.",
            ),
            "warp_energy_gain": PluginParameter(
                name="warp_energy_gain",
                label="Warp energy gain",
                type="float",
                default=1.35,
                minimum=0.0,
                maximum=6.0,
                step=0.05,
                description="How much overall energy amplifies warp.",
            ),

            # Helix (time)
            "helix_growth": PluginParameter(
                name="helix_growth",
                label="Helix growth",
                type="float",
                default=0.018,
                minimum=0.0,
                maximum=0.2,
                step=0.001,
                description="How fast the arms transition into a vertical helix over time (per second).",
            ),
            "helix_height": PluginParameter(
                name="helix_height",
                label="Helix height",
                type="float",
                default=1.55,
                minimum=0.0,
                maximum=6.0,
                step=0.01,
                description="Vertical amplitude of the helix stretch.",
            ),
            "helix_pitch": PluginParameter(
                name="helix_pitch",
                label="Helix pitch",
                type="float",
                default=1.10,
                minimum=0.0,
                maximum=6.0,
                step=0.01,
                description="How fast the helix oscillates with radius.",
            ),
            "helix_twist_speed": PluginParameter(
                name="helix_twist_speed",
                label="Helix twist speed",
                type="float",
                default=0.65,
                minimum=0.0,
                maximum=6.0,
                step=0.01,
                description="Temporal twist applied to the helix.",
            ),

            # Beams (treble spikes)
            "beam_strength": PluginParameter(
                name="beam_strength",
                label="Beam strength",
                type="float",
                default=2.25,
                minimum=0.0,
                maximum=8.0,
                step=0.05,
                description="Accretion beam intensity multiplier.",
            ),
            "beam_length": PluginParameter(
                name="beam_length",
                label="Beam length",
                type="float",
                default=6.0,
                minimum=0.5,
                maximum=18.0,
                step=0.05,
                description="How far beams extend along Y.",
            ),
            "beam_radius": PluginParameter(
                name="beam_radius",
                label="Beam radius",
                type="float",
                default=0.25,
                minimum=0.05,
                maximum=2.0,
                step=0.01,
                description="Beam thickness around the core (XZ).",
            ),
            "treble_spike_threshold": PluginParameter(
                name="treble_spike_threshold",
                label="Treble spike threshold",
                type="float",
                default=0.22,
                minimum=0.01,
                maximum=1.5,
                step=0.01,
                description="Spike triggers when treble rises faster than this amount.",
            ),
            "treble_spike_min": PluginParameter(
                name="treble_spike_min",
                label="Treble minimum",
                type="float",
                default=0.18,
                minimum=0.0,
                maximum=2.0,
                step=0.01,
                description="Minimum treble level required to allow a spike burst.",
            ),
            "treble_spike_cooldown_s": PluginParameter(
                name="treble_spike_cooldown_s",
                label="Spike cooldown (s)",
                type="float",
                default=0.12,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="Minimum time between spikes.",
            ),
            "treble_spike_decay": PluginParameter(
                name="treble_spike_decay",
                label="Spike decay",
                type="float",
                default=0.88,
                minimum=0.5,
                maximum=0.98,
                step=0.01,
                description="How quickly spike bursts fade (closer to 1 = longer bursts).",
            ),

            # Camera + orbit hue shift
            "camera_yaw": PluginParameter(
                name="camera_yaw",
                label="Camera yaw",
                type="float",
                default=0.85,
                minimum=-6.283,
                maximum=6.283,
                step=0.01,
                description="Base camera yaw (radians).",
            ),
            "camera_pitch": PluginParameter(
                name="camera_pitch",
                label="Camera pitch",
                type="float",
                default=0.62,
                minimum=-1.2,
                maximum=1.2,
                step=0.01,
                description="Base camera pitch (radians).",
            ),
            "camera_distance": PluginParameter(
                name="camera_distance",
                label="Camera distance",
                type="float",
                default=4.1,
                minimum=2.0,
                maximum=20.0,
                step=0.05,
                description="Camera distance from center.",
            ),
            "fov": PluginParameter(
                name="fov",
                label="FOV",
                type="float",
                default=1.55,
                minimum=0.35,
                maximum=2.0,
                step=0.01,
                description="Field of view factor (higher = wider).",
            ),
            "camera_orbit_amount": PluginParameter(
                name="camera_orbit_amount",
                label="Orbit amount",
                type="float",
                default=0.14,
                minimum=0.0,
                maximum=1.2,
                step=0.01,
                description="Amplitude of the subtle camera orbit (0 disables orbit).",
            ),
            "camera_orbit_speed": PluginParameter(
                name="camera_orbit_speed",
                label="Orbit speed",
                type="float",
                default=0.18,
                minimum=0.0,
                maximum=2.0,
                step=0.01,
                description="Speed of camera orbit (radians/sec).",
            ),
            "hue_orbit_amount": PluginParameter(
                name="hue_orbit_amount",
                label="Hue orbit amount",
                type="float",
                default=0.10,
                minimum=0.0,
                maximum=0.6,
                step=0.01,
                description="How much orbit phase shifts palette (subtle hue drift).",
            ),

            # Quality / fog / glow
            "fog": PluginParameter(
                name="fog",
                label="Fog",
                type="float",
                default=1.15,
                minimum=0.0,
                maximum=3.0,
                step=0.05,
                description="Distance fog (keeps background clean/black).",
            ),
            "detail_steps": PluginParameter(
                name="detail_steps",
                label="Detail steps",
                type="int",
                default=120,
                minimum=32,
                maximum=140,
                step=1,
                description="Raymarch step cap (higher = richer, slower).",
            ),
            "glow_gain": PluginParameter(
                name="glow_gain",
                label="Glow gain",
                type="float",
                default=1.35,
                minimum=0.0,
                maximum=4.0,
                step=0.01,
                description="Extra emissive glow multiplier.",
            ),

            # Post process
            "bloom": PluginParameter(
                name="bloom",
                label="Bloom intensity",
                type="float",
                default=1.55,
                minimum=0.0,
                maximum=3.0,
                step=0.05,
                description="Bloom contribution in final combine.",
            ),
            "bloom_radius": PluginParameter(
                name="bloom_radius",
                label="Bloom radius",
                type="float",
                default=2.4,
                minimum=0.5,
                maximum=6.0,
                step=0.05,
                description="Bloom blur radius.",
            ),
            "bloom_threshold": PluginParameter(
                name="bloom_threshold",
                label="Bloom threshold",
                type="float",
                default=0.55,
                minimum=0.0,
                maximum=2.0,
                step=0.01,
                description="Only bright parts above this contribute to bloom.",
            ),
            "bloom_soft_knee": PluginParameter(
                name="bloom_soft_knee",
                label="Bloom soft knee",
                type="float",
                default=0.65,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="Soft transition around threshold.",
            ),
            "bloom_downsample": PluginParameter(
                name="bloom_downsample",
                label="Bloom downsample",
                type="int",
                default=2,
                minimum=1,
                maximum=4,
                step=1,
                description="Downsampling factor for bloom buffers.",
            ),
            "chromatic_aberration": PluginParameter(
                name="chromatic_aberration",
                label="Chromatic aberration",
                type="float",
                default=0.010,
                minimum=0.0,
                maximum=0.03,
                step=0.001,
                description="Post-process RGB split amount.",
            ),
            "exposure": PluginParameter(
                name="exposure",
                label="Exposure",
                type="float",
                default=1.25,
                minimum=0.1,
                maximum=3.0,
                step=0.01,
                description="Overall brightness before tone mapping.",
            ),
            "contrast": PluginParameter(
                name="contrast",
                label="Contrast",
                type="float",
                default=1.05,
                minimum=0.6,
                maximum=1.8,
                step=0.01,
                description="Contrast shaping.",
            ),
            "scanlines": PluginParameter(
                name="scanlines",
                label="Scanlines",
                type="float",
                default=0.10,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="CRT-like scanlines (subtle).",
            ),
            "vignette": PluginParameter(
                name="vignette",
                label="Vignette",
                type="float",
                default=0.22,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="Darken edges.",
            ),
        }

    def create_widget(self, parent: Optional[QWidget] = None) -> QWidget:
        self._widget = _SDFGalaxyWidget(config=self.config, parent=parent)
        return self._widget

    def on_audio_features(self, features: Dict[str, Any]) -> None:
        if self._widget is None:
            return

        time_ms = int(features.get("time_ms", 0))
        inputs = features.get("inputs", {}) or {}
        inp = inputs.get("input_1", {}) or {}

        energy = _safe_float(inp.get("rms", 0.0), 0.0)

        # Bass + treble (best-effort extraction; falls back to RMS)
        bass = _extract_band_energy(inp, ("bass", "low", "low_energy"))  # type: ignore[arg-type]
        treble = _extract_band_energy(inp, ("treble", "high", "high_energy"))  # type: ignore[arg-type]
        if bass is None:
            bass = energy
        if treble is None:
            treble = energy

        # Treble spike detector (cheap + robust)
        t_s = float(time_ms) / 1000.0
        thr = float(_clamp(_safe_float(self.config.get("treble_spike_threshold", 0.22), 0.22), 0.01, 2.0))
        tmin = float(_clamp(_safe_float(self.config.get("treble_spike_min", 0.18), 0.18), 0.0, 2.0))
        cooldown = float(_clamp(_safe_float(self.config.get("treble_spike_cooldown_s", 0.12), 0.12), 0.0, 2.0))

        # Use a slightly smoothed treble for stability (but keep responsiveness)
        sm = 0.35
        treble_s = (1.0 - sm) * self._prev_treble + sm * float(treble)

        spike = 0.0
        if (t_s - self._last_spike_t) >= cooldown:
            if treble_s >= tmin and (treble_s - self._prev_treble) >= thr:
                spike = 1.0
                self._last_spike_t = t_s

        self._prev_treble = treble_s

        self._widget.update_audio_features(
            time_ms=time_ms,
            energy=float(energy),
            bass=float(bass),
            treble=float(treble_s),
            treble_spike=float(spike),
        )
