
"""pulsing_runes.py

Olaf Visualization Plugin (VisPy / Gloo) - Pulsing Runes

Goal
----
- n concentric circular rows of "runes".
- Each ring rotates in opposite direction to its neighbor.
- Drum/kick pulses illuminate successive rings (one ring per pulse, cycling).

Notes
-----
- Top-down view only; camera is intentionally NOT parameterized.
- Multi-pass bloom pipeline reused (scene -> bloom prefilter -> blur -> combine), matching
  the project plugin architecture.

License: same as project
"""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    from vispy.gloo import gl  # type: ignore

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



def _db_to_linear(db_val: float) -> float:
    """
    Converts a dB value (typically negative) to a linear amplitude.
    We use 20*log10(amplitude) convention.
    """
    try:
        return float(10.0 ** (db_val / 20.0))
    except Exception:
        return 0.0


def _extract_scalar(input_dict: Dict[str, Any], keys: tuple[str, ...]) -> Optional[float]:
    """
    Extracts a single scalar feature from an input dict.

    Supports common key variants such as:
      - "rms", "energy", "level", "loudness", "amplitude"
      - dB variants: "rms_db", "level_db", "loudness_db" (converted to linear)

    If the value is a list/tuple, the first element is used.
    """
    for k in keys:
        if k in input_dict:
            v = input_dict.get(k)
            if isinstance(v, (list, tuple)) and len(v):
                v = v[0]
            fv = _safe_float(v, 0.0)

            # Heuristic: treat *_db keys as decibels.
            if k.endswith("_db") or k.endswith("db"):
                fv = _db_to_linear(fv)

            return fv

    return None

def _extract_band_energy(input_dict: Dict[str, Any], keys: tuple[str, ...]) -> Optional[float]:
    """
    Tries common shapes:
      - input_1["bass"] / ["low"] / ...
      - input_1["bands"] / ["band_energies"] dict
    """
    for k in keys:
        if k in input_dict:
            return _safe_float(input_dict.get(k, 0.0), 0.0)

    for container_key in ("bands", "band_energies", "eq", "spectrum_bands"):
        bd = input_dict.get(container_key)

        # Dict-shaped: {"bass": 0.2, "mid": 0.1, ...}
        if isinstance(bd, dict):
            for k in keys:
                if k in bd:
                    return _safe_float(bd.get(k, 0.0), 0.0)

        # Array-shaped (common in Olaf): [band0, band1, ...] where band0~low/bass
        if isinstance(bd, (list, tuple)):
            if len(bd) > 0:
                n = min(3, len(bd))  # take the lowest bands as "bass"
                try:
                    return float(sum(float(bd[i]) for i in range(n)) / float(n))
                except Exception:
                    return None

        # Numpy arrays are supported when available
        try:
            import numpy as _np  # local import (optional)
            if isinstance(bd, _np.ndarray) and bd.size > 0:
                n = int(min(3, bd.size))
                return float(_np.mean(bd[:n]))
        except Exception:
            pass

    return None


@dataclass
class _AudioState:
    t: float = 0.0
    energy: float = 0.0
    bass: float = 0.0
    # Separate smoothed envelopes for specific modulation targets.
    # These exist because the host may provide relatively sparse RMS updates.
    rotation_energy: float = 0.0
    camera_energy: float = 0.0
    pulse: float = 0.0
    active_ring: float = 0.0  # stored as float for GLSL
    _prev_bass: float = 0.0
    _last_pulse_t: float = -999.0


# ---------------------------------------------------------------------
# Style presets
# ---------------------------------------------------------------------
# We keep the preset count small but cover very different looks.
_STYLE_PRESETS = [
    "Runes Classic",
    "HUD Brackets",
    "Target Reticle",
    "Barcode Glitch",
    "Circuit Traces",
    "Hex Tiles",
    "Tech Mandala",
    "Sigils Abstract",
    "Constellations",
    "Circular Sequencer",
]
_STYLE_PRESET_TO_ID = {name: float(i) for i, name in enumerate(_STYLE_PRESETS)}


# ---------------------------------------------------------------------
# Colorways
# ---------------------------------------------------------------------
_COLORWAYS = [
    "Neon Violet",
    "Cyan Tech",
    "Amber Ritual",
    "Lime Acid",
    "Crimson Void",
    "Ocean Teal",
    "Solar Gold",
    "Monochrome Ice",
]
_COLORWAY_TO_ID = {name: float(i) for i, name in enumerate(_COLORWAYS)}

# Backward compatibility: old projects may still contain `color_preset`.
_LEGACY_COLOR_PRESETS = [
    "Cyan/Magenta",
    "Amber/Teal",
    "Green",
    "Purple/Blue",
    "Monochrome",
]
_LEGACY_COLOR_PRESET_TO_COLORWAY = {
    "Cyan/Magenta": "Cyan Tech",
    "Amber/Teal": "Amber Ritual",
    "Green": "Cyan Tech",
    "Purple/Blue": "Neon Violet",
    "Monochrome": "Neon Violet",
}

# ---------------------------------------------------------------------
# Preset bundles
# ---------------------------------------------------------------------
# Each preset applies a coherent set of parameters. We intentionally avoid
# touching the pulse detector thresholds here (pulse_threshold),
# because these are highly dependent on the user's stems/preview backend.
_STYLE_PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Runes Classic": {
        "style_id": 0.0,
        "rings": 7,
        "runes_per_ring": 24,
        "runes_step": 3,
        "rune_length": 0.80,
        "rune_thickness": 0.015,
        "rotation_speed": 0.55,
        "rotation_step": 0.08,
        "pulse_gain": 6.0,
        "pulse_trail": 0.35,
        "pulse_step_delay_s": 0.06,
        "glow_gain": 1.25,
        "exposure": 1.15,
        "contrast": 1.10,
        "chromatic_aberration": 0.008,
        "vignette": 0.32,
    },
    "HUD Brackets": {
        "style_id": 1.0,
        "rings": 7,
        "runes_per_ring": 18,
        "runes_step": 2,
        "rune_length": 0.95,
        "rune_thickness": 0.013,
        "rotation_speed": 0.40,
        "rotation_step": 0.06,
        "pulse_gain": 7.0,
        "pulse_trail": 0.28,
        "pulse_step_delay_s": 0.055,
        "glow_gain": 1.15,
        "exposure": 1.10,
        "contrast": 1.18,
        "chromatic_aberration": 0.006,
        "vignette": 0.28,
    },
    "Target Reticle": {
        "style_id": 2.0,
        "rings": 6,
        "runes_per_ring": 16,
        "runes_step": 2,
        "rune_length": 1.05,
        "rune_thickness": 0.014,
        "rotation_speed": 0.30,
        "rotation_step": 0.05,
        "pulse_gain": 8.0,
        "pulse_trail": 0.25,
        "pulse_step_delay_s": 0.06,
        "glow_gain": 1.10,
        "exposure": 1.08,
        "contrast": 1.22,
        "chromatic_aberration": 0.007,
        "vignette": 0.26,
    },
    "Barcode Glitch": {
        "style_id": 3.0,
        "rings": 8,
        "runes_per_ring": 44,
        "runes_step": 4,
        "rune_length": 0.65,
        "rune_thickness": 0.012,
        "rotation_speed": 0.70,
        "rotation_step": 0.10,
        "pulse_gain": 6.5,
        "pulse_trail": 0.20,
        "pulse_step_delay_s": 0.045,
        "glow_gain": 1.05,
        "exposure": 1.05,
        "contrast": 1.28,
        "chromatic_aberration": 0.010,
        "scanlines": 0.10,
        "vignette": 0.34,
    },
    "Circuit Traces": {
        "style_id": 4.0,
        "rings": 7,
        "runes_per_ring": 22,
        "runes_step": 3,
        "rune_length": 0.88,
        "rune_thickness": 0.013,
        "rotation_speed": 0.45,
        "rotation_step": 0.07,
        "pulse_gain": 7.5,
        "pulse_trail": 0.30,
        "pulse_step_delay_s": 0.06,
        "glow_gain": 1.20,
        "exposure": 1.12,
        "contrast": 1.14,
        "chromatic_aberration": 0.007,
        "vignette": 0.30,
    },
    "Hex Tiles": {
        "style_id": 5.0,
        "rings": 7,
        "runes_per_ring": 20,
        "runes_step": 2,
        "rune_length": 0.95,
        "rune_thickness": 0.012,
        "rotation_speed": 0.38,
        "rotation_step": 0.06,
        "pulse_gain": 6.0,
        "pulse_trail": 0.40,
        "pulse_step_delay_s": 0.065,
        "glow_gain": 1.05,
        "exposure": 1.12,
        "contrast": 1.06,
        "chromatic_aberration": 0.005,
        "vignette": 0.28,
    },
    "Tech Mandala": {
        "style_id": 6.0,
        "rings": 9,
        "runes_per_ring": 14,
        "runes_step": 1,
        "rune_length": 1.10,
        "rune_thickness": 0.014,
        "rotation_speed": 0.22,
        "rotation_step": 0.03,
        "pulse_gain": 8.0,
        "pulse_trail": 0.55,
        "pulse_step_delay_s": 0.080,
        "glow_gain": 1.35,
        "exposure": 1.18,
        "contrast": 1.00,
        "chromatic_aberration": 0.004,
        "vignette": 0.38,
    },
    "Sigils Abstract": {
        "style_id": 7.0,
        "rings": 6,
        "runes_per_ring": 12,
        "runes_step": 1,
        "rune_length": 1.15,
        "rune_thickness": 0.016,
        "rotation_speed": 0.32,
        "rotation_step": 0.04,
        "pulse_gain": 9.0,
        "pulse_trail": 0.35,
        "pulse_step_delay_s": 0.070,
        "glow_gain": 1.30,
        "exposure": 1.12,
        "contrast": 1.10,
        "chromatic_aberration": 0.006,
        "vignette": 0.34,
    },
    "Constellations": {
        "style_id": 8.0,
        "rings": 6,
        "runes_per_ring": 10,
        "runes_step": 1,
        "rune_length": 1.20,
        "rune_thickness": 0.014,
        "rotation_speed": 0.26,
        "rotation_step": 0.04,
        "pulse_gain": 8.5,
        "pulse_trail": 0.45,
        "pulse_step_delay_s": 0.075,
        "glow_gain": 1.45,
        "exposure": 1.16,
        "contrast": 1.02,
        "chromatic_aberration": 0.005,
        "vignette": 0.42,
    },
    "Circular Sequencer": {
        "style_id": 9.0,
        "rings": 5,
        "runes_per_ring": 32,
        "runes_step": 2,
        "rune_length": 0.75,
        "rune_thickness": 0.012,
        "rotation_speed": 0.55,
        "rotation_step": 0.05,
        "pulse_gain": 7.0,
        "pulse_trail": 0.22,
        "pulse_step_delay_s": 0.050,
        "glow_gain": 1.20,
        "exposure": 1.10,
        "contrast": 1.16,
        "chromatic_aberration": 0.008,
        "vignette": 0.30,
    },
}


# ---------------------------------------------------------------------
# GLSL
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

_FRAGMENT_SCENE = r"""
precision highp float;

varying vec2 v_uv;

uniform vec2  u_resolution;
uniform float u_time;

uniform float u_energy;
uniform float u_bass;

uniform float u_rings;
uniform float u_inner_radius;
uniform float u_ring_thickness;
uniform float u_ring_gap;
uniform float u_camera_distance;

uniform float u_pulse_age;
uniform float u_pulse_step_s;

uniform float u_runes_per_ring;
uniform float u_rune_step;          // additional runes per outer ring
uniform float u_rune_length;        // tangent length (in segment space)
uniform float u_rune_thickness;

uniform float u_rotation_speed;     // radians/sec at ring 0
uniform float u_rotation_step;      // extra speed per ring

uniform float u_active_ring;
uniform float u_pulse;              // 0..1 envelope
uniform float u_pulse_trail;        // trail size in rings (>=0.1)
uniform float u_pulse_gain;         // intensity multiplier

uniform float u_palette_id;
uniform float u_glow_gain;
uniform float u_style_id;

#define TAU 6.28318530718
#define MAX_RINGS 16

float hash11(float p)
{
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

float sdSegment(vec2 p, vec2 a, vec2 b)
{
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

float sdBox(vec2 p, vec2 b)
{
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

vec3 palette(float k, float preset)
{
    // Colorways (high-contrast neon palettes).
    // preset:
    //  0 = Neon Violet
    //  1 = Cyan Tech
    //  2 = Amber Ritual
    //  3 = Lime Acid
    //  4 = Crimson Void
    //  5 = Ocean Teal
    //  6 = Solar Gold
    //  7 = Monochrome Ice
    k = fract(k);
    float s = 0.5 + 0.5 * sin(TAU * k);

    if (preset < 0.5)      return mix(vec3(0.22, 0.40, 1.00), vec3(0.95, 0.18, 1.00), s); // violet/magenta
    else if (preset < 1.5) return mix(vec3(0.05, 0.95, 0.95), vec3(0.10, 0.35, 1.00), s); // cyan/blue
    else if (preset < 2.5) return mix(vec3(1.00, 0.62, 0.12), vec3(1.00, 0.15, 0.55), s); // amber/pink
    else if (preset < 3.5) return mix(vec3(0.25, 1.00, 0.18), vec3(0.98, 1.00, 0.08), s); // lime/yellow
    else if (preset < 4.5) return mix(vec3(1.00, 0.10, 0.20), vec3(0.55, 0.00, 0.85), s); // red/purple
    else if (preset < 5.5) return mix(vec3(0.00, 0.85, 0.75), vec3(0.05, 0.35, 0.55), s); // teal/deep teal
    else if (preset < 6.5) return mix(vec3(1.00, 0.80, 0.20), vec3(1.00, 0.95, 0.55), s); // gold/ivory
    else                   return mix(vec3(0.85, 0.95, 1.00), vec3(0.35, 0.70, 1.00), s); // icy mono
}

float rune_mask(vec2 q, float seed, float thickness)
{
    // Small "rune" made of a few strokes chosen by seed.
    float h1 = hash11(seed + 1.0);
    float h2 = hash11(seed + 2.0);
    float h3 = hash11(seed + 3.0);

    float d = 1e6;

    // vertical spine (often)
    if (h1 > 0.20)
        d = min(d, sdSegment(q, vec2(0.0, -0.35), vec2(0.0, 0.35)));

    // diagonal
    if (h2 > 0.25)
    {
        float s = sign(h2 - 0.5);
        d = min(d, sdSegment(q, vec2(-0.32, -0.22 * s), vec2(0.32, 0.22 * s)));
    }

    // bar / hook
    if (h3 > 0.35)
    {
        float y = mix(-0.22, 0.22, h3);
        d = min(d, sdSegment(q, vec2(-0.28, y), vec2(0.28, y)));
    }

    // tiny cap box (rare)
    if (h1 < 0.12)
        d = min(d, sdBox(q - vec2(0.0, 0.26), vec2(0.10, 0.05)));

    float ink = smoothstep(thickness, 0.0, d);
    float halo = smoothstep(thickness * 7.0, thickness, d) * 0.35;
    return clamp(ink + halo, 0.0, 1.0);
}

float _stroke(float d, float thickness)
{
    float ink = smoothstep(thickness, 0.0, d);
    float halo = smoothstep(thickness * 7.0, thickness, d) * 0.30;
    return clamp(ink + halo, 0.0, 1.0);
}

float sdCircle(vec2 p, float r)
{
    return length(p) - r;
}

// Inigo Quilez style SDF for a regular hexagon centered at origin.
float sdHex(vec2 p, float r)
{
    vec3 k = vec3(-0.8660254, 0.5, 0.57735);
    p = abs(p);
    p -= 2.0 * min(dot(k.xy, p), 0.0) * k.xy;
    p -= vec2(clamp(p.x, -k.z * r, k.z * r), r);
    return length(p) * sign(p.y);
}

float sdEquilateralTriangle(vec2 p, float r)
{
    // r ~ "radius"
    const float k = 1.7320508;
    p.x = abs(p.x) - r;
    p.y = p.y + r / k;
    if (p.x + k * p.y > 0.0)
        p = vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0;
    p.x -= clamp(p.x, -2.0 * r, 0.0);
    return -length(p) * sign(p.y);
}

float hud_mask(vec2 q, float seed, float thickness)
{
    float h1 = hash11(seed + 11.0);
    float h2 = hash11(seed + 17.0);

    float d = 1e6;

    // Brackets on left/right
    float x = 0.28;
    float y = 0.28 + 0.10 * (h1 - 0.5);

    d = min(d, sdSegment(q, vec2(-x, -y), vec2(-x,  y)));
    d = min(d, sdSegment(q, vec2( x, -y), vec2( x,  y)));

    // Small caps (randomly top or bottom to add variety)
    float capY = (h2 > 0.5) ? y : -y;
    d = min(d, sdSegment(q, vec2(-x, capY), vec2(-x + 0.20, capY)));
    d = min(d, sdSegment(q, vec2( x, capY), vec2( x - 0.20, capY)));

    // Inner tick
    if (h1 > 0.35)
        d = min(d, sdSegment(q, vec2(-0.10, 0.0), vec2(0.10, 0.0)));

    return _stroke(d, thickness);
}

float reticle_mask(vec2 q, float seed, float thickness)
{
    // Use seed in a no-op to avoid unused-parameter warnings on some GLSL compilers.
    float d = 1e6 + 0.0 * seed;

    // Cross
    d = min(d, sdSegment(q, vec2(-0.30, 0.0), vec2(0.30, 0.0)));
    d = min(d, sdSegment(q, vec2(0.0, -0.30), vec2(0.0, 0.30)));

    // Small ring
    float ring = abs(sdCircle(q, 0.22)) - thickness * 0.65;
    d = min(d, ring);

    // Two small corner notches
    d = min(d, sdSegment(q, vec2(-0.32, 0.18), vec2(-0.18, 0.32)));
    d = min(d, sdSegment(q, vec2( 0.32,-0.18), vec2( 0.18,-0.32)));

    return _stroke(d, thickness);
}

float barcode_mask(vec2 q, float seed, float thickness)
{
    float h1 = hash11(seed + 5.0);
    float h2 = hash11(seed + 6.0);
    float h3 = hash11(seed + 7.0);
    float h4 = hash11(seed + 8.0);

    float d = 1e6;

    // 4 vertical bars with varying widths/heights.
    vec2 p = q;
    float x0 = -0.30;
    float w0 = mix(0.02, 0.06, h1);
    float h0 = mix(0.16, 0.36, h2);
    d = min(d, sdBox(p - vec2(x0, 0.0), vec2(w0, h0)));

    float x1 = -0.10;
    float w1 = mix(0.02, 0.07, h2);
    float h1b = mix(0.12, 0.34, h3);
    d = min(d, sdBox(p - vec2(x1, 0.0), vec2(w1, h1b)));

    float x2 = 0.10;
    float w2 = mix(0.02, 0.08, h3);
    float h2b = mix(0.12, 0.34, h4);
    d = min(d, sdBox(p - vec2(x2, 0.0), vec2(w2, h2b)));

    float x3 = 0.30;
    float w3 = mix(0.02, 0.06, h4);
    float h3b = mix(0.16, 0.36, h1);
    d = min(d, sdBox(p - vec2(x3, 0.0), vec2(w3, h3b)));

    // Convert box SDF to outline-ish bars
    return _stroke(d, thickness * 1.4);
}

float circuit_mask(vec2 q, float seed, float thickness)
{
    float h1 = hash11(seed + 21.0);
    float h2 = hash11(seed + 22.0);

    float d = 1e6;

    // An L-shaped trace
    float x = mix(-0.20, 0.20, h1);
    float y = mix(-0.20, 0.20, h2);

    d = min(d, sdSegment(q, vec2(-0.30, y), vec2(x, y)));
    d = min(d, sdSegment(q, vec2(x, y), vec2(x, 0.30)));

    // A node/pad
    float pad = abs(sdCircle(q - vec2(x, y), 0.06)) - thickness * 0.8;
    d = min(d, pad);

    // A second small pad
    float pad2 = abs(sdCircle(q - vec2(-0.30, y), 0.05)) - thickness * 0.8;
    d = min(d, pad2);

    return _stroke(d, thickness);
}

float hex_tiles_mask(vec2 q, float seed, float thickness)
{
    // Use seed in a no-op to avoid unused-parameter warnings on some GLSL compilers.
    float d = abs(sdHex(q, 0.34)) - thickness * 0.8 + 0.0 * seed;
    return _stroke(d, thickness);
}

float mandala_mask(vec2 q, float seed, float thickness)
{
    float h1 = hash11(seed + 31.0);
    float d = 1e6;

    // Triangle outline
    float tri = abs(sdEquilateralTriangle(q * (0.95 + 0.15 * h1), 0.34)) - thickness * 0.75;
    d = min(d, tri);

    // Inner circle
    float ring = abs(sdCircle(q, 0.18 + 0.04 * (h1 - 0.5))) - thickness * 0.55;
    d = min(d, ring);

    // Small radial spoke
    d = min(d, sdSegment(q, vec2(0.0, -0.26), vec2(0.0, 0.26)));

    return _stroke(d, thickness);
}

float sigil_mask(vec2 q, float seed, float thickness)
{
    // Abstract "sigil": random connected strokes + dots.
    float h1 = hash11(seed + 41.0);
    float h2 = hash11(seed + 42.0);
    float h3 = hash11(seed + 43.0);

    vec2 p0 = vec2(mix(-0.28, 0.28, h1), mix(-0.25, 0.25, h2));
    vec2 p1 = vec2(mix(-0.28, 0.28, h2), mix(-0.25, 0.25, h3));
    vec2 p2 = vec2(mix(-0.28, 0.28, h3), mix(-0.25, 0.25, h1));

    float d = 1e6;
    d = min(d, sdSegment(q, p0, p1));
    d = min(d, sdSegment(q, p1, p2));

    // Two dots
    d = min(d, abs(sdCircle(q - p0, 0.045)) - thickness * 0.7);
    d = min(d, abs(sdCircle(q - p2, 0.035)) - thickness * 0.7);

    return _stroke(d, thickness);
}

float constellation_mask(vec2 q, float seed, float thickness)
{
    float h1 = hash11(seed + 51.0);
    float h2 = hash11(seed + 52.0);
    float h3 = hash11(seed + 53.0);

    vec2 p0 = vec2(mix(-0.26, -0.05, h1), mix(-0.20, 0.20, h2));
    vec2 p1 = vec2(mix(-0.05, 0.18, h2), mix(-0.22, 0.22, h3));
    vec2 p2 = vec2(mix(0.05, 0.28, h3), mix(-0.20, 0.20, h1));

    float d = 1e6;

    // Links
    d = min(d, sdSegment(q, p0, p1));
    d = min(d, sdSegment(q, p1, p2));

    // Stars (dots)
    d = min(d, abs(sdCircle(q - p0, 0.05)) - thickness * 0.9);
    d = min(d, abs(sdCircle(q - p1, 0.04)) - thickness * 0.9);
    d = min(d, abs(sdCircle(q - p2, 0.035)) - thickness * 0.9);

    return _stroke(d, thickness);
}

float sequencer_mask(vec2 q, float seed, float thickness)
{
    float h = hash11(seed + 61.0);
    float d = 1e6;

    // 5 steps in a line, some "active"
    for (int i = 0; i < 5; i++)
    {
        float fi = float(i);
        float x = mix(-0.28, 0.28, fi / 4.0);
        float r = (hash11(seed + 100.0 + fi) > 0.55) ? 0.045 : 0.028;
        d = min(d, abs(sdCircle(q - vec2(x, 0.0), r)) - thickness * 0.9);
    }

    // Measure tick
    d = min(d, sdSegment(q, vec2(-0.32, -0.26), vec2(0.32, -0.26)));

    // Small accent dot
    d = min(d, abs(sdCircle(q - vec2(0.0, 0.18), 0.03 + 0.02 * h)) - thickness * 0.8);

    return _stroke(d, thickness);
}

float glyph_mask(vec2 q, float seed, float thickness, float style)
{
    if (style < 0.5)      return rune_mask(q, seed, thickness);
    else if (style < 1.5) return hud_mask(q, seed, thickness);
    else if (style < 2.5) return reticle_mask(q, seed, thickness);
    else if (style < 3.5) return barcode_mask(q, seed, thickness);
    else if (style < 4.5) return circuit_mask(q, seed, thickness);
    else if (style < 5.5) return hex_tiles_mask(q, seed, thickness);
    else if (style < 6.5) return mandala_mask(q, seed, thickness);
    else if (style < 7.5) return sigil_mask(q, seed, thickness);
    else if (style < 8.5) return constellation_mask(q, seed, thickness);
    else                  return sequencer_mask(q, seed, thickness);
}

void main()
{
    vec2 fc = v_uv * u_resolution;

    // Centered normalized coords (top-down)
    vec2 uv = (fc / u_resolution) * 2.0 - 1.0;
    uv.x *= u_resolution.x / max(1.0, u_resolution.y);
    uv *= max(0.25, u_camera_distance);

    float r = length(uv);
    float ang = atan(uv.y, uv.x); // -pi..pi
    float a01 = (ang + 3.14159265) / TAU; // 0..1

    vec3 col = vec3(0.0);

    float rings = clamp(u_rings, 1.0, float(MAX_RINGS));
    float inner = max(0.02, u_inner_radius);

    // Rings are always "glued": u_ring_gap is reinterpreted as extra ring height (thickness),
    // NOT spacing between rings.
    float thick_base = max(0.008, u_ring_thickness);
    float height = max(0.0, u_ring_gap);               // 0..0.25
    float thick = thick_base * (1.0 + 3.0 * height);   // up to ~1.75x
    float pitch = thick;

    float pulse = clamp(u_pulse, 0.0, 2.0);
    float trail = max(0.10, u_pulse_trail);
    float step_s = max(0.01, u_pulse_step_s);
    // Discrete stepping: highlighted ring index jumps (0 -> 1 -> 2 ...)
    float wave_ring = clamp(floor(u_pulse_age / step_s), 0.0, rings - 1.0);
    // Total travel time (used for subtle core glow decay)
    float spread_total = max(0.05, step_s * rings);
// Gentle breathing + global energy gain
    float breathe = 0.04 * sin(1.7 * u_time + 2.0 * r);
    float global_gain = (0.60 + 0.80 * clamp(u_energy, 0.0, 1.5)) + breathe;

    for (int i = 0; i < MAX_RINGS; i++)
    {
        if (float(i) >= rings) break;

        float fi = float(i);
        float radius = inner + fi * pitch;

        // Alternate rotation direction
        float dir = (mod(fi, 2.0) < 0.5) ? 1.0 : -1.0;
        float rot = dir * (u_rotation_speed + fi * u_rotation_step) * u_time;

        // Segment count grows with ring index
        float segs = max(6.0, u_runes_per_ring + fi * u_rune_step);
        float seg_angle = TAU / segs;

        float ang_r = ang + rot;
        float idx = floor(((ang_r + 3.14159265) / TAU) * segs);

        float center = (idx + 0.5) * seg_angle - 3.14159265;
        float local_ang = (ang_r - center) / seg_angle; // ~ [-0.5..0.5]

        // Radial coord scaled to tile space (thicker than a thin ring line)
        float qy = (r - radius) / (thick * 0.62);
        vec2 qt = vec2(local_ang, qy);

        // Tile SDF: little "slabs" around the ring (as in the reference image)
        float tile_y = 0.82 + 0.50 * height; // fill ring more; height fattens tiles
        vec2 tile_half = vec2(0.46, tile_y);
        float tile_d = sdBox(qt, tile_half);

        float tile_fill = smoothstep(0.020, -0.020, tile_d);
        float tile_edge = smoothstep(0.030, 0.000, abs(tile_d));

        // Subtle ring boundary (helps separate rings)
        float ring_band = smoothstep(0.85, 0.12, abs(qy));

        // Pulse highlight weights per ring
        float d_ring = abs(fi - wave_ring);
        float pulse_w = exp(-d_ring / trail);
        float bass_gain = 0.6 + 0.8 * clamp(u_bass, 0.0, 1.5);
        float pulse_boost = u_pulse_gain * pulse * pulse_w * bass_gain;

        // Palette per ring/angle (for runes & borders)
        float k = (fi / max(1.0, rings)) + 0.22 * a01 + 0.015 * u_time;
        vec3 pal = palette(k, u_palette_id);

        // Stone base: deep blue-violet with slight per-tile variation
        float h = hash11(idx + 37.0 * fi);
        vec3 stone = mix(vec3(0.03, 0.04, 0.10), vec3(0.09, 0.12, 0.32), 0.55 + 0.35 * h);
        stone = mix(stone, pal * 0.22, 0.45);

        // Make the central ring darker (more black than violet) to match the requested look.
        if (fi < 0.5) {
            stone = mix(stone, vec3(0.0), 0.92);
        }
        stone *= tile_fill;

        // Border: brighter violet-ish
        vec3 border_col = pal * 0.75 + vec3(0.02);
        vec3 tile_col = stone + tile_edge * border_col * (0.55 + 0.35 * ring_band);

        // Rune inside each tile
        vec2 qr = vec2(local_ang / max(0.10, u_rune_length), qy);
        float thickness = max(0.002, u_rune_thickness);
        float rune = glyph_mask(qr, idx + 37.0 * fi, thickness, u_style_id) * tile_fill;

        // Rune color (mostly white with palette tint)
        vec3 rune_col = mix(vec3(1.0), pal, 0.40);

        // Add emissive glow feeding bloom
        vec3 emissive = rune_col * rune * (0.95 + 0.65 * u_glow_gain);

        // Pulse adds a brighter/whiter spark on the active ring tiles
        vec3 pulse_col = mix(pal, vec3(1.0), 0.65);
        tile_col += pulse_boost * pulse_col * (0.85 * tile_fill + 0.55 * tile_edge);
        emissive += pulse_boost * pulse_col * rune * 1.35;

        // Outer rings fade slightly (keeps center readable)
        float fade = 1.0 - 0.52 * (fi / max(1.0, rings - 1.0));
        vec3 c = (tile_col * 0.85 + emissive) * fade;

        col += c;
    }

    // Center core (dark core with subtle pulse glow)
    float core_r = inner * 0.55;
    float core = smoothstep(core_r + 0.06, core_r - 0.02, r);
    float core_boost = pulse * exp(-u_pulse_age / max(0.05, spread_total * 0.45));
    float core_hole = smoothstep(inner * 0.15, inner * 0.10, r);

    // Darken the center instead of tinting it purple
    col *= (1.0 - 0.85 * core);

    // Add a subtle white glow on drum pulse (kept intentionally small)
    col += core * vec3(1.0) * (0.06 * core_boost);

    col *= (1.0 - 0.65 * core_hole); // small hole

    col *= global_gain;

    gl_FragColor = vec4(max(col, 0.0), 1.0);
}
"""


# Bloom shaders (same style as other Olaf shaders)
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

    // Tiny grain to avoid banding
    float g = (rand(uv + vec2(u_energy, u_energy * 1.37)) - 0.5) * 0.010 * (0.35 + u_energy);
    col += g;

    float expo = max(0.05, u_exposure);
    float cont = max(0.65, u_contrast);

    col *= expo;
    col = tonemap_aces(col);
    col = pow(col, vec3(0.90 / cont));

    col = pow(max(col, 0.0), vec3(0.4545)); // gamma

    gl_FragColor = vec4(col, 1.0);
}
"""


# ---------------------------------------------------------------------
# VisPy canvas
# ---------------------------------------------------------------------
if HAVE_VISPY:

    class _PulsingRunesCanvas(app.Canvas):
        """Multi-pass pipeline (scene -> bloom -> combine)."""

        def __init__(self, config: Dict[str, Any]) -> None:
            self._config = config
            self._audio = _AudioState()

            # Optional style override (used to keep cycling stable during export
            # where the host may re-apply config snapshots between frames).
            self._style_id_override: Optional[float] = None

            super().__init__(keys=None, size=(640, 360), show=False)

            # If an OpenGL driver rejects our offscreen FBO pipeline (common during
            # export preview grabs on some Windows setups), we fall back to a single-pass
            # draw directly to the default framebuffer so export can continue.
            self._fbo_failed = False

            # Cached OpenGL limits (texture size / viewport dims). Populated lazily
            # on the first draw, when a context is guaranteed to be current.
            self._gl_limits: Optional[Dict[str, Any]] = None

            self._prog_scene = gloo.Program(_VERTEX, _FRAGMENT_SCENE)
            self._prog_prefilter = gloo.Program(_VERTEX, _FRAGMENT_PREFILTER)
            self._prog_blur = gloo.Program(_VERTEX, _FRAGMENT_BLUR)
            self._prog_combine = gloo.Program(_VERTEX, _FRAGMENT_COMBINE)

            verts = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
            v = np.array(verts, dtype="float32") if HAVE_NUMPY else verts

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

        def reset_audio_state(self) -> None:
            """Reset all audio-driven state.

            This is important when the user stops and restarts preview: Olaf may reset the
            timeline back to 0.0s, while the plugin's in-memory detector keeps its previous
            timestamps. Without a reset, (t - last_pulse_t) can stay negative for the entire
            playback, preventing any new triggers.
            """
            self._audio = _AudioState()
            self.update()

        def _get_gl_limits(self) -> Dict[str, Any]:
            """Query and cache a few OpenGL limits (robust for export/offscreen contexts)."""
            if self._gl_limits is not None:
                return self._gl_limits

            # Reasonable conservative defaults (used if glGetIntegerv is unavailable).
            limits: Dict[str, Any] = {"max_tex": 4096, "max_vp": (4096, 4096)}

            def _as_int(value: Any, default: int) -> int:
                try:
                    # vispy may return numpy arrays / sequences
                    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
                        value = list(value)[0]
                    return int(value)
                except Exception:
                    return int(default)

            def _as_pair(value: Any, default: tuple[int, int]) -> tuple[int, int]:
                try:
                    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
                        seq = list(value)
                        if len(seq) >= 2:
                            return (int(seq[0]), int(seq[1]))
                    return default
                except Exception:
                    return default

            try:
                limits["max_tex"] = _as_int(gl.glGetIntegerv(gl.GL_MAX_TEXTURE_SIZE), limits["max_tex"])
            except Exception:
                pass

            try:
                limits["max_vp"] = _as_pair(gl.glGetIntegerv(gl.GL_MAX_VIEWPORT_DIMS), limits["max_vp"])
            except Exception:
                pass

            # Clamp to at least 64px to avoid nonsense values on buggy drivers.
            limits["max_tex"] = max(64, int(limits.get("max_tex", 4096)))
            mv = limits.get("max_vp", (4096, 4096))
            limits["max_vp"] = (max(64, int(mv[0])), max(64, int(mv[1])))

            self._gl_limits = limits
            return limits

        # -----------------------------
        # Resources
        # -----------------------------
        def _ensure_fbos(self) -> None:
            """(Re)allocate offscreen textures/FBOs with driver-safe clamping."""
            # Do not allocate offscreen resources if we already detected a driver issue.
            if getattr(self, "_fbo_failed", False):
                return

            limits = self._get_gl_limits()
            max_tex = int(limits.get("max_tex", 4096))

            w, h = self.physical_size
            w = max(1, int(w))
            h = max(1, int(h))

            # Clamp to the maximum supported texture size. Oversized export previews can otherwise
            # trigger GL_INVALID_VALUE during texture allocation on some GPUs/drivers.
            w = min(w, max_tex)
            h = min(h, max_tex)

            # Bloom downsample is fixed (not exposed as a parameter).
            down = 2

            if (w, h, down) == self._last_sizes and self._scene_tex is not None:
                return

            self._last_sizes = (w, h, down)

            # Scene FBO
            self._scene_tex = self._make_texture(w, h)
            self._scene_fbo = gloo.FrameBuffer(color=self._scene_tex)

            # Bloom FBOs (small but never < 16x16)
            bw = max(16, min(max_tex, w // down))
            bh = max(16, min(max_tex, h // down))
            self._bloom_tex0 = self._make_texture(bw, bh)
            self._bloom_tex1 = self._make_texture(bw, bh)
            self._bloom_fbo0 = gloo.FrameBuffer(color=self._bloom_tex0)
            self._bloom_fbo1 = gloo.FrameBuffer(color=self._bloom_tex1)

        @staticmethod
        def _make_texture(w: int, h: int):
            """Create a small RGBA texture for FBO attachments (export-safe defaults)."""
            w = max(1, int(w))
            h = max(1, int(h))

            # Prefer 8-bit textures for maximum driver compatibility (float textures + linear
            # filtering can be problematic on some Windows/ANGLE configurations).
            if HAVE_NUMPY:
                data = np.zeros((h, w, 4), dtype=np.uint8)
                return gloo.Texture2D(data, interpolation="linear")

            return gloo.Texture2D((h, w, 4), interpolation="linear")

        def _update_uniforms(self, scene_w: int, scene_h: int, screen_w: int, screen_h: int) -> None:
            """Update shader uniforms from the current config + audio state.

            IMPORTANT:
            - Use *scene* resolution for the scene pass (FBO) and *screen* resolution for the final combine.
            - Never leave resolution at (0, 0) (export preview grabs can transiently report 0-sized widgets).
            """
            scene_w = max(1, int(scene_w))
            scene_h = max(1, int(scene_h))
            screen_w = max(1, int(screen_w))
            screen_h = max(1, int(screen_h))

            # Scene shader expects its own render resolution.
            self._prog_scene["u_resolution"] = (float(scene_w), float(scene_h))

            # Common time/audio
            self._prog_scene["u_time"] = float(self._audio.t)
            self._prog_scene["u_energy"] = float(self._audio.energy)
            self._prog_scene["u_bass"] = float(self._audio.bass)

            # Geometry
            self._prog_scene["u_rings"] = float(_clamp(_safe_float(self._config.get("rings", 7), 7), 1.0, 16.0))
            self._prog_scene["u_inner_radius"] = float(_clamp(_safe_float(self._config.get("inner_radius", 0.18), 0.18), 0.02, 0.80))
            self._prog_scene["u_ring_thickness"] = float(_clamp(_safe_float(self._config.get("ring_thickness", 0.10), 0.10), 0.008, 0.40))
            self._prog_scene["u_ring_gap"] = float(_clamp(_safe_float(self._config.get("ring_gap", 0.00), 0.00), 0.00, 0.25))

            # Camera distance can optionally be modulated by input_1 intensity.
            base_cam = float(_clamp(_safe_float(self._config.get("camera_distance", 0.50), 0.50), 0.35, 3.00))
            cam_amount = float(_clamp(_safe_float(self._config.get("camera_distance_audio_amount", 0.0), 0.0), -2.0, 2.0))
            cam_e = float(_clamp(self._audio.camera_energy, 0.0, 1.5)) / 1.5  # 0..1
            cam_dist = base_cam + (cam_amount * cam_e)
            self._prog_scene["u_camera_distance"] = float(_clamp(cam_dist, 0.35, 3.00))

            # Glyph layout
            self._prog_scene["u_runes_per_ring"] = float(_clamp(_safe_float(self._config.get("runes_per_ring", 24), 24), 3.0, 128.0))
            self._prog_scene["u_rune_step"] = float(_clamp(_safe_float(self._config.get("runes_step", 3), 3), 0.0, 16.0))
            self._prog_scene["u_rune_length"] = float(_clamp(_safe_float(self._config.get("rune_length", 0.80), 0.80), 0.15, 2.0))
            self._prog_scene["u_rune_thickness"] = float(_clamp(_safe_float(self._config.get("rune_thickness", 0.015), 0.015), 0.003, 0.060))
            # Rotation (audio-reactive)
            base_speed = float(_clamp(_safe_float(self._config.get("rotation_speed", 0.55), 0.55), -6.0, 6.0))
            base_step  = float(_clamp(_safe_float(self._config.get("rotation_step", 0.08), 0.08), -1.5, 1.5))
            react = float(_clamp(_safe_float(self._config.get("rotation_audio_reactivity", 0.8), 0.8), 0.0, 3.0))
            # Louder sections spin faster. Use a dedicated (faster) smoothed envelope so
            # rotation remains responsive even when the host RMS updates are sparse.
            e = float(_clamp(self._audio.rotation_energy, 0.0, 1.5))
            mult = 1.0 + (react * e * 0.05)
            self._prog_scene["u_rotation_speed"] = base_speed * mult
            self._prog_scene["u_rotation_step"]  = base_step * mult

            # Pulse state + propagation timing
            self._prog_scene["u_active_ring"] = float(self._audio.active_ring)
            self._prog_scene["u_pulse"] = float(self._audio.pulse)
            self._prog_scene["u_pulse_trail"] = float(_clamp(_safe_float(self._config.get("pulse_trail", 0.35), 0.35), 0.10, 4.0))
            self._prog_scene["u_pulse_gain"] = float(_clamp(_safe_float(self._config.get("pulse_gain", 6.0), 6.0), 0.0, 20.0))

            # Age since last pulse (seconds). Used by the shader to step the highlight ring-by-ring.
            last_pulse_t = float(getattr(self._audio, "_last_pulse_t", -999.0))
            pulse_age = max(0.0, float(self._audio.t) - last_pulse_t)
            self._prog_scene["u_pulse_age"] = float(pulse_age)

            # Per-ring step delay (seconds)
            self._prog_scene["u_pulse_step_s"] = float(_clamp(_safe_float(self._config.get("pulse_step_delay_s", 0.06), 0.06), 0.01, 0.50))

            # Colorway -> palette ID (float)
            cw = self._config.get("colorway_id", None)
            palette_id = 0.0
            if isinstance(cw, str) and cw in _COLORWAY_TO_ID:
                palette_id = float(_COLORWAY_TO_ID.get(cw, 0.0))
            else:
                # Backward compatibility: `colorway` string, legacy `color_preset`, or int index.
                cw_name = str(self._config.get("colorway") or "")
                if cw_name not in _COLORWAY_TO_ID:
                    legacy_palette = str(self._config.get("color_preset") or "")
                    cw_name = _LEGACY_COLOR_PRESET_TO_COLORWAY.get(legacy_palette, _COLORWAYS[0])
                if cw_name in _COLORWAY_TO_ID:
                    palette_id = float(_COLORWAY_TO_ID.get(cw_name, 0.0))
                else:
                    try:
                        palette_id = float(int(cw)) if cw is not None else 0.0
                    except Exception:
                        palette_id = 0.0
            self._prog_scene["u_palette_id"] = float(_clamp(palette_id, 0.0, float(len(_COLORWAYS) - 1)))

            # Style selector (the actual "design" switch)
            style_val = self._style_id_override if self._style_id_override is not None else self._config.get("style_id", 0.0)
            self._prog_scene["u_style_id"] = float(_clamp(_safe_float(style_val, 0.0), 0.0, float(len(_STYLE_PRESETS) - 1)))

            # Emissive glow feeding bloom
            self._prog_scene["u_glow_gain"] = float(_clamp(_safe_float(self._config.get("glow_gain", 1.25), 1.25), 0.0, 4.0))

            # Combine shader uses the *screen* resolution.
            self._prog_combine["u_resolution"] = (float(screen_w), float(screen_h))
            self._prog_combine["u_energy"] = float(self._audio.energy)

            self._prog_combine["u_exposure"] = float(_clamp(_safe_float(self._config.get("exposure", 1.15), 1.15), 0.10, 5.0))
            self._prog_combine["u_contrast"] = float(_clamp(_safe_float(self._config.get("contrast", 1.10), 1.10), 0.60, 2.0))

            # Bloom combine intensity
            self._prog_combine["u_bloom_intensity"] = float(_clamp(_safe_float(self._config.get("bloom_intensity", 1.25), 1.25), 0.0, 3.0))

            # Post FX
            self._prog_combine["u_chroma"] = float(_clamp(_safe_float(self._config.get("chromatic_aberration", 0.008), 0.008), 0.0, 0.03))
            self._prog_combine["u_scanlines"] = float(_clamp(_safe_float(self._config.get("scanlines", 0.0), 0.0), 0.0, 1.0))
            self._prog_combine["u_vignette"] = float(_clamp(_safe_float(self._config.get("vignette", 0.32), 0.32), 0.0, 1.0))


        def set_audio_features(
            self,
            *,
            time_s: float,
            energy: float,
            bass: float,
            pulse_trigger: bool,
            style_id: Optional[float] = None,
        ) -> None:
            # Time + smoothing
            prev_t = float(self._audio.t)
            self._audio.t = float(time_s)
            dt = float(self._audio.t - prev_t)
            # Export preview grabs / timeline resets may provide a non-monotonic clock.
            # We still want stable smoothing coefficients.
            if dt <= 0.0 or dt > 1.0:
                dt = 1.0 / 60.0

            # Style override: when provided by the plugin, it takes precedence over
            # config['style_id'] (helps during export where config may be snapshotted).
            if style_id is not None:
                try:
                    self._style_id_override = float(style_id)
                except Exception:
                    self._style_id_override = None

            a_e = 0.25
            a_b = 0.28
            e_in = float(_clamp(energy, 0.0, 2.0))
            b_in = float(_clamp(bass, 0.0, 2.0))
            self._audio.energy = (1.0 - a_e) * self._audio.energy + a_e * e_in
            self._audio.bass = (1.0 - a_b) * self._audio.bass + a_b * b_in

            # Faster envelope used for rotation modulation (user-tunable in seconds).
            rot_tau = float(_clamp(_safe_float(self._config.get("rotation_audio_smoothing_s", 0.03), 0.03), 0.0, 2.0))
            if rot_tau <= 1e-6:
                rot_alpha = 1.0
            else:
                rot_alpha = 1.0 - math.exp(-dt / rot_tau)
            rot_alpha = float(_clamp(rot_alpha, 0.0, 1.0))
            self._audio.rotation_energy = (1.0 - rot_alpha) * self._audio.rotation_energy + rot_alpha * e_in

            # Camera envelope (defaults to slightly smoother to avoid jitter).
            cam_tau = float(_clamp(_safe_float(self._config.get("camera_distance_audio_smoothing_s", 0.18), 0.18), 0.0, 2.0))
            if cam_tau <= 1e-6:
                cam_alpha = 1.0
            else:
                cam_alpha = 1.0 - math.exp(-dt / cam_tau)
            cam_alpha = float(_clamp(cam_alpha, 0.0, 1.0))
            self._audio.camera_energy = (1.0 - cam_alpha) * self._audio.camera_energy + cam_alpha * e_in

            # Pulse envelope
            decay = float(_clamp(_safe_float(self._config.get("pulse_decay", 0.86), 0.86), 0.50, 0.98))
            self._audio.pulse *= decay

            if pulse_trigger:
                # Reset the pulse "age" so the wave starts at the center again.
                # The shader uses (t - _last_pulse_t) to advance ring-by-ring.
                self._audio._last_pulse_t = float(self._audio.t)
                self._audio.active_ring = 0.0  # legacy/diagnostic; not required by the shader
                self._audio.pulse = 1.0
                self._audio._last_pulse_t = float(self._audio.t)

            self.update()

        # -----------------------------
        # VisPy callbacks
        # -----------------------------
        def on_resize(self, event) -> None:  # type: ignore[override]
            gloo.set_viewport(0, 0, *event.physical_size)
            self.update()

        def on_draw(self, event) -> None:  # type: ignore[override]
            """Render one frame (robust for export preview grabs)."""
            limits = self._get_gl_limits()
            max_vp_w, max_vp_h = limits.get("max_vp", (4096, 4096))

            # Clamp screen viewport to avoid GL_INVALID_VALUE on oversized export previews.
            w, h = self.physical_size
            screen_w = max(1, min(int(w), int(max_vp_w)))
            screen_h = max(1, min(int(h), int(max_vp_h)))

            # If FBO path failed previously, always use the single-pass fallback.
            if getattr(self, "_fbo_failed", False):
                try:
                    self._update_uniforms(scene_w=screen_w, scene_h=screen_h, screen_w=screen_w, screen_h=screen_h)
                    gloo.set_viewport(0, 0, screen_w, screen_h)
                    gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                    self._prog_scene.draw("triangle_strip")
                except Exception:
                    # Last-resort: swallow draw errors during grab so export can continue.
                    return
                return

            # Allocate/refresh offscreen resources
            self._ensure_fbos()
            if self._scene_tex is None or self._scene_fbo is None:
                return
            if self._bloom_tex0 is None or self._bloom_tex1 is None:
                return
            if self._bloom_fbo0 is None or self._bloom_fbo1 is None:
                return

            # Use actual FBO sizes (which may be clamped to GL limits).
            scene_h, scene_w = self._scene_tex.shape[0], self._scene_tex.shape[1]
            bloom_h, bloom_w = self._bloom_tex0.shape[0], self._bloom_tex0.shape[1]

            # Update uniforms with per-target resolutions.
            self._update_uniforms(scene_w=scene_w, scene_h=scene_h, screen_w=screen_w, screen_h=screen_h)

            try:
                # 1) Scene -> scene_fbo
                with self._scene_fbo:
                    gloo.set_viewport(0, 0, scene_w, scene_h)
                    gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                    self._prog_scene.draw("triangle_strip")

                # 2) Prefilter/downsample -> bloom0
                self._prog_prefilter["u_scene"] = self._scene_tex
                
                # Bloom prefilter controls (kept conservative for export stability).
                self._prog_prefilter["u_threshold"] = float(_clamp(_safe_float(self._config.get("bloom_threshold", 0.85), 0.85), 0.0, 5.0))
                self._prog_prefilter["u_soft_knee"] = float(_clamp(_safe_float(self._config.get("bloom_soft_knee", 0.50), 0.50), 0.0, 1.0))
                with self._bloom_fbo0:
                    gloo.set_viewport(0, 0, bloom_w, bloom_h)
                    gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                    self._prog_prefilter.draw("triangle_strip")

                # 3) Blur horizontal bloom0 -> bloom1
                self._prog_blur["u_tex"] = self._bloom_tex0
                self._prog_blur["u_texel"] = (1.0 / float(bloom_w), 1.0 / float(bloom_h))
                self._prog_blur["u_dir"] = (1.0, 0.0)
                
                self._prog_blur["u_radius"] = float(_clamp(_safe_float(self._config.get("bloom_radius", 1.25), 1.25), 0.1, 6.0))
                with self._bloom_fbo1:
                    gloo.set_viewport(0, 0, bloom_w, bloom_h)
                    gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                    self._prog_blur.draw("triangle_strip")

                # 4) Blur vertical bloom1 -> bloom0
                self._prog_blur["u_tex"] = self._bloom_tex1
                self._prog_blur["u_dir"] = (0.0, 1.0)
                
                self._prog_blur["u_radius"] = float(_clamp(_safe_float(self._config.get("bloom_radius", 1.25), 1.25), 0.1, 6.0))
                with self._bloom_fbo0:
                    gloo.set_viewport(0, 0, bloom_w, bloom_h)
                    gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                    self._prog_blur.draw("triangle_strip")

                # 5) Combine -> screen
                self._prog_combine["u_scene"] = self._scene_tex
                self._prog_combine["u_bloom"] = self._bloom_tex0
                gloo.set_viewport(0, 0, screen_w, screen_h)
                gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                self._prog_combine.draw("triangle_strip")

            except Exception:
                # If anything goes wrong (often during widget.grab()), switch to a safe fallback.
                self._fbo_failed = True
                try:
                    self._update_uniforms(scene_w=screen_w, scene_h=screen_h, screen_w=screen_w, screen_h=screen_h)
                    gloo.set_viewport(0, 0, screen_w, screen_h)
                    gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                    self._prog_scene.draw("triangle_strip")
                except Exception:
                    return

class _PulsingRunesWidget(QWidget):
    """Qt widget embedding the VisPy canvas."""

    def __init__(self, config: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._canvas = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not HAVE_VISPY:
            label = QLabel(
                "VisPy is not available.\n"
                "Pulsing Runes is disabled.\n"
                "Please install 'vispy' to enable this visualization.",
                self,
            )
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            return

        app.use_app("pyqt6")  # type: ignore[arg-type]
        self._canvas = _PulsingRunesCanvas(config=self._config)  # type: ignore[name-defined]
        self._canvas.native.setParent(self)
        layout.addWidget(self._canvas.native)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(640, 360)

    def update_audio_features(self, *, time_ms: int, energy: float, bass: float, pulse_trigger: bool, style_id: Optional[float] = None) -> None:
        if self._canvas is None:
            return
        self._canvas.set_audio_features(
            time_s=float(time_ms) / 1000.0,
            energy=float(energy),
            bass=float(bass),
            pulse_trigger=bool(pulse_trigger),
            style_id=style_id,
        )

    def reset_audio_state(self) -> None:
        """Reset the canvas audio state.

        On some backends, preview restart resets the timeline but not the plugin instance.
        We expose an explicit reset hook so the detector and pulse envelope do not get
        stuck with stale timestamps.
        """
        if self._canvas is None:
            return
        try:
            self._canvas.reset_audio_state()
        except Exception:
            pass


# ---------------------------------------------------------------------
# Olaf plugin class
# ---------------------------------------------------------------------
class SDFPulsingRunesVisualization(BaseVisualization):
    plugin_id = "sdf_pulsing_runes"
    plugin_name = "SDF Pulsing Glyph Rings"
    plugin_description = "Top-down concentric glyph rings rotating alternately; input_2 (drums) triggers a stepped center-to-outside pulse (ring-by-ring). Includes style presets."
    plugin_author = "DrDLP"
    plugin_version = "0.7.9"
    plugin_max_inputs = 2

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._widget: Optional[_PulsingRunesWidget] = None

        # Visual defaults
        self.config.setdefault("rings", 7)
        self.config.setdefault("inner_radius", 0.18)
        self.config.setdefault("ring_thickness", 0.10)
        self.config.setdefault("ring_gap", 0.0)
        self.config.setdefault("camera_distance", 0.5)

        self.config.setdefault("runes_per_ring", 24)
        self.config.setdefault("runes_step", 3)
        self.config.setdefault("rune_length", 0.80)
        self.config.setdefault("rune_thickness", 0.015)

        self.config.setdefault("rotation_speed", 0.55)
        self.config.setdefault("rotation_step", 0.08)
        self.config.setdefault("rotation_audio_reactivity", 0.8)
        # Smaller values => faster response (more frequent rotation modulation changes).
        self.config.setdefault("rotation_audio_smoothing_s", 0.03)

        # Optional camera distance modulation driven by input_1 intensity.
        # A positive amount moves the camera further away on louder sections.
        self.config.setdefault("camera_distance_audio_amount", 0.0)
        self.config.setdefault("camera_distance_audio_smoothing_s", 0.18)

        self.config.setdefault("pulse_decay", 0.92)
        # In Olaf preview, per-input features are currently RMS-only.
        # A low default makes transients detectable without forcing users
        # to set an extremely low value manually.
        self.config.setdefault("pulse_threshold", 0.01)
        self.config.setdefault("pulse_cooldown_s", 0.12)
        self.config.setdefault("pulse_spread_s", 0.35)
        # Delay between ring steps for the pulse propagation (one ring jump every N seconds).
        # Backward compatible: if an older preset only stores pulse_spread_s (total travel time),
        # we derive a per-ring delay using (rings - 1).
        if "pulse_step_delay_s" not in self.config:
            _rings = int(self.config.get("rings", 7) or 7)
            _total = float(self.config.get("pulse_spread_s", 0.35) or 0.35)
            self.config["pulse_step_delay_s"] = max(0.01, _total / max(1, _rings - 1))
        self.config.setdefault("pulse_trail", 0.35)

        self.config.setdefault("pulse_gain", 6.0)

        self.config.setdefault("glow_gain", 1.25)

        # Bloom / post defaults (kept conservative for export stability)
        self.config.setdefault("bloom_threshold", 0.85)
        self.config.setdefault("bloom_soft_knee", 0.50)
        self.config.setdefault("bloom_radius", 1.25)
        self.config.setdefault("bloom_intensity", 1.25)

        self.config.setdefault("style_id", 0.0)

        # Presets
        # NOTE: Some Olaf UI builds only expose a single enum dropdown. We prioritize the
        # *render* preset as an enum (so it is selectable as a combo box) and represent the
        # colorway as a small int slider (0..2). To keep backward compatibility, we still
        # accept legacy `color_preset` and the older `colorway` string.
        #
        # Cycling (pulse-triggered preset cycling)
        self.config.setdefault("cycling_enabled", False)
        self.config.setdefault("cycling_interval_s", 3.0)

        # Colorway (stored as a string choice). Backward compatible with older int ids.
        cw = self.config.get("colorway_id", None)
        if isinstance(cw, str) and cw in _COLORWAY_TO_ID:
            cw_name = cw
        else:
            cw_name = ""
            if isinstance(cw, (int, float)):
                idx = int(cw)
                if 0 <= idx < len(_COLORWAYS):
                    cw_name = _COLORWAYS[idx]
            if not cw_name:
                cw_name = str(self.config.get("colorway") or "")
            if cw_name not in _COLORWAY_TO_ID:
                legacy_palette = str(self.config.get("color_preset") or "")
                cw_name = _LEGACY_COLOR_PRESET_TO_COLORWAY.get(legacy_palette, _COLORWAYS[0])

        self.config["colorway_id"] = cw_name

        # Render preset (enum string). Backward compatible with older int-based configs.
        rp = self.config.get("render_preset", _STYLE_PRESETS[0])
        if isinstance(rp, (int, float)):
            rp_idx = int(rp)
            rp_idx = max(0, min(len(_STYLE_PRESETS) - 1, rp_idx))
            rp_name = _STYLE_PRESETS[rp_idx]
        else:
            rp_name = str(rp)
            if rp_name not in _STYLE_PRESETS:
                rp_name = _STYLE_PRESETS[0]
        self.config["render_preset"] = rp_name

        # Remember last applied preset name to avoid overwriting user configs continuously.
        self._last_render_preset: str = str(self.config.get("render_preset", _STYLE_PRESETS[0]))
        self._last_cycle_t: float = -999.0  # last time a cycle was applied
        # Runtime style state (export-safe): some export backends may re-apply
        # config snapshots between frames. We keep the current cycled style here
        # and push it to the canvas every frame.
        self._runtime_render_preset: str = str(self.config.get("render_preset", _STYLE_PRESETS[0]) or _STYLE_PRESETS[0])
        self._runtime_style_id: float = float(self.config.get("style_id", 0.0) or 0.0)
# Post-process defaults

        self.config.setdefault("chromatic_aberration", 0.008)
        self.config.setdefault("exposure", 1.15)
        self.config.setdefault("contrast", 1.10)
        self.config.setdefault("scanlines", 0.0)
        self.config.setdefault("vignette", 0.32)

        # Internal pulse detector state
        self._prev_bass: float = 0.0
        self._last_pulse_t: float = -999.0
        self._prev_t_s: float = -1.0

        # Export preview in export_tab builds a dummy features payload with time_ms=0.
        # To keep pulse detection and cycling functional in that mode, we synthesize a
        # monotonic time when the host does not advance the timestamp.
        self._synth_time_ms: int = 0
        self._last_seen_time_ms: int = -1

    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        return {
            "render_preset": PluginParameter(
                name="render_preset",
                label="Render preset",
                type="enum",
                default=_STYLE_PRESETS[0],
                choices=_STYLE_PRESETS,
                description="Select a visual style bundle (one-shot initializer).",
            ),
                        "cycling_enabled": PluginParameter(
                name="cycling_enabled",
                label="Cycling",
                type="bool",
                default=False,
                description="If enabled, cycle to the next render preset when a pulse triggers.",
            ),
            "cycling_interval_s": PluginParameter(
                name="cycling_interval_s",
                label="Cycling interval (s)",
                type="float",
                default=3.0,
                minimum=0.0,
                maximum=30.0,
                step=0.1,
                description="Minimum time between cycles (in seconds). 0 = cycle on every pulse.",
            ),

"colorway_id": PluginParameter(
                name="colorway_id",
                label="Colorway",
                type="enum",
                default=_COLORWAYS[0],
                choices=_COLORWAYS,
                description="Select a neon colorway preset.",
            ),

            "rings": PluginParameter(
                name="rings",
                label="Rings",
                type="int",
                default=7,
                minimum=1,
                maximum=16,
                step=1,
                description="Number of concentric rune rings.",
            ),
            "inner_radius": PluginParameter(
                name="inner_radius",
                label="Inner radius",
                type="float",
                default=0.18,
                minimum=0.02,
                maximum=0.8,
                step=0.01,
                description="Radius of the innermost ring (normalized).",
            ),
            "ring_thickness": PluginParameter(
                name="ring_thickness",
                label="Ring thickness",
                type="float",
                default=0.10,
                minimum=0.008,
                maximum=0.40,
                step=0.01,
                description="Radial thickness of each ring (normalized).",
            ),
            "ring_gap": PluginParameter(
                name="ring_gap",
                label="Ring height",
                type="float",
                default=0.00,
                minimum=0.00,
                maximum=0.25,
                step=0.005,
                description="Extra ring height (adds thickness) without creating spacing. 0 = baseline thickness.",
            ),
            "camera_distance": PluginParameter(
                name="camera_distance",
                label="Camera distance",
                type="float",
                default=0.5,
                minimum=0.35,
                maximum=3.00,
                step=0.01,
                description="Top-down zoom-out factor (higher = further away).",
            ),

            "camera_distance_audio_amount": PluginParameter(
                name="camera_distance_audio_amount",
                label="Camera distance (audio amount)",
                type="float",
                default=0.0,
                minimum=-2.0,
                maximum=2.0,
                step=0.01,
                description=(
                    "Adds (amount * input_1 intensity) to the camera distance. "
                    "Positive values pull the camera back on loud sections; negative values push in."
                ),
            ),
            "camera_distance_audio_smoothing_s": PluginParameter(
                name="camera_distance_audio_smoothing_s",
                label="Camera distance smoothing (s)",
                type="float",
                default=0.18,
                minimum=0.0,
                maximum=2.0,
                step=0.01,
                description="Smoothing time-constant for the camera modulation (0 = no smoothing).",
            ),


            "runes_per_ring": PluginParameter(
                name="runes_per_ring",
                label="Runes per ring",
                type="int",
                default=24,
                minimum=3,
                maximum=128,
                step=1,
                description="How many rune 'slots' on the inner ring.",
            ),
            "runes_step": PluginParameter(
                name="runes_step",
                label="Runes step",
                type="int",
                default=3,
                minimum=0,
                maximum=16,
                step=1,
                description="Additional rune slots per outer ring.",
            ),
            "rune_length": PluginParameter(
                name="rune_length",
                label="Rune length",
                type="float",
                default=0.80,
                minimum=0.15,
                maximum=2.0,
                step=0.01,
                description="Tangent stretch of rune glyphs (higher = shorter glyphs).",
            ),
            "rune_thickness": PluginParameter(
                name="rune_thickness",
                label="Rune thickness",
                type="float",
                default=0.015,
                minimum=0.003,
                maximum=0.06,
                step=0.001,
                description="Stroke thickness for rune lines.",
            ),

            "rotation_speed": PluginParameter(
                name="rotation_speed",
                label="Rotation speed",
                type="float",
                default=0.55,
                minimum=-6.0,
                maximum=6.0,
                step=0.01,
                description="Base angular speed (radians/sec). Rings alternate direction automatically.",
            ),
            "rotation_step": PluginParameter(
                name="rotation_step",
                label="Rotation step",
                type="float",
                default=0.08,
                minimum=-1.5,
                maximum=1.5,
                step=0.01,
                description="Extra angular speed per ring index (can be negative).",
            ),
            "rotation_audio_reactivity": PluginParameter(
                name="rotation_audio_reactivity",
                label="Rotation audio reactivity",
                type="float",
                default=0.8,
                minimum=0.0,
                maximum=3.0,
                step=0.05,
                description="How strongly input_1 energy multiplies the rotation speed (0 = no audio reactivity).",
            ),

            "rotation_audio_smoothing_s": PluginParameter(
                name="rotation_audio_smoothing_s",
                label="Rotation smoothing (s)",
                type="float",
                default=0.03,
                minimum=0.0,
                maximum=2.0,
                step=0.01,
                description=(
                    "Smoothing time-constant for the rotation modulation (0 = immediate). "
                    "Lower values make the audio-reactive rotation update more frequently."
                ),
            ),


            "pulse_threshold": PluginParameter(
                name="pulse_threshold",
                label="Pulse threshold (ΔRMS)",
                type="float",
                default=0.01,
                minimum=0.0,
                maximum=0.2,
                step=0.001,
                description=(
                    "Pulse triggers when the source RMS rises faster than this amount between frames. "
                    "In Preview mode the stems usually expose only 'rms', so small thresholds (e.g. 0.01) are expected."
                ),
            ),
            "pulse_cooldown_s": PluginParameter(
                name="pulse_cooldown_s",
                label="Pulse cooldown (s)",
                type="float",
                default=0.12,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="Minimum time between pulses.",
            ),            "pulse_step_delay_s": PluginParameter(
                name="pulse_step_delay_s",
                label="Pulse step delay (s)",
                type="float",
                default=0.06,
                minimum=0.01,
                maximum=0.50,
                step=0.005,
                description="Delay between ring jumps of the pulse (center -> outer).",
            ),
            "pulse_decay": PluginParameter(
                name="pulse_decay",
                label="Pulse decay",
                type="float",
                default=0.86,
                minimum=0.50,
                maximum=0.98,
                step=0.01,
                description="How quickly the pulse highlight fades (closer to 1 = longer).",
            ),
            "pulse_trail": PluginParameter(
                name="pulse_trail",
                label="Pulse trail",
                type="float",
                default=0.35,
                minimum=0.10,
                maximum=4.0,
                step=0.05,
                description="How many rings are softly affected around the active ring.",
            ),
            "pulse_gain": PluginParameter(
                name="pulse_gain",
                label="Pulse intensity",
                type="float",
                default=6.0,
                minimum=0.0,
                maximum=20.0,
                step=0.25,
                description="Intensity multiplier for the ring-by-ring pulse (input 2).",
            ),
            "glow_gain": PluginParameter(
                name="glow_gain",
                label="Glow gain",
                type="float",
                default=1.25,
                minimum=0.0,
                maximum=4.0,
                step=0.01,
                description="Extra emissive glow multiplier (feeds bloom).",
            ),

            # Post-process
            "chromatic_aberration": PluginParameter(
                name="chromatic_aberration",
                label="Chromatic aberration",
                type="float",
                default=0.008,
                minimum=0.0,
                maximum=0.03,
                step=0.001,
                description="Post-process RGB split amount.",
            ),
            "exposure": PluginParameter(
                name="exposure",
                label="Exposure",
                type="float",
                default=1.15,
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
                description="Contrast shaping.",
            ),
            "scanlines": PluginParameter(
                name="scanlines",
                label="Scanlines",
                type="float",
                default=0.0,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="CRT-like scanlines.",
            ),
            "vignette": PluginParameter(
                name="vignette",
                label="Vignette",
                type="float",
                default=0.32,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="Darken edges.",
            ),
        }

    def _apply_render_preset_if_needed(self) -> None:
        """Apply the selected render preset exactly once when it changes.

        Presets are treated as a *one-shot* initializer: selecting a preset writes a coherent
        bundle of settings into `self.config`, then the user can fine-tune sliders without
        being overridden on every frame.
        """
        preset_name = str(self.config.get("render_preset", _STYLE_PRESETS[0]) or _STYLE_PRESETS[0])
        if preset_name not in _STYLE_PRESETS:
            # Backward compatibility: numeric preset
            try:
                preset_idx = int(float(preset_name))
                preset_idx = max(0, min(len(_STYLE_PRESETS) - 1, preset_idx))
                preset_name = _STYLE_PRESETS[preset_idx]
            except Exception:
                preset_name = _STYLE_PRESETS[0]

        if preset_name == getattr(self, "_last_render_preset", ""):
            return

        bundle = _STYLE_PRESET_CONFIGS.get(preset_name)
        if isinstance(bundle, dict):
            for k, v in bundle.items():
                # Do not overwrite pulse detector thresholds (dependent on stems/backend).
                if k in ("pulse_threshold", "pulse_cooldown_s", "pulse_decay"):
                    continue
                self.config[k] = v

            # Make the shader dispatch explicit (also helps when exporting configs).
            self.config["style_id"] = float(bundle.get("style_id", float(_STYLE_PRESETS.index(preset_name))))

        self._last_render_preset = preset_name

        # Keep runtime state in sync (used by export-safe cycling).
        self._runtime_render_preset = preset_name
        self._runtime_style_id = float(self.config.get("style_id", 0.0) or 0.0)


    def _maybe_cycle_on_pulse(self, *, t_s: float, pulse_trigger: bool) -> None:
        """Cycle to the next render preset on a pulse trigger.

        This cycling mode is intentionally *non-destructive*:
        - It changes `render_preset` + `style_id` so the glyph design changes.
        - It does NOT re-apply the full preset bundle (which would overwrite user tweaks).
        """
        if not pulse_trigger:
            return
        if not bool(self.config.get("cycling_enabled", False)):
            return

        try:
            interval = float(self.config.get("cycling_interval_s", 3.0) or 0.0)
        except Exception:
            interval = 0.0
        if interval < 0.0:
            interval = 0.0

        # Interval is evaluated *only when a pulse happens*.
        # Special case: 0s => cycle on every pulse.
        if interval > 0.0 and (float(t_s) - float(getattr(self, "_last_cycle_t", -999.0))) < interval:
            return

        current = str(self.config.get("render_preset", _STYLE_PRESETS[0]) or _STYLE_PRESETS[0])
        if current not in _STYLE_PRESETS:
            current = _STYLE_PRESETS[0]
        idx = _STYLE_PRESETS.index(current)

        next_name = _STYLE_PRESETS[(idx + 1) % len(_STYLE_PRESETS)]
        # Update runtime state first (this is what the canvas will actually use).
        self._runtime_render_preset = next_name
        self.config["render_preset"] = next_name

        bundle = _STYLE_PRESET_CONFIGS.get(next_name, {}) if isinstance(_STYLE_PRESET_CONFIGS.get(next_name), dict) else {}
        self.config["style_id"] = float(bundle.get("style_id", float(_STYLE_PRESETS.index(next_name))))
        self._runtime_style_id = float(self.config.get("style_id", 0.0) or 0.0)

        # Prevent `_apply_render_preset_if_needed()` from writing the whole bundle on next frame.
        self._last_render_preset = next_name
        self._last_cycle_t = float(t_s)



    def create_widget(self, parent: Optional[QWidget] = None) -> QWidget:
        self._widget = _PulsingRunesWidget(config=self.config, parent=parent)
        return self._widget

    def on_audio_features(self, features: Dict[str, Any]) -> None:
        """Receive Olaf audio features and drive a stepped ring-by-ring pulse from input 2 (drums).

        We keep a small amount of state for transient detection. On some backends, stopping and
        restarting preview resets the timeline to 0.0s while keeping the plugin instance alive,
        so we detect time "rewinds" and reset our detector state.
        """
        if self._widget is None:
            return

        # Export safety: some backends re-apply a config snapshot between frames.
        # When cycling is enabled, we keep the runtime preset/style authoritative.
        if bool(self.config.get("cycling_enabled", False)):
            self.config["render_preset"] = self._runtime_render_preset
            self.config["style_id"] = float(self._runtime_style_id)
        self._apply_render_preset_if_needed()

        # Build a robust timestamp (export preview may keep time_ms=0).
        raw_ms = features.get("time_ms", features.get("t_ms", None))
        if raw_ms is None:
            # Some hosts use seconds-based keys.
            for sec_key in ("time_sec", "time_s", "t_s", "time_seconds"):
                if sec_key in features:
                    try:
                        raw_ms = int(round(float(features.get(sec_key) or 0.0) * 1000.0))
                    except Exception:
                        raw_ms = 0
                    break
        try:
            time_ms = int(raw_ms) if raw_ms is not None else 0
        except Exception:
            time_ms = 0

        # If the host does not advance time (export preview dummy payload), synthesize
        # a monotonic clock from dt/fps so cooldown/cycling continue to work.
        if time_ms <= self._last_seen_time_ms:
            dt = features.get("dt", None)
            if dt is None:
                fps = features.get("fps", None)
                try:
                    dt = 1.0 / float(fps) if fps else 1.0 / 30.0
                except Exception:
                    dt = 1.0 / 30.0
            try:
                self._synth_time_ms += int(round(float(dt) * 1000.0))
            except Exception:
                self._synth_time_ms += 33
            time_ms = self._synth_time_ms
        else:
            self._synth_time_ms = time_ms
            self._last_seen_time_ms = time_ms

        t_s = float(time_ms) / 1000.0

        # Preview restart / seek detection (timeline went backwards)
        if self._prev_t_s >= 0.0 and (t_s + 0.25) < self._prev_t_s:
            self._prev_bass = 0.0
            self._last_pulse_t = -999.0
            self._last_cycle_t = -999.0
            try:
                self._widget.reset_audio_state()
            except Exception:
                pass

        inputs_any = features.get("inputs", None)

        def _pick_input(n: int) -> Dict[str, Any]:
            # 1) Standard mapping: features["inputs"] is a dict
            if isinstance(inputs_any, dict):
                for k in (f"input_{n}", f"input{n}", str(n), n):
                    v = inputs_any.get(k)
                    if isinstance(v, dict):
                        return v

            # 2) Alternate: features["inputs"] is a list/tuple
            if isinstance(inputs_any, (list, tuple)) and len(inputs_any) >= n:
                v = inputs_any[n - 1]
                if isinstance(v, dict):
                    return v

            # 3) Some backends store input dicts at the root level
            for k in (f"input_{n}", f"input{n}", str(n), n):
                v = features.get(k)
                if isinstance(v, dict):
                    return v

            return {}

        # Mix fallback: the root dict is often the only source available in preview.
        mix_src: Dict[str, Any] = features

        inp1 = _pick_input(1) or mix_src
        inp2 = _pick_input(2)

        # Main energy (music bed)
        energy = _extract_scalar(
            inp1,
            ("rms", "energy", "level", "loudness", "amplitude", "gain", "rms_db", "level_db", "loudness_db"),
        )
        if energy is None:
            energy = 0.0

        # Pulse driver (input 2). If missing, fall back silently to Mix so preview still animates.
        drum_src: Dict[str, Any] = inp2 if isinstance(inp2, dict) and len(inp2) else mix_src

        # Try to grab a bass/kick-related band first; otherwise fallback to generic scalar energy.
        bass = _extract_band_energy(drum_src, ("bass", "low", "kick", "sub", "low_energy", "bass_energy"))  # type: ignore[arg-type]
        if bass is None:
            bass = _extract_scalar(
                drum_src,
                ("bass", "low", "kick", "sub", "low_energy", "bass_energy", "rms", "energy", "level", "loudness", "amplitude", "rms_db", "level_db", "loudness_db"),
            )
        if bass is None:
            # Export rendering often only provides per-input RMS envelopes.
            bass = _extract_scalar(drum_src, ("rms", "energy", "level", "amplitude", "gain"))
        if bass is None:
            bass = float(energy)

        # If the backend provides a beat/onset boolean, prefer it over threshold-based detection.
        onset_flag = False
        for k in ("onset", "beat", "kick_onset", "snare_onset", "transient", "is_beat"):
            v = drum_src.get(k) if isinstance(drum_src, dict) else None
            if isinstance(v, bool) and v:
                onset_flag = True
                break
            if isinstance(v, (int, float)) and float(v) >= 0.5:
                onset_flag = True
                break

        # Mild smoothing, keep responsiveness
        sm = 0.35
        bass_s = (1.0 - sm) * self._prev_bass + sm * float(bass)

        pulse_trigger = False
        cooldown = float(self.config.get("pulse_cooldown_s", 0.12) or 0.12)
        if (t_s - self._last_pulse_t) >= cooldown:
            thr = float(self.config.get("pulse_threshold", 0.01) or 0.01)

            # Prefer explicit beat/onset flags when available.
            if onset_flag:
                pulse_trigger = True
                self._last_pulse_t = t_s
            # Fallback: detect a rising edge on the smoothed curve.
            elif (bass_s - self._prev_bass) >= thr:
                pulse_trigger = True
                self._last_pulse_t = t_s

        self._prev_bass = bass_s
        self._prev_t_s = t_s

        # Optional: cycle to the next preset on pulse triggers
        self._maybe_cycle_on_pulse(t_s=t_s, pulse_trigger=pulse_trigger)

        self._widget.update_audio_features(
            time_ms=time_ms,
            energy=float(energy),
            bass=float(bass_s),
            pulse_trigger=pulse_trigger,
            style_id=float(self._runtime_style_id),
        )
