"""
sdf_void_cathedral_warp.py

Olaf Visualization Plugin (VisPy / Gloo) - Void Cathedral Warp (Balanced) v3.3
-----------------------------------------------------------------------------

v3.3 changes
- Remove "Preview FPS" parameter (fixed internal redraw cadence).
- Camera: remove audio-driven forward acceleration. Add orbital rotation instead (user param).
- Make "Warp (bass)" clearly visible (helical twist + radial breathing).
- Make "Wall thickness" more noticeable (wider range + slightly stronger rib relief).
- Keep VisPy timer safety to prevent "CanvasBackendDesktop has been deleted".

3 inputs (stems)
- input_1: Energy -> global glow + (mild) rotation intensity
- input_2: Drums  -> emissive flicker + bloom boost
- input_3: Bass   -> warp/twist strength (primary)

Notes
- Comments are in English for GitHub readability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

try:
    from olaf_app.visualization_api import BaseVisualization, PluginParameter
except Exception:
    from visualization_api import BaseVisualization, PluginParameter  # type: ignore


# --------
# Optional VisPy dependency
# --------
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


def _extract_rms(d: Dict[str, Any], default: float = 0.0) -> float:
    v = d.get("rms", default)
    if isinstance(v, (list, tuple)) and v:
        v = v[0]
    return float(_safe_float(v, default))


def _preset_colors(name: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    presets = {
        "Neon Cyan / Magenta": ((0.10, 0.90, 1.00), (0.95, 0.20, 1.00)),
        "Solar Gold": ((1.00, 0.80, 0.25), (1.00, 0.35, 0.10)),
        "Void Green": ((0.15, 1.00, 0.55), (0.05, 0.35, 0.20)),
        "Inferno Red": ((1.00, 0.20, 0.15), (1.00, 0.65, 0.15)),
        "Royal Purple": ((0.65, 0.35, 1.00), (0.10, 0.85, 1.00)),
    }
    return presets.get(name, presets["Neon Cyan / Magenta"])


@dataclass
class _AudioState:
    t: float = 0.0
    energy: float = 0.0
    drums: float = 0.0
    bass: float = 0.0


_VERTEX = r"""
attribute vec2 a_position;
varying vec2 v_uv;

void main() {
    v_uv = (a_position + vec2(1.0)) * 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

_FRAGMENT_SCENE = r"""
precision highp float;
varying vec2 v_uv;

uniform vec2  u_resolution;
uniform float u_time;

// 3-stem controls
uniform float u_energy; // input_1
uniform float u_drums;  // input_2
uniform float u_bass;   // input_3

// user params
uniform float u_speed;          // forward speed (constant, not audio-modulated)
uniform float u_rotation_speed; // camera orbital rotation speed
uniform float u_warp;           // overall warp scale (bass makes it explode)
uniform float u_rune_density;

uniform float u_radius;
uniform float u_wall;

uniform vec3  u_colA;
uniform vec3  u_colB;

#define MAX_STEPS 104
#define MAX_DIST  90.0
#define SURF_EPS  0.0016

float hash11(float p) {
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

// Shell around a hollow tunnel: distance to the wall around radius R.
float sdTubeWall(vec3 p, float R, float wall) {
    float r = length(p.xy);
    return abs(r - R) - wall;
}

float runeMask(vec2 q, float seed, float thickness) {
    float h1 = hash11(seed + 1.0);
    float h2 = hash11(seed + 2.0);
    float h3 = hash11(seed + 3.0);

    float d = 1e6;

    // Vertical bar
    if (h1 > 0.15) {
        vec2 a = vec2(0.0, -0.35);
        vec2 b = vec2(0.0,  0.35);
        vec2 pa = q - a, ba = b - a;
        float t = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        d = min(d, length(pa - ba * t));
    }

    // Diagonal bar
    if (h2 > 0.45) {
        float s = sign(h2 - 0.65);
        vec2 a = vec2(-0.30, -0.20 * s);
        vec2 b = vec2( 0.30,  0.20 * s);
        vec2 pa = q - a, ba = b - a;
        float t = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        d = min(d, length(pa - ba * t));
    }

    // Horizontal bar
    if (h3 > 0.60) {
        float y = mix(-0.18, 0.18, h3);
        vec2 a = vec2(-0.28, y);
        vec2 b = vec2( 0.28, y);
        vec2 pa = q - a, ba = b - a;
        float t = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        d = min(d, length(pa - ba * t));
    }

    float ink  = smoothstep(thickness, 0.0, d);
    float halo = smoothstep(thickness * 8.0, thickness, d) * 0.30;
    return clamp(ink + halo, 0.0, 1.0);
}

float mapScene(vec3 p) {
    float e = clamp(u_energy, 0.0, 1.5);
    float b = clamp(u_bass,   0.0, 1.5);

    // Bass-driven warp: make it VERY visible (twist + displacement + breathing).
    float baseWarp = u_warp;
    float warp = baseWarp * (0.20 + 1.35 * b + 0.15 * e); // bass dominates
    float tt = u_time * (0.85 + 0.30 * e);

    // Helical twist (very visible)
    float twist = warp * (0.55 * p.z + 0.90 * tt);
    p.xy = rot(twist) * p.xy;

    // Additional lateral warp
    p.x += 0.18 * warp * sin(1.20 * p.z - 0.70 * tt);
    p.y += 0.18 * warp * cos(1.05 * p.z + 0.80 * tt);

    // Radial breathing of the tunnel radius (also very visible)
    float R = max(0.25, u_radius);
    float breathe = 1.0 + 0.10 * warp * sin(1.35 * p.z + 0.90 * tt);
    R *= breathe;

    float wall = max(0.002, u_wall);
    float dWall = sdTubeWall(p, R, wall);

    // Cathedral ribs: slightly stronger for readability
    float ring = abs(sin(p.z * 2.0)) * 0.055;
    dWall -= ring;

    return dWall;
}

vec3 calcNormal(vec3 p) {
    vec2 e = vec2(0.0017, 0.0);
    float d = mapScene(p);
    vec3 n = d - vec3(
        mapScene(p - e.xyy),
        mapScene(p - e.yxy),
        mapScene(p - e.yyx)
    );
    return normalize(n);
}

vec3 background(vec2 uv) {
    vec2 p = uv * 2.0 - 1.0;
    p.x *= u_resolution.x / max(1.0, u_resolution.y);

    float n = 0.0;
    vec2 q = p * 0.9;
    for (int i = 0; i < 4; i++) {
        n += sin(q.x * 2.2 + float(i) * 1.7 + u_time * 0.06) *
             cos(q.y * 2.0 + float(i) * 1.3 - u_time * 0.05);
        q *= 1.6;
    }
    n = 0.5 + 0.5 * n / 4.0;

    vec3 neb = mix(vec3(0.02, 0.03, 0.07), vec3(0.18, 0.06, 0.24), n);
    neb += 0.02 * vec3(hash11(dot(p, vec2(31.7, 57.3)) + floor(u_time * 0.1)));
    return neb;
}

void main() {
    vec2 uv = v_uv;
    vec2 p = uv * 2.0 - 1.0;
    p.x *= u_resolution.x / max(1.0, u_resolution.y);

    float e = clamp(u_energy, 0.0, 1.5);
    float d = clamp(u_drums,  0.0, 1.5);

    // Constant forward travel (no audio acceleration => no "back and forth" feel)
    float z = u_time * max(0.0, u_speed);

    // Orbital camera rotation around the tunnel axis (audio can intensify it slightly)
    float rotSpeed = u_rotation_speed * (0.75 + 0.35 * e + 0.25 * d);
    float phi = u_time * rotSpeed;

    float orbit = 0.07 + 0.03 * e; // small orbit, energy makes it a bit wider
    vec3 ro = vec3(orbit * cos(phi), orbit * sin(phi), z);

    // Look forward (slightly toward the centerline)
    vec3 target = vec3(0.0, 0.0, z + 1.2);
    vec3 fw = normalize(target - ro);
    vec3 up0 = vec3(0.0, 1.0, 0.0);
    vec3 rt = normalize(cross(up0, fw));
    vec3 up = cross(fw, rt);

    // Ray direction
    vec3 rd = normalize(rt * p.x + up * p.y + fw * 1.15);

    float t = 0.0;
    for (int i = 0; i < MAX_STEPS; i++) {
        vec3 pos = ro + rd * t;
        float dist = mapScene(pos);

        if (t > MAX_DIST) break;

        // Robust march (defensive against negative values)
        if (dist < SURF_EPS) {
            if (dist < 0.0) {
                t += (-dist) + SURF_EPS;
                continue;
            }
            break;
        }

        t += dist;
    }

    if (t > MAX_DIST) {
        gl_FragColor = vec4(background(uv), 1.0);
        return;
    }

    vec3 pos = ro + rd * t;
    vec3 n = calcNormal(pos);

    // Metallic shading
    vec3 L = normalize(vec3(0.2, 0.65, -0.45));
    float ndl = clamp(dot(n, L), 0.0, 1.0);

    // Make geometry a bit easier to read
    vec3 base = mix(vec3(0.010, 0.012, 0.018), vec3(0.08, 0.09, 0.12), ndl);

    // Runes on tunnel wall (angle + z)
    float ang = atan(pos.y, pos.x);
    float a01 = (ang + 3.14159265) / 6.2831853;
    float dens = max(1.0, u_rune_density);

    vec2 cell = vec2(fract(a01 * dens) - 0.5,
                     fract(pos.z * 0.35) - 0.5);

    float seed = floor(a01 * dens) + 97.0 * floor(pos.z * 0.35);
    float rune = runeMask(cell, seed, 0.020);

    // Drums flicker + energy glow
    float flick = 0.85 + 0.35 * sin(u_time * (6.0 + 10.0 * d));
    float glow = (0.30 + 0.95 * e) * mix(1.0, flick, clamp(d, 0.0, 1.0));

    vec3 emissive = mix(u_colA, u_colB, fract(0.06 * pos.z)) * rune * glow;

    float fog = exp(-0.030 * t * t);
    vec3 col = base + emissive;
    col = mix(background(uv), col, fog);

    gl_FragColor = vec4(max(col, 0.0), 1.0);
}
"""

_FRAGMENT_PREFILTER = r"""
precision highp float;
varying vec2 v_uv;
uniform sampler2D u_scene;

vec3 prefilter(vec3 c) {
    float threshold = 0.85;
    float soft_knee = 0.50;

    float br = max(max(c.r, c.g), c.b);
    float knee = threshold * soft_knee + 1e-5;
    float soft = clamp((br - threshold + knee) / (2.0 * knee), 0.0, 1.0);
    float contrib = max(br - threshold, 0.0) + soft * soft * knee;
    return c * (contrib / max(br, 1e-5));
}

void main() {
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

void main() {
    float radius = 1.25;
    vec2 d = u_dir * u_texel * radius;

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
uniform float u_drums; // slight bloom boost on drums

vec3 tonemap_aces(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec2 uv = v_uv;

    vec3 scene = texture2D(u_scene, uv).rgb;
    vec3 bloom = texture2D(u_bloom, uv).rgb;

    float d = clamp(u_drums, 0.0, 1.5);
    float bloomBoost = 1.0 + 0.40 * d;

    vec3 col = scene + (u_bloom_intensity * bloomBoost) * bloom;

    col *= max(0.05, u_exposure);
    col = tonemap_aces(col);

    float cont = max(0.65, u_contrast);
    col = pow(col, vec3(0.90 / cont));

    col = pow(max(col, 0.0), vec3(0.4545));
    gl_FragColor = vec4(col, 1.0);
}
"""


if HAVE_VISPY:

    class _VoidCathedralCanvas(app.Canvas):
        def __init__(self, config: Dict[str, Any]) -> None:
            self._config = config
            self._audio = _AudioState()

            super().__init__(keys=None, size=(640, 360), show=False)

            self._fbo_failed = False
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

            # Timer safety (fixed 30 FPS internal cadence)
            self._timer: Optional[app.Timer] = None
            self._timer_running = False
            self._start_redraw_timer()

            try:
                self.events.close.connect(self._on_canvas_close)  # type: ignore[attr-defined]
            except Exception:
                pass

        def _start_redraw_timer(self) -> None:
            # Fixed internal cadence; no UI parameter
            self._timer = app.Timer(interval=1.0 / 30.0, connect=self._tick, start=True)
            self._timer_running = True

        def _stop_redraw_timer(self) -> None:
            if self._timer is None:
                return
            try:
                self._timer.stop()
            except Exception:
                pass
            self._timer_running = False
            self._timer = None

        def _tick(self, event=None) -> None:
            if not self._timer_running:
                return
            try:
                self.update()
            except RuntimeError:
                self._stop_redraw_timer()
            except Exception:
                self._stop_redraw_timer()

        def _on_canvas_close(self, event=None) -> None:
            self._stop_redraw_timer()

        def reset_audio_state(self) -> None:
            self._audio = _AudioState()
            try:
                self.update()
            except Exception:
                pass

        def _get_gl_limits(self) -> Dict[str, Any]:
            if self._gl_limits is not None:
                return self._gl_limits
            limits: Dict[str, Any] = {"max_tex": 4096, "max_vp": (4096, 4096)}
            try:
                limits["max_tex"] = int(gl.glGetIntegerv(gl.GL_MAX_TEXTURE_SIZE))
            except Exception:
                pass
            try:
                mv = gl.glGetIntegerv(gl.GL_MAX_VIEWPORT_DIMS)
                if hasattr(mv, "__len__"):
                    mv = list(mv)
                    if len(mv) >= 2:
                        limits["max_vp"] = (int(mv[0]), int(mv[1]))
            except Exception:
                pass
            limits["max_tex"] = max(64, int(limits.get("max_tex", 4096)))
            w, h = limits.get("max_vp", (4096, 4096))
            limits["max_vp"] = (max(64, int(w)), max(64, int(h)))
            self._gl_limits = limits
            return limits

        @staticmethod
        def _make_texture(w: int, h: int):
            w = max(1, int(w))
            h = max(1, int(h))
            if HAVE_NUMPY:
                data = np.zeros((h, w, 4), dtype=np.uint8)
                return gloo.Texture2D(data, interpolation="linear")
            return gloo.Texture2D((h, w, 4), interpolation="linear")

        def _ensure_fbos(self) -> None:
            if self._fbo_failed:
                return
            limits = self._get_gl_limits()
            max_tex = int(limits.get("max_tex", 4096))

            w, h = self.physical_size
            w = min(max(1, int(w)), max_tex)
            h = min(max(1, int(h)), max_tex)
            down = 2

            if (w, h, down) == self._last_sizes and self._scene_tex is not None:
                return
            self._last_sizes = (w, h, down)

            self._scene_tex = self._make_texture(w, h)
            self._scene_fbo = gloo.FrameBuffer(color=self._scene_tex)

            bw = max(16, min(max_tex, w // down))
            bh = max(16, min(max_tex, h // down))
            self._bloom_tex0 = self._make_texture(bw, bh)
            self._bloom_tex1 = self._make_texture(bw, bh)
            self._bloom_fbo0 = gloo.FrameBuffer(color=self._bloom_tex0)
            self._bloom_fbo1 = gloo.FrameBuffer(color=self._bloom_tex1)

        def _update_uniforms(self, scene_w: int, scene_h: int, screen_w: int, screen_h: int) -> None:
            scene_w = max(1, int(scene_w))
            scene_h = max(1, int(scene_h))
            screen_w = max(1, int(screen_w))
            screen_h = max(1, int(screen_h))

            preset = str(self._config.get("color_preset", "Neon Cyan / Magenta"))
            colA, colB = _preset_colors(preset)

            self._prog_scene["u_resolution"] = (float(scene_w), float(scene_h))
            self._prog_scene["u_time"] = float(self._audio.t)

            self._prog_scene["u_energy"] = float(self._audio.energy)
            self._prog_scene["u_drums"] = float(self._audio.drums)
            self._prog_scene["u_bass"] = float(self._audio.bass)

            self._prog_scene["u_speed"] = float(_clamp(_safe_float(self._config.get("speed", 1.20), 1.20), 0.0, 6.0))
            self._prog_scene["u_rotation_speed"] = float(_clamp(_safe_float(self._config.get("rotation_speed", 0.75), 0.75), 0.0, 6.0))
            self._prog_scene["u_warp"] = float(_clamp(_safe_float(self._config.get("warp", 0.90), 0.90), 0.0, 3.0))
            self._prog_scene["u_rune_density"] = float(_clamp(_safe_float(self._config.get("rune_density", 18.0), 18.0), 4.0, 64.0))

            self._prog_scene["u_radius"] = float(_clamp(_safe_float(self._config.get("tunnel_radius", 0.75), 0.75), 0.25, 2.5))
            self._prog_scene["u_wall"] = float(_clamp(_safe_float(self._config.get("wall_thickness", 0.07), 0.07), 0.01, 0.60))

            self._prog_scene["u_colA"] = colA
            self._prog_scene["u_colB"] = colB

            self._prog_combine["u_resolution"] = (float(screen_w), float(screen_h))
            self._prog_combine["u_exposure"] = float(_clamp(_safe_float(self._config.get("exposure", 1.10), 1.10), 0.1, 3.0))
            self._prog_combine["u_contrast"] = float(_clamp(_safe_float(self._config.get("contrast", 1.05), 1.05), 0.6, 2.0))
            self._prog_combine["u_bloom_intensity"] = float(_clamp(_safe_float(self._config.get("bloom", 1.10), 1.10), 0.0, 3.0))
            self._prog_combine["u_drums"] = float(self._audio.drums)

        def set_audio_features(self, *, time_s: float, energy: float, drums: float, bass: float) -> None:
            self._audio.t = float(time_s)

            # Soft smoothing for stable motion
            a = 0.20
            self._audio.energy = (1.0 - a) * self._audio.energy + a * float(_clamp(energy, 0.0, 2.0))
            self._audio.drums = (1.0 - a) * self._audio.drums + a * float(_clamp(drums, 0.0, 2.0))
            self._audio.bass = (1.0 - a) * self._audio.bass + a * float(_clamp(bass, 0.0, 2.0))

            try:
                self.update()
            except Exception:
                pass

        def on_resize(self, event) -> None:  # type: ignore[override]
            try:
                gloo.set_viewport(0, 0, *event.physical_size)
            except Exception:
                pass

        def on_draw(self, event) -> None:  # type: ignore[override]
            limits = self._get_gl_limits()
            max_vp_w, max_vp_h = limits.get("max_vp", (4096, 4096))

            w, h = self.physical_size
            screen_w = max(1, min(int(w), int(max_vp_w)))
            screen_h = max(1, min(int(h), int(max_vp_h)))

            if self._fbo_failed:
                try:
                    self._update_uniforms(screen_w, screen_h, screen_w, screen_h)
                    gloo.set_viewport(0, 0, screen_w, screen_h)
                    gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                    self._prog_scene.draw("triangle_strip")
                except Exception:
                    return
                return

            self._ensure_fbos()
            if self._scene_tex is None or self._scene_fbo is None:
                return
            if self._bloom_tex0 is None or self._bloom_tex1 is None:
                return
            if self._bloom_fbo0 is None or self._bloom_fbo1 is None:
                return

            scene_h, scene_w = self._scene_tex.shape[0], self._scene_tex.shape[1]
            bloom_h, bloom_w = self._bloom_tex0.shape[0], self._bloom_tex0.shape[1]
            self._update_uniforms(scene_w, scene_h, screen_w, screen_h)

            try:
                with self._scene_fbo:
                    gloo.set_viewport(0, 0, scene_w, scene_h)
                    gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                    self._prog_scene.draw("triangle_strip")

                self._prog_prefilter["u_scene"] = self._scene_tex
                with self._bloom_fbo0:
                    gloo.set_viewport(0, 0, bloom_w, bloom_h)
                    gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                    self._prog_prefilter.draw("triangle_strip")

                self._prog_blur["u_tex"] = self._bloom_tex0
                self._prog_blur["u_texel"] = (1.0 / float(bloom_w), 1.0 / float(bloom_h))
                self._prog_blur["u_dir"] = (1.0, 0.0)
                with self._bloom_fbo1:
                    gloo.set_viewport(0, 0, bloom_w, bloom_h)
                    gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                    self._prog_blur.draw("triangle_strip")

                self._prog_blur["u_tex"] = self._bloom_tex1
                self._prog_blur["u_dir"] = (0.0, 1.0)
                with self._bloom_fbo0:
                    gloo.set_viewport(0, 0, bloom_w, bloom_h)
                    gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                    self._prog_blur.draw("triangle_strip")

                self._prog_combine["u_scene"] = self._scene_tex
                self._prog_combine["u_bloom"] = self._bloom_tex0
                gloo.set_viewport(0, 0, screen_w, screen_h)
                gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                self._prog_combine.draw("triangle_strip")

            except Exception:
                self._fbo_failed = True
                try:
                    self._update_uniforms(screen_w, screen_h, screen_w, screen_h)
                    gloo.set_viewport(0, 0, screen_w, screen_h)
                    gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                    self._prog_scene.draw("triangle_strip")
                except Exception:
                    return


class _VoidCathedralWidget(QWidget):
    def __init__(self, config: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._canvas = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not HAVE_VISPY:
            label = QLabel(
                "VisPy is not available.\n"
                "Void Cathedral Warp is disabled.\n"
                "Please install 'vispy' to enable this visualization.",
                self,
            )
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            return

        app.use_app("pyqt6")  # type: ignore[arg-type]
        self._canvas = _VoidCathedralCanvas(config=self._config)  # type: ignore[name-defined]
        self._canvas.native.setParent(self)
        layout.addWidget(self._canvas.native)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(640, 360)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._canvas is not None:
            try:
                self._canvas.close()
            except Exception:
                pass
            self._canvas = None
        super().closeEvent(event)

    def update_audio_features(self, *, time_ms: int, energy: float, drums: float, bass: float) -> None:
        if self._canvas is None:
            return
        self._canvas.set_audio_features(
            time_s=float(time_ms) / 1000.0,
            energy=energy,
            drums=drums,
            bass=bass,
        )

    def reset_audio_state(self) -> None:
        if self._canvas is None:
            return
        try:
            self._canvas.reset_audio_state()
        except Exception:
            pass


class SDFVoidCathedralWarpVisualization(BaseVisualization):
    plugin_id = "sdf_void_cathedral_warp"
    plugin_name = "SDF Void Cathedral Warp (Balanced) v3.3"
    plugin_description = (
        "Orbital tunnel flight with emissive runes and bloom. "
        "3 inputs: Energy (1), Drums (2), Bass (3)."
    )
    plugin_author = "DrDLP"
    plugin_version = "0.3.3"
    plugin_max_inputs = 3

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._widget: Optional[_VoidCathedralWidget] = None

        # UI-friendly defaults
        self.config.setdefault("color_preset", "Neon Cyan / Magenta")

        self.config.setdefault("speed", 1.25)
        self.config.setdefault("rotation_speed", 0.75)  # new
        self.config.setdefault("warp", 0.95)
        self.config.setdefault("rune_density", 18.0)

        self.config.setdefault("tunnel_radius", 0.75)
        self.config.setdefault("wall_thickness", 0.07)  # wider range => more visible

        # Gains & floors
        self.config.setdefault("energy_gain", 1.00)
        self.config.setdefault("drums_gain", 1.00)
        self.config.setdefault("bass_gain", 1.00)
        self.config.setdefault("energy_floor", 0.10)

        self.config.setdefault("bloom", 1.10)
        self.config.setdefault("exposure", 1.10)
        self.config.setdefault("contrast", 1.05)

        self._prev_t_s: float = -1.0

    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        color_choices: List[str] = [
            "Neon Cyan / Magenta",
            "Solar Gold",
            "Void Green",
            "Inferno Red",
            "Royal Purple",
        ]
        return {
            "color_preset": PluginParameter(
                name="color_preset",
                label="Color preset",
                type="enum",
                default="Neon Cyan / Magenta",
                choices=color_choices,
            ),

            "speed": PluginParameter(name="speed", label="Forward speed", type="float", default=1.25, minimum=0.0, maximum=6.0, step=0.01),
            "rotation_speed": PluginParameter(name="rotation_speed", label="Rotation speed", type="float", default=0.75, minimum=0.0, maximum=6.0, step=0.01),
            "warp": PluginParameter(name="warp", label="Warp (Bass)", type="float", default=0.95, minimum=0.0, maximum=3.0, step=0.01),
            "rune_density": PluginParameter(name="rune_density", label="Rune density", type="float", default=18.0, minimum=4.0, maximum=64.0, step=1.0),

            "tunnel_radius": PluginParameter(name="tunnel_radius", label="Tunnel radius", type="float", default=0.75, minimum=0.25, maximum=2.5, step=0.01),
            "wall_thickness": PluginParameter(name="wall_thickness", label="Wall thickness", type="float", default=0.07, minimum=0.01, maximum=0.60, step=0.005),

            "energy_gain": PluginParameter(name="energy_gain", label="Gain (Energy / input_1)", type="float", default=1.00, minimum=0.0, maximum=3.0, step=0.01),
            "drums_gain": PluginParameter(name="drums_gain", label="Gain (Drums / input_2)", type="float", default=1.00, minimum=0.0, maximum=3.0, step=0.01),
            "bass_gain": PluginParameter(name="bass_gain", label="Gain (Bass / input_3)", type="float", default=1.00, minimum=0.0, maximum=3.0, step=0.01),
            "energy_floor": PluginParameter(name="energy_floor", label="Energy floor", type="float", default=0.10, minimum=0.0, maximum=1.0, step=0.01),

            "bloom": PluginParameter(name="bloom", label="Bloom", type="float", default=1.10, minimum=0.0, maximum=3.0, step=0.01),
            "exposure": PluginParameter(name="exposure", label="Exposure", type="float", default=1.10, minimum=0.1, maximum=3.0, step=0.01),
            "contrast": PluginParameter(name="contrast", label="Contrast", type="float", default=1.05, minimum=0.6, maximum=2.0, step=0.01),
        }

    def create_widget(self, parent: Optional[QWidget] = None) -> QWidget:
        self._widget = _VoidCathedralWidget(config=self.config, parent=parent)
        return self._widget

    def on_audio_features(self, features: Dict[str, Any]) -> None:
        if self._widget is None:
            return

        raw_ms = features.get("time_ms", 0)
        try:
            time_ms = int(raw_ms)
        except Exception:
            time_ms = 0
        t_s = float(time_ms) / 1000.0

        # Seek/restart detection
        if self._prev_t_s >= 0.0 and (t_s + 0.25) < self._prev_t_s:
            try:
                self._widget.reset_audio_state()
            except Exception:
                pass

        inputs = features.get("inputs", {}) if isinstance(features.get("inputs", {}), dict) else {}

        i1 = inputs.get("input_1", {}) if isinstance(inputs.get("input_1", {}), dict) else {}
        i2 = inputs.get("input_2", {}) if isinstance(inputs.get("input_2", {}), dict) else {}
        i3 = inputs.get("input_3", {}) if isinstance(inputs.get("input_3", {}), dict) else {}

        rms1 = _extract_rms(i1, 0.0)
        rms2 = _extract_rms(i2, 0.0)
        rms3 = _extract_rms(i3, 0.0)

        g1 = float(_clamp(_safe_float(self.config.get("energy_gain", 1.0), 1.0), 0.0, 3.0))
        g2 = float(_clamp(_safe_float(self.config.get("drums_gain", 1.0), 1.0), 0.0, 3.0))
        g3 = float(_clamp(_safe_float(self.config.get("bass_gain", 1.0), 1.0), 0.0, 3.0))

        floor = float(_clamp(_safe_float(self.config.get("energy_floor", 0.10), 0.10), 0.0, 1.0))

        energy = max(floor, float(_clamp(rms1 * g1, 0.0, 2.0)))
        drums = float(_clamp(rms2 * g2, 0.0, 2.0))
        bass = float(_clamp(rms3 * g3, 0.0, 2.0))

        self._prev_t_s = t_s

        self._widget.update_audio_features(
            time_ms=time_ms,
            energy=energy,
            drums=drums,
            bass=bass,
        )
