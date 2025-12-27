"""
mecha_choir_constellations.py

Olaf Visualization Plugin (VisPy / Gloo)
- Mecha-Choir Constellations (Cyber ritual orbits) v1.9

Visual pitch
- Central stage (disc + ring)
- Multiple orbital constellations (segmented rings)
- Guitars create jitter + "arc lightning" along segments
- Vocals drive the central stage (disc size) + subtle glyph traces
- Drums drive global orbit speed + kick strobe + phase resync (cuts)
- Bass drives camera zoom (depth)

Inputs (stems)
- input_1: Full mix (energy)
- input_2: Bass
- input_3: Drums
- input_4: Vocals

Parameters
- orbit_count, orbit_radius, orbit_speed
- segment_count, segment_thickness
- lightning_intensity, camera_distance, camera_motion
- rhythm_reactivity, lightning_reactivity, disc_reactivity
- color_preset (extra, but very practical)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
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
        "Cyan / Magenta": ((0.10, 0.90, 1.00), (0.95, 0.20, 1.00)),
        "Gold / Ember": ((1.00, 0.82, 0.25), (1.00, 0.28, 0.10)),
        "Acid Green": ((0.20, 1.00, 0.55), (0.05, 0.25, 0.18)),
        "Inferno": ((1.00, 0.20, 0.15), (1.00, 0.65, 0.15)),
        "Royal": ((0.65, 0.35, 1.00), (0.10, 0.85, 1.00)),
        "Mono Ice": ((0.65, 0.90, 1.00), (0.25, 0.55, 0.90)),
    }
    return presets.get(name, presets["Cyan / Magenta"])


@dataclass
class _AudioState:
    # timeline (seconds)
    t: float = 0.0

    # Smoothed stem envelopes (0..~2)
    drums: float = 0.0
    bass: float = 0.0
    guitars: float = 0.0  # derived from full mix energy (used for lightning/jitter)
    vocals: float = 0.0

    # kick detection / cut sync (drums)
    prev_drums: float = 0.0
    kick: float = 0.0
    cut_time: float = 0.0

    # Camera motion driven by drums (smooth "orbiting" camera)
    cam_phase: float = 0.0
    cam_rot: float = 0.0
    cam_pan_x: float = 0.0
    cam_pan_y: float = 0.0


_VERTEX = r"""
attribute vec2 a_position;
varying vec2 v_uv;

void main() {
    v_uv = (a_position + vec2(1.0)) * 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

# -----------------------------
# Scene shader (2.5D ritual orbits)
# -----------------------------
_FRAGMENT_SCENE = r"""
precision highp float;
varying vec2 v_uv;

uniform vec2  u_resolution;
uniform float u_time;

// stems (smoothed RMS envelopes)
uniform float u_drums;
uniform float u_bass;
uniform float u_guitars;
uniform float u_vocals;

// kick/cut sync
uniform float u_kick;
uniform float u_cut_time;

// camera motion (drums-driven)
uniform vec2  u_cam_pan;
uniform float u_cam_rot;

// params
uniform float u_orbit_count;
uniform float u_orbit_radius;
uniform float u_orbit_speed;

uniform float u_segment_count;
uniform float u_segment_thickness;

uniform float u_lightning_intensity;
uniform float u_camera_distance;

// reactivity (per-system)
uniform float u_rhythm_react;
uniform float u_lightning_react;
uniform float u_disc_react;

// background motif (4 corner clones)
uniform float u_bg_scale;
uniform float u_bg_intensity;

// colors
uniform vec3 u_colA;
uniform vec3 u_colB;

#define MAX_ORBITS 8

float hash11(float p) {
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

mat2 rot(float a){
    float c = cos(a), s = sin(a);
    return mat2(c,-s,s,c);
}

float sdCircle(vec2 p, float r){
    return length(p) - r;
}

float sdRing(vec2 p, float r, float th){
    return abs(length(p) - r) - th;
}

vec3 background(vec2 uv){
    // Strict black background (requested).
    // Keep it truly black to maximize contrast for the ritual rings.
    return vec3(0.0);
}

// segmented ring mask (SDF-ish)
float segmentedRing(vec2 p, float r, float th, float segCount, float phase, float guitars){
    // radius band
    float dRing = sdRing(p, r, th);

    // angle segmentation
    float a = atan(p.y, p.x);               // -pi..pi
    float a01 = (a + 3.14159265) / 6.2831853; // 0..1
    float gJ = guitars * 0.018 * u_lightning_react;
    a01 += gJ * sin(20.0 * a + 3.0 * phase);

    float u = fract((a01 + phase) * segCount);
    float onRatio = 0.68;

    // soften segment edges
    float edge = 0.10 + 0.20 * guitars * u_lightning_react;
    float segMask = smoothstep(0.0, edge, u) * (1.0 - smoothstep(onRatio - edge, onRatio, u));
    segMask = clamp(segMask, 0.0, 1.0);

    // convert SDF to alpha-like mask
    float ringMask = smoothstep(0.012, 0.0, dRing);
    return ringMask * segMask;
}

float lightning(vec2 p, float r, float segCount, float phase, float guitars, float intensity){
    float a = atan(p.y, p.x);
    float a01 = (a + 3.14159265) / 6.2831853;

    float seg = floor((a01 + phase) * segCount);
    float rnd = hash11(seg + 13.7 * r);

    // randomize local lightning frequency per segment
    float f = mix(18.0, 42.0, rnd);
    float w = mix(10.0, 22.0, rnd);

    // arc spikes
    float spike = sin((a01 + phase) * f * 6.2831853 + u_time * w);
    spike = pow(max(spike, 0.0), 10.0);

    // radial fade close to ring
    float dr = abs(length(p) - r);
    float band = smoothstep(0.06, 0.0, dr);

    float g01 = clamp(guitars / 2.0, 0.0, 1.0);
    float gShaped = pow(g01, 1.25);
    return (intensity * u_lightning_react) * (0.15 + 1.35 * gShaped) * band * spike;
}


// Render a subtle "mini-scene" (stage + a few segmented rings) used as a background motif.
// This is intentionally screen-space (it uses p0, before camera transforms) and low intensity.
vec3 bgMotif(vec2 q, float scale, float segCount, vec3 colA, vec3 colB) {
    if (scale <= 0.0001) return vec3(0.0);

    // Scale up/down the mini motif (bigger scale => bigger motif).
    q /= max(0.10, scale);

    // Soft fade so it doesn't look like a sticker.
    float fade = smoothstep(1.10, 0.40, length(q));

    // Mini stage
    float discR = 0.16;
    float disc = smoothstep(0.02, 0.0, sdCircle(q, discR));
    float ring = smoothstep(0.018, 0.0, sdRing(q, discR * 1.22, 0.010));

    vec3 stage = mix(colA, colB, 0.5) * (0.14 * disc + 0.22 * ring);

    // A few segmented rings (no lightning, no jitter => calm background)
    vec3 rings = vec3(0.0);
    for (int i = 0; i < 3; i++) {
        float k = float(i) / 2.0;
        float r = 0.34 + 0.18 * k;
        float ph = 0.10 * u_time + 0.07 * float(i);
        float m = segmentedRing(q, r, 0.014, segCount * 0.70, ph, 0.0);
        vec3 rc = mix(colA, colB, k);
        rings += rc * (0.10 + 0.35 * m);
    }

    // Very low intensity overall: keep it "present" but not distracting.
    return (stage + rings) * fade * (0.30 * u_bg_intensity);
}

void main() {
    vec2 uv = v_uv;
    vec2 p = uv * 2.0 - 1.0;
    float aspect = u_resolution.x / max(1.0, u_resolution.y);
    p.x *= aspect;

    // p0 is screen-space coordinates (pre-camera). Used for background motifs.
    vec2 p0 = p;

    float drums = clamp(u_drums, 0.0, 3.0);
    float bass  = clamp(u_bass,  0.0, 3.0);
    float guitars = clamp(u_guitars, 0.0, 3.0);
    float vocals  = clamp(u_vocals,  0.0, 3.0);

    // camera zoom (bass): larger => zoom out
    float cam = max(0.35, u_camera_distance * (1.0 - 0.26 * bass));
    p *= cam;

    // camera motion (drums): smooth pan + rotation (replaces halos)
    p = rot(u_cam_rot) * p + u_cam_pan;

    vec3 col = background(uv);

    // 4 background motif clones (one per corner), size controlled by u_bg_scale.
    float segCountBG = clamp(u_segment_count, 6.0, 96.0);
    vec2 c1 = vec2(-0.78 * aspect, -0.78);
    vec2 c2 = vec2( 0.78 * aspect, -0.78);
    vec2 c3 = vec2(-0.78 * aspect,  0.78);
    vec2 c4 = vec2( 0.78 * aspect,  0.78);

    col += bgMotif(p0 - c1, u_bg_scale, segCountBG, u_colA, u_colB);
    col += bgMotif(p0 - c2, u_bg_scale, segCountBG, u_colA, u_colB);
    col += bgMotif(p0 - c3, u_bg_scale, segCountBG, u_colA, u_colB);
    col += bgMotif(p0 - c4, u_bg_scale, segCountBG, u_colA, u_colB);

    // kick strobe
    float strobe = 1.0 + (0.85 + 0.25 * drums * u_rhythm_react) * clamp(u_kick, 0.0, 1.0);

    // central stage: disc + ring "ritual platform"
    float discBase = 0.18;
    // Disc size follows vocals envelope (more reactive, shaped for punch).
    float v01 = clamp(vocals / 2.0, 0.0, 1.0);
    float vShaped = pow(v01, 1.35);
    float discR = discBase * (1.0 + (2.20 * u_disc_react) * vShaped);
    // Rhythmic wobble so sustained vocals still feel alive (drums modulate the wobble speed).
    discR *= 1.0 + (0.26 * u_disc_react) * vShaped * sin(u_time * (2.4 + 3.2 * drums * u_rhythm_react));
    // Kick punch (keeps the stage feeling "locked" to the beat).
    discR *= 1.0 + 0.16 * clamp(u_kick, 0.0, 1.0);

    float dDisc = sdCircle(p, discR);
    float discMask = smoothstep(0.02, 0.0, dDisc);

    float dStageRing = sdRing(p, discR * 1.22, 0.010 + 0.010 * (0.35 * bass + 0.65 * vocals));
    float ringMask = smoothstep(0.018, 0.0, dStageRing);

    float stageBoost = 1.10 + 0.40 * clamp((vocals * u_disc_react) / 2.0, 0.0, 1.0);
    vec3 stageCol = mix(u_colA, u_colB, 0.5) * stageBoost * (0.24 * discMask + 0.70 * ringMask);

    // orbits
    float oc = clamp(u_orbit_count, 1.0, float(MAX_ORBITS));
    // Keep overall brightness stable when orbit_count increases.
    // Reference is tuned for the default orbit_count=5.
    float orbitNorm = clamp(sqrt(5.0 / oc), 0.55, 1.60);
    float segCount = clamp(u_segment_count, 6.0, 96.0);
    float th = clamp(u_segment_thickness, 0.003, 0.08);

    // drums speed + cut resync: use local time from last kick cut
    float tLocal = max(0.0, u_time - u_cut_time);
    float baseSpeed = u_orbit_speed * (0.45 + 1.20 * drums * u_rhythm_react);

    // draw far to near (smaller to bigger), so big rings look "closer"
    for (int i = 0; i < MAX_ORBITS; i++) {
        float fi = float(i);
        if (fi >= oc) break;

        float k = (oc <= 1.0) ? 0.0 : (fi / (oc - 1.0));
        float r = u_orbit_radius * (0.55 + 0.75 * k);

        // per-orbit phase and slight speed variation
        float ph = tLocal * baseSpeed * (0.70 + 0.55 * k);
        ph += 0.15 * fi;

        // ring segmentation
        float m = segmentedRing(p, r, th, segCount, ph, guitars);

        // base ring color gradient along radius
        vec3 rc = mix(u_colA, u_colB, k);

        // guitar lightning (derived from full mix energy)
        float li = lightning(p, r, segCount, ph, guitars, u_lightning_intensity);

        // ring glow / emissive (drums strobe)
        // IMPORTANT: avoid a constant baseline glow when m==0, otherwise the whole screen gets tinted,
        // and the effect gets worse with higher orbit_count.
        float drumLift = 0.85 + 0.65 * clamp(drums / 2.0, 0.0, 1.0);
        float mm = clamp(m, 0.0, 1.0);
        float glow = (0.20 + 1.65 * mm) * mm * strobe * drumLift;
        vec3 emissive = rc * glow + rc * li * (2.0 + 1.2 * mm);
        emissive *= orbitNorm;

        // subtle depth: far rings dimmer
        float depth = mix(0.60, 1.0, k);
        col += emissive * depth;
    }

    // stage over everything
    col += stageCol * strobe;

    // Vocals "trace" (localized): avoid tinting the entire background.
    float rTrace = u_orbit_radius * 0.95;
    float traceBand = smoothstep(0.40, 0.0, abs(length(p) - rTrace));
    col += (0.14 * vocals) * traceBand * mix(u_colB, vec3(1.0), 0.30);

    // mild vignette
    float v = smoothstep(1.45, 0.2, length(p));
    col *= (0.90 + 0.10 * v);

    gl_FragColor = vec4(max(col, 0.0), 1.0);
}
"""

_FRAGMENT_PREFILTER = r"""
precision highp float;
varying vec2 v_uv;
uniform sampler2D u_scene;

vec3 prefilter(vec3 c) {
    // Lower threshold => more pixels contribute to bloom (more visible bloom).
    float threshold = 0.65;
    float soft_knee = 0.75;

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
    float radius = 1.55;
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

uniform float u_exposure;
uniform float u_contrast;
uniform float u_bloom_intensity;

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

    float lum = dot(scene, vec3(0.2126, 0.7152, 0.0722));
    // Mask bloom so it doesn't tint pure black areas.
    float bloomMask = smoothstep(0.05, 0.22, lum);
    vec3 col = scene + (u_bloom_intensity * bloomMask) * bloom;

    col *= max(0.05, u_exposure);
    col = tonemap_aces(col);

    float cont = max(0.65, u_contrast);
    col = pow(col, vec3(0.90 / cont));

    // gamma-ish
    col = pow(max(col, 0.0), vec3(0.4545));
    gl_FragColor = vec4(col, 1.0);
}
"""


if HAVE_VISPY:

    class _MechaChoirCanvas(app.Canvas):
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

            # Timer safety: fixed cadence (no UI param)
            self._timer: Optional[app.Timer] = None
            self._timer_running = False
            self._start_redraw_timer()

            try:
                self.events.close.connect(self._on_canvas_close)  # type: ignore[attr-defined]
            except Exception:
                pass

        def _start_redraw_timer(self) -> None:
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

            preset = str(self._config.get("color_preset", "Cyan / Magenta"))
            colA, colB = _preset_colors(preset)

            # Scene
            self._prog_scene["u_resolution"] = (float(scene_w), float(scene_h))
            self._prog_scene["u_time"] = float(self._audio.t)

            self._prog_scene["u_drums"] = float(self._audio.drums)
            self._prog_scene["u_bass"] = float(self._audio.bass)
            self._prog_scene["u_guitars"] = float(self._audio.guitars)
            self._prog_scene["u_vocals"] = float(self._audio.vocals)

            self._prog_scene["u_kick"] = float(self._audio.kick)
            self._prog_scene["u_cut_time"] = float(self._audio.cut_time)

            # Drums-driven camera motion (smooth orbiting pan + gentle roll)
            self._prog_scene["u_cam_pan"] = (float(self._audio.cam_pan_x), float(self._audio.cam_pan_y))
            self._prog_scene["u_cam_rot"] = float(self._audio.cam_rot)

            self._prog_scene["u_orbit_count"] = float(_clamp(_safe_float(self._config.get("orbit_count", 5.0), 5.0), 1.0, 8.0))
            self._prog_scene["u_orbit_radius"] = float(_clamp(_safe_float(self._config.get("orbit_radius", 0.58), 0.58), 0.20, 1.20))
            self._prog_scene["u_orbit_speed"] = float(_clamp(_safe_float(self._config.get("orbit_speed", 1.10), 1.10), 0.0, 6.0))

            self._prog_scene["u_segment_count"] = float(_clamp(_safe_float(self._config.get("segment_count", 28.0), 28.0), 6.0, 96.0))
            self._prog_scene["u_segment_thickness"] = float(_clamp(_safe_float(self._config.get("segment_thickness", 0.020), 0.020), 0.003, 0.08))

            self._prog_scene["u_lightning_intensity"] = float(_clamp(_safe_float(self._config.get("lightning_intensity", 1.10), 1.10), 0.0, 3.0))
            self._prog_scene["u_camera_distance"] = float(_clamp(_safe_float(self._config.get("camera_distance", 0.95), 0.95), 0.35, 2.20))
            self._prog_scene["u_rhythm_react"] = float(_clamp(_safe_float(self._config.get("rhythm_reactivity", 1.35), 1.35), 0.25, 5.0))
            self._prog_scene["u_lightning_react"] = float(_clamp(_safe_float(self._config.get("lightning_reactivity", 1.35), 1.35), 0.25, 5.0))
            self._prog_scene["u_disc_react"] = float(_clamp(_safe_float(self._config.get("disc_reactivity", 1.60), 1.60), 0.25, 5.0))
            self._prog_scene["u_bg_scale"] = float(_clamp(_safe_float(self._config.get("bg_motif_scale", 0.55), 0.55), 0.0, 1.50))
            self._prog_scene["u_bg_intensity"] = float(_clamp(_safe_float(self._config.get("bg_motif_intensity", 0.18), 0.18), 0.0, 1.0))

            self._prog_scene["u_colA"] = colA
            self._prog_scene["u_colB"] = colB

            # Combine
            self._prog_combine["u_exposure"] = float(_clamp(_safe_float(self._config.get("exposure", 1.10), 1.10), 0.1, 3.0))
            self._prog_combine["u_contrast"] = float(_clamp(_safe_float(self._config.get("contrast", 1.05), 1.05), 0.6, 2.0))
            self._prog_combine["u_bloom_intensity"] = float(_clamp(_safe_float(self._config.get("bloom", 1.15), 1.15), 0.0, 3.0))

        
        def set_audio(self, *, time_s: float, drums: float, bass: float, guitars: float, vocals: float) -> None:
            # Compute a stable dt (preview restarts / export grabs can produce non-monotonic clocks).
            prev_t = float(self._audio.t)
            t = float(time_s)
            dt = t - prev_t
            if dt <= 0.0 or dt > 1.0:
                dt = 1.0 / 30.0

            self._audio.t = t

            # Mild smoothing for stem envelopes
            a = 0.18
            self._audio.drums = (1.0 - a) * self._audio.drums + a * float(_clamp(drums, 0.0, 3.0))
            self._audio.bass = (1.0 - a) * self._audio.bass + a * float(_clamp(bass, 0.0, 3.0))
            self._audio.guitars = (1.0 - a) * self._audio.guitars + a * float(_clamp(guitars, 0.0, 3.0))
            self._audio.vocals = (1.0 - a) * self._audio.vocals + a * float(_clamp(vocals, 0.0, 3.0))

            # Kick detection: transient on the (smoothed) drums envelope.
            d_now = self._audio.drums
            d_prev = self._audio.prev_drums
            self._audio.prev_drums = d_now

            delta = max(0.0, d_now - d_prev)
            threshold = 0.10 + 0.30 * d_now
            triggered = delta > threshold

            # Kick envelope (decay)
            self._audio.kick *= 0.86
            if triggered:
                self._audio.kick = 1.0
                # "Cut" sync: reset orbit phases at kick time.
                self._audio.cut_time = t

            # -------------------------------------------------
            # Drums-driven camera motion (smooth, rhythmic)
            # -------------------------------------------------
            try:
                motion = float(self._config.get("camera_motion", 1.0) or 1.0)
            except Exception:
                motion = 1.0
            motion = float(_clamp(motion, 0.0, 5.0))

            # Advance a phase at a tempo proportional to drum energy.
            self._audio.cam_phase += dt * (0.55 + 3.40 * d_now)

            kick = float(_clamp(self._audio.kick, 0.0, 1.0))

            # The camera amplitude is proportional to drums, with a kick accent (intentionally strong).
            amp = motion * (0.04 + 0.22 * d_now) * (1.0 + 0.90 * kick)

            target_pan_x = math.cos(self._audio.cam_phase) * amp * 0.28
            target_pan_y = math.sin(self._audio.cam_phase * 1.11) * amp * 0.22
            target_rot = math.sin(self._audio.cam_phase * 0.75 + 1.7) * amp * 1.30

            # Safety clamps: strong motion, but never lose the scene completely.
            target_pan_x = float(_clamp(target_pan_x, -0.25, 0.25))
            target_pan_y = float(_clamp(target_pan_y, -0.20, 0.20))
            target_rot = float(_clamp(target_rot, -0.55, 0.55))
            # Smooth the camera to avoid jitter (stronger smoothing when quiet).
            alpha = float(_clamp(0.10 + 0.30 * d_now, 0.08, 0.45))
            self._audio.cam_pan_x = (1.0 - alpha) * self._audio.cam_pan_x + alpha * float(target_pan_x)
            self._audio.cam_pan_y = (1.0 - alpha) * self._audio.cam_pan_y + alpha * float(target_pan_y)
            self._audio.cam_rot = (1.0 - alpha) * self._audio.cam_rot + alpha * float(target_rot)

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
            w, h = self.physical_size
            screen_w = max(1, int(w))
            screen_h = max(1, int(h))

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
                # scene
                with self._scene_fbo:
                    gloo.set_viewport(0, 0, scene_w, scene_h)
                    gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                    self._prog_scene.draw("triangle_strip")

                # prefilter
                self._prog_prefilter["u_scene"] = self._scene_tex
                with self._bloom_fbo0:
                    gloo.set_viewport(0, 0, bloom_w, bloom_h)
                    gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                    self._prog_prefilter.draw("triangle_strip")

                # blur H
                self._prog_blur["u_tex"] = self._bloom_tex0
                self._prog_blur["u_texel"] = (1.0 / float(bloom_w), 1.0 / float(bloom_h))
                self._prog_blur["u_dir"] = (1.0, 0.0)
                with self._bloom_fbo1:
                    gloo.set_viewport(0, 0, bloom_w, bloom_h)
                    gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                    self._prog_blur.draw("triangle_strip")

                # blur V
                self._prog_blur["u_tex"] = self._bloom_tex1
                self._prog_blur["u_dir"] = (0.0, 1.0)
                with self._bloom_fbo0:
                    gloo.set_viewport(0, 0, bloom_w, bloom_h)
                    gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                    self._prog_blur.draw("triangle_strip")

                # combine
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


class _MechaChoirWidget(QWidget):
    def __init__(self, config: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._canvas = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not HAVE_VISPY:
            label = QLabel(
                "VisPy is not available.\n"
                "Mecha-Choir Constellations is disabled.\n"
                "Please install 'vispy' to enable this visualization.",
                self,
            )
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            return

        app.use_app("pyqt6")  # type: ignore[arg-type]
        self._canvas = _MechaChoirCanvas(config=self._config)  # type: ignore[name-defined]
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

    def update_audio(self, *, time_ms: int, drums: float, bass: float, guitars: float, vocals: float) -> None:
        if self._canvas is None:
            return
        self._canvas.set_audio(
            time_s=float(time_ms) / 1000.0,
            drums=drums,
            bass=bass,
            guitars=guitars,
            vocals=vocals,
        )

    def reset_audio_state(self) -> None:
        if self._canvas is None:
            return
        try:
            self._canvas.reset_audio_state()
        except Exception:
            pass


class MechaChoirConstellationsVisualization(BaseVisualization):
    plugin_id = "mecha_choir_constellations"
    plugin_name = "Mecha-Choir Constellations v1.9"
    plugin_description = (
        "Cyber ritual: segmented orbit rings + lightning arcs. "
        "Drums drive orbit speed, kick cuts, and smooth camera motion; "
        "Bass drives zoom; Vocals drive the central stage size; "
        "Full mix drives lightning/jitter."
    )
    plugin_author = "DrDLP"
    plugin_version = "1.9.0"
    plugin_max_inputs = 4

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._widget: Optional[_MechaChoirWidget] = None
        self._prev_t_s: float = -1.0

        # Defaults (keep them readable)
        self.config.setdefault("color_preset", "Cyan / Magenta")

        self.config.setdefault("orbit_count", 5.0)
        self.config.setdefault("orbit_radius", 0.58)
        self.config.setdefault("orbit_speed", 1.10)

        self.config.setdefault("segment_count", 28.0)
        self.config.setdefault("segment_thickness", 0.020)

        self.config.setdefault("lightning_intensity", 1.10)
        self.config.setdefault("camera_distance", 0.95)
        self.config.setdefault("camera_motion", 2.50)
        # Separate reactivity controls (less confusing than one global multiplier)
        self.config.setdefault("rhythm_reactivity", 1.35)   # drums -> speed/strobe/cuts
        self.config.setdefault("lightning_reactivity", 1.35) # full mix -> lightning/jitter
        self.config.setdefault("disc_reactivity", 1.60)     # vocals -> central disc size

        # background motif (4 corner clones). 0 disables.
        self.config.setdefault("bg_motif_scale", 0.55)
        self.config.setdefault("bg_motif_intensity", 0.18)

        # post
        self.config.setdefault("bloom", 1.15)
        self.config.setdefault("exposure", 1.10)
        self.config.setdefault("contrast", 1.05)

    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        # NOTE: Keep this list in sync with defaults in __init__ and shader uniforms.
        return {
            "color_preset": PluginParameter(
                name="color_preset",
                label="Color preset",
                type="enum",
                default="Cyan / Magenta",
                choices=["Cyan / Magenta", "Gold / Ember", "Acid Green", "Inferno", "Royal", "Mono Ice"],
            ),

            "orbit_count": PluginParameter(
                name="orbit_count", label="Orbit count", type="float",
                default=5.0, minimum=1.0, maximum=8.0, step=1.0
            ),
            "orbit_radius": PluginParameter(
                name="orbit_radius", label="Orbit radius", type="float",
                default=0.58, minimum=0.20, maximum=1.20, step=0.01
            ),
            "orbit_speed": PluginParameter(
                name="orbit_speed", label="Orbit speed", type="float",
                default=1.10, minimum=0.0, maximum=6.0, step=0.01
            ),

            "segment_count": PluginParameter(
                name="segment_count", label="Segment count", type="float",
                default=28.0, minimum=6.0, maximum=96.0, step=1.0
            ),
            "segment_thickness": PluginParameter(
                name="segment_thickness", label="Segment thickness", type="float",
                default=0.020, minimum=0.003, maximum=0.08, step=0.001
            ),

            "lightning_intensity": PluginParameter(
                name="lightning_intensity", label="Lightning intensity", type="float",
                default=1.10, minimum=0.0, maximum=3.0, step=0.01
            ),

            "camera_distance": PluginParameter(
                name="camera_distance", label="Camera distance", type="float",
                default=0.95, minimum=0.35, maximum=2.20, step=0.01
            ),
            "camera_motion": PluginParameter(
                name="camera_motion",
                label="Camera motion",
                type="float",
                default=2.50,
                minimum=0.0,
                maximum=5.0,
                step=0.01,
                description="Drums-driven smooth camera orbit amplitude (0 = static camera).",
            ),

            # Reactivity controls (these were not visible due to a malformed parameters() dict).
            "rhythm_reactivity": PluginParameter(
                name="rhythm_reactivity",
                label="Rhythm reactivity",
                type="float",
                default=1.35,
                minimum=0.25,
                maximum=5.0,
                step=0.01,
                description="Drums-driven strength for orbit speed, kick strobe, and cut resets.",
            ),
            "lightning_reactivity": PluginParameter(
                name="lightning_reactivity",
                label="Lightning reactivity",
                type="float",
                default=1.35,
                minimum=0.25,
                maximum=5.0,
                step=0.01,
                description="Full-mix-driven strength for lightning and segment jitter.",
            ),
            "disc_reactivity": PluginParameter(
                name="disc_reactivity",
                label="Disc reactivity",
                type="float",
                default=1.60,
                minimum=0.25,
                maximum=5.0,
                step=0.01,
                description="Vocals-driven strength for central disc size and wobble.",
            ),

            "bg_motif_scale": PluginParameter(
                name="bg_motif_scale",
                label="BG motif scale",
                type="float",
                default=0.55,
                minimum=0.0,
                maximum=1.50,
                step=0.01,
                description="Size of 4 background motif clones (corners). 0 disables.",
            ),
            "bg_motif_intensity": PluginParameter(
                name="bg_motif_intensity",
                label="BG motif intensity",
                type="float",
                default=0.18,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                description="Brightness of the 4 background motif clones. 0 disables their contribution.",
            ),

            # Post
            "bloom": PluginParameter(
                name="bloom", label="Bloom", type="float",
                default=1.15, minimum=0.0, maximum=3.0, step=0.01
            ),
            "exposure": PluginParameter(
                name="exposure", label="Exposure", type="float",
                default=1.10, minimum=0.1, maximum=3.0, step=0.01
            ),
            "contrast": PluginParameter(
                name="contrast", label="Contrast", type="float",
                default=1.05, minimum=0.6, maximum=2.0, step=0.01
            ),
        }

    def create_widget(self, parent: Optional[QWidget] = None) -> QWidget:
        self._widget = _MechaChoirWidget(config=self.config, parent=parent)
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
        i4 = inputs.get("input_4", {}) if isinstance(inputs.get("input_4", {}), dict) else {}

        # Internal scaling (kept simple; adjust in stems if needed)
        # Requested stem mapping:
        #   input_1: full mix (energy)  -> used as "guitars" driver for lightning/jitter
        #   input_2: bass              -> zoom / depth
        #   input_3: drums             -> orbit speed + kick/cuts + camera motion
        #   input_4: vocals            -> central stage size
        # More reactive mapping (still clamped for stability).
        # Per-system reactivity (less confusing than one global multiplier).
        try:
            rhythm_react = float(self.config.get("rhythm_reactivity", 1.35) or 1.35)
        except Exception:
            rhythm_react = 1.35
        rhythm_react = _clamp(rhythm_react, 0.25, 5.0)

        try:
            lightning_react = float(self.config.get("lightning_reactivity", 1.35) or 1.35)
        except Exception:
            lightning_react = 1.35
        lightning_react = _clamp(lightning_react, 0.25, 5.0)

        try:
            disc_react = float(self.config.get("disc_reactivity", 1.60) or 1.60)
        except Exception:
            disc_react = 1.60
        disc_react = _clamp(disc_react, 0.25, 5.0)

        # Full mix drives lightning/jitter: emphasize transients.
        full = _clamp((_extract_rms(i1, 0.0) ** 0.75) * 3.4 * lightning_react, 0.0, 4.0)

        # Bass drives zoom: keep it strong but stable (no extra reactivity slider for now).
        bass = _clamp((_extract_rms(i2, 0.0) ** 0.80) * 3.8, 0.0, 4.0)

        # Drums drive speed/strobe/cuts/camera: emphasize kicks.
        drums = _clamp((_extract_rms(i3, 0.0) ** 0.70) * 4.2 * rhythm_react, 0.0, 4.0)

        # Vocals drive disc size.
        vocals = _clamp((_extract_rms(i4, 0.0) ** 0.75) * 4.2 * disc_react, 0.0, 4.0)

        # We keep the shader input name "guitars" for the lightning system, but drive it from the full mix.
        guitars = full

        self._prev_t_s = t_s

        self._widget.update_audio(
            time_ms=time_ms,
            drums=drums,
            bass=bass,
            guitars=guitars,
            vocals=vocals,
        )