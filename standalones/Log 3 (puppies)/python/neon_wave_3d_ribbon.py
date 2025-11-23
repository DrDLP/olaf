"""
neon_wave_3d_ribbon.py

3D-looking neon waveform "ribbon" for the full mix.

Input
-----
- stems/full.wav : full song mix

Output
------
- output/neon_wave_3d_ribbon.mp4
  1920x1080 @ 24 fps, H.264, with audio from full.wav

Concept
-------
At each frame t:

1. We build several "layers" of waveform segments corresponding to
   times [t, t - Δt, t - 2Δt, ...] (a trailing history).
2. Each layer is a polyline in 3D space:
      X = span across screen
      Y = amplitude (audio)
      Z = depth (farther back in time)
3. We apply a perspective projection:
      x_screen = cx + f * X / Z
      y_screen = cy - f * Y / Z
4. We draw these lines back-to-front with Skia, add glow with OpenCV,
   and apply a subtle vignette.

Dependencies
------------
- numpy
- librosa
- opencv-python (cv2)
- skia-python
- moviepy

Run this script from the `python` folder.
"""

import os
import math
import numpy as np
import cv2
import librosa
import skia
from moviepy import VideoClip, AudioFileClip

# ------------------- CONFIG -------------------

W, H, FPS = 1280, 720, 24

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

STEMS_DIR = os.path.join(ROOT_DIR, "stems")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FULL_WAV = os.path.join(STEMS_DIR, "full.wav")
OUT_MP4 = os.path.join(OUTPUT_DIR, "neon_wave_3d_ribbon.mp4")
BITRATE = "14M"

# Horizontal coverage of the wave in "world units"
X_SPAN = 4.0          # larger -> wider wave in 3D
Y_SPAN = 1.8          # vertical scale in world units

# Trail configuration: how many depth layers, separated by TIME_STEP seconds
N_LAYERS = 50         # number of past layers in the trail
TIME_STEP = 0.07      # seconds between layers (total trail ~ N_LAYERS*TIME_STEP)
Z_START = 4.0         # nearest layer depth
Z_STEP  = 0.4         # depth increment per layer

# Number of points in each polyline along X
N_POINTS = 700

# Waveform window per layer (seconds)
WIN_SEC = 1.8

# Colors (BGR) for low/high spectral centroid
COL_LOW  = np.array([160,  80, 255], np.float32)  # bluish-magenta
COL_HIGH = np.array([255,  80, 255], np.float32)  # hot magenta

# Glow parameters (multi-scale)
GLOW_SIGMAS = [4.0, 8.0, 16.0, 28.0]
GLOW_GAINS  = [0.7, 0.5, 0.32, 0.22]

# Vignette strength
VIGNETTE_GAIN = 0.22

# ------------------------------------------------


def load_audio(path):
    """
    Load mono audio, gentle normalization.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio missing: {path}")
    y, sr = librosa.load(path, mono=True)
    p95 = np.percentile(np.abs(y), 95)
    y = np.clip(y / (p95 + 1e-12), -1.0, 1.0)
    dur = len(y) / sr
    return y, sr, dur


def rms_track(y, sr, hop=512, frame_length=2048, smooth=0.93):
    """
    Smoothed RMS energy, normalized 0..1.
    """
    rms = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop
    )[0]
    t = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=hop
    )

    out = np.empty_like(rms, dtype=float)
    acc = 0.0
    for i, v in enumerate(rms):
        acc = smooth * acc + (1.0 - smooth) * float(v)
        out[i] = acc

    lo, hi = np.percentile(out, [5, 95])
    out = (out - lo) / (hi - lo + 1e-12)
    out = np.clip(out, 0.0, 1.0)
    return t, out


def spectral_centroid_track(y, sr, hop=512, n_fft=2048, smooth=0.93):
    """
    Smoothed spectral centroid, normalized 0..1.
    """
    c = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop, n_fft=n_fft
    )[0]
    t = librosa.frames_to_time(
        np.arange(len(c)), sr=sr, hop_length=hop
    )
    c = (c - c.min())/(c.max() - c.min() + 1e-12)

    out = np.empty_like(c, dtype=float)
    acc = 0.0
    for i, v in enumerate(c):
        acc = smooth * acc + (1.0 - smooth) * float(v)
        out[i] = acc

    return t, np.clip(out, 0.0, 1.0)


def interp_fn(t, v):
    """
    Simple 1D interpolator over (t, v).
    """
    if v.size <= 1:
        return lambda x: 0.0
    return lambda x: float(np.interp(x, t, v))


def waveform_segment(y, sr, t_center, win_sec, n_points):
    """
    Extract a 1D waveform slice around t_center of duration win_sec,
    resampled to n_points samples in [-1, +1] (normalized time).
    """
    half = win_sec * 0.5
    t0 = max(0.0, t_center - half)
    t1 = min(len(y)/sr, t_center + half)
    if t1 <= t0:
        t1 = min(len(y)/sr, t_center + 1e-3)
    n = max(2, int((t1 - t0) * sr))
    s0 = int(t0 * sr)
    s1 = min(len(y), s0 + n)
    seg = y[s0:s1]
    if len(seg) < n:
        seg = np.pad(seg, (0, n-len(seg)), mode="edge")

    xs = np.linspace(0, len(seg) - 1, n_points, dtype=np.float32)
    seg_res = np.interp(xs, np.arange(len(seg), dtype=np.float32), seg)
    return seg_res.astype(np.float32)


def color_from_centroid(c01):
    """
    BGR color between COL_LOW and COL_HIGH for c in [0,1].
    """
    col = COL_LOW * (1.0 - c01) + COL_HIGH * c01
    return np.clip(col, 0, 255)


def to_skia_path(pts):
    """
    Convert 2D polyline pts (N,2) to Skia Path with quad smoothing.
    """
    path = skia.Path()
    x0, y0 = float(pts[0, 0]), float(pts[0, 1])
    path.moveTo(x0, y0)
    for i in range(1, len(pts) - 1):
        xm = (pts[i, 0] + pts[i+1, 0]) * 0.5
        ym = (pts[i, 1] + pts[i+1, 1]) * 0.5
        path.quadTo(float(pts[i, 0]), float(pts[i, 1]), float(xm), float(ym))
    path.lineTo(float(pts[-1, 0]), float(pts[-1, 1]))
    return path


def apply_glow(img_bgr, col_hint):
    """
    Multi-scale bloom/glow based on intensity, tinted by col_hint (BGR).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    acc = np.zeros_like(img_bgr, dtype=np.float32)

    for sigma, gain in zip(GLOW_SIGMAS, GLOW_GAINS):
        g = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        g = (g.astype(np.float32) / 255.0) * gain

        layer = np.zeros_like(acc)
        layer[:, :, 0] = g * (col_hint[0] / 255.0) * 255.0
        layer[:, :, 1] = g * (col_hint[1] / 255.0) * 255.0
        layer[:, :, 2] = g * (col_hint[2] / 255.0) * 255.0
        acc += layer

    out = np.clip(img_bgr.astype(np.float32) + acc, 0, 255).astype(np.uint8)
    return out


def add_vignette(img_bgr, strength=0.2):
    """
    Radial vignette.
    """
    h, w = img_bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2) / max(cx, cy)
    v = np.clip(1.0 - strength * (r**1.5), 0.18, 1.0)
    out = (img_bgr.astype(np.float32) * v[..., None]).astype(np.uint8)
    return out


# ------------ 3D projection helpers ------------

def build_3d_ribbon_layer(seg_amp, z_layer):
    """
    Build a 3D polyline for one layer.

    seg_amp : (N_POINTS,) amplitudes in [-1,1]
    z_layer : depth (world units)

    Returns: pts_cam (N_POINTS, 3) in camera coordinates, no rotation
    (camera at origin looking along +Z).
    """
    # x in [-X_SPAN, +X_SPAN]
    xs = np.linspace(-X_SPAN, X_SPAN, N_POINTS, dtype=np.float32)
    ys = seg_amp * Y_SPAN
    zs = np.full_like(xs, z_layer, dtype=np.float32)
    pts = np.stack([xs, ys, zs], axis=1)
    return pts


def project_points(pts_cam, f, cx, cy):
    """
    Perspective projection of 3D points in camera coords (X,Y,Z>0)
    to 2D screen coords (pixels).
    """
    x = pts_cam[:, 0]
    y = pts_cam[:, 1]
    z = pts_cam[:, 2] + 1e-6  # avoid div0

    sx = cx + f * (x / z)
    sy = cy - f * (y / z)

    return np.stack([sx, sy], axis=1)


# ----------------- Audio prep ------------------

print(f"[neon_wave_3d_ribbon] Loading full mix: {FULL_WAV}")
y, sr, D = load_audio(FULL_WAV)
print(f"[neon_wave_3d_ribbon] Duration: {D:.2f} s, SR={sr}")
print("[neon_wave_3d_ribbon] Computing RMS and spectral centroid tracks...")

t_rms, rms = rms_track(y, sr)
t_cent, cent = spectral_centroid_track(y, sr)

f_rms = interp_fn(t_rms, rms)
f_cent = interp_fn(t_cent, cent)

# Precompute times and maybe cap duration if needed
duration = D
n_video_frames = int(math.ceil(duration * FPS))

# Camera focal length (pixels)
F = W / 1.6
CX, CY = W / 2.0, H / 2.0

# ----------------- Frame function --------------

def make_frame(t):
    """
    MoviePy callback: generate one RGB frame at time t.
    """
    t = float(max(0.0, min(duration, t)))

    # Prepare Skia surface
    surface = skia.Surface(W, H)
    canvas = surface.getCanvas()
    canvas.clear(skia.ColorBLACK)

    # For glow tint, we'll accumulate an approximate color
    # based on centroid at the current time
    col_center = color_from_centroid(f_cent(t))

    # Draw layers from farthest to nearest (back-to-front)
    for layer_idx in reversed(range(N_LAYERS)):
        t_layer = t - layer_idx * TIME_STEP
        if t_layer < 0:
            continue

        seg = waveform_segment(y, sr, t_layer, WIN_SEC, N_POINTS)
        z_layer = Z_START + layer_idx * Z_STEP

        pts3 = build_3d_ribbon_layer(seg, z_layer)

        # We can modulate Y-scale slightly with RMS at that time
        e = f_rms(t_layer)
        pts3[:, 1] *= (0.7 + 0.9 * e)

        # Project to 2D
        pts2 = project_points(pts3, F, CX, CY)

        # Filter points which are inside screen bounds (slightly extended)
        # (Not strictly needed, but avoids drawing wildly offscreen)
        mask = (
            (pts2[:, 0] > -0.1 * W) & (pts2[:, 0] < 1.1 * W) &
            (pts2[:, 1] > -0.1 * H) & (pts2[:, 1] < 1.1 * H)
        )
        if not np.any(mask):
            continue

        pts2_clip = pts2[mask]

        # Build path
        path = to_skia_path(pts2_clip)

        # Color and width depend on depth & centroid
        # centroid closer to now (t_layer) for color.
        c_layer = color_from_centroid(f_cent(t_layer))
        r, g, b = int(c_layer[2]), int(c_layer[1]), int(c_layer[0])

        depth_norm = layer_idx / max(1, N_LAYERS - 1)
        # more distant => thinner & more transparent
        width = 3.5 * (1.0 - 0.6 * depth_norm)
        alpha = int(255 * (0.9 * (1.0 - 0.7 * depth_norm)))

        paint = skia.Paint(
            AntiAlias=True,
            Style=skia.Paint.kStroke_Style
        )
        paint.setStrokeWidth(width)
        paint.setColor(skia.Color(r, g, b, alpha))

        canvas.drawPath(path, paint)

    # Convert Skia surface to BGR image
    img = np.frombuffer(
        surface.makeImageSnapshot().tobytes(), dtype=np.uint8
    )
    img = img.reshape((H, W, 4))[:, :, :3][:, :, ::-1]  # RGBA->RGB->BGR

    # Glow
    img = apply_glow(img, col_center)

    # Vignette
    img = add_vignette(img, strength=VIGNETTE_GAIN)

    # MoviePy wants RGB
    return img[:, :, ::-1]


# ----------------- Render video ----------------

print(f"[neon_wave_3d_ribbon] Rendering to {OUT_MP4}")
clip = VideoClip(make_frame, duration=duration)

clip.write_videofile(
    OUT_MP4,
    fps=FPS,
    codec="libx264",
    audio_codec="aac",
    audio_bitrate="192k",
    bitrate=BITRATE,
)
print("[neon_wave_3d_ribbon] Done.")
