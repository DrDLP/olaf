"""
guitar_strings_screen.py

Audio-reactive "glowing strings" visualization for guitar.

Behavior
--------
- Input stem: stems/guitar.wav
- Output: vertical green-screen video with 5 diagonal "strings"
  that vibrate and emit sparks on guitar attacks.

Visual mapping
--------------
- Amplitude envelope (RMS) -> amplitude of the string vibration.
- Pitch (from pyin, normalized) -> spatial frequency and temporal
  speed of the wave along the string.
- Onsets -> small "sparks" that slide along the strings.

Output
------
- 720x1280 H.264 video, green background (#94C264),
  suitable for green-screen compositing inside a rectangular "screen"
  behind the guitarist.

Run this script from the `python` folder.
"""

import os
import numpy as np
import cv2
import librosa
from moviepy import VideoClip

# =========================
# Configuration
# =========================

# Output resolution (taller than wide for a vertical "screen")
OUT_WIDTH = 720
OUT_HEIGHT = 1280
OUT_FPS = 24
OUT_BITRATE = "8M"
OUT_FILENAME = "bass_strings_screen.mp4"

# Background color (green screen) #94C264 in BGR
BG_COLOR = (100, 194, 148)

# Guitar stem filename (relative to repo root)
GTR_STEM_NAME = "bass.wav"

# Number of strings
N_STRINGS = 4

# Geometry of the strings (in output image coordinates)
# Strings are roughly vertical, slightly diagonal.
TOP_MARGIN = 0.10  # fraction of height
BOT_MARGIN = 0.90
LEFT_MARGIN = 0.20  # fraction of width for leftmost string
STRING_SPACING = 0.12  # fraction of width between strings
DIAG_OFFSET = 0.05  # fraction of width added to x at top (diagonal tilt)

# String visual parameters
STRING_COLOR_BASE = 230  # base gray level for strings
STRING_THICKNESS = 7
STRING_VIB_AMP_MAX = 20.0  # maximum vibration amplitude (pixels)
STRING_VIB_POWER = 0.7     # non-linear mapping of energy -> amplitude

# Spark parameters
SPARK_LIFETIME = 0.7       # seconds that a spark travels along the string
SPARK_RADIUS_WEAK = 6
SPARK_RADIUS_STRONG = 10
SPARK_BRIGHTNESS_WEAK = 220
SPARK_BRIGHTNESS_STRONG = 255

# How we classify onsets by energy (percentiles)
ONSET_WEAK_PCT = 40.0
ONSET_STRONG_PCT = 75.0

# Wave parameters
# Spatial frequency along the string (cycles from bottom to top)
SPATIAL_FREQ_MIN = 1.0
SPATIAL_FREQ_MAX = 6.0

# Temporal wave speed (how fast the wave travels along the string)
TIME_FREQ_BASE = 1.0   # Hz
TIME_FREQ_EXTRA = 4.0  # additional Hz when pitch is high


# =========================
# Path helpers
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

STEMS_DIR = os.path.join(ROOT_DIR, "stems")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

GTR_STEM_PATH = os.path.join(STEMS_DIR, GTR_STEM_NAME)
OUT_PATH = os.path.join(OUTPUT_DIR, OUT_FILENAME)


# =========================
# Audio utilities
# =========================

def load_audio_normalized(path: str):
    """
    Load a mono audio file and apply gentle peak normalization.

    Returns
    -------
    y : np.ndarray
        Normalized mono audio signal.
    sr : int
        Sample rate.
    duration : float
        Duration in seconds.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    y, sr = librosa.load(path, mono=True)
    p95 = np.percentile(np.abs(y), 95)
    y = np.clip(y / (p95 + 1e-12), -1.0, 1.0)
    duration = len(y) / sr
    return y, sr, duration


def rms_envelope(y, sr, hop_length: int = 512,
                 frame_length: int = 2048,
                 smooth: float = 0.9):
    """
    Compute a smoothed RMS energy envelope and normalize it to 0..1.

    Returns
    -------
    t : np.ndarray
        Time points in seconds.
    e : np.ndarray
        Normalized energy envelope (0..1).
    """
    rms = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length
    )[0]
    t = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=hop_length
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


def make_interpolator(t, v):
    """
    Return a function f(t_query) that interpolates v at arbitrary times.
    If there is not enough data, return a constant 0 function.
    """
    if v.size <= 1:
        return lambda x: 0.0
    return lambda x: float(np.interp(x, t, v))


# =========================
# Load guitar & analyze
# =========================

print(f"[guitar_strings_screen] Loading guitar stem: {GTR_STEM_PATH}")
y_gtr, sr_gtr, D = load_audio_normalized(GTR_STEM_PATH)
print(f"[guitar_strings_screen] Audio duration: {D:.2f} s")

print("[guitar_strings_screen] Computing RMS envelope...")
t_env, e_env = rms_envelope(y_gtr, sr_gtr)
gtr_energy_at = make_interpolator(t_env, e_env)

print("[guitar_strings_screen] Estimating pitch (pyin)...")
f0, voiced_flag, voiced_prob = librosa.pyin(
    y_gtr,
    fmin=librosa.note_to_hz("E2"),  # typical low guitar range
    fmax=librosa.note_to_hz("E6"),
    frame_length=2048,
    hop_length=512,
)
t_pitch = librosa.times_like(f0, sr=sr_gtr, hop_length=512)

midi = librosa.hz_to_midi(f0)
mask = voiced_flag & np.isfinite(midi)

pitch_norm = np.zeros_like(midi, dtype=float)
if np.any(mask):
    midi_valid = midi[mask]
    lo, hi = np.percentile(midi_valid, [10, 90])
    if hi <= lo:
        hi = lo + 1.0
    pitch_norm[mask] = (midi[mask] - lo) / (hi - lo)
pitch_norm = np.clip(pitch_norm, 0.0, 1.0)
pitch_norm_at = make_interpolator(t_pitch, pitch_norm)

print("[guitar_strings_screen] Detecting onsets...")
onset_env = librosa.onset.onset_strength(y=y_gtr, sr=sr_gtr)
onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr_gtr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr_gtr)

n_onsets = len(onset_times)
print(f"[guitar_strings_screen] Found {n_onsets} onsets")

if n_onsets > 0:
    onset_energy = np.array([gtr_energy_at(float(t)) for t in onset_times])
    thr_weak = np.percentile(onset_energy, ONSET_WEAK_PCT)
    thr_strong = np.percentile(onset_energy, ONSET_STRONG_PCT)
    print(f"[guitar_strings_screen] Onset thresholds: weak<{thr_weak:.3f}, "
          f"strong>{thr_strong:.3f}")

    onset_strength = np.zeros(n_onsets, dtype=np.int8)  # 0=weak,1=medium,2=strong
    for i, e in enumerate(onset_energy):
        if e < thr_weak:
            onset_strength[i] = 0
        elif e < thr_strong:
            onset_strength[i] = 1
        else:
            onset_strength[i] = 2
else:
    onset_energy = np.array([], dtype=float)
    onset_strength = np.zeros(0, dtype=np.int8)

# Build "spark events": which onset belongs to which string
spark_events = []
for i, t0 in enumerate(onset_times):
    e = onset_energy[i] if n_onsets > 0 else 0.0
    p = pitch_norm_at(float(t0))
    # Map pitch to string index (low pitch -> lower index)
    idx = int(round(p * (N_STRINGS - 1)))
    idx = max(0, min(N_STRINGS - 1, idx))
    s = int(onset_strength[i])
    spark_events.append({
        "t0": float(t0),
        "string_idx": idx,
        "strength": s,
    })


# =========================
# Precompute string geometry
# =========================

# Define base endpoints for each string, from bottom to top, slightly diagonal
strings_base = []  # list of dicts with (p0, p1, dir, n)

for i in range(N_STRINGS):
    x0 = int(OUT_WIDTH * (LEFT_MARGIN + i * STRING_SPACING))
    y0 = int(OUT_HEIGHT * BOT_MARGIN)

    x1 = int(x0 + OUT_WIDTH * DIAG_OFFSET)
    y1 = int(OUT_HEIGHT * TOP_MARGIN)

    p0 = np.array([x0, y0], dtype=float)
    p1 = np.array([x1, y1], dtype=float)
    v = p1 - p0
    L = np.linalg.norm(v)
    if L < 1e-6:
        v = np.array([0.0, -1.0], dtype=float)
        L = 1.0
    dir_vec = v / L
    # Perpendicular unit vector (for vibration)
    n = np.array([-dir_vec[1], dir_vec[0]], dtype=float)

    strings_base.append({
        "p0": p0,
        "p1": p1,
        "dir": dir_vec,
        "normal": n,
        "length": L,
    })


# =========================
# Frame generator
# =========================

def make_frame(t: float):
    """
    MoviePy callback: return an RGB frame at time t.

    - Draw all strings with a sin wave deformation depending on energy & pitch.
    - Draw sparks moving along strings at onsets.
    """
    t = float(np.clip(t, 0.0, D))

    # Green background
    frame_bgr = np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=np.uint8)
    frame_bgr[:] = BG_COLOR

    # Current energy & pitch
    e = float(np.clip(gtr_energy_at(t), 0.0, 1.0))
    p = float(np.clip(pitch_norm_at(t), 0.0, 1.0))

    # String vibration amplitude
    vib_amp = STRING_VIB_AMP_MAX * (e ** STRING_VIB_POWER)

    # Wave spatial & temporal frequency
    spatial_freq = SPATIAL_FREQ_MIN + (SPATIAL_FREQ_MAX - SPATIAL_FREQ_MIN) * p
    time_freq = TIME_FREQ_BASE + TIME_FREQ_EXTRA * p
    phase_t = 2.0 * np.pi * time_freq * t

    # Draw strings
    N_POINTS = 200
    us = np.linspace(0.0, 1.0, N_POINTS)

    # Brightness can also depend a bit on energy
    brightness_f = STRING_COLOR_BASE + (255 - STRING_COLOR_BASE) * e
    brightness = int(np.clip(brightness_f, 0, 255))
    string_color = (brightness, brightness, brightness)

    for s in strings_base:
        p0 = s["p0"]
        p1 = s["p1"]
        n = s["normal"]

        pts = []
        for u in us:
            base = p0 + (p1 - p0) * u
            # Sinusoidal displacement along the normal
            disp = vib_amp * np.sin(2.0 * np.pi * spatial_freq * u - phase_t)
            pos = base + disp * n
            x, y = int(pos[0]), int(pos[1])
            pts.append([x, y])

        pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            frame_bgr,
            [pts],
            isClosed=False,
            color=string_color,
            thickness=STRING_THICKNESS,
            lineType=cv2.LINE_AA,
        )

    # Draw sparks
    for ev in spark_events:
        age = t - ev["t0"]
        if age < 0.0 or age > SPARK_LIFETIME:
            continue

        string_idx = ev["string_idx"]
        strength = ev["strength"]

        # Progress along the string: 0 -> 1 over SPARK_LIFETIME
        u = age / SPARK_LIFETIME
        u = float(np.clip(u, 0.0, 1.0))

        s = strings_base[string_idx]
        p0 = s["p0"]
        p1 = s["p1"]
        n = s["normal"]

        # Base position along string
        base = p0 + (p1 - p0) * u
        disp = vib_amp * np.sin(2.0 * np.pi * spatial_freq * u - phase_t)
        pos = base + disp * n
        x, y = int(pos[0]), int(pos[1])

        # Spark radius and base brightness depending on strength
        if strength >= 2:
            radius = SPARK_RADIUS_STRONG
            b0 = SPARK_BRIGHTNESS_STRONG
        elif strength == 1:
            radius = (SPARK_RADIUS_WEAK + SPARK_RADIUS_STRONG) // 2
            b0 = (SPARK_BRIGHTNESS_WEAK + SPARK_BRIGHTNESS_STRONG) // 2
        else:
            radius = SPARK_RADIUS_WEAK
            b0 = SPARK_BRIGHTNESS_WEAK

        # Fade out with age
        fade = 1.0 - (age / SPARK_LIFETIME)
        fade = float(np.clip(fade, 0.0, 1.0))
        b = int(np.clip(b0 * fade, 0, 255))
        color = (b, b, b)

        cv2.circle(
            frame_bgr,
            (x, y),
            radius,
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    # MoviePy expects RGB
    return frame_bgr[:, :, ::-1]


# =========================
# Render video
# =========================

print(f"[guitar_strings_screen] Rendering to {OUT_PATH}")
clip = VideoClip(make_frame, duration=D)
clip.write_videofile(
    OUT_PATH,
    fps=OUT_FPS,
    codec="libx264",
    audio=False,
    bitrate=OUT_BITRATE,
)
print("[guitar_strings_screen] Done.")
