"""
drums_circles_screen.py

Audio-reactive concentric circles for drums, on a green background
suitable for green-screen compositing behind the band.

Behavior
--------
- Input drum stem: stems/drums_merged.wav
- Detect percussive onsets and their relative strength.
- Each hit spawns an expanding ring:
    * radius grows with time since hit
    * thickness and brightness depend on hit strength
    * rings fade out smoothly
- Multiple hits overlap in time, creating concentric circles.

Output
------
- 1920x1080 H.264 video with green background (#94C264).
- File: output/drums_circles_screen.mp4

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

# Output video
OUT_WIDTH = 1920
OUT_HEIGHT = 1080
OUT_FPS = 24
OUT_BITRATE = "8M"
OUT_FILENAME = "drums_circles_screen.mp4"

# Background color (green screen) #94C264 in BGR
BG_COLOR = (100, 194, 148)

# Drum stem filename (relative to repo root)
DRUM_STEM_NAME = "drums_merged.wav"

# Circle animation parameters
CENTER_X = OUT_WIDTH // 2
CENTER_Y = OUT_HEIGHT // 2

MIN_RADIUS = 80.0          # starting radius at hit time
RADIUS_SPEED = 600.0       # pixels per second (base speed)
MAX_LIFETIME = 0.8         # seconds: ring visible from t_hit to t_hit + MAX_LIFETIME

# Hit strength thresholds (percentiles of energy at onsets)
HIT_WEAK_PCT = 30.0
HIT_STRONG_PCT = 70.0

# For each strength, define visual scales
# strength 0 = weak, 1 = medium, 2 = strong
THICKNESS_BY_STRENGTH = {
    0: 12,
    1: 24,
    2: 32,
}
BRIGHTNESS_BY_STRENGTH = {
    0: 180,
    1: 220,
    2: 255,
}

# Fade curve: alpha = (1 - age / MAX_LIFETIME) ** FADE_POWER
FADE_POWER = 1.8


# =========================
# Path helpers
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

STEMS_DIR = os.path.join(ROOT_DIR, "stems")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DRUM_STEM_PATH = os.path.join(STEMS_DIR, DRUM_STEM_NAME)
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
# Load drums & detect hits
# =========================

print(f"[drums_circles_screen] Loading drum stem: {DRUM_STEM_PATH}")
y_d, sr_d, D = load_audio_normalized(DRUM_STEM_PATH)
print(f"[drums_circles_screen] Audio duration: {D:.2f} s")

print("[drums_circles_screen] Computing drum energy envelope...")
t_env_d, e_env_d = rms_envelope(y_d, sr_d)
energy_at = make_interpolator(t_env_d, e_env_d)

print("[drums_circles_screen] Detecting onsets...")
onset_env = librosa.onset.onset_strength(y=y_d, sr=sr_d)
onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr_d)
onset_times = librosa.frames_to_time(onset_frames, sr=sr_d)

n_hits = len(onset_times)
print(f"[drums_circles_screen] Found {n_hits} onsets")

if n_hits > 0:
    hit_energy = np.array([energy_at(float(t)) for t in onset_times])

    thr_weak = np.percentile(hit_energy, HIT_WEAK_PCT)
    thr_strong = np.percentile(hit_energy, HIT_STRONG_PCT)
    print(f"[drums_circles_screen] Hit thresholds: weak<{thr_weak:.3f}, "
          f"strong>{thr_strong:.3f}")

    # strength: 0=weak, 1=medium, 2=strong
    hit_strength = np.zeros(n_hits, dtype=np.int8)
    for i, e in enumerate(hit_energy):
        if e < thr_weak:
            hit_strength[i] = 0
        elif e < thr_strong:
            hit_strength[i] = 1
        else:
            hit_strength[i] = 2
else:
    hit_energy = np.array([], dtype=float)
    hit_strength = np.zeros(0, dtype=np.int8)

# Pack hits in a simple list of dicts (time + strength)
hits = [{"t": float(t), "s": int(hit_strength[i])}
        for i, t in enumerate(onset_times)]


# =========================
# Frame generator
# =========================

def make_frame(t: float):
    """
    MoviePy callback: return an RGB frame (H, W, 3) at time t.

    For each drum hit in the last MAX_LIFETIME seconds, draw an expanding
    circle centered in the frame, with radius and brightness depending on
    hit age and strength.
    """
    t = float(np.clip(t, 0.0, D))

    # Base green background (BGR)
    frame_bgr = np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=np.uint8)
    frame_bgr[:] = BG_COLOR

    if n_hits == 0:
        return frame_bgr[:, :, ::-1]  # to RGB

    # Only consider hits that are "alive" at time t
    t_min = t - MAX_LIFETIME
    if t_min < 0.0:
        t_min = 0.0

    for h in hits:
        th = h["t"]
        if th < t_min or th > t:
            continue
        age = t - th  # 0 .. MAX_LIFETIME
        life = age / MAX_LIFETIME
        life = float(np.clip(life, 0.0, 1.0))

        strength = h["s"]
        thickness = THICKNESS_BY_STRENGTH.get(strength, 3)
        base_brightness = BRIGHTNESS_BY_STRENGTH.get(strength, 220)

        # Fade-out factor
        alpha = (1.0 - life) ** FADE_POWER
        brightness = int(base_brightness * alpha)

        # Radius grows with age and strength
        radius = MIN_RADIUS + RADIUS_SPEED * age * (1.0 + 0.5 * strength)

        # Color (BGR): white-ish with brightness
        col = (brightness, brightness, brightness)

        # Draw the circle
        cv2.circle(
            frame_bgr,
            (CENTER_X, CENTER_Y),
            int(radius),
            col,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    # MoviePy expects RGB
    return frame_bgr[:, :, ::-1]


# =========================
# Render video
# =========================

print(f"[drums_circles_screen] Rendering to {OUT_PATH}")
clip = VideoClip(make_frame, duration=D)
clip.write_videofile(
    OUT_PATH,
    fps=OUT_FPS,
    codec="libx264",
    audio=False,
    bitrate=OUT_BITRATE,
)
print("[drums_circles_screen] Done.")
