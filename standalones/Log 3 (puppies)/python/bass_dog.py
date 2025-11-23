"""
bass_dog.py

Audio-reactive kawaii bass player.

Behavior
--------
- Uses stems/bass.wav to detect when the bass is active and how strong it is.
- Uses stems/rythm_merged.wav to drive idle wobble and play animation speed.

Sprites (relative to repo root, under assets/sprites):
    bass_idle.png   -> idle pose (no bass playing)
    bass_play_1.png -> playing pose A (normal bass activity)
    bass_play2.png  -> playing pose B (normal bass activity)
    bass_high.png   -> strong accent pose (when bass is very loud)

Rules:
- When bass energy is very low -> stay on bass_idle.png, with a small tilt
  (left/base/right/base) driven by the rhythm stem.
- When bass energy is moderate -> alternate bass_play_1.png and bass_play2.png,
  with a switching speed that depends on the rhythm energy.
- When bass energy is strong (>= 70% of the normalized maximum) -> use
  bass_high.png and apply a stronger wobble/tilt driven by the bass signal.

Output:
- 480x480 H.264 video with a green background (#94C264) suitable for
  green-screen compositing.

This script is meant to be executed from the `python` folder.
"""

import os
import numpy as np
import cv2
import librosa
from moviepy import VideoClip

# =========================
# Global configuration
# =========================

OUT_WIDTH = 480
OUT_HEIGHT = 480
OUT_FPS = 24
OUT_BITRATE = "6M"
OUT_FILENAME = "bass_dog_480.mp4"

# Background color (green screen) #94C264 in BGR
BG_COLOR = (100, 194, 148)

# Max sprite size inside the output frame
MAX_SPRITE_W = 440
MAX_SPRITE_H = 440

# Padding for rotation (avoid clipping)
PAD_PIX = 40

# Bass energy thresholds (on normalized 0..1 envelope)
BASS_IDLE_THR = 0.06   # below -> idle
BASS_STRONG_THR = 0.70 # above -> "high" pose

# Idle tilt parameters (when bass is silent, driven by rhythm stem)
IDLE_ANGLE_MIN_DEG = 0.2
IDLE_ANGLE_MAX_DEG = 0.5
IDLE_TILT_CYCLE_HZ = 1.5  # pattern L -> base -> R -> base

# Play animation speed (alternation of play_1/play2) in Hz
PLAY_BASE_FREQ = 1.5  # minimum alternation speed
PLAY_EXTRA_FREQ = 3.0 # extra speed when rhythm energy is high

# Strong wobble when bass is loud (on bass_high pose)
HIGH_ANGLE_MIN_DEG = 1.5
HIGH_ANGLE_MAX_DEG = 5.0
HIGH_WOBBLE_BASE_HZ = 0.8
HIGH_WOBBLE_EXTRA_HZ = 1.5


# =========================
# Path helpers
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

STEMS_DIR = os.path.join(ROOT_DIR, "stems")
SPRITES_DIR = os.path.join(ROOT_DIR, "assets", "sprites")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASS_WAV = os.path.join(STEMS_DIR, "bass.wav")
RYTHM_WAV = os.path.join(STEMS_DIR, "rythm_merged.wav")

SPR_IDLE = os.path.join(SPRITES_DIR, "bass_idle.png")
SPR_PLAY1 = os.path.join(SPRITES_DIR, "bass_play1.png")  
SPR_PLAY2 = os.path.join(SPRITES_DIR, "bass_play2.png")
SPR_HIGH = os.path.join(SPRITES_DIR, "bass_high.png")

OUT_PATH = os.path.join(OUTPUT_DIR, OUT_FILENAME)


# =========================
# Audio utilities
# =========================

def load_audio_normalized(path: str):
    """
    Load a mono audio file and apply a gentle peak normalization.

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

    # Exponential moving average for smoother motion
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
# Load audio & envelopes
# =========================

print(f"[bass_dog] Loading bass stem: {BASS_WAV}")
y_bass, sr_bass, D_bass = load_audio_normalized(BASS_WAV)

print("[bass_dog] Computing bass energy envelope...")
t_bass_env, e_bass_env = rms_envelope(y_bass, sr_bass)
bass_energy_at = make_interpolator(t_bass_env, e_bass_env)

print(f"[bass_dog] Loading rhythm stem: {RYTHM_WAV}")
y_r, sr_r, D_r = load_audio_normalized(RYTHM_WAV)

print("[bass_dog] Computing rhythm envelope...")
t_r_env, e_r_env = rms_envelope(y_r, sr_r)
rythm_energy_at = make_interpolator(t_r_env, e_r_env)

# Clip duration: minimum of both stems
D = min(D_bass, D_r)
print(f"[bass_dog] Clip duration: {D:.2f} seconds")


# =========================
# Load sprites
# =========================

def load_sprite(path: str):
    """
    Load sprite as BGR image (discard any alpha channel).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sprite not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read sprite: {path}")
    return img[:, :, :3]


print("[bass_dog] Loading sprites...")
spr_idle = load_sprite(SPR_IDLE)
spr_play1 = load_sprite(SPR_PLAY1)
spr_play2 = load_sprite(SPR_PLAY2)
spr_high = load_sprite(SPR_HIGH)

h0, w0 = spr_idle.shape[:2]
scale = min(MAX_SPRITE_W / w0, MAX_SPRITE_H / h0, 1.0)
new_w = int(w0 * scale)
new_h = int(h0 * scale)

base_frames = [
    cv2.resize(spr_idle,   (new_w, new_h), interpolation=cv2.INTER_NEAREST),
    cv2.resize(spr_play1,  (new_w, new_h), interpolation=cv2.INTER_NEAREST),
    cv2.resize(spr_play2,  (new_w, new_h), interpolation=cv2.INTER_NEAREST),
    cv2.resize(spr_high,   (new_w, new_h), interpolation=cv2.INTER_NEAREST),
]

IDX_IDLE = 0
IDX_PLAY1 = 1
IDX_PLAY2 = 2
IDX_HIGH = 3

# Padding for rotation (idle & high poses)
PH = new_h + 2 * PAD_PIX
PW = new_w + 2 * PAD_PIX

padded_frames = []
for f in base_frames:
    canvas = np.zeros((PH, PW, 3), dtype=np.uint8)
    canvas[:] = BG_COLOR
    y0 = (PH - new_h) // 2
    x0 = (PW - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = f
    padded_frames.append(canvas)


# =========================
# Precompute per-frame state
# =========================

n_frames = int(np.ceil(D * OUT_FPS))
frame_times = np.arange(n_frames) / OUT_FPS

# Energy at each video frame
bass_e_frames = np.array([bass_energy_at(float(t)) for t in frame_times])
rythm_e_frames = np.array([rythm_energy_at(float(t)) for t in frame_times])

# State: 0 = idle, 1 = play (normal), 2 = high (strong)
state = np.zeros(n_frames, dtype=np.int8)
for k in range(n_frames):
    e = bass_e_frames[k]
    if e <= BASS_IDLE_THR:
        state[k] = 0
    elif e < BASS_STRONG_THR:
        state[k] = 1
    else:
        state[k] = 2

# Idle tilt angles
idle_angles = np.zeros(n_frames, dtype=np.float32)
for k, t in enumerate(frame_times):
    e_r = float(np.clip(rythm_energy_at(float(t)), 0.0, 1.0))
    amp = IDLE_ANGLE_MIN_DEG + (IDLE_ANGLE_MAX_DEG - IDLE_ANGLE_MIN_DEG) * e_r
    phase = t * IDLE_TILT_CYCLE_HZ * 4.0
    step = int(np.floor(phase)) % 4
    if step == 0:
        idle_angles[k] = -amp
    elif step == 2:
        idle_angles[k] = +amp
    else:
        idle_angles[k] = 0.0

# Play animation alternation (play1 / play2)
play_indices = np.zeros(n_frames, dtype=np.int8)
play_phase = 0.0
for k in range(n_frames):
    e_r = rythm_e_frames[k]
    freq = PLAY_BASE_FREQ + PLAY_EXTRA_FREQ * e_r
    play_phase += freq / OUT_FPS
    play_indices[k] = int(np.floor(play_phase)) % 2  # 0 or 1

# Strong wobble angles for high state
high_angles = np.zeros(n_frames, dtype=np.float32)
phase_high = 0.0
for k in range(n_frames):
    e_b = bass_e_frames[k]
    freq = HIGH_WOBBLE_BASE_HZ + HIGH_WOBBLE_EXTRA_HZ * e_b
    # Integrate phase for a smooth continuous wobble
    phase_high += freq / OUT_FPS * 2.0 * np.pi
    amp = HIGH_ANGLE_MIN_DEG + (HIGH_ANGLE_MAX_DEG - HIGH_ANGLE_MIN_DEG) * e_b
    high_angles[k] = amp * np.sin(phase_high)


# =========================
# Frame generator
# =========================

def make_frame(t: float):
    """
    MoviePy callback: return an RGB frame at time t.
    """
    k = int(round(t * OUT_FPS))
    if k < 0:
        k = 0
    elif k >= n_frames:
        k = n_frames - 1

    st = state[k]

    if st == 0:
        # Idle: small tilt based on rhythm, idle sprite
        sprite_idx = IDX_IDLE
        angle_deg = idle_angles[k]
    elif st == 1:
        # Normal play: alternate play_1 / play2, no tilt
        sprite_idx = IDX_PLAY1 if play_indices[k] == 0 else IDX_PLAY2
        angle_deg = 0.0
    else:
        # Strong level: high pose with stronger wobble based on bass
        sprite_idx = IDX_HIGH
        angle_deg = high_angles[k]

    sprite_padded = padded_frames[sprite_idx]

    # Apply rotation if needed
    if abs(angle_deg) > 1e-3:
        M = cv2.getRotationMatrix2D((PW / 2.0, PH / 2.0), angle_deg, 1.0)
        rot = cv2.warpAffine(
            sprite_padded, M, (PW, PH),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=BG_COLOR,
        )
    else:
        rot = sprite_padded

    frame = np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=np.uint8)
    frame[:] = BG_COLOR

    cx = OUT_WIDTH // 2
    cy = OUT_HEIGHT // 2

    x1 = cx - PW // 2
    y1 = cy - PH // 2
    x2 = x1 + PW
    y2 = y1 + PH

    # Safe clipping to avoid broadcasting errors if PW/PH > OUT_WIDTH/HEIGHT
    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(OUT_WIDTH, x2)
    y2c = min(OUT_HEIGHT, y2)

    sx1 = x1c - x1
    sy1 = y1c - y1
    sx2 = sx1 + (x2c - x1c)
    sy2 = sy1 + (y2c - y1c)

    if x2c > x1c and y2c > y1c:
        frame[y1c:y2c, x1c:x2c] = rot[sy1:sy2, sx1:sx2]

    # MoviePy expects RGB
    return frame[:, :, ::-1]


# =========================
# Render video
# =========================

print(f"[bass_dog] Rendering to {OUT_PATH}")
clip = VideoClip(make_frame, duration=D)
clip.write_videofile(
    OUT_PATH,
    fps=OUT_FPS,
    codec="libx264",
    audio=False,
    bitrate=OUT_BITRATE,
)
print("[bass_dog] Done.")
