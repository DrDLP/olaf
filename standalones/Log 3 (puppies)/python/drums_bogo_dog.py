"""
drums_bongo_dog.py

Audio-reactive bongo drummer using four cute sprites.

Rules:
- Input drum stem:  stems/drum_merged.wav
- Idle sprite:      assets/sprites/bongo_hold.png
- Medium hits:      alternate left/right paw
                    (bongo_left_paw.png / bongo_right_paw.png)
- Strong hits:      bongo_double_paw.png
- Very weak hits (below a threshold) are ignored (stay idle).
- After each hit, the sprite automatically returns to idle
  once the hit display window is over; if hits are closer than 0.3 s,
  they chain directly from one action sprite to the next.
- In idle state, the sprite slightly tilts left/right according to
  the energy of stems/rythm_merged.wav.

Output:
- A 480x480 H.264 video with green background (#94C264),
  ready to be used as a green-screen overlay.

This script is meant to be executed from the `python` folder.
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
OUT_WIDTH = 480
OUT_HEIGHT = 480
OUT_FPS = 24
OUT_BITRATE = "6M"
OUT_FILENAME = "drums_bongo_dog_480.mp4"

# Background color (green screen) #94C264 in BGR
BG_COLOR = (100, 194, 148)

# Maximum sprite size inside the 480x480 frame
MAX_SPRITE_W = 440
MAX_SPRITE_H = 440

# Padding around the sprite to allow small rotations without clipping
PAD_PIX = 40

# Hit display duration (seconds)
HIT_DISPLAY_DURATION = 0.05
# Below this hit energy, we ignore the hit
HIT_IGNORE_PERCENTILE = 10.0  # percentile for "weak" hits

# Above this percentile, a hit is considered "strong" (double paw)
HIT_STRONG_PERCENTILE = 70.0  # percentile for "strong" hits

# Idle tilt parameters (based on rhythm stem energy)
IDLE_ANGLE_MIN_DEG = 0.2
IDLE_ANGLE_MAX_DEG = 0.5
IDLE_TILT_CYCLE_HZ = 1.5  # L -> base -> R -> base

# =========================
# Path helpers
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

STEMS_DIR = os.path.join(ROOT_DIR, "stems")
SPRITES_DIR = os.path.join(ROOT_DIR, "assets", "sprites")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DRUM_STEM_PATH = os.path.join(STEMS_DIR, "drums_merged.wav")
RHYTHM_STEM_PATH = os.path.join(STEMS_DIR, "rythm_merged.wav")

SPR_IDLE = os.path.join(SPRITES_DIR, "bongo_hold.png")
SPR_LEFT = os.path.join(SPRITES_DIR, "bongo_left_paw.png")
SPR_RIGHT = os.path.join(SPRITES_DIR, "bongo_right_paw.png")
SPR_DOUBLE = os.path.join(SPRITES_DIR, "bongo_double_paw.png")

OUT_PATH = os.path.join(OUTPUT_DIR, OUT_FILENAME)


# =========================
# Audio utilities
# =========================

def load_audio_normalized(path: str):
    """
    Load a mono audio file and apply gentle peak normalization.

    Returns:
        y (np.ndarray): normalized mono signal
        sr (int): sample rate
        duration (float): duration in seconds
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

    Returns:
        t (np.ndarray): times in seconds
        e (np.ndarray): normalized energy (0..1)
    """
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    t = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr,
        hop_length=hop_length
    )

    # Exponential moving average smoothing
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
    Return a function f(t_query) interpolating v at arbitrary times.
    If v has size <= 1, return a constant 0 function.
    """
    if v.size <= 1:
        return lambda x: 0.0
    return lambda x: float(np.interp(x, t, v))


# =========================
# Load drum stem and detect hits
# =========================

print(f"[drums_bongo_dog] Loading drum stem: {DRUM_STEM_PATH}")
y_d, sr_d, D_d = load_audio_normalized(DRUM_STEM_PATH)

print("[drums_bongo_dog] Computing drum energy envelope...")
t_env_d, e_env_d = rms_envelope(y_d, sr_d)
energy_d_at = make_interpolator(t_env_d, e_env_d)

print("[drums_bongo_dog] Detecting onsets...")
onset_env = librosa.onset.onset_strength(y=y_d, sr=sr_d)
onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr_d)
onset_times = librosa.frames_to_time(onset_frames, sr=sr_d)

n_hits = len(onset_times)
print(f"[drums_bongo_dog] Found {n_hits} onsets")

if n_hits > 0:
    hit_energy = np.array([energy_d_at(float(t)) for t in onset_times])
    # Thresholds based on percentiles of hit energy
    thr_ignore = np.percentile(hit_energy, HIT_IGNORE_PERCENTILE)
    thr_strong = np.percentile(hit_energy, HIT_STRONG_PERCENTILE)
    print(f"[drums_bongo_dog] Hit thresholds: ignore<{thr_ignore:.3f}, "
          f"strong>{thr_strong:.3f}")

    # hit_types: 0 = ignored, 1 = medium, 2 = strong
    hit_types = np.zeros(n_hits, dtype=np.int8)
    for i, e in enumerate(hit_energy):
        if e < thr_ignore:
            hit_types[i] = 0
        elif e < thr_strong:
            hit_types[i] = 1
        else:
            hit_types[i] = 2

    # For medium hits, precompute left/right alternation
    medium_side = np.full(n_hits, -1, dtype=np.int8)  # -1=not medium, 0=left,1=right
    next_side = 0  # start with left
    for i in range(n_hits):
        if hit_types[i] == 1:
            medium_side[i] = next_side
            next_side = 1 - next_side
else:
    hit_energy = np.array([], dtype=float)
    hit_types = np.zeros(0, dtype=np.int8)
    medium_side = np.zeros(0, dtype=np.int8)

# =========================
# Load rhythm stem for idle tilt
# =========================

print(f"[drums_bongo_dog] Loading rhythm stem: {RHYTHM_STEM_PATH}")
y_r, sr_r, D_r = load_audio_normalized(RHYTHM_STEM_PATH)

print("[drums_bongo_dog] Computing rhythm envelope...")
t_env_r, e_env_r = rms_envelope(y_r, sr_r)
energy_r_at = make_interpolator(t_env_r, e_env_r)

# Clip duration = minimum of both stems
D = min(D_d, D_r)
print(f"[drums_bongo_dog] Clip duration: {D:.2f} seconds")

# =========================
# Load sprites
# =========================

def load_sprite(path: str):
    """
    Load a sprite as BGR (discard any existing alpha channel).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sprite not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read sprite: {path}")
    return img[:, :, :3]  # keep BGR only


print("[drums_bongo_dog] Loading sprites...")
spr_idle = load_sprite(SPR_IDLE)
spr_left = load_sprite(SPR_LEFT)
spr_right = load_sprite(SPR_RIGHT)
spr_double = load_sprite(SPR_DOUBLE)

h0, w0 = spr_idle.shape[:2]
scale = min(MAX_SPRITE_W / w0, MAX_SPRITE_H / h0, 1.0)
new_w = int(w0 * scale)
new_h = int(h0 * scale)

base_frames = [
    cv2.resize(spr_idle,   (new_w, new_h), interpolation=cv2.INTER_NEAREST),
    cv2.resize(spr_left,   (new_w, new_h), interpolation=cv2.INTER_NEAREST),
    cv2.resize(spr_right,  (new_w, new_h), interpolation=cv2.INTER_NEAREST),
    cv2.resize(spr_double, (new_w, new_h), interpolation=cv2.INTER_NEAREST),
]

IDX_IDLE = 0
IDX_LEFT = 1
IDX_RIGHT = 2
IDX_DOUBLE = 3

# Pad sprites so we can rotate idle without cropping
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
# Helper: which hit is active at time t?
# =========================

def current_hit_index(t: float):
    """
    Return the index of the active hit at time t, or -1 if none.

    A hit is "active" for HIT_DISPLAY_DURATION seconds after its onset.
    This ensures the sprite returns to idle between separated hits,
    while closely spaced hits can chain directly from one action to another.
    """
    if n_hits == 0:
        return -1

    dt = t - onset_times
    mask = (dt >= 0.0) & (dt < HIT_DISPLAY_DURATION)
    if not np.any(mask):
        return -1

    # Choose the most recent active hit
    dt_sel = dt.copy()
    dt_sel[~mask] = -1.0
    i = int(np.argmax(dt_sel))
    return i


def idle_angle_from_rhythm(t: float) -> float:
    """
    Compute a small idle tilt angle (in degrees) based on rhythm energy.
    Pattern: left -> base -> right -> base.
    """
    e = energy_r_at(t)
    e = float(np.clip(e, 0.0, 1.0))
    amp = IDLE_ANGLE_MIN_DEG + (IDLE_ANGLE_MAX_DEG - IDLE_ANGLE_MIN_DEG) * e
    phase = t * IDLE_TILT_CYCLE_HZ * 4.0  # four states per cycle
    k = int(np.floor(phase)) % 4
    if k == 0:
        return -amp
    elif k == 2:
        return +amp
    else:
        return 0.0


# =========================
# Frame generator
# =========================

def make_frame(t: float):
    """
    MoviePy callback: return an RGB frame (H, W, 3) at time t.
    """
    t = float(np.clip(t, 0.0, D))

    hit_idx = current_hit_index(t)

    if hit_idx < 0 or hit_types[hit_idx] == 0:
        # Idle state: small tilt based on rhythm
        sprite_idx = IDX_IDLE
        angle_deg = idle_angle_from_rhythm(t)
    else:
        # Active hit: choose left/right/ double paw, no tilt
        htype = hit_types[hit_idx]
        if htype == 2:
            sprite_idx = IDX_DOUBLE
        else:
            side = medium_side[hit_idx]
            if side == 0:
                sprite_idx = IDX_LEFT
            else:
                sprite_idx = IDX_RIGHT
        angle_deg = 0.0

    sprite_padded = padded_frames[sprite_idx]

    # Apply rotation for idle (only)
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

    # Compose on final 480x480 frame (with safe clipping)
    frame = np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=np.uint8)
    frame[:] = BG_COLOR

    cx = OUT_WIDTH // 2
    cy = OUT_HEIGHT // 2

    x1 = cx - PW // 2
    y1 = cy - PH // 2
    x2 = x1 + PW
    y2 = y1 + PH

    # Clip to valid bounds
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

print(f"[drums_bongo_dog] Rendering to {OUT_PATH}")
clip = VideoClip(make_frame, duration=D)
clip.write_videofile(
    OUT_PATH,
    fps=OUT_FPS,
    codec="libx264",
    audio=False,
    bitrate=OUT_BITRATE,
)
print("[drums_bongo_dog] Done.")
