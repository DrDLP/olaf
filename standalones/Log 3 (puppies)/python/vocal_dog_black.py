"""
vocal_dog_black.py

Audio-reactive kawaii backing/low-lead singer (black French bulldog).

Behavior
--------
- Uses stems/backing_vocals.wav as full backing energy.
- Uses stems/lead_vocals.wav BUT only the lowest 50% of detected pitches.
  Higher-pitched lead notes are ignored for this dog.
- Effective "vocal energy" = backing energy + low-pitch lead energy.
- Uses stems/rythm_merged.wav to drive:
    * idle alternation between idle1/idle2
    * light idle wobble (tilt)
    * singing alternation speed (sing1/sing2)

Sprites (relative to repo root, under assets/sprites):
    singer_black_idle1.png -> idle pose A
    singer_black_idle2.png -> idle pose B
    singer_black_sing1.png -> singing pose A
    singer_black_sing2.png -> singing pose B

Rules
-----
- When effective vocal energy is very low:
    - alternate idle1 / idle2 based on rhythm energy
    - apply a small tilt left/base/right/base based on rhythm
- When vocals are present (effective energy above a threshold):
    - alternate between sing1 and sing2, with a switching speed depending
      on the rhythm energy
    - apply a light zoom-in/out depending on vocal intensity
      (stronger vocals -> slightly larger sprite)

Output
------
- 480x480 H.264 video with green background (#94C264),
  suitable for green-screen compositing.

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
OUT_FILENAME = "vocal_dog_black_480.mp4"

# Background color (green screen) #94C264 in BGR
BG_COLOR = (100, 194, 148)

# Max sprite size inside the output frame
MAX_SPRITE_W = 440
MAX_SPRITE_H = 440

# Padding for rotation of idle sprites
PAD_PIX = 40

# Vocal energy thresholds (on normalized 0..1)
# Here "vocal energy" = backing + low-pitch part of lead
VOCAL_IDLE_THR = 0.01   # below -> idle

# Idle tilt parameters (driven by rhythm stem)
IDLE_ANGLE_MIN_DEG = 0.2
IDLE_ANGLE_MAX_DEG = 0.5
IDLE_TILT_CYCLE_HZ = 1.5  # pattern L -> base -> R -> base

# Idle alternation between idle1 / idle2
IDLE_BASE_FREQ = 1.0
IDLE_EXTRA_FREQ = 2.0  # extra oscillation speed if rhythm energy is high

# Singing alternation and zoom
SING_BASE_FREQ = 2.0   # minimum alternation speed sing1/sing2
SING_EXTRA_FREQ = 4.0  # extra alternation speed when rhythm energy is high

ZOOM_MIN = 1.00        # 1.0 = no zoom
ZOOM_MAX = 1.12        # light zoom at max vocal energy


# =========================
# Path helpers
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

STEMS_DIR = os.path.join(ROOT_DIR, "stems")
SPRITES_DIR = os.path.join(ROOT_DIR, "assets", "sprites")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LEAD_WAV = os.path.join(STEMS_DIR, "cleaned_vocals.wav")
BACK_WAV = os.path.join(STEMS_DIR, "backing_vocals.wav")
RYTHM_WAV = os.path.join(STEMS_DIR, "rythm_merged.wav")

SPR_IDLE1 = os.path.join(SPRITES_DIR, "singer_black_idle1.png")
SPR_IDLE2 = os.path.join(SPRITES_DIR, "singer_black_idle2.png")
SPR_SING1 = os.path.join(SPRITES_DIR, "singer_black_sing1.png")
SPR_SING2 = os.path.join(SPRITES_DIR, "singer_black_sing2.png")

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

print(f"[vocal_dog_black] Loading lead vocals: {LEAD_WAV}")
y_lead, sr_lead, D_lead = load_audio_normalized(LEAD_WAV)

print(f"[vocal_dog_black] Loading backing vocals: {BACK_WAV}")
y_back, sr_back, D_back = load_audio_normalized(BACK_WAV)

if sr_lead != sr_back:
    raise RuntimeError("Lead and backing vocals must have same sample rate")

print("[vocal_dog_black] Computing vocal envelopes...")
t_lead, e_lead = rms_envelope(y_lead, sr_lead)
t_back, e_back = rms_envelope(y_back, sr_back)

f_e_lead = make_interpolator(t_lead, e_lead)
f_e_back = make_interpolator(t_back, e_back)

# --- Pitch analysis on lead to get "low 50%" mask ---
print("[vocal_dog_black] Estimating lead pitch (pyin)...")
f0, voiced_flag, voiced_prob = librosa.pyin(
    y_lead,
    fmin=librosa.note_to_hz("A2"),  # adapt range if needed
    fmax=librosa.note_to_hz("C6"),
    frame_length=2048,
    hop_length=512,
)
t_pitch = librosa.times_like(f0, sr=sr_lead, hop_length=512)

midi = librosa.hz_to_midi(f0)
mask_valid = voiced_flag & np.isfinite(midi)

low_flag = np.zeros_like(midi, dtype=float)
if np.any(mask_valid):
    midi_valid = midi[mask_valid]
    # Threshold at 50th percentile (median): lower half of pitches
    thr = np.percentile(midi_valid, 50)
    low_flag[mask_valid] = (midi[mask_valid] <= thr).astype(float)
else:
    # Fallback: if no valid pitch, treat all as "low"
    low_flag[:] = 1.0

low_flag = np.clip(low_flag, 0.0, 1.0)
f_low_pitch_mask = make_interpolator(t_pitch, low_flag)

# Rhythm stem
print(f"[vocal_dog_black] Loading rhythm stem: {RYTHM_WAV}")
y_r, sr_r, D_r = load_audio_normalized(RYTHM_WAV)

print("[vocal_dog_black] Computing rhythm envelope...")
t_r, e_r = rms_envelope(y_r, sr_r)
f_e_r = make_interpolator(t_r, e_r)

# Clip duration: minimum of the three stems
D = min(D_lead, D_back, D_r)
print(f"[vocal_dog_black] Clip duration: {D:.2f} seconds")


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


print("[vocal_dog_black] Loading sprites...")
spr_idle1 = load_sprite(SPR_IDLE1)
spr_idle2 = load_sprite(SPR_IDLE2)
spr_sing1 = load_sprite(SPR_SING1)
spr_sing2 = load_sprite(SPR_SING2)

h0, w0 = spr_idle1.shape[:2]
scale = min(MAX_SPRITE_W / w0, MAX_SPRITE_H / h0, 1.0)
new_w = int(w0 * scale)
new_h = int(h0 * scale)

# Base frames resized
idle1_resized = cv2.resize(spr_idle1, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
idle2_resized = cv2.resize(spr_idle2, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
sing1_resized = cv2.resize(spr_sing1, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
sing2_resized = cv2.resize(spr_sing2, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

# Pad idle frames for rotation
PH = new_h + 2 * PAD_PIX
PW = new_w + 2 * PAD_PIX

idle_padded = []
for f in [idle1_resized, idle2_resized]:
    canvas = np.zeros((PH, PW, 3), dtype=np.uint8)
    canvas[:] = BG_COLOR
    y0 = (PH - new_h) // 2
    x0 = (PW - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = f
    idle_padded.append(canvas)

# Singing frames do not need padding; we will scale them directly.


# =========================
# Precompute per-frame state
# =========================

n_frames = int(np.ceil(D * OUT_FPS))
frame_times = np.arange(n_frames) / OUT_FPS

# Energy at each video frame
lead_e_frames = np.array([f_e_lead(float(t)) for t in frame_times])
back_e_frames = np.array([f_e_back(float(t)) for t in frame_times])
rythm_e_frames = np.array([f_e_r(float(t))   for t in frame_times])

# Low-pitch mask for lead, sampled at video frames (0 or 1-ish)
low_pitch_mask = np.array([f_low_pitch_mask(float(t)) for t in frame_times])
low_pitch_mask = np.clip(low_pitch_mask, 0.0, 1.0)

# Only keep lead energy where pitch is in the lower 50%
lead_low_e_frames = lead_e_frames * low_pitch_mask

# Effective vocal energy for this dog: backing + low-pitch lead
vocal_e_frames = np.clip(lead_low_e_frames + back_e_frames, 0.0, 1.0)

# State: 0 = idle, 1 = singing
state = np.zeros(n_frames, dtype=np.int8)
for k in range(n_frames):
    if vocal_e_frames[k] <= VOCAL_IDLE_THR:
        state[k] = 0
    else:
        state[k] = 1

# Idle tilt angles and idle1/idle2 alternation
idle_angles = np.zeros(n_frames, dtype=np.float32)
idle_variant = np.zeros(n_frames, dtype=np.int8)
phase_idle_toggle = 0.0

for k, t in enumerate(frame_times):
    # Rhythm energy for both tilt amplitude and alternation speed
    e_r = float(np.clip(rythm_e_frames[k], 0.0, 1.0))

    # Tilt amplitude
    amp = IDLE_ANGLE_MIN_DEG + (IDLE_ANGLE_MAX_DEG - IDLE_ANGLE_MIN_DEG) * e_r
    phase_tilt = t * IDLE_TILT_CYCLE_HZ * 4.0
    step = int(np.floor(phase_tilt)) % 4
    if step == 0:
        idle_angles[k] = -amp
    elif step == 2:
        idle_angles[k] = +amp
    else:
        idle_angles[k] = 0.0

    # Idle1 / idle2 alternation
    freq_toggle = IDLE_BASE_FREQ + IDLE_EXTRA_FREQ * e_r
    phase_idle_toggle += freq_toggle / OUT_FPS
    idle_variant[k] = int(np.floor(phase_idle_toggle)) % 2  # 0 or 1

# Singing alternation and zoom
sing_variant = np.zeros(n_frames, dtype=np.int8)
sing_zoom = np.ones(n_frames, dtype=np.float32)
phase_sing_toggle = 0.0

for k, t in enumerate(frame_times):
    e_r = float(np.clip(rythm_e_frames[k], 0.0, 1.0))
    e_v = float(np.clip(vocal_e_frames[k], 0.0, 1.0))

    # Alternate between sing1 and sing2
    freq_sing = SING_BASE_FREQ + SING_EXTRA_FREQ * e_r
    phase_sing_toggle += freq_sing / OUT_FPS
    sing_variant[k] = int(np.floor(phase_sing_toggle)) % 2  # 0 or 1

    # Zoom factor based on vocal intensity
    sing_zoom[k] = ZOOM_MIN + (ZOOM_MAX - ZOOM_MIN) * e_v


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

    frame = np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=np.uint8)
    frame[:] = BG_COLOR

    cx = OUT_WIDTH // 2
    cy = OUT_HEIGHT // 2

    if st == 0:
        # ----- IDLE -----
        idx_idle = idle_variant[k]
        angle_deg = idle_angles[k]

        sprite_padded = idle_padded[idx_idle]

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

        h_s, w_s = rot.shape[:2]
        x1 = cx - w_s // 2
        y1 = cy - h_s // 2
        x2 = x1 + w_s
        y2 = y1 + h_s

        # Safe clipping
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

    else:
        # ----- SINGING -----
        zoom = float(sing_zoom[k])
        base = sing1_resized if sing_variant[k] == 0 else sing2_resized

        # Apply zoom by resizing sprite around its center
        h_s, w_s = base.shape[:2]
        new_w_z = max(1, int(w_s * zoom))
        new_h_z = max(1, int(h_s * zoom))
        sprite_zoomed = cv2.resize(base, (new_w_z, new_h_z), interpolation=cv2.INTER_LINEAR)

        x1 = cx - new_w_z // 2
        y1 = cy - new_h_z // 2
        x2 = x1 + new_w_z
        y2 = y1 + new_h_z

        # Safe clipping
        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(OUT_WIDTH, x2)
        y2c = min(OUT_HEIGHT, y2)

        sx1 = x1c - x1
        sy1 = y1c - y1
        sx2 = sx1 + (x2c - x1c)
        sy2 = sy1 + (y2c - y1c)

        if x2c > x1c and y2c > y1c:
            frame[y1c:y2c, x1c:x2c] = sprite_zoomed[sy1:sy2, sx1:sx2]

    # MoviePy expects RGB
    return frame[:, :, ::-1]


# =========================
# Render video
# =========================

print(f"[vocal_dog_black] Rendering to {OUT_PATH}")
clip = VideoClip(make_frame, duration=D)
clip.write_videofile(
    OUT_PATH,
    fps=OUT_FPS,
    codec="libx264",
    audio=False,
    bitrate=OUT_BITRATE,
)
print("[vocal_dog_black] Done.")
