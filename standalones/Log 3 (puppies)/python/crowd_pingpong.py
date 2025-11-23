"""
crowd_pingpong.py

Simplified audio-reactive crowd loop.

Inputs
------
Crowd frames (loop, first 30 only):
    assets/crowd/00001.png ... assets/crowd/00030.png

Stems (relative to repo root):
    stems/rythm_merged.wav   : global rhythm -> controls ping-pong speed
    stems/cleaned_vocals.wav : vocals -> pull the crowd towards high position
    stems/drums_merged.wav   : drums -> very slight horizontal oscillations
    stems/bass.wav           : bass  -> very fine zoom in/out

Behavior
--------
- Frames 0..29 correspond to:
    index 0   -> 00001.png (highest position)
    index 29  -> 00030.png (lowest position)

- The animation runs as a ping-pong loop over indices:
    [0,1,2,...,29,28,...,1] and so on.
  This loop is driven by rythm_merged (speed modulation).
  We start at the LOW position (index 29).

- Vocals:
    The energy of cleaned_vocals pulls the crowd towards the top:
        high vocal energy -> index near 0
        low vocal energy  -> index near 29

- If the crowd stays near the top for more than 0.2 s, we add a small
  local jitter to avoid freezing:
        index_final = base_index + pattern[-1, 0, +1, 0, ...] (clamped)

- Drums:
    Very slight horizontal oscillation (a few pixels) based on drums energy.

- Bass:
    Very fine zoom based on bass energy (a few percent).

Output
------
- H.264 video with same resolution as crowd frames.
- Duration = min duration of the stems used.
- File: output/crowd_control_simple.mp4

Run this script from the `python` folder.
"""

import os
import cv2
import glob
import numpy as np
import librosa
from moviepy import VideoClip

# =========================
# Configuration
# =========================

OUT_FPS = 24
OUT_BITRATE = "8M"
OUT_FILENAME = "crowd_loop.mp4"

# Ping-pong speed parameters (frames per video frame)
BASE_SPEED = 2.0           # base ping-pong speed
SPEED_BOOST_R = 1.5        # extra speed from rhythm energy

# Envelopes
RMS_FRAME_LENGTH = 2048
RMS_HOP_LENGTH = 512
RMS_SMOOTH = 0.9           # smoothing factor for RMS

# Influence of vocals on vertical position
VOCAL_INFLUENCE = 0.6      # 0 = ignore vocals, 1 = follow vocals only

# Definition of "high position"
HIGH_INDEX_THRESHOLD = 2   # indices <= this are considered "high"
HIGH_MIN_TIME = 0.1        # seconds above threshold before jitter starts

# Jitter pattern when staying high
JITTER_PATTERN = [0, 1, 0, -1]  # applied around base index (clamped)

# Drums lateral oscillation
DRUM_SWAY_MAX = 1.0        # max horizontal sway in pixels
DRUM_SWAY_FREQ = 2.0       # Hz

# Bass zoom
BASS_ZOOM_MAX = 0.03       # max additional scale factor (3%)


# =========================
# Paths
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

ASSETS_CROWD_DIR = os.path.join(ROOT_DIR, "assets", "crowd")
STEMS_DIR = os.path.join(ROOT_DIR, "stems")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RYTHM_WAV = os.path.join(STEMS_DIR, "rythm_merged.wav")
VOX_WAV   = os.path.join(STEMS_DIR, "cleaned_vocals.wav")
DRUMS_WAV = os.path.join(STEMS_DIR, "drums_merged.wav")
BASS_WAV  = os.path.join(STEMS_DIR, "bass.wav")

OUT_PATH = os.path.join(OUTPUT_DIR, OUT_FILENAME)


# =========================
# Audio utilities
# =========================

def load_audio_normalized(path: str):
    """
    Load mono audio and apply gentle peak normalization (95th percentile).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    y, sr = librosa.load(path, mono=True)
    p95 = np.percentile(np.abs(y), 95)
    y = np.clip(y / (p95 + 1e-12), -1.0, 1.0)
    duration = len(y) / sr
    return y, sr, duration


def rms_envelope(y, sr, hop_length: int, frame_length: int, smooth: float):
    """
    Compute a smoothed RMS energy envelope and normalize it to 0..1.
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
    Return a function f(t_query) interpolating v over t.
    If not enough data, returns constant 0.
    """
    if v.size <= 1:
        return lambda x: 0.0
    return lambda x: float(np.interp(x, t, v))


# =========================
# Load crowd frames (first 30 only)
# =========================

print("[crowd_control_simple] Loading crowd frames (00001..00030)...")

frame_files = []
for i in range(1, 31):
    fname = os.path.join(ASSETS_CROWD_DIR, f"{i:05d}.png")
    if os.path.isfile(fname):
        frame_files.append(fname)

if not frame_files:
    raise RuntimeError(
        f"No crowd frames 00001.png..00030.png found in {ASSETS_CROWD_DIR}"
    )

crowd_frames_bgr = []
for path in frame_files:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    crowd_frames_bgr.append(img)

N_FRAMES = len(crowd_frames_bgr)
print(f"[crowd_control_simple] Loaded {N_FRAMES} frames")

H, W = crowd_frames_bgr[0].shape[:2]
print(f"[crowd_control_simple] Frame size: {W}x{H}")

# Build ping-pong sequence indices for 0..N_FRAMES-1
# Example for N=4 : [0,1,2,3,2,1]
pingpong_seq = np.concatenate([
    np.arange(N_FRAMES, dtype=int),
    np.arange(N_FRAMES - 2, 0, -1, dtype=int)
])
PP_LEN = len(pingpong_seq)

# We want to START at the low position (index N_FRAMES-1)
# so we offset initial ping-pong position accordingly.
idx_low = N_FRAMES - 1  # 29 for 30 frames
start_pp_index = int(np.where(pingpong_seq == idx_low)[0][0])


# =========================
# Load stems & envelopes
# =========================

print(f"[crowd_control_simple] Loading stems:")

print(f"  rhythm: {RYTHM_WAV}")
y_r, sr_r, D_r = load_audio_normalized(RYTHM_WAV)

print(f"  vocals: {VOX_WAV}")
y_v, sr_v, D_v = load_audio_normalized(VOX_WAV)

print(f"  drums : {DRUMS_WAV}")
y_d, sr_d, D_d = load_audio_normalized(DRUMS_WAV)

print(f"  bass  : {BASS_WAV}")
y_b, sr_b, D_b = load_audio_normalized(BASS_WAV)

print("[crowd_control_simple] Computing RMS envelopes...")
t_r, e_r = rms_envelope(
    y_r, sr_r, hop_length=RMS_HOP_LENGTH,
    frame_length=RMS_FRAME_LENGTH, smooth=RMS_SMOOTH
)
t_v, e_v = rms_envelope(
    y_v, sr_v, hop_length=RMS_HOP_LENGTH,
    frame_length=RMS_FRAME_LENGTH, smooth=RMS_SMOOTH
)
t_d, e_d = rms_envelope(
    y_d, sr_d, hop_length=RMS_HOP_LENGTH,
    frame_length=RMS_FRAME_LENGTH, smooth=RMS_SMOOTH
)
t_b, e_b = rms_envelope(
    y_b, sr_b, hop_length=RMS_HOP_LENGTH,
    frame_length=RMS_FRAME_LENGTH, smooth=RMS_SMOOTH
)

f_e_r = make_interpolator(t_r, e_r)
f_e_v = make_interpolator(t_v, e_v)
f_e_d = make_interpolator(t_d, e_d)
f_e_b = make_interpolator(t_b, e_b)

# Duration = min of stems
D = min(D_r, D_v, D_d, D_b)
print(f"[crowd_control_simple] Clip duration: {D:.2f} s")

n_frames = int(np.ceil(D * OUT_FPS))
frame_times = np.arange(n_frames) / OUT_FPS

# Sample envelopes at video frame times
e_r_frames = np.array([f_e_r(t) for t in frame_times])
e_v_frames = np.array([f_e_v(t) for t in frame_times])
e_d_frames = np.array([f_e_d(t) for t in frame_times])
e_b_frames = np.array([f_e_b(t) for t in frame_times])

e_r_frames = np.clip(e_r_frames, 0.0, 1.0)
e_v_frames = np.clip(e_v_frames, 0.0, 1.0)
e_d_frames = np.clip(e_d_frames, 0.0, 1.0)
e_b_frames = np.clip(e_b_frames, 0.0, 1.0)

# =========================
# Precompute loop position, indices, jitter, offsets, zoom
# =========================

# 1) Ping-pong loop position driven by rhythm
loop_pos = np.zeros(n_frames, dtype=float)
pos = float(start_pp_index)

for k in range(n_frames):
    loop_pos[k] = pos
    # local speed based on rhythm energy
    er = float(e_r_frames[k])
    speed = BASE_SPEED * (1.0 + SPEED_BOOST_R * er)  # frames of ping-pong per video frame
    pos += speed

# 2) Base ping-pong index
base_pp_indices = pingpong_seq[(loop_pos.astype(int) % PP_LEN)]

# 3) Combine with vocals: pull towards high position (index 0)
combined_indices = np.zeros(n_frames, dtype=int)
for k in range(n_frames):
    base_idx = base_pp_indices[k]
    ev = float(e_v_frames[k])
    # Desired index from vocals: ev=0 -> low (N_FRAMES-1), ev=1 -> high (0)
    target_idx_vocal = (1.0 - ev) * (N_FRAMES - 1)
    idx_float = (1.0 - VOCAL_INFLUENCE) * base_idx + VOCAL_INFLUENCE * target_idx_vocal
    idx_int = int(round(idx_float))
    idx_int = max(0, min(N_FRAMES - 1, idx_int))
    combined_indices[k] = idx_int

# 4) Jitter when staying high > 0.2 s
final_indices = np.zeros_like(combined_indices)
time_in_high = 0.0
jitter_phase = 0
dt = 1.0 / OUT_FPS
pattern = np.array(JITTER_PATTERN, dtype=int)
pat_len = len(pattern)

for k in range(n_frames):
    idx = combined_indices[k]
    # Consider "high" if index <= HIGH_INDEX_THRESHOLD and vocal energy not trivial
    ev = float(e_v_frames[k])
    is_high = (idx <= HIGH_INDEX_THRESHOLD) and (ev > 0.1)

    if is_high:
        time_in_high += dt
    else:
        time_in_high = 0.0
        jitter_phase = 0  # reset phase when leaving high

    jitter_offset = 0
    if time_in_high > HIGH_MIN_TIME:
        # cycle pattern
        jitter_offset = int(pattern[jitter_phase % pat_len])
        jitter_phase += 1

    idx_j = idx + jitter_offset
    idx_j = max(0, min(N_FRAMES - 1, idx_j))

    final_indices[k] = idx_j

# 5) Drums-based horizontal sway
x_offsets = np.zeros(n_frames, dtype=float)
for k in range(n_frames):
    ed = float(e_d_frames[k])
    # amplitude grows with drums energy, but stays very small
    amp = DRUM_SWAY_MAX * (0.2 + 0.8 * ed)
    x_offsets[k] = amp * np.sin(2.0 * np.pi * DRUM_SWAY_FREQ * frame_times[k])

# 6) Bass-based fine zoom
zoom_factors = np.ones(n_frames, dtype=float)
for k in range(n_frames):
    eb = float(e_b_frames[k])
    zoom_factors[k] = 1.0 + BASS_ZOOM_MAX * eb  # scale slightly >1


# =========================
# Frame generator
# =========================

def make_frame(t: float):
    """
    MoviePy callback: return an RGB frame at time t.

    - Selects crowd frame according to final_indices[k].
    - Applies fine zoom based on bass.
    - Applies small horizontal shift based on drums.
    """
    k = int(round(t * OUT_FPS))
    if k < 0:
        k = 0
    elif k >= n_frames:
        k = n_frames - 1

    idx = int(final_indices[k])
    base = crowd_frames_bgr[idx]

    # 1) Zoom based on bass
    scale = float(zoom_factors[k])
    if abs(scale - 1.0) < 1e-3:
        zoomed = base
    else:
        new_w = max(1, int(round(W * scale)))
        new_h = max(1, int(round(H * scale)))
        resized = cv2.resize(base, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Center-crop back to (H, W)
        if new_w > W:
            x0 = (new_w - W) // 2
        else:
            x0 = 0
        if new_h > H:
            y0 = (new_h - H) // 2
        else:
            y0 = 0

        x1 = min(x0 + W, new_w)
        y1 = min(y0 + H, new_h)

        crop = resized[y0:y1, x0:x1]

        # If crop is smaller (possible when scale<1 by future changes),
        # pad it to (H, W)
        zoomed = np.zeros((H, W, 3), dtype=np.uint8)
        h_c, w_c = crop.shape[:2]
        zoomed[:h_c, :w_c] = crop

    # 2) Horizontal sway from drums
    x_off = int(round(x_offsets[k]))
    frame_bgr = np.zeros_like(zoomed)

    x_src_start = max(0, -x_off)
    x_src_end   = min(W, W - x_off)

    x_dst_start = max(0, x_off)
    x_dst_end   = x_dst_start + (x_src_end - x_src_start)

    # Vertical is unchanged
    y_src_start = 0
    y_src_end   = H
    y_dst_start = 0
    y_dst_end   = H

    if x_src_end > x_src_start and y_src_end > y_src_start:
        frame_bgr[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = \
            zoomed[y_src_start:y_src_end, x_src_start:x_src_end]

    # MoviePy expects RGB
    return frame_bgr[:, :, ::-1]


# =========================
# Render video
# =========================

print(f"[crowd_control_simple] Rendering to {OUT_PATH}")
clip = VideoClip(make_frame, duration=D)
clip.write_videofile(
    OUT_PATH,
    fps=OUT_FPS,
    codec="libx264",
    audio=False,   # visual only, audio added in post
    bitrate=OUT_BITRATE,
)
print("[crowd_pingpong] Done.")
