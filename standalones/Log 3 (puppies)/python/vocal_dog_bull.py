"""
vocal_dog_bull.py

Audio-reactive kawaii lead singer (metal bulldog).

Behavior (updated)
------------------
- Uses:
    * stems/cleaned_vocals.wav  -> lead vocals (energy + pitch for screams)
    * stems/backing_vocals.wav  -> normal singing (sing1/sing2)
    * stems/rythm_merged.wav    -> idle wobble + general drive

Sprites (relative to repo root, under assets/sprites):
    singer_bull_idle1.png   -> idle pose A
    singer_bull_idle2.png   -> idle pose B
    singer_bull_sing1.png   -> singing pose A (normal)
    singer_bull_sing2.png   -> singing pose B (normal)
    singer_bull_scream1.png -> high-note "scream" pose
    singer_bull_scream2.png -> scream pose (used for medium/high alt + low growl)

Rules
-----
- When total vocal energy is very low:
    - idle with idle1/idle2 + small tilt (rythm-driven)

- When vocals are present:
    * Normal singing:
        - triggered by backing_vocals (+ un peu de lead)
        - pose = sing1 ou sing2
        - IMPORTANT: sing1 ↔ sing2 only when pitch changes
          (|pitch(t)-pitch(t-1)| > PITCH_DELTA_SING)
        - light zoom according to vocal intensity

    * Screams (lead only):
        - triggered when lead energy high + pitch high / very low
        - high scream: scream_type = 1
            > uses scream1 / scream2, but only switches when pitch changes
        - low scream: scream_type = 2
            > always scream2 (growl)
        - once started, hold at least SCREAM_MIN_HOLD seconds

Priority:
    scream_high/low > normal singing > idle

Output
------
- 480x480 H.264 video with green background (#94C264),
  suitable for green-screen compositing.

Run from the `python` folder.
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
OUT_FILENAME = "vocal_dog_bull_480.mp4"

# Background color (green screen) #94C264 in BGR
BG_COLOR = (100, 194, 148)

# Max sprite size inside the output frame
MAX_SPRITE_W = 440
MAX_SPRITE_H = 440

# Padding for rotation of idle and scream sprites
PAD_PIX = 40

# Vocal energy thresholds (normalized 0..1)
VOCAL_IDLE_THR = 0.010      # below -> idle
BACK_SING_THR  = 0.005      # backing energy threshold for normal singing

# Scream thresholds based on lead vocals only
SCREAM_ENERGY_THR = 0.35    # lead energy threshold for scream
PITCH_HIGH_THR    = 0.50    # normalized pitch > this -> high scream
PITCH_LOW_THR     = 0.35    # normalized pitch < this -> low scream

# Pitch-change thresholds for sprite switching
PITCH_DELTA_SING   = 0.02   # change sprite sing1<->sing2 only if |Δpitch|>...
PITCH_DELTA_SCREAM = 0.04   # change scream1<->scream2 only if |Δpitch|>...

# Scream temporal logic
SCREAM_MIN_HOLD = 0.10      # seconds minimum in scream state

# Idle tilt parameters (driven by rhythm stem)
IDLE_ANGLE_MIN_DEG = 0.05
IDLE_ANGLE_MAX_DEG = 0.1
IDLE_TILT_CYCLE_HZ = 1.5    # pattern L -> base -> R -> base

# Idle alternation between idle1 / idle2
IDLE_BASE_FREQ = 1.0
IDLE_EXTRA_FREQ = 2.0       # extra oscillation speed if rhythm energy is high

# Singing zoom (no more time-based switching)
ZOOM_MIN = 1.00
ZOOM_MAX = 1.08

# Scream zoom and wobble
HIGH_SCREAM_ZOOM_MIN = 1.02
HIGH_SCREAM_ZOOM_MAX = 1.18
LOW_SCREAM_ZOOM_MIN  = 1.00
LOW_SCREAM_ZOOM_MAX  = 1.10

HIGH_SCREAM_ANGLE_MIN_DEG = 0.5
HIGH_SCREAM_ANGLE_MAX_DEG = 3.0
LOW_SCREAM_ANGLE_MIN_DEG  = 0.3
LOW_SCREAM_ANGLE_MAX_DEG  = 1.5

SCREAM_WOBBLE_BASE_HZ  = 1.0
SCREAM_WOBBLE_EXTRA_HZ = 2.0


# =========================
# Paths
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

STEMS_DIR = os.path.join(ROOT_DIR, "stems")
SPRITES_DIR = os.path.join(ROOT_DIR, "assets", "sprites")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LEAD_WAV   = os.path.join(STEMS_DIR, "cleaned_vocals.wav")
BACK_WAV   = os.path.join(STEMS_DIR, "backing_vocals.wav")
RYTHM_WAV  = os.path.join(STEMS_DIR, "rythm_merged.wav")

SPR_IDLE1   = os.path.join(SPRITES_DIR, "singer_bull_idle1.png")
SPR_IDLE2   = os.path.join(SPRITES_DIR, "singer_bull_idle2.png")
SPR_SING1   = os.path.join(SPRITES_DIR, "singer_bull_sing1.png")
SPR_SING2   = os.path.join(SPRITES_DIR, "singer_bull_sing2.png")
SPR_SCREAM1 = os.path.join(SPRITES_DIR, "singer_bull_scream1.png")
SPR_SCREAM2 = os.path.join(SPRITES_DIR, "singer_bull_scream2.png")

OUT_PATH = os.path.join(OUTPUT_DIR, OUT_FILENAME)


# =========================
# Audio utilities
# =========================

def load_audio_normalized(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    y, sr = librosa.load(path, mono=True)
    p95 = np.percentile(np.abs(y), 95)
    y = np.clip(y / (p95 + 1e-12), -1.0, 1.0)
    duration = len(y) / sr
    return y, sr, duration


def rms_envelope(y, sr, hop_length=512, frame_length=2048, smooth=0.9):
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
    if v.size <= 1:
        return lambda x: 0.0
    return lambda x: float(np.interp(x, t, v))


# =========================
# Load audio & envelopes
# =========================

print(f"[vocal_dog_bull] Loading lead vocals (cleaned): {LEAD_WAV}")
y_lead, sr_lead, D_lead = load_audio_normalized(LEAD_WAV)

print(f"[vocal_dog_bull] Loading backing vocals: {BACK_WAV}")
y_back, sr_back, D_back = load_audio_normalized(BACK_WAV)

if sr_lead != sr_back:
    raise RuntimeError("Lead and backing vocals must share the same sample rate")

print("[vocal_dog_bull] Computing vocal envelopes...")
t_lead, e_lead = rms_envelope(y_lead, sr_lead)
t_back, e_back = rms_envelope(y_back, sr_back)

f_e_lead = make_interpolator(t_lead, e_lead)
f_e_back = make_interpolator(t_back, e_back)

print("[vocal_dog_bull] Estimating pitch (pyin) from lead...")
f0, voiced_flag, _ = librosa.pyin(
    y_lead,
    fmin=librosa.note_to_hz("A2"),
    fmax=librosa.note_to_hz("C6"),
    frame_length=2048,
    hop_length=512,
)
t_pitch = librosa.times_like(f0, sr=sr_lead, hop_length=512)
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
f_pitch = make_interpolator(t_pitch, pitch_norm)

print(f"[vocal_dog_bull] Loading rhythm stem: {RYTHM_WAV}")
y_r, sr_r, D_r = load_audio_normalized(RYTHM_WAV)
print("[vocal_dog_bull] Computing rhythm envelope...")
t_r, e_r = rms_envelope(y_r, sr_r)
f_e_r = make_interpolator(t_r, e_r)

D = min(D_lead, D_back, D_r)
print(f"[vocal_dog_bull] Clip duration: {D:.2f} seconds")


# =========================
# Load sprites
# =========================

def load_sprite(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sprite not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read sprite: {path}")
    return img[:, :, :3]


print("[vocal_dog_bull] Loading sprites...")
spr_idle1   = load_sprite(SPR_IDLE1)
spr_idle2   = load_sprite(SPR_IDLE2)
spr_sing1   = load_sprite(SPR_SING1)
spr_sing2   = load_sprite(SPR_SING2)
spr_scream1 = load_sprite(SPR_SCREAM1)
spr_scream2 = load_sprite(SPR_SCREAM2)

h0, w0 = spr_idle1.shape[:2]
scale = min(MAX_SPRITE_W / w0, MAX_SPRITE_H / h0, 1.0)
new_w = int(w0 * scale)
new_h = int(h0 * scale)

idle1_resized   = cv2.resize(spr_idle1,   (new_w, new_h), interpolation=cv2.INTER_NEAREST)
idle2_resized   = cv2.resize(spr_idle2,   (new_w, new_h), interpolation=cv2.INTER_NEAREST)
sing1_resized   = cv2.resize(spr_sing1,   (new_w, new_h), interpolation=cv2.INTER_NEAREST)
sing2_resized   = cv2.resize(spr_sing2,   (new_w, new_h), interpolation=cv2.INTER_NEAREST)
scream1_resized = cv2.resize(spr_scream1, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
scream2_resized = cv2.resize(spr_scream2, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

PH = new_h + 2 * PAD_PIX
PW = new_w + 2 * PAD_PIX

idle_padded = []
for f in (idle1_resized, idle2_resized):
    canvas = np.zeros((PH, PW, 3), dtype=np.uint8)
    canvas[:] = BG_COLOR
    y0 = (PH - new_h) // 2
    x0 = (PW - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = f
    idle_padded.append(canvas)


# =========================
# Precompute per-frame state & variants
# =========================

n_frames = int(np.ceil(D * OUT_FPS))
frame_times = np.arange(n_frames) / OUT_FPS

lead_e_frames  = np.array([f_e_lead(float(t))  for t in frame_times])
back_e_frames  = np.array([f_e_back(float(t))  for t in frame_times])
pitch_frames   = np.array([f_pitch(float(t))   for t in frame_times])
rythm_e_frames = np.array([f_e_r(float(t))     for t in frame_times])

lead_e_frames  = np.clip(lead_e_frames,  0.0, 1.0)
back_e_frames  = np.clip(back_e_frames,  0.0, 1.0)
pitch_frames   = np.clip(pitch_frames,   0.0, 1.0)
rythm_e_frames = np.clip(rythm_e_frames, 0.0, 1.0)

total_vocal_frames = np.clip(back_e_frames + 0.5 * lead_e_frames, 0.0, 1.0)

# state: 0=idle, 1=sing_normal, 2=scream
state = np.zeros(n_frames, dtype=np.int8)
# scream_type: 0=no, 1=high scream, 2=low scream
scream_type = np.zeros(n_frames, dtype=np.int8)

scream_active = False
scream_start_time = 0.0
current_scream_type = 0

for k, t in enumerate(frame_times):
    eL = float(lead_e_frames[k])
    eB = float(back_e_frames[k])
    eT = float(total_vocal_frames[k])
    pV = float(pitch_frames[k])

    inst_high = (eL >= SCREAM_ENERGY_THR) and (pV >= PITCH_HIGH_THR)
    inst_low  = (eL >= SCREAM_ENERGY_THR) and (pV <= PITCH_LOW_THR)

    if scream_active:
        if (t - scream_start_time) < SCREAM_MIN_HOLD:
            pass
        else:
            if inst_high:
                current_scream_type = 1
                scream_start_time = t
            elif inst_low:
                current_scream_type = 2
                scream_start_time = t
            else:
                scream_active = False
                current_scream_type = 0
    else:
        if inst_high:
            scream_active = True
            current_scream_type = 1
            scream_start_time = t
        elif inst_low:
            scream_active = True
            current_scream_type = 2
            scream_start_time = t
        else:
            current_scream_type = 0

    if scream_active and current_scream_type != 0:
        state[k] = 2
        scream_type[k] = current_scream_type
    else:
        scream_type[k] = 0
        if eT <= VOCAL_IDLE_THR and eB <= BACK_SING_THR:
            state[k] = 0
        else:
            if eB >= BACK_SING_THR or eT > VOCAL_IDLE_THR:
                state[k] = 1
            else:
                state[k] = 0

# Idle tilt + idle1/idle2
idle_angles = np.zeros(n_frames, dtype=np.float32)
idle_variant = np.zeros(n_frames, dtype=np.int8)
phase_idle_toggle = 0.0

for k, t in enumerate(frame_times):
    e_r = float(rythm_e_frames[k])
    amp = IDLE_ANGLE_MIN_DEG + (IDLE_ANGLE_MAX_DEG - IDLE_ANGLE_MIN_DEG) * e_r
    phase_tilt = t * IDLE_TILT_CYCLE_HZ * 4.0
    step = int(np.floor(phase_tilt)) % 4
    if step == 0:
        idle_angles[k] = -amp
    elif step == 2:
        idle_angles[k] = +amp
    else:
        idle_angles[k] = 0.0

    freq_toggle = IDLE_BASE_FREQ + IDLE_EXTRA_FREQ * e_r
    phase_idle_toggle += freq_toggle / OUT_FPS
    idle_variant[k] = int(np.floor(phase_idle_toggle)) % 2

# Singing zoom
sing_zoom = np.ones(n_frames, dtype=np.float32)
for k in range(n_frames):
    eT = float(total_vocal_frames[k])
    sing_zoom[k] = ZOOM_MIN + (ZOOM_MAX - ZOOM_MIN) * eT

# Singing sprite variant (sing1/sing2) – only when pitch changes
sing_variant = np.zeros(n_frames, dtype=np.int8)
for k in range(1, n_frames):
    if state[k] == 1 and state[k-1] == 1:
        dp = abs(pitch_frames[k] - pitch_frames[k-1])
        if dp > PITCH_DELTA_SING:
            sing_variant[k] = 1 - sing_variant[k-1]
        else:
            sing_variant[k] = sing_variant[k-1]
    elif state[k] == 1 and state[k-1] != 1:
        sing_variant[k] = 0  # start from sing1
    else:
        sing_variant[k] = sing_variant[k-1]

# Scream zoom/angle
scream_zoom_high = np.ones(n_frames, dtype=np.float32)
scream_zoom_low  = np.ones(n_frames, dtype=np.float32)
scream_angle_high = np.zeros(n_frames, dtype=np.float32)
scream_angle_low  = np.zeros(n_frames, dtype=np.float32)

phase_scream = 0.0
for k, t in enumerate(frame_times):
    eL = float(lead_e_frames[k])
    pV = float(pitch_frames[k])
    level = float(np.clip(0.5 * (eL + pV), 0.0, 1.0))

    scream_zoom_high[k] = HIGH_SCREAM_ZOOM_MIN + (HIGH_SCREAM_ZOOM_MAX - HIGH_SCREAM_ZOOM_MIN) * level
    scream_zoom_low[k]  = LOW_SCREAM_ZOOM_MIN  + (LOW_SCREAM_ZOOM_MAX  - LOW_SCREAM_ZOOM_MIN)  * level

    freq = SCREAM_WOBBLE_BASE_HZ + SCREAM_WOBBLE_EXTRA_HZ * level
    phase_scream += freq / OUT_FPS * 2.0 * np.pi
    s = np.sin(phase_scream)

    amp_high = HIGH_SCREAM_ANGLE_MIN_DEG + (HIGH_SCREAM_ANGLE_MAX_DEG - HIGH_SCREAM_ANGLE_MIN_DEG) * level
    amp_low  = LOW_SCREAM_ANGLE_MIN_DEG  + (LOW_SCREAM_ANGLE_MAX_DEG  - LOW_SCREAM_ANGLE_MIN_DEG)  * level

    scream_angle_high[k] = amp_high * s
    scream_angle_low[k]  = amp_low  * s

# Scream sprite choice: 1=scream1, 2=scream2; switch only on pitch change for high screams
scream_sprite_choice = np.zeros(n_frames, dtype=np.int8)
prev_pitch = pitch_frames[0]
prev_choice = 1  # default scream1

for k in range(n_frames):
    if state[k] == 2:
        if scream_type[k] == 2:
            # low scream -> always scream2
            scream_sprite_choice[k] = 2
            prev_choice = 2
        else:
            # high scream -> toggle 1<->2 only when pitch changes
            dp = abs(pitch_frames[k] - prev_pitch)
            if dp > PITCH_DELTA_SCREAM:
                prev_choice = 1 if prev_choice == 2 else 2
            scream_sprite_choice[k] = prev_choice
        prev_pitch = pitch_frames[k]
    else:
        scream_sprite_choice[k] = 0  # not used in non-scream frames


# =========================
# Frame generator
# =========================

def make_frame(t: float):
    k = int(round(t * OUT_FPS))
    if k < 0:
        k = 0
    elif k >= n_frames:
        k = n_frames - 1

    st = int(state[k])
    frame = np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=np.uint8)
    frame[:] = BG_COLOR

    cx = OUT_WIDTH // 2
    cy = OUT_HEIGHT // 2

    if st == 0:
        # IDLE
        idx_idle = int(idle_variant[k])
        angle_deg = float(idle_angles[k])
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

    elif st == 1:
        # NORMAL SINGING
        zoom = float(sing_zoom[k])
        base = sing1_resized if int(sing_variant[k]) == 0 else sing2_resized

        h_s, w_s = base.shape[:2]
        new_w_z = max(1, int(w_s * zoom))
        new_h_z = max(1, int(h_s * zoom))
        sprite_zoomed = cv2.resize(base, (new_w_z, new_h_z), interpolation=cv2.INTER_LINEAR)

        x1 = cx - new_w_z // 2
        y1 = cy - new_h_z // 2
        x2 = x1 + new_w_z
        y2 = y1 + new_h_z

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

    else:
        # SCREAM
        stype = int(scream_type[k])
        choice = int(scream_sprite_choice[k])

        if stype == 2:
            zoom = float(scream_zoom_low[k])
            angle_deg = float(scream_angle_low[k])
        else:
            zoom = float(scream_zoom_high[k])
            angle_deg = float(scream_angle_high[k])

        if choice == 1:
            base_scream = scream1_resized
        else:
            base_scream = scream2_resized

        h_s, w_s = base_scream.shape[:2]
        new_w_z = max(1, int(w_s * zoom))
        new_h_z = max(1, int(h_s * zoom))
        zoomed = cv2.resize(base_scream, (new_w_z, new_h_z), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((PH, PW, 3), dtype=np.uint8)
        canvas[:] = BG_COLOR

        cx_c = PW // 2
        cy_c = PH // 2

        x1 = cx_c - new_w_z // 2
        y1 = cy_c - new_h_z // 2
        x2 = x1 + new_w_z
        y2 = y1 + new_h_z

        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(PW, x2)
        y2c = min(PH, y2)

        sx1 = x1c - x1
        sy1 = y1c - y1
        sx2 = sx1 + (x2c - x1c)
        sy2 = sy1 + (y2c - y1c)

        if x2c > x1c and y2c > y1c:
            canvas[y1c:y2c, x1c:x2c] = zoomed[sy1:sy2, sx1:sx2]

        if abs(angle_deg) > 1e-3:
            M = cv2.getRotationMatrix2D((PW / 2.0, PH / 2.0), angle_deg, 1.0)
            rot = cv2.warpAffine(
                canvas, M, (PW, PH),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=BG_COLOR,
            )
        else:
            rot = canvas

        h_s2, w_s2 = rot.shape[:2]
        x1 = cx - w_s2 // 2
        y1 = cy - h_s2 // 2
        x2 = x1 + w_s2
        y2 = y1 + h_s2

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

    return frame[:, :, ::-1]


# =========================
# Render video
# =========================

print(f"[vocal_dog_bull] Rendering to {OUT_PATH}")
clip = VideoClip(make_frame, duration=D)
clip.write_videofile(
    OUT_PATH,
    fps=OUT_FPS,
    codec="libx264",
    audio=False,
    bitrate=OUT_BITRATE,
)
print("[vocal_dog_bull] Done.")
