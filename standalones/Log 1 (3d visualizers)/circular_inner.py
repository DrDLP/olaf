"""
Circular audio waveform histogram around a reactive black circle.

Concept
-------
- The audio waveform is sampled around the current time and mapped
  to a circular histogram of radial bars around a center.
- A black circle in the center changes its radius with the audio
  energy (percussive / overall loudness feeling).
- Colors of the bars outside this circle rotate / pulse with the rhythm.

Pipeline
--------
1. Load "song.wav" (mono or stereo).
2. Compute an RMS envelope with librosa (used as "energy").
3. For each video frame:
   - Sample a small window of the waveform around the corresponding time.
   - Build a circular histogram: one bar per sample/bin.
   - Inner radius = base circle radius + gain * (energy ** exponent).
   - Bar lengths = function of local sample amplitude.
   - Colors = hue around the circle + a phase driven by energy.
4. Render with pygfx to a silent MP4.
5. Mux original audio back onto the video with ffmpeg.

Dependencies
------------
pip install librosa pygfx wgpu pylinalg imageio[ffmpeg] imageio-ffmpeg
"""

import math
import subprocess
from pathlib import Path

import numpy as np
import librosa
from imageio import v2 as iio
import imageio_ffmpeg

import pygfx as gfx
import wgpu.gui.offscreen as wgpu_offscreen


# --------------------------------------------------------------------
# Global configuration / artist controls
# --------------------------------------------------------------------

# ---- I/O ---------------------------------------------------------------------

AUDIO_PATH = "song.wav"  # input audio file (mono or stereo)

OUTPUT_VIDEO_SILENT = "output_circular_hist_silent_inner.mp4"
OUTPUT_VIDEO_WITH_AUDIO = "output_circular_hist_with_audio_inner.mp4"

# ---- Video settings ----------------------------------------------------------

FPS = 24                 # frames per second for the output video
OUT_WIDTH = 1280         # output width in pixels
OUT_HEIGHT = 720         # output height in pixels

# Limit duration for quick previews; set to None for full track
MAX_DURATION_SEC = None  # e.g. 20.0 for 20 seconds

# ---- Audio analysis settings -------------------------------------------------

N_FFT = 2048             # FFT window size for RMS analysis
HOP_LENGTH = 512         # hop length for RMS envelope

# ---- Waveform histogram settings --------------------------------------------

N_BINS = 400             # number of radial bars around the circle
WAVE_WINDOW_SAMPLES = 400  # # of waveform samples used per frame (~N_BINS)

# Bar length parameters
MIN_BAR_LENGTH = 0.5     # minimum bar length (inner radius -> inner+MIN_BAR)
BAR_LENGTH_SCALE = 6.0   # scale factor for amplitude -> extra bar length

# Waveform amplitude normalization reference:
# 95th percentile of |waveform| is mapped sensibly to BAR_LENGTH_SCALE
AMP_REF_PERCENTILE = 95.0

# ---- Central black circle (radius reacts to energy) -------------------------

BASE_CIRCLE_RADIUS = 1.0   # base radius when energy is zero
CIRCLE_GAIN = 2.0          # how much circle radius grows with energy
CIRCLE_ENERGY_EXP = 1.5    # nonlinearity: energy ** exponent

# ---- Color / rhythm settings -------------------------------------------------

# Hue rotation over time: h = base_hue + phase + position_factor
BASE_HUE = 0.05          # base hue (0..1) ~ red/orange
HUE_SPREAD = 1.0         # how much the hue changes around the circle (0..1)

COLOR_PHASE_BASE = 0.01   # base hue phase increment per frame (slow rotation)
COLOR_PHASE_GAIN = 0.08   # additional phase increment proportional to energy

SATURATION = 0.9          # bar color saturation
VALUE_BASE = 0.4          # base brightness
VALUE_ENERGY_GAIN = 0.6   # extra brightness proportional to energy

# ---- Camera / viewing setup --------------------------------------------------

# The histogram lives in the XY-plane around the origin.
# We use an orthographic camera centered on the origin.
VIEW_RADIUS = 6.0         # extent of the visible region in world units

# ---- Central black circle (radius reacts to energy) -------------------------

BASE_CIRCLE_RADIUS = 1.0   # base radius when energy is zero (smaller inner disk)
CIRCLE_GAIN = 2.0          # how much circle radius grows with energy
CIRCLE_ENERGY_EXP = 1.5    # nonlinearity: energy ** exponent

# ---- Outer ring (bars) geometry ---------------------------------------------

# Fixed outer radius for the bar tips.
# The inner radius will move with the circle radius (energy),
# but the outer radius stays constant, so only the inner circle "pulses".
OUTER_RING_RADIUS = 5.0

# Shape of amplitude -> bar length mapping
BAR_EXPONENT = 0.8   # < 1.0 makes mid-levels more visible; > 1.0 emphasizes peaks only


# --------------------------------------------------------------------
# 1) AUDIO ANALYSIS
# --------------------------------------------------------------------

def load_audio_stereo(path: str):
    """Load audio (mono or stereo), normalize, return L/R, sr, duration."""
    print(f"[audio] Loading: {path}")
    y, sr = librosa.load(path, sr=None, mono=False)

    if y.ndim == 1:
        y_L = y.astype(np.float32)
        y_R = y.astype(np.float32)
        print("[audio] Detected mono, duplicating to L/R.")
    else:
        y_L = y[0].astype(np.float32)
        y_R = y[1].astype(np.float32)
        print("[audio] Detected stereo.")

    max_val = float(max(np.max(np.abs(y_L)), np.max(np.abs(y_R)), 1e-6))
    y_L /= max_val
    y_R /= max_val

    duration = len(y_L) / sr
    print(f"[audio] sr={sr}, duration={duration:.2f}s, samples={len(y_L)}")
    return y_L, y_R, sr, duration


def compute_rms_envelope(y_mono: np.ndarray):
    """Compute a normalized RMS envelope in [0, 1]."""
    print("[audio] Computing RMS envelope...")
    rms = librosa.feature.rms(
        y=y_mono,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
        center=True,
    )[0]  # shape (frames,)

    rms_min, rms_max = float(rms.min()), float(rms.max())
    if rms_max <= 1e-9:
        rms_norm = np.zeros_like(rms, dtype=np.float32)
    else:
        rms_norm = ((rms - rms_min) / (rms_max - rms_min + 1e-9)).astype(np.float32)

    print(f"[audio] RMS frames={len(rms_norm)}, min={rms_min:.4e}, max={rms_max:.4e}")
    return rms_norm


def get_energy_at_alpha(rms_norm: np.ndarray, alpha: float) -> float:
    """
    Return the normalized RMS energy at normalized time alpha in [0, 1].

    Parameters
    ----------
    rms_norm : np.ndarray
        Normalized RMS envelope, shape (frames,).
    alpha : float
        Normalized time in [0, 1].

    Returns
    -------
    float
        Energy value in [0, 1] for the given alpha.
    """
    n = len(rms_norm)
    idx = int(alpha * (n - 1))
    idx = max(0, min(idx, n - 1))
    return float(rms_norm[idx])


def compute_amp_ref(y_mono: np.ndarray) -> float:
    """
    Compute an amplitude reference for waveform bars.

    We use the AMP_REF_PERCENTILE of |y_mono| as reference so that
    "normal loudness" is mapped to a visually reasonable bar length.
    """
    amp = np.abs(y_mono)
    ref = float(np.percentile(amp, AMP_REF_PERCENTILE))
    ref = max(ref, 1e-4)
    print(f"[audio] Amplitude reference (p{AMP_REF_PERCENTILE:.1f}) = {ref:.4f}")
    return ref


# --------------------------------------------------------------------
# 2) PYGFX SCENE: histogram + black circle + camera
# --------------------------------------------------------------------

def create_circle_mesh(num_segments: int = 128):
    """
    Create a unit-radius filled disk mesh centered at the origin (XY plane).

    The actual radius will be adjusted later via local scaling.
    """
    # One center vertex + ring vertices
    positions = np.zeros((num_segments + 1, 3), dtype=np.float32)
    positions[0] = [0.0, 0.0, -0.1]  # slightly behind the bars

    for i in range(num_segments):
        angle = 2.0 * math.pi * i / num_segments
        x = math.cos(angle)
        y = math.sin(angle)
        positions[i + 1] = [x, y, -0.1]

    # Triangulate the disk as a fan
    indices = []
    for i in range(1, num_segments):
        indices.append([0, i, i + 1])
    indices.append([0, num_segments, 1])  # close the fan

    indices = np.array(indices, dtype=np.uint32)

    # Black circle
    geometry = gfx.Geometry(indices=indices, positions=positions)
    material = gfx.MeshBasicMaterial(color=(0.0, 0.0, 0.0, 1.0))
    circle = gfx.Mesh(geometry, material)
    return circle


def create_scene_with_histogram():
    """
    Create the pygfx scene:

      - A black circle mesh centered at the origin.
      - A circular histogram of waveform bars as a LineSegments object.
      - A dark background and an orthographic camera.
    """
    print("[gfx] Initializing circular histogram scene...")

    canvas = wgpu_offscreen.WgpuCanvas(size=(OUT_WIDTH, OUT_HEIGHT), pixel_ratio=1)
    renderer = gfx.renderers.WgpuRenderer(canvas)
    scene = gfx.Scene()

    # Background
    bg = gfx.Background.from_color("#000000")
    scene.add(bg)

    # Black circle at the center (unit radius initially)
    circle = create_circle_mesh(num_segments=128)
    scene.add(circle)

    # Histogram geometry: N_BINS line segments -> 2 vertices per bin
    positions = np.zeros((N_BINS * 2, 3), dtype=np.float32)
    colors = np.zeros((N_BINS * 2, 4), dtype=np.float32)
    colors[:, 3] = 1.0

    hist_geom = gfx.Geometry(positions=positions, colors=colors)
    hist_mat = gfx.LineMaterial(
        thickness=2.0,
        color_mode="vertex",
        aa=True,
    )
    hist_mat.alpha_mode = "blend"

    hist_line = gfx.Line(hist_geom, hist_mat)
    scene.add(hist_line)

    # Perspective camera centered on the origin, looking at XY plane.
    # This approximates an orthographic view for our flat histogram.
    aspect = OUT_WIDTH / OUT_HEIGHT
    cam = gfx.PerspectiveCamera(50, aspect)
    cam.local.position = (0.0, 0.0, 15.0)
    cam.look_at((0.0, 0.0, 0.0))


    renderer.effect_passes = []  # no post-processing

    canvas.request_draw(lambda: renderer.render(scene, cam))
    return canvas, renderer, scene, circle, hist_line, cam


# --------------------------------------------------------------------
# 3) HELPER: vectorized HSV -> RGB
# --------------------------------------------------------------------

def hsv_to_rgb_vectorized(h: np.ndarray, s: np.ndarray, v: np.ndarray):
    """
    Vectorized HSV -> RGB conversion for arrays in [0,1].
    h, s, v are np.ndarray of the same shape.
    Returns r, g, b each of same shape.
    """
    c = v * s
    h_ = (h * 6.0) % 6.0
    x_c = c * (1.0 - np.abs(h_ % 2.0 - 1.0))
    m = v - c

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    mask0 = (0.0 <= h_) & (h_ < 1.0)
    mask1 = (1.0 <= h_) & (h_ < 2.0)
    mask2 = (2.0 <= h_) & (h_ < 3.0)
    mask3 = (3.0 <= h_) & (h_ < 4.0)
    mask4 = (4.0 <= h_) & (h_ < 5.0)
    mask5 = (5.0 <= h_) & (h_ < 6.0)

    r[mask0], g[mask0], b[mask0] = c[mask0], x_c[mask0], 0.0
    r[mask1], g[mask1], b[mask1] = x_c[mask1], c[mask1], 0.0
    r[mask2], g[mask2], b[mask2] = 0.0, c[mask2], x_c[mask2]
    r[mask3], g[mask3], b[mask3] = 0.0, x_c[mask3], c[mask3]
    r[mask4], g[mask4], b[mask4] = x_c[mask4], 0.0, c[mask4]
    r[mask5], g[mask5], b[mask5] = c[mask5], 0.0, x_c[mask5]

    r += m
    g += m
    b += m

    return r, g, b


# --------------------------------------------------------------------
# 4) FRAME-BY-FRAME HISTOGRAM UPDATE
# --------------------------------------------------------------------

def compute_waveform_window(
    y_mono: np.ndarray,
    sr: int,
    t_sec: float,
    window_samples: int,
) -> np.ndarray:
    """
    Extract a waveform window around time t_sec from y_mono.

    Parameters
    ----------
    y_mono : np.ndarray
        Mono waveform in [-1, 1].
    sr : int
        Sample rate.
    t_sec : float
        Target time in seconds.
    window_samples : int
        Number of samples in the window.

    Returns
    -------
    np.ndarray
        Window of shape (window_samples,), zero-padded at boundaries if needed.
    """
    n = len(y_mono)
    center_idx = int(t_sec * sr)
    half = window_samples // 2
    start = center_idx - half
    end = start + window_samples

    # Clamp to valid range
    left_pad = 0
    right_pad = 0

    if start < 0:
        left_pad = -start
        start = 0
    if end > n:
        right_pad = end - n
        end = n

    window = y_mono[start:end]

    if left_pad > 0 or right_pad > 0:
        window = np.pad(window, (left_pad, right_pad), mode="constant")

    return window.astype(np.float32)


def update_histogram_geometry(
    hist_line: gfx.Line,
    circle: gfx.Mesh,
    y_mono: np.ndarray,
    sr: int,
    t_sec: float,
    energy: float,
    amp_ref: float,
    color_phase: float,
):
    """
    Update the circular histogram geometry and the black circle radius
    for a given frame.

    Parameters
    ----------
    hist_line : gfx.Line
        Line object representing the radial bars.
    circle : gfx.Mesh
        Black circle mesh (unit radius, scaled per frame).
    y_mono : np.ndarray
        Mono waveform.
    sr : int
        Sample rate.
    t_sec : float
        Current time in seconds.
    energy : float
        Audio energy in [0,1] at this time.
    amp_ref : float
        Normalization factor for waveform amplitude.
    color_phase : float
        Global phase influencing the hue rotation.
    """
    # 1) Circle radius from energy
    circle_radius = BASE_CIRCLE_RADIUS + CIRCLE_GAIN * (energy ** CIRCLE_ENERGY_EXP)

    # 2) Waveform window for this frame
    window = compute_waveform_window(y_mono, sr, t_sec, WAVE_WINDOW_SAMPLES)

    # Map window to exactly N_BINS bins (simple resampling via linear interpolation)
    if WAVE_WINDOW_SAMPLES != N_BINS:
        x_src = np.linspace(0.0, 1.0, WAVE_WINDOW_SAMPLES, endpoint=False)
        x_dst = np.linspace(0.0, 1.0, N_BINS, endpoint=False)
        window_bins = np.interp(x_dst, x_src, window)
    else:
        window_bins = window

    # 3) Amplitude -> bar length
    amp = np.abs(window_bins)
    amp_norm = np.clip(amp / amp_ref, 0.0, 1.0)

    # Optional shaping of the amplitude response:
    #   - BAR_EXPONENT < 1.0  => mid-levels accentuated
    #   - BAR_EXPONENT > 1.0  => only strong peaks get very long
    amp_shaped = amp_norm ** BAR_EXPONENT

    bar_lengths = MIN_BAR_LENGTH + BAR_LENGTH_SCALE * amp_shaped


    # 4) Build positions and colors
    positions = np.zeros((N_BINS * 2, 3), dtype=np.float32)
    colors = np.zeros((N_BINS * 2, 4), dtype=np.float32)

    # Hue along the circle, with global phase and spread
    bin_angles = np.linspace(0.0, 2.0 * math.pi, N_BINS, endpoint=False)
    bin_fracs = np.linspace(0.0, 1.0, N_BINS, endpoint=False)
    hues = (BASE_HUE + color_phase + HUE_SPREAD * bin_fracs) % 1.0

    # Brightness depends on energy (global)
    v = np.clip(VALUE_BASE + VALUE_ENERGY_GAIN * energy, 0.0, 1.0)
    s = np.full_like(hues, SATURATION, dtype=np.float32)
    v_arr = np.full_like(hues, v, dtype=np.float32)

    r, g, b = hsv_to_rgb_vectorized(hues, s, v_arr)

    for i in range(N_BINS):
        angle = bin_angles[i]

        # Fixed outer radius: the bar tips stay at OUTER_RING_RADIUS.
        outer_r = OUTER_RING_RADIUS

        # Interpret bar_lengths[i] as the *desired* thickness (radial extent) of the bar.
        # If there were no circle, the bar would go from (outer_r - thickness) to outer_r.
        desired_thickness = bar_lengths[i]
        inner_r_raw = outer_r - desired_thickness

        # The inner radius cannot go inside the black circle:
        # when the circle grows on peaks, it "pushes" the bars outward,
        # shortening them, but the outer edge remains fixed.
        inner_r = max(circle_radius, inner_r_raw)

        # Inner and outer positions (bars pointing outward, from inner -> outer)
        x0 = inner_r * math.cos(angle)
        y0 = inner_r * math.sin(angle)
        x1 = outer_r * math.cos(angle)
        y1 = outer_r * math.sin(angle)

        idx0 = 2 * i
        idx1 = 2 * i + 1

        positions[idx0] = [x0, y0, 0.0]
        positions[idx1] = [x1, y1, 0.0]

        colors[idx0] = [r[i], g[i], b[i], 1.0]
        colors[idx1] = [r[i], g[i], b[i], 1.0]



    # Update histogram geometry
    hist_line.geometry = gfx.Geometry(
        positions=positions.astype(np.float32),
        colors=colors.astype(np.float32),
    )

    # Update circle scale (unit circle -> circle_radius)
    circle.local.scale = (circle_radius, circle_radius, 1.0)


# --------------------------------------------------------------------
# 5) RENDER LOOP
# --------------------------------------------------------------------

def render_circular_hist_video(
    y_L: np.ndarray,
    y_R: np.ndarray,
    sr: int,
    duration: float,
    rms_norm: np.ndarray,
    amp_ref: float,
    out_path: str,
):
    """
    Render the circular waveform histogram as a silent MP4.

    Parameters
    ----------
    y_L, y_R : np.ndarray
        Left and right audio channels (we use y_mono internally).
    sr : int
        Sample rate.
    duration : float
        Duration of the video in seconds.
    rms_norm : np.ndarray
        Normalized RMS envelope for energy.
    amp_ref : float
        Amplitude reference for waveform normalization.
    out_path : str
        Output path for the silent video.
    """
    canvas, renderer, scene, circle, hist_line, cam = create_scene_with_histogram()

    y_mono = 0.5 * (y_L + y_R)

    total_frames = int(duration * FPS)
    total_frames = max(total_frames, 1)

    print(f"[render] FPS={FPS}, frames={total_frames}, video_duration={total_frames / FPS:.2f}s")

    writer = iio.get_writer(out_path, fps=FPS)

    # Color phase for hue rotation
    color_phase = 0.0

    try:
        for i in range(total_frames):
            alpha = i / max(1, total_frames - 1)
            t_sec = alpha * duration

            # Energy for this frame
            energy = get_energy_at_alpha(rms_norm, alpha)

            # Update color phase (controls hue rotation) based on energy
            color_phase = (color_phase + COLOR_PHASE_BASE +
                           COLOR_PHASE_GAIN * energy) % 1.0

            # Update histogram + circle
            update_histogram_geometry(
                hist_line,
                circle,
                y_mono,
                sr,
                t_sec,
                energy,
                amp_ref,
                color_phase,
            )

            # Offscreen render
            frame = canvas.draw()
            frame_rgba = np.asarray(frame)

            if frame_rgba.ndim != 3 or frame_rgba.shape[2] < 3:
                raise RuntimeError(
                    f"canvas.draw() returned unexpected shape {frame_rgba.shape}; "
                    "this usually indicates a GPU/backend error."
                )

            frame_rgb = frame_rgba[..., :3]
            writer.append_data(frame_rgb)

            if (i + 1) % 50 == 0 or i == total_frames - 1:
                print(f"[render] frame {i + 1}/{total_frames}")
    finally:
        writer.close()
        print(f"[render] Silent video written to: {out_path}")


# --------------------------------------------------------------------
# 6) MUX AUDIO
# --------------------------------------------------------------------

def mux_audio(video_path: str, audio_path: str, output_path: str):
    """
    Mux the original audio into the rendered silent video using ffmpeg.

    Parameters
    ----------
    video_path : str
        Path to the silent video file.
    audio_path : str
        Path to the original audio file.
    output_path : str
        Path to the output video file with audio.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_path,
    ]

    print("[mux] Running ffmpeg to add audio...")
    subprocess.run(cmd, check=True)
    print(f"[mux] Done. Output with audio: {output_path}")


# --------------------------------------------------------------------
# 7) MAIN
# --------------------------------------------------------------------

def main():
    """Entry point: load audio, analyze, render video, mux audio."""
    audio_file = Path(AUDIO_PATH)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    y_L, y_R, sr, duration = load_audio_stereo(str(audio_file))

    if MAX_DURATION_SEC is not None:
        duration = min(duration, MAX_DURATION_SEC)
        print(f"[main] Clamped duration to {duration:.2f}s for preview.")

    y_mono = 0.5 * (y_L + y_R)
    rms_norm = compute_rms_envelope(y_mono)
    amp_ref = compute_amp_ref(y_mono)

    # 1) Render silent video
    render_circular_hist_video(
        y_L,
        y_R,
        sr,
        duration,
        rms_norm,
        amp_ref,
        OUTPUT_VIDEO_SILENT,
    )

    # 2) Mux original audio
    mux_audio(OUTPUT_VIDEO_SILENT, str(audio_file), OUTPUT_VIDEO_WITH_AUDIO)


if __name__ == "__main__":
    print("[main] Starting circular waveform histogram render...")
    main()
    print("[main] Done.")
