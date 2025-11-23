"""
Audio-driven dense particle cloud (Python + librosa + pygfx).

Concept
-------
- A dense 3D cloud of colored particles around the origin.
- The camera moves inside the cloud, with a decentered look-at point.
- Audio energy (RMS) controls:
    * central flashes (brightness of particles near the center),
    * the rotation speed of the camera (faster on loud sections).
- Particle positions are static; only colors and camera motion change.
  This keeps the visual stable and avoids "against-rotation" illusions.

Outputs
-------
- output_particles_silent.mp4      : video only
- output_particles_with_audio.mp4  : video + original audio

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

AUDIO_PATH = "song.wav"  # Input audio file (mono or stereo)

OUTPUT_VIDEO_SILENT = "output_particles_silent.mp4"      # Video without audio
OUTPUT_VIDEO_WITH_AUDIO = "output_particles_with_audio.mp4"  # Final muxed output

# ---- Video settings ----------------------------------------------------------

FPS = 24                # Frames per second of the output video
OUT_WIDTH = 1280        # Output video width in pixels
OUT_HEIGHT = 720        # Output video height in pixels

# ---- Audio analysis settings -------------------------------------------------

N_FFT = 2048            # FFT window size for RMS analysis
HOP_LENGTH = 512        # Hop length between analysis frames

# ---- Particle cloud settings -------------------------------------------------

N_PARTICLES = 20000    # Number of particles in the cloud (performance-sensitive)
CLOUD_RADIUS = 10.0     # Target radius of the dense core after normalization
MAX_RADIUS = 10.0       # Maximum allowed radius; outliers are resampled

# Color / density shaping for radial distribution
R_NORM_EXPONENT = 4.0   # Exponent applied to normalized radius to compress the center
CENTER_WEIGHT_SIGMA = 0.35  # Width of the central "hot core" for flashes

# Outlier resampling
RESAMPLE_OUTLIER_STD = 0.5  # Std dev for re-sampling out-of-bounds particles

# ---- Visual style: particles & background ------------------------------------

BG_COLOR = "#020209"    # Background color of the scene
POINT_SIZE = 2.0        # Visual size of each particle in the renderer
POINT_ALPHA_MODE = "blend"  # Alpha blending mode for points

# Hue and saturation mapping (radial color gradient)
HUE_BASE = 0.02         # Hue at the very center of the cloud (0 = red)
HUE_RANGE = 0.88        # Hue range from center to edge (~warm to cool)
SAT_MAX = 0.85          # Maximum saturation at the center
SAT_DROP = 0.35         # Decrease in saturation towards the edge

# Value / brightness base levels
BASE_V_MIN = 0.25       # Minimum base value at the edge of the cloud
BASE_V_RANGE = 0.45     # Additional base value towards the center

# Flash shaping
FLASH_POWER = 2.5       # Nonlinearity for converting energy -> flash target
FLASH_SMOOTH = 0.4      # Smoothing factor for flash response in time [0..1]
FLASH_CENTER_EXP = 1.3  # Exponent applied to center_weight to emphasize the core
FLASH_STRENGTH = 4.0    # Strength of flash effect on brightness

# ---- Camera & motion controls -----------------------------------------------

# RMS smoothing for rotation control
ROT_SMOOTH = 0.3        # Smoothing factor for energy used for rotation speed
ROT_GAIN = 2.4          # Gain of audio energy on camera angular speed

# Base camera orbit
BASE_ORBITS = 1.8       # Approximate number of camera orbits for the whole clip

# Camera height / elevation evolution over normalized time t_norm in [0, 1]
CAM_HEIGHT_SMOOTH_FACTOR = 1.5  # How fast the elevation curve reaches its plateau
CAM_ELEV_START_DEG = 10.0       # Elevation at the beginning (near-horizontal)
CAM_ELEV_END_DEG = 45.0         # Elevation towards the end (more top-down)

# Camera radial motion (inside the cloud)
CAM_BASE_RADIUS = 7.0           # Mean camera radius (inside the cloud)
CAM_DEPTH_AMPLITUDE = 2.0       # Amplitude of "breathing" in/out motion
CAM_DEPTH_CYCLES = 3.0          # Number of in/out cycles over the whole video
CAM_DEPTH_MIN = 4.0             # Minimum radius clamp (avoid hitting the exact center)
CAM_DEPTH_MAX = 10.0            # Maximum radius clamp (avoid leaving the cloud)

# Target (look-at) motion inside the cloud
TARGET_BASE_RADIUS = 4.0        # Mean radius of the look-at point
TARGET_PHASE = 0.5              # Angular offset between camera angle and target angle (in radians)
TARGET_ELEV_SCALE = 0.4         # Relative elevation of the target vs. camera elevation
TARGET_DEPTH_AMPLITUDE = 1.0    # Amplitude of radial motion of the look-at point
TARGET_DEPTH_FREQ = 1.0         # Frequency of target depth oscillation over t_norm
TARGET_DEPTH_PHASE = 1.3        # Phase offset for target depth oscillation
TARGET_DEPTH_MIN = 2.0          # Minimum radius clamp for the target point
TARGET_DEPTH_MAX = 7.0          # Maximum radius clamp for the target point

# ---- Duration / preview ------------------------------------------------------

# Set to None for full track, or to a float (seconds) for a shorter preview
MAX_DURATION_SEC = None


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


# --------------------------------------------------------------------
# 2) PYGFX SCENE: dense particle cloud + camera
# --------------------------------------------------------------------

def create_scene_with_particles():
    """
    Create a pygfx scene with:

      - a dense particle cloud (gfx.Points) centered at the origin,
      - a dark background,
      - a perspective camera.

    Initial positions/colors are placeholders; they are updated after
    the particles are initialized.
    """
    print("[gfx] Initializing particle cloud scene...")

    canvas = wgpu_offscreen.WgpuCanvas(size=(OUT_WIDTH, OUT_HEIGHT), pixel_ratio=1)
    renderer = gfx.renderers.WgpuRenderer(canvas)
    scene = gfx.Scene()

    # Background
    bg = gfx.Background.from_color(BG_COLOR)
    scene.add(bg)

    # Dummy initial geometry; will be replaced once real positions/colors are computed
    positions = np.zeros((N_PARTICLES, 3), dtype=np.float32)
    colors = np.zeros((N_PARTICLES, 4), dtype=np.float32)
    colors[:, 3] = 1.0

    geometry = gfx.Geometry(positions=positions, colors=colors)
    material = gfx.PointsMaterial(
        size=POINT_SIZE,
        color_mode="vertex",
    )
    material.alpha_mode = POINT_ALPHA_MODE

    points = gfx.Points(geometry, material)
    scene.add(points)

    # Camera (its position will be controlled by animate_camera())
    aspect = OUT_WIDTH / OUT_HEIGHT
    camera = gfx.PerspectiveCamera(60, aspect)
    camera.local.position = (0.0, 0.0, 40.0)
    camera.look_at((0.0, 0.0, 0.0))

    renderer.effect_passes = []  # keep it simple, no post-processing

    canvas.request_draw(lambda: renderer.render(scene, camera))
    return canvas, renderer, scene, points, camera, material


# --------------------------------------------------------------------
# 3) PARTICLE INITIALIZATION + COLOR COMPUTATION
# --------------------------------------------------------------------

def initialize_particles():
    """
    Initialize static particle positions and derived radial info.

    - Positions: 3D Gaussian cloud, scaled so that most points lie
      within CLOUD_RADIUS, and outliers clipped.
    - r_norm: radius normalized to [0, 1] (after shaping).
    - center_weight: weight in [0, 1] emphasizing the central core for flashes.
    """
    print("[particles] Initializing dense cloud...")

    # Sample from 3D Gaussian around origin
    positions = np.random.normal(loc=0.0, scale=1.0, size=(N_PARTICLES, 3)).astype(np.float32)

    # Normalize radial distribution so that most points are within CLOUD_RADIUS
    radius = np.linalg.norm(positions, axis=1)
    eps = 1e-6
    r95 = np.percentile(radius, 95.0) + eps
    positions *= (CLOUD_RADIUS / r95)

    # Recompute radius after scaling and resample outliers beyond MAX_RADIUS
    radius = np.linalg.norm(positions, axis=1)
    too_far = radius > MAX_RADIUS
    if np.any(too_far):
        n = int(too_far.sum())
        positions[too_far] = np.random.normal(
            loc=0.0,
            scale=RESAMPLE_OUTLIER_STD,
            size=(n, 3),
        )

    # Compute normalized radius in [0, 1] (based on the 99th percentile)
    radius = np.linalg.norm(positions, axis=1)
    max_r = np.percentile(radius, 99.0) + eps
    r_norm = np.clip(radius / max_r, 0.0, 1.0).astype(np.float32)

    # Radial shaping: compress inner region to make the core visually denser
    r_norm = r_norm ** R_NORM_EXPONENT

    # Center weight: Gaussian kernel emphasizing the central core
    center_weight = np.exp(-(r_norm / CENTER_WEIGHT_SIGMA) ** 2).astype(np.float32)

    print("[particles] Cloud ready.")
    return positions, r_norm, center_weight


def compute_particle_colors(
    r_norm: np.ndarray,
    center_weight: np.ndarray,
    flash_level: float,
):
    """
    Compute per-particle colors (RGBA) based on:

      - radial distance (r_norm),
      - central weight (center_weight),
      - global flash_level in [0, 1].

    The visual style is a "heatmap" gradient:
      - warm, bright colors near the center,
      - cooler, darker colors towards the outer shell,
      - flashes amplify brightness from the core outward.
    """
    n = r_norm.shape[0]
    colors = np.empty((n, 4), dtype=np.float32)

    # Hue: center -> warm (red/orange), outer -> cooler (magenta/blue)
    h = (HUE_BASE + HUE_RANGE * r_norm).astype(np.float32)

    # Saturation: high, slightly decreasing with radius
    s = (SAT_MAX - SAT_DROP * r_norm).astype(np.float32)

    # Base value/brightness: higher near center, slightly dimmer at edge
    base_v = (BASE_V_MIN + BASE_V_RANGE * (1.0 - r_norm)).astype(np.float32)

    # Flash amplification:
    # - flash_level is a smoothed energy signal in [0, 1],
    # - center_weight emphasizes the core,
    # - FLASH_CENTER_EXP sharpens the effect at the core,
    # - FLASH_STRENGTH controls overall impact.
    center_boost = center_weight ** FLASH_CENTER_EXP
    flash_factor = 1.0 + FLASH_STRENGTH * flash_level * center_boost

    v = np.clip(base_v * flash_factor, 0.0, 1.0)

    # Vectorized HSV -> RGB conversion
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

    colors[:, 0] = r
    colors[:, 1] = g
    colors[:, 2] = b
    colors[:, 3] = 1.0

    return colors


# --------------------------------------------------------------------
# 4) CAMERA ANIMATION
# --------------------------------------------------------------------

def animate_camera(
    camera: gfx.PerspectiveCamera,
    angle: float,
    t_norm: float,
):
    """
    Camera inside the particle cloud, with:
      - orbital motion driven by 'angle' (audio-reactive rotation),
      - breathing depth (radius) over time,
      - a decentered look-at point that also moves inside the cloud.

    The result is:
      - the camera is immersed in the cloud (not outside),
      - it does NOT look at the origin, but at another point,
        which gives a more subjective / FPV feeling.
    """

    def smoothstep(x: float) -> float:
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    # Elevation: smooth increase then plateau over time
    h_alpha = smoothstep(min(1.0, t_norm * CAM_HEIGHT_SMOOTH_FACTOR))

    elev_deg = CAM_ELEV_START_DEG + (CAM_ELEV_END_DEG - CAM_ELEV_START_DEG) * h_alpha
    elev = math.radians(elev_deg)

    # ----------------------------------------------------------------
    # 1) Camera radial depth (inside the cloud)
    # ----------------------------------------------------------------
    # Camera radius oscillates between CAM_DEPTH_MIN and CAM_DEPTH_MAX,
    # centered around CAM_BASE_RADIUS, to "breathe" in and out.
    depth_osc = math.sin(2.0 * math.pi * CAM_DEPTH_CYCLES * t_norm)
    radius = CAM_BASE_RADIUS + CAM_DEPTH_AMPLITUDE * depth_osc
    radius = max(CAM_DEPTH_MIN, min(CAM_DEPTH_MAX, radius))

    cam_x = radius * math.cos(elev) * math.cos(angle)
    cam_y = radius * math.sin(elev)
    cam_z = radius * math.cos(elev) * math.sin(angle)

    # ----------------------------------------------------------------
    # 2) Decentered look-at point inside the cloud
    # ----------------------------------------------------------------
    # The target also moves, with a smaller radius and an angular offset.
    target_angle = angle + TARGET_PHASE
    target_elev = elev * TARGET_ELEV_SCALE

    target_depth_osc = math.sin(2.0 * math.pi * TARGET_DEPTH_FREQ * t_norm + TARGET_DEPTH_PHASE)
    target_radius = TARGET_BASE_RADIUS + TARGET_DEPTH_AMPLITUDE * target_depth_osc
    target_radius = max(TARGET_DEPTH_MIN, min(TARGET_DEPTH_MAX, target_radius))

    tgt_x = target_radius * math.cos(target_elev) * math.cos(target_angle)
    tgt_y = target_radius * math.sin(target_elev)
    tgt_z = target_radius * math.cos(target_elev) * math.sin(target_angle)

    # ----------------------------------------------------------------
    # 3) Apply camera transform
    # ----------------------------------------------------------------
    camera.local.position = (cam_x, cam_y, cam_z)
    camera.look_at((tgt_x, tgt_y, tgt_z))


# --------------------------------------------------------------------
# 5) RENDER LOOP
# --------------------------------------------------------------------

def render_particles_video(
    y_L: np.ndarray,
    y_R: np.ndarray,
    sr: int,
    duration: float,
    rms_norm: np.ndarray,
    out_path: str,
):
    """
    Render the dense particle cloud as a silent MP4.

    Parameters
    ----------
    y_L, y_R : np.ndarray
        Left and right audio channels (not directly used here but kept
        for future extensions).
    sr : int
        Sample rate of the audio.
    duration : float
        Duration of the video in seconds.
    rms_norm : np.ndarray
        Normalized RMS envelope of the mono mix.
    out_path : str
        Path to the silent MP4 file to be written.
    """
    canvas, renderer, scene, points, camera, material = create_scene_with_particles()

    positions, r_norm, center_weight = initialize_particles()

    # Set initial geometry once
    colors_init = compute_particle_colors(r_norm, center_weight, flash_level=0.0)
    points.geometry = gfx.Geometry(
        positions=positions.astype(np.float32),
        colors=colors_init.astype(np.float32),
    )

    total_frames = int(duration * FPS)
    total_frames = max(total_frames, 1)
    dt = duration / total_frames

    print(f"[render] FPS={FPS}, frames={total_frames}, video_duration={total_frames / FPS:.2f}s")

    writer = iio.get_writer(out_path, fps=FPS)

    # Camera state: angle for orbit, and smoothed energy/flash
    angle = 0.0
    rot_energy_state = 0.0
    flash_state = 0.0

    # Base angular speed (rad/s) for ~BASE_ORBITS over the clip
    base_ang_speed = 2.0 * math.pi * BASE_ORBITS / max(duration, 1e-3)

    try:
        for i in range(total_frames):
            t_norm = i / max(1, total_frames - 1)

            # 1) Audio energy at this time
            energy = get_energy_at_alpha(rms_norm, t_norm)

            # 2) Flash level: non-linear + temporal smoothing
            flash_target = energy ** FLASH_POWER
            flash_state = (1.0 - FLASH_SMOOTH) * flash_state + FLASH_SMOOTH * flash_target

            # 3) Rotation energy state: smoothed energy for speed control
            rot_energy_state = (1.0 - ROT_SMOOTH) * rot_energy_state + ROT_SMOOTH * energy
            speed_factor = 1.0 + ROT_GAIN * rot_energy_state  # always > 0

            # 4) Update camera angle (strictly increasing)
            angle += base_ang_speed * speed_factor * dt

            # 5) Update particle colors based on flash_state
            colors = compute_particle_colors(r_norm, center_weight, flash_level=flash_state)
            points.geometry = gfx.Geometry(
                positions=positions.astype(np.float32),
                colors=colors.astype(np.float32),
            )

            # 6) Update camera position and orientation
            animate_camera(camera, angle, t_norm)

            # 7) Offscreen render
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

    # 1) Render silent video
    render_particles_video(
        y_L,
        y_R,
        sr,
        duration,
        rms_norm,
        OUTPUT_VIDEO_SILENT,
    )

    # 2) Mux original audio
    mux_audio(OUTPUT_VIDEO_SILENT, str(audio_file), OUTPUT_VIDEO_WITH_AUDIO)


if __name__ == "__main__":
    print("[main] Starting dense audio-driven particle cloud...")
    main()
    print("[main] Done.")
