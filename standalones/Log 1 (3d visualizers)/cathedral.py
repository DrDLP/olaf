"""
Audio-driven procedural "sound cathedral" / city (Python + librosa + pygfx).

Concept
-------
Given an input WAV file (song.wav), this script:

1. Loads the audio with librosa and computes a Mel spectrogram.
2. Aggregates frequency bands into "columns" of a 3D city:
   each column's height is controlled by the energy in a frequency band.
3. Uses pygfx box meshes to render a grid of buildings (columns, pillars, towers).
4. Animates the city over time: the architecture grows, shrinks, and morphs
   with the music, while a camera flies around it.
5. Renders an MP4 video via imageio/ffmpeg.

Visual mapping
--------------
- Each frame of the video corresponds to a moment in the audio.
- Low frequencies → taller central "nave" / foundations.
- Higher frequencies → more variation in peripheral towers.
- A radial weighting makes the center of the grid taller, evoking a cathedral
  or dense city center rather than a flat histogram.

Dependencies
------------
pip install librosa pygfx wgpu pylinalg imageio[ffmpeg]

Tested with:
- Python 3.10
- librosa >= 0.11
- pygfx >= 0.15, wgpu >= 0.16
- imageio >= 2.24 with ffmpeg backend
"""

import math
from pathlib import Path
import colorsys

import numpy as np
import librosa
from imageio import v2 as iio

import pygfx as gfx
import pylinalg as la  # currently unused, kept for future camera/transform work
import wgpu.gui.offscreen as wgpu_offscreen
from pygfx.renderers.wgpu import DDAAPass


# --------------------------------------------------------------------
# Global configuration / artist controls
# --------------------------------------------------------------------

# ---- I/O ---------------------------------------------------------------------

# Input audio file (mono or stereo; mono is recommended for predictable behavior).
AUDIO_PATH = "song.wav"

# Output video file (silent; audio will be muxed to a second file).
OUTPUT_VIDEO = "output_cathedral.mp4"
OUTPUT_VIDEO_WITH_AUDIO = "output_cathedral_with_audio.mp4"


# ---- Video settings ----------------------------------------------------------

FPS = 24                 # frames per second for the output video
OUT_WIDTH = 1280         # video width in pixels
OUT_HEIGHT = 720         # video height in pixels

# Limit duration for testing; set to None for full track
MAX_DURATION_SEC = None  # e.g. 20.0 for 20 seconds, None for full song


# ---- Audio analysis settings -------------------------------------------------

N_MELS = 128             # number of Mel bands for the spectrogram
N_FFT = 2048             # FFT window size
HOP_LENGTH = 512         # hop size between spectrogram frames


# ---- City / cathedral layout -------------------------------------------------

GRID_X = 16              # number of columns along X (street direction)
GRID_Z = 16              # number of columns along Z (depth)
BUILDING_SPACING = 1.3   # world-space distance between building centers

# Radial weight for building heights (center vs. edge)
RADIAL_WEIGHT_MAX = 1.6  # maximum multiplier at the very center
RADIAL_WEIGHT_MIN = 0.3  # minimum multiplier at the far corners

# Additional boost on cross shape (nave + transept)
CROSS_WEIGHT_FACTOR = 1.4  # multiplier for central X/Z axis lines


# ---- Building height / animation --------------------------------------------

# Fraction of the video over which the city “builds up” from nothing.
BUILDUP_FRACTION = 0.10   # first 10% of the clip is a growth phase

# Height formula: h = HEIGHT_BASE + HEIGHT_SCALE * (energy ** HEIGHT_EXPONENT) * weight * build_up
HEIGHT_BASE = 0.45         # minimal height offset (even when energy is low)
HEIGHT_SCALE = 16.0       # global scale for building height amplitude
HEIGHT_EXPONENT = 1.4     # nonlinearity on band energy ( >1 for more contrast)

# Mirrored city underneath: relative height vs. top buildings
BOTTOM_HEIGHT_RATIO = 0.85


# ---- Neon color style for columns -------------------------------------------

# Radial “hotspot” neon:
# - near the center: bright hot colors (red/yellow/orange),
# - towards the edges: cooler / darker (magenta/blue).
NEON_HUE_CENTER = 0    # base hue near the center (wrapped mod 1.0)
NEON_HUE_RANGE = 0.6     # how far hue moves from center to edge

NEON_LIGHTNESS_CENTER = 0.35  # lightness near the center (0..1)
NEON_LIGHTNESS_EDGE = 0.55    # lightness near the edges (0..1)
NEON_LIGHTNESS_MIN = 0.25     # clamp minimum lightness
NEON_LIGHTNESS_MAX = 0.80     # clamp maximum lightness

NEON_COLOR_BOOST = 1.5        # overall multiplier for “neon” punch

# Top material emissive / shininess
TOP_EMISSIVE_FACTOR = 0.8     # emissive = color * this factor
TOP_SHININESS = 80.0          # Phong shininess for top columns

# Bottom (mirrored) material tweaks
BOTTOM_COLOR_ATTENUATION = (0.8, 0.8, 0.9)  # RGB multipliers vs. top neon color
BOTTOM_EMISSIVE_FACTOR = 0.4                # relative to top emissive
BOTTOM_SHININESS = 50.0


# ---- Background & lighting ---------------------------------------------------

# Background vertical gradient (top to bottom).
BACKGROUND_COLOR_TOP = "#050514"
BACKGROUND_COLOR_BOTTOM = "#000000"

# Ambient light intensity (soft global fill).
AMBIENT_INTENSITY = 0.6

# Directional “sun” light position in world space.
DIR_LIGHT_POSITION = (20.0, 80.0, 20.0)


# ---- Camera motion -----------------------------------------------------------

# Camera field of view (degrees).
CAM_FOV_DEG = 50.0

# Number of full rotations the camera makes over the entire clip.
CAM_BASE_ORBITS = 20.0

# Camera elevation (pitch) control: base + sinusoidal variation over time.
CAM_ELEV_BASE_DEG = 40.0   # average elevation angle above horizon
CAM_ELEV_AMP_DEG = 32.0    # amplitude of elevation oscillation (± degrees)
CAM_ELEV_FREQ = 0.25       # number of up/down oscillations over normalized [0,1]

# Radial distance from the city center in units of “city radius”.
# city_radius ~ max(GRID_X, GRID_Z) * BUILDING_SPACING * 0.8
CAM_RADIUS_FACTOR = 1.8


# --------------------------------------------------------------------
# 1) AUDIO ANALYSIS: Mel spectrogram
# --------------------------------------------------------------------

def analyze_audio(path: str):
    """
    Load the audio file and compute a normalized Mel spectrogram.

    Parameters
    ----------
    path : str
        Path to the audio file.

    Returns
    -------
    S_norm : np.ndarray, shape (n_mels, time_bins), float32
        Mel spectrogram in dB, normalized to [0, 1].
    duration : float
        Duration of the audio in seconds.
    """
    print(f"[audio] Loading: {path}")
    y, sr = librosa.load(path, sr=None, mono=True)
    duration = len(y) / sr
    print(f"[audio] sr={sr}, duration={duration:.2f}s")

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_min, S_max = S_db.min(), S_db.max()
    S_norm = (S_db - S_min) / (S_max - S_min + 1e-9)
    S_norm = S_norm.astype(np.float32)

    n_mels, t_bins = S_norm.shape
    print(f"[audio] mel-spectrogram: n_mels={n_mels}, time_bins={t_bins}")

    return S_norm, duration


def compute_band_energies(S_norm: np.ndarray, t_idx: int, n_bands: int):
    """
    Aggregate Mel spectrogram bins into n_bands frequency bands
    at a given time index.

    Parameters
    ----------
    S_norm : np.ndarray, shape (n_mels, time_bins)
        Normalized Mel spectrogram.
    t_idx : int
        Time index (clamped to valid range).
    n_bands : int
        Number of bands to aggregate to.

    Returns
    -------
    band_energies : np.ndarray, shape (n_bands,)
        Average energy per band in [0, 1].
    """
    n_mels, t_bins = S_norm.shape
    t_idx = max(0, min(t_idx, t_bins - 1))

    band_edges = np.linspace(0, n_mels, n_bands + 1, dtype=int)
    band_energies = np.zeros(n_bands, dtype=np.float32)

    for b in range(n_bands):
        start = band_edges[b]
        end = band_edges[b + 1]
        if end <= start:
            band_energies[b] = 0.0
        else:
            band_energies[b] = float(np.mean(S_norm[start:end, t_idx]))

    return band_energies


# --------------------------------------------------------------------
# 2) CITY / CATHEDRAL LAYOUT
# --------------------------------------------------------------------

def build_city_layout(grid_x: int, grid_z: int, spacing: float):
    """
    Build a regular grid of "building slots" and a radial weighting,
    so the center tends to be taller (cathedral-like).

    Parameters
    ----------
    grid_x : int
        Number of buildings along X.
    grid_z : int
        Number of buildings along Z.
    spacing : float
        Distance between building centers.

    Returns
    -------
    positions_xz : np.ndarray, shape (N, 2)
        (x, z) ground positions for each building instance.
    radial_weights : np.ndarray, shape (N,)
        Weight in [RADIAL_WEIGHT_MIN, ~RADIAL_WEIGHT_MAX * CROSS_WEIGHT_FACTOR]
        that modulates building height based on distance from the center and a
        cross-shaped pattern (nave + transept).
    """
    N = grid_x * grid_z
    positions_xz = np.zeros((N, 2), dtype=np.float32)
    radial_weights = np.zeros((N,), dtype=np.float32)

    cx = (grid_x - 1) / 2.0
    cz = (grid_z - 1) / 2.0

    idx = 0
    for iz in range(grid_z):
        for ix in range(grid_x):
            # World-space position on the ground plane
            px = (ix - cx) * spacing
            pz = (iz - cz) * spacing
            positions_xz[idx, 0] = px
            positions_xz[idx, 1] = pz

            # Normalized radial distance from center in index space
            rx = (ix - cx) / max(cx, 1e-6)
            rz = (iz - cz) / max(cz, 1e-6)
            r = math.sqrt(rx * rx + rz * rz)  # 0 at center, ~1 at corners

            # Base radial weight: tall at center, shorter at edges
            w = max(RADIAL_WEIGHT_MIN, RADIAL_WEIGHT_MAX - r)

            # Emphasize cross shape (nave + transept)
            if abs(ix - cx) < 1.0 or abs(iz - cz) < 1.0:
                w *= CROSS_WEIGHT_FACTOR

            radial_weights[idx] = w
            idx += 1

    return positions_xz, radial_weights


# --------------------------------------------------------------------
# 3) PYGFX SCENE: city / cathedral
# --------------------------------------------------------------------

def create_scene_with_city(positions_xz: np.ndarray):
    """
    Create the pygfx scene containing:

      - A set of building meshes on a grid (top side).
      - A mirrored set of buildings below y=0 (bottom side).
      - Ambient + directional light.
      - A perspective camera.

    Each building gets its own MeshPhongMaterial (neon colors based on
    radial distance), and we create a mirrored "reflection" with a
    slightly dimmer, cooler emissive color.

    Parameters
    ----------
    positions_xz : np.ndarray, shape (N, 2)
        Ground positions for each building.

    Returns
    -------
    canvas : wgpu.gui.offscreen.WgpuCanvas
    renderer : gfx.WgpuRenderer
    scene : gfx.Scene
    buildings_top : list[gfx.Mesh]
        List of building meshes above y=0.
    buildings_bottom : list[gfx.Mesh]
        List of mirrored building meshes below y=0.
    camera : gfx.PerspectiveCamera
    """
    print("[gfx] Initializing city scene")

    canvas = wgpu_offscreen.WgpuCanvas(size=(OUT_WIDTH, OUT_HEIGHT), pixel_ratio=1)
    renderer = gfx.renderers.WgpuRenderer(canvas)
    scene = gfx.Scene()

    # Background vertical gradient
    background = gfx.Background.from_color(BACKGROUND_COLOR_TOP, BACKGROUND_COLOR_BOTTOM)
    scene.add(background)

    n_instances = positions_xz.shape[0]
    geometry = gfx.box_geometry(1.0, 1.0, 1.0)  # base unit cube

    buildings_top: list[gfx.Mesh] = []
    buildings_bottom: list[gfx.Mesh] = []

    # Precompute center indices for radial coloring
    cx = (GRID_X - 1) / 2.0
    cz = (GRID_Z - 1) / 2.0

    for idx in range(n_instances):
        px, pz = positions_xz[idx]
        iz = idx // GRID_X
        ix = idx % GRID_X

        # ------------------------------
        # Neon "hotspot" color by radius
        # ------------------------------
        # Distance normalized to [0, 1] from grid center
        dx = (ix - cx) / max(cx, 1e-6)
        dz = (iz - cz) / max(cz, 1e-6)
        r_norm = math.sqrt(dx * dx + dz * dz)
        r_norm = min(r_norm, 1.0)

        # "Thermal" mapping in HLS:
        # - center (r_norm ~ 0): red / yellow (hot, bright),
        # - mid radius: orange / magenta,
        # - edges (r_norm ~ 1): violet / blue (cooler, darker).
        h = (NEON_HUE_CENTER + NEON_HUE_RANGE * r_norm) % 1.0

        s = 1.0  # max saturation for neon look

        # Lightness decreases from center to edge, clamped to a reasonable range
        l = NEON_LIGHTNESS_CENTER - (NEON_LIGHTNESS_CENTER - NEON_LIGHTNESS_EDGE) * r_norm
        l = max(NEON_LIGHTNESS_MIN, min(NEON_LIGHTNESS_MAX, l))

        # HLS -> RGB
        r, g, b = colorsys.hls_to_rgb(h, l, s)

        # Slight neon boost and clamp
        neon_rgb = np.clip(np.array([r, g, b]) * NEON_COLOR_BOOST, 0.0, 1.0)

        color_rgba = (neon_rgb[0], neon_rgb[1], neon_rgb[2], 1.0)
        emissive_rgba = (neon_rgb[0] * TOP_EMISSIVE_FACTOR,
                         neon_rgb[1] * TOP_EMISSIVE_FACTOR,
                         neon_rgb[2] * TOP_EMISSIVE_FACTOR,
                         1.0)

        # Top material: bright neon
        material_top = gfx.MeshPhongMaterial(
            color=color_rgba,
            emissive=emissive_rgba,
            shininess=TOP_SHININESS,
        )

        # Bottom material: dimmer + slightly "colder" reflection
        bottom_color = (
            neon_rgb[0] * BOTTOM_COLOR_ATTENUATION[0],
            neon_rgb[1] * BOTTOM_COLOR_ATTENUATION[1],
            neon_rgb[2] * BOTTOM_COLOR_ATTENUATION[2],
            1.0,
        )
        material_bottom = gfx.MeshPhongMaterial(
            color=bottom_color,
            emissive=(emissive_rgba[0] * BOTTOM_EMISSIVE_FACTOR,
                      emissive_rgba[1] * BOTTOM_EMISSIVE_FACTOR,
                      emissive_rgba[2] * BOTTOM_EMISSIVE_FACTOR,
                      1.0),
            shininess=BOTTOM_SHININESS,
        )

        mesh_top = gfx.Mesh(geometry, material_top)
        mesh_bottom = gfx.Mesh(geometry, material_bottom)

        # Initial very small height
        h0 = 0.1
        mesh_top.local.scale = (1.0, h0, 1.0)
        mesh_top.local.position = (px, h0 / 2.0, pz)

        mesh_bottom.local.scale = (1.0, h0, 1.0)
        mesh_bottom.local.position = (px, -h0 / 2.0, pz)

        scene.add(mesh_top)
        scene.add(mesh_bottom)

        buildings_top.append(mesh_top)
        buildings_bottom.append(mesh_bottom)

    # Lighting
    scene.add(gfx.AmbientLight(intensity=AMBIENT_INTENSITY))

    sun = gfx.DirectionalLight()
    sun.local.position = DIR_LIGHT_POSITION
    scene.add(sun)

    # Camera (initial position; will be animated later)
    aspect = OUT_WIDTH / OUT_HEIGHT
    camera = gfx.PerspectiveCamera(CAM_FOV_DEG, aspect)
    camera.local.position = (0.0, 40.0, 60.0)
    camera.look_at((0.0, 0.0, 0.0))

    # Simple anti-aliasing pass
    renderer.effect_passes = [DDAAPass()]

    canvas.request_draw(lambda: renderer.render(scene, camera))

    return canvas, renderer, scene, buildings_top, buildings_bottom, camera


# --------------------------------------------------------------------
# 4) PER-FRAME UPDATES: city + camera
# --------------------------------------------------------------------

def update_city_for_frame(
    buildings_top: list[gfx.Mesh],
    buildings_bottom: list[gfx.Mesh],
    positions_xz: np.ndarray,
    radial_weights: np.ndarray,
    S_norm: np.ndarray,
    frame_idx: int,
    total_frames: int,
    duration: float,
):
    """
    Update all building meshes for a given frame.

    Mapping:
      - Each column along X is associated with a frequency band.
      - band_energies[b] in [0, 1] controls the base height of that column.
      - A radial weight makes center columns taller (cathedral style).
      - A global build-up factor slowly ramps up the city at the beginning.

    Both:
      - buildings_top  (above y=0), and
      - buildings_bottom (mirrored below y=0)
    are updated.

    Returns
    -------
    global_energy : float
        Global mean band energy at this frame (for potential camera/audio coupling).
    """
    n_instances = positions_xz.shape[0]
    time_bins = S_norm.shape[1]

    alpha = frame_idx / max(1, total_frames - 1)
    t_idx = int(alpha * (time_bins - 1))

    band_energies = compute_band_energies(S_norm, t_idx, GRID_X)

    # City "builds up" over the initial fraction of the video
    if BUILDUP_FRACTION <= 0.0:
        build_up = 1.0
    else:
        build_up = min(1.0, alpha / BUILDUP_FRACTION)

    # Global energy in [0,1] (mean over bands)
    global_energy = float(np.mean(band_energies))

    for idx in range(n_instances):
        px, pz = positions_xz[idx]
        w = radial_weights[idx]

        iz = idx // GRID_X
        ix = idx % GRID_X

        base_energy = band_energies[ix]

        # Height per column: non-linear mapping of energy with radial weighting
        h = HEIGHT_BASE + HEIGHT_SCALE * (base_energy ** HEIGHT_EXPONENT) * w * build_up

        # Top building
        mesh_top = buildings_top[idx]
        mesh_top.local.scale = (1.0, h, 1.0)
        mesh_top.local.position = (px, h / 2.0, pz)

        # Mirrored building below y=0 (slightly shorter)
        h_bottom = h * BOTTOM_HEIGHT_RATIO
        mesh_bottom = buildings_bottom[idx]
        mesh_bottom.local.scale = (1.0, h_bottom, 1.0)
        mesh_bottom.local.position = (px, -h_bottom / 2.0, pz)

    return global_energy


def animate_camera(
    camera: gfx.PerspectiveCamera,
    frame_idx: int,
    total_frames: int,
    energy: float,
):
    """
    Smooth orbital camera around the mirrored neon city.

    - Rotation speed is constant (currently independent of audio).
    - Camera stays at a fixed radius from the city center.
    - Height (elevation) varies smoothly over time.

    Parameters
    ----------
    camera : gfx.PerspectiveCamera
        Camera to update.
    frame_idx : int
        Current frame index.
    total_frames : int
        Total number of frames in the animation.
    energy : float
        Global energy for this frame (0..1). Currently unused, but kept
        for future extensions (e.g. subtle speed modulation).
    """
    if total_frames <= 1:
        alpha = 0.0
    else:
        alpha = frame_idx / (total_frames - 1)

    # 1) Azimuth: simple linear rotation over time
    angle = 2.0 * math.pi * CAM_BASE_ORBITS * alpha

    # 2) Elevation: smooth sinusoidal variation over time
    elev_base = math.radians(CAM_ELEV_BASE_DEG)
    elev_amp = math.radians(CAM_ELEV_AMP_DEG)
    elev = elev_base + elev_amp * math.sin(2.0 * math.pi * CAM_ELEV_FREQ * alpha)

    # 3) Radius: constant distance proportional to the city footprint
    city_radius = max(GRID_X, GRID_Z) * BUILDING_SPACING * 0.8
    radius = city_radius * CAM_RADIUS_FACTOR

    # Spherical to Cartesian
    x = radius * math.cos(elev) * math.cos(angle)
    y = radius * math.sin(elev)
    z = radius * math.cos(elev) * math.sin(angle)

    camera.local.position = (x, y, z)
    camera.look_at((0.0, 0.0, 0.0))


# --------------------------------------------------------------------
# 5) RENDER LOOP: offscreen -> MP4
# --------------------------------------------------------------------

def render_city_video(S_norm: np.ndarray, duration: float, out_path: str):
    """
    Render the procedural city/cathedral into an MP4 file.

    Parameters
    ----------
    S_norm : np.ndarray, shape (n_mels, time_bins)
        Normalized Mel spectrogram used to drive building heights.
    duration : float
        Duration of the clip in seconds.
    out_path : str
        Output video filename (e.g. "output_cathedral.mp4").
    """
    positions_xz, radial_weights = build_city_layout(GRID_X, GRID_Z, BUILDING_SPACING)
    canvas, renderer, scene, buildings_top, buildings_bottom, camera = create_scene_with_city(positions_xz)

    total_frames = int(duration * FPS)
    total_frames = max(total_frames, 1)
    print(f"[render] FPS={FPS}, frames={total_frames}, video_duration={total_frames / FPS:.2f}s")

    writer = iio.get_writer(out_path, fps=FPS)

    try:
        for i in range(total_frames):
            # Update city geometry for this frame and get global energy
            energy = update_city_for_frame(
                buildings_top,
                buildings_bottom,
                positions_xz,
                radial_weights,
                S_norm,
                i,
                total_frames,
                duration,
            )

            # Camera motion (currently not using energy, but energy is
            # passed through for possible future use).
            animate_camera(camera, i, total_frames, energy)

            # Offscreen render: canvas.draw() -> RGBA ndarray
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
        print(f"[render] Video written to: {out_path}")


# --------------------------------------------------------------------
# 6) AUDIO MUXING (ffmpeg via imageio-ffmpeg)
# --------------------------------------------------------------------

import subprocess
import imageio_ffmpeg


def mux_audio(video_path: str, audio_path: str, output_path: str):
    """
    Use ffmpeg to mux the original audio into the rendered video.

    video_path : path to the silent MP4 rendered by this script.
    audio_path : path to the input audio file (song.wav).
    output_path : final MP4 with both video and audio.

    Command:
        ffmpeg -i video -i audio -c:v copy -c:a aac -b:a 192k -shortest out

    - Video is not re-encoded (copy),
    - Audio is encoded to AAC,
    - Output duration is the shortest of (video, audio).
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
    audio_file = Path(AUDIO_PATH)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    S_norm, duration = analyze_audio(str(audio_file))

    # Clamp duration if a preview limit is set
    if MAX_DURATION_SEC is not None:
        duration = min(duration, MAX_DURATION_SEC)

    # 1) Render silent video driven by the spectrogram
    render_city_video(S_norm, duration, OUTPUT_VIDEO)

    # 2) Mux original audio into the rendered video
    mux_audio(OUTPUT_VIDEO, str(audio_file), OUTPUT_VIDEO_WITH_AUDIO)


if __name__ == "__main__":
    main()
