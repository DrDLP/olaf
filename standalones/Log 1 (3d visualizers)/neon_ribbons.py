"""
Neon ribbons music visualizer.

This script renders an audio-reactive "neon ribbons" tunnel using:
- librosa for audio analysis (RMS envelope),
- skia-python for vector drawing,
- OpenCV (cv2) for bloom / glow post-processing,
- MoviePy v2 for assembling the frames into an MP4 video.

Requirements (Python 3.9+ recommended):
    pip install moviepy librosa opencv-python skia-python numpy

MoviePy v2 note:
    - Import from `moviepy`, not `moviepy.editor`.
    - Use `.with_audio(...)` instead of `.set_audio(...)`.
"""

import os
from typing import Tuple

import numpy as np
import librosa
import skia
import cv2
from moviepy import VideoClip, AudioFileClip

# ------------------- CONFIGURATION -------------------

# Render resolution and frame rate
W, H, FPS = 1280, 720, 30
BITRATE = "14M"

# Audio input and video output
FULL_WAV = "song.wav"  # Make sure this file exists
OUT_MP4 = "neon_ribbons.mp4"

# Visual parameters
NUM_LINES = 200        # Number of "ribbons" in depth
POINTS_PER_LINE = 100  # Horizontal resolution of each ribbon polyline
GRID_DEPTH = 30.0      # Total depth of the 3D scene
SPEED = 4.0            # Wave scrolling speed

# Colors (BGR for OpenCV / Skia RGBA mapping)
# We interpolate from far (background) to near (foreground)
COLOR_FAR = np.array([255, 0, 128])   # Deep pink / violet (BGR)
COLOR_NEAR = np.array([0, 255, 255])  # Cyan / electric yellow (BGR)


# ------------------- AUDIO ANALYSIS -------------------

def load_audio_data(path: str) -> Tuple[np.ndarray, int, float, np.ndarray, np.ndarray]:
    """
    Load audio and precompute a smoothed RMS envelope.

    Parameters
    ----------
    path:
        Path to the WAV (or any format supported by librosa).

    Returns
    -------
    y : np.ndarray
        Raw audio samples.
    sr : int
        Sample rate (Hz).
    duration : float
        Audio duration in seconds.
    times : np.ndarray
        Time axis for the RMS envelope (seconds).
    rms_smooth : np.ndarray
        Normalized, smoothed RMS values in [0, 1].
    """
    if not os.path.isfile(path):
        # Fallback: generate a short silent buffer to test the pipeline
        print(f"Warning: audio file not found at '{path}'. Using generated silence.")
        sr = 44100
        duration = 5.0
        y = np.zeros(int(sr * duration), dtype=np.float32)
    else:
        y, sr = librosa.load(path, sr=None)

    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]

    # Simple temporal smoothing of the RMS to avoid jitter
    rms = np.interp(np.arange(len(rms)), np.arange(len(rms)), rms)
    rms_smooth = np.convolve(rms, np.ones(10) / 10, mode="same")

    # Normalize so the maximum value is 1.0
    if rms_smooth.max() > 0:
        rms_smooth = rms_smooth / rms_smooth.max()

    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    duration = len(y) / sr

    return y, sr, duration, times, rms_smooth


def get_audio_loudness(t: float, times: np.ndarray, rms_data: np.ndarray) -> float:
    """
    Get the interpolated RMS loudness at time t.

    Parameters
    ----------
    t:
        Time in seconds.
    times:
        Time axis of the RMS data.
    rms_data:
        RMS envelopes sampled at `times`.

    Returns
    -------
    Interpolated loudness (float).
    """
    return float(np.interp(t, times, rms_data))


# ------------------- 3D PROJECTION -------------------

def project_3d_point(
    x: float,
    y: float,
    z: float,
    width: int,
    height: int,
) -> Tuple[float, float, float]:
    """
    Simple 3D perspective projection.

    Parameters
    ----------
    x, y:
        World coordinates (roughly in [-1, 1]).
    z:
        Depth coordinate (must be > 0, e.g. 1.0 to 10.0).
    width, height:
        Target viewport size in pixels.

    Returns
    -------
    x_proj, y_proj:
        Projected 2D coordinates in pixel space.
    scale:
        Perspective scale factor derived from depth.
    """
    fov = 300.0  # Field of view
    if z <= 0.1:
        z = 0.1  # Avoid division by zero

    scale = fov / z

    x_proj = (x * scale) + (width / 2.0)
    y_proj = (y * scale) + (height / 2.0)

    return x_proj, y_proj, scale


# ------------------- RENDERING -------------------

def make_neon_ribbons_frame(
    t: float,
    times: np.ndarray,
    rms_curve: np.ndarray,
) -> np.ndarray:
    """
    Generate a single RGB frame of the neon ribbons effect at time t.

    Parameters
    ----------
    t:
        Time in seconds.
    times:
        Time axis for the RMS envelope.
    rms_curve:
        RMS envelope values (normalized in [0, 1]).

    Returns
    -------
    frame_rgb: np.ndarray
        RGB frame as a uint8 HxWx3 array.
    """
    # 1. Create a Skia surface and clear to black
    surface = skia.Surface(W, H)
    canvas = surface.getCanvas()
    canvas.clear(skia.ColorBLACK)

    # Current music "energy" (RMS) with a small offset so the scene never fully stops
    energy = get_audio_loudness(t, times, rms_curve) * 1.5 + 0.1

    # 2. Generate the depth-stacked ribbon lines
    # Painter's algorithm: draw from far (high z) to near (low z)
    for i in range(NUM_LINES, 0, -1):
        # Normalized depth (0 = near, 1 = far)
        z_norm = i / NUM_LINES

        # World space depth; offset so we are never at z=0
        z_world = 1.0 + (z_norm * GRID_DEPTH)

        # Linear interpolation of color between near and far
        col_mix = COLOR_NEAR * (1.0 - z_norm) + COLOR_FAR * z_norm

        # Alpha: more distant ribbons are more transparent
        alpha = int(255 * (1.0 - (z_norm * 0.6)))
        color = skia.Color(
            int(col_mix[2]),  # R
            int(col_mix[1]),  # G
            int(col_mix[0]),  # B
            alpha,
        )

        path = skia.Path()
        first_point = True

        # Generate points along the ribbon in X
        for j in range(POINTS_PER_LINE):
            # Normalized horizontal coordinate: [-0.5, 0.5]
            x_norm = (j / (POINTS_PER_LINE - 1)) - 0.5
            x_world = x_norm * (GRID_DEPTH * 1.5)  # Wider scene at larger depth

            # --- Wave shaping ---
            # Main sinus wave that travels forward
            offset_x = x_world * 2.0
            offset_z = z_world * 0.5
            phase = t * SPEED

            y_base = np.sin(offset_x + phase + offset_z)
            y_detail = np.sin(offset_x * 3.0 - phase) * 0.3

            # Edge fade so the tunnel "closes" toward the sides
            fade_edges = 1.0 - (2.0 * abs(x_norm)) ** 2

            amplitude = energy * 1.2 * fade_edges
            y_world = (y_base + y_detail) * amplitude * 0.8

            # Global vertical offset to center the scene in view
            y_world += 0.5

            # Project the 3D point to screen space
            px, py, scale = project_3d_point(x_world, y_world, z_world, W, H)

            if first_point:
                path.moveTo(px, py)
                first_point = False
            else:
                # With a high horizontal resolution, lineTo is smooth enough
                path.lineTo(px, py)

        # Draw the ribbon as a stroked path
        paint = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style)
        # Lines closer to the camera are slightly thicker
        paint.setStrokeWidth(4.0 * (2.0 / z_world))
        paint.setColor(color)

        canvas.drawPath(path, paint)

    # 3. Post-process glow using OpenCV
    # Convert Skia surface to RGBA bytes -> numpy array
    img_array = np.frombuffer(surface.makeImageSnapshot().tobytes(), dtype=np.uint8)
    img_array = img_array.reshape((H, W, 4))  # RGBA

    # Convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

    # Gaussian blur for bloom / glow
    glow = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=15, sigmaY=15)

    # Additive blend: sharp image + scaled glow
    final_bgr = cv2.addWeighted(img_bgr, 1.0, glow, 0.8, 0.0)

    # Return RGB for MoviePy (flip channel order)
    return final_bgr[:, :, ::-1]


# ------------------- MOVIEPY PIPELINE -------------------

def main() -> None:
    """Entry point: build and render the neon ribbons video."""
    print(f"Loading audio from '{FULL_WAV}'...")
    _, _, duration, times, rms_curve = load_audio_data(FULL_WAV)

    # Wrap our frame generator so MoviePy gets a single-argument function f(t)
    def frame_fn(t: float) -> np.ndarray:
        return make_neon_ribbons_frame(t, times, rms_curve)

    print(f"Rendering video to '{OUT_MP4}'...")
    try:
        # Create the dynamic video clip
        clip = VideoClip(frame_fn, duration=duration)

        # Attach audio if the file actually exists
        audio_clip = None
        if os.path.isfile(FULL_WAV):
            audio_clip = AudioFileClip(FULL_WAV)
            # MoviePy v2 style: use with_audio() to attach the track
            clip = clip.with_audio(audio_clip)

        # Export to MP4
        clip.write_videofile(
            OUT_MP4,
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            bitrate=BITRATE,
            threads=4,
        )
        print("Render completed successfully.")
    except Exception as exc:
        print(f"Error during render: {exc}")
    finally:
        # Clean up resources
        try:
            clip.close()
        except Exception:
            pass
        if "audio_clip" in locals() and audio_clip is not None:
            try:
                audio_clip.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
