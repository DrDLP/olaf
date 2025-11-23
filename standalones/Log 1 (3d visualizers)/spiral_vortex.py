"""
Spiral Vortex music visualizer.

Hypnotic concentric rings that:
- rotate over time,
- slightly twist into a spiral,
- pulse according to the audio RMS loudness.

Technologies:
- NumPy for math and vector operations
- librosa for audio analysis (RMS envelope)
- skia-python for drawing
- OpenCV for bloom/glow post-processing
- MoviePy v2 for assembling frames into an MP4 video

Requirements:
    pip install moviepy librosa opencv-python skia-python numpy
"""

import os
from typing import Tuple
import colorsys
import numpy as np
import librosa
import skia
import cv2
from moviepy import VideoClip, AudioFileClip

# ------------------- CONFIG -------------------

W, H, FPS = 1280, 720, 30
BITRATE = "14M"

AUDIO_PATH = "song.wav"
OUTPUT_PATH = "spiral_vortex.mp4"

# Visual parameters
NUM_RINGS = 40           # Number of concentric rings
POINTS_PER_RING = 360    # Angular resolution per ring
BASE_RADIUS = min(W, H) * 0.08
RADIUS_SPACING = min(W, H) * 0.03
COLOR_ROT_SPEED = 0.25  # controls how fast the colors appear to rotate

GLOBAL_ROT_SPEED = 0.4   # Global rotation speed
SPIRAL_STRENGTH = 0.12   # Angular twist per ring
PULSE_AMOUNT = 0.35      # How much the audio modulates radius

# Color from center to border (BGR)
COLOR_INNER = np.array([0, 255, 200])    # cyan / teal
COLOR_OUTER = np.array([255, 0, 150])    # magenta / purple


# ------------------- AUDIO ANALYSIS -------------------

def load_audio_data(path: str) -> Tuple[np.ndarray, int, float, np.ndarray, np.ndarray]:
    """
    Load audio and compute a normalized RMS envelope.

    Parameters
    ----------
    path:
        Path to the audio file. If missing, silence is generated.

    Returns
    -------
    y : np.ndarray
        Audio samples (mono).
    sr : int
        Sample rate in Hz.
    duration : float
        Audio duration in seconds.
    times : np.ndarray
        Time axis for the RMS data (seconds).
    rms_smooth : np.ndarray
        Smoothed RMS values normalized to [0, 1].
    """
    if not os.path.isfile(path):
        print(f"Warning: audio file '{path}' not found. Using generated silence.")
        sr = 44100
        duration = 10.0
        y = np.zeros(int(sr * duration), dtype=np.float32)
    else:
        # librosa is commonly used for music/audio analysis :contentReference[oaicite:0]{index=0}
        y, sr = librosa.load(path, sr=None)

    hop_length = 512
    rms = librosa.feature.rms(
        y=y, frame_length=2048, hop_length=hop_length
    )[0]

    # Simple moving average smoothing
    rms_smooth = np.convolve(rms, np.ones(10) / 10.0, mode="same")

    # Normalize to [0, 1]
    max_val = rms_smooth.max()
    if max_val > 0:
        rms_smooth = rms_smooth / max_val

    times = librosa.frames_to_time(np.arange(len(rms_smooth)),
                                   sr=sr, hop_length=hop_length)
    duration = len(y) / sr
    return y, sr, duration, times, rms_smooth


def get_audio_loudness(t: float, times: np.ndarray, rms_data: np.ndarray) -> float:
    """
    Interpolate the RMS envelope at time t.

    Parameters
    ----------
    t:
        Time in seconds.
    times:
        Time stamps of the RMS samples.
    rms_data:
        RMS values corresponding to `times`.

    Returns
    -------
    loudness : float
        Interpolated value in [0, 1].
    """
    if len(times) == 0:
        return 0.0
    return float(np.interp(t, times, rms_data))


# ------------------- FRAME GENERATION -------------------

def make_spiral_vortex_frame(
    t: float,
    times: np.ndarray,
    rms_curve: np.ndarray,
) -> np.ndarray:
    """
    Generate one RGB frame of the spiral vortex at time t.

    Parameters
    ----------
    t:
        Time in seconds.
    times:
        RMS time axis.
    rms_curve:
        RMS amplitude curve (normalized).

    Returns
    -------
    frame_rgb : np.ndarray
        Frame as an (H, W, 3) RGB uint8 array.
    """
    surface = skia.Surface(W, H)
    canvas = surface.getCanvas()
    canvas.clear(skia.ColorBLACK)

    cx, cy = W / 2.0, H / 2.0

    energy = get_audio_loudness(t, times, rms_curve)
    # Add a small offset so visuals never fully freeze
    energy_boosted = energy * 1.3 + 0.1

    # Angles for all points on a ring
    angles = np.linspace(0.0, 2.0 * np.pi, POINTS_PER_RING, endpoint=True)

    for i in range(NUM_RINGS):
        # Normalized radius index [0, 1]
        r_norm = i / max(NUM_RINGS - 1, 1)

        # Base radius + pulsing
        base_r = BASE_RADIUS + i * RADIUS_SPACING
        pulse = 1.0 + PULSE_AMOUNT * energy_boosted * (1.0 - r_norm)
        radius = base_r * pulse

        # Rotation: inner rings rotate faster for a vortex feeling
        ring_rot = (GLOBAL_ROT_SPEED * t * (2.0 - r_norm)) + SPIRAL_STRENGTH * i

        # Angles for this ring
        theta = angles + ring_rot

        # Screen coordinates for all points on the ring
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)

        # Stroke width (similar to your original version)
        stroke_width = 3.0 * (1.5 - 0.8 * r_norm)

        # Phase for color rotation:
        #  - depends on time t
        #  - slightly on ring index (r_norm) so rings are out of phase
        color_phase = COLOR_ROT_SPEED * t + 0.7 * r_norm

        # Draw the ring as many small colored segments
        for k in range(len(x) - 1):
            # Angle of this segment (between -pi and pi)
            theta_seg = theta[k]

            # Map angle + time phase to hue in [0, 1]
            # theta_seg in [-pi, pi] → [0, 1]
            base_hue = (theta_seg + np.pi) / (2.0 * np.pi)
            hue = (base_hue + color_phase) % 1.0

            # Saturation and value depend slightly on energy and radius
            sat = 0.8 + 0.2 * energy_boosted
            val = 0.5 + 0.5 * energy_boosted * (1.0 - 0.5 * r_norm)

            # HSV → RGB in [0, 1]
            r_col, g_col, b_col = colorsys.hsv_to_rgb(hue, sat, val)

            # Convert to 0–255 and CLAMP
            r_int = int(np.clip(r_col * 255.0, 0, 255))
            g_int = int(np.clip(g_col * 255.0, 0, 255))
            b_int = int(np.clip(b_col * 255.0, 0, 255))

            color = skia.Color(
                r_int,
                g_int,
                b_int,
                215,  # alpha
            )

            paint = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style)
            paint.setStrokeWidth(stroke_width)
            paint.setColor(color)

            path = skia.Path()
            path.moveTo(float(x[k]), float(y[k]))
            path.lineTo(float(x[k + 1]), float(y[k + 1]))
            canvas.drawPath(path, paint)



    # Convert Skia surface to numpy array
    img_array = np.frombuffer(
        surface.makeImageSnapshot().tobytes(), dtype=np.uint8
    ).reshape((H, W, 4))  # RGBA

    # Convert RGBA -> BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

    # Bloom: blur + additively blend to create a glow
    glow = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=12, sigmaY=12)
    final_bgr = cv2.addWeighted(img_bgr, 1.0, glow, 0.9, 0.0)

    # Convert to RGB for MoviePy
    frame_rgb = final_bgr[:, :, ::-1]
    return frame_rgb


# ------------------- MOVIEPY PIPELINE -------------------

def main() -> None:
    """Load audio, build the VideoClip and render the MP4."""
    print(f"Loading audio from '{AUDIO_PATH}'...")
    _, _, duration, times, rms_curve = load_audio_data(AUDIO_PATH)

    def frame_fn(t: float) -> np.ndarray:
        return make_spiral_vortex_frame(t, times, rms_curve)

    print(f"Rendering video to '{OUTPUT_PATH}'...")
    clip = None
    audio_clip = None

    try:
        clip = VideoClip(frame_fn, duration=duration)

        if os.path.isfile(AUDIO_PATH):
            audio_clip = AudioFileClip(AUDIO_PATH)
            # MoviePy v2 uses with_audio() instead of set_audio() :contentReference[oaicite:1]{index=1}
            clip = clip.with_audio(audio_clip)

        clip.write_videofile(
            OUTPUT_PATH,
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
        if clip is not None:
            try:
                clip.close()
            except Exception:
                pass
        if audio_clip is not None:
            try:
                audio_clip.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
