#!/usr/bin/env python3
"""
Automatic audio clean-up script for "boxy / underwater / muffled" vocal or music recordings.

This version is especially suited for already-mastered material (e.g. AI-generated
songs from Suno): it will NOT increase the overall loudness above the original track.
It can only keep it similar or slightly lower.

Processing chain:
1. Optional light noise reduction (noisereduce, spectral gating), only if the file
   is not already extremely hot in level.
2. Optional peak attenuation at input (only if the file is already very hot).
3. Pedalboard effects (tonal shaping + very gentle dynamics):
   - High-pass filter at 80 Hz (remove rumble / unnecessary low-end)
   - Two low-mid cuts (~300 Hz and ~450 Hz) to reduce "boxiness" / "mud"
   - Simple de-esser: narrow cut around 7 kHz to tame harsh "s" sounds
   - Presence boost around 3.5 kHz (clarity)
   - High-shelf boost from 6 kHz (air / openness)
   - Gentle compressor (low ratio, high threshold, mainly catches peaks)
   - Very soft noise gate / expander (lightly reduces ambience between phrases)
   (No limiter here: we avoid adding extra brickwall limiting.)
4. Loudness matching:
   - Measure original loudness (after noise reduction)
   - Measure processed loudness
   - If processed is louder, bring it DOWN to be ~3 dB below the original
   - If processed is quieter, leave it as-is (NO upward boost)
5. Final peak safety: if peaks exceed ~0 dBFS (|sample| > 0.995), they are attenuated;
   otherwise they are untouched.

Requirements (Python packages):
    - numpy
    - pedalboard
    - noisereduce
    - pyloudnorm

You can install them with:

    pip install numpy pedalboard noisereduce pyloudnorm

Usage:
    python auto_clean.py input.wav output.wav
    python auto_clean.py input.wav        # -> input_cleaned.wav

Notes:
- This is a "preset" style processor.
- Designed to keep Suno-like mastered tracks at a safe loudness,
  mainly improving tone / clarity and slightly *reducing* loudness
  to alleviate the "flattened / saturated" feel.
"""

import argparse
import numpy as np
import noisereduce as nr
import pyloudnorm as pyln

from pedalboard import (
    Pedalboard,
    HighpassFilter,
    PeakFilter,
    HighShelfFilter,
    Compressor,
    NoiseGate,
)
from pedalboard.io import AudioFile


def build_board() -> Pedalboard:
    """
    Build the effect chain ("board") used for tonal shaping and very gentle dynamics.

    Parameters are chosen to be safe for already-processed mixes (e.g. Suno songs),
    focusing on clarity rather than heavy compression or limiting.
    """
    board = Pedalboard([
        # 1) Remove unnecessary low-end / rumble
        HighpassFilter(
            cutoff_frequency_hz=80.0
        ),

        # 2) Low-mid cuts to reduce "boxy" / "muddy" sound
        PeakFilter(
            cutoff_frequency_hz=300.0,
            gain_db=-3.0,
            q=1.0,  # moderately narrow
        ),
        PeakFilter(
            cutoff_frequency_hz=450.0,
            gain_db=-2.0,
            q=1.0,
        ),

        # 3) Simple de-esser: narrow dip around 7 kHz
        #    (helps tame harsh "s", "sh" sounds after adding air)
        PeakFilter(
            cutoff_frequency_hz=7000.0,
            gain_db=-2.0,  # gentle
            q=3.0,         # relatively narrow notch
        ),

        # 4) Presence boost to make the voice/instrument more intelligible
        PeakFilter(
            cutoff_frequency_hz=3500.0,
            gain_db=2.0,
            q=0.7,  # wider band
        ),

        # 5) Air / openness (high-shelf from ~6 kHz)
        HighShelfFilter(
            cutoff_frequency_hz=6000.0,
            gain_db=2.0,
            q=0.7,
        ),

        # 6) Very light compressor: mostly catching occasional peaks
        Compressor(
            threshold_db=-3.0,   # higher threshold -> much less often active
            ratio=1.15,          # very low ratio
            attack_ms=25.0,
            release_ms=250.0,
        ),

        # 7) Very soft noise gate / expander
        NoiseGate(
            threshold_db=-65.0,  # very low -> almost never intrusive
            ratio=1.1,           # very gentle expansion
            attack_ms=5.0,
            release_ms=300.0,
        ),

        # No limiter here: we avoid re-flattening an already mastered track.
    ])

    return board


def apply_noise_reduction(audio: np.ndarray, samplerate: int) -> np.ndarray:
    """
    Apply a light spectral noise reduction using noisereduce.

    The goal is to reduce constant background noise (fan, hiss, room tone)
    without destroying the main content. Settings are intentionally moderate.

    For AI-generated songs (Suno), this step will usually do almost nothing,
    which is fine.
    """
    if audio.ndim == 1:
        # Mono signal
        audio_denoised = nr.reduce_noise(
            y=audio,
            sr=samplerate,
            prop_decrease=0.15,  # lighter reduction than before
            stationary=True,
        )
    else:
        # Multi-channel (e.g., stereo), shape: [channels, frames]
        channels = []
        for ch in audio:
            ch_denoised = nr.reduce_noise(
                y=ch,
                sr=samplerate,
                prop_decrease=0.15,
                stationary=True,
            )
            channels.append(ch_denoised)
        audio_denoised = np.vstack(channels)

    return audio_denoised


def match_loudness_to_reference(
    reference: np.ndarray,
    processed: np.ndarray,
    samplerate: int,
    offset_db: float = -3.0,
) -> np.ndarray:
    """
    Match the processed audio loudness to a reference audio, without ever
    making it louder than the reference.

    - Measure integrated loudness (LUFS) of both signals.
    - If the processed audio is louder than (reference + offset_db),
      bring it DOWN to (reference + offset_db).
    - If it is already quieter, leave it as-is (no upward gain).

    This is ideal for already-mastered material: we only attenuate.

    Args:
        reference: np.ndarray, original signal (after noise reduction),
                   shape [channels, frames] or [frames]
        processed: np.ndarray, processed signal (after EQ/comp),
                   same shape convention.
        samplerate: sample rate in Hz.
        offset_db: how much below the reference you want the processed track
                   to be, in dB. Example:
                   - offset_db = -3.0 -> processed target is ~3 dB quieter.

    Returns:
        Processed audio whose loudness is <= reference loudness + offset_db.
    """
    # Helper: convert [channels, frames] -> [frames, channels] for pyloudnorm
    def to_lufs_shape(x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            return x.T
        return x

    ref_for_lufs = to_lufs_shape(reference)
    proc_for_lufs = to_lufs_shape(processed)

    meter = pyln.Meter(samplerate)

    ref_loudness = meter.integrated_loudness(ref_for_lufs)
    proc_loudness = meter.integrated_loudness(proc_for_lufs)

    # Target loudness: slightly below the reference
    target_loudness = ref_loudness + offset_db  # offset_db is negative

    # If processed is louder than target, attenuate it.
    # Otherwise, leave it unchanged (no upward boost).
    if proc_loudness > target_loudness:
        normalized = pyln.normalize.loudness(
            proc_for_lufs,
            proc_loudness,
            target_loudness,
        )
    else:
        normalized = proc_for_lufs

    # Convert back to [channels, frames] if needed
    if normalized.ndim == 2:
        return normalized.T
    return normalized


def process_file(input_path: str, output_path: str) -> None:
    """
    Read an audio file, process it through optional noise reduction + effect chain,
    match loudness to the original (never louder), and write the result.

    Key points:
    - Noise reduction is skipped if the input is already extremely hot (peak > 0.9).
    - No limiter in the Pedalboard: we avoid making the waveform more "brick-like".
    - Loudness is typically brought ~3 dB below the original to reduce the
      impression of constant saturation.
    """
    # --- 0) Read input audio file ---
    with AudioFile(input_path) as f:
        # shape: [channels, frames] or [frames]
        audio = f.read(f.frames)
        samplerate = f.samplerate

    # --- 1) Decide whether to apply noise reduction ---
    peak_raw = np.max(np.abs(audio))
    if peak_raw > 0.9:
        # Already very hot (likely mastered): skip noise reduction to avoid artefacts
        audio_nr = audio
    else:
        # Apply light NR
        audio_nr = apply_noise_reduction(audio, samplerate)

    # Keep a copy for loudness reference (after NR decision, before processing)
    loudness_reference = np.copy(audio_nr)

    # --- 2) Input peak safety (attenuation only) ---
    # If the file is already peaking very high, bring it slightly down.
    peak_in = np.max(np.abs(audio_nr))
    if peak_in > 0.99:
        audio_nr = audio_nr / peak_in * 0.99

    # --- 3) Pedalboard processing: tonal shaping + very gentle dynamics ---
    board = build_board()
    processed = board(audio_nr, samplerate)

    # --- 4) Loudness matching to original (only attenuate if louder) ---
    # offset_db = -3.0 -> processed will be about 3 dB quieter than reference
    processed = match_loudness_to_reference(
        reference=loudness_reference,
        processed=processed,
        samplerate=samplerate,
        offset_db=-3.0,
    )

    # --- 5) Final peak safety (attenuation only) ---
    peak_out = np.max(np.abs(processed))
    if peak_out > 0.995:
        # Only turn down if we are very close to 0 dBFS
        processed = processed / peak_out * 0.995

    # --- 6) Ensure shape is [channels, frames] for writing ---
    if processed.ndim == 1:
        processed = processed.reshape(1, -1)

    num_channels = processed.shape[0]

    # --- 7) Write processed audio to output file ---
    # AudioFile expects: (path, mode, samplerate, num_channels)
    with AudioFile(output_path, 'w', samplerate, num_channels) as f:
        f.write(processed.astype(np.float32))

    print(f"Processed file written to: {output_path}")


def main() -> None:
    """
    Command-line entry point.

    Examples:
        python auto_clean.py input.wav output.wav
        python auto_clean.py input.wav  # -> input_cleaned.wav
    """
    parser = argparse.ArgumentParser(
        description=(
            "Automatic clean-up of a 'boxy / underwater / muffled' audio track, "
            "keeping overall loudness at or below the original (good for Suno songs)."
        )
    )
    parser.add_argument(
        "input",
        help="Path to the input audio file (wav, flac, etc.)",
    )
    parser.add_argument(
        "output",
        nargs="?",  # optional: if omitted, will generate <input>_cleaned.<ext>
        help="Path to the output audio file (optional). "
             "Default: <input>_cleaned.<ext>",
    )
    args = parser.parse_args()

    from pathlib import Path

    in_path = Path(args.input)
    if args.output is None:
        # song.wav -> song_cleaned.wav
        out_path = in_path.with_name(in_path.stem + "_cleaned" + in_path.suffix)
    else:
        out_path = Path(args.output)

    process_file(str(in_path), str(out_path))


if __name__ == "__main__":
    main()
