from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import json
import re
import difflib

import torch

import whisper

# phonémisation optionnelle (pour plus tard, karaoké avancé, etc.)
try:
    from phonemizer import phonemize
    HAVE_PHONEMIZER = True
except ImportError:
    HAVE_PHONEMIZER = False


# ---------- Types ----------

@dataclass
class Phrase:
    line_index: int
    text: str
    start: Optional[float]  # seconds
    end: Optional[float]    # seconds


@dataclass
class WordTimed:
    line_index: int
    word_index: int
    text: str
    matched_text: str
    start: float
    end: float


@dataclass
class PhonemeTimed:
    line_index: int
    word_index: int
    word: str
    phoneme: str
    start: float
    end: float


@dataclass
class AlignmentResult:
    phrases: List[Phrase]
    words: List[WordTimed]
    phonemes: List[PhonemeTimed]


ProgressCallback = Callable[[float, str], None]


# ---------- Utilitaires texte / temps (repris de ton script) ----------

def normalize_word(w: str) -> str:
    """Normalize a word for alignment (lowercase, strip simple punctuation)."""
    w = w.lower().strip()
    w = re.sub(r"[.,!?;:«»\"“”()\-–—']", "", w)
    return w


def format_srt_timestamp(t: float) -> str:
    """Convert float seconds to SRT timestamp hh:mm:ss,mmm."""
    if t < 0:
        t = 0.0
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    millis = int(round((t - int(t)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


# ---------- Étape 1 : transcription Whisper ----------

def transcribe_with_whisper(
    audio_path: Path,
    model_name: str = "medium",
    language: Optional[str] = None,
    device: Optional[str] = None,
    beam_size: int = 5,
    patience: float = 1.0,
    no_speech_threshold: Optional[float] = None,
    compression_ratio_threshold: Optional[float] = None,
    logprob_threshold: Optional[float] = None,
    condition_on_previous_text: Optional[bool] = None,
    initial_prompt: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
):
    """
    Run Whisper on the given audio file and return (segments, recognized_words).

    Parameters
    ----------
    beam_size:
        Number of beams for Whisper's beam search. Values between 4 and 8
        generally improve quality at the cost of speed.
    patience:
        Patience parameter for beam search. Values > 1.0 make the search
        more exhaustive and can slightly improve quality.
    no_speech_threshold:
        If set, overrides Whisper's default threshold to classify segments
        as 'no speech'. Lower values keep more low-intelligibility segments.
    compression_ratio_threshold:
        If set, adjusts the filter for repetitive / hallucinated segments.
        Higher values are more tolerant and drop fewer segments.
    logprob_threshold:
        If set, adjusts the average log-probability threshold for rejecting
        uncertain segments. Lower (more negative) values are more tolerant.
    condition_on_previous_text:
        If set, toggles whether each segment uses the previous text as
        decoding context. Disabling this can help with multilingual / Latin
        material to reduce language drift.
    initial_prompt:
        Optional textual prompt to bias Whisper towards a specific language
        or vocabulary (e.g. Latin liturgy, harsh vocals).
    """
    # Choose device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if progress_cb:
        progress_cb(5.0, f"Loading Whisper model '{model_name}' on {device}...")

    model = whisper.load_model(model_name, device=device)

    if progress_cb:
        progress_cb(20.0, "Transcribing with word-level timestamps...")

    # Clamp and sanitize decoding parameters
    beam_size = max(1, int(beam_size))
    patience = max(0.0, float(patience))

    transcribe_kwargs = {
        "word_timestamps": True,
        "task": "transcribe",
        "verbose": False,
        "fp16": (device == "cuda"),
        "beam_size": beam_size,
        "patience": patience,
    }

    if language:
        transcribe_kwargs["language"] = language

    if no_speech_threshold is not None:
        transcribe_kwargs["no_speech_threshold"] = float(no_speech_threshold)

    if compression_ratio_threshold is not None:
        transcribe_kwargs["compression_ratio_threshold"] = float(compression_ratio_threshold)

    if logprob_threshold is not None:
        transcribe_kwargs["logprob_threshold"] = float(logprob_threshold)

    if condition_on_previous_text is not None:
        transcribe_kwargs["condition_on_previous_text"] = bool(condition_on_previous_text)

    if initial_prompt:
        transcribe_kwargs["initial_prompt"] = str(initial_prompt)

    result = model.transcribe(str(audio_path), **transcribe_kwargs)
    segments = result["segments"]

    recognized_words = []
    for seg in segments:
        seg_id = seg.get("id", None)
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_text = seg["text"]

        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                wt = w["word"].strip()
                if not wt:
                    continue
                recognized_words.append(
                    {
                        "text": wt,
                        "norm": normalize_word(wt),
                        "start": float(w["start"]),
                        "end": float(w["end"]),
                        "segment_index": seg_id,
                        "segment_start": float(seg_start),
                        "segment_end": float(seg_end),
                    }
                )
        else:
            # Fallback: split segment into words and distribute time uniformly
            raw_words = [x for x in re.split(r"\s+", seg_text) if x.strip()]
            if not raw_words:
                continue
            duration = seg_end - seg_start
            step = duration / len(raw_words)
            for i, wt in enumerate(raw_words):
                wt = wt.strip()
                if not wt:
                    continue
                w_start = seg_start + i * step
                w_end = seg_start + (i + 1) * step
                recognized_words.append(
                    {
                        "text": wt,
                        "norm": normalize_word(wt),
                        "start": float(w_start),
                        "end": float(w_end),
                        "segment_index": seg_id,
                        "segment_start": float(seg_start),
                        "segment_end": float(seg_end),
                    }
                )

    if progress_cb:
        progress_cb(50.0, "Transcription done. Aligning lyrics...")

    return segments, recognized_words

# ---------- Étape 2 : alignement lyrics ↔ mots reconnus ----------

def read_lyrics_from_text(lyrics_text: str) -> List[str]:
    """
    Convert lyrics from a big text block to a list of non-empty lines.
    (1 phrase per line in the UI = 1 phrase alignée)
    """
    lines = []
    for line in lyrics_text.splitlines():
        line = line.strip()
        if not line:
            continue
        lines.append(line)
    return lines


def align_lyrics_to_words(
    lyrics_lines: List[str],
    recognized_words,
    max_search_window: int = 5,
    min_similarity: float = 0.6,
    progress_cb: Optional[ProgressCallback] = None,
):
    """
    Simple alignment:
    - for each word in each lyrics line,
    - find the best matching recognized word in a sliding window starting at cursor.

    Parameters
    ----------
    max_search_window:
        How many recognized words to look ahead (from the current cursor)
        when searching for the best match for a lyrics word.
        Larger values are more robust to insertions/omissions but slower.
    min_similarity:
        Minimal difflib similarity (0–1) required to accept a match.
        Higher values give stricter matches (fewer false positives),
        but may leave more lyrics words unmatched.
    """
    if progress_cb:
        progress_cb(55.0, "Aligning lyrics to recognized words...")

    # Sanitize parameters
    max_search_window = max(1, int(max_search_window))
    min_similarity = max(0.0, min(1.0, float(min_similarity)))

    phrases: List[Phrase] = []
    words_timed: List[WordTimed] = []

    cursor = 0  # index into recognized_words
    n_rec = len(recognized_words)

    for line_idx, line in enumerate(lyrics_lines):
        line_words = [w for w in re.split(r"\s+", line) if w]
        norm_line_words = [normalize_word(w) for w in line_words]

        matched_indices = []
        for w_idx, (word, norm_word) in enumerate(zip(line_words, norm_line_words)):
            if not norm_word:
                continue

            best_j = None
            best_score = 0.0

            # search window [cursor, cursor + max_search_window)
            for j in range(cursor, min(cursor + max_search_window, n_rec)):
                cand = recognized_words[j]
                score = difflib.SequenceMatcher(a=norm_word, b=cand["norm"]).ratio()
                if score > best_score:
                    best_score = score
                    best_j = j

            if best_j is not None and best_score >= min_similarity:
                rec_w = recognized_words[best_j]
                matched_indices.append((w_idx, best_j, best_score))

                words_timed.append(
                    WordTimed(
                        line_index=line_idx,
                        word_index=w_idx,
                        text=word,
                        matched_text=rec_w["text"],
                        start=rec_w["start"],
                        end=rec_w["end"],
                    )
                )

                cursor = best_j + 1

        # phrase timing
        if matched_indices:
            idxs = [j for (_, j, _) in matched_indices]
            starts = [recognized_words[j]["start"] for j in idxs]
            ends = [recognized_words[j]["end"] for j in idxs]
            phrase_start = float(min(starts))
            phrase_end = float(max(ends))
        else:
            phrase_start = None
            phrase_end = None

        phrases.append(
            Phrase(
                line_index=line_idx,
                text=line,
                start=phrase_start,
                end=phrase_end,
            )
        )

    if progress_cb:
        progress_cb(80.0, "Alignment done. Preparing outputs...")

    return phrases, words_timed

# ---------- Étape 3 : SRT & JSON ----------

def write_srt(phrases: List[Phrase], srt_path: Path):
    lines: List[str] = []
    idx = 1
    for p in phrases:
        if p.start is None or p.end is None:
            continue

        start_ts = format_srt_timestamp(p.start)
        end_ts = format_srt_timestamp(p.end)

        lines.append(str(idx))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(p.text)
        lines.append("")  # blank line
        idx += 1

    srt_path.write_text("\n".join(lines), encoding="utf-8")


def phonemize_words(
    words_timed: List[WordTimed],
    phoneme_lang: str = "fr-fr",
    progress_cb: Optional[ProgressCallback] = None,
) -> List[PhonemeTimed]:
    """
    From the timed words, compute a simple timing for each phoneme,
    by splitting the word duration uniformly.
    """
    if not HAVE_PHONEMIZER:
        if progress_cb:
            progress_cb(90.0, "Phonemizer not installed, skipping phonemes.")
        return []

    if progress_cb:
        progress_cb(90.0, "Phonemizing words...")

    texts = [w.text for w in words_timed]
    ph_strings = phonemize(
        texts,
        language=phoneme_lang,
        backend="espeak",
        strip=True,
        njobs=1,
    )

    if isinstance(ph_strings, str):
        ph_strings = [ph_strings]

    phonemes_timed: List[PhonemeTimed] = []

    for w, ph_str in zip(words_timed, ph_strings):
        phs = [p for p in ph_str.split() if p]
        if not phs:
            continue

        duration = float(w.end - w.start)
        if duration <= 0:
            continue

        step = duration / len(phs)
        for i, ph in enumerate(phs):
            ph_start = float(w.start + i * step)
            ph_end = float(w.start + (i + 1) * step)
            phonemes_timed.append(
                PhonemeTimed(
                    line_index=w.line_index,
                    word_index=w.word_index,
                    word=w.text,
                    phoneme=ph,
                    start=ph_start,
                    end=ph_end,
                )
            )

    return phonemes_timed


def _infer_phoneme_lang_from_whisper_lang(lang: Optional[str]) -> str:
    """
    Very simple mapping from Whisper language code to phonemizer language code.
    Adjust as needed.
    """
    if not lang:
        return "fr-fr"
    lang = lang.lower()
    if lang == "fr":
        return "fr-fr"
    if lang == "en":
        return "en-us"
    if lang == "de":
        return "de-de"
    return "fr-fr"


# ---------- Entrée principale pour Olaf ----------

def run_alignment_for_project(
    project,
    audio_path: Path,
    lyrics_text: str,
    model_name: str = "medium",
    whisper_language: Optional[str] = None,
    phoneme_language: Optional[str] = None,
    device: Optional[str] = None,
    beam_size: int = 5,
    patience: float = 1.0,
    max_search_window: int = 5,
    min_similarity: float = 0.6,
    no_speech_threshold: Optional[float] = None,
    compression_ratio_threshold: Optional[float] = None,
    logprob_threshold: Optional[float] = None,
    condition_on_previous_text: Optional[bool] = None,
    initial_prompt: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> AlignmentResult:
    """
    High-level function used by the Vocal tab.

    Input
    -----
    project:
        Project instance (used to resolve folders).
    audio_path:
        Audio file path (main track or stem).
    lyrics_text:
        Lyrics from the UI (one line per phrase).
    model_name:
        Whisper model name ("tiny", "base", "small", "medium", "large-v2", "large-v3", ...).
    whisper_language:
        Language code for Whisper (e.g., 'fr', 'en') or None for auto-detect.
    phoneme_language:
        Phonemizer language code (e.g., 'fr-fr'), optional.
    device:
        "cuda" or "cpu". If None, it is inferred from torch.cuda.is_available().
    beam_size / patience:
        Beam search parameters forwarded to Whisper.
    max_search_window / min_similarity:
        Alignment refinement parameters forwarded to align_lyrics_to_words().
    no_speech_threshold / compression_ratio_threshold / logprob_threshold:
        Advanced Whisper filters to keep or reject low-confidence segments.
    condition_on_previous_text:
        Whisper context usage; disabling can help with strongly multilingual
        or Latin material.
    initial_prompt:
        Optional text shown to Whisper before decoding. Can hint the model
        about domain, language, or vocabulary (e.g. Latin liturgy).
    """
    audio_path = Path(audio_path).resolve()

    if progress_cb:
        progress_cb(0.0, "Starting alignment...")

    # 1) Read lyrics from text
    lyrics_lines = read_lyrics_from_text(lyrics_text)
    if not lyrics_lines and progress_cb:
        progress_cb(2.0, "Warning: no lyrics provided; alignment will be poor.")

    # 2) Whisper transcription with advanced parameters
    segments, recognized_words = transcribe_with_whisper(
        audio_path,
        model_name=model_name,
        language=whisper_language,
        device=device,
        beam_size=beam_size,
        patience=patience,
        no_speech_threshold=no_speech_threshold,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        condition_on_previous_text=condition_on_previous_text,
        initial_prompt=initial_prompt,
        progress_cb=progress_cb,
    )

    # 3) Lyrics ↔ words alignment
    phrases, words_timed = align_lyrics_to_words(
        lyrics_lines,
        recognized_words,
        max_search_window=max_search_window,
        min_similarity=min_similarity,
        progress_cb=progress_cb,
    )

    # 4) Optional phonemes (for future usage)
    if not phoneme_language:
        phoneme_language = _infer_phoneme_lang_from_whisper_lang(whisper_language)
    phonemes_timed = phonemize_words(
        words_timed,
        phoneme_lang=phoneme_language,
        progress_cb=progress_cb,
    )

    result = AlignmentResult(
        phrases=phrases,
        words=words_timed,
        phonemes=phonemes_timed,
    )

    # 5) Write JSON + SRT into the project's vocal_align folder
    align_dir = project.folder / "vocal_align"
    align_dir.mkdir(parents=True, exist_ok=True)

    phrases_json = [
        {
            "line_index": p.line_index,
            "text": p.text,
            "start": p.start,
            "end": p.end,
        }
        for p in phrases
    ]
    words_json = [
        {
            "line_index": w.line_index,
            "word_index": w.word_index,
            "text": w.text,
            "matched_text": w.matched_text,
            "start": w.start,
            "end": w.end,
        }
        for w in words_timed
    ]
    phonemes_json = [
        {
            "line_index": ph.line_index,
            "word_index": ph.word_index,
            "word": ph.word,
            "phoneme": ph.phoneme,
            "start": ph.start,
            "end": ph.end,
        }
        for ph in phonemes_timed
    ]

    (align_dir / "phrases.json").write_text(
        json.dumps(phrases_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (align_dir / "words.json").write_text(
        json.dumps(words_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (align_dir / "phonemes.json").write_text(
        json.dumps(phonemes_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    srt_path = align_dir / "subtitles.srt"
    write_srt(phrases, srt_path)

    if progress_cb:
        progress_cb(100.0, "Alignment finished.")

    return result
