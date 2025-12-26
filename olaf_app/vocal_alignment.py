from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json
import re
import difflib
import unicodedata

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
    # Debug / scoring metadata for UI and exports.
    # This mirrors the content written to vocal_align/alignment_metrics.json.
    metrics: Dict[str, Any]
    chosen_whisper_language: str


ProgressCallback = Callable[[float, str], None]


# ---------- Utilitaires texte / temps (repris de ton script) ----------

def normalize_word(w: str) -> str:
    """Normalize a word for alignment.

    - Lowercase, strip punctuation.
    - Strip Unicode accents (Latin choirs often come with inconsistent spelling).
    - Normalize common ligatures.

    The goal is to make Whisper output and user lyrics comparable without
    being language-agnostic (English + Latin works well).
    """
    w = (w or "").strip().lower()
    if not w:
        return ""

    # Common ligatures / special cases
    w = (w.replace("æ", "ae")
           .replace("œ", "oe")
           .replace("ß", "ss"))

    # Unicode normalization: NFKD + remove diacritics/combining marks
    w = unicodedata.normalize("NFKD", w)
    w = "".join(ch for ch in w if not unicodedata.combining(ch))

    # Remove typical punctuation/apostrophes/dashes
    w = re.sub(r"[.,!?;:«»\"“”()\[\]{}<>\\/\-–—'’`´]+", "", w)
    w = re.sub(r"\s+", "", w)

    # Keep only ascii letters/digits
    w = re.sub(r"[^0-9a-z]+", "", w)

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

    recognized_words = _postprocess_recognized_word_timings(recognized_words)
    
    return segments, recognized_words

def _postprocess_recognized_word_timings(recognized_words: list[dict]) -> list[dict]:
    """Improve Whisper word-level timings for singing material.

    Whisper often truncates the end time of held notes (long vowels),
    especially at segment boundaries. This post-process:
    - clamps words inside their segment bounds,
    - ensures monotonic non-overlapping timings inside each segment,
    - extends word end time up to the next word start (or segment end for last word),
      when Whisper ended too early.

    This is a heuristic; it does not replace forced alignment.
    """
    if not recognized_words:
        return recognized_words

    # Group by segment_index (None-safe)
    by_seg: dict[str, list[dict]] = {}
    for w in recognized_words:
        seg = w.get("segment_index", None)
        key = str(seg) if seg is not None else "__NOSEG__"
        by_seg.setdefault(key, []).append(w)

    for _seg_key, words in by_seg.items():
        # Sort by start time
        words.sort(key=lambda x: float(x.get("start", 0.0)))

        # Segment bounds (fallback if missing)
        seg_start = float(words[0].get("segment_start", 0.0) or 0.0)
        seg_end = float(words[0].get("segment_end", seg_start) or seg_start)
        if seg_end < seg_start:
            seg_end = seg_start

        # Pass 1: clamp inside segment + enforce minimal duration
        MIN_DUR = 0.060  # 60 ms
        for w in words:
            s = float(w.get("start", seg_start) or seg_start)
            e = float(w.get("end", s) or s)

            s = max(seg_start, min(s, seg_end))
            e = max(s, min(e, seg_end))

            if (e - s) < MIN_DUR:
                e = min(seg_end, s + MIN_DUR)

            w["start"] = s
            w["end"] = e

        # Pass 2: extend ends to reduce premature cutoffs
        # - For last word: end = segment_end (common held note case)
        # - For others: end <= next.start and can be extended up to next.start if Whisper ended too early
        for i, w in enumerate(words):
            s = float(w["start"])
            e = float(w["end"])

            if i == len(words) - 1:
                # Strong heuristic: last word usually stretches to seg_end in singing
                e = max(e, seg_end)
                e = min(e, seg_end)
            else:
                nxt = words[i + 1]
                nxt_s = float(nxt["start"])

                # Prevent overlap
                if e > nxt_s:
                    e = max(s, nxt_s)

                # If Whisper ended *well before* the next word, extend up to next word start
                # (helps held vowels inside a segment)
                GAP_EXTEND_THRESHOLD = 0.120  # 120 ms
                if (nxt_s - e) > GAP_EXTEND_THRESHOLD:
                    e = min(nxt_s, seg_end)

            # Re-apply minimal duration just in case
            if (e - s) < MIN_DUR:
                e = min(seg_end, s + MIN_DUR)

            w["end"] = float(e)

        # Pass 3: ensure monotonic start/end overall
        prev_end = seg_start
        for w in words:
            s = max(float(w["start"]), prev_end)
            e = max(float(w["end"]), s)
            w["start"] = s
            w["end"] = e
            prev_end = e

    return recognized_words

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



def _tokenize_lyrics_lines(lyrics_lines: List[str]):
    """Flatten lyrics into tokens while preserving (line_index, word_index)."""
    tokens = []
    for line_idx, line in enumerate(lyrics_lines):
        words = [w for w in re.split(r"\s+", line.strip()) if w]
        for w_idx, w in enumerate(words):
            nw = normalize_word(w)
            if not nw:
                continue
            tokens.append(
                {"line_index": line_idx, "word_index": w_idx, "raw": w, "norm": nw}
            )
    return tokens


class _SimilarityCache:
    """Tiny similarity cache to reduce difflib overhead in DP."""
    def __init__(self, max_items: int = 200_000):
        self._max_items = int(max_items)
        self._d = {}

    def get(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        key = (a, b) if a <= b else (b, a)
        v = self._d.get(key)
        if v is not None:
            return v
        v = difflib.SequenceMatcher(a=a, b=b).ratio()
        if len(self._d) < self._max_items:
            self._d[key] = v
        return v


def _similarity(a: str, b: str, cache: Optional[_SimilarityCache] = None) -> float:
    if cache is None:
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(a=a, b=b).ratio()
    return cache.get(a, b)


def _dp_global_align(
    lyr_norms: List[str],
    rec_norms: List[str],
    scoring_min_similarity: float,
    gap_lyr: float,
    gap_rec: float,
    cache: Optional[_SimilarityCache] = None,
):
    """Global DP alignment between two token sequences.

    Returns a monotonic path of (lyr_index, rec_index, similarity).
    Gaps are encoded as (lyr_index, None, 0) or (None, rec_index, 0).
    """
    n = len(lyr_norms)
    m = len(rec_norms)

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]  # 1=match, 2=gap_lyr, 3=gap_rec

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_lyr
        bt[i][0] = 2
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap_rec
        bt[0][j] = 3

    scoring_min_similarity = max(0.0, min(1.0, float(scoring_min_similarity)))

    for i in range(1, n + 1):
        a = lyr_norms[i - 1]
        for j in range(1, m + 1):
            b = rec_norms[j - 1]
            sim = _similarity(a, b, cache=cache)

            if sim >= scoring_min_similarity:
                match_score = 2.0 + 6.0 * sim
            else:
                match_score = -4.0 + 2.0 * sim

            s_match = dp[i - 1][j - 1] + match_score
            s_gap_lyr = dp[i - 1][j] + gap_lyr
            s_gap_rec = dp[i][j - 1] + gap_rec

            best = s_match
            bcode = 1
            if s_gap_lyr > best:
                best = s_gap_lyr
                bcode = 2
            if s_gap_rec > best:
                best = s_gap_rec
                bcode = 3

            dp[i][j] = best
            bt[i][j] = bcode

    path = []
    i, j = n, m
    while i > 0 or j > 0:
        bcode = bt[i][j]
        if bcode == 1 and i > 0 and j > 0:
            a = lyr_norms[i - 1]
            b = rec_norms[j - 1]
            sim = _similarity(a, b, cache=cache)
            path.append((i - 1, j - 1, float(sim)))
            i -= 1
            j -= 1
        elif bcode == 2 and i > 0:
            path.append((i - 1, None, 0.0))
            i -= 1
        else:
            path.append((None, j - 1, 0.0))
            j -= 1

    path.reverse()
    return path


def _compute_alignment_metrics(lyr_tokens, matches: List[tuple], n_lines: int) -> dict:
    """Compute metrics used to compare runs and debug poor results."""
    total = len(lyr_tokens)
    matched = len(matches)
    sims = [float(s) for *_rest, s in matches] if matches else []
    mean_sim = float(sum(sims) / len(sims)) if sims else 0.0
    med_sim = float(sorted(sims)[len(sims) // 2]) if sims else 0.0

    by_line_total = [0] * n_lines
    by_line_matched = [0] * n_lines
    for t in lyr_tokens:
        li = int(t["line_index"])
        if 0 <= li < n_lines:
            by_line_total[li] += 1
    for lyr_i, _rec_i, _sim in matches:
        li = int(lyr_tokens[lyr_i]["line_index"])
        if 0 <= li < n_lines:
            by_line_matched[li] += 1

    by_line_coverage = []
    for li in range(n_lines):
        denom = by_line_total[li] or 1
        by_line_coverage.append(float(by_line_matched[li] / denom))

    return {
        "lyrics_words_total": int(total),
        "lyrics_words_matched": int(matched),
        "coverage": float(matched / (total or 1)),
        "mean_similarity": mean_sim,
        "median_similarity": med_sim,
        "by_line_coverage": by_line_coverage,
    }


def _parse_language_candidates(whisper_language: Optional[str]) -> List[Optional[str]]:
    """Parse a language field that may contain multiple candidates.

    Examples:
    - None / "auto" -> [None]
    - "en" -> ["en"]
    - "en,la,auto" -> ["en", "la", None]
    """
    if whisper_language is None:
        return [None]
    s = str(whisper_language).strip().lower()
    if not s or s in ("auto", "detect", "none"):
        return [None]
    parts = [p.strip() for p in re.split(r"[,;|/]+", s) if p.strip()]
    if len(parts) <= 1:
        return [s]
    out: List[Optional[str]] = []
    for p in parts:
        if p in ("auto", "detect", "none"):
            out.append(None)
        else:
            out.append(p)
    seen = set()
    dedup = []
    for x in out:
        key = x or "__AUTO__"
        if key in seen:
            continue
        seen.add(key)
        dedup.append(x)
    return dedup

def _fill_missing_words_between_neighbors(
    lyr_tokens: List[dict],
    words_timed: List[WordTimed],
    min_word_dur: float = 0.060,
) -> tuple[List[WordTimed], int]:
    """Fill missing lyrics words between two timed neighbors.

    Original behavior:
        If there is not enough room in (left.end -> right.start) to allocate
        k * min_word_dur, we skip the gap entirely (leaving missing words).

    Improved behavior (salvage-friendly):
        - If the gap is too small, try to create room by:
            1) shortening left neighbor end a bit, and/or
            2) moving right neighbor start a bit later (keeping a minimal duration),
            3) if still not enough, shift the right neighbor AND all subsequent words
               of the same line to the right (ripple shift).
        - Then allocate missing words uniformly in the created window.

    Returns:
        (new_words_timed, filled_count)
    """
    if not lyr_tokens or not words_timed:
        return words_timed, 0

    # Fast lookup for already-timed words (matched or previously inserted)
    timed_by_key: dict[tuple[int, int], WordTimed] = {
        (int(w.line_index), int(w.word_index)): w for w in words_timed
    }

    # Group lyric tokens by line, ordered by word_index (lyrics order)
    tokens_by_line: dict[int, List[dict]] = {}
    for t in lyr_tokens:
        li = int(t["line_index"])
        tokens_by_line.setdefault(li, []).append(t)
    for li in tokens_by_line:
        tokens_by_line[li].sort(key=lambda x: int(x["word_index"]))

    filled: List[WordTimed] = []
    filled_count = 0

    # Tunables (salvage heuristics)
    # Keep at least this duration when squeezing neighbor words.
    # (We allow a bit smaller than min_word_dur so we can create space in tight gaps.)
    KEEP_NEIGHBOR_MIN_DUR = max(0.030, float(min_word_dur) * 0.50)
    # If still not enough room after squeezing, we shift right-side words.
    ALLOW_RIPPLE_SHIFT = True

    def _try_squeeze_neighbors(left_w: WordTimed, right_w: WordTimed, need_total: float) -> None:
        """Try to free time by shrinking left end and moving right start later."""
        # Current window
        window_start = float(left_w.end)
        window_end = float(right_w.start)
        avail = window_end - window_start
        if avail >= need_total:
            return

        deficit = need_total - max(0.0, avail)

        # How much can we shrink left word end while keeping minimal duration?
        left_len = float(left_w.end) - float(left_w.start)
        left_slack = max(0.0, left_len - KEEP_NEIGHBOR_MIN_DUR)

        # How much can we move right word start later while keeping minimal duration?
        right_len = float(right_w.end) - float(right_w.start)
        right_slack = max(0.0, right_len - KEEP_NEIGHBOR_MIN_DUR)

        if deficit <= 0.0 or (left_slack <= 0.0 and right_slack <= 0.0):
            return

        # Split the borrowing across both sides, then top-up from remaining slack.
        borrow_left = min(left_slack, deficit * 0.5)
        borrow_right = min(right_slack, deficit - borrow_left)

        rem = deficit - (borrow_left + borrow_right)
        if rem > 0.0 and left_slack > borrow_left:
            extra = min(left_slack - borrow_left, rem)
            borrow_left += extra
            rem -= extra
        if rem > 0.0 and right_slack > borrow_right:
            extra = min(right_slack - borrow_right, rem)
            borrow_right += extra
            rem -= extra

        if borrow_left > 0.0:
            new_left_end = float(left_w.end) - borrow_left
            # Safety: never invert start/end
            left_w.end = max(float(left_w.start) + 0.010, new_left_end)

        if borrow_right > 0.0:
            new_right_start = float(right_w.start) + borrow_right
            # Safety: keep right duration >= 10ms
            right_w.start = min(new_right_start, float(right_w.end) - 0.010)

    def _ripple_shift_line_words(line_index: int, from_word_index: int, shift: float) -> None:
        """Shift right-side words of the same line to create time budget."""
        if shift <= 0.0:
            return
        for w in words_timed:
            if int(w.line_index) != int(line_index):
                continue
            if int(w.word_index) < int(from_word_index):
                continue
            w.start = float(w.start) + float(shift)
            w.end = float(w.end) + float(shift)

    for li, toks in tokens_by_line.items():
        if not toks:
            continue

        # Find gaps between two timed neighbors in lyrics order
        for left_pos in range(len(toks) - 1):
            left_t = toks[left_pos]
            left_key = (li, int(left_t["word_index"]))
            if left_key not in timed_by_key:
                continue

            # Find the next timed token to the right
            right_pos = None
            for rp in range(left_pos + 1, len(toks)):
                rt = toks[rp]
                if (li, int(rt["word_index"])) in timed_by_key:
                    right_pos = rp
                    break
            if right_pos is None:
                continue

            # Missing tokens between left_pos and right_pos
            gap_tokens = toks[left_pos + 1 : right_pos]
            if not gap_tokens:
                continue

            gap_tokens = [
                t for t in gap_tokens
                if (li, int(t["word_index"])) not in timed_by_key
            ]
            if not gap_tokens:
                continue

            left_w = timed_by_key[left_key]
            right_t = toks[right_pos]
            right_w = timed_by_key[(li, int(right_t["word_index"]))]

            k = len(gap_tokens)
            need_total = float(min_word_dur) * float(k)

            # Window to allocate missing words
            window_start = float(left_w.end)
            window_end = float(right_w.start)
            avail = window_end - window_start

            # If not enough room, try squeeze + (optionally) ripple-shift to create space.
            if avail < need_total:
                _try_squeeze_neighbors(left_w, right_w, need_total)

                window_start = float(left_w.end)
                window_end = float(right_w.start)
                avail = window_end - window_start

                if ALLOW_RIPPLE_SHIFT and avail < need_total:
                    shift = need_total - max(0.0, avail)
                    # Shift right_w and all subsequent words in the same line
                    _ripple_shift_line_words(li, int(right_w.word_index), shift)

                    window_start = float(left_w.end)
                    window_end = float(right_w.start)
                    avail = window_end - window_start

            # If still no positive window, we cannot allocate reliably.
            # (This should be extremely rare after ripple shift.)
            if avail <= 0.0:
                continue

            # Allocate equal partitions
            step = avail / float(k)

            for i, t in enumerate(gap_tokens):
                raw = str(t.get("raw", "")).strip()
                if not raw:
                    continue

                s = window_start + float(i) * step
                e = window_start + float(i + 1) * step

                # Enforce minimal duration inside the window (after squeeze/shift, this should fit)
                if (e - s) < float(min_word_dur):
                    e = min(window_end, s + float(min_word_dur))
                if e > window_end:
                    e = window_end
                if e <= s:
                    continue

                wt = WordTimed(
                    line_index=int(li),
                    word_index=int(t["word_index"]),
                    text=raw,
                    matched_text="",  # empty => estimated / not a real Whisper match
                    start=float(s),
                    end=float(e),
                )
                timed_by_key[(int(li), int(t["word_index"]))] = wt
                filled.append(wt)
                filled_count += 1

    if filled_count == 0:
        return words_timed, 0

    merged = list(words_timed) + filled
    merged.sort(key=lambda w: (int(w.line_index), float(w.start), int(w.word_index)))
    return merged, filled_count

def _fill_missing_words_at_line_edges(
    lyr_tokens: List[dict],
    words_timed: List[WordTimed],
    min_word_dur: float = 0.060,
    max_edge_extend_sec: float = 5.0,
) -> tuple[List[WordTimed], int]:
    """Fill missing lyrics words at the BEGINNING / END of each line.

    This is a late-stage UI/UX salvage. If Whisper missed words at the line
    edges (before the first matched word or after the last matched word), we
    can still create synthetic timings as long as we have a safe window:

    - Missing BEFORE the first timed word of a line:
        place words in a window ending at first.start, starting at:
          max(prev_line_end, first.start - max_edge_extend_sec)
        unless more time is needed to fit k * min_word_dur.

    - Missing AFTER the last timed word of a line:
        place words in a window starting at last.end, ending at:
          min(next_line_start, last.end + max_edge_extend_sec)
        unless more time is needed to fit k * min_word_dur.

    We never cross the previous line end / next line start, to avoid overlap.
    Generated words have matched_text="" (estimated).

    Returns:
        (new_words_timed, filled_count)
    """
    if not lyr_tokens or not words_timed:
        return words_timed, 0

    max_edge_extend_sec = max(0.0, float(max_edge_extend_sec))
    min_word_dur = max(0.010, float(min_word_dur))

    # Fast lookup for already-timed words (matched or previously inserted)
    timed_by_key: dict[tuple[int, int], WordTimed] = {
        (int(w.line_index), int(w.word_index)): w for w in words_timed
    }

    # Group lyric tokens by line, ordered by word_index
    tokens_by_line: dict[int, List[dict]] = {}
    for t in lyr_tokens:
        li = int(t["line_index"])
        tokens_by_line.setdefault(li, []).append(t)
    for li in tokens_by_line:
        tokens_by_line[li].sort(key=lambda x: int(x["word_index"]))

    # Per-line timing bounds from already timed words
    line_first_start: dict[int, float] = {}
    line_last_end: dict[int, float] = {}
    for w in words_timed:
        li = int(w.line_index)
        s = float(w.start)
        e = float(w.end)
        if li not in line_first_start or s < line_first_start[li]:
            line_first_start[li] = s
        if li not in line_last_end or e > line_last_end[li]:
            line_last_end[li] = e

    sorted_lines = sorted(tokens_by_line.keys())

    def prev_line_end(li: int) -> Optional[float]:
        try:
            idx = sorted_lines.index(li)
        except ValueError:
            return None
        if idx <= 0:
            return None
        pli = sorted_lines[idx - 1]
        return line_last_end.get(pli)

    def next_line_start(li: int) -> Optional[float]:
        try:
            idx = sorted_lines.index(li)
        except ValueError:
            return None
        if idx >= len(sorted_lines) - 1:
            return None
        nli = sorted_lines[idx + 1]
        return line_first_start.get(nli)

    filled: List[WordTimed] = []
    filled_count = 0

    for li, toks in tokens_by_line.items():
        # Need at least one timed word in this line
        if li not in line_first_start or li not in line_last_end:
            continue

        timed_indices = sorted(
            int(t["word_index"]) for t in toks if (li, int(t["word_index"])) in timed_by_key
        )
        if not timed_indices:
            continue

        first_idx = timed_indices[0]
        last_idx = timed_indices[-1]

        # ---- Fill before first timed word
        missing_before = [
            t
            for t in toks
            if int(t["word_index"]) < first_idx and (li, int(t["word_index"])) not in timed_by_key
        ]
        if missing_before:
            missing_before.sort(key=lambda x: int(x["word_index"]))
            k = len(missing_before)
            first_w = timed_by_key[(li, first_idx)]
            window_end = float(first_w.start)

            ple = prev_line_end(li)

            cap_start = window_end - max_edge_extend_sec
            if ple is not None:
                cap_start = max(cap_start, float(ple))

            need_start = window_end - (k * min_word_dur)
            if ple is not None:
                need_start = max(need_start, float(ple))

            window_start = min(cap_start, need_start)

            if (window_end - window_start) >= (k * min_word_dur) and window_end > window_start:
                step = (window_end - window_start) / k
                for i, t in enumerate(missing_before):
                    raw = str(t.get("raw", "")).strip()
                    if not raw:
                        continue
                    s = window_start + i * step
                    e = window_start + (i + 1) * step
                    if e <= s:
                        continue
                    wt = WordTimed(
                        line_index=int(li),
                        word_index=int(t["word_index"]),
                        text=raw,
                        matched_text="",
                        start=float(s),
                        end=float(e),
                    )
                    timed_by_key[(li, int(t["word_index"]))] = wt
                    filled.append(wt)
                    filled_count += 1

        # ---- Fill after last timed word
        missing_after = [
            t
            for t in toks
            if int(t["word_index"]) > last_idx and (li, int(t["word_index"])) not in timed_by_key
        ]
        if missing_after:
            missing_after.sort(key=lambda x: int(x["word_index"]))
            k = len(missing_after)
            last_w = timed_by_key[(li, last_idx)]
            window_start = float(last_w.end)

            nls = next_line_start(li)

            cap_end = window_start + max_edge_extend_sec
            if nls is not None:
                cap_end = min(cap_end, float(nls))

            need_end = window_start + (k * min_word_dur)
            if nls is not None:
                need_end = min(need_end, float(nls))

            window_end = max(cap_end, need_end)

            if (window_end - window_start) >= (k * min_word_dur) and window_end > window_start:
                step = (window_end - window_start) / k
                for i, t in enumerate(missing_after):
                    raw = str(t.get("raw", "")).strip()
                    if not raw:
                        continue
                    s = window_start + i * step
                    e = window_start + (i + 1) * step
                    if e <= s:
                        continue
                    wt = WordTimed(
                        line_index=int(li),
                        word_index=int(t["word_index"]),
                        text=raw,
                        matched_text="",
                        start=float(s),
                        end=float(e),
                    )
                    timed_by_key[(li, int(t["word_index"]))] = wt
                    filled.append(wt)
                    filled_count += 1

    if filled_count == 0:
        return words_timed, 0

    merged = list(words_timed) + filled
    merged.sort(key=lambda w: (int(w.line_index), float(w.start), int(w.word_index)))
    return merged, int(filled_count)


def _recompute_phrase_timings_from_words(phrases: List[Phrase], words_timed: List[WordTimed]) -> None:
    """Update Phrase.start/end from the current words_timed list (in-place)."""
    phrase_min: dict[int, Optional[float]] = {}
    phrase_max: dict[int, Optional[float]] = {}
    for w in words_timed:
        li = int(w.line_index)
        s = float(w.start)
        e = float(w.end)
        if li not in phrase_min or phrase_min[li] is None or s < float(phrase_min[li]):
            phrase_min[li] = s
        if li not in phrase_max or phrase_max[li] is None or e > float(phrase_max[li]):
            phrase_max[li] = e
    for p in phrases:
        p.start = phrase_min.get(int(p.line_index))
        p.end = phrase_max.get(int(p.line_index))


def _insert_missing_phrases_with_estimated_timings(
    phrases: List[Phrase],
    lyr_tokens: List[dict],
    words_timed: List[WordTimed],
    recognized_words: List[dict],
    min_word_dur: float = 0.060,
    max_gap_sec: float = 5.0,
) -> tuple[List[Phrase], List[WordTimed], int, int]:
    """Final salvage: insert phrases that have *no timed words*.

    If an entire lyrics line could not be aligned at all, the UI becomes hard
    to edit because the phrase has no timing. Here we insert synthetic timings
    for the whole phrase and all its words.

    Important:
    - We keep the *lyrics order* (line_index) and insert timings *between*
      the nearest timed neighbors (prev.end -> next.start).
    - If multiple consecutive phrases are missing, we allocate a single
      window for the whole run and split it in order. This prevents all
      missing phrases from collapsing to the same timestamp (often 0.0).
    """
    min_word_dur = max(0.010, float(min_word_dur))
    max_gap_sec = max(0.0, float(max_gap_sec))

    phrase_by_line: dict[int, Phrase] = {int(p.line_index): p for p in phrases}

    # Group lyric tokens by line, ordered by word_index (lyrics order)
    tokens_by_line: dict[int, List[dict]] = {}
    for t in lyr_tokens:
        li = int(t["line_index"])
        tokens_by_line.setdefault(li, []).append(t)
    for li in tokens_by_line:
        tokens_by_line[li].sort(key=lambda x: int(x["word_index"]))

    # Which phrases already have at least one timed word
    has_words = {int(w.line_index) for w in words_timed}

    # Track bounds from recognized words to handle global edge cases
    if recognized_words:
        try:
            track_start = float(min(float(w.get("start", 0.0) or 0.0) for w in recognized_words))
            track_end = float(max(float(w.get("end", 0.0) or 0.0) for w in recognized_words))
        except Exception:
            track_start = 0.0
            track_end = 0.0
    else:
        track_start = 0.0
        track_end = 0.0

    inserted_phrases = 0
    inserted_words = 0
    filled: List[WordTimed] = []

    def _nonempty_tokens_for_line(line_index: int) -> List[dict]:
        toks = tokens_by_line.get(int(line_index), [])
        return [t for t in toks if str(t.get("raw", "")).strip()]

    # Helper: find nearest timed neighbor bounds around a [start_i, end_i] run.
    def _neighbor_bounds_for_run(start_i: int, end_i: int) -> tuple[Optional[float], Optional[float]]:
        """Return (prev_end, next_start) around a missing run.

        Important nuance:
        If the immediate *next* phrase has a start time that is <= prev_end (overlap),
        using it would yield a non-positive gap and would push insertions to track start
        (0.0) via the allocator fallback. To avoid that, we keep looking forward for the
        first neighbor whose start is strictly after prev_end.
        """
        prev_end: Optional[float] = None
        for j in range(start_i - 1, -1, -1):
            pj = phrases[j]
            if pj.end is not None:
                prev_end = float(pj.end)
                break

        next_start: Optional[float] = None
        min_next = (float(prev_end) + 1e-3) if prev_end is not None else None
        for j in range(end_i + 1, len(phrases)):
            pj = phrases[j]
            if pj.start is None:
                continue
            s = float(pj.start)
            if (min_next is None) or (s > min_next):
                next_start = s
                break

        return prev_end, next_start

    # Helper: allocate a window for a run of missing phrases.
    def _allocate_window(prev_end: Optional[float], next_start: Optional[float], run_len: int) -> tuple[float, float]:
        if run_len <= 0:
            return (track_start, track_start)

        # Synthetic "budget" for the whole run.
        # With max_gap_sec enabled, we cap to max_gap_sec per phrase.
        desired_total = float(run_len) * float(max_gap_sec) if max_gap_sec > 0 else 0.0

        if prev_end is not None and next_start is not None and float(next_start) > float(prev_end):
            gap = float(next_start) - float(prev_end)
            total = gap if desired_total <= 0 else min(gap, desired_total)

            # Center inside the available gap (better UX than hugging the left bound).
            start = float(prev_end) + 0.5 * max(0.0, gap - total)
            end = start + total
            return (start, end)

        if prev_end is None and next_start is not None:
            end = float(next_start)
            total = desired_total if desired_total > 0 else max(0.5, end - track_start)
            start = max(track_start, end - total)
            return (start, end)

        if prev_end is not None and next_start is None:
            start = float(prev_end)
            total = desired_total if desired_total > 0 else max(0.5, max(track_end, start + 0.5) - start)
            end = min(max(track_end, start + 0.5), start + total) if track_end > 0 else (start + total)
            if end <= start:
                end = start + total
            return (start, end)

        # No neighbors (no timings at all)
        start = float(track_start)
        total = desired_total if desired_total > 0 else max(0.5, float(run_len) * 0.5)
        end = start + total
        return (start, end)

    # Helper: split the run window across phrases proportionally to token count.
    def _split_window_across_run(run_tokens: List[List[dict]], window_len: float) -> List[float]:
        mins = [max(min_word_dur * max(1, len(toks)), 0.010) for toks in run_tokens]
        total_min = sum(mins)

        if window_len <= 0:
            return [m for m in mins]

        if window_len >= total_min:
            rem = window_len - total_min
            weights = [max(1, len(toks)) for toks in run_tokens]
            wsum = float(sum(weights)) or 1.0
            extra = [rem * (w / wsum) for w in weights]
            return [mins[i] + extra[i] for i in range(len(mins))]

        scale = window_len / (total_min or 1.0)
        return [m * scale for m in mins]

    i = 0
    while i < len(phrases):
        p = phrases[i]
        li = int(p.line_index)

        # Only care about lines with no timed words but with actual lyrics tokens
        toks0 = _nonempty_tokens_for_line(li)
        if li in has_words or not toks0:
            i += 1
            continue

        # Start a run of consecutive missing phrases (lyrics order)
        run_start = i
        run_end = i

        run_lines: List[int] = [li]
        run_tokens: List[List[dict]] = [toks0]

        j = i + 1
        while j < len(phrases):
            pj = phrases[j]
            lij = int(pj.line_index)
            toksj = _nonempty_tokens_for_line(lij)
            if lij in has_words or not toksj:
                break
            run_end = j
            run_lines.append(lij)
            run_tokens.append(toksj)
            j += 1

        prev_end, next_start = _neighbor_bounds_for_run(run_start, run_end)
        win_start, win_end = _allocate_window(prev_end, next_start, len(run_lines))
        win_len = max(0.010, float(win_end) - float(win_start))

        per_phrase_dur = _split_window_across_run(run_tokens, win_len)

        cursor = float(win_start)
        for k, line_index in enumerate(run_lines):
            toks = run_tokens[k]
            if not toks:
                continue

            dur = max(0.010, float(per_phrase_dur[k]))
            start = float(cursor)
            end = float(cursor + dur)
            cursor = end

            # Create synthetic word timings inside [start, end]
            n = max(1, len(toks))
            step = (end - start) / float(n)

            for wi, t in enumerate(toks):
                raw = str(t.get("raw", "")).strip()
                if not raw:
                    continue
                ws = start + wi * step
                we = start + (wi + 1) * step
                if we <= ws:
                    continue
                filled.append(
                    WordTimed(
                        line_index=int(line_index),
                        word_index=int(t["word_index"]),
                        text=raw,
                        matched_text="",
                        start=float(ws),
                        end=float(we),
                    )
                )
                inserted_words += 1

            pp = phrase_by_line.get(int(line_index))
            if pp is not None:
                pp.start = float(start)
                pp.end = float(end)
                inserted_phrases += 1

        i = run_end + 1

    if filled:
        words_timed = list(words_timed) + filled
        words_timed.sort(key=lambda w: (int(w.line_index), float(w.start), int(w.word_index)))

    return phrases, words_timed, int(inserted_phrases), int(inserted_words)

def align_lyrics_to_words(
    lyrics_lines: List[str],
    recognized_words,
    max_search_window: int = 5,   # kept for API compatibility (unused here)
    min_similarity: float = 0.6,
    progress_cb: Optional[ProgressCallback] = None,
    passes: int = 2,
    strict_similarity: Optional[float] = None,
):
    """Align lyrics tokens to Whisper words with 1/2/3-pass strategy.

    Added heuristic:
    - Fill missing lyrics words BETWEEN two matched neighbors within the same line
      by interpolating timings (improves karaoke continuity).

    Returns
    -------
    phrases, words_timed, metrics
    """
    if progress_cb:
        progress_cb(55.0, "Aligning lyrics to recognized words (DP)...")

    min_similarity = max(0.0, min(1.0, float(min_similarity)))
    passes = int(passes)
    if passes < 1:
        passes = 1
    if passes > 3:
        passes = 3

    if strict_similarity is None:
        strict_similarity = min(0.92, max(0.70, min_similarity + 0.15))
    strict_similarity = max(0.0, min(1.0, float(strict_similarity)))

    salvage_similarity = max(0.35, min(min_similarity - 0.10, 0.55))

    lyr_tokens = _tokenize_lyrics_lines(lyrics_lines)
    rec_tokens = [{"idx": i, "norm": w.get("norm", ""), "raw": w.get("text", "")} for i, w in enumerate(recognized_words)]
    lyr_tokens = [t for t in lyr_tokens if t["norm"]]
    rec_tokens = [t for t in rec_tokens if t["norm"]]

    n_lines = len(lyrics_lines)
    n = len(lyr_tokens)
    m = len(rec_tokens)

    phrases: List[Phrase] = [Phrase(line_index=i, text=line, start=None, end=None) for i, line in enumerate(lyrics_lines)]
    words_timed: List[WordTimed] = []

    if n == 0 or m == 0:
        metrics = _compute_alignment_metrics(lyr_tokens, [], n_lines)
        if progress_cb:
            progress_cb(80.0, "Alignment done (empty tokens).")
        return phrases, words_timed, metrics

    lyr_norms = [t["norm"] for t in lyr_tokens]
    rec_norms = [t["norm"] for t in rec_tokens]

    cache = _SimilarityCache()

    GAP_LYR = -3.0
    GAP_REC = -2.0

    def path_to_matches(path, out_threshold: float):
        out = []
        for li, rj, sim in path:
            if li is None or rj is None:
                continue
            if sim >= out_threshold:
                out.append((li, rec_tokens[rj]["idx"], float(sim)))
        return out

    if progress_cb:
        progress_cb(58.0, f"DP pass 1/{passes} (score_min={min_similarity:.2f})...")

    base_path = _dp_global_align(
        lyr_norms,
        rec_norms,
        scoring_min_similarity=min_similarity,
        gap_lyr=GAP_LYR,
        gap_rec=GAP_REC,
        cache=cache,
    )

    if passes == 1:
        matches = path_to_matches(base_path, min_similarity)
    else:
        if progress_cb:
            progress_cb(62.0, f"DP anchors (strict={strict_similarity:.2f})...")

        anchor_path = _dp_global_align(
            lyr_norms,
            rec_norms,
            scoring_min_similarity=strict_similarity,
            gap_lyr=GAP_LYR,
            gap_rec=GAP_REC,
            cache=cache,
        )
        anchors = path_to_matches(anchor_path, strict_similarity)

        if len(anchors) < 3:
            matches = path_to_matches(base_path, min_similarity)
        else:
            anchors_sorted = sorted(anchors, key=lambda x: (x[0], x[1]))
            boundaries = [(-1, -1, 1.0)] + anchors_sorted + [(n, m, 1.0)]

            def align_chunk(l0, l1, r0, r1, scoring_min_sim):
                if l1 < l0 or r1 < r0:
                    return []
                sub_lyr = lyr_norms[l0:l1 + 1]
                sub_rec = rec_norms[r0:r1 + 1]
                sub_path = _dp_global_align(
                    sub_lyr,
                    sub_rec,
                    scoring_min_similarity=scoring_min_sim,
                    gap_lyr=GAP_LYR,
                    gap_rec=GAP_REC,
                    cache=cache,
                )
                out = []
                for li, rj, sim in sub_path:
                    if li is None or rj is None:
                        continue
                    abs_li = l0 + li
                    abs_rj = r0 + rj
                    if sim >= min_similarity:
                        out.append((abs_li, rec_tokens[abs_rj]["idx"], float(sim)))
                return out

            matches_map = {(li, rj): sim for li, rj, sim in anchors_sorted}

            if progress_cb:
                progress_cb(70.0, f"DP fill (lenient={min_similarity:.2f})...")
            for (l_prev, r_prev, _), (l_next, r_next, _) in zip(boundaries[:-1], boundaries[1:]):
                l0 = l_prev + 1
                l1 = l_next - 1
                r0 = r_prev + 1
                r1 = r_next - 1
                for li, rj, sim in align_chunk(l0, l1, r0, r1, scoring_min_sim=min_similarity):
                    matches_map[(li, rj)] = sim

            if passes >= 3:
                if progress_cb:
                    progress_cb(74.0, f"DP salvage (score_min={salvage_similarity:.2f})...")
                for (l_prev, r_prev, _), (l_next, r_next, _) in zip(boundaries[:-1], boundaries[1:]):
                    l0 = l_prev + 1
                    l1 = l_next - 1
                    r0 = r_prev + 1
                    r1 = r_next - 1
                    if l1 < l0 or r1 < r0:
                        continue
                    chunk_total = (l1 - l0 + 1)
                    chunk_matched = sum(1 for (li, _rj) in matches_map.keys() if l0 <= li <= l1)
                    if chunk_total <= 0:
                        continue
                    if (chunk_matched / chunk_total) >= 0.25:
                        continue
                    for li, rj, sim in align_chunk(l0, l1, r0, r1, scoring_min_sim=salvage_similarity):
                        matches_map[(li, rj)] = max(matches_map.get((li, rj), 0.0), sim)

            matches = [(li, rj, sim) for (li, rj), sim in matches_map.items()]
            matches.sort(key=lambda x: (x[0], x[1]))

            if len(matches) < max(5, int(0.20 * n)):
                matches = path_to_matches(base_path, min_similarity)

    # Convert matches to WordTimed + phrase timings (from matched words)
    phrase_min = {line_idx: None for line_idx in range(n_lines)}
    phrase_max = {line_idx: None for line_idx in range(n_lines)}

    for lyr_i, rec_idx, _sim in matches:
        t = lyr_tokens[lyr_i]
        rw = recognized_words[rec_idx]

        wt = WordTimed(
            line_index=int(t["line_index"]),
            word_index=int(t["word_index"]),
            text=t["raw"],
            matched_text=str(rw.get("text", "")),
            start=float(rw.get("start", 0.0)),
            end=float(rw.get("end", 0.0)),
        )
        words_timed.append(wt)

        li = int(t["line_index"])
        s = float(wt.start)
        e = float(wt.end)
        if phrase_min[li] is None or s < phrase_min[li]:
            phrase_min[li] = s
        if phrase_max[li] is None or e > phrase_max[li]:
            phrase_max[li] = e

    for p in phrases:
        p.start = phrase_min.get(p.line_index)
        p.end = phrase_max.get(p.line_index)

    # --- Heuristic 1: fill missing words BETWEEN matched neighbors (karaoke continuity)
    if progress_cb:
        progress_cb(78.0, "Filling missing words between matched neighbors...")

    words_timed, filled_between = _fill_missing_words_between_neighbors(
        lyr_tokens=lyr_tokens,
        words_timed=words_timed,
        min_word_dur=0.060,
    )

    # --- Heuristic 2: fill missing words at line EDGES (safe windows)
    if progress_cb:
        progress_cb(79.0, "Filling missing words at line edges (safe windows)...")

    words_timed, filled_edges = _fill_missing_words_at_line_edges(
        lyr_tokens=lyr_tokens,
        words_timed=words_timed,
        min_word_dur=0.060,
        max_edge_extend_sec=5.0,
    )

    # Update phrase timings from the (possibly) expanded / filled word timings.
    _recompute_phrase_timings_from_words(phrases, words_timed)

    # --- Final salvage: if a whole line has no timing at all, insert it anyway
    # (synthetic timings) so the editor always has something to work with.
    phrases, words_timed, inserted_phrases, inserted_words = _insert_missing_phrases_with_estimated_timings(
        phrases=phrases,
        lyr_tokens=lyr_tokens,
        words_timed=words_timed,
        recognized_words=recognized_words,
        min_word_dur=0.060,
        max_gap_sec=5.0,
    )

    # Recompute again after insertion
    _recompute_phrase_timings_from_words(phrases, words_timed)

    metrics = _compute_alignment_metrics(lyr_tokens, matches, n_lines)
    metrics.update(
        {
            "passes": int(passes),
            "min_similarity": float(min_similarity),
            "strict_similarity": float(strict_similarity),
            "lyrics_words_filled_between": int(filled_between),
            "lyrics_words_filled_edges": int(filled_edges),
            "phrases_inserted": int(inserted_phrases),
            "lyrics_words_inserted": int(inserted_words),
        }
    )

    if progress_cb:
        progress_cb(80.0, "Alignment done. Preparing outputs...")

    return phrases, words_timed, metrics

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
    """Map Whisper language code -> phonemizer language code.

    For ESpeak backend, Latin is typically 'la'.
    """
    if not lang:
        return "fr-fr"
    lang = str(lang).lower().strip()
    if lang in ("fr", "fr-fr"):
        return "fr-fr"
    if lang in ("en", "en-us", "en-gb"):
        return "en-us"
    if lang in ("de", "de-de"):
        return "de-de"
    if lang in ("la", "lat", "latin"):
        return "la"
    return "fr-fr"


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
    alignment_passes: int = 2,
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
    alignment_passes:
        1 = single DP pass, 2 = anchors+fill (recommended), 3 = anchors+fill+salvage.
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
    # Power-user feature: allow multiple language candidates (e.g. "en,la,auto")
    # and keep the run that aligns best to the provided lyrics.
    lang_candidates = _parse_language_candidates(whisper_language)

    best = None  # (score_tuple, metrics, segments, recognized_words, chosen_lang, phrases, words_timed)

    for idx_lang, lang in enumerate(lang_candidates):
        if progress_cb and len(lang_candidates) > 1:
            shown = (lang if lang is not None else "auto")
            progress_cb(3.0, f"Trying Whisper language candidate {idx_lang+1}/{len(lang_candidates)}: {shown}")

        segments, recognized_words = transcribe_with_whisper(
            audio_path,
            model_name=model_name,
            language=lang,
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

        # 3) Lyrics ↔ words alignment (multi-pass DP)
        phrases, words_timed, metrics = align_lyrics_to_words(
            lyrics_lines,
            recognized_words,
            max_search_window=max_search_window,
            min_similarity=min_similarity,
            progress_cb=progress_cb,
            passes=int(alignment_passes),  # 1/2/3 pass alignment
        )

        score = (float(metrics.get("coverage", 0.0)), float(metrics.get("mean_similarity", 0.0)))
        if best is None or score > best[0]:
            best = (score, metrics, segments, recognized_words, lang, phrases, words_timed)

    assert best is not None
    _score, metrics, segments, recognized_words, chosen_lang, phrases, words_timed = best


    # 4) Optional phonemes (for future usage)
    if not phoneme_language:
        phoneme_language = _infer_phoneme_lang_from_whisper_lang(chosen_lang)
    phonemes_timed = phonemize_words(
        words_timed,
        phoneme_lang=phoneme_language,
        progress_cb=progress_cb,
    )

    result = AlignmentResult(
        phrases=phrases,
        words=words_timed,
        phonemes=phonemes_timed,
        metrics=dict(metrics),
        chosen_whisper_language=(chosen_lang if chosen_lang is not None else "auto"),
    )
    # 5) Write JSON + SRT into the project's vocal_align folder
    align_dir = project.folder / "vocal_align"
    align_dir.mkdir(parents=True, exist_ok=True)

    # Debug exports: diagnose poor alignment (harsh vocals / choirs are hard).
    try:
        (align_dir / "recognized_words.json").write_text(
            json.dumps(recognized_words, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        recognized_text = " ".join([w.get("text", "").strip() for w in recognized_words if w.get("text", "").strip()])
        (align_dir / "recognized_text.txt").write_text(recognized_text, encoding="utf-8")
        (align_dir / "alignment_metrics.json").write_text(
            json.dumps(
                {
                    "chosen_whisper_language": chosen_lang if chosen_lang is not None else "auto",
                    "metrics": metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass

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
