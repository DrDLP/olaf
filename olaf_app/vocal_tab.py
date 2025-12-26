from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Any
import json
import shutil
import copy
from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QSettings
from PyQt6.QtGui import QColor, QPalette, QPainter, QPen, QBrush, QFontMetrics
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QComboBox,
    QProgressBar,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QCheckBox,
    QSpinBox,          
    QDoubleSpinBox,
    QApplication,
    QTabWidget,
    QSlider,
    QInputDialog,
    QLineEdit,
    QSizePolicy,
)

from .project_manager import Project
from .vocal_alignment import run_alignment_for_project

# torch is optional; we handle the case where it is not installed
try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class WordTimelineWidget(QWidget):
    """
    Visual editor for word timings inside a single phrase.

    UX goals:
    - Allow gaps ("holes") between words (do NOT auto-fill).
    - Edit each word independently (drag start or end handle).
    - Prevent overlaps with neighboring words (soft clamps).
    - Optional: drag phrase start/end range (does NOT clamp/move words).

    Playback / alignment helpers:
    - A playhead line is drawn when set_playhead() is used.
    - Word segments progressively "light up" as the playhead advances.

    Important: the horizontal time mapping uses a *local viewport* around the phrase
    (typically 2–5 seconds), clamped inside the neighbor-bounded allowed window
    (phrase_min..phrase_max). This keeps the editor readable while also avoiding
    degenerate behavior when dragging the outer handles (range collapsing).
    """

    timingsChanged = pyqtSignal(list)          # [(gidx, start, end), ...]
    phraseChanged = pyqtSignal(float, float)   # (phrase_start, phrase_end)
    wordSelected = pyqtSignal(object)          # global word idx or None
    playheadMoved = pyqtSignal(float)        # playhead time in seconds (user moved)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(170)
        self.setMouseTracking(True)

        # Phrase range (editable)
        self._phrase_idx: Optional[int] = None
        self._phrase_start: float = 0.0
        self._phrase_end: float = 1.0

        # Allowed phrase window (neighbor-bounded), used for the timeline mapping
        self._phrase_min: float = 0.0
        self._phrase_max: float = 1.0

        # View range used for mapping (typically equals phrase_min/max)
        self._view_start: float = 0.0
        self._view_end: float = 1.0

        # Words for current phrase (local copies)
        # Each: {"gidx": int, "text": str, "start": float, "end": float, "word_index": int}
        self._words: list[dict] = []

        # Playback helpers
        self._playhead_s: Optional[float] = None
        self._play_window: Optional[tuple[float, float]] = None

        # Selection / hover / drag state
        self._selected_local_idx: Optional[int] = None
        self._hover_handle: Optional[tuple[str, int]] = None  # ("w_start"/"w_end", local_idx) or ("p_start"/"p_end", -1)
        self._drag_handle: Optional[tuple[str, int]] = None
        self._dragging: bool = False

        # Constraints
        self._min_word_dur = 0.060
        self._min_phrase_span = 0.050
        self._handle_px = 6
        self._margin = 12


        # Viewport policy (readability):
        # Keep the visible window around the phrase within a small span (2–5s),
        # clamped to the allowed neighbor window (phrase_min..phrase_max).
        self._view_min_span_s = 2.0
        self._view_max_span_s = 5.0
        self._view_pad_s = 0.75
        # Render geometry
        self._ruler_y = 40
        self._bar_y = 72
        self._bar_h = 34

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_phrase(
        self,
        phrase_idx: int,
        phrase_start: float,
        phrase_end: float,
        words: list[dict],
        phrase_min: float,
        phrase_max: float,
    ) -> None:
        """Update the phrase and word list displayed by the editor."""
        self._phrase_idx = int(phrase_idx)
        self._phrase_start = float(phrase_start)
        self._phrase_end = float(phrase_end)

        # Allowed bounds (neighbor constraints)
        self._phrase_min = float(phrase_min)
        self._phrase_max = float(phrase_max)
        if self._phrase_max <= self._phrase_min + 1e-6:
            # Defensive: ensure a usable view span
            self._phrase_max = self._phrase_min + max(self._phrase_end - self._phrase_start, 0.25)

        # Defensive: avoid zero-length phrase
        if self._phrase_end <= self._phrase_start + 1e-6:
            self._phrase_end = self._phrase_start + 0.25

        # View range used for mapping:
        # Keep a small readable viewport around the phrase (2–5 seconds),
        # clamped inside the allowed neighbor window (phrase_min..phrase_max).
        phrase_dur = max(self._phrase_end - self._phrase_start, 0.0)
        # Ensure the viewport contains the full phrase. If the phrase itself is longer than the max,
        # we fall back to the phrase duration (cannot reasonably cap it).
        base_span = phrase_dur + 2.0 * float(self._view_pad_s)
        span = min(float(self._view_max_span_s), max(float(self._view_min_span_s), base_span))
        if phrase_dur > span:
            span = phrase_dur

        center = 0.5 * (self._phrase_start + self._phrase_end)
        view_start = center - 0.5 * span
        view_end = view_start + span

        # Clamp viewport to allowed bounds by shifting (preserve span when possible)
        if view_start < self._phrase_min:
            shift = self._phrase_min - view_start
            view_start += shift
            view_end += shift
        if view_end > self._phrase_max:
            shift = view_end - self._phrase_max
            view_start -= shift
            view_end -= shift

        # Final clamp (in case the allowed window is smaller than our desired span)
        view_start = max(self._phrase_min, view_start)
        view_end = min(self._phrase_max, view_end)
        if view_end <= view_start + 1e-6:
            # Defensive: ensure a usable view span
            view_start = self._phrase_start
            view_end = max(self._phrase_end, view_start + 1.0)

        self._view_start = float(view_start)
        self._view_end = float(view_end)

        # Keep a local copy, sorted by (start, word_index) for deterministic visuals
        self._words = sorted(
            [dict(w) for w in (words or [])],
            key=lambda x: (float(x.get("start", 0.0)), int(x.get("word_index", 0))),
        )

        # Reset selection if out of range
        if self._selected_local_idx is not None and not (0 <= self._selected_local_idx < len(self._words)):
            self._selected_local_idx = None
            self.wordSelected.emit(None)

        self.update()

    def set_selected_global_idx(self, gidx: Optional[int]) -> None:
        """Select a word by its global index (used by parent to sync selection)."""
        if gidx is None:
            self._selected_local_idx = None
            self.wordSelected.emit(None)
            self.update()
            return

        for i, w in enumerate(self._words):
            if int(w.get("gidx", -1)) == int(gidx):
                self._selected_local_idx = i
                self.wordSelected.emit(int(gidx))
                self.update()
                return

        self._selected_local_idx = None
        self.wordSelected.emit(None)
        self.update()

    def get_selected_global_idx(self) -> Optional[int]:
        """Return selected global word index or None."""
        if self._selected_local_idx is None:
            return None
        if 0 <= self._selected_local_idx < len(self._words):
            return int(self._words[self._selected_local_idx].get("gidx", -1))
        return None

    def set_playhead(self, playhead_s: Optional[float]) -> None:
        """
        Set the current playhead time (seconds) for the visual helper.
        This does NOT move any timings: it only affects painting.
        """
        if playhead_s is None:
            self._playhead_s = None
            self.update()
            return
        self._playhead_s = float(playhead_s)
        self.update()

    def get_playhead(self) -> Optional[float]:
        """Return the current playhead time in seconds (or None)."""
        return None if self._playhead_s is None else float(self._playhead_s)

    def set_play_window(self, start_s: Optional[float], end_s: Optional[float]) -> None:
        """
        Optional: show a (start, end) playback window on the timeline.
        Useful when playing a phrase with pre/post-roll and/or looping.
        """
        if start_s is None or end_s is None:
            self._play_window = None
            self.update()
            return
        a = float(start_s)
        b = float(end_s)
        if b < a:
            a, b = b, a
        self._play_window = (a, b)
        self.update()

    # ------------------------------------------------------------------
    # Mapping helpers (based on view_start/view_end)
    # ------------------------------------------------------------------

    def _time_to_x(self, t: float) -> float:
        w = max(self.width() - 2 * self._margin, 10)
        dur = max(self._view_end - self._view_start, 1e-6)
        return self._margin + (float(t) - self._view_start) / dur * w

    def _x_to_time(self, x: float) -> float:
        w = max(self.width() - 2 * self._margin, 10)
        dur = max(self._view_end - self._view_start, 1e-6)
        u = (float(x) - self._margin) / w
        u = max(0.0, min(1.0, u))
        return self._view_start + u * dur

    def _bar_rect(self) -> tuple[int, int, int, int]:
        x0 = self._margin
        x1 = self.width() - self._margin
        return (x0, self._bar_y, x1 - x0, self._bar_h)

    def _choose_tick_step(self, dur: float) -> float:
        """
        Pick a nice tick step to get a readable ruler (roughly 6-12 ticks).
        """
        dur = max(float(dur), 1e-6)
        candidates = [0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0]
        target = dur / 9.0
        best = candidates[0]
        best_err = abs(best - target)
        for c in candidates[1:]:
            err = abs(c - target)
            if err < best_err:
                best = c
                best_err = err
        return best

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        bar_x, bar_y, bar_w, bar_h = self._bar_rect()
        bar_x0 = bar_x
        bar_x1 = bar_x + bar_w

        # Phrase header
        p.setPen(QPen(QColor(200, 200, 200, 220), 1))
        p.drawText(
            self._margin,
            22,
            f"Phrase range: {self._phrase_start:0.3f} → {self._phrase_end:0.3f}  (drag word start/end handles; gaps allowed)",
        )

        # Ruler (ticks) across the allowed window
        view_dur = max(self._view_end - self._view_start, 1e-6)
        tick_step = self._choose_tick_step(view_dur)

        p.setPen(QPen(QColor(140, 140, 140, 130), 1))
        p.drawLine(int(bar_x0), self._ruler_y, int(bar_x1), self._ruler_y)

        fm = QFontMetrics(p.font())
        n_ticks = int(view_dur / tick_step) + 1
        for k in range(n_ticks + 1):
            t = self._view_start + k * tick_step
            if t > self._view_end + 1e-6:
                break
            x = self._time_to_x(t)
            p.setPen(QPen(QColor(140, 140, 140, 130), 1))
            p.drawLine(int(x), self._ruler_y - 4, int(x), self._ruler_y + 4)

            # Show labels relative to phrase start (may be negative on the left side)
            label = f"{(t - self._phrase_start):+0.2f}"
            tw = fm.horizontalAdvance(label)
            p.setPen(QPen(QColor(170, 170, 170, 160), 1))
            p.drawText(int(x - tw / 2), self._ruler_y - 8, label)

        # Timeline background (allowed window)
        p.setPen(QPen(QColor(120, 120, 120, 160), 1))
        p.setBrush(QBrush(QColor(80, 80, 80, 50)))
        p.drawRoundedRect(bar_x0, bar_y, bar_w, bar_h, 6, 6)

        # Phrase window overlay (where words are expected to live)
        ps = max(self._view_start, min(self._phrase_start, self._view_end))
        pe = max(self._view_start, min(self._phrase_end, self._view_end))
        if pe < ps:
            ps, pe = pe, ps

        psx = self._time_to_x(ps)
        pex = self._time_to_x(pe)

        if pex > psx + 1:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(0, 180, 255, 18)))
            p.drawRoundedRect(int(psx), bar_y + 1, max(int(pex - psx), 2), bar_h - 2, 6, 6)

        # Optional play window overlay (pre/post-roll)
        if self._play_window is not None:
            a, b = self._play_window
            a = max(self._view_start, min(a, self._view_end))
            b = max(self._view_start, min(b, self._view_end))
            if b > a + 1e-6:
                ax = self._time_to_x(a)
                bx = self._time_to_x(b)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(QColor(0, 255, 255, 22)))
                p.drawRoundedRect(int(ax), bar_y + 1, max(int(bx - ax), 2), bar_h - 2, 6, 6)

        # Phrase handles at actual phrase start/end (not at bar extremes)
        # Make them visually distinct from word handles (taller + double line).
        ph_top = bar_y - 14
        ph_bot = bar_y + bar_h + 14
        ph_gap = 3

        for xh, kind in ((psx, "p_start"), (pex, "p_end")):
            is_hover = (self._hover_handle == (kind, -1))
            col = QColor(255, 200, 80, 240) if is_hover else QColor(240, 240, 240, 220)
            w = 3 if is_hover else 2
            p.setPen(QPen(col, w))
            # Double line (two close parallels)
            p.drawLine(int(xh - ph_gap), ph_top, int(xh - ph_gap), ph_bot)
            p.drawLine(int(xh + ph_gap), ph_top, int(xh + ph_gap), ph_bot)

        # Draw word segments (with playhead "illumination")
        playhead = self._playhead_s
        for i, w in enumerate(self._words):
            ws = float(w.get("start", self._phrase_start))
            we = float(w.get("end", ws))

            # For display only: clamp inside view
            ws_d = max(self._view_start, min(ws, self._view_end))
            we_d = max(self._view_start, min(we, self._view_end))
            if we_d < ws_d:
                we_d = ws_d

            x0 = self._time_to_x(ws_d)
            x1 = self._time_to_x(we_d)

            is_selected = (self._selected_local_idx == i)

            # Time-based illumination
            alpha = 70
            border_alpha = 190
            progress_u = 0.0
            is_current = False
            if playhead is not None:
                if playhead >= we:
                    alpha = 140
                    border_alpha = 220
                elif ws <= playhead < we:
                    alpha = 190
                    border_alpha = 240
                    is_current = True
                    denom = max(we - ws, 1e-6)
                    progress_u = max(0.0, min(1.0, (playhead - ws) / denom))
                else:
                    alpha = 55
                    border_alpha = 160

            # Selection still wins
            if is_selected:
                p.setPen(QPen(QColor(30, 140, 255, 240), 2))
                p.setBrush(QBrush(QColor(30, 140, 255, max(alpha, 110))))
            else:
                p.setPen(QPen(QColor(30, 140, 255, border_alpha), 1))
                p.setBrush(QBrush(QColor(30, 140, 255, alpha)))

            p.drawRoundedRect(int(x0), bar_y + 2, max(int(x1 - x0), 2), bar_h - 4, 4, 4)

            # Current-word progress overlay (sub-fill)
            if is_current and (x1 - x0) > 2:
                px1 = x0 + (x1 - x0) * progress_u
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(QColor(255, 255, 255, 35)))
                p.drawRoundedRect(int(x0), bar_y + 2, max(int(px1 - x0), 2), bar_h - 4, 4, 4)

            # Start/end handles for this word
            sx = self._time_to_x(ws_d)
            ex = self._time_to_x(we_d)
            start_col = QColor(255, 200, 80, 240) if self._hover_handle == ("w_start", i) else QColor(220, 220, 220, 160)
            end_col = QColor(255, 200, 80, 240) if self._hover_handle == ("w_end", i) else QColor(220, 220, 220, 160)

            p.setPen(QPen(start_col, 2 if self._hover_handle == ("w_start", i) else 1))
            p.drawLine(int(sx), bar_y - 6, int(sx), bar_y + bar_h + 6)

            p.setPen(QPen(end_col, 2 if self._hover_handle == ("w_end", i) else 1))
            p.drawLine(int(ex), bar_y - 6, int(ex), bar_y + bar_h + 6)


            # In-segment drag affordances: grab << (start) or >> (end) inside the segment.
            # This helps when multiple handles overlap (end of previous == start of next).
            seg_w = float(x1 - x0)
            icon_w = 18
            icon_h = 14
            icon_y = int(bar_y + (bar_h - icon_h) / 2)

            if seg_w >= (2 * icon_w + 10):
                # Start icon rect
                sx0 = int(x0) + 3
                sx1 = sx0 + icon_w
                ex1 = int(x1) - 3
                ex0 = ex1 - icon_w

                # Start icon
                start_hover = (self._hover_handle == ("w_start", i))
                pen_col = QColor(255, 200, 80, 240) if start_hover else QColor(255, 255, 255, 140)
                p.setPen(QPen(pen_col, 1))
                p.setBrush(QBrush(QColor(0, 0, 0, 70)))
                p.drawRoundedRect(sx0, icon_y, icon_w, icon_h, 4, 4)
                p.drawText(sx0 + 3, icon_y + icon_h - 3, "<<")

                # End icon
                end_hover = (self._hover_handle == ("w_end", i))
                pen_col = QColor(255, 200, 80, 240) if end_hover else QColor(255, 255, 255, 140)
                p.setPen(QPen(pen_col, 1))
                p.setBrush(QBrush(QColor(0, 0, 0, 70)))
                p.drawRoundedRect(ex0, icon_y, icon_w, icon_h, 4, 4)
                p.drawText(ex0 + 4, icon_y + icon_h - 3, ">>")

            # Word label (centered on the segment, with elide if needed)
            txt = str(w.get("text", "")).strip()
            if txt:
                # Define a safe text area inside the segment.
                # If the in-segment icons (<< / >>) are visible, keep the text away from them.
                seg_left = float(x0) + 3.0
                seg_right = float(x1) - 3.0

                if seg_w >= (2 * icon_w + 10):
                    seg_left = float(x0) + 3.0 + float(icon_w) + 6.0
                    seg_right = float(x1) - 3.0 - float(icon_w) - 6.0

                max_w = max(int(seg_right - seg_left), 0)
                label = fm.elidedText(txt, Qt.TextElideMode.ElideRight, max_w)

                tw = fm.horizontalAdvance(label)
                tx = seg_left + max(0.0, (seg_right - seg_left - tw) / 2.0)

                # Vertically center inside the bar. drawText() expects a baseline (not a top-left).
                ty = int(bar_y + (bar_h + fm.ascent() - fm.descent()) / 2)

                p.setPen(QPen(QColor(240, 240, 240, 230), 1))
                p.drawText(int(tx), int(ty), label)

        # Playhead line + small label
        if playhead is not None:
            t = max(self._view_start, min(float(playhead), self._view_end))
            x = self._time_to_x(t)

            p.setPen(QPen(QColor(255, 255, 255, 190), 2))
            p.drawLine(int(x), self._ruler_y - 2, int(x), bar_y + bar_h + 10)

            rel = float(playhead) - float(self._phrase_start)
            badge = f"{rel:+0.3f}s"
            tw = fm.horizontalAdvance(badge)
            bx = int(x - tw / 2) - 4
            by = self._ruler_y + 8
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(0, 0, 0, 120)))
            p.drawRoundedRect(bx, by, tw + 8, 18, 6, 6)
            p.setPen(QPen(QColor(255, 255, 255, 220), 1))
            p.drawText(bx + 4, by + 13, badge)

        p.end()

    # ------------------------------------------------------------------
    # Hit-testing helpers
    # ------------------------------------------------------------------

    def _hit_test_handle(self, x: float, y: float) -> Optional[tuple[str, int]]:
        """Return which handle is under the mouse, if any.

        Priority:
          1) Phrase edges
          2) In-segment icons (<< / >>) to disambiguate overlapping word handles
          3) Raw handle lines
        """
        if not (self._bar_y - 20 <= y <= self._bar_y + self._bar_h + 20):
            return None

        # Phrase edges (at phrase start/end positions)
        psx = self._time_to_x(self._phrase_start)
        pex = self._time_to_x(self._phrase_end)
        if abs(x - psx) <= self._handle_px:
            return ("p_start", -1)
        if abs(x - pex) <= self._handle_px:
            return ("p_end", -1)

        # Word start/end handles (icons first, then raw lines)
        icon_w = 18
        icon_h = 14
        icon_y0 = self._bar_y + int((self._bar_h - icon_h) / 2)
        icon_y1 = icon_y0 + icon_h

        for i, w in enumerate(self._words):
            ws = float(w.get("start", self._phrase_start))
            we = float(w.get("end", ws))

            # Clamp to current viewport for hit-testing / drawing coherence
            ws_d = max(self._view_start, min(ws, self._view_end))
            we_d = max(self._view_start, min(we, self._view_end))

            x0 = self._time_to_x(min(ws_d, we_d))
            x1 = self._time_to_x(max(ws_d, we_d))

            # Prefer clicking on the icons inside the segment
            if (x1 - x0) >= (2 * icon_w + 10) and (icon_y0 <= y <= icon_y1):
                sx0 = int(x0) + 3
                sx1 = sx0 + icon_w
                ex1 = int(x1) - 3
                ex0 = ex1 - icon_w

                if sx0 <= x <= sx1:
                    return ("w_start", i)
                if ex0 <= x <= ex1:
                    return ("w_end", i)

            # Raw handle lines
            sx = self._time_to_x(ws_d)
            ex = self._time_to_x(we_d)
            if abs(x - sx) <= self._handle_px:
                return ("w_start", i)
            if abs(x - ex) <= self._handle_px:
                return ("w_end", i)

        return None

        # Phrase edges (at phrase start/end positions)
        psx = self._time_to_x(self._phrase_start)
        pex = self._time_to_x(self._phrase_end)
        if abs(x - psx) <= self._handle_px:
            return ("p_start", -1)
        if abs(x - pex) <= self._handle_px:
            return ("p_end", -1)

        # Word start/end handles
        for i, w in enumerate(self._words):
            ws = float(w.get("start", self._phrase_start))
            we = float(w.get("end", ws))
            ws_d = max(self._view_start, min(ws, self._view_end))
            we_d = max(self._view_start, min(we, self._view_end))
            sx = self._time_to_x(ws_d)
            ex = self._time_to_x(we_d)
            if abs(x - sx) <= self._handle_px:
                return ("w_start", i)
            if abs(x - ex) <= self._handle_px:
                return ("w_end", i)

        return None

    def _hit_test_word(self, x: float, y: float) -> Optional[int]:
        """Return local word index if a word segment is clicked, else None."""
        if not (self._bar_y <= y <= self._bar_y + self._bar_h):
            return None

        for i, w in enumerate(self._words):
            ws = float(w.get("start", self._phrase_start))
            we = float(w.get("end", ws))
            ws_d = max(self._view_start, min(ws, self._view_end))
            we_d = max(self._view_start, min(we, self._view_end))
            x0 = self._time_to_x(ws_d)
            x1 = self._time_to_x(we_d)
            if x0 <= x <= x1:
                return i
        return None

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def _set_playhead_from_x(self, x: float) -> None:
        """Move playhead from an x coordinate and notify listeners."""
        t = float(self._x_to_time(float(x)))
        self.set_playhead(t)
        try:
            self.playheadMoved.emit(float(t))
        except Exception:
            pass

    def mouseMoveEvent(self, event):  # noqa: N802
        x = float(event.position().x())
        y = float(event.position().y())

        self._hover_handle = self._hit_test_handle(x, y)

        if self._dragging and self._drag_handle is not None:
            kind, idx = self._drag_handle
            if kind == "playhead":
                self._set_playhead_from_x(x)
                self.update()
                return
            if kind in ("w_start", "w_end"):
                self._apply_word_handle_drag(kind, idx, x)
                self.update()
                return
            if kind in ("p_start", "p_end"):
                self._apply_phrase_edge_drag(kind, x)
                self.update()
                return

        self.update()

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return

        x = float(event.position().x())
        y = float(event.position().y())

        self._dragging = False
        self._drag_handle = None

        # 1) Handles have priority (word/phrase edges)
        handle = self._hit_test_handle(x, y)
        if handle is not None:
            self._dragging = True
            self._drag_handle = handle
            return

        # 2) Clicking a word selects it (no timing change)
        widx = self._hit_test_word(x, y)
        if widx is not None:
            self._selected_local_idx = int(widx)
            gidx = int(self._words[widx].get("gidx", -1))
            self.wordSelected.emit(gidx if gidx >= 0 else None)
            self.update()
            return

        # 3) Clicking empty timeline moves the playhead (and can be dragged)
        in_timeline_band = (self._ruler_y - 18) <= y <= (self._bar_y + self._bar_h + 18)
        if in_timeline_band:
            self._dragging = True
            self._drag_handle = ("playhead", -1)
            self._set_playhead_from_x(x)
            self.update()
            return

        # 4) Click outside clears selection
        if self._selected_local_idx is not None:
            self._selected_local_idx = None
            self.wordSelected.emit(None)
            self.update()

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._dragging and self._drag_handle is not None:
            kind, _idx = self._drag_handle
            if kind in ("w_start", "w_end"):
                self._emit_word_timings()
            elif kind in ("p_start", "p_end"):
                self.phraseChanged.emit(float(self._phrase_start), float(self._phrase_end))

        self._dragging = False
        self._drag_handle = None
        self.update()


    # ------------------------------------------------------------------
    # Drag application logic
    # ------------------------------------------------------------------

    def _apply_word_handle_drag(self, kind: str, local_idx: int, x: float) -> None:
        """Drag a word start or end handle without forcing gaps to close."""
        if local_idx < 0 or local_idx >= len(self._words):
            return

        t = self._x_to_time(x)

        # Clamp inside phrase range
        t = max(self._phrase_start, min(t, self._phrase_end))

        w = self._words[local_idx]
        cur_s = float(w.get("start", self._phrase_start))
        cur_e = float(w.get("end", cur_s))

        # Neighbor overlap prevention (allow gaps)
        prev_end = None
        next_start = None
        if local_idx > 0:
            prev_end = float(self._words[local_idx - 1].get("end", self._phrase_start))
        if local_idx + 1 < len(self._words):
            next_start = float(self._words[local_idx + 1].get("start", self._phrase_end))

        if kind == "w_start":
            t_max = cur_e - self._min_word_dur
            if prev_end is not None:
                t = max(t, prev_end)
            t = min(t, t_max)
            w["start"] = float(t)

        elif kind == "w_end":
            t_min = cur_s + self._min_word_dur
            if next_start is not None:
                t = min(t, next_start)
            t = max(t, t_min)
            w["end"] = float(t)

        # Final safety
        s2 = float(w.get("start", cur_s))
        e2 = float(w.get("end", cur_e))
        if e2 < s2 + self._min_word_dur:
            w["end"] = float(s2 + self._min_word_dur)

    def _apply_phrase_edge_drag(self, kind: str, x: float) -> None:
        """
        Drag phrase start/end; does NOT clamp/move words (prevents agglutination).

        Uses the allowed window mapping (phrase_min..phrase_max), so the range
        will not collapse to the minimal span when adjusting edges.
        """
        t = self._x_to_time(x)

        if kind == "p_start":
            t = max(self._phrase_min, min(t, self._phrase_end - self._min_phrase_span))
            self._phrase_start = float(t)
        elif kind == "p_end":
            t = min(self._phrase_max, max(t, self._phrase_start + self._min_phrase_span))
            self._phrase_end = float(t)

        # Keep phrase inside view window
        if self._phrase_end <= self._phrase_start + 1e-6:
            self._phrase_end = self._phrase_start + max(self._min_phrase_span, 0.10)

    def _emit_word_timings(self) -> None:
        """Emit updated timings for parent to apply to the global word list."""
        payload = []
        for w in self._words:
            gidx = w.get("gidx", None)
            if gidx is None:
                continue
            payload.append((int(gidx), float(w.get("start", 0.0)), float(w.get("end", 0.0))))
        self.timingsChanged.emit(payload)

class VocalTab(QWidget):
    """
    Vocal tab:
    - Manages lyrics (stored inside the project)
    - Runs Whisper-based alignment (phrases + words)
    - Lets the user edit phrase timings
    - Saves updated timings to JSON + SRT
    - Provides a simple karaoke preview driven by the shared QMediaPlayer
    """
    audioStarted = pyqtSignal(str)

    def __init__(self, player: QMediaPlayer, parent=None):
        super().__init__(parent)
        self.player = player
        self._project: Optional[Project] = None

        # Alignment data (generic: dicts or dataclasses)
        self._phrases: List[Any] = []
        self._words: List[Any] = []
        # Snapshots for "Reset" (last saved / loaded state)
        self._saved_phrases_snapshot: List[Any] = []
        self._saved_words_snapshot: List[Any] = []

        # Phrase selection sync between Numeric and Visual subtabs
        self._syncing_phrase_selection: bool = False
        self._current_phrase_idx: Optional[int] = None

        # Visual editor playback loop state
        self._visual_play_active: bool = False
        self._visual_loop_enabled: bool = False
        self._visual_loop_start_s: float = 0.0
        self._visual_loop_end_s: float = 0.0
        # Previous playback rate before entering visual slow-motion
        self._visual_prev_playback_rate: Optional[float] = None

        
        # Cached media duration in milliseconds (for sliders)
        self._media_duration_ms: int = 0
       

        # GPU detection state
        self._gpu_available: bool = False
        self._gpu_reason: str = ""

        self._build_ui()
        self._detect_gpu()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_visual_alignment_tab(self) -> QWidget:
        """Build the 'Lyrics alignment (visual)' sub-tab."""
        tab = QWidget(self)
        root = QHBoxLayout(tab)

        # Left column: phrase list + playback
        left = QVBoxLayout()
        left.addWidget(QLabel("Phrases:", tab))

        self.visual_phrase_list = QListWidget(tab)
        self.visual_phrase_list.setMinimumWidth(360)
        left.addWidget(self.visual_phrase_list, 1)

        play_group = QGroupBox("Phrase playback", tab)
        pg = QVBoxLayout(play_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Pre-roll (s):", play_group))
        self.spin_visual_preroll = QDoubleSpinBox(play_group)
        self.spin_visual_preroll.setRange(0.0, 10.0)
        self.spin_visual_preroll.setDecimals(2)
        self.spin_visual_preroll.setSingleStep(0.25)
        self.spin_visual_preroll.setValue(0.50)
        row1.addWidget(self.spin_visual_preroll)

        row1.addWidget(QLabel("Post-roll (s):", play_group))
        self.spin_visual_postroll = QDoubleSpinBox(play_group)
        self.spin_visual_postroll.setRange(0.0, 10.0)
        self.spin_visual_postroll.setDecimals(2)
        self.spin_visual_postroll.setSingleStep(0.25)
        self.spin_visual_postroll.setValue(0.50)
        row1.addWidget(self.spin_visual_postroll)
        row1.addStretch(1)
        pg.addLayout(row1)


        row_speed = QHBoxLayout()
        row_speed.addWidget(QLabel("Speed:", play_group))
        self.combo_visual_speed = QComboBox(play_group)
        self.combo_visual_speed.addItem("1.00x", 1.0)
        self.combo_visual_speed.addItem("0.75x", 0.75)
        self.combo_visual_speed.addItem("0.50x", 0.50)
        self.combo_visual_speed.addItem("0.33x", 1.0 / 3.0)
        self.combo_visual_speed.addItem("0.25x", 0.25)
        self.combo_visual_speed.setCurrentIndex(0)
        row_speed.addWidget(self.combo_visual_speed)
        row_speed.addStretch(1)
        pg.addLayout(row_speed)

        row2 = QHBoxLayout()
        self.chk_visual_loop = QCheckBox("Loop", play_group)
        self.chk_visual_loop.setChecked(False)
        row2.addWidget(self.chk_visual_loop)

        self.btn_visual_play = QPushButton("Play phrase", play_group)
        self.btn_visual_stop = QPushButton("Stop", play_group)
        row2.addWidget(self.btn_visual_play)
        row2.addWidget(self.btn_visual_stop)
        row2.addStretch(1)
        pg.addLayout(row2)

        play_group.setLayout(pg)
        left.addWidget(play_group, 0)

        root.addLayout(left, 0)

        # Right column: timeline + word editor
        right = QVBoxLayout()

        tl_group = QGroupBox("Visual timing editor", tab)
        tl_layout = QVBoxLayout(tl_group)

        self.lbl_visual_playhead = QLabel("Playhead: --", tl_group)
        tl_layout.addWidget(self.lbl_visual_playhead)

        self.visual_timeline = WordTimelineWidget(tl_group)
        tl_layout.addWidget(self.visual_timeline, 1)

        tl_help = QLabel(
            "Tip: drag a word's START or END handle independently (gaps are allowed).\n"
            "Drag the outermost phrase edges only if you need to adjust the phrase range.",
            tl_group,
        )
        tl_help.setWordWrap(True)
        tl_layout.addWidget(tl_help)

        tl_group.setLayout(tl_layout)
        right.addWidget(tl_group, 1)

        editor_group = QGroupBox("Selected word", tab)
        eg = QVBoxLayout(editor_group)

        row_txt = QHBoxLayout()
        row_txt.addWidget(QLabel("Text:", editor_group))
        self.edit_visual_word_text = QLineEdit(editor_group)
        row_txt.addWidget(self.edit_visual_word_text, 1)
        eg.addLayout(row_txt)

        row_t = QHBoxLayout()
        row_t.addWidget(QLabel("Start (s):", editor_group))
        self.spin_visual_word_start = QDoubleSpinBox(editor_group)
        self.spin_visual_word_start.setRange(0.0, 99999.0)
        self.spin_visual_word_start.setDecimals(3)
        self.spin_visual_word_start.setSingleStep(0.01)
        row_t.addWidget(self.spin_visual_word_start)

        row_t.addWidget(QLabel("End (s):", editor_group))
        self.spin_visual_word_end = QDoubleSpinBox(editor_group)
        self.spin_visual_word_end.setRange(0.0, 99999.0)
        self.spin_visual_word_end.setDecimals(3)
        self.spin_visual_word_end.setSingleStep(0.01)
        row_t.addWidget(self.spin_visual_word_end)

        row_t.addStretch(1)
        eg.addLayout(row_t)

        row_btn = QHBoxLayout()
        self.btn_visual_apply_word = QPushButton("Apply", editor_group)
        self.btn_visual_insert_word = QPushButton("Insert word…", editor_group)
        self.btn_visual_delete_word = QPushButton("Delete word", editor_group)
        row_btn.addWidget(self.btn_visual_apply_word)
        row_btn.addWidget(self.btn_visual_insert_word)
        row_btn.addWidget(self.btn_visual_delete_word)
        row_btn.addStretch(1)
        eg.addLayout(row_btn)

        editor_group.setLayout(eg)
        right.addWidget(editor_group, 0)

        bottom = QHBoxLayout()
        self.btn_visual_save = QPushButton("Save timings", tab)
        self.btn_visual_reset = QPushButton("Reset (last saved)", tab)
        bottom.addWidget(self.btn_visual_save)
        bottom.addWidget(self.btn_visual_reset)
        bottom.addStretch(1)
        right.addLayout(bottom)

        root.addLayout(right, 1)

        # Connections (visual tab)
        self.visual_phrase_list.currentItemChanged.connect(self._on_visual_phrase_selected)
        self.visual_phrase_list.itemClicked.connect(self._on_visual_phrase_clicked)

        self.visual_timeline.timingsChanged.connect(self._on_visual_word_timings_changed)
        self.visual_timeline.playheadMoved.connect(self._on_visual_timeline_playhead_moved)
        self.visual_timeline.phraseChanged.connect(self._on_visual_phrase_range_changed)
        self.visual_timeline.wordSelected.connect(self._on_visual_word_selected)

        self.btn_visual_play.clicked.connect(self._play_current_visual_phrase)
        self.btn_visual_stop.clicked.connect(self._stop_visual_phrase)
        self.chk_visual_loop.toggled.connect(self._on_visual_loop_toggled)
        self.btn_visual_apply_word.clicked.connect(self._apply_visual_word_edit)
        self.btn_visual_insert_word.clicked.connect(self._insert_visual_word)
        self.btn_visual_delete_word.clicked.connect(self._delete_visual_word)

        self.btn_visual_save.clicked.connect(self.save_timings_to_files)
        self.btn_visual_reset.clicked.connect(self._reset_alignment_to_saved_snapshot)

        tab.setLayout(root)
        return tab


    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # Project label is now hidden; the active project is shown
        # globally in the bottom player bar.
        self.lbl_project = QLabel("", self)
        self.lbl_project.setVisible(False)
        main_layout.addWidget(self.lbl_project)

        # Sub-tabs inside the Vocal tab
        self.sub_tabs = QTabWidget(self)
        main_layout.addWidget(self.sub_tabs)

        # --------------------------------------------------------------
        # Tab 1: Lyrics alignment (lyrics + alignment settings)
        # --------------------------------------------------------------
        tab_lyrics = QWidget(self)

        # Two columns:
        # - Left: Lyrics editor (full height)
        # - Right: Everything else (alignment settings)
        tab_lyrics_layout = QHBoxLayout(tab_lyrics)
        tab_lyrics_layout.setContentsMargins(0, 0, 0, 0)
        tab_lyrics_layout.setSpacing(10)

        # -------------------------
        # LEFT COLUMN: Lyrics
        # -------------------------
        left_col = QWidget(tab_lyrics)
        left_layout = QVBoxLayout(left_col)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        lyrics_group = QGroupBox("Lyrics", tab_lyrics)
        lyrics_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lg_layout = QVBoxLayout(lyrics_group)

        btn_row = QHBoxLayout()
        self.btn_load_lyrics = QPushButton("Load .txt…", lyrics_group)
        self.btn_save_lyrics = QPushButton("Save .txt…", lyrics_group)
        btn_row.addWidget(self.btn_load_lyrics)
        btn_row.addWidget(self.btn_save_lyrics)
        btn_row.addStretch(1)
        lg_layout.addLayout(btn_row)

        self.lyrics_edit = QTextEdit(lyrics_group)
        self.lyrics_edit.setPlaceholderText(
            "One line per phrase (verse). These lyrics are stored inside the project."
        )
        self.lyrics_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Make the editor take all remaining vertical space
        lg_layout.addWidget(self.lyrics_edit, 1)

        lyrics_group.setLayout(lg_layout)
        left_layout.addWidget(lyrics_group, 1)
        left_col.setLayout(left_layout)

        # -------------------------
        # RIGHT COLUMN: Settings
        # -------------------------
        right_col = QWidget(tab_lyrics)
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # ----- Alignment settings group -----
        settings_group = QGroupBox("Alignment settings", tab_lyrics)
        sg_layout = QVBoxLayout(settings_group)

        # Model selection
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Whisper model:", settings_group))
        self.model_combo = QComboBox(settings_group)
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large-v2", "large-v3"])
        self.model_combo.setCurrentText("medium")
        model_row.addWidget(self.model_combo)
        model_row.addStretch(1)
        sg_layout.addLayout(model_row)

        # Language
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Language:", settings_group))
        self.lang_combo = QComboBox(settings_group)

        # First entry: auto-detect (None => Whisper auto)
        self.lang_combo.addItem("Auto (detect language)", userData=None)

        # Explicit languages (data = Whisper language code)
        self.lang_combo.addItem("English (en)", userData="en")
        self.lang_combo.addItem("French (fr)", userData="fr")
        self.lang_combo.addItem("German (de)", userData="de")
        self.lang_combo.addItem("Latin (la)", userData="la")

        # Power-user multi-language candidates:
        # The backend will try each candidate and keep the best alignment vs lyrics.
        self.lang_combo.addItem("English + Latin + Auto (en,la,auto)", userData="en,la,auto")
        self.lang_combo.addItem("Latin + English + Auto (la,en,auto)", userData="la,en,auto")

        # Convenience preset for typical mixed tracks:
        # Mostly English, with a Latin section in the middle.
        # Note: this is NOT time-splitting; it's a candidate list evaluated globally.
        self.lang_combo.addItem("English / Latin / English (en, la, en)", userData="en,la,en")

        self.lang_combo.setCurrentIndex(0)  # default = auto
        lang_row.addWidget(self.lang_combo)
        lang_row.addStretch(1)
        sg_layout.addLayout(lang_row)
        
        # Audio source for alignment and karaoke
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Audio source:", settings_group))
        self.audio_source_combo = QComboBox(settings_group)
        self.audio_source_combo.addItem("Full mix (project audio)", userData="full_mix")
        self.audio_source_combo.addItem("Manual file (browse…)", userData="manual")
        self.audio_source_combo.setCurrentIndex(0)
        source_row.addWidget(self.audio_source_combo)

        self.btn_browse_manual = QPushButton("Browse…", settings_group)
        self.btn_browse_manual.setEnabled(False)
        source_row.addWidget(self.btn_browse_manual)

        self.lbl_manual_path = QLabel("", settings_group)
        self.lbl_manual_path.setMinimumWidth(220)
        self.lbl_manual_path.setWordWrap(False)
        self.lbl_manual_path.setToolTip("Selected manual audio file")
        source_row.addWidget(self.lbl_manual_path, 1)

        source_row.addStretch(1)
        sg_layout.addLayout(source_row)

        # Device (CPU / GPU) + status
        device_row = QHBoxLayout()
        self.chk_use_gpu = QCheckBox("Use GPU if available", settings_group)
        self.lbl_gpu_status = QLabel("", settings_group)
        self.lbl_gpu_status.setWordWrap(True)
        device_row.addWidget(self.chk_use_gpu)
        device_row.addWidget(self.lbl_gpu_status, 1)
        sg_layout.addLayout(device_row)

        # Whisper decoding quality (beam search parameters)
        decode_row = QHBoxLayout()
        decode_row.addWidget(QLabel("Beam size:", settings_group))
        self.spin_beam_size = QSpinBox(settings_group)
        self.spin_beam_size.setRange(1, 16)
        self.spin_beam_size.setValue(5)
        self.spin_beam_size.setToolTip(
            "Number of parallel paths explored by Whisper.\n"
            "Higher = better quality, but slower decoding."
        )
        decode_row.addWidget(self.spin_beam_size)

        decode_row.addWidget(QLabel("Patience:", settings_group))
        self.spin_patience = QDoubleSpinBox(settings_group)
        self.spin_patience.setDecimals(2)
        self.spin_patience.setRange(0.0, 2.0)
        self.spin_patience.setSingleStep(0.1)
        self.spin_patience.setValue(1.0)
        self.spin_patience.setToolTip(
            "Beam search patience factor.\n"
            "Values > 1.0 let Whisper explore more candidates before stopping."
        )
        decode_row.addWidget(self.spin_patience)
        decode_row.addStretch(1)
        sg_layout.addLayout(decode_row)

        # Alignment refinement parameters
        align_params_row = QHBoxLayout()
        align_params_row.addWidget(QLabel("Max search window:", settings_group))
        self.spin_max_search_window = QSpinBox(settings_group)
        self.spin_max_search_window.setRange(1, 50)
        self.spin_max_search_window.setValue(5)
        self.spin_max_search_window.setToolTip(
            "How many recognized words to look ahead when aligning each lyrics word.\n"
            "Larger window is more robust to insertions but slightly slower."
        )
        align_params_row.addWidget(self.spin_max_search_window)

        align_params_row.addWidget(QLabel("Min similarity:", settings_group))
        self.spin_min_similarity = QDoubleSpinBox(settings_group)
        self.spin_min_similarity.setDecimals(2)
        self.spin_min_similarity.setRange(0.0, 1.0)
        self.spin_min_similarity.setSingleStep(0.05)
        self.spin_min_similarity.setValue(0.60)
        self.spin_min_similarity.setToolTip(
            "Minimal similarity (0–1) between lyrics word and recognized word.\n"
            "Higher = stricter matches (fewer false positives, more 'unmatched' words)."
        )
        align_params_row.addWidget(self.spin_min_similarity)
        align_params_row.addStretch(1)

        # Alignment passes
        passes_row = QHBoxLayout()
        passes_row.addWidget(QLabel("Alignment passes:", settings_group))
        self.alignment_passes_combo = QComboBox(settings_group)
        self.alignment_passes_combo.addItem("1 (single DP pass – fast)", userData=1)
        self.alignment_passes_combo.addItem("2 (anchors + fill – recommended)", userData=2)
        self.alignment_passes_combo.addItem("3 (salvage – harsh vocals)", userData=3)
        self.alignment_passes_combo.setCurrentIndex(1)
        self.alignment_passes_combo.setToolTip(
            "Number of alignment passes.\n"
            "1: single global DP (fast).\n"
            "2: strict anchors + lenient fill (recommended for music).\n"
            "3: adds a salvage pass for very hard sections (growls/choirs)."
        )
        passes_row.addWidget(self.alignment_passes_combo)
        passes_row.addStretch(1)
        sg_layout.addLayout(passes_row)

        sg_layout.addLayout(align_params_row)

        # Advanced Whisper options (with presets)
        adv_group = QGroupBox("Whisper – advanced options", settings_group)
        adv_layout = QVBoxLayout(adv_group)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:", adv_group))
        self.combo_whisper_preset = QComboBox(adv_group)
        self.combo_whisper_preset.addItem("Balanced music (default)", userData="balanced")
        self.combo_whisper_preset.addItem("Harsh / noisy vocals", userData="harsh")
        self.combo_whisper_preset.addItem("Custom (manual tuning)", userData="custom")
        self.combo_whisper_preset.setCurrentIndex(0)
        self.combo_whisper_preset.setToolTip(
            "Balanced: good default for most songs.\n"
            "Harsh / noisy: more tolerant to growls, screams, noisy mixes,\n"
            "and multilingual/Latin content, at the cost of more noise."
        )
        preset_row.addWidget(self.combo_whisper_preset)
        preset_row.addStretch(1)
        adv_layout.addLayout(preset_row)

        row_ns = QHBoxLayout()
        row_ns.addWidget(QLabel("No-speech threshold:", adv_group))
        self.spin_no_speech_threshold = QDoubleSpinBox(adv_group)
        self.spin_no_speech_threshold.setDecimals(2)
        self.spin_no_speech_threshold.setRange(0.0, 1.0)
        self.spin_no_speech_threshold.setSingleStep(0.05)
        self.spin_no_speech_threshold.setValue(0.60)
        self.spin_no_speech_threshold.setToolTip(
            "Whisper's threshold to classify segments as 'no speech'.\n"
            "Lower values keep more low-intelligibility segments\n"
            "(useful for harsh vocals), but may include more noise."
        )
        row_ns.addWidget(self.spin_no_speech_threshold)

        self.chk_condition_prev = QCheckBox("Use previous text as context", adv_group)
        self.chk_condition_prev.setChecked(True)
        self.chk_condition_prev.setToolTip(
            "If enabled, each segment is decoded using the previous text as context.\n"
            "Disable this for strongly multilingual or Latin content to reduce\n"
            "language drift back to English/French."
        )
        row_ns.addWidget(self.chk_condition_prev)
        row_ns.addStretch(1)
        adv_layout.addLayout(row_ns)

        row_filters = QHBoxLayout()
        row_filters.addWidget(QLabel("Compression ratio max:", adv_group))
        self.spin_compression_ratio = QDoubleSpinBox(adv_group)
        self.spin_compression_ratio.setDecimals(2)
        self.spin_compression_ratio.setRange(0.0, 10.0)
        self.spin_compression_ratio.setSingleStep(0.1)
        self.spin_compression_ratio.setValue(2.40)
        self.spin_compression_ratio.setToolTip(
            "Filter for hallucinated / repetitive segments.\n"
            "Higher values are more tolerant (fewer segments dropped)."
        )
        row_filters.addWidget(self.spin_compression_ratio)

        row_filters.addWidget(QLabel("Log-prob threshold:", adv_group))
        self.spin_logprob_threshold = QDoubleSpinBox(adv_group)
        self.spin_logprob_threshold.setDecimals(2)
        self.spin_logprob_threshold.setRange(-10.0, 0.0)
        self.spin_logprob_threshold.setSingleStep(0.1)
        self.spin_logprob_threshold.setValue(-1.00)
        self.spin_logprob_threshold.setToolTip(
            "Average log-probability threshold for rejecting segments.\n"
            "Lower (more negative) = more tolerant (keeps more uncertain segments)."
        )
        row_filters.addWidget(self.spin_logprob_threshold)
        row_filters.addStretch(1)
        adv_layout.addLayout(row_filters)

        prompt_label = QLabel("Initial prompt (optional):", adv_group)
        prompt_label.setToolTip(
            "Optional text shown to Whisper before decoding.\n"
            "Can be used to hint the model about Latin / specific vocabulary,\n"
            "e.g. 'Ecclesiastical Latin liturgical lyrics'."
        )
        adv_layout.addWidget(prompt_label)

        self.txt_initial_prompt = QTextEdit(adv_group)
        self.txt_initial_prompt.setPlaceholderText(
            "Example: 'Latin liturgical lyrics, ecclesiastical pronunciation.'"
        )
        self.txt_initial_prompt.setFixedHeight(60)
        adv_layout.addWidget(self.txt_initial_prompt)

        adv_help = QLabel(
            "Balanced preset: good starting point for most songs.\n"
            "Harsh / noisy: more tolerant settings for death metal, growls,\n"
            "and multilingual/Latin content – at the cost of more noise.",
            adv_group,
        )
        adv_help.setWordWrap(True)
        adv_layout.addWidget(adv_help)

        adv_group.setLayout(adv_layout)
        sg_layout.addWidget(adv_group)

        # Apply default Whisper preset and connect changes
        self._apply_whisper_preset("balanced")
        self.combo_whisper_preset.currentIndexChanged.connect(self._on_whisper_preset_changed)

        # Run alignment
        align_row = QHBoxLayout()
        self.btn_align = QPushButton("Align lyrics with audio", settings_group)
        align_row.addWidget(self.btn_align)
        align_row.addStretch(1)
        sg_layout.addLayout(align_row)

        self.align_progress = QProgressBar(settings_group)
        self.align_progress.setRange(0, 100)
        self.align_progress.setValue(0)

        # One-line status (latest message)
        self.lbl_align_status = QLabel("Ready.", settings_group)

        # Alignment summary shown after completion (coverage, passes, fills, etc.)
        self.lbl_align_summary = QLabel("", settings_group)
        self.lbl_align_summary.setWordWrap(True)
        self.lbl_align_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        # Live alignment log (mirrors console debug)
        self.txt_align_log = QTextEdit(settings_group)
        self.txt_align_log.setReadOnly(True)
        self.txt_align_log.setFixedHeight(140)
        self.txt_align_log.setPlaceholderText("Alignment log will appear here…")

        sg_layout.addWidget(self.align_progress)
        sg_layout.addWidget(self.lbl_align_status)
        sg_layout.addWidget(self.lbl_align_summary)
        sg_layout.addWidget(self.txt_align_log)

        settings_group.setLayout(sg_layout)

        # Put settings group in right column, and keep it top-aligned
        right_layout.addWidget(settings_group, 0)
        right_layout.addStretch(1)
        right_col.setLayout(right_layout)

        # -------------------------
        # Assemble the two columns
        # -------------------------
        tab_lyrics_layout.addWidget(left_col, 3)
        tab_lyrics_layout.addWidget(right_col, 2)
        tab_lyrics.setLayout(tab_lyrics_layout)


        # --------------------------------------------------------------
        # Tab 2: Export (ancien Alignment management caché)
        # --------------------------------------------------------------
        tab_manage = QWidget(self)
        tab_manage_layout = QVBoxLayout(tab_manage)

        # Ancien preview (toujours là pour la logique, mais caché)
        preview_group = QGroupBox("Alignment preview", tab_manage)
        pg_layout = QVBoxLayout(preview_group)

        main_row = QHBoxLayout()

        # Left: list of segments
        self.segments_list = QListWidget(preview_group)
        main_row.addWidget(self.segments_list, 1)

        # Right: timing editor
        editor_col = QVBoxLayout()

        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("Start (s):", preview_group))
        self.spin_start = QDoubleSpinBox(preview_group)
        self.spin_start.setRange(0.0, 100000.0)
        self.spin_start.setDecimals(3)
        time_row.addWidget(self.spin_start)

        time_row.addWidget(QLabel("End (s):", preview_group))
        self.spin_end = QDoubleSpinBox(preview_group)
        self.spin_end.setRange(0.0, 100000.0)
        self.spin_end.setDecimals(3)
        time_row.addWidget(self.spin_end)

        editor_col.addLayout(time_row)

        self.btn_apply_timing = QPushButton("Apply to selected", preview_group)
        editor_col.addWidget(self.btn_apply_timing, alignment=Qt.AlignmentFlag.AlignLeft)

        self.btn_save_timings = QPushButton("Save timings to JSON + SRT", preview_group)
        editor_col.addWidget(self.btn_save_timings, alignment=Qt.AlignmentFlag.AlignLeft)

        main_row.addLayout(editor_col, 1)
        pg_layout.addLayout(main_row)

        preview_group.setLayout(pg_layout)
        tab_manage_layout.addWidget(preview_group)

        # On garde ce group pour la logique mais on le cache dans l'onglet Export
        preview_group.hide()

        # ---- Export / Import group ----
        export_group = QGroupBox("Export / Import alignment files", tab_manage)
        export_layout = QVBoxLayout(export_group)

        export_info = QLabel(
            "Export or import alignment in several formats:\n"
            "- Subtitles (SRT) for YouTube / DaVinci Resolve\n"
            "- Phrase-level timings (JSON)\n"
            "- Word-level timings (JSON)\n"
            "- Phoneme-level timings (JSON)\n"
            "- Plain lyrics text (TXT)",
            export_group,
        )
        export_info.setWordWrap(True)
        export_layout.addWidget(export_info)

        # ---- SRT ----
        row_srt = QHBoxLayout()
        row_srt.addWidget(QLabel("Subtitles (SRT):", export_group))
        self.btn_export_srt = QPushButton("Export SRT…", export_group)
        self.btn_import_srt = QPushButton("Import SRT…", export_group)
        row_srt.addWidget(self.btn_export_srt)
        row_srt.addWidget(self.btn_import_srt)
        row_srt.addStretch(1)
        export_layout.addLayout(row_srt)

        # ---- Phrases ----
        row_phr = QHBoxLayout()
        row_phr.addWidget(QLabel("Phrase timings:", export_group))
        self.btn_export_phrases_json = QPushButton("Export JSON…", export_group)
        self.btn_import_phrases_json = QPushButton("Import JSON…", export_group)
        row_phr.addWidget(self.btn_export_phrases_json)
        row_phr.addWidget(self.btn_import_phrases_json)
        row_phr.addStretch(1)
        export_layout.addLayout(row_phr)

        # ---- Words ----
        row_words = QHBoxLayout()
        row_words.addWidget(QLabel("Word timings:", export_group))
        self.btn_export_words_json = QPushButton("Export JSON…", export_group)
        self.btn_import_words_json = QPushButton("Import JSON…", export_group)
        row_words.addWidget(self.btn_export_words_json)
        row_words.addWidget(self.btn_import_words_json)
        row_words.addStretch(1)
        export_layout.addLayout(row_words)

        # ---- Phonemes ----
        row_phon = QHBoxLayout()
        row_phon.addWidget(QLabel("Phoneme timings:", export_group))
        self.btn_export_phonemes_json = QPushButton("Export JSON…", export_group)
        self.btn_import_phonemes_json = QPushButton("Import JSON…", export_group)
        row_phon.addWidget(self.btn_export_phonemes_json)
        row_phon.addWidget(self.btn_import_phonemes_json)
        row_phon.addStretch(1)
        export_layout.addLayout(row_phon)

        # ---- Lyrics TXT ----
        row_txt = QHBoxLayout()
        row_txt.addWidget(QLabel("Lyrics text:", export_group))
        self.btn_export_lyrics_txt = QPushButton("Export TXT…", export_group)
        self.btn_import_lyrics_txt = QPushButton("Import TXT…", export_group)
        row_txt.addWidget(self.btn_export_lyrics_txt)
        row_txt.addWidget(self.btn_import_lyrics_txt)
        row_txt.addStretch(1)
        export_layout.addLayout(row_txt)

        # ---- Status + bulk export ----
        self.lbl_export_status = QLabel("", export_group)
        export_layout.addWidget(self.lbl_export_status)

        self.btn_export_all = QPushButton("Export all files…", export_group)
        export_layout.addWidget(self.btn_export_all)

        tab_manage_layout.addWidget(export_group)
        tab_manage_layout.addStretch(1)
        tab_manage.setLayout(tab_manage_layout)


        # --------------------------------------------------------------
        # Tab 3: Karaoke preview (Alignment management (karaoke preview))
        # --------------------------------------------------------------
        tab_kara = QWidget(self)
        tab_kara_layout = QVBoxLayout(tab_kara)

        kara_group = QGroupBox("Karaoke preview", tab_kara)
        kara_layout = QHBoxLayout(kara_group)

        # Left column: current line + word + local player + phrase + word timing
        left_col = QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(4)  # compact vertical spacing

        # Current phrase and current word
        self.lbl_kara_line = QLabel("", kara_group)
        self.lbl_kara_word = QLabel("", kara_group)
        self.lbl_kara_line.setStyleSheet("font-weight: bold;")
        self.lbl_kara_word.setStyleSheet("color: #00ccff;")

        left_col.addWidget(self.lbl_kara_line)
        left_col.addWidget(self.lbl_kara_word)

        # Timer label and position slider for the current audio
        self.lbl_kara_track_time = QLabel("0.000 s", kara_group)
        left_col.addWidget(self.lbl_kara_track_time)

        self.kara_slider = QSlider(Qt.Orientation.Horizontal, kara_group)
        self.kara_slider.setRange(0, 0)

        # Neon-style fake waveform background for the karaoke track slider
        self.kara_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 18px;
                margin: 0px;
                border-radius: 9px;
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0.00 #050008,
                    stop: 0.15 #200020,
                    stop: 0.30 #ff00ff,
                    stop: 0.45 #300030,
                    stop: 0.60 #ff66ff,
                    stop: 0.75 #300030,
                    stop: 0.90 #ff00ff,
                    stop: 1.00 #050008
                );
            }
            QSlider::handle:horizontal {
                background: #000000;
                border: 1px solid #ff66ff;
                width: 10px;
                margin: -6px 0;
                border-radius: 5px;
            }
            QSlider::sub-page:horizontal {
                background: rgba(0, 255, 255, 120);
                border-radius: 9px;
            }
            QSlider::add-page:horizontal {
                background: transparent;
                border-radius: 9px;
            }
            """
        )

        left_col.addWidget(self.kara_slider)

        # Local player controls (Play / Pause / Stop) for the selected audio source
        kara_controls = QHBoxLayout()
        self.btn_karaoke_play = QPushButton("Play", kara_group)
        self.btn_karaoke_pause = QPushButton("Pause", kara_group)
        self.btn_karaoke_stop = QPushButton("Stop", kara_group)
        kara_controls.addWidget(self.btn_karaoke_play)
        kara_controls.addWidget(self.btn_karaoke_pause)
        kara_controls.addWidget(self.btn_karaoke_stop)
        kara_controls.addStretch(1)
        left_col.addLayout(kara_controls)

        # Sliders to adjust the start/end of the selected phrase
        phrase_group = QGroupBox("Selected phrase timing", kara_group)
        phrase_layout = QVBoxLayout(phrase_group)

        self.lbl_phrase_start_value = QLabel("Start: 0.000 s", phrase_group)
        self.phrase_start_slider = QSlider(Qt.Orientation.Horizontal, phrase_group)
        self.phrase_start_slider.setRange(0, 0)

        self.lbl_phrase_end_value = QLabel("End: 0.000 s", phrase_group)
        self.phrase_end_slider = QSlider(Qt.Orientation.Horizontal, phrase_group)
        self.phrase_end_slider.setRange(0, 0)

        # 10 ms nudge buttons for phrase start
        phrase_start_row = QHBoxLayout()
        phrase_start_row.addWidget(self.lbl_phrase_start_value)
        self.btn_phrase_start_minus = QPushButton("-10 ms", phrase_group)
        self.btn_phrase_start_plus = QPushButton("+10 ms", phrase_group)
        phrase_start_row.addWidget(self.btn_phrase_start_minus)
        phrase_start_row.addWidget(self.btn_phrase_start_plus)
        phrase_start_row.addStretch(1)
        phrase_layout.addLayout(phrase_start_row)
        phrase_layout.addWidget(self.phrase_start_slider)

        # 10 ms nudge buttons for phrase end
        phrase_end_row = QHBoxLayout()
        phrase_end_row.addWidget(self.lbl_phrase_end_value)
        self.btn_phrase_end_minus = QPushButton("-10 ms", phrase_group)
        self.btn_phrase_end_plus = QPushButton("+10 ms", phrase_group)
        phrase_end_row.addWidget(self.btn_phrase_end_minus)
        phrase_end_row.addWidget(self.btn_phrase_end_plus)
        phrase_end_row.addStretch(1)
        phrase_layout.addLayout(phrase_end_row)
        phrase_layout.addWidget(self.phrase_end_slider)

        left_col.addWidget(phrase_group)


        # Sliders to adjust the start/end of the selected word + insert/delete
        word_group = QGroupBox("Selected word timing", kara_group)
        wg_layout = QVBoxLayout(word_group)

        self.lbl_word_text = QLabel("(no word selected)", word_group)
        wg_layout.addWidget(self.lbl_word_text)

        self.lbl_word_start_value = QLabel("Start: 0.000 s", word_group)
        self.word_start_slider = QSlider(Qt.Orientation.Horizontal, word_group)
        self.word_start_slider.setRange(0, 0)

        self.lbl_word_end_value = QLabel("End: 0.000 s", word_group)
        self.word_end_slider = QSlider(Qt.Orientation.Horizontal, word_group)
        self.word_end_slider.setRange(0, 0)

        # 10 ms nudge buttons for word start
        word_start_row = QHBoxLayout()
        word_start_row.addWidget(self.lbl_word_start_value)
        self.btn_word_start_minus = QPushButton("-10 ms", word_group)
        self.btn_word_start_plus = QPushButton("+10 ms", word_group)
        word_start_row.addWidget(self.btn_word_start_minus)
        word_start_row.addWidget(self.btn_word_start_plus)
        word_start_row.addStretch(1)
        wg_layout.addLayout(word_start_row)
        wg_layout.addWidget(self.word_start_slider)

        # 10 ms nudge buttons for word end
        word_end_row = QHBoxLayout()
        word_end_row.addWidget(self.lbl_word_end_value)
        self.btn_word_end_minus = QPushButton("-10 ms", word_group)
        self.btn_word_end_plus = QPushButton("+10 ms", word_group)
        word_end_row.addWidget(self.btn_word_end_minus)
        word_end_row.addWidget(self.btn_word_end_plus)
        word_end_row.addStretch(1)
        wg_layout.addLayout(word_end_row)
        wg_layout.addWidget(self.word_end_slider)

        word_btn_row = QHBoxLayout()
        self.btn_insert_word = QPushButton("Insert word", word_group)
        self.btn_delete_word = QPushButton("Delete word", word_group)
        word_btn_row.addWidget(self.btn_insert_word)
        word_btn_row.addWidget(self.btn_delete_word)
        word_btn_row.addStretch(1)
        wg_layout.addLayout(word_btn_row)

        word_group.setLayout(wg_layout)
        left_col.addWidget(word_group)


        # Wrap left column in a QWidget so we can align it to the top
        left_widget = QWidget(kara_group)
        left_widget.setLayout(left_col)

        # Center column: phrases list
        self.kara_scroll_list = QListWidget(kara_group)
        # Use default palette / theme for background and selection

        # Right column: words of the selected/current phrase
        self.kara_words_list = QListWidget(kara_group)
        # Use default palette / theme for background and selection


        # Use a very soft alternating background so items stay readable
        # on dark / neon themes without being too aggressive.
        subtle_list_style = """
            QListWidget {
                padding-left: 10px;
                padding-right: 10px;
                border: none;
                background-color: transparent;
            }
            QListWidget::item {
                padding: 2px 6px;
                background-color: transparent;
            }
            /* Slightly lighter background for every other row */
            QListWidget::item:alternate {
                background-color: rgba(255, 255, 255, 18);
            }
            /* Keep selection clearly visible whatever the theme */
            QListWidget::item:selected {
                background-color: rgba(70, 194, 255, 150);
                color: #ffffff;
            }
        """

        self.kara_scroll_list.setStyleSheet(subtle_list_style)
        self.kara_words_list.setStyleSheet(subtle_list_style)


        # Add widgets to main horizontal layout
        kara_layout.addWidget(left_widget)
        kara_layout.addWidget(self.kara_scroll_list)
        kara_layout.addWidget(self.kara_words_list)

        # Force left column to be top-aligned inside the horizontal layout
        kara_layout.setAlignment(left_widget, Qt.AlignmentFlag.AlignTop)

        # 3 columns with equal width
        kara_layout.setStretch(0, 1)  # left column
        kara_layout.setStretch(1, 1)  # phrases
        kara_layout.setStretch(2, 1)  # words

        kara_group.setLayout(kara_layout)
        tab_kara_layout.addWidget(kara_group)
        tab_kara.setLayout(tab_kara_layout)

        tab_visual = self._build_visual_alignment_tab()

        # --------------------------------------------------------------
        # Add tabs in desired order
        # --------------------------------------------------------------
        self.sub_tabs.addTab(tab_lyrics, "Lyrics alignment")
        self.sub_tabs.addTab(tab_kara, "Lyrics alignment (numeric)")
        self.sub_tabs.addTab(tab_visual, "Lyrics alignment (visual)")
        self.sub_tabs.addTab(tab_manage, "Export / Import")
        # Remember the visual tab index to manage playback lifecycle (stop when leaving the tab)
        self._tab_visual_index = self.sub_tabs.indexOf(tab_visual)
        self.sub_tabs.currentChanged.connect(self._on_subtab_changed)

        # --------------------------------------------------------------
        # Connections at the end of UI setup
        # --------------------------------------------------------------
        self.btn_load_lyrics.clicked.connect(self.load_lyrics_from_file)
        self.btn_save_lyrics.clicked.connect(self.save_lyrics_to_file)
        self.btn_align.clicked.connect(self.run_alignment)

        self.segments_list.currentItemChanged.connect(self.on_segment_selected)
        self.btn_apply_timing.clicked.connect(self.apply_timing_to_selected)
        self.btn_save_timings.clicked.connect(self.save_timings_to_files)

        # Export: individual files
        self.btn_export_srt.clicked.connect(self.export_srt)
        self.btn_export_phrases_json.clicked.connect(self.export_phrases_json)
        self.btn_export_words_json.clicked.connect(self.export_words_json)
        self.btn_export_phonemes_json.clicked.connect(self.export_phonemes_json)
        self.btn_export_lyrics_txt.clicked.connect(self.export_lyrics_txt)

        # Import: individual files (JSON / TXT / SRT)
        self.btn_import_srt.clicked.connect(self.import_srt)
        self.btn_import_phrases_json.clicked.connect(self.import_phrases_json)
        self.btn_import_words_json.clicked.connect(self.import_words_json)
        self.btn_import_phonemes_json.clicked.connect(self.import_phonemes_json)
        self.btn_import_lyrics_txt.clicked.connect(self.import_lyrics_txt)

        # Export: bulk
        self.btn_export_all.clicked.connect(self.export_alignment_files)


        # Karaoke controls
        self.btn_karaoke_play.clicked.connect(self.play_karaoke)
        self.btn_karaoke_pause.clicked.connect(self.pause_karaoke)
        self.btn_karaoke_stop.clicked.connect(self.stop_karaoke)
        self.kara_slider.sliderMoved.connect(self._on_kara_slider_moved)

        # Phrase timing sliders (edit selected phrase)
        self.kara_scroll_list.currentItemChanged.connect(self._on_kara_phrase_selected)
        self.kara_scroll_list.itemClicked.connect(self._on_kara_phrase_clicked)
        self.phrase_start_slider.sliderMoved.connect(self._on_phrase_start_slider_moved)
        self.phrase_end_slider.sliderMoved.connect(self._on_phrase_end_slider_moved)
        self.phrase_start_slider.sliderReleased.connect(self._on_phrase_sliders_released)
        self.phrase_end_slider.sliderReleased.connect(self._on_phrase_sliders_released)

        # Word timing sliders (edit selected word)
        self.kara_words_list.currentItemChanged.connect(self._on_kara_word_selected)
        self.word_start_slider.sliderMoved.connect(self._on_word_start_slider_moved)
        self.word_end_slider.sliderMoved.connect(self._on_word_end_slider_moved)
        self.word_start_slider.sliderReleased.connect(self._on_word_sliders_released)
        self.word_end_slider.sliderReleased.connect(self._on_word_sliders_released)
        self.btn_insert_word.clicked.connect(self.insert_word_for_current_phrase)
        self.btn_delete_word.clicked.connect(self.delete_selected_word)
        
        # Nudge buttons (10 ms)
        self.btn_phrase_start_minus.clicked.connect(self._nudge_phrase_start_minus)
        self.btn_phrase_start_plus.clicked.connect(self._nudge_phrase_start_plus)
        self.btn_phrase_end_minus.clicked.connect(self._nudge_phrase_end_minus)
        self.btn_phrase_end_plus.clicked.connect(self._nudge_phrase_end_plus)

        self.btn_word_start_minus.clicked.connect(self._nudge_word_start_minus)
        self.btn_word_start_plus.clicked.connect(self._nudge_word_start_plus)
        self.btn_word_end_minus.clicked.connect(self._nudge_word_end_minus)
        self.btn_word_end_plus.clicked.connect(self._nudge_word_end_plus)
        

        # Extra connections for audio source + persistence
        self.audio_source_combo.currentIndexChanged.connect(self._on_audio_source_changed)
        self.btn_browse_manual.clicked.connect(self._browse_manual_audio)

        # Persist lyrics as the user types
        self.lyrics_edit.textChanged.connect(self._save_persisted_lyrics)

        # Load persisted settings (lyrics text, audio source mode, manual path)
        self._load_persisted_prefs()

    # ------------------------------------------------------------------
    # Whisper presets (balanced / harsh / custom)
    # ------------------------------------------------------------------

    def _on_whisper_preset_changed(self) -> None:
        """React to preset combo changes and update advanced parameters."""
        idx = self.combo_whisper_preset.currentIndex()
        key = self.combo_whisper_preset.itemData(idx)
        if not isinstance(key, str):
            return
        if key == "custom":
            # Do not override manual tuning in custom mode.
            return
        self._apply_whisper_preset(key)

    def _apply_whisper_preset(self, preset_key: str) -> None:
        """
        Apply a preset for Whisper + alignment parameters.

        'balanced' is suited for most songs.
        'harsh' is more tolerant to extreme vocals / noisy mixes.
        """
        # Safe defaults in case some widgets are missing
        if not hasattr(self, "spin_beam_size"):
            return

        if preset_key == "harsh":
            # More exhaustive search + more tolerant filters
            self.spin_beam_size.setValue(8)
            self.spin_patience.setValue(1.20)

            self.spin_no_speech_threshold.setValue(0.35)
            self.spin_compression_ratio.setValue(3.00)
            self.spin_logprob_threshold.setValue(-1.50)

            self.chk_condition_prev.setChecked(False)

            # Alignment: more tolerant to recognition errors
            self.spin_max_search_window.setValue(8)
            self.spin_min_similarity.setValue(0.55)
        else:
            # 'balanced' (default) preset
            self.spin_beam_size.setValue(5)
            self.spin_patience.setValue(1.00)

            self.spin_no_speech_threshold.setValue(0.60)
            self.spin_compression_ratio.setValue(2.40)
            self.spin_logprob_threshold.setValue(-1.00)

            self.chk_condition_prev.setChecked(True)

            # Alignment: stricter but cleaner
            self.spin_max_search_window.setValue(5)
            self.spin_min_similarity.setValue(0.60)

    # ------------------------------------------------------------------
    # GPU detection
    # ------------------------------------------------------------------

    def _detect_gpu(self):
        """Detect if CUDA / GPU is usable and explain why / why not."""
        if torch is None:
            self._gpu_available = False
            self._gpu_reason = "PyTorch is not installed in this environment."
            self.chk_use_gpu.setChecked(False)
            self.chk_use_gpu.setEnabled(False)
            self.lbl_gpu_status.setText(self._gpu_reason)
            return

        try:
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                self._gpu_available = True
                self._gpu_reason = f"GPU available: {name}"
                self.chk_use_gpu.setEnabled(True)
                self.chk_use_gpu.setChecked(True)
                self.lbl_gpu_status.setText(self._gpu_reason)
            else:
                self._gpu_available = False
                # Try to give a meaningful reason
                if getattr(torch.version, "cuda", None) is None:
                    reason = "PyTorch build has no CUDA support (CPU-only install)."
                elif torch.cuda.device_count() == 0:
                    reason = "No CUDA GPU detected (cuda device_count == 0)."
                else:
                    reason = "CUDA runtime not available (drivers/toolkit issue)."
                self._gpu_reason = reason
                self.chk_use_gpu.setChecked(False)
                # You may want to allow forcing GPU later; for now disable
                self.chk_use_gpu.setEnabled(False)
                self.lbl_gpu_status.setText(f"GPU not available: {reason}")
        except Exception as e:  # pragma: no cover
            self._gpu_available = False
            self._gpu_reason = f"Could not query CUDA: {e}"
            self.chk_use_gpu.setChecked(False)
            self.chk_use_gpu.setEnabled(False)
            self.lbl_gpu_status.setText(self._gpu_reason)

    # ------------------------------------------------------------------
    # Public API used by MainWindow
    # ------------------------------------------------------------------

    def set_project(self, project: Optional[Project]):
        """Set the current project whose vocal alignment will be managed."""
        self._project = project
        if project:
            self.lbl_project.setText(f"Current project: {project.name}")
            # Restore lyrics from project if available; otherwise keep current (persisted) lyrics
            text = getattr(project, "lyrics_text", None)
            if text:
                self.lyrics_edit.setPlainText(text)
        else:
            self.lbl_project.setText("Current project: (none)")
            # Do not clear lyrics_edit here: keep last persisted lyrics

        # Reset alignment data and karaoke visuals
        self._phrases = []
        self._words = []
        self.segments_list.clear()
        self.lbl_kara_line.setText("")
        self.lbl_kara_word.setText("")
        if hasattr(self, "kara_scroll_list"):
            self.kara_scroll_list.clear()

        # Try to load existing alignment from disk (if any)
        self._load_alignment_from_disk()


    def on_position_changed(self, position_ms: int):
        """
        Update karaoke preview, visual editor playhead, and local player UI based on current playback position.
        """
        # --- Always keep the karaoke slider + time label in sync (even if no alignment data)
        if hasattr(self, "kara_slider"):
            try:
                duration_ms = int(self.player.duration() or 0)
            except Exception:
                duration_ms = 0

            if duration_ms > 0:
                self._media_duration_ms = duration_ms
                if self.kara_slider.maximum() == 0:
                    self.kara_slider.setRange(0, duration_ms)

            try:
                self.kara_slider.blockSignals(True)
                self.kara_slider.setValue(int(position_ms))
            finally:
                self.kara_slider.blockSignals(False)

        if hasattr(self, "lbl_kara_track_time"):
            try:
                seconds = float(position_ms) / 1000.0
                self.lbl_kara_track_time.setText(f"{seconds:7.3f} s")
            except Exception:
                pass

        pos_s = float(position_ms) / 1000.0

        # Keep the visual cursor aligned with actual playback time.
        try:
            if self.player is not None and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self._visual_cursor_time_s = float(pos_s)
        except Exception:
            pass


        # --- Visual editor helpers: playhead line + numeric timer
        if hasattr(self, "visual_timeline"):
            try:
                self.visual_timeline.set_playhead(pos_s)
            except Exception:
                pass

        if hasattr(self, "lbl_visual_playhead"):
            try:
                if getattr(self, "_current_phrase_idx", None) is not None and self._current_phrase_idx < len(self._phrases):
                    ph = self._phrases[int(self._current_phrase_idx)]
                    ph_s = float(self._get_phrase_field(ph, "start") or 0.0)
                    ph_e = float(self._get_phrase_field(ph, "end") or ph_s)
                    rel = pos_s - ph_s
                    self.lbl_visual_playhead.setText(
                        f"Playhead: {pos_s:0.3f} s   |   phrase: {rel:+0.3f} s / {max(ph_e - ph_s, 0.0):0.3f} s"
                    )
                else:
                    self.lbl_visual_playhead.setText(f"Playhead: {pos_s:0.3f} s")
            except Exception:
                pass

        # --- Visual editor phrase playback loop (optional)
        try:
            if getattr(self, "_visual_play_active", False):
                if pos_s >= float(getattr(self, "_visual_loop_end_s", 0.0)):
                    if getattr(self, "_visual_loop_enabled", False):
                        self.player.setPosition(int(float(getattr(self, "_visual_loop_start_s", 0.0)) * 1000.0))
                    else:
                        self.player.pause()
                        self._visual_play_active = False
        except Exception:
            pass

        # --- Karaoke highlight update only if we have alignment data
        if not self._phrases:
            return

        self._update_karaoke_for_time(pos_s)

    def _on_subtab_changed(self, idx: int) -> None:
        """Stop visual phrase playback when leaving the visual editor tab.

        This prevents stale looping state when the user toggles 'Loop' off/on and switches tabs.
        """
        try:
            visual_idx = getattr(self, "_tab_visual_index", None)
            if visual_idx is None:
                return

            leaving_visual = (int(idx) != int(visual_idx))
            if leaving_visual:
                # Leaving the visual tab -> stop the phrase playback and clear loop state.
                if getattr(self, "_visual_play_active", False):
                    self._stop_visual_phrase()

                # Always force loop OFF when leaving, to avoid confusing persistent UI state.
                self._visual_loop_enabled = False
                try:
                    if hasattr(self, "chk_visual_loop"):
                        self.chk_visual_loop.setChecked(False)
                except Exception:
                    pass

                # Safety: restore playback rate even if playback was interrupted elsewhere.
                try:
                    if self._visual_prev_playback_rate is not None and hasattr(self.player, "setPlaybackRate"):
                        self.player.setPlaybackRate(float(self._visual_prev_playback_rate))
                except Exception:
                    pass
                self._visual_prev_playback_rate = None
        except Exception:
            pass

    def load_lyrics_from_file(self):
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        start_dir = str(self._project.folder)
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load lyrics (.txt)",
            start_dir,
            "Text files (*.txt);;All files (*.*)",
        )
        if not path_str:
            return

        path = Path(path_str)
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read lyrics file:\n{e}")
            return

        self.lyrics_edit.setPlainText(text)

        # Persist in project (if helper exists)
        if hasattr(self._project, "set_lyrics_text"):
            try:
                self._project.set_lyrics_text(text)
            except Exception:
                self._project.lyrics_text = text
        else:
            self._project.lyrics_text = text

    def save_lyrics_to_file(self):
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        text = self.lyrics_edit.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "Empty lyrics", "Lyrics text is empty.")
            return

        default_path = self._project.folder / "lyrics.txt"
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save lyrics (.txt)",
            str(default_path),
            "Text files (*.txt);;All files (*.*)",
        )
        if not path_str:
            return

        path = Path(path_str)
        try:
            path.write_text(text, encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not write lyrics file:\n{e}")
            return

        # Persist into project
        if hasattr(self._project, "set_lyrics_text"):
            try:
                self._project.set_lyrics_text(text)
            except Exception:
                self._project.lyrics_text = text
        else:
            self._project.lyrics_text = text

    def _get_audio_path_for_current_mode(self) -> Optional[Path]:
        """
        Return audio path to use for alignment and karaoke based on current audio source.
        Supported modes:
          - 'full_mix': use the project's main audio file
          - 'manual'  : use the user-picked file
        """
        if not self._project:
            return None

        mode = self.audio_source_combo.currentData() if self.audio_source_combo is not None else "full_mix"

        # Manual file mode: use the path shown in the label
        if mode == "manual":
            manual = self.lbl_manual_path.text().strip()
            if manual:
                p = Path(manual)
                if p.is_file():
                    return p
                return None

        # Default/fallback: full mix from project
        if hasattr(self._project, "get_audio_path"):
            return self._project.get_audio_path()

        return None

        
    # ------------------------------------------------------------------
    # Persistence helpers (per-user, across app launches)
    # ------------------------------------------------------------------

    def _settings(self) -> QSettings:
        """
        Return a QSettings instance. On Windows it stores in the Registry under:
        HKEY_CURRENT_USER\Software\Olaf\OlafApp
        """
        return QSettings("Olaf", "OlafApp")

    def _load_persisted_prefs(self) -> None:
        """Load last used lyrics and audio source settings from QSettings."""
        s = self._settings()

        # Lyrics text
        last_lyrics = s.value("lyrics_alignment/last_lyrics_text", type=str)
        if last_lyrics:
            # Do not emit textChanged while setting programmatically
            block = self.lyrics_edit.blockSignals(True)
            self.lyrics_edit.setPlainText(last_lyrics)
            self.lyrics_edit.blockSignals(block)

        # Audio source mode
        mode = s.value("lyrics_alignment/audio_source_mode", "full_mix", type=str)
        if mode not in ("full_mix", "manual"):
            mode = "full_mix"

        # Set combo index based on userData
        index_to_set = 0
        for i in range(self.audio_source_combo.count()):
            if self.audio_source_combo.itemData(i) == mode:
                index_to_set = i
                break
        self.audio_source_combo.setCurrentIndex(index_to_set)

        # Manual file path (if any)
        manual_path = s.value("lyrics_alignment/manual_audio_path", "", type=str)
        if manual_path:
            self.lbl_manual_path.setText(manual_path)
            self.lbl_manual_path.setToolTip(manual_path)

        # Enable/disable browse according to mode
        self.btn_browse_manual.setEnabled(mode == "manual")

    def _save_persisted_lyrics(self) -> None:
        """Save current lyrics text to QSettings (called on textChanged)."""
        s = self._settings()
        s.setValue("lyrics_alignment/last_lyrics_text", self.lyrics_edit.toPlainText())

    def _on_audio_source_changed(self, idx: int) -> None:
        """Enable browse button only for 'manual' and persist the choice."""
        mode = self.audio_source_combo.currentData()
        self.btn_browse_manual.setEnabled(mode == "manual")
        # Persist the mode
        s = self._settings()
        s.setValue("lyrics_alignment/audio_source_mode", mode if mode else "full_mix")

    def _browse_manual_audio(self) -> None:
        """Pick a manual audio file and persist its path."""
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select audio file",
            "",
            "Audio files (*.wav *.mp3 *.flac *.m4a *.aac);;All files (*.*)",
        )
        if not file_path:
            return
        self.lbl_manual_path.setText(file_path)
        self.lbl_manual_path.setToolTip(file_path)
        s = self._settings()
        s.setValue("lyrics_alignment/manual_audio_path", file_path)

    def _load_alignment_from_disk(self) -> None:
        """
        Load existing alignment (phrases + words) from project/vocal_align/*.json
        if available, so timings survive across sessions.
        """
        if not self._project:
            return

        align_dir = self._project.folder / "vocal_align"
        phrases_path = align_dir / "phrases.json"
        words_path = align_dir / "words.json"

        if not phrases_path.is_file():
            # No previous alignment saved for this project
            return

        try:
            phrases_data = json.loads(phrases_path.read_text(encoding="utf-8"))
            if isinstance(phrases_data, list):
                self._phrases = phrases_data
            else:
                self._phrases = []
        except Exception as e:
            self._phrases = []
            self.lbl_align_status.setText(f"Could not read phrases.json: {e}")
            return

        if words_path.is_file():
            try:
                words_data = json.loads(words_path.read_text(encoding="utf-8"))
                if isinstance(words_data, list):
                    self._words = words_data
                else:
                    self._words = []
            except Exception:
                # If words fail to load, we still keep phrases
                self._words = []
        else:
            self._words = []

        # Refresh UI to reflect loaded alignment
        self.populate_segments_preview()
        self.lbl_align_status.setText("Loaded existing alignment from disk.")

        # If available, also show the last saved alignment summary.
        if self._project is not None and hasattr(self, "lbl_align_summary"):
            try:
                align_dir = self._project.folder / "vocal_align"
                mpath = align_dir / "alignment_metrics.json"
                if mpath.is_file():
                    data = json.loads(mpath.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        metrics = data.get("metrics")
                        chosen_lang = str(data.get("chosen_whisper_language", "") or "")
                        if isinstance(metrics, dict):
                            self.lbl_align_summary.setText(self._format_alignment_summary(metrics, chosen_lang))
            except Exception:
                pass
        
    # -------------------- Import helpers --------------------
        # Cache as "last saved" state for the Reset button
        self._saved_phrases_snapshot = copy.deepcopy(self._phrases)
        self._saved_words_snapshot = copy.deepcopy(self._words)

        # Refresh both phrase lists / editors
        self._refresh_karaoke_scroll()
        if self._current_phrase_idx is not None:
            self._refresh_visual_editor_for_phrase(self._current_phrase_idx)


    def _open_file_dialog(self, title: str, filter_str: str) -> Optional[Path]:
        """Small helper to open a file dialog and return a Path or None if cancelled."""
        file_str, _ = QFileDialog.getOpenFileName(
            self,
            title,
            "",
            filter_str,
        )
        if not file_str:
            return None
        return Path(file_str)


    # ------------------------------------------------------------------
    # Alignment + GPU / CPU selection
    # ------------------------------------------------------------------

    def run_alignment(self):
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        # Check audio according to selected source (full mix or manual file)
        audio_path = self._get_audio_path_for_current_mode()
        if not audio_path or not Path(audio_path).is_file():
            mode = "full_mix"
            if hasattr(self, "audio_source_combo") and self.audio_source_combo is not None:
                data = self.audio_source_combo.currentData()
                if isinstance(data, str) and data:
                    mode = data

            if mode == "manual":
                QMessageBox.warning(
                    self,
                    "Invalid audio file",
                    "No valid manual audio file selected.\n"
                    "Please click 'Browse…' or switch the audio source to 'Full mix (project audio)'.",
                )
            else:
                QMessageBox.warning(
                    self,
                    "No audio file",
                    "This project has no valid audio file associated.",
                )
            return

        # Lyrics
        lyrics_text = self.lyrics_edit.toPlainText().strip()
        if not lyrics_text:
            QMessageBox.warning(self, "No lyrics", "Please enter or load lyrics first.")
            return

        # Persist lyrics in the project + to disk so the backend is forced to use them
        if hasattr(self._project, "set_lyrics_text"):
            try:
                self._project.set_lyrics_text(lyrics_text)
            except Exception:
                self._project.lyrics_text = lyrics_text
        else:
            self._project.lyrics_text = lyrics_text

        # Write lyrics.txt in both project root and vocal_align/ for safety
        main_lyrics = self._project.folder / "lyrics.txt"
        align_dir = self._project.folder / "vocal_align"
        align_dir.mkdir(parents=True, exist_ok=True)
        align_lyrics = align_dir / "lyrics.txt"
        for p in (main_lyrics, align_lyrics):
            try:
                p.write_text(lyrics_text, encoding="utf-8")
            except Exception:
                pass

        model_name = self.model_combo.currentText()

        # Use combo userData to decide if we force a language or let Whisper auto-detect
        lang_data = self.lang_combo.currentData()
        whisper_language = None if not lang_data else str(lang_data)

        device = "cuda" if (self.chk_use_gpu.isChecked() and self._gpu_available) else "cpu"

        # Advanced decoding / alignment parameters from the UI
        beam_size = int(self.spin_beam_size.value()) if hasattr(self, "spin_beam_size") else 5
        patience = float(self.spin_patience.value()) if hasattr(self, "spin_patience") else 1.0
        max_search_window = int(self.spin_max_search_window.value()) if hasattr(self, "spin_max_search_window") else 5
        min_similarity = float(self.spin_min_similarity.value()) if hasattr(self, "spin_min_similarity") else 0.60

        alignment_passes = 2
        if hasattr(self, "alignment_passes_combo") and self.alignment_passes_combo is not None:
            try:
                alignment_passes = int(self.alignment_passes_combo.currentData() or 2)
            except Exception:
                alignment_passes = 2

        # Advanced Whisper filters / context / prompt
        no_speech_threshold = float(self.spin_no_speech_threshold.value()) if hasattr(self, "spin_no_speech_threshold") else None
        compression_ratio_threshold = float(self.spin_compression_ratio.value()) if hasattr(self, "spin_compression_ratio") else None
        logprob_threshold = float(self.spin_logprob_threshold.value()) if hasattr(self, "spin_logprob_threshold") else None
        condition_on_previous_text = bool(self.chk_condition_prev.isChecked()) if hasattr(self, "chk_condition_prev") else None

        initial_prompt = None
        if hasattr(self, "txt_initial_prompt"):
            txt = self.txt_initial_prompt.toPlainText().strip()
            if txt:
                initial_prompt = txt

        self.btn_align.setEnabled(False)
        self.align_progress.setValue(0)
        self.lbl_align_status.setText(f"Running alignment on {device}…")
        if hasattr(self, "lbl_align_summary"):
            self.lbl_align_summary.setText("")
        if hasattr(self, "txt_align_log"):
            try:
                self.txt_align_log.clear()
            except Exception:
                pass
        QApplication.processEvents()

        def progress_cb(percent: float, message: str):
            # Keep UI responsive + log for debugging
            try:
                self.align_progress.setValue(int(percent))
                self.lbl_align_status.setText(message[:200])
                if hasattr(self, "txt_align_log"):
                    try:
                        self.txt_align_log.append(f"{percent:5.1f}%  {message}")
                    except Exception:
                        pass
                QApplication.processEvents()
            except Exception:
                pass

            try:
                print(f"[Whisper/Align] {percent:5.1f}% - {message}")
            except Exception:
                pass

        # Immediate ping: proves that the alignment call chain actually started.
        progress_cb(0.0, f"Starting alignment (passes={alignment_passes}, device={device})…")

        try:
            result = run_alignment_for_project(
                project=self._project,
                audio_path=audio_path,
                lyrics_text=lyrics_text,
                model_name=model_name,
                whisper_language=whisper_language,  # None => auto-detect
                phoneme_language=None,
                device=device,
                beam_size=beam_size,
                patience=patience,
                max_search_window=max_search_window,
                min_similarity=min_similarity,
                alignment_passes=alignment_passes,
                no_speech_threshold=no_speech_threshold,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                condition_on_previous_text=condition_on_previous_text,
                initial_prompt=initial_prompt,
                progress_cb=progress_cb,
            )

        except Exception as e:
            QMessageBox.critical(self, "Alignment error", f"Whisper / alignment failed:\n{e}")
            self.lbl_align_status.setText("Error during alignment.")
            return
        finally:
            self.btn_align.setEnabled(True)

        # Normalize alignment result (dict or dataclass)
        self._phrases = getattr(result, "phrases", getattr(result, "lines", [])) or []
        self._words = getattr(result, "words", []) or []

        self.align_progress.setValue(100)
        self.lbl_align_status.setText("Alignment done.")

        # Show debug summary (coverage, fills, inserted phrases, etc.)
        metrics = getattr(result, "metrics", None)
        chosen_lang = getattr(result, "chosen_whisper_language", "") or ""
        if metrics is None and self._project is not None:
            # Fallback: load from disk
            try:
                align_dir = self._project.folder / "vocal_align"
                mpath = align_dir / "alignment_metrics.json"
                if mpath.is_file():
                    data = json.loads(mpath.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        metrics = data.get("metrics")
                        chosen_lang = str(data.get("chosen_whisper_language", chosen_lang) or chosen_lang)
            except Exception:
                pass

        if isinstance(metrics, dict) and hasattr(self, "lbl_align_summary"):
            self.lbl_align_summary.setText(self._format_alignment_summary(metrics, chosen_lang))
            if hasattr(self, "txt_align_log"):
                try:
                    self.txt_align_log.append("")
                    self.txt_align_log.append(self._format_alignment_summary(metrics, chosen_lang))
                except Exception:
                    pass

        self.populate_segments_preview()


    def progress_cb(percent: float, message: str):
        self.align_progress.setValue(int(percent))
        self.lbl_align_status.setText(message[:150])
        QApplication.processEvents()

        # Also print progress to stdout so pass 3 messages don't get lost in the UI.
        try:
            print(f"[Whisper/Align] {percent:5.1f}% - {message}")
        except Exception:
            pass


        try:
            # run_alignment_for_project(
            #     project,
            #     audio_path,
            #     lyrics_text,
            #     model_name,
            #     whisper_language,
            #     phoneme_language,
            #     device,
            #     progress_cb,
            # )
            result = run_alignment_for_project(
                project=self._project,
                audio_path=audio_path,
                lyrics_text=lyrics_text,
                model_name=model_name,
                whisper_language=whisper_language,  # None => auto-detect
                phoneme_language=None,
                device=device,
                beam_size=beam_size,
                patience=patience,
                max_search_window=max_search_window,
                min_similarity=min_similarity,
                alignment_passes=alignment_passes,
                no_speech_threshold=no_speech_threshold,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                condition_on_previous_text=condition_on_previous_text,
                initial_prompt=initial_prompt,
                progress_cb=progress_cb,
            )



        except Exception as e:
            QMessageBox.critical(self, "Alignment error", f"Whisper / alignment failed:\n{e}")
            self.btn_align.setEnabled(True)
            self.lbl_align_status.setText("Error during alignment.")
            return
        finally:
            self.btn_align.setEnabled(True)


        # Normalize alignment result (dict or dataclass)
        self._phrases = getattr(result, "phrases", getattr(result, "lines", [])) or []
        self._words = getattr(result, "words", []) or []

        self.align_progress.setValue(100)
        self.lbl_align_status.setText("Alignment done.")
        self.populate_segments_preview()

    def _format_alignment_summary(self, metrics: dict, chosen_lang: str) -> str:
        """Return a short human-readable alignment report for the UI."""
        if not isinstance(metrics, dict):
            return ""

        total = metrics.get("lyrics_words_total")
        matched = metrics.get("lyrics_words_matched")
        coverage = metrics.get("coverage")
        passes = metrics.get("passes")

        def _fmt_int(x):
            try:
                return str(int(x))
            except Exception:
                return "?"

        def _fmt_float(x, nd=3):
            try:
                return f"{float(x):.{nd}f}"
            except Exception:
                return "?"

        parts = []
        if passes is not None:
            parts.append(f"passes={_fmt_int(passes)}")
        if chosen_lang:
            parts.append(f"chosen_lang={chosen_lang}")
        if total is not None and matched is not None:
            parts.append(f"coverage={_fmt_float(coverage, 3)} ({_fmt_int(matched)}/{_fmt_int(total)})")
        elif coverage is not None:
            parts.append(f"coverage={_fmt_float(coverage, 3)}")

        # Requested debug counters
        for k in (
            "lyrics_words_filled_between",
            "lyrics_words_filled_edges",
            "phrases_inserted",
            "lyrics_words_inserted",
        ):
            if k in metrics:
                parts.append(f"{k}={_fmt_int(metrics.get(k))}")

        # Similarity (useful for diagnosing over-lenient salvage)
        if "mean_similarity" in metrics:
            parts.append(f"mean_similarity={_fmt_float(metrics.get('mean_similarity'), 3)}")
        if "median_similarity" in metrics:
            parts.append(f"median_similarity={_fmt_float(metrics.get('median_similarity'), 3)}")

        # Also keep a JSON-ish one-liner for copy/paste into issues.
        compact_keys = [
            "passes",
            "coverage",
            "lyrics_words_total",
            "lyrics_words_matched",
            "lyrics_words_filled_between",
            "lyrics_words_filled_edges",
            "phrases_inserted",
            "lyrics_words_inserted",
        ]
        compact = {k: metrics.get(k) for k in compact_keys if k in metrics}
        try:
            compact_json = json.dumps(compact, ensure_ascii=False)
        except Exception:
            compact_json = str(compact)

        return "Alignment summary: " + ", ".join(parts) + "\n" + compact_json


    # ------------------------------------------------------------------
    # Phrases list + editing
    # ------------------------------------------------------------------

    def _get_phrase_field(self, phrase: Any, field: str):
        if phrase is None:
            return None
        if isinstance(phrase, dict):
            return phrase.get(field)
        return getattr(phrase, field, None)

    def _set_phrase_field(self, phrase: Any, field: str, value: Any):
        if isinstance(phrase, dict):
            phrase[field] = value
        else:
            setattr(phrase, field, value)

    def _get_word_field(self, word: Any, field: str):
        if word is None:
            return None
        if isinstance(word, dict):
            return word.get(field)
        return getattr(word, field, None)

    def _set_word_field(self, word: Any, field: str, value: Any):
        if isinstance(word, dict):
            word[field] = value
        else:
            setattr(word, field, value)

    def _get_phrase_neighbor_bounds(self, idx: int) -> tuple[float, float]:
        """
        Return (min_s, max_s) allowed for phrase idx, based on previous/next phrases
        and the track duration. Used to prevent phrase overlaps.
        """
        if not self._phrases or idx < 0 or idx >= len(self._phrases):
            return (0.0, max(self._media_duration_ms / 1000.0, 0.01))

        phrase = self._phrases[idx]
        start = self._get_phrase_field(phrase, "start") or 0.0
        end = self._get_phrase_field(phrase, "end") or start

        # Left bound: end of previous phrase (or 0.0)
        min_s = 0.0
        if idx > 0:
            prev = self._phrases[idx - 1]
            prev_end = self._get_phrase_field(prev, "end")
            if prev_end is not None:
                min_s = float(prev_end)

        # Right bound: start of next phrase (or track duration / end+10s)
        if self._media_duration_ms > 0:
            track_s = self._media_duration_ms / 1000.0
        else:
            track_s = max(end, start) + 10.0

        max_s = track_s
        if idx + 1 < len(self._phrases):
            nxt = self._phrases[idx + 1]
            nxt_start = self._get_phrase_field(nxt, "start")
            if nxt_start is not None:
                max_s = float(nxt_start)

        if max_s < min_s + 0.010:
            max_s = min_s + 0.010

        return (min_s, max_s)

    def _get_word_neighbor_bounds(self, line_index: int, word_global_idx: int) -> tuple[float, float]:
        """
        Return (min_s, max_s) allowed for the given word (global index) within a line_index,
        based on previous/next words in that line and the phrase boundaries.
        """
        if line_index is None:
            # Fallback: whole track
            if self._media_duration_ms > 0:
                return (0.0, self._media_duration_ms / 1000.0)
            return (0.0, 9999.0)

        # Collect all words for this line_index
        words_for_line: list[tuple[int, Any]] = []
        for gidx, w in enumerate(self._words):
            if self._get_word_field(w, "line_index") == line_index:
                words_for_line.append((gidx, w))

        if not words_for_line:
            if self._media_duration_ms > 0:
                return (0.0, self._media_duration_ms / 1000.0)
            return (0.0, 9999.0)

        # Sort by lyrics order first (word_index), then by time as tie-breaker
        def _word_sort_key(item: tuple[int, Any]):
            wobj = item[1]
            wi = self._get_word_field(wobj, "word_index")
            ws = self._get_word_field(wobj, "start")
            return (wi if wi is not None else 0, ws if ws is not None else 0.0)

        words_for_line.sort(key=_word_sort_key)

        # Find position of this word in the sorted list
        idx_in_line = None
        for i, (gidx, _) in enumerate(words_for_line):
            if gidx == word_global_idx:
                idx_in_line = i
                break

        # Phrase bounds for this line
        phrase_start = None
        phrase_end = None
        for phrase in self._phrases:
            if self._get_phrase_field(phrase, "line_index") == line_index:
                phrase_start = self._get_phrase_field(phrase, "start")
                phrase_end = self._get_phrase_field(phrase, "end")
                break

        if self._media_duration_ms > 0:
            track_s = self._media_duration_ms / 1000.0
        else:
            track_s = 9999.0

        # Default min/max from phrase or track
        min_s = phrase_start if phrase_start is not None else 0.0
        max_s = phrase_end if phrase_end is not None else track_s

        if idx_in_line is not None:
            # Previous word end as left bound
            if idx_in_line > 0:
                prev_word = words_for_line[idx_in_line - 1][1]
                prev_end = self._get_word_field(prev_word, "end")
                if prev_end is not None:
                    min_s = float(prev_end)
            # Next word start as right bound
            if idx_in_line + 1 < len(words_for_line):
                next_word = words_for_line[idx_in_line + 1][1]
                next_start = self._get_word_field(next_word, "start")
                if next_start is not None:
                    max_s = float(next_start)

        if max_s < min_s + 0.010:
            max_s = min_s + 0.010

        return (min_s, max_s)


    def _on_kara_word_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        """When a word is selected in the words list, sync the 'Selected word timing' panel."""
        if current is None:
            return
        idx = current.data(Qt.ItemDataRole.UserRole)
        if idx is None or not isinstance(idx, int):
            return
        if idx < 0 or idx >= len(self._words):
            return

        w = self._words[idx]
        text = self._get_word_field(w, "text") or ""
        start = self._get_word_field(w, "start") or 0.0
        end = self._get_word_field(w, "end") or 0.0
        line_index = self._get_word_field(w, "line_index")

        self.lbl_word_text.setText(f"Word: {text}")

        # Determine slider bounds from neighbor words and phrase
        if line_index is not None:
            min_s, max_s = self._get_word_neighbor_bounds(line_index, idx)
        else:
            if self._media_duration_ms > 0:
                min_s, max_s = (0.0, self._media_duration_ms / 1000.0)
            else:
                min_s, max_s = (0.0, max(end, start) + 1.0)

        # Clamp start/end in those bounds
        start = max(min_s, min(float(start), max_s))
        end = max(start + 0.010, min(float(end), max_s))

        # Convert to ms for slider
        min_ms = int(min_s * 1000.0)
        max_ms = int(max_s * 1000.0)

        self.word_start_slider.blockSignals(True)
        self.word_end_slider.blockSignals(True)

        self.word_start_slider.setRange(min_ms, max_ms)
        self.word_end_slider.setRange(min_ms, max_ms)

        self.word_start_slider.setValue(int(start * 1000.0))
        self.word_end_slider.setValue(int(end * 1000.0))

        self.word_start_slider.blockSignals(False)
        self.word_end_slider.blockSignals(False)

        self.lbl_word_start_value.setText(f"Start: {start:7.3f} s")
        self.lbl_word_end_value.setText(f"End:   {end:7.3f} s")


    def _on_word_start_slider_moved(self, position_ms: int) -> None:
        """Update start label while dragging the word start slider."""
        seconds = position_ms / 1000.0
        self.lbl_word_start_value.setText(f"Start: {seconds:7.3f} s")

    def _on_word_end_slider_moved(self, position_ms: int) -> None:
        """Update end label while dragging the word end slider."""
        seconds = position_ms / 1000.0
        self.lbl_word_end_value.setText(f"End:   {seconds:7.3f} s")

    def _on_word_sliders_released(self) -> None:
        """
        When either start/end slider is released, apply the new timings
        to the selected word and save to disk.
        The new timings are clamped between previous/next words for the same line
        so that no overlap is created. The owning phrase is expanded if needed
        but phrase neighbors are left unchanged.
        """
        current = self.kara_words_list.currentItem()
        if current is None:
            return
        idx = current.data(Qt.ItemDataRole.UserRole)
        if idx is None or not isinstance(idx, int):
            return
        if idx < 0 or idx >= len(self._words):
            return

        w = self._words[idx]
        line_index = self._get_word_field(w, "line_index")

        # Neighbor-based bounds
        if line_index is not None:
            min_s, max_s = self._get_word_neighbor_bounds(line_index, idx)
        else:
            if self._media_duration_ms > 0:
                min_s, max_s = (0.0, self._media_duration_ms / 1000.0)
            else:
                min_s, max_s = (0.0, 9999.0)

        start_s = self.word_start_slider.value() / 1000.0
        end_s = self.word_end_slider.value() / 1000.0

        # Clamp to neighbor bounds
        if start_s < min_s:
            start_s = min_s
        if end_s > max_s:
            end_s = max_s

        # Ensure end > start
        if end_s <= start_s:
            end_s = min(start_s + 0.010, max_s)
            if end_s <= start_s:
                start_s = max(min_s, max_s - 0.010)
                end_s = max_s

        # Push clamped values back to sliders
        self.word_start_slider.blockSignals(True)
        self.word_end_slider.blockSignals(True)
        self.word_start_slider.setValue(int(start_s * 1000.0))
        self.word_end_slider.setValue(int(end_s * 1000.0))
        self.word_start_slider.blockSignals(False)
        self.word_end_slider.blockSignals(False)

        # Update word timings
        self._set_word_field(w, "start", float(start_s))
        self._set_word_field(w, "end", float(end_s))

        self.lbl_word_start_value.setText(f"Start: {start_s:7.3f} s")
        self.lbl_word_end_value.setText(f"End:   {end_s:7.3f} s")

        # Expand the owning phrase if needed (but do not touch neighbor phrases)
        phrase_idx_for_line: Optional[int] = None
        phrase_obj: Any = None
        if line_index is not None and self._phrases:
            for p_idx, phrase in enumerate(self._phrases):
                if self._get_phrase_field(phrase, "line_index") == line_index:
                    phrase_idx_for_line = p_idx
                    phrase_obj = phrase
                    break

        if phrase_obj is not None:
            pstart = self._get_phrase_field(phrase_obj, "start")
            pend = self._get_phrase_field(phrase_obj, "end")
            changed = False

            if pstart is None or start_s < float(pstart) - 1e-6:
                self._set_phrase_field(phrase_obj, "start", float(start_s))
                changed = True
            if pend is None or end_s > float(pend) + 1e-6:
                self._set_phrase_field(phrase_obj, "end", float(end_s))
                changed = True

            if changed:
                # Refresh phrase lists and keep selection + sliders in sync
                self.populate_segments_preview()
                if phrase_idx_for_line is not None:
                    if 0 <= phrase_idx_for_line < self.segments_list.count():
                        self.segments_list.setCurrentRow(phrase_idx_for_line)
                    if 0 <= phrase_idx_for_line < self.kara_scroll_list.count():
                        self.kara_scroll_list.setCurrentRow(phrase_idx_for_line)
                    self._sync_phrase_timing_panel(phrase_idx_for_line, phrase_obj)

        # Refresh words list for this line (keeps ordering by time)
        if line_index is not None:
            self._refresh_karaoke_words(line_index)

            # Reselect this word (if still present)
            for row in range(self.kara_words_list.count()):
                item = self.kara_words_list.item(row)
                gidx = item.data(Qt.ItemDataRole.UserRole)
                if gidx == idx:
                    self.kara_words_list.setCurrentItem(item)
                    break

        # Persist new timings to JSON (+ SRT via phrases)
        self.lbl_align_status.setText("Updated word timing from sliders.")
        self.save_timings_to_files()


    # --- 10 ms nudge for phrases ---

    def _nudge_phrase_start(self, delta_ms: int) -> None:
        if self.phrase_start_slider.maximum() <= self.phrase_start_slider.minimum():
            return
        new_val = self.phrase_start_slider.value() + delta_ms
        new_val = max(self.phrase_start_slider.minimum(), min(self.phrase_start_slider.maximum(), new_val))
        self.phrase_start_slider.setValue(new_val)
        self._on_phrase_sliders_released()

    def _nudge_phrase_end(self, delta_ms: int) -> None:
        if self.phrase_end_slider.maximum() <= self.phrase_end_slider.minimum():
            return
        new_val = self.phrase_end_slider.value() + delta_ms
        new_val = max(self.phrase_end_slider.minimum(), min(self.phrase_end_slider.maximum(), new_val))
        self.phrase_end_slider.setValue(new_val)
        self._on_phrase_sliders_released()

    def _nudge_phrase_start_minus(self) -> None:
        self._nudge_phrase_start(-10)

    def _nudge_phrase_start_plus(self) -> None:
        self._nudge_phrase_start(+10)

    def _nudge_phrase_end_minus(self) -> None:
        self._nudge_phrase_end(-10)

    def _nudge_phrase_end_plus(self) -> None:
        self._nudge_phrase_end(+10)

    # --- 10 ms nudge for words ---

    def _nudge_word_start(self, delta_ms: int) -> None:
        if self.word_start_slider.maximum() <= self.word_start_slider.minimum():
            return
        new_val = self.word_start_slider.value() + delta_ms
        new_val = max(self.word_start_slider.minimum(), min(self.word_start_slider.maximum(), new_val))
        self.word_start_slider.setValue(new_val)
        self._on_word_sliders_released()

    def _nudge_word_end(self, delta_ms: int) -> None:
        if self.word_end_slider.maximum() <= self.word_end_slider.minimum():
            return
        new_val = self.word_end_slider.value() + delta_ms
        new_val = max(self.word_end_slider.minimum(), min(self.word_end_slider.maximum(), new_val))
        self.word_end_slider.setValue(new_val)
        self._on_word_sliders_released()

    def _nudge_word_start_minus(self) -> None:
        self._nudge_word_start(-10)

    def _nudge_word_start_plus(self) -> None:
        self._nudge_word_start(+10)

    def _nudge_word_end_minus(self) -> None:
        self._nudge_word_end(-10)

    def _nudge_word_end_plus(self) -> None:
        self._nudge_word_end(+10)


    def insert_word_for_current_phrase(self) -> None:
        """Insert a new word into the current phrase (line_index) with manually editable timing."""
        if not self._phrases:
            QMessageBox.warning(self, "No phrases", "No alignment has been run yet.")
            return

        # Need a current phrase selection
        phrase_item = self.kara_scroll_list.currentItem()
        if phrase_item is None:
            QMessageBox.warning(self, "No phrase selected", "Select a phrase in the list first.")
            return

        phrase_idx = phrase_item.data(Qt.ItemDataRole.UserRole)
        if phrase_idx is None or not isinstance(phrase_idx, int):
            return
        if phrase_idx < 0 or phrase_idx >= len(self._phrases):
            return

        phrase = self._phrases[phrase_idx]
        line_index = self._get_phrase_field(phrase, "line_index")
        if line_index is None:
            line_index = phrase_idx

        # Ask for the new word text
        text, ok = QInputDialog.getText(self, "Insert word", "New word text:")
        if not ok or not text.strip():
            return
        text = text.strip()

        # Base time from current playback position
        position_ms = 0
        try:
            position_ms = self.player.position()
        except Exception:
            pass
        base_time = position_ms / 1000.0

        pstart = self._get_phrase_field(phrase, "start")
        pend = self._get_phrase_field(phrase, "end")

        # Clamp base_time in phrase bounds if known
        if pstart is not None and pend is not None:
            if base_time < pstart:
                base_time = float(pstart)
            if base_time > pend:
                base_time = float(pend)

        # Ask user for start time (seconds, 3 decimals)
        default_start = float(base_time)
        start_s, ok = QInputDialog.getDouble(
            self,
            "Word start time",
            "Start (seconds):",
            default_start,
            0.0,
            100000.0,
            3,
        )
        if not ok:
            return

        # Default end = start + 0.200, clamped to phrase end if known
        default_end = start_s + 0.200
        if pstart is not None and pend is not None and default_end > pend:
            default_end = float(pend)

        end_s, ok = QInputDialog.getDouble(
            self,
            "Word end time",
            "End (seconds):",
            default_end,
            start_s,
            100000.0,
            3,
        )
        if not ok:
            return

        if end_s <= start_s:
            end_s = start_s + 0.010

        # Determine word_index: append at the end for this line_index
        word_indices = []
        for w in self._words:
            if self._get_word_field(w, "line_index") == line_index:
                wi = self._get_word_field(w, "word_index")
                if wi is not None:
                    word_indices.append(int(wi))
        if word_indices:
            new_word_index = max(word_indices) + 1
        else:
            new_word_index = 0

        new_word = {
            "line_index": int(line_index),
            "word_index": int(new_word_index),
            "text": text,
            "matched_text": text,
            "start": float(start_s),
            "end": float(end_s),
        }

        self._words.append(new_word)
        global_idx = len(self._words) - 1

        # Refresh words list and select the new word (list is sorted by time)
        self._refresh_karaoke_words(line_index)
        for row in range(self.kara_words_list.count()):
            item = self.kara_words_list.item(row)
            gidx = item.data(Qt.ItemDataRole.UserRole)
            if gidx == global_idx:
                self.kara_words_list.setCurrentItem(item)
                break

        self.lbl_align_status.setText(f"Inserted word '{text}' in phrase {phrase_idx + 1}.")
        self.save_timings_to_files()


    def delete_selected_word(self) -> None:
        """Delete the currently selected word from the alignment."""
        current = self.kara_words_list.currentItem()
        if current is None:
            QMessageBox.warning(self, "No word selected", "Select a word in the right list first.")
            return

        idx = current.data(Qt.ItemDataRole.UserRole)
        if idx is None or not isinstance(idx, int):
            return
        if idx < 0 or idx >= len(self._words):
            return

        w = self._words[idx]
        line_index = self._get_word_field(w, "line_index")
        word_index = self._get_word_field(w, "word_index")
        text = self._get_word_field(w, "text") or ""

        # Remove the word
        del self._words[idx]

        # Optionally reindex following word_index for this line_index
        if line_index is not None and word_index is not None:
            for w2 in self._words:
                if self._get_word_field(w2, "line_index") == line_index:
                    wi2 = self._get_word_field(w2, "word_index")
                    if wi2 is not None and wi2 > word_index:
                        self._set_word_field(w2, "word_index", int(wi2) - 1)

        # Refresh words list
        self._refresh_karaoke_words(line_index)

        # Clear word panel
        self.lbl_word_text.setText("(no word selected)")
        self.lbl_word_start_value.setText("Start: 0.000 s")
        self.lbl_word_end_value.setText("End:   0.000 s")
        self.word_start_slider.setRange(0, 0)
        self.word_end_slider.setRange(0, 0)

        self.lbl_align_status.setText(f"Deleted word '{text}'.")
        self.save_timings_to_files()


    def populate_segments_preview(self):
        self.segments_list.clear()

        if not self._phrases:
            placeholder = QListWidgetItem("No aligned phrases yet.")
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            self.segments_list.addItem(placeholder)
            return

        for idx, phrase in enumerate(self._phrases):
            text = self._get_phrase_field(phrase, "text") or ""
            start = self._get_phrase_field(phrase, "start")
            end = self._get_phrase_field(phrase, "end")
            if start is None or end is None:
                times = "[no timing]"
            else:
                times = f"[{start:7.3f} → {end:7.3f}]"
            item = QListWidgetItem(f"{idx + 1:03d} {times}  {text}")
            item.setData(Qt.ItemDataRole.UserRole, idx)
            self.segments_list.addItem(item)
            
        # Also refresh karaoke scrolling list on the right
        self._refresh_karaoke_scroll()            

    def _refresh_karaoke_scroll(self) -> None:
        """Populate phrase lists (Numeric and Visual) with all phrases."""

        kara_list = getattr(self, "kara_scroll_list", None)
        vis_list = getattr(self, "visual_phrase_list", None)

        if kara_list is not None:
            kara_list.clear()
        if vis_list is not None:
            vis_list.clear()

        if not self._phrases:
            return

        for idx, phrase in enumerate(self._phrases):
            text = self._get_phrase_field(phrase, "text") or ""
            start = self._get_phrase_field(phrase, "start")
            end = self._get_phrase_field(phrase, "end")
            if start is None or end is None:
                times = "[no timing]"
            else:
                times = f"[{start:7.3f} → {end:7.3f}]"

            label = f"{times}  {text}"

            if kara_list is not None:
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, idx)
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft)
                kara_list.addItem(item)

            if vis_list is not None:
                item2 = QListWidgetItem(label)
                item2.setData(Qt.ItemDataRole.UserRole, idx)
                item2.setTextAlignment(Qt.AlignmentFlag.AlignLeft)
                vis_list.addItem(item2)


    def _refresh_karaoke_words(self, line_index: Optional[int]) -> None:
        """
        Populate the words column with all words belonging to the given lyrics line_index.
        If line_index is None, the list is cleared.
        """
        if not hasattr(self, "kara_words_list"):
            return

        self.kara_words_list.clear()

        if line_index is None or not self._words:
            return

        # Filter words for the given line_index, keeping their global index
        filtered = []
        for global_idx, w in enumerate(self._words):
            if self._get_word_field(w, "line_index") == line_index:
                filtered.append((global_idx, w))

        if not filtered:
            return

        def _word_sort_key(item):
            wobj = item[1]
            wi = self._get_word_field(wobj, "word_index")
            ws = self._get_word_field(wobj, "start")
            # Put unknown word_index at the end instead of the beginning:
            if wi is None:
                wi = 10**9
            if ws is None:
                ws = 0.0
            return (int(wi), float(ws))


        for global_idx, w in filtered:
            text = self._get_word_field(w, "text") or ""
            ws = self._get_word_field(w, "start")
            we = self._get_word_field(w, "end")
            if ws is None or we is None:
                times = "[no timing]"
            else:
                times = f"[{ws:7.3f} → {we:7.3f}]"
            item = QListWidgetItem(f"{times}  {text}")
            item.setTextAlignment(Qt.AlignmentFlag.AlignLeft)
            # store global word index for editing
            item.setData(Qt.ItemDataRole.UserRole, global_idx)
            self.kara_words_list.addItem(item)

    def _sync_phrase_timing_panel(self, idx: Optional[int], phrase: Any) -> None:
        """
        Update the 'Selected phrase timing' sliders and labels based on the given phrase.
        Does NOT seek audio or change selection; pure UI sync.
        Slider range is constrained by previous/next phrase to avoid overlaps.
        """
        if idx is None or idx < 0 or idx >= len(self._phrases):
            return

        raw_start = self._get_phrase_field(phrase, "start") or 0.0
        raw_end = self._get_phrase_field(phrase, "end") or raw_start

        min_s, max_s = self._get_phrase_neighbor_bounds(idx)

        # Clamp phrase inside neighbor bounds
        start = max(min_s, min(float(raw_start), max_s))
        end = max(start + 0.010, min(float(raw_end), max_s))

        min_ms = int(min_s * 1000.0)
        max_ms = int(max_s * 1000.0)

        self.phrase_start_slider.blockSignals(True)
        self.phrase_end_slider.blockSignals(True)

        self.phrase_start_slider.setRange(min_ms, max_ms)
        self.phrase_end_slider.setRange(min_ms, max_ms)

        self.phrase_start_slider.setValue(int(start * 1000.0))
        self.phrase_end_slider.setValue(int(end * 1000.0))

        self.phrase_start_slider.blockSignals(False)
        self.phrase_end_slider.blockSignals(False)

        self.lbl_phrase_start_value.setText(f"Start: {start:7.3f} s")
        self.lbl_phrase_end_value.setText(f"End:   {end:7.3f} s")
    def _refresh_visual_editor_for_phrase(self, phrase_idx: int) -> None:
        """Push the selected phrase + its words to the timeline widget."""
        if not hasattr(self, "visual_timeline") or self.visual_timeline is None:
            return
        if phrase_idx < 0 or phrase_idx >= len(self._phrases):
            return

        phrase = self._phrases[phrase_idx]
        p_start = float(self._get_phrase_field(phrase, "start") or 0.0)
        p_end = float(self._get_phrase_field(phrase, "end") or p_start)
        line_index = self._get_phrase_field(phrase, "line_index")
        if line_index is None:
            line_index = phrase_idx

        min_s, max_s = self._get_phrase_neighbor_bounds(phrase_idx)

        words_for_line = []
        for gidx, w in enumerate(self._words):
            if self._get_word_field(w, "line_index") == line_index:
                words_for_line.append(
                    {
                        "gidx": gidx,
                        "text": self._get_word_field(w, "text") or "",
                        "start": float(self._get_word_field(w, "start") or p_start),
                        "end": float(self._get_word_field(w, "end") or p_start),
                        "word_index": int(self._get_word_field(w, "word_index") or 0),
                    }
                )

        self.visual_timeline.set_phrase(
            phrase_idx=phrase_idx,
            phrase_start=p_start,
            phrase_end=p_end,
            words=words_for_line,
            phrase_min=float(min_s),
            phrase_max=float(max_s),
        )

        self._set_visual_word_editor_enabled(False)

    def _set_visual_word_editor_enabled(self, enabled: bool) -> None:
        if not hasattr(self, "edit_visual_word_text"):
            return
        self.edit_visual_word_text.setEnabled(enabled)
        self.spin_visual_word_start.setEnabled(enabled)
        self.spin_visual_word_end.setEnabled(enabled)
        self.btn_visual_apply_word.setEnabled(enabled)
        self.btn_visual_delete_word.setEnabled(enabled)

        if not enabled:
            self.edit_visual_word_text.setText("")
            self.spin_visual_word_start.setValue(0.0)
            self.spin_visual_word_end.setValue(0.0)

    def _on_visual_word_selected(self, gidx_obj):
        if gidx_obj is None:
            self._set_visual_word_editor_enabled(False)
            return

        gidx = int(gidx_obj)
        if gidx < 0 or gidx >= len(self._words):
            self._set_visual_word_editor_enabled(False)
            return

        w = self._words[gidx]
        self._set_visual_word_editor_enabled(True)
        self.edit_visual_word_text.setText(str(self._get_word_field(w, "text") or ""))
        self.spin_visual_word_start.setValue(float(self._get_word_field(w, "start") or 0.0))
        self.spin_visual_word_end.setValue(float(self._get_word_field(w, "end") or 0.0))

    def _on_visual_word_timings_changed(self, payload: list) -> None:
        if not payload:
            return

        for gidx, new_s, new_e in payload:
            if gidx < 0 or gidx >= len(self._words):
                continue
            self._set_word_field(self._words[gidx], "start", float(new_s))
            self._set_word_field(self._words[gidx], "end", float(new_e))

        if self._current_phrase_idx is not None and 0 <= self._current_phrase_idx < len(self._phrases):
            line_index = self._get_phrase_field(self._phrases[self._current_phrase_idx], "line_index")
            if line_index is None:
                line_index = self._current_phrase_idx
            self._refresh_karaoke_words(line_index)

    def _on_visual_phrase_range_changed(self, new_start: float, new_end: float) -> None:
        if self._current_phrase_idx is None:
            return
        idx = int(self._current_phrase_idx)
        if idx < 0 or idx >= len(self._phrases):
            return

        self._set_phrase_field(self._phrases[idx], "start", float(new_start))
        self._set_phrase_field(self._phrases[idx], "end", float(new_end))

        self._refresh_karaoke_scroll()
        self._syncing_phrase_selection = True
        try:
            if hasattr(self, "kara_scroll_list") and self.kara_scroll_list is not None:
                self.kara_scroll_list.setCurrentRow(idx)
            if hasattr(self, "visual_phrase_list") and self.visual_phrase_list is not None:
                self.visual_phrase_list.setCurrentRow(idx)
        finally:
            self._syncing_phrase_selection = False

        self._sync_phrase_timing_panel(idx, self._phrases[idx])

    def _apply_visual_word_edit(self) -> None:
        if not hasattr(self, "visual_timeline") or self.visual_timeline is None:
            return
        gidx = self.visual_timeline.get_selected_global_idx()
        if gidx is None or gidx < 0 or gidx >= len(self._words):
            return

        new_text = self.edit_visual_word_text.text().strip()
        new_s = float(self.spin_visual_word_start.value())
        new_e = float(self.spin_visual_word_end.value())
        if new_e < new_s:
            new_e = new_s

        self._set_word_field(self._words[gidx], "text", new_text)
        self._set_word_field(self._words[gidx], "start", new_s)
        self._set_word_field(self._words[gidx], "end", new_e)

        if self._current_phrase_idx is not None:
            self._refresh_visual_editor_for_phrase(self._current_phrase_idx)
            self.visual_timeline.set_selected_global_idx(gidx)

            line_index = self._get_phrase_field(self._phrases[self._current_phrase_idx], "line_index")
            if line_index is None:
                line_index = self._current_phrase_idx
            self._refresh_karaoke_words(line_index)

    def _insert_visual_word(self) -> None:
        """Insert a new word at the current *visual cursor* (playhead).

        The cursor can be moved by clicking/dragging the playhead in the visual timeline.
        We do NOT auto-fill gaps; the word is inserted into the currently available gap
        between its nearest matched neighbors around the cursor.
        """
        if self._current_phrase_idx is None:
            return
        phrase = self._phrases[self._current_phrase_idx]
        line_index = self._get_phrase_field(phrase, "line_index")
        if line_index is None:
            line_index = self._current_phrase_idx

        word_text, ok = QInputDialog.getText(self, "Insert word", "Word text:")
        if not ok:
            return
        word_text = (word_text or "").strip()
        if not word_text:
            return

        p_start = float(self._get_phrase_field(phrase, "start") or 0.0)
        p_end = float(self._get_phrase_field(phrase, "end") or p_start)
        if p_end <= p_start:
            p_end = p_start + 0.25

        # Prefer the explicit cursor/playhead set by the user in the visual timeline.
        cursor_s: float
        if getattr(self, "_visual_cursor_time_s", None) is not None:
            cursor_s = float(self._visual_cursor_time_s)
        else:
            cursor_s = float(self.player.position()) / 1000.0 if self.player is not None else p_start
        cursor_s = max(p_start, min(cursor_s, p_end))

        # Collect existing word intervals for the current line (allowing gaps).
        local_words = []
        for gidx, w in enumerate(self._words):
            if self._get_word_field(w, "line_index") == line_index:
                ws = float(self._get_word_field(w, "start") or p_start)
                we = float(self._get_word_field(w, "end") or ws)
                local_words.append((ws, we, gidx))
        local_words.sort(key=lambda it: (it[0], it[1]))

        # Determine neighbor bounds around the cursor.
        prev_end = p_start
        next_start = p_end

        # If the cursor falls inside an existing word, insert *after* it by default.
        inside_idx = None
        for ws, we, gidx in local_words:
            if ws <= cursor_s <= we:
                inside_idx = gidx
                prev_end = we
                break

        if inside_idx is None:
            # Find the closest previous word (by end time)
            for ws, we, gidx in local_words:
                if we <= cursor_s:
                    prev_end = max(prev_end, we)

        # Find the closest next word start strictly after prev_end (or cursor if no prev)
        for ws, we, gidx in local_words:
            if ws >= max(cursor_s, prev_end) and ws < next_start:
                next_start = ws
                break

        gap_start = float(prev_end)
        gap_end = float(next_start)

        min_dur = 0.060
        default_dur = 0.200

        if gap_end - gap_start < min_dur:
            QMessageBox.warning(
                self,
                "No room",
                "No free space at the current cursor position (neighbor overlap).\nMove the playhead to a gap or adjust neighbors first.",
            )
            return

        new_s = max(gap_start, min(cursor_s, gap_end - min_dur))
        new_e = min(gap_end, new_s + min(default_dur, max(0.0, gap_end - new_s)))
        if new_e - new_s < min_dur:
            new_e = min(gap_end, new_s + min_dur)

        new_word = {
            "line_index": int(line_index),
            "text": word_text,
            "start": float(new_s),
            "end": float(new_e),
        }
        self._words.append(new_word)
        new_gidx = len(self._words) - 1
        self._reindex_words_for_line(int(line_index))
        
        self._refresh_karaoke_words(line_index)
        self._refresh_visual_editor_for_phrase(self._current_phrase_idx)

        # Select the newly inserted word for immediate editing
        try:
            if hasattr(self, "visual_timeline") and self.visual_timeline is not None:
                self.visual_timeline.set_selected_global_idx(new_gidx)
        except Exception:
            pass


    def _delete_visual_word(self) -> None:
        if not hasattr(self, "visual_timeline") or self.visual_timeline is None:
            return
        gidx = self.visual_timeline.get_selected_global_idx()
        if gidx is None or gidx < 0 or gidx >= len(self._words):
            return

        w = self._words[gidx]
        txt = str(self._get_word_field(w, "text") or "")
        reply = QMessageBox.question(
            self,
            "Delete word",
            f"Delete this word?\n\n{txt}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        del self._words[gidx]
        self._reindex_words_for_line(int(line_index))

        if self._current_phrase_idx is not None:
            phrase = self._phrases[self._current_phrase_idx]
            line_index = self._get_phrase_field(phrase, "line_index")
            if line_index is None:
                line_index = self._current_phrase_idx
            self._refresh_karaoke_words(line_index)
            self._refresh_visual_editor_for_phrase(self._current_phrase_idx)

    def _play_current_visual_phrase(self) -> None:
        if self._current_phrase_idx is None or self._current_phrase_idx >= len(self._phrases):
            return
        phrase = self._phrases[self._current_phrase_idx]
        p_start = float(self._get_phrase_field(phrase, "start") or 0.0)
        p_end = float(self._get_phrase_field(phrase, "end") or p_start)

        pre = float(self.spin_visual_preroll.value())
        post = float(self.spin_visual_postroll.value())

        start_s = max(0.0, p_start - pre)
        end_s = max(start_s, p_end + post)

        audio_path = self._get_audio_path_for_current_mode()
        if audio_path:
            if self.player.source().toString() != QUrl.fromLocalFile(str(audio_path)).toString():
                self.player.setSource(QUrl.fromLocalFile(str(audio_path)))

        self._visual_loop_enabled = bool(self.chk_visual_loop.isChecked())

        # Apply visual slow-motion (phrase playback only).
        # We snapshot the previous rate so we can restore it when leaving the visual tab.
        try:
            if self._visual_prev_playback_rate is None and hasattr(self.player, "playbackRate"):
                self._visual_prev_playback_rate = float(self.player.playbackRate())
        except Exception:
            self._visual_prev_playback_rate = self._visual_prev_playback_rate or 1.0

        rate = 1.0
        try:
            if hasattr(self, "combo_visual_speed"):
                data = self.combo_visual_speed.currentData()
                if data is not None:
                    rate = float(data)
        except Exception:
            rate = 1.0

        try:
            if hasattr(self.player, "setPlaybackRate"):
                self.player.setPlaybackRate(rate)
        except Exception:
            pass

        self._visual_loop_start_s = float(start_s)
        self._visual_loop_end_s = float(end_s)
        self._visual_play_active = True

        # Visual helper: show the playback window and reset playhead in the timeline
        try:
            if hasattr(self, "visual_timeline"):
                self.visual_timeline.set_play_window(start_s, end_s)
                self.visual_timeline.set_playhead(start_s)
        except Exception:
            pass

        self.player.setPosition(int(start_s * 1000.0))
        self.player.play()

    def _stop_visual_phrase(self) -> None:
        self._visual_play_active = False
        self._visual_loop_enabled = False
        try:
            self.player.pause()
        except Exception:
            pass

        # Clear the playback window overlay in the timeline
        try:
            if hasattr(self, "visual_timeline"):
                self.visual_timeline.set_play_window(None, None)
        except Exception:
            pass

        # Restore global playback speed (slow-motion is visual-tab only).
        try:
            if self._visual_prev_playback_rate is not None and hasattr(self.player, "setPlaybackRate"):
                self.player.setPlaybackRate(float(self._visual_prev_playback_rate))
        except Exception:
            pass
        self._visual_prev_playback_rate = None

    def _on_visual_loop_toggled(self, checked: bool) -> None:
        """Keep the visual playback loop flag in sync with the checkbox."""
        try:
            self._visual_loop_enabled = bool(checked)
        except Exception:
            self._visual_loop_enabled = False

    def _reset_alignment_to_saved_snapshot(self) -> None:
        if not self._saved_phrases_snapshot:
            QMessageBox.information(self, "Reset", "No saved snapshot available yet. Save timings first.")
            return

        reply = QMessageBox.question(
            self,
            "Reset timings",
            "Reset all phrase/word timings to the last saved state?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._phrases = copy.deepcopy(self._saved_phrases_snapshot)
        self._words = copy.deepcopy(self._saved_words_snapshot)

        self._refresh_karaoke_scroll()

        if self._current_phrase_idx is not None and self._current_phrase_idx < len(self._phrases):
            self._select_phrase_idx(self._current_phrase_idx, source="visual")
        elif self._phrases:
            self._select_phrase_idx(0, source="visual")


    def _on_kara_phrase_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Numeric tab phrase selection handler (syncs with Visual tab)."""
        if getattr(self, "_syncing_phrase_selection", False):
            return
        if current is None:
            return
        idx = current.data(Qt.ItemDataRole.UserRole)
        if idx is None or not isinstance(idx, int):
            return
        self._select_phrase_idx(idx, source="kara")

    
    def _on_kara_phrase_clicked(self, item: QListWidgetItem) -> None:
        """User clicked a phrase in the Numeric editor list.

        `currentItemChanged` does not fire when the user clicks the already-selected item.
        This handler ensures we can always restart playback from the phrase start.
        """
        if item is None:
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is None or not isinstance(idx, int):
            return
        self._select_phrase_idx(idx, source="kara")
        try:
            self.player.play()
        except Exception:
            pass

    def _on_visual_phrase_clicked(self, item: QListWidgetItem) -> None:
        """User clicked a phrase in the Visual editor list (always restart)."""
        if item is None:
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is None or not isinstance(idx, int):
            return
        self._select_phrase_idx(idx, source="visual")
        try:
            self.player.play()
        except Exception:
            pass

    def _on_visual_timeline_playhead_moved(self, t_s: float) -> None:
        """Seek the player when the user moves the playhead in the Visual timeline.

        This makes 'Insert word' predictable: it will be inserted at (or just after) this cursor.
        """
        try:
            self._visual_cursor_time_s = float(t_s)
        except Exception:
            self._visual_cursor_time_s = None

        # Keep audio and timeline in sync
        try:
            if self.player is not None:
                self.player.setPosition(int(float(t_s) * 1000.0))
        except Exception:
            pass

    def _on_visual_phrase_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Visual tab phrase selection handler (syncs with Numeric tab)."""
        if getattr(self, "_syncing_phrase_selection", False):
            return
        if current is None:
            return
        idx = current.data(Qt.ItemDataRole.UserRole)
        if idx is None or not isinstance(idx, int):
            return
        self._select_phrase_idx(idx, source="visual")

    def _select_phrase_idx(self, idx: int, source: str = "kara"):
        """Shared phrase selection logic for Numeric and Visual editors."""
        if idx < 0 or idx >= len(self._phrases):
            return

        self._current_phrase_idx = int(idx)
        phrase = self._phrases[idx]

        start = float(self._get_phrase_field(phrase, "start") or 0.0)

        # Determine line_index for words
        line_index = self._get_phrase_field(phrase, "line_index")
        if line_index is None:
            line_index = idx

        # Sync both phrase lists (avoid recursion)
        self._syncing_phrase_selection = True
        try:
            if source != "kara" and hasattr(self, "kara_scroll_list") and self.kara_scroll_list is not None:
                if self.kara_scroll_list.currentRow() != idx:
                    self.kara_scroll_list.setCurrentRow(idx)
            if source != "visual" and hasattr(self, "visual_phrase_list") and self.visual_phrase_list is not None:
                if self.visual_phrase_list.currentRow() != idx:
                    self.visual_phrase_list.setCurrentRow(idx)
        finally:
            self._syncing_phrase_selection = False

        # Sync the 'Selected phrase timing' panel UI (no audio seek here)
        self._sync_phrase_timing_panel(idx, phrase)

        # Seek audio to the beginning of this phrase
        position_ms = int(start * 1000.0)
        if hasattr(self, "kara_slider"):
            self.kara_slider.blockSignals(True)
            self.kara_slider.setValue(position_ms)
            self.kara_slider.blockSignals(False)

        if hasattr(self, "lbl_kara_track_time"):
            self.lbl_kara_track_time.setText(f"{start:7.3f} s")

        self.player.setPosition(position_ms)

        # Update karaoke labels/highlight immediately
        self._update_karaoke_for_time(start)

        # Update the words column for this phrase (numeric tab)
        self._refresh_karaoke_words(line_index)

        # Update the visual editor (if available)
        self._refresh_visual_editor_for_phrase(idx)



    def _on_phrase_start_slider_moved(self, position_ms: int) -> None:
        """Update start label while dragging the start slider."""
        seconds = position_ms / 1000.0
        self.lbl_phrase_start_value.setText(f"Start: {seconds:7.3f} s")

    def _on_phrase_end_slider_moved(self, position_ms: int) -> None:
        """Update end label while dragging the end slider."""
        seconds = position_ms / 1000.0
        self.lbl_phrase_end_value.setText(f"End:   {seconds:7.3f} s")

    def _on_phrase_sliders_released(self) -> None:
        """
        When either start/end slider is released, apply the new timings
        to the selected phrase, clamped between previous/next phrases
        so that no overlap is created, then save to disk.
        """
        current = self.kara_scroll_list.currentItem()
        if current is None:
            return
        idx = current.data(Qt.ItemDataRole.UserRole)
        if idx is None or not isinstance(idx, int):
            return
        if idx < 0 or idx >= len(self._phrases):
            return

        min_s, max_s = self._get_phrase_neighbor_bounds(idx)

        start_s = self.phrase_start_slider.value() / 1000.0
        end_s = self.phrase_end_slider.value() / 1000.0

        # Clamp to neighbor bounds
        if start_s < min_s:
            start_s = min_s
        if end_s > max_s:
            end_s = max_s

        # Ensure end > start
        if end_s <= start_s:
            end_s = min(start_s + 0.010, max_s)
            if end_s <= start_s:
                # Degenerate case: force minimal span inside [min_s, max_s]
                start_s = max(min_s, max_s - 0.010)
                end_s = max_s

        # Push clamped values back to sliders
        self.phrase_start_slider.blockSignals(True)
        self.phrase_end_slider.blockSignals(True)
        self.phrase_start_slider.setValue(int(start_s * 1000.0))
        self.phrase_end_slider.setValue(int(end_s * 1000.0))
        self.phrase_start_slider.blockSignals(False)
        self.phrase_end_slider.blockSignals(False)

        phrase = self._phrases[idx]
        self._set_phrase_field(phrase, "start", float(start_s))
        self._set_phrase_field(phrase, "end", float(end_s))

        # Refresh both alignment list and karaoke list
        self.populate_segments_preview()

        # Restore selection in both lists
        if 0 <= idx < self.segments_list.count():
            self.segments_list.setCurrentRow(idx)
        if 0 <= idx < self.kara_scroll_list.count():
            self.kara_scroll_list.setCurrentRow(idx)

        # Update status and labels
        self.lbl_align_status.setText(f"Updated phrase {idx + 1} from sliders.")
        self.lbl_phrase_start_value.setText(f"Start: {start_s:7.3f} s")
        self.lbl_phrase_end_value.setText(f"End:   {end_s:7.3f} s")

        # Persist new timings to JSON + SRT
        self.save_timings_to_files()

    def on_segment_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        if current is None:
            return
        idx = current.data(Qt.ItemDataRole.UserRole)
        if idx is None or not isinstance(idx, int):
            return
        if idx < 0 or idx >= len(self._phrases):
            return

        phrase = self._phrases[idx]
        start = self._get_phrase_field(phrase, "start") or 0.0
        end = self._get_phrase_field(phrase, "end") or 0.0

        self.spin_start.blockSignals(True)
        self.spin_end.blockSignals(True)
        self.spin_start.setValue(float(start))
        self.spin_end.setValue(float(end))
        self.spin_start.blockSignals(False)
        self.spin_end.blockSignals(False)

    def apply_timing_to_selected(self):
        current = self.segments_list.currentItem()
        if current is None:
            QMessageBox.warning(self, "No phrase selected", "Select a phrase in the list first.")
            return
        idx = current.data(Qt.ItemDataRole.UserRole)
        if idx is None or not isinstance(idx, int):
            return
        if idx < 0 or idx >= len(self._phrases):
            return

        start = float(self.spin_start.value())
        end = float(self.spin_end.value())
        if end <= start:
            QMessageBox.warning(self, "Invalid timing", "End time must be greater than start time.")
            return

        phrase = self._phrases[idx]
        self._set_phrase_field(phrase, "start", start)
        self._set_phrase_field(phrase, "end", end)

        text = self._get_phrase_field(phrase, "text") or ""
        times = f"[{start:7.3f} → {end:7.3f}]"
        current.setText(f"{idx + 1:03d} {times}  {text}")
        self.lbl_align_status.setText(f"Updated timing of phrase {idx + 1}.")

    def _reindex_words_for_line(self, line_index: int) -> None:
        """
        Ensure every word of the given line_index has a consistent 'word_index'
        based on time order. This makes repeated-word highlighting deterministic.
        """
        items = []
        for gidx, w in enumerate(self._words or []):
            if self._get_word_field(w, "line_index") != line_index:
                continue
            ws = float(self._get_word_field(w, "start") or 0.0)
            we = float(self._get_word_field(w, "end") or ws)
            items.append((ws, we, gidx, w))

        items.sort(key=lambda it: (it[0], it[1], it[2]))

        for new_idx, (_, __, ___, w) in enumerate(items):
            self._set_word_field(w, "word_index", int(new_idx))


    # ------------------------------------------------------------------
    # Save timings → JSON + SRT
    # ------------------------------------------------------------------

    def _format_srt_timestamp(self, t: float) -> str:
        if t < 0:
            t = 0.0
        hours = int(t // 3600)
        minutes = int((t % 3600) // 60)
        seconds = int(t % 60)
        millis = int(round((t - int(t)) * 1000))
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

    def _parse_srt_timestamp(self, s: str) -> float:
        """Parse 'HH:MM:SS,mmm' or 'HH:MM:SS.mmm' into float seconds."""
        s = s.strip().replace(",", ".")
        parts = s.split(":")
        if len(parts) != 3:
            return 0.0
        h, m, sec = parts
        try:
            return int(h) * 3600 + int(m) * 60 + float(sec)
        except ValueError:
            return 0.0

    def save_timings_to_files(self):
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        if not self._phrases:
            QMessageBox.warning(self, "No phrases", "No alignment has been run yet.")
            return

        align_dir = self._project.folder / "vocal_align"
        align_dir.mkdir(parents=True, exist_ok=True)
        phrases_path = align_dir / "phrases.json"
        srt_path = align_dir / "subtitles.srt"
        words_path = align_dir / "words.json"

        # Save phrases JSON (convert to plain dicts so dataclasses are serializable)
        try:
            phrases_json = []
            for phrase in self._phrases:
                phrases_json.append(
                    {
                        "line_index": self._get_phrase_field(phrase, "line_index"),
                        "text": self._get_phrase_field(phrase, "text"),
                        "start": self._get_phrase_field(phrase, "start"),
                        "end": self._get_phrase_field(phrase, "end"),
                    }
                )

            with phrases_path.open("w", encoding="utf-8") as f:
                json.dump(phrases_json, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not write phrases.json:\n{e}")
            return

        # Save words JSON (if any) – déjà en dicts uniformisés
        try:
            words_json = []
            for w in self._words:
                words_json.append(
                    {
                        "line_index": self._get_word_field(w, "line_index"),
                        "word_index": self._get_word_field(w, "word_index"),
                        "text": self._get_word_field(w, "text"),
                        "matched_text": self._get_word_field(w, "matched_text"),
                        "start": self._get_word_field(w, "start"),
                        "end": self._get_word_field(w, "end"),
                    }
                )
            with words_path.open("w", encoding="utf-8") as f:
                json.dump(words_json, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not write words.json:\n{e}")
            return

        # Save SRT (comme avant)
        try:
            with srt_path.open("w", encoding="utf-8") as f:
                idx_srt = 1
                for phrase in self._phrases:
                    text = self._get_phrase_field(phrase, "text") or ""
                    start = self._get_phrase_field(phrase, "start")
                    end = self._get_phrase_field(phrase, "end")
                    if not text or start is None or end is None:
                        continue
                    start_srt = self._format_srt_timestamp(float(start))
                    end_srt = self._format_srt_timestamp(float(end))
                    f.write(f"{idx_srt}\n")
                    f.write(f"{start_srt} --> {end_srt}\n")
                    f.write(text + "\n\n")
                    idx_srt += 1
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not write SRT file:\n{e}")
            return

        # Status message only (no popup)
        # Update "last saved" snapshot (used by Reset)
        self._saved_phrases_snapshot = copy.deepcopy(self._phrases)
        self._saved_words_snapshot = copy.deepcopy(self._words)
        self.lbl_align_status.setText(
            f"Timings saved to phrases.json, words.json and SRT."
        )

    def _get_project_id_for_export(self) -> str:
        """Return a short ID used to suffix exported filenames."""
        if self._project is not None and getattr(self._project, "id", None):
            return self._project.id
        return "project"

    def _get_alignment_dir(self) -> Optional[Path]:
        """Return the vocal_align folder for the current project."""
        if not self._project:
            return None
        return self._project.folder / "vocal_align"

    # -------------------- SRT --------------------

    def export_srt(self) -> None:
        """Export subtitles.srt (for YouTube / DaVinci Resolve)."""
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        align_dir = self._get_alignment_dir()
        assert align_dir is not None
        src = align_dir / "subtitles.srt"
        if not src.is_file():
            QMessageBox.warning(
                self,
                "File not found",
                "No 'subtitles.srt' found in the vocal_align folder.\n"
                "Please run the alignment and save timings first.",
            )
            return

        project_id = self._get_project_id_for_export()
        default_name = f"subtitles_{project_id}.srt"
        default_path = align_dir / default_name

        dest_path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Export SRT subtitles",
            str(default_path),
            "SubRip subtitles (*.srt);;All files (*.*)",
        )
        if not dest_path_str:
            return

        dest_path = Path(dest_path_str)
        try:
            shutil.copy2(src, dest_path)
        except Exception as e:
            QMessageBox.critical(self, "Export error", f"Could not export SRT file:\n{e}")
            return

        self.lbl_export_status.setText(f"Exported SRT to: {dest_path}")

    # -------------------- JSON / CSV helpers --------------------

    def _export_json_with_dialog(self, src_name: str, default_base: str, title: str) -> None:
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return
        align_dir = self._get_alignment_dir()
        assert align_dir is not None
        src = align_dir / src_name
        if not src.is_file():
            QMessageBox.warning(
                self,
                "File not found",
                f"No '{src_name}' found in the vocal_align folder.\n"
                "Please run the alignment and save timings first.",
            )
            return
        project_id = self._get_project_id_for_export()
        default_name = f"{default_base}_{project_id}.json"
        default_path = align_dir / default_name

        dest_str, _ = QFileDialog.getSaveFileName(
            self,
            title,
            str(default_path),
            "JSON files (*.json);;All files (*.*)",
        )
        if not dest_str:
            return
        dest = Path(dest_str)
        try:
            shutil.copy2(src, dest)
        except Exception as e:
            QMessageBox.critical(self, "Export error", f"Could not export JSON file:\n{e}")
            return
        self.lbl_export_status.setText(f"Exported {src_name} to: {dest}")


    # -------------------- Phrases --------------------

    def export_phrases_json(self) -> None:
        self._export_json_with_dialog(
            src_name="phrases.json",
            default_base="phrases",
            title="Export phrase timings (JSON)",
        )

    # -------------------- Words --------------------

    def export_words_json(self) -> None:
        self._export_json_with_dialog(
            src_name="words.json",
            default_base="words",
            title="Export word timings (JSON)",
        )

    # -------------------- Phonemes --------------------

    def export_phonemes_json(self) -> None:
        self._export_json_with_dialog(
            src_name="phonemes.json",
            default_base="phonemes",
            title="Export phoneme timings (JSON)",
        )

    # -------------------- Lyrics TXT --------------------

    def export_lyrics_txt(self) -> None:
        """Export lyrics text as a plain TXT file."""
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        text = getattr(self._project, "lyrics_text", "") or ""
        if not text.strip():
            QMessageBox.warning(
                self,
                "No lyrics",
                "No lyrics text is stored for this project.",
            )
            return

        project_id = self._get_project_id_for_export()
        default_name = f"lyrics_{project_id}.txt"
        default_path = self._project.folder / default_name

        dest_str, _ = QFileDialog.getSaveFileName(
            self,
            "Export lyrics (TXT)",
            str(default_path),
            "Text files (*.txt);;All files (*.*)",
        )
        if not dest_str:
            return
        dest = Path(dest_str)

        try:
            dest.write_text(text, encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Export error", f"Could not write TXT file:\n{e}")
            return

        self.lbl_export_status.setText(f"Exported lyrics TXT to: {dest}")


    def export_alignment_files(self) -> None:
        """Export SRT, words/phonemes JSON and lyrics TXT to a user-chosen folder."""
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        align_dir = self._project.folder / "vocal_align"
        if not align_dir.is_dir():
            QMessageBox.warning(
                self,
                "No alignment folder",
                "No 'vocal_align' folder was found for this project.\n"
                "Please run the alignment first.",
            )
            return

        # Let the user choose an export directory (e.g. for YouTube / DaVinci)
        target_dir_str = QFileDialog.getExistingDirectory(
            self,
            "Choose export folder",
            str(self._project.folder),
        )
        if not target_dir_str:
            # User cancelled
            return

        target_dir = Path(target_dir_str)

        # Files we try to export. They are produced by the alignment pipeline.
        files_to_copy = [
            "subtitles.srt",   # phrases to SRT (YouTube / DaVinci Resolve)
            "phrases.json",    # phrase-level timings
            "words.json",      # word-level timings
            "phonemes.json",   # phoneme-level timings (if created)
            "lyrics.txt",      # plain lyrics text
        ]

        copied = 0
        missing = []

        for name in files_to_copy:
            src = align_dir / name
            if src.is_file():
                try:
                    shutil.copy2(src, target_dir / name)
                    copied += 1
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Export error",
                        f"Could not copy '{name}':\n{e}",
                    )
            else:
                missing.append(name)

        msg = f"Exported {copied} file(s) to:\n{target_dir}"
        if missing:
            msg += "\nMissing (not created yet): " + ", ".join(missing)

        self.lbl_export_status.setText(msg)

    # ==================== IMPORT FUNCTIONS ====================

    def import_phrases_json(self) -> None:
        """Import phrase timings from a JSON file and replace current phrases."""
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        path = self._open_file_dialog(
            "Import phrase timings (JSON)",
            "JSON files (*.json);;All files (*.*)",
        )
        if path is None:
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            QMessageBox.critical(self, "Import error", f"Could not read JSON file:\n{e}")
            return

        if not isinstance(data, list):
            QMessageBox.critical(
                self,
                "Import error",
                "Expected a JSON list of phrase objects.",
            )
            return

        self._phrases = data
        self.populate_segments_preview()
        self.save_timings_to_files()
        self.lbl_export_status.setText(f"Imported phrase timings from: {path}")

    def import_words_json(self) -> None:
        """Import word timings from a JSON file and replace current words."""
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        path = self._open_file_dialog(
            "Import word timings (JSON)",
            "JSON files (*.json);;All files (*.*)",
        )
        if path is None:
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            QMessageBox.critical(self, "Import error", f"Could not read JSON file:\n{e}")
            return

        if not isinstance(data, list):
            QMessageBox.critical(
                self,
                "Import error",
                "Expected a JSON list of word objects.",
            )
            return

        self._words = data
        self._refresh_karaoke_words(None)
        self.save_timings_to_files()
        self.lbl_export_status.setText(f"Imported word timings from: {path}")

    def import_phonemes_json(self) -> None:
        """Import phoneme timings JSON and overwrite phonemes.json in vocal_align."""
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        path = self._open_file_dialog(
            "Import phoneme timings (JSON)",
            "JSON files (*.json);;All files (*.*)",
        )
        if path is None:
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            QMessageBox.critical(self, "Import error", f"Could not read JSON file:\n{e}")
            return

        if not isinstance(data, list):
            QMessageBox.critical(
                self,
                "Import error",
                "Expected a JSON list of phoneme objects.",
            )
            return

        align_dir = self._get_alignment_dir()
        target = align_dir / "phonemes.json"
        try:
            target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Import error", f"Could not write phonemes.json:\n{e}")
            return

        self.lbl_export_status.setText(f"Imported phoneme timings from: {path}")

    def import_lyrics_txt(self) -> None:
        """Import lyrics from a TXT file and replace the current lyrics."""
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        path = self._open_file_dialog(
            "Import lyrics (TXT)",
            "Text files (*.txt);;All files (*.*)",
        )
        if path is None:
            return

        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Import error", f"Could not read TXT file:\n{e}")
            return

        # Update project object and text editor
        self.lyrics_edit.setPlainText(text)
        if hasattr(self._project, "lyrics_text"):
            self._project.lyrics_text = text

        # Also write into project folder and vocal_align folder
        try:
            (self._project.folder / "lyrics.txt").write_text(text, encoding="utf-8")
            align_dir = self._get_alignment_dir()
            (align_dir / "lyrics.txt").write_text(text, encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Import error", f"Could not write lyrics.txt:\n{e}")
            return

        self.lbl_export_status.setText(f"Imported lyrics TXT from: {path}")


    def import_srt(self) -> None:
        """Import an SRT file and replace phrase timings accordingly."""
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        src_path = self._open_file_dialog(
            "Import SRT subtitles",
            "SubRip subtitles (*.srt);;All files (*.*)",
        )
        if src_path is None:
            return

        try:
            content = src_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            QMessageBox.critical(self, "Import error", f"Could not read SRT file:\n{e}")
            return

        # Split into SRT blocks without using regex
        blocks: List[str] = []
        current: List[str] = []
        for line in content.splitlines():
            if line.strip() == "":
                if current:
                    blocks.append("\n".join(current))
                    current = []
            else:
                current.append(line)
        if current:
            blocks.append("\n".join(current))

        new_phrases: List[dict] = []
        for block in blocks:
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            if len(lines) < 2:
                continue

            # Optional numeric index on first line
            idx_line = 0
            if lines[0].isdigit():
                idx_line = 1
            if idx_line >= len(lines):
                continue

            timing_line = lines[idx_line]
            if "-->" not in timing_line:
                continue
            start_raw, end_raw = timing_line.split("-->", 1)
            start = self._parse_srt_timestamp(start_raw.strip())
            end = self._parse_srt_timestamp(end_raw.strip())
            if end <= start:
                continue

            text = " ".join(lines[idx_line + 1 :]) if len(lines) > idx_line + 1 else ""
            new_phrases.append(
                {
                    "line_index": len(new_phrases),
                    "text": text,
                    "start": float(start),
                    "end": float(end),
                }
            )

        if not new_phrases:
            QMessageBox.warning(
                self,
                "Import error",
                "No valid entries were found in the SRT file.",
            )
            return

        # Replace in-memory phrases
        self._phrases = new_phrases

        # Refresh UI (lists + karaoke preview)
        self.populate_segments_preview()

        # Persist everything (phrases.json, words.json, phonemes.json, subtitles.srt)
        self.save_timings_to_files()
        self.lbl_export_status.setText(f"Imported SRT from: {src_path}")


    # ------------------------------------------------------------------
    # Karaoke playback controls
    # ------------------------------------------------------------------

    def play_karaoke(self):
        """Play or resume the selected audio (full mix or manual file)."""

        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        audio_path = self._get_audio_path_for_current_mode()
        if not audio_path or not Path(audio_path).is_file():
            mode = "full_mix"
            if hasattr(self, "audio_source_combo") and self.audio_source_combo is not None:
                data = self.audio_source_combo.currentData()
                if isinstance(data, str) and data:
                    mode = data

            if mode == "manual":
                QMessageBox.warning(
                    self,
                    "Invalid audio file",
                    "No valid manual audio file selected.\n"
                    "Please click 'Browse…' or switch the audio source to 'Full mix (project audio)'.",
                )
            else:
                QMessageBox.warning(
                    self,
                    "No audio file",
                    "This project has no valid audio file associated.",
                )
            return

        url = QUrl.fromLocalFile(str(audio_path.resolve()))

        # If the current source differs from the desired audio, load it and start from 0
        if self.player.source() != url:
            self.player.setSource(url)
            self.player.setPosition(0)
        else:
            # If we are in StoppedState, ensure we start from 0
            if self.player.playbackState() == QMediaPlayer.PlaybackState.StoppedState:
                self.player.setPosition(0)
            # If we are in PausedState, keep current position and just resume

        # Start or resume playback
        self.player.play()

        # Notify global player bar if connected
        try:
            mode = self.audio_source_combo.currentData() if self.audio_source_combo is not None else "full_mix"
            label_mode = "manual file" if mode == "manual" else "full mix"
            self.audioStarted.emit(f"Karaoke: {self._project.name} — {label_mode}")
        except Exception:
            pass


    def stop_karaoke(self):
        """Stop playback for karaoke."""
        self.player.stop()

    def pause_karaoke(self):
        """Pause playback for karaoke."""
        self.player.pause()

    def _on_kara_slider_moved(self, position_ms: int) -> None:
        """Seek in the current audio when moving the karaoke slider and update karaoke preview."""
        # Seek the shared player
        self.player.setPosition(position_ms)

        # Update local UI immediately (even before positionChanged is emitted)
        if hasattr(self, "lbl_kara_track_time"):
            seconds = position_ms / 1000.0
            self.lbl_kara_track_time.setText(f"{seconds:7.3f} s")

        if self._phrases:
            pos_s = position_ms / 1000.0
            self._update_karaoke_for_time(pos_s)



    def _on_player_position_changed_karaoke(self, position_ms: int) -> None:
        """Update slider and time label for karaoke playback."""
        if not hasattr(self, "kara_position_slider"):
            return
        self.kara_position_slider.blockSignals(True)
        self.kara_position_slider.setValue(position_ms)
        self.kara_position_slider.blockSignals(False)
        seconds = position_ms / 1000.0
        if hasattr(self, "lbl_kara_time"):
            self.lbl_kara_time.setText(f"{seconds:7.3f} s")

    def _on_player_duration_changed_karaoke(self, duration_ms: int) -> None:
        """Update slider range when media duration changes."""
        if not hasattr(self, "kara_position_slider"):
            return
        self.kara_position_slider.setRange(0, duration_ms)

    def _on_karaoke_slider_moved(self, position_ms: int) -> None:
        """Seek in the current audio when moving the karaoke slider."""
        self.player.setPosition(position_ms)

    def apply_time_offsets(self) -> None:
        """Apply global offsets to phrase and word timings."""
        phrase_offset = float(self.spin_phrase_offset.value()) if hasattr(self, "spin_phrase_offset") else 0.0
        word_offset = float(self.spin_word_offset.value()) if hasattr(self, "spin_word_offset") else 0.0

        if not self._phrases and not self._words:
            QMessageBox.warning(self, "No alignment", "No alignment has been run yet.")
            return

        if phrase_offset == 0.0 and word_offset == 0.0:
            QMessageBox.information(self, "No change", "Offsets are both zero; nothing to apply.")
            return

        # Apply phrase offset to all phrases
        if phrase_offset != 0.0:
            for phrase in self._phrases:
                start = self._get_phrase_field(phrase, "start")
                end = self._get_phrase_field(phrase, "end")
                if start is not None:
                    self._set_phrase_field(phrase, "start", float(start) + phrase_offset)
                if end is not None:
                    self._set_phrase_field(phrase, "end", float(end) + phrase_offset)

        # Apply word offset to all words
        if word_offset != 0.0:
            for w in self._words:
                ws = self._get_word_field(w, "start")
                we = self._get_word_field(w, "end")
                if ws is not None:
                    if isinstance(w, dict):
                        w["start"] = float(ws) + word_offset
                    else:
                        setattr(w, "start", float(ws) + word_offset)
                if we is not None:
                    if isinstance(w, dict):
                        w["end"] = float(we) + word_offset
                    else:
                        setattr(w, "end", float(we) + word_offset)

        # Refresh views and status
        self.populate_segments_preview()
        self.lbl_align_status.setText("Applied phrase/word offsets.")




    # ------------------------------------------------------------------
    # Karaoke preview helper
    # ------------------------------------------------------------------
    
    
    def _update_karaoke_scroll_highlight(self, current_idx: Optional[int]) -> None:
        """
        Update colors and bold style in the karaoke scrolling list:
        previous / current / next.

        Default text color is taken from the current palette so the list
        stays readable across light / dark / neon themes.
        """
        if not hasattr(self, "kara_scroll_list"):
            return

        # Use the QListWidget's palette as the base text color
        palette = self.kara_scroll_list.palette()
        default_color = palette.color(QPalette.ColorRole.Text)

        for row in range(self.kara_scroll_list.count()):
            item = self.kara_scroll_list.item(row)
            idx = item.data(Qt.ItemDataRole.UserRole)

            # Default style: normal weight + theme text color
            color = default_color
            font = item.font()
            font.setBold(False)

            if isinstance(idx, int) and current_idx is not None:
                if idx == current_idx - 1:
                    # Previous phrase: red
                    color = QColor(200, 0, 0)
                elif idx == current_idx:
                    # Current phrase: green + bold
                    color = QColor(0, 150, 0)
                    font.setBold(True)
                elif idx == current_idx + 1:
                    # Next phrase: orange
                    color = QColor(220, 120, 0)

            item.setForeground(color)
            item.setFont(font)

            # Keep current phrase visible in the viewport
            if isinstance(idx, int) and current_idx is not None and idx == current_idx:
                self.kara_scroll_list.scrollToItem(item)

    def _update_karaoke_word_highlight(
        self,
        line_index: Optional[int],
        global_word_idx: Optional[int],
    ) -> None:
        """
        Highlight the current word in the right words list:
        - bold + blue for the current word
        - normal + theme text color for others.
        """
        if not hasattr(self, "kara_words_list"):
            return

        # Use the words list palette so colors remain consistent with the theme
        palette = self.kara_words_list.palette()
        default_color = palette.color(QPalette.ColorRole.Text)

        for row in range(self.kara_words_list.count()):
            item = self.kara_words_list.item(row)
            gidx = item.data(Qt.ItemDataRole.UserRole)

            font = item.font()
            if global_word_idx is not None and isinstance(gidx, int) and gidx == global_word_idx:
                font.setBold(True)
                item.setFont(font)
                # Current word: blue highlight
                item.setForeground(QColor(0, 0, 220))
                # Keep highlighted word visible
                self.kara_words_list.scrollToItem(item)
            else:
                font.setBold(False)
                item.setFont(font)
                # Other words: normal theme text color
                item.setForeground(default_color)

    def _update_karaoke_for_time(self, pos_s: float):
        """
        Set current line + word labels and highlights based on current playback time.
        Also keeps the words column synchronized with the currently sung phrase.
        """
        current_phrase = None
        current_idx: Optional[int] = None
        phrase_start = None
        phrase_end = None

        # Find current phrase for the given time
        for idx, phrase in enumerate(self._phrases):
            start = self._get_phrase_field(phrase, "start")
            end = self._get_phrase_field(phrase, "end")
            if start is None or end is None:
                continue
            if start <= pos_s <= end:
                current_phrase = phrase
                current_idx = idx
                phrase_start = start
                phrase_end = end
                break

        if current_phrase is None:
            # Outside any phrase: hide karaoke line/word, clear highlight and words list
            self.lbl_kara_line.setText("")
            self.lbl_kara_word.setText("")
            self._update_karaoke_scroll_highlight(None)
            self._refresh_karaoke_words(None)
            # Clear word highlight state
            if hasattr(self, "_kara_current_line_index"):
                self._kara_current_line_index = None
            self._update_karaoke_word_highlight(None, None)
            return

        # Update scrolling lyrics highlight (previous / current / next)
        self._update_karaoke_scroll_highlight(current_idx)

        # Phrase text + timer
        line_text = self._get_phrase_field(current_phrase, "text") or ""
        if phrase_start is not None and phrase_end is not None:
            self.lbl_kara_line.setText(
                f"[{phrase_start:7.3f} → {phrase_end:7.3f}]  {line_text}"
            )
        else:
            self.lbl_kara_line.setText(line_text)

        # Determine which line_index is used for words
        line_index = self._get_phrase_field(current_phrase, "line_index")
        if line_index is None:
            line_index = current_idx

        # Make sure the words list shows the words for the current phrase
        if not hasattr(self, "_kara_current_line_index"):
            self._kara_current_line_index = None
        if self._kara_current_line_index != line_index:
            self._refresh_karaoke_words(line_index)
            self._kara_current_line_index = line_index

        # Find current word within that line
        current_word_text = ""
        current_word_start: Optional[float] = None
        current_word_end: Optional[float] = None
        current_word_global_idx: Optional[int] = None

        for gidx, w in enumerate(self._words or []):
            if self._get_word_field(w, "line_index") != line_index:
                continue
            ws = self._get_word_field(w, "start")
            we = self._get_word_field(w, "end")
            if ws is None or we is None:
                continue
            if ws <= pos_s <= we:
                current_word_text = self._get_word_field(w, "text") or ""
                current_word_start = ws
                current_word_end = we
                current_word_global_idx = gidx
                break

        # Update word label with timer
        if current_word_text:
            if current_word_start is not None and current_word_end is not None:
                self.lbl_kara_word.setText(
                    f"[{current_word_start:7.3f} → {current_word_end:7.3f}]  {current_word_text}"
                )
            else:
                self.lbl_kara_word.setText(current_word_text)
        else:
            self.lbl_kara_word.setText("")

        # Highlight current word in the right column
        self._update_karaoke_word_highlight(line_index, current_word_global_idx)