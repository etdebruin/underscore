"""Assemble the scored track: insert stings into pauses, duck beds under speech."""

import numpy as np

from .cues import CueSheet, Insert
from .music import generate_bed

# Insert stings bleed slightly under the surrounding speech so the edit sounds produced.
PRE_OVERLAP = 0.8
POST_OVERLAP = 1.4
SNAP_MAX_DIST = 6.0

UNDERLAY_DUCK = 0.28   # bed gain while speech is present
UNDERLAY_FULL = 0.85   # bed gain in pauses within an underlay region
UNDERLAY_LEVEL = 0.55  # overall underlay level relative to the (already quiet) bed
EDGE_FADE = 1.5        # underlay fade in/out, seconds
SMOOTH_SECONDS = 0.35  # gain-envelope smoothing (attack/release feel)


def find_gaps(speech_spans: list[tuple[float, float]], total: float) -> list[tuple[float, float]]:
    """Complement of the speech spans within [0, total]."""
    gaps = []
    cursor = 0.0
    for start, end in sorted(speech_spans):
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < total:
        gaps.append((cursor, total))
    return gaps


def snap_to_gap(t: float, gaps: list[tuple[float, float]], max_dist: float = SNAP_MAX_DIST) -> float:
    """Move an insert point to the center of the nearest silence, if one is close."""
    best, best_dist = t, max_dist
    for start, end in gaps:
        center = (start + end) / 2.0
        dist = abs(center - t)
        if dist < best_dist:
            best, best_dist = center, dist
    return best


def shift_time(t: float, inserts: list[Insert]) -> float:
    """Map a time on the original track to the track with pauses inserted."""
    return t + sum(i.duration for i in inserts if i.time < t)


def duck_envelope(
    n: int,
    sr: int,
    speech_spans: list[tuple[float, float]],
    duck: float,
    full: float,
) -> np.ndarray:
    env = np.full(n, full, dtype=np.float32)
    for start, end in speech_spans:
        a, b = int(start * sr), int(end * sr)
        env[max(a, 0) : min(b, n)] = duck
    return _smooth(env, sr)


def _smooth(env: np.ndarray, sr: int) -> np.ndarray:
    win = int(SMOOTH_SECONDS * sr)
    if win < 2:
        return env
    kernel = np.hanning(win)
    kernel /= kernel.sum()
    return np.convolve(env, kernel, mode="same").astype(np.float32)


def _place(canvas: np.ndarray, clip: np.ndarray, at: int) -> None:
    """Add a stereo clip onto the canvas at sample offset `at`, clipping to bounds."""
    start = max(at, 0)
    end = min(at + clip.shape[0], canvas.shape[0])
    if end > start:
        canvas[start:end] += clip[start - at : end - at]


def _synth_bed(mood: str, duration: float, sr: int, seed: int = 0, reason: str = "") -> np.ndarray:
    return generate_bed(mood, duration, sr, seed=seed)


def assemble(
    voice: np.ndarray,
    sr: int,
    sheet: CueSheet,
    speech_spans: list[tuple[float, float]],
    seed: int = 1,
    bed_fn=None,
) -> np.ndarray:
    """Mix voice + cue sheet into a stereo master. `voice` is mono float32.

    `bed_fn(mood, duration, sr, seed, reason)` supplies music clips; defaults to
    the built-in procedural synth pads.
    """
    bed_fn = bed_fn or _synth_bed
    total = len(voice) / sr
    gaps = find_gaps(speech_spans, total)

    # Snap insert points to real silences, keep them ordered.
    inserts = [
        Insert(snap_to_gap(i.time, gaps), i.duration, i.mood, i.reason) for i in sheet.inserts
    ]
    inserts.sort(key=lambda i: i.time)

    # Build the stretched voice track by splicing silence in at each insert point.
    pieces, cursor = [], 0
    for ins in inserts:
        cut = int(ins.time * sr)
        pieces.append(voice[cursor:cut])
        pieces.append(np.zeros(int(ins.duration * sr), dtype=np.float32))
        cursor = cut
    pieces.append(voice[cursor:])
    stretched = np.concatenate(pieces)

    out = np.zeros((len(stretched), 2), dtype=np.float32)
    out += stretched[:, None]  # center the voice

    # Insert stings: full level in the pause, ramping in/out under adjacent speech.
    for k, ins in enumerate(inserts):
        at = shift_time(ins.time, inserts)
        clip_dur = PRE_OVERLAP + ins.duration + POST_OVERLAP
        bed = bed_fn(ins.mood, clip_dur, sr, seed=seed + k, reason=ins.reason)
        env = np.ones(bed.shape[0], dtype=np.float32)
        pre, post = int(PRE_OVERLAP * sr), int(POST_OVERLAP * sr)
        env[:pre] = np.linspace(0.0, 1.0, pre)
        env[-post:] = np.linspace(1.0, 0.0, post)
        _place(out, bed * env[:, None], int((at - PRE_OVERLAP) * sr))

    # Underlays: bed under the narration, ducked while speech is present.
    shifted_speech = [
        (shift_time(s, inserts), shift_time(e, inserts)) for s, e in speech_spans
    ]
    for k, u in enumerate(sheet.underlays):
        start, end = shift_time(u.start, inserts), shift_time(u.end, inserts)
        n = int((end - start) * sr)
        if n <= 0:
            continue
        bed = bed_fn(u.mood, end - start, sr, seed=seed + 100 + k, reason=u.reason)
        n = min(n, bed.shape[0])
        bed = bed[:n]
        local_speech = [(s - start, e - start) for s, e in shifted_speech if e > start and s < end]
        env = duck_envelope(n, sr, local_speech, duck=UNDERLAY_DUCK, full=UNDERLAY_FULL)
        fade = min(int(EDGE_FADE * sr), n // 2)
        env[:fade] *= np.linspace(0.0, 1.0, fade)
        env[-fade:] *= np.linspace(1.0, 0.0, fade)
        _place(out, bed * (env * UNDERLAY_LEVEL)[:, None], int(start * sr))

    # Soft-knee safety limiter.
    peak = np.max(np.abs(out))
    if peak > 0.98:
        out = np.tanh(out / peak * 1.2) * (0.98 / np.tanh(1.2))
    return out.astype(np.float32)
