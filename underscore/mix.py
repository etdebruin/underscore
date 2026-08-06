"""Assemble the scored track: insert stings into pauses, duck beds under speech."""

from dataclasses import dataclass

import numpy as np

from .cues import Clip, CueSheet, Insert
from .music import generate_bed

# Insert stings bleed slightly under the surrounding speech so the edit sounds produced.
PRE_OVERLAP = 0.8
POST_OVERLAP = 1.4
SNAP_MAX_DIST = 6.0

# Voice clips get a breath of silence either side instead of a musical overlap,
# and only a click-guard fade at the edges — the quote itself plays verbatim.
CLIP_PAD_PRE = 0.35
CLIP_PAD_POST = 0.45
CLIP_EDGE_FADE = 0.02

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


@dataclass
class _Splice:
    """A stretch of silence spliced into the voice track (for a sting or a clip)."""

    time: float
    duration: float


def shift_time(t: float, splices: list) -> float:
    """Map a time on the original track to the track with pauses inserted.

    Accepts anything with `.time` and `.duration` (Insert, _Splice, ...).
    """
    return t + sum(s.duration for s in splices if s.time < t)


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


def _synth_bed(mood: str, duration: float, sr: int, seed: int = 0, reason: str = "",
               clip_id: str | None = None) -> np.ndarray:
    return generate_bed(mood, duration, sr, seed=seed)


def assemble(
    voice: np.ndarray,
    sr: int,
    sheet: CueSheet,
    speech_spans: list[tuple[float, float]],
    seed: int = 1,
    bed_fn=None,
    clip_fn=None,
) -> np.ndarray:
    """Mix voice + cue sheet into a stereo master. `voice` is mono float32.

    `bed_fn(mood, duration, sr, seed, reason)` supplies music clips; defaults to
    the built-in procedural synth pads. `clip_fn(clip_id, sr)` supplies voice
    clips verbatim (stereo, natural length) — required when the sheet has clips.
    """
    bed_fn = bed_fn or _synth_bed
    if sheet.clips and clip_fn is None:
        raise ValueError("cue sheet has voice clips but no clip_fn was provided")
    total = len(voice) / sr
    gaps = find_gaps(speech_spans, total)

    # Snap insert points to real silences, keep them ordered.
    inserts = [
        Insert(snap_to_gap(i.time, gaps), i.duration, i.mood, i.reason, i.clip_id)
        for i in sheet.inserts
    ]
    inserts.sort(key=lambda i: i.time)

    # Voice clips splice the same way; their duration is the audio's natural length.
    clips: list[tuple[Clip, np.ndarray]] = []
    for c in sheet.clips:
        audio = clip_fn(c.clip_id, sr)
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=1)
        clips.append((Clip(snap_to_gap(c.time, gaps), c.clip_id, c.reason, c.gain), audio))
    clips.sort(key=lambda pair: pair[0].time)

    splices = sorted(
        [_Splice(i.time, i.duration) for i in inserts]
        + [_Splice(c.time, CLIP_PAD_PRE + a.shape[0] / sr + CLIP_PAD_POST) for c, a in clips],
        key=lambda s: s.time,
    )

    # Build the stretched voice track by splicing silence in at each splice point.
    pieces, cursor = [], 0
    for sp in splices:
        cut = int(sp.time * sr)
        pieces.append(voice[cursor:cut])
        pieces.append(np.zeros(int(sp.duration * sr), dtype=np.float32))
        cursor = cut
    pieces.append(voice[cursor:])
    stretched = np.concatenate(pieces)

    out = np.zeros((len(stretched), 2), dtype=np.float32)
    out += stretched[:, None]  # center the voice

    # Insert stings: full level in the pause, ramping in/out under adjacent speech.
    for k, ins in enumerate(inserts):
        at = shift_time(ins.time, splices)
        clip_dur = PRE_OVERLAP + ins.duration + POST_OVERLAP
        bed = bed_fn(ins.mood, clip_dur, sr, seed=seed + k, reason=ins.reason,
                     clip_id=ins.clip_id)
        env = np.ones(bed.shape[0], dtype=np.float32)
        pre, post = int(PRE_OVERLAP * sr), int(POST_OVERLAP * sr)
        env[:pre] = np.linspace(0.0, 1.0, pre)
        env[-post:] = np.linspace(1.0, 0.0, post)
        _place(out, bed * env[:, None], int((at - PRE_OVERLAP) * sr))

    # Voice clips: verbatim in their spliced pause, click-guard fades only.
    clip_spans: list[tuple[float, float]] = []
    for c, audio in clips:
        at = shift_time(c.time, splices) + CLIP_PAD_PRE
        clip = audio * c.gain
        fade = int(CLIP_EDGE_FADE * sr)
        if 1 < fade and clip.shape[0] > 2 * fade:
            clip[:fade] *= np.linspace(0.0, 1.0, fade)[:, None]
            clip[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None]
        _place(out, clip, int(at * sr))
        clip_spans.append((at, at + audio.shape[0] / sr))

    # Underlays: bed under the narration, ducked while anyone is speaking —
    # the host (speech spans) or the subject (spliced-in clips).
    shifted_speech = [
        (shift_time(s, splices), shift_time(e, splices)) for s, e in speech_spans
    ] + clip_spans
    for k, u in enumerate(sheet.underlays):
        start, end = shift_time(u.start, splices), shift_time(u.end, splices)
        n = int((end - start) * sr)
        if n <= 0:
            continue
        bed = bed_fn(u.mood, end - start, sr, seed=seed + 100 + k, reason=u.reason,
                     clip_id=u.clip_id)
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
