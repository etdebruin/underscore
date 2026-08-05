"""Procedurally generated mood beds.

Self-contained stand-in for a licensed music library or a generation API
(MusicGen, ElevenLabs music): slow synth pads over a chord progression,
one recipe per mood. Swap this module out without touching the mixer.
"""

import numpy as np

# Chord progressions as semitone offsets from the root, one chord list per step.
_RECIPES: dict[str, dict] = {
    "warm": {
        "root": 220.0,  # A3
        "chords": [[0, 4, 7, 12], [5, 9, 12, 16], [7, 11, 14, 17], [0, 4, 7, 12]],
        "wave": "triangle",
        "lp_hz": 1800.0,
        "pulse": 0.0,
    },
    "tense": {
        "root": 138.6,  # C#3
        "chords": [[0, 3, 7, 13], [0, 3, 6, 13], [1, 3, 7, 13], [0, 3, 6, 12]],
        "wave": "saw",
        "lp_hz": 1200.0,
        "pulse": 2.0,  # slow amplitude pulse, Hz
    },
    "wistful": {
        "root": 196.0,  # G3
        "chords": [[0, 3, 7, 14], [-4, 0, 3, 12], [-2, 2, 5, 12], [0, 3, 7, 10]],
        "wave": "triangle",
        "lp_hz": 1400.0,
        "pulse": 0.0,
    },
    "uplifting": {
        "root": 261.6,  # C4
        "chords": [[0, 4, 7, 12], [7, 12, 16, 19], [5, 9, 12, 17], [7, 11, 14, 19]],
        "wave": "triangle",
        "lp_hz": 2400.0,
        "pulse": 4.0,
    },
    "mysterious": {
        "root": 164.8,  # E3
        "chords": [[0, 5, 7, 14], [0, 6, 7, 13], [0, 5, 10, 14], [0, 6, 7, 12]],
        "wave": "sine",
        "lp_hz": 1000.0,
        "pulse": 0.0,
    },
}

_CHORD_SECONDS = 4.0
_TARGET_PEAK = 0.7


def _osc(freq: float, t: np.ndarray, wave: str, phase: float) -> np.ndarray:
    x = freq * t + phase
    if wave == "saw":
        return 2.0 * (x % 1.0) - 1.0
    if wave == "triangle":
        return 2.0 * np.abs(2.0 * (x % 1.0) - 1.0) - 1.0
    return np.sin(2 * np.pi * x)


def _lowpass(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    # One-pole lowpass approximated by a truncated exponential FIR kernel
    # (vectorized; the recursive form was a per-sample Python loop). Applied
    # twice for a gentler top end.
    alpha = 1.0 - np.exp(-2 * np.pi * cutoff_hz / sr)
    taps = int(np.ceil(np.log(1e-4) / np.log(1.0 - alpha)))
    kernel = alpha * (1.0 - alpha) ** np.arange(max(taps, 2))
    kernel /= kernel.sum()
    y = x
    for _ in range(2):
        y = np.convolve(y, kernel)[: len(x)]
    return y


def generate_bed(mood: str, duration: float, sr: int, seed: int = 0) -> np.ndarray:
    """Render a stereo float32 pad of `duration` seconds for the given mood."""
    recipe = _RECIPES.get(mood, _RECIPES["warm"])
    rng = np.random.default_rng(seed)
    n = round(duration * sr)
    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)

    chord_len = int(_CHORD_SECONDS * sr)
    xfade = int(0.5 * sr)
    chords = recipe["chords"]
    pos = 0
    step = 0
    while pos < n:
        seg_len = min(chord_len + xfade, n - pos)
        t = np.arange(seg_len) / sr
        seg_l = np.zeros(seg_len)
        seg_r = np.zeros(seg_len)
        for semitone in chords[step % len(chords)]:
            freq = recipe["root"] * 2 ** (semitone / 12.0)
            detune = 1.0 + rng.uniform(-0.0015, 0.0015)
            phase_l, phase_r = rng.uniform(0, 1, size=2)
            seg_l += _osc(freq * detune, t, recipe["wave"], phase_l)
            seg_r += _osc(freq / detune, t, recipe["wave"], phase_r)
        # per-chord attack/release so chord changes don't click
        env = np.ones(seg_len)
        a = min(xfade, seg_len)
        env[:a] = np.linspace(0.0, 1.0, a)
        env[-a:] *= np.linspace(1.0, 0.0, a)
        left[pos : pos + seg_len] += seg_l * env
        right[pos : pos + seg_len] += seg_r * env
        pos += chord_len
        step += 1

    left = _lowpass(left, sr, recipe["lp_hz"])
    right = _lowpass(right, sr, recipe["lp_hz"])

    if recipe["pulse"] > 0:
        t = np.arange(n) / sr
        pulse = 0.75 + 0.25 * np.sin(2 * np.pi * recipe["pulse"] * t)
        left *= pulse
        right *= pulse

    bed = np.stack([left, right], axis=1)
    peak = np.max(np.abs(bed))
    if peak > 0:
        bed *= _TARGET_PEAK / peak
    return bed.astype(np.float32)
