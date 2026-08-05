"""Voice treatment: broadcast-style leveling and warmth, via ffmpeg filters.

Two independent stages, both applied to the dry voice before music is mixed in:

- warm: high-pass rumble, low-mid warmth boost, tame harshness, a little air,
  and a de-esser. Runs first so the leveler reacts to the corrected tone.
- level: gentle compression followed by a windowed dynamic normalizer, evening
  out a speaker who drifts between quiet and loud.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

WARM_CHAIN = (
    "highpass=f=75,"
    "equalizer=f=170:width_type=q:width=1.0:gain=2.5,"
    "equalizer=f=3200:width_type=q:width=1.2:gain=-1.5,"
    "equalizer=f=9500:width_type=q:width=0.8:gain=1.2,"
    "deesser"
)

LEVEL_CHAIN = (
    "acompressor=threshold=-22dB:ratio=2.5:attack=10:release=200:makeup=3,"
    "dynaudnorm=f=300:g=15:m=6:p=0.9"
)


def build_filter(level: bool, warm: bool) -> str:
    chains = []
    if warm:
        chains.append(WARM_CHAIN)
    if level:
        chains.append(LEVEL_CHAIN)
    return ",".join(chains)


def process_voice(path: str, sr: int, level: bool = True, warm: bool = True) -> np.ndarray:
    """Decode any audio file to mono float32 at `sr`, with optional treatment."""
    filters = build_filter(level=level, warm=warm)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(sr)]
    if filters:
        cmd += ["-af", filters]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        subprocess.run(cmd + [tmp.name], check=True)
        data, _ = sf.read(tmp.name, dtype="float32")
    Path(tmp.name).unlink(missing_ok=True)
    return data
