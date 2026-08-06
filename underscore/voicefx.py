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


def main(argv: list[str] | None = None) -> int:
    """Apply the voice treatment to a file, so a take can be auditioned dry vs treated.

    uv run python -m underscore.voicefx take.webm -o treated.wav [--no-warm] [--no-level]
    """
    import argparse

    parser = argparse.ArgumentParser(description="Treat a voice recording")
    parser.add_argument("input")
    parser.add_argument("-o", "--output", required=True, help="Output .wav")
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--level", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warm", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    voice = process_voice(args.input, args.sr, level=args.level, warm=args.warm)
    sf.write(args.output, voice, args.sr)
    print(f"{len(voice) / args.sr:.1f}s -> {args.output}")
    return 0


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


if __name__ == "__main__":
    import sys

    sys.exit(main())
