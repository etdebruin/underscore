"""Import an external audio file into the clip library.

    uv run python -m underscore.import_clip quote.wav --clip-id ilya-neurips-1 \
        --description "Ilya on stage at NeurIPS, 2015"

The file is stored as-is (any container ffmpeg reads, including video — decode
pulls the audio stream) and registered as a speech clip, so it can be placed
verbatim via the `clips` section of a cue sheet but is never offered to the
music producer for reuse.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from .library import Library


def probe_length_ms(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return round(float(out) * 1000)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import an audio file into the clip library")
    parser.add_argument("input", help="Audio file (any format ffmpeg reads)")
    parser.add_argument("--clip-id", help="Library id (default: input filename stem)")
    parser.add_argument("--description", default="", help="Who/where/when — shown in the editor")
    parser.add_argument("--kind", default="speech", choices=["speech", "music"])
    parser.add_argument("--library", help="Library root (default: ~/.underscore/library)")
    args = parser.parse_args(argv)

    src = Path(args.input)
    if not src.exists():
        print(f"error: no such file: {src}", file=sys.stderr)
        return 1
    clip_id = args.clip_id or src.stem

    lib = Library(args.library)
    length_ms = probe_length_ms(src)
    shutil.copyfile(src, lib.path_for(clip_id))
    lib.register(clip_id, mood="", description=args.description,
                 length_ms=length_ms, kind=args.kind)
    print(f"imported {clip_id}  {length_ms / 1000:.1f}s  kind={args.kind}  -> {lib.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
