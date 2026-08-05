"""CLI: voice track in, scored track out.

    uv run python -m underscore.cli input.wav -o scored.wav [--cues cues.json]
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf

from .analyze import analyze
from .cues import CueSheet, Insert, Underlay
from .library import Library, catalog_text
from .mix import assemble
from .transcribe import transcribe
from .voicefx import process_voice

SR = 44100


def _loudnorm(in_path: str, out_path: str) -> None:
    """Normalize to -16 LUFS (podcast standard); encodes by output extension."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", in_path,
         "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", out_path],
        check=True,
    )


def _load_dotenv() -> None:
    env = Path(".env")
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Score a podcast voice track with music")
    parser.add_argument("input", help="Voice track (any format ffmpeg reads)")
    parser.add_argument("-o", "--output", required=True, help="Output file (.wav/.mp3/.m4a)")
    parser.add_argument("--cues", help="Use/save cue sheet JSON instead of calling Claude")
    parser.add_argument("--save-cues", help="Write the generated cue sheet to this JSON file")
    parser.add_argument(
        "--music",
        choices=["elevenlabs", "synth"],
        default=None,
        help="Music backend (default: elevenlabs when ELEVENLABS_API_KEY is set, else synth)",
    )
    parser.add_argument(
        "--scoring",
        choices=["light", "standard", "rich"],
        default="standard",
        help="How densely to score: stings at theme transitions scale with this",
    )
    parser.add_argument(
        "--level", action=argparse.BooleanOptionalAction, default=True,
        help="Even out the speaker's volume (compressor + dynamic normalizer)",
    )
    parser.add_argument(
        "--warm", action=argparse.BooleanOptionalAction, default=True,
        help="Warm up the voice (EQ: rumble cut, low-mid boost, de-ess)",
    )
    args = parser.parse_args(argv)

    print(f"[1/4] Transcribing {args.input} ...", flush=True)
    transcript = transcribe(args.input)
    print(f"      {len(transcript.words)} words, {transcript.duration:.1f}s", flush=True)

    if args.cues and Path(args.cues).exists():
        print(f"[2/4] Loading cue sheet from {args.cues}", flush=True)
        raw = json.loads(Path(args.cues).read_text())
        sheet = CueSheet(
            inserts=[Insert(**i) for i in raw["inserts"]],
            underlays=[Underlay(**u) for u in raw["underlays"]],
        )
    else:
        print(f"[2/4] Asking Claude for a cue sheet ({args.scoring} scoring) ...", flush=True)
        library = Library()
        sheet = analyze(transcript, scoring=args.scoring,
                        catalog=catalog_text(library.catalog()))

    sheet_json = json.dumps(dataclasses.asdict(sheet), indent=2)
    for ins in sheet.inserts:
        print(f"      insert   @{ins.time:6.1f}s  {ins.duration:.0f}s {ins.mood:<10} {ins.reason}")
    for u in sheet.underlays:
        print(f"      underlay {u.start:6.1f}-{u.end:.1f}s  {u.mood:<10} {u.reason}")
    save_to = args.save_cues or args.cues
    if save_to:
        Path(save_to).write_text(sheet_json)
        print(f"      cue sheet saved to {save_to}", flush=True)

    backend = args.music or ("elevenlabs" if os.environ.get("ELEVENLABS_API_KEY") else "synth")
    bed_fn = None
    if backend == "elevenlabs":
        from .elevenlabs_music import ElevenLabsMusic

        bed_fn = ElevenLabsMusic().bed
    fx = [name for name, on in (("warm", args.warm), ("level", args.level)) if on]
    print(f"[3/4] Mixing (music: {backend}; voice fx: {', '.join(fx) or 'none'}) ...", flush=True)
    voice = process_voice(args.input, SR, level=args.level, warm=args.warm)
    master = assemble(voice, SR, sheet, transcript.speech_spans(), bed_fn=bed_fn)

    print(f"[4/4] Loudness-normalizing -> {args.output}", flush=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, master, SR)
        _loudnorm(tmp.name, args.output)
    Path(tmp.name).unlink(missing_ok=True)

    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
