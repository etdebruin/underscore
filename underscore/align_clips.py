"""Align [CLIP ...] markers in a weave script to a recording of the narration.

The host reads the weave script straight through, skipping the clip markers.
This tool matches each marker's preceding narration (the "anchor tail") against
the recording's transcript and emits a cue sheet placing every clip in the
pause right after its anchor sentence was spoken:

    uv run python -m underscore.align_clips weave.md --audio voice.wav -o cues.json

Marker format, on its own line between narration paragraphs:

    [CLIP <clip_id> | <label: who/where/when> | "first words of the quote"]

Only the clip_id is required; the label becomes the cue's reason.
"""

import argparse
import difflib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .cues import Clip
from .transcribe import Transcript, Word

MARKER_RE = re.compile(r"^\s*\[CLIP\s+([^\s|\]]+)\s*(?:\|([^|\]]*))?(?:\|([^\]]*))?\]\s*$",
                       re.MULTILINE)
TAIL_WORDS = 14      # narration words before the marker used as the match anchor
MIN_RATIO = 0.65     # fuzzy-match floor: below this the anchor was not found


@dataclass
class Anchor:
    clip_id: str
    label: str
    tail: list[str]  # the last narration words before the marker, verbatim


def parse_weave(text: str) -> list[Anchor]:
    anchors = []
    cursor = 0
    for m in MARKER_RE.finditer(text):
        narration = text[cursor:m.start()]
        tail = narration.split()[-TAIL_WORDS:]
        anchors.append(Anchor(
            clip_id=m.group(1),
            label=(m.group(2) or "").strip(),
            tail=tail,
        ))
        cursor = m.end()
    return anchors


def _norm(words: list[str]) -> str:
    out = [re.sub(r"[^a-z0-9']+", "", w.lower()) for w in words]
    return " ".join(w for w in out if w)


def _find_tail(words: list[Word], tail: list[str], start_idx: int) -> tuple[float, int]:
    """Best fuzzy match of `tail` in words[start_idx:]; returns (end_time, next_idx)."""
    target = _norm(tail)
    n = len(tail)
    best_ratio, best_i = 0.0, -1
    for i in range(start_idx, len(words) - n + 1):
        window = _norm([w.text for w in words[i : i + n]])
        ratio = difflib.SequenceMatcher(None, target, window).ratio()
        if ratio > best_ratio:
            best_ratio, best_i = ratio, i
    if best_i == -1 or best_ratio < MIN_RATIO:
        raise ValueError(
            f"anchor not found in the recording (best match {best_ratio:.2f}): "
            f"\"...{' '.join(tail)}\""
        )
    return words[best_i + n - 1].end, best_i + n


def align(weave_text: str, transcript: Transcript) -> list[Clip]:
    """Place each weave clip right after its anchor tail in the recording."""
    clips = []
    idx = 0
    for a in parse_weave(weave_text):
        if not a.tail:
            raise ValueError(f"clip {a.clip_id} has no narration before it to anchor on")
        end_time, idx = _find_tail(transcript.words, a.tail, idx)
        clips.append(Clip(time=round(end_time, 2), clip_id=a.clip_id, reason=a.label))
    return clips


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Place weave clip markers on a recording")
    parser.add_argument("weave", help="Weave script with [CLIP ...] markers")
    parser.add_argument("--audio", help="The narration recording (transcribed here)")
    parser.add_argument("--transcript", help="Existing transcript JSON instead of --audio")
    parser.add_argument("-o", "--output", required=True, help="Cue sheet JSON to write")
    args = parser.parse_args(argv)

    if args.transcript:
        data = json.loads(Path(args.transcript).read_text())
        transcript = Transcript(text=data["text"],
                                words=[Word(**w) for w in data["words"]],
                                duration=data["duration"])
    elif args.audio:
        from .transcribe import transcribe

        print(f"transcribing {args.audio} ...", flush=True)
        transcript = transcribe(args.audio)
    else:
        print("error: pass --audio or --transcript", file=sys.stderr)
        return 1

    clips = align(Path(args.weave).read_text(), transcript)
    for c in clips:
        print(f"  clip {c.clip_id:<28} @{c.time:7.2f}s  {c.reason}")
    sheet = {"inserts": [], "underlays": [],
             "clips": [{"time": c.time, "clip_id": c.clip_id,
                        "reason": c.reason, "gain": c.gain} for c in clips]}
    Path(args.output).write_text(json.dumps(sheet, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
