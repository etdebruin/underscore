"""Ask Claude, acting as a podcast producer, where music belongs."""

from pydantic import BaseModel, Field

from .cues import CueSheet, validate_cues
from .transcribe import Transcript

MODEL = "claude-opus-5"

SYSTEM = """You are an experienced podcast producer scoring an episode with music.
You receive a transcript with word-level timestamps. Propose a restrained cue sheet:

- "inserts": moments where the narration should PAUSE for a short musical sting
  (3-8 seconds). Use these only at genuine act breaks, cliffhangers, or emotional
  beats where a beat of silence-plus-music heightens the moment. The time you give
  is where the pause opens, and it must land at a sentence boundary.
- "underlays": stretches where a quiet music bed should play UNDER the narration
  (intro, outro, emotionally charged or scene-setting passages). At least 10 seconds
  long, and they must not overlap each other or an insert.

Moods available: warm, tense, wistful, uplifting, mysterious.

Less is more: for a few minutes of audio, 1-3 inserts and 1-3 underlays. Never
score wall-to-wall. Give a short reason for each cue."""


class InsertCue(BaseModel):
    time: float = Field(description="Seconds into the track where the pause opens")
    duration: float = Field(description="Sting length in seconds, 3-8")
    mood: str
    reason: str


class UnderlayCue(BaseModel):
    start: float
    end: float
    mood: str
    reason: str


class RawCueSheet(BaseModel):
    inserts: list[InsertCue]
    underlays: list[UnderlayCue]


def _timed_transcript(transcript: Transcript) -> str:
    """Render the transcript with a [mm:ss.s] marker at each speech-span start."""
    lines = []
    line: list[str] = []
    line_start = 0.0
    for w in transcript.words:
        if not line:
            line_start = w.start
        line.append(w.text)
        if w.text.endswith((".", "!", "?")):
            lines.append(f"[{line_start:7.2f}s] {' '.join(line)}")
            line = []
    if line:
        lines.append(f"[{line_start:7.2f}s] {' '.join(line)}")
    return "\n".join(lines)


def analyze(transcript: Transcript, client=None) -> CueSheet:
    import anthropic

    client = client or anthropic.Anthropic()
    response = client.messages.parse(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Track duration: {transcript.duration:.1f}s.\n"
                    f"Transcript with timestamps:\n\n{_timed_transcript(transcript)}"
                ),
            }
        ],
        output_format=RawCueSheet,
    )
    raw = response.parsed_output
    return validate_cues(raw.model_dump(), total_duration=transcript.duration)
