"""Ask Claude, acting as a podcast producer, where music belongs."""

from pydantic import BaseModel, Field

from .cues import Clip, CueSheet, validate_cues
from .transcribe import Transcript

MODEL = "claude-opus-5"

SYSTEM = """You are an experienced podcast producer scoring an episode with music,
in the style of a modern narrative news show. You receive a transcript with
word-level timestamps. Propose a cue sheet:

- "inserts": moments where the narration should PAUSE for a short musical sting
  (3-8 seconds). These mark structure and heighten storytelling: use one whenever
  the episode moves between distinct themes, segments, or chapters, and at genuine
  act breaks, cliffhangers, reveals, or emotional beats where a breath of
  silence-plus-music lands the moment. The time you give is where the pause opens,
  and it must land at a sentence boundary.
- "underlays": stretches where a quiet music bed should play UNDER the narration
  (intro, outro, emotionally charged, scene-setting, or montage-like passages).
  At least 10 seconds long, and they must not overlap each other or an insert.

Moods available: warm, tense, wistful, uplifting, mysterious.

Never score wall-to-wall; silence between cues is what makes them land.
Give a short reason for each cue."""

DENSITY = {
    "light": (
        "Be very sparing: roughly one insert per 10 minutes of audio and at most "
        "2-3 underlays total, reserved for the strongest moments only."
    ),
    "standard": (
        "Aim for roughly one insert per 4-6 minutes of audio at real theme "
        "transitions, plus an intro bed, an outro bed, and 1-2 beds under the most "
        "emotionally charged passages."
    ),
    "rich": (
        "Score it like a heavily produced narrative episode: an insert at every "
        "theme or segment transition (roughly one per 2-4 minutes), and beds under "
        "the intro, outro, and each emotionally significant or scene-setting "
        "passage. Still leave stretches of dry narration between cues."
    ),
}


class InsertCue(BaseModel):
    time: float = Field(description="Seconds into the track where the pause opens")
    duration: float = Field(description="Sting length in seconds, 3-8")
    mood: str
    reason: str
    clip_id: str | None = Field(
        None, description="Reuse this existing library clip; null to generate new music"
    )


class UnderlayCue(BaseModel):
    start: float
    end: float
    mood: str
    reason: str
    clip_id: str | None = Field(
        None, description="Reuse this existing library clip; null to generate new music"
    )


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


CLIPS_GUIDANCE = """
This episode is a narrated documentary: at the marked CLIP moments the narration
stops and the subject's own recorded voice plays. Those moments are already a
break in the narration, so:
- Do NOT put an insert at a clip moment — the clip is the break. Keep inserts at
  least ~15 seconds away from one.
- A bed may run under the narration leading into a clip and stop there, or pick
  up after it. Beds continue under the clip if you span it, so only span one when
  music under the subject's voice genuinely serves the moment.
- The strongest cue is often a short sting shortly AFTER a clip, letting the
  quote land before the host resumes."""


def _clips_block(clips: list[Clip]) -> str:
    lines = [
        f"[{c.time:7.2f}s] CLIP {c.clip_id} — {c.reason or 'the subject speaks'}"
        for c in sorted(clips, key=lambda c: c.time)
    ]
    return "The subject's voice clips land here (times on this same track):\n" + "\n".join(lines)


def analyze(
    transcript: Transcript,
    client=None,
    scoring: str = "standard",
    catalog: str = "",
    clips: list[Clip] | None = None,
) -> CueSheet:
    import anthropic

    client = client or anthropic.Anthropic()
    system = f"{SYSTEM}\n\n{DENSITY.get(scoring, DENSITY['standard'])}"
    if clips:
        system += f"\n{CLIPS_GUIDANCE}"
    if catalog:
        system += (
            f"\n\n{catalog}\n\nReusing a clip keeps the show sonically consistent and is "
            "free; set clip_id when a clip's mood and original scene fit the new moment. "
            "Set clip_id to null when the moment deserves bespoke music."
        )
    content = f"Track duration: {transcript.duration:.1f}s.\n"
    if clips:
        content += f"\n{_clips_block(clips)}\n"
    content += f"\nTranscript with timestamps:\n\n{_timed_transcript(transcript)}"
    response = client.messages.parse(
        model=MODEL,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": content}],
        output_format=RawCueSheet,
    )
    raw = response.parsed_output
    return validate_cues(raw.model_dump(), total_duration=transcript.duration)
