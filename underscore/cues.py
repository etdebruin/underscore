"""Cue sheet model: where music goes, and validation of LLM-proposed cues."""

from dataclasses import dataclass, field

MOODS = {"warm", "tense", "wistful", "uplifting", "mysterious"}
DEFAULT_MOOD = "warm"

MIN_INSERT_DUR = 2.0
MAX_INSERT_DUR = 12.0
MIN_INSERT_SPACING = 10.0
MIN_UNDERLAY_DUR = 5.0


@dataclass
class Insert:
    """Music played during a pause deliberately opened in the narration."""

    time: float
    duration: float
    mood: str
    reason: str
    clip_id: str | None = None  # reuse this library clip instead of generating


@dataclass
class Underlay:
    """Music bed playing softly underneath a stretch of narration."""

    start: float
    end: float
    mood: str
    reason: str
    clip_id: str | None = None


@dataclass
class CueSheet:
    inserts: list[Insert] = field(default_factory=list)
    underlays: list[Underlay] = field(default_factory=list)


def _mood(value: str) -> str:
    return value if value in MOODS else DEFAULT_MOOD


def validate_cues(raw: dict, total_duration: float) -> CueSheet:
    """Sanitize an LLM-proposed cue sheet: clamp, sort, drop overlaps."""
    inserts: list[Insert] = []
    for item in sorted(raw.get("inserts", []), key=lambda d: d["time"]):
        time = min(max(float(item["time"]), 0.0), total_duration)
        duration = min(max(float(item["duration"]), MIN_INSERT_DUR), MAX_INSERT_DUR)
        if inserts and time - inserts[-1].time < MIN_INSERT_SPACING:
            continue
        inserts.append(Insert(time, duration, _mood(item.get("mood", "")),
                              item.get("reason", ""), item.get("clip_id")))

    underlays: list[Underlay] = []
    for item in sorted(raw.get("underlays", []), key=lambda d: d["start"]):
        start = min(max(float(item["start"]), 0.0), total_duration)
        end = min(max(float(item["end"]), 0.0), total_duration)
        if end - start < MIN_UNDERLAY_DUR:
            continue
        if underlays and start < underlays[-1].end:
            continue
        underlays.append(Underlay(start, end, _mood(item.get("mood", "")),
                                  item.get("reason", ""), item.get("clip_id")))

    return CueSheet(inserts=inserts, underlays=underlays)
