import dataclasses
import itertools

from underscore.cues import MOODS, sheet_from_dict, validate_cues

RAW = {
    "inserts": [
        {"time": 30.0, "duration": 6.0, "mood": "tense", "reason": "act break"},
        {"time": 32.0, "duration": 5.0, "mood": "warm", "reason": "too close to previous"},
        {"time": 90.0, "duration": 100.0, "mood": "wistful", "reason": "duration out of range"},
    ],
    "underlays": [
        {"start": 10.0, "end": 25.0, "mood": "mysterious", "reason": "opening"},
        {"start": 20.0, "end": 40.0, "mood": "warm", "reason": "overlaps previous"},
        {"start": 50.0, "end": 52.0, "mood": "warm", "reason": "too short"},
        {"start": 100.0, "end": 500.0, "mood": "alien-disco", "reason": "bad mood, clamp end"},
    ],
}


def test_inserts_sorted_and_deduped():
    sheet = validate_cues(RAW, total_duration=120.0)
    times = [i.time for i in sheet.inserts]
    assert times == sorted(times)
    # the 32.0 insert is within 10s of the 30.0 one and should be dropped
    assert len(sheet.inserts) == 2


def test_insert_duration_clamped():
    sheet = validate_cues(RAW, total_duration=120.0)
    for ins in sheet.inserts:
        assert 2.0 <= ins.duration <= 12.0


def test_underlay_overlap_dropped():
    sheet = validate_cues(RAW, total_duration=120.0)
    spans = [(u.start, u.end) for u in sheet.underlays]
    for (s1, e1), (s2, e2) in itertools.pairwise(spans):
        assert e1 <= s2, "underlays must not overlap"


def test_underlay_too_short_dropped():
    sheet = validate_cues(RAW, total_duration=120.0)
    for u in sheet.underlays:
        assert u.end - u.start >= 5.0


def test_underlay_clamped_to_duration():
    sheet = validate_cues(RAW, total_duration=120.0)
    for u in sheet.underlays:
        assert u.end <= 120.0


def test_unknown_mood_defaults():
    sheet = validate_cues(RAW, total_duration=120.0)
    for cue in list(sheet.inserts) + list(sheet.underlays):
        assert cue.mood in MOODS


def test_validate_cues_parses_clips_sorted_and_clamped():
    raw = {
        "clips": [
            {"time": 40.0, "clip_id": "q2", "reason": "later quote"},
            {"time": 10.0, "clip_id": "q1", "gain": 5.0},
            {"time": 20.0, "reason": "no clip_id — dropped"},
        ],
    }
    sheet = validate_cues(raw, total_duration=120.0)
    assert [c.clip_id for c in sheet.clips] == ["q1", "q2"]
    assert 0.0 < sheet.clips[0].gain <= 2.0


def test_sheet_from_dict_round_trips_clips():
    raw = {
        "inserts": [{"time": 30.0, "duration": 5.0, "mood": "warm",
                     "reason": "x", "clip_id": None}],
        "underlays": [],
        "clips": [{"time": 12.0, "clip_id": "q1", "reason": "the quote", "gain": 1.0}],
    }
    sheet = sheet_from_dict(raw)
    assert dataclasses.asdict(sheet) == raw


def test_sheet_from_dict_tolerates_missing_clips_key():
    raw = {"inserts": [], "underlays": []}
    sheet = sheet_from_dict(raw)
    assert sheet.clips == []
