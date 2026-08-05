import itertools

from underscore.cues import MOODS, validate_cues

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
