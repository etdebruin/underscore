import numpy as np

from underscore.cues import CueSheet, Insert, Underlay
from underscore.mix import assemble, duck_envelope, find_gaps, shift_time, snap_to_gap

SR = 44100


def test_find_gaps_between_speech():
    spans = [(0.5, 3.0), (4.0, 8.0)]
    gaps = find_gaps(spans, total=10.0)
    assert gaps == [(0.0, 0.5), (3.0, 4.0), (8.0, 10.0)]


def test_snap_to_nearest_gap_center():
    gaps = [(3.0, 4.0), (8.0, 10.0)]
    assert snap_to_gap(3.7, gaps) == 3.5
    assert snap_to_gap(8.4, gaps) == 9.0


def test_snap_falls_back_when_no_gap_near():
    gaps = [(100.0, 101.0)]
    assert snap_to_gap(5.0, gaps, max_dist=6.0) == 5.0


def test_shift_time_accumulates_insert_durations():
    inserts = [Insert(time=10.0, duration=4.0, mood="warm", reason=""),
               Insert(time=20.0, duration=6.0, mood="warm", reason="")]
    assert shift_time(5.0, inserts) == 5.0
    assert shift_time(15.0, inserts) == 19.0
    assert shift_time(25.0, inserts) == 35.0


def test_duck_envelope_ducks_speech_and_opens_gaps():
    n = SR * 10
    speech = [(0.0, 4.0), (6.0, 10.0)]
    env = duck_envelope(n, SR, speech, duck=0.2, full=1.0)
    assert env.shape == (n,)
    # middle of a speech span is ducked
    assert env[SR * 2] < 0.35
    # middle of the gap opens up
    assert env[SR * 5] > 0.6


def test_assemble_lengthens_by_insert_durations():
    dur = 30.0
    voice = np.zeros(int(SR * dur), dtype=np.float32)
    sheet = CueSheet(
        inserts=[Insert(time=15.0, duration=5.0, mood="warm", reason="")],
        underlays=[],
    )
    out = assemble(voice, SR, sheet, speech_spans=[(0.0, 14.5), (15.5, 30.0)])
    expected = int(SR * (dur + 5.0))
    assert abs(out.shape[0] - expected) < SR * 0.1
    assert out.ndim == 2 and out.shape[1] == 2


def test_assemble_output_does_not_clip():
    rng = np.random.default_rng(0)
    voice = (rng.standard_normal(SR * 20) * 0.3).astype(np.float32)
    sheet = CueSheet(
        inserts=[Insert(time=10.0, duration=4.0, mood="tense", reason="")],
        underlays=[Underlay(start=2.0, end=9.0, mood="warm", reason="")],
    )
    out = assemble(voice, SR, sheet, speech_spans=[(0.0, 9.8), (10.2, 20.0)])
    assert np.max(np.abs(out)) <= 1.0
