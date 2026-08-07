import numpy as np
import pytest

from underscore.cues import Clip, CueSheet, Insert, Underlay
from underscore.mix import (
    CLIP_PAD_POST,
    CLIP_PAD_PRE,
    assemble,
    duck_envelope,
    find_gaps,
    shift_time,
    snap_to_gap,
)

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


def test_bed_fn_receives_insert_clip_id():
    seen = []

    def bed_fn(mood, duration, sr, seed=0, reason="", clip_id=None):
        seen.append(clip_id)
        return np.zeros((int(duration * sr), 2), dtype=np.float32)

    voice = np.zeros(SR * 30, dtype=np.float32)
    sheet = CueSheet(
        inserts=[Insert(time=15.0, duration=4.0, mood="warm", reason="", clip_id="abc123")],
    )
    assemble(voice, SR, sheet, speech_spans=[(0.0, 14.5), (15.5, 30.0)], bed_fn=bed_fn)
    assert seen == ["abc123"]


def _const_clip(seconds: float, value: float = 0.25) -> np.ndarray:
    return np.full((int(seconds * SR), 2), value, dtype=np.float32)


def test_clip_spliced_verbatim_at_natural_length():
    voice = np.zeros(SR * 30, dtype=np.float32)
    sheet = CueSheet(clips=[Clip(time=15.0, clip_id="quote-1")])
    out = assemble(
        voice, SR, sheet, speech_spans=[(0.0, 14.5), (15.5, 30.0)],
        clip_fn=lambda cid, sr: _const_clip(3.0),
    )
    expected = int(SR * (30.0 + CLIP_PAD_PRE + 3.0 + CLIP_PAD_POST))
    assert abs(out.shape[0] - expected) < SR * 0.1
    # the quote plays unscaled (no loop, no peak-normalize) in the spliced pause
    mid = int((15.0 + CLIP_PAD_PRE + 1.5) * SR)
    assert np.isclose(out[mid, 0], 0.25, atol=0.01)
    assert np.isclose(np.max(np.abs(out)), 0.25, atol=0.01)


def test_clip_gain_applied():
    voice = np.zeros(SR * 30, dtype=np.float32)
    sheet = CueSheet(clips=[Clip(time=15.0, clip_id="quote-1", gain=0.5)])
    out = assemble(
        voice, SR, sheet, speech_spans=[(0.0, 14.5), (15.5, 30.0)],
        clip_fn=lambda cid, sr: _const_clip(3.0),
    )
    assert np.isclose(np.max(np.abs(out)), 0.125, atol=0.01)


def test_clips_require_clip_fn():
    voice = np.zeros(SR * 10, dtype=np.float32)
    sheet = CueSheet(clips=[Clip(time=5.0, clip_id="quote-1")])
    with pytest.raises(ValueError):
        assemble(voice, SR, sheet, speech_spans=[(0.0, 10.0)])


def test_shift_time_counts_clip_splices():
    voice = np.zeros(SR * 30, dtype=np.float32)
    sheet = CueSheet(
        inserts=[Insert(time=25.0, duration=4.0, mood="warm", reason="")],
        clips=[Clip(time=15.0, clip_id="quote-1")],
    )
    out = assemble(
        voice, SR, sheet,
        speech_spans=[(0.0, 14.5), (15.5, 24.5), (25.5, 30.0)],
        clip_fn=lambda cid, sr: _const_clip(3.0),
    )
    expected = int(SR * (30.0 + CLIP_PAD_PRE + 3.0 + CLIP_PAD_POST + 4.0))
    assert abs(out.shape[0] - expected) < SR * 0.1


def test_underlay_ducks_under_clip():
    voice = np.zeros(SR * 30, dtype=np.float32)
    speech = [(0.0, 12.0), (18.0, 30.0)]
    sheet = CueSheet(
        underlays=[Underlay(start=1.0, end=29.0, mood="warm", reason="")],
        clips=[Clip(time=15.0, clip_id="quote-1")],
    )

    def bed_fn(mood, duration, sr, seed=0, reason="", clip_id=None):
        return np.full((int(duration * sr), 2), 0.5, dtype=np.float32)

    out = assemble(
        voice, SR, sheet, speech_spans=speech, bed_fn=bed_fn,
        clip_fn=lambda cid, sr: np.zeros((SR * 3, 2), dtype=np.float32),  # silent: isolate the bed
    )
    clip_center = int((15.0 + CLIP_PAD_PRE + 1.5) * SR)
    open_gap = int(13.0 * SR)  # inside the same silence, before the clip: bed at full
    assert abs(out[clip_center, 0]) < abs(out[open_gap, 0]) * 0.6


def test_spliced_gap_carries_room_tone_not_digital_silence():
    """A hole of absolute silence between two different rooms is what makes a
    clip sound like the audio stopped and restarted."""
    rng = np.random.default_rng(0)
    voice = (rng.standard_normal(SR * 30) * 0.01).astype(np.float32)  # real room floor
    sheet = CueSheet(clips=[Clip(time=15.0, clip_id="q")])
    out = assemble(
        voice, SR, sheet, speech_spans=[(0.0, 14.5), (15.5, 30.0)],
        clip_fn=lambda cid, sr: np.zeros((SR * 3, 2), dtype=np.float32),
    )
    pad = out[int(15.05 * SR) : int(15.3 * SR), 0]
    assert float(np.sqrt(np.mean(pad**2))) > 0.002, "spliced pad fell to digital silence"


def test_dead_air_between_joined_takes_gets_room_tone():
    """Paragraph takes are joined with synthesized silence, leaving holes in the
    middle of the read — not just at the clips."""
    rng = np.random.default_rng(1)
    voice = (rng.standard_normal(SR * 30) * 0.01).astype(np.float32)
    voice[int(10 * SR) : int(10.45 * SR)] = 0.0  # a join between two takes
    sheet = CueSheet()
    # a natural pause mid-take (real room tone) alongside the synthetic join
    out = assemble(voice, SR, sheet, speech_spans=[(0.0, 5.0), (6.0, 10.0), (10.45, 30.0)])
    hole = out[int(10.1 * SR) : int(10.35 * SR), 0]
    assert float(np.sqrt(np.mean(hole**2))) > 0.002, "join stayed digitally silent"


def test_room_tone_fill_invents_nothing_when_the_source_is_silent():
    voice = np.zeros(SR * 30, dtype=np.float32)
    sheet = CueSheet(clips=[Clip(time=15.0, clip_id="q")])
    out = assemble(
        voice, SR, sheet, speech_spans=[(0.0, 14.5), (15.5, 30.0)],
        clip_fn=lambda cid, sr: np.zeros((SR * 3, 2), dtype=np.float32),
    )
    assert float(np.max(np.abs(out))) == 0.0


def test_assemble_output_does_not_clip():
    rng = np.random.default_rng(0)
    voice = (rng.standard_normal(SR * 20) * 0.3).astype(np.float32)
    sheet = CueSheet(
        inserts=[Insert(time=10.0, duration=4.0, mood="tense", reason="")],
        underlays=[Underlay(start=2.0, end=9.0, mood="warm", reason="")],
    )
    out = assemble(voice, SR, sheet, speech_spans=[(0.0, 9.8), (10.2, 20.0)])
    assert np.max(np.abs(out)) <= 1.0
