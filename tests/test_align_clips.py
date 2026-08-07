from underscore.align_clips import align, parse_weave
from underscore.transcribe import Transcript, Word

WEAVE = """He landed at the University of Toronto, and that's where the story turns.

[CLIP ilya-q1 | Ilya on stage at NeurIPS, 2015 | "the models, they just want to learn"]

That confidence — that was the whole bet.

Years later the bet paid off, and everyone wanted to know what he saw first.

[CLIP ilya-q2 | Ilya on the No Priors podcast]

Which brings us to this week.
"""


def _words(text: str, start: float = 0.0, per: float = 0.4) -> list[Word]:
    out, t = [], start
    for w in text.split():
        out.append(Word(text=w, start=t, end=t + per))
        t += per
    return out


def _transcript() -> Transcript:
    # The narration as actually read (skipping the markers), with light drift:
    # whisper hears "that is" where the weave says "that's".
    text = (
        "He landed at the University of Toronto and that is where the story turns. "
        "That confidence that was the whole bet. "
        "Years later the bet paid off and everyone wanted to know what he saw first. "
        "Which brings us to this week."
    )
    words = _words(text)
    return Transcript(text=text, words=words, duration=words[-1].end)


def test_parse_weave_finds_markers_with_anchor_tails():
    anchors = parse_weave(WEAVE)
    assert [a.clip_id for a in anchors] == ["ilya-q1", "ilya-q2"]
    assert anchors[0].label == "Ilya on stage at NeurIPS, 2015"
    assert anchors[1].label == "Ilya on the No Priors podcast"
    assert anchors[0].tail[-3:] == ["the", "story", "turns."]


def test_align_places_clips_after_their_anchor_sentences():
    transcript = _transcript()
    clips = align(WEAVE, transcript)
    assert [c.clip_id for c in clips] == ["ilya-q1", "ilya-q2"]
    # first clip lands right after "...the story turns." (word 14 of the read)
    turns_end = transcript.words[13].end
    assert abs(clips[0].time - turns_end) < 1.0
    # anchors are matched monotonically — the second clip comes later
    assert clips[1].time > clips[0].time
    saw_first = transcript.words[-7].end  # "...what he saw first."
    assert abs(clips[1].time - saw_first) < 2.0


def test_align_carries_label_as_reason():
    clips = align(WEAVE, _transcript())
    assert clips[0].reason == "Ilya on stage at NeurIPS, 2015"


def test_align_raises_when_anchor_not_spoken():
    weave = "A sentence that was never actually read aloud.\n\n[CLIP q1 | somewhere]\n"
    try:
        align(weave, _transcript())
        raised = False
    except ValueError:
        raised = True
    assert raised


COLD_OPEN_WEAVE = """[CLIP ilya-q0 | Ilya on the No Priors podcast | "it was obvious to me"]

He landed at the University of Toronto, and that's where the story turns.

[CLIP ilya-q1 | Ilya on stage at NeurIPS, 2015]

That confidence — that was the whole bet.
"""


def test_leading_marker_is_a_cold_open_at_time_zero():
    """A marker above all narration opens the episode: it plays before the read."""
    clips = align(COLD_OPEN_WEAVE, _transcript())
    assert [c.clip_id for c in clips] == ["ilya-q0", "ilya-q1"]
    assert clips[0].time == 0.0
    assert clips[0].reason == "Ilya on the No Priors podcast"
    # the anchored clip still lands after its narration
    assert clips[1].time > 1.0


def test_align_still_raises_for_a_later_clip_with_no_narration_before_it():
    weave = (
        "[CLIP q0 | cold open]\n\n"
        "[CLIP q1 | nothing to anchor on]\n\n"
        "He landed at the University of Toronto, and that's where the story turns.\n"
    )
    try:
        align(weave, _transcript())
        raised = False
    except ValueError:
        raised = True
    assert raised
