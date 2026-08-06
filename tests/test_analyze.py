from dataclasses import dataclass

from underscore.analyze import analyze
from underscore.cues import Clip
from underscore.transcribe import Transcript, Word


@dataclass
class _Parsed:
    parsed_output: object


class _StubClient:
    """Captures the prompt and returns an empty cue sheet."""

    def __init__(self):
        self.calls = []
        self.messages = self

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        from underscore.analyze import RawCueSheet

        return _Parsed(RawCueSheet(inserts=[], underlays=[]))


def _transcript() -> Transcript:
    words = [Word(text="Hello.", start=0.0, end=1.0), Word(text="World.", start=1.0, end=2.0)]
    return Transcript(text="Hello. World.", words=words, duration=60.0)


def test_analyze_without_clips_says_nothing_about_them():
    client = _StubClient()
    analyze(_transcript(), client=client)
    user = client.calls[0]["messages"][0]["content"]
    assert "subject's own recorded voice" not in client.calls[0]["system"]
    assert "CLIP" not in user


def test_analyze_tells_the_producer_where_the_voice_clips_land():
    client = _StubClient()
    clips = [
        Clip(time=14.6, clip_id="q1", reason="Ilya at TED, 2023"),
        Clip(time=90.0, clip_id="q2", reason="Ilya on Dwarkesh"),
    ]
    analyze(_transcript(), client=client, clips=clips)
    call = client.calls[0]
    assert "subject's own recorded voice" in call["system"]
    user = call["messages"][0]["content"]
    # same [  ss.sss] shape as the transcript lines, so times read consistently
    assert "14.60s] CLIP q1 — Ilya at TED, 2023" in user
    assert "90.00s] CLIP q2 — Ilya on Dwarkesh" in user
