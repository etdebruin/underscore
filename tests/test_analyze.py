from dataclasses import dataclass

from underscore.analyze import analyze
from underscore.cues import Clip
from underscore.transcribe import Transcript, Word


@dataclass
class _Parsed:
    parsed_output: object


class _StubClient:
    """Captures the prompt and returns an empty cue sheet."""

    def __init__(self, fail_first: int = 0):
        self.calls = []
        self.messages = self
        self.fail_first = fail_first

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        from underscore.analyze import RawCueSheet

        if len(self.calls) <= self.fail_first:
            return _Unparsed()
        return _Parsed(RawCueSheet(inserts=[], underlays=[]))


@dataclass
class _Unparsed:
    """What the SDK hands back when the model didn't finish a valid response."""

    parsed_output: object = None
    stop_reason: str = "max_tokens"
    content: tuple = ()


def _transcript() -> Transcript:
    words = [Word(text="Hello.", start=0.0, end=1.0), Word(text="World.", start=1.0, end=2.0)]
    return Transcript(text="Hello. World.", words=words, duration=60.0)


def test_analyze_without_clips_says_nothing_about_them():
    client = _StubClient()
    analyze(_transcript(), client=client)
    user = client.calls[0]["messages"][0]["content"]
    assert "subject's own recorded voice" not in client.calls[0]["system"]
    assert "CLIP" not in user


def test_analyze_retries_with_more_room_when_the_response_was_truncated():
    client = _StubClient(fail_first=1)
    analyze(_transcript(), client=client)
    assert len(client.calls) == 2
    assert client.calls[1]["max_tokens"] > client.calls[0]["max_tokens"]


def test_analyze_reports_why_instead_of_crashing_on_an_attribute():
    client = _StubClient(fail_first=99)
    try:
        analyze(_transcript(), client=client)
        raised = None
    except RuntimeError as exc:
        raised = str(exc)
    assert raised is not None
    assert "max_tokens" in raised  # says why, not 'NoneType has no attribute'


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
