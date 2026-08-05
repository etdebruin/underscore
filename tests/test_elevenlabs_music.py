import io
import subprocess

import numpy as np
import soundfile as sf

from underscore.elevenlabs_music import ElevenLabsMusic, build_prompt

SR = 44100


def _wav_bytes(seconds: float, sr: int = SR) -> bytes:
    """A tiny valid stereo wav clip standing in for the API's mp3 response."""
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, np.stack([tone, tone], axis=1), sr, format="WAV")
    return buf.getvalue()


class FakeFetcher:
    def __init__(self, seconds: float = 6.0):
        self.calls: list[tuple[str, int]] = []
        self.seconds = seconds

    def __call__(self, prompt: str, length_ms: int) -> bytes:
        self.calls.append((prompt, length_ms))
        return _wav_bytes(self.seconds)


def test_build_prompt_mentions_mood_and_reason():
    p = build_prompt("wistful", "tender passage about the birds")
    assert "instrumental" in p.lower()
    assert "tender passage about the birds" in p


def test_bed_shape_level_and_exact_length(tmp_path):
    fetcher = FakeFetcher(seconds=6.0)
    backend = ElevenLabsMusic(api_key="k", cache_dir=tmp_path, fetcher=fetcher)
    bed = backend.bed("warm", duration=5.0, sr=SR, reason="intro")
    assert bed.shape == (SR * 5, 2)
    assert bed.dtype == np.float32
    assert 0.05 < np.max(np.abs(bed)) <= 0.9


def test_bed_padded_when_clip_is_short(tmp_path):
    fetcher = FakeFetcher(seconds=4.0)
    backend = ElevenLabsMusic(api_key="k", cache_dir=tmp_path, fetcher=fetcher)
    bed = backend.bed("warm", duration=8.0, sr=SR, reason="")
    assert bed.shape == (SR * 8, 2)


def test_cache_avoids_second_fetch(tmp_path):
    fetcher = FakeFetcher()
    backend = ElevenLabsMusic(api_key="k", cache_dir=tmp_path, fetcher=fetcher)
    backend.bed("tense", duration=5.0, sr=SR, reason="act break")
    backend.bed("tense", duration=5.0, sr=SR, reason="act break")
    assert len(fetcher.calls) == 1


def test_requested_length_never_below_api_minimum(tmp_path):
    fetcher = FakeFetcher(seconds=4.0)
    backend = ElevenLabsMusic(api_key="k", cache_dir=tmp_path, fetcher=fetcher)
    backend.bed("warm", duration=1.5, sr=SR, reason="")
    _, length_ms = fetcher.calls[0]
    assert length_ms >= 3000


def test_ffmpeg_available_for_decode():
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, check=False)
    assert result.returncode == 0
