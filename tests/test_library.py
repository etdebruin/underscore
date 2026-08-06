import io

import numpy as np
import soundfile as sf

from underscore.cues import validate_cues
from underscore.elevenlabs_music import ElevenLabsMusic
from underscore.library import Library, catalog_text

SR = 44100


def _wav_bytes(seconds: float) -> bytes:
    t = np.linspace(0, seconds, int(seconds * SR), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, np.stack([tone, tone], axis=1), SR, format="WAV")
    return buf.getvalue()


def _seed_clip(lib: Library, clip_id: str = "abc123", seconds: float = 6.0) -> str:
    lib.path_for(clip_id).write_bytes(_wav_bytes(seconds))
    lib.register(clip_id, mood="warm", description="show open theme", length_ms=6000)
    return clip_id


def test_register_and_catalog(tmp_path):
    lib = Library(tmp_path)
    _seed_clip(lib)
    (entry,) = lib.catalog()
    assert entry.clip_id == "abc123"
    assert entry.mood == "warm"
    assert entry.description == "show open theme"
    assert entry.length_ms == 6000


def test_load_trims_to_duration(tmp_path):
    lib = Library(tmp_path)
    _seed_clip(lib, seconds=6.0)
    clip = lib.load("abc123", duration=4.0, sr=SR)
    assert clip.shape == (SR * 4, 2)
    assert 0.05 < np.max(np.abs(clip)) <= 0.9


def test_load_loops_short_clip(tmp_path):
    lib = Library(tmp_path)
    _seed_clip(lib, seconds=3.0)
    clip = lib.load("abc123", duration=8.0, sr=SR)
    assert clip.shape == (SR * 8, 2)


def test_delete_removes_audio_and_sidecar(tmp_path):
    lib = Library(tmp_path)
    _seed_clip(lib)
    lib.delete("abc123")
    assert not lib.has("abc123")
    assert lib.catalog() == []


def test_catalog_text_lists_clips(tmp_path):
    lib = Library(tmp_path)
    _seed_clip(lib)
    text = catalog_text(lib.catalog())
    assert "abc123" in text and "warm" in text and "show open theme" in text


def test_validate_cues_preserves_clip_id():
    raw = {
        "inserts": [{"time": 30.0, "duration": 5.0, "mood": "warm",
                     "reason": "x", "clip_id": "abc123"}],
        "underlays": [{"start": 50.0, "end": 70.0, "mood": "warm",
                       "reason": "y", "clip_id": None}],
    }
    sheet = validate_cues(raw, total_duration=120.0)
    assert sheet.inserts[0].clip_id == "abc123"
    assert sheet.underlays[0].clip_id is None


def test_load_raw_natural_length_unscaled(tmp_path):
    lib = Library(tmp_path)
    _seed_clip(lib, seconds=3.0)  # tone amplitude 0.5
    clip = lib.load_raw("abc123", sr=SR)
    assert abs(clip.shape[0] - SR * 3) < SR * 0.05
    assert clip.shape[1] == 2
    # verbatim: not peak-normalized to library TARGET_PEAK
    assert 0.45 < np.max(np.abs(clip)) < 0.55


def test_speech_clips_kept_out_of_music_catalog_text(tmp_path):
    lib = Library(tmp_path)
    _seed_clip(lib)
    lib.path_for("quote-1").write_bytes(_wav_bytes(3.0))
    lib.register("quote-1", mood="", description="Ilya on stage at NeurIPS",
                 length_ms=3000, kind="speech")
    entries = lib.catalog()
    assert {e.clip_id for e in entries} == {"abc123", "quote-1"}
    assert next(e for e in entries if e.clip_id == "quote-1").kind == "speech"
    text = catalog_text(entries)
    assert "abc123" in text and "quote-1" not in text


def test_elevenlabs_registers_generated_clip(tmp_path):
    calls = []

    def fetcher(prompt, length_ms):
        calls.append(prompt)
        return _wav_bytes(6.0)

    lib = Library(tmp_path)
    backend = ElevenLabsMusic(api_key="k", library=lib, fetcher=fetcher)
    backend.bed("tense", duration=5.0, sr=SR, reason="act break into part two")
    (entry,) = lib.catalog()
    assert entry.mood == "tense"
    assert entry.description == "act break into part two"
