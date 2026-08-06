import io

import numpy as np
import soundfile as sf

from underscore.import_clip import main
from underscore.library import Library

SR = 44100


def _wav_bytes(seconds: float) -> bytes:
    t = np.linspace(0, seconds, int(seconds * SR), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, np.stack([tone, tone], axis=1), SR, format="WAV")
    return buf.getvalue()


def test_import_registers_speech_clip(tmp_path):
    src = tmp_path / "quote.wav"
    src.write_bytes(_wav_bytes(3.0))
    rc = main([
        str(src),
        "--clip-id", "ilya-neurips-1",
        "--description", "Ilya on stage at NeurIPS",
        "--library", str(tmp_path / "lib"),
    ])
    assert rc == 0
    lib = Library(tmp_path / "lib")
    assert lib.has("ilya-neurips-1")
    (entry,) = lib.catalog()
    assert entry.clip_id == "ilya-neurips-1"
    assert entry.kind == "speech"
    assert entry.description == "Ilya on stage at NeurIPS"
    assert 2900 <= entry.length_ms <= 3100


def test_import_default_clip_id_from_filename(tmp_path):
    src = tmp_path / "bret-taylor-quote.wav"
    src.write_bytes(_wav_bytes(2.0))
    rc = main([str(src), "--library", str(tmp_path / "lib")])
    assert rc == 0
    assert Library(tmp_path / "lib").has("bret-taylor-quote")
