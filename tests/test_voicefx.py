import numpy as np
import soundfile as sf

from underscore.voicefx import build_filter, process_voice

SR = 44100


def test_filter_off_is_empty():
    assert build_filter(level=False, warm=False) == ""


def test_filter_level_only():
    f = build_filter(level=True, warm=False)
    assert "acompressor" in f and "dynaudnorm" in f
    assert "equalizer" not in f


def test_filter_warm_only():
    f = build_filter(level=False, warm=True)
    assert "highpass" in f and "equalizer" in f and "deesser" in f
    assert "acompressor" not in f


def test_filter_warm_precedes_level():
    f = build_filter(level=True, warm=True)
    assert f.index("highpass") < f.index("acompressor")


def test_process_voice_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    t = np.linspace(0, 3.0, SR * 3, endpoint=False)
    speechish = (0.3 * np.sin(2 * np.pi * 180 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))
                 + 0.01 * rng.standard_normal(SR * 3)).astype(np.float32)
    src = tmp_path / "v.wav"
    sf.write(src, speechish, SR)

    out = process_voice(str(src), SR, level=True, warm=True)
    assert out.ndim == 1
    assert out.dtype == np.float32
    assert abs(len(out) - SR * 3) < SR * 0.2
    assert np.max(np.abs(out)) <= 1.0
