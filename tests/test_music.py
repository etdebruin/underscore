import numpy as np
import pytest

from underscore.cues import MOODS
from underscore.music import generate_bed

SR = 44100


@pytest.mark.parametrize("mood", sorted(MOODS))
def test_bed_shape_and_level(mood):
    bed = generate_bed(mood, duration=8.0, sr=SR, seed=7)
    assert bed.shape == (SR * 8, 2)
    assert bed.dtype == np.float32
    peak = np.max(np.abs(bed))
    assert 0.05 < peak <= 0.9, f"{mood} peak={peak}"


def test_bed_deterministic_with_seed():
    a = generate_bed("warm", duration=2.0, sr=SR, seed=42)
    b = generate_bed("warm", duration=2.0, sr=SR, seed=42)
    assert np.array_equal(a, b)
