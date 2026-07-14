import pytest
from src.metrics import compression_ratio

def test_invariance():
    """ Test that the ratio for grayscale images is the same as for RGB images."""
    ratio1 = compression_ratio((100, 50), 10)
    ratio2 = compression_ratio((100, 50, 3), 10)
    assert pytest.approx(ratio1) == pytest.approx(ratio2)

def test_monotonicity():
    """ Test that higher k gives a lower compression ratio."""
    ratio1 = compression_ratio((100, 50), 5)
    ratio2 = compression_ratio((100, 50), 20)
    assert ratio1 > ratio2
