import os
import sys

import numpy as np
import pytest

# Ensure local src/ is importable when tests are run directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from helios.sim.mmi import _compute_symmetric_port_positions


def test_default_positions_match_historical_formula():
    """Default spacing must reproduce the historical (i+0.5)*W/N placement.

    This ensures backward compatibility of the port geometry when Din/Dout are not provided.
    """
    W = 10.0e-6
    N = 4

    positions = _compute_symmetric_port_positions(N, W, spacing=None, name="input")
    expected = [(i + 0.5) * (W / N) for i in range(N)]

    assert np.allclose(positions, expected)

    # Symmetry about x=W/2
    centered = np.array(positions) - 0.5 * W
    assert np.allclose(centered, -centered[::-1])


def test_custom_spacing_is_symmetric_about_centerline():
    """Custom spacing must place ports symmetrically around x=W/2."""
    W = 20.0e-6
    N = 4
    Din = 4.0e-6

    positions = _compute_symmetric_port_positions(N, W, spacing=Din, name="input")

    centered = np.array(positions) - 0.5 * W
    assert np.allclose(centered, -centered[::-1])

    # Physical bounds
    assert np.min(positions) >= 0.0
    assert np.max(positions) <= W


def test_spacing_too_large_raises_value_error():
    """If spacing pushes any port outside [0, W], the helper must raise."""
    W = 10.0e-6
    N = 4

    # Far too large for N=4.
    Din = W

    with pytest.raises(ValueError):
        _compute_symmetric_port_positions(N, W, spacing=Din, name="input")


def test_non_positive_spacing_raises_value_error():
    """Spacing must be strictly positive when provided."""
    W = 10.0e-6
    N = 2

    with pytest.raises(ValueError):
        _compute_symmetric_port_positions(N, W, spacing=0.0, name="input")

    with pytest.raises(ValueError):
        _compute_symmetric_port_positions(N, W, spacing=-1.0e-6, name="input")
