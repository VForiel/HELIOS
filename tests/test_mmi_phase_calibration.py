import os
import sys

import numpy as np
import pytest

# Ensure local src/ is importable when tests are run directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from helios.sim.mmi import _calibrate_phases_genetic_like, _wrap_phase_radians


def test_genetic_like_calibration_improves_metric_on_mock_objective():
    """The genetic-like optimizer should reduce a simple smooth metric.

    We use a deterministic objective with a known optimum away from the origin.
    This avoids relying on the physical MMI simulation (which can be heavier and
    may change with model details).
    """

    target = np.array([0.0, 0.3, 5.1, 2.2])  # radians (wrapped)
    n = len(target)

    def metric(phases):
        phases = _wrap_phase_radians(phases)
        # Quadratic bowl around target with phase wrap handled via shortest distance.
        d = np.angle(np.exp(1j * (phases - target)))
        return float(np.sum(d**2))

    initial = np.array([0.0, 4.0, 1.0, 0.1])
    res = _calibrate_phases_genetic_like(
        evaluate_metric=metric,
        n_phases=n,
        beta=0.8,
        initial_step=np.pi / 2,
        epsilon=1e-3,
        initial_phases=initial,
        fixed_indices={0},
        max_outer_iterations=80,
        verbose=False,
    )

    assert res["metric"].ndim == 1
    assert res["phases"].shape[1] == n
    assert np.isfinite(res["best_metric"])
    assert res["best_metric"] <= metric(initial)


def test_invalid_beta_raises():
    def metric(phases):
        return float(np.sum(np.asarray(phases) ** 2))

    with pytest.raises(ValueError):
        _calibrate_phases_genetic_like(metric, n_phases=2, beta=0.4)

    with pytest.raises(ValueError):
        _calibrate_phases_genetic_like(metric, n_phases=2, beta=1.0)
