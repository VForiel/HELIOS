import os
import sys

import numpy as np
import pytest

# Ensure local src/ is importable when tests are run directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import helios.sim.mmi as mmi_module
from helios.sim.mmi import (
    _calibrate_phases_genetic_like,
    _wrap_phase_radians,
    calibrate_input_phases_genetic,
)


def test_calibrate_input_phases_does_not_fix_phase_of_input_1(monkeypatch):
    """The MMI input-phase calibration should not clamp input-1 as a reference.

    Historically, the calibration routine removed the global phase degeneracy by forcing
    the first input phase (index 0) to zero. This is convenient numerically but can be
    undesirable when interfacing with hardware where there is no absolute phase origin.

    This test monkeypatches the heavy physical simulation with a lightweight linear
    model whose intensities are invariant to global phase. We assert that the first call
    to the simulation receives the user-provided phase on input 1, i.e. it is not
    overwritten to 0.
    """

    N = 4
    M = 4

    expected_phase0 = 0.7
    input_amplitudes = (np.ones(N, dtype=complex) * np.exp(1j * 0.0)) / np.sqrt(N)
    input_amplitudes[0] = np.exp(1j * expected_phase0) / np.sqrt(N)

    rng = np.random.default_rng(0)
    mixing = rng.normal(size=(M, N)) + 1j * rng.normal(size=(M, N))

    observed = {}

    def fake_simulate(*, N, M, input_amplitudes, **kwargs):
        amps = np.asarray(input_amplitudes, dtype=complex)
        if "phase0" not in observed:
            observed["phase0"] = float(np.mod(np.angle(amps[0]), 2 * np.pi))
        return mixing @ amps

    monkeypatch.setattr(mmi_module, "simulate", fake_simulate)

    calibrate_input_phases_genetic(
        N=N,
        M=M,
        input_amplitudes=input_amplitudes,
        bright_output_idx=0,
        epsilon=10.0,  # Skip optimization loop; still evaluates the metric once.
        verbose=False,
    )

    assert "phase0" in observed
    assert np.isclose(observed["phase0"], expected_phase0, atol=1e-12)


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
