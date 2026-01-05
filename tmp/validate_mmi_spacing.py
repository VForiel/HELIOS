"""Validation script for Din/Dout port spacing in helios.sim.mmi.

This script is intentionally lightweight and avoids plotting so it can run reliably
in headless CI or agent environments.
"""

import os
import sys

import numpy as np

# Ensure local src/ is importable when run directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from helios.sim.mmi import compute_contributions


def main() -> None:
    W = 10.0e-6

    data = compute_contributions(
        N=2,
        M=2,
        L=50e-6,
        W=W,
        n_eff=2.0458,
        wavelength=1.55e-6,
        input_amplitudes=[1 / np.sqrt(2), 1j / np.sqrt(2)],
        num_modes=20,
        num_z_steps=10,
        z_resolution=None,
        Din=4.0e-6,
        Dout=4.0e-6,
        verbose=False,
    )

    input_pos = np.array(data["input_positions"], dtype=float)
    output_pos = np.array(data["output_positions"], dtype=float)

    assert input_pos.shape == (2,)
    assert output_pos.shape == (2,)
    assert np.all(input_pos >= 0) and np.all(input_pos <= W)
    assert np.all(output_pos >= 0) and np.all(output_pos <= W)

    # Symmetry about x=W/2
    assert np.allclose(input_pos - 0.5 * W, -(input_pos[::-1] - 0.5 * W))
    assert np.allclose(output_pos - 0.5 * W, -(output_pos[::-1] - 0.5 * W))

    # Spacing too large must fail.
    try:
        compute_contributions(
            N=4,
            M=4,
            L=50e-6,
            W=20e-6,
            n_eff=2.0458,
            wavelength=1.55e-6,
            input_amplitudes=[0.5, 0.5j, 0.5, 0.5j],
            num_modes=20,
            num_z_steps=5,
            z_resolution=None,
            Din=20e-6,
            Dout=20e-6,
            verbose=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for oversized Din/Dout.")


if __name__ == "__main__":
    main()
    print("MMI Din/Dout validation OK")
