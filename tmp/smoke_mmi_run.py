import numpy as np
from helios.sim.mmi import simulate, calibrate_input_phases_genetic, calibrate_n_core_and_phases


def main() -> None:
    out = simulate(
        N=2,
        M=2,
        W=10.0e-6,
        L=None,
        n_core=2.0458,
        delta_n=0.0958,
        wavelength=1.55e-6,
        input_amplitudes=np.array([1.0, 1.0j], dtype=complex),
        num_modes=30,
        num_z_steps=20,
        Din=None,
        Dout=None,
        Sin=2.0e-6,
        Sout=2.0e-6,
        verbose=False,
    )
    print("simulate OK", np.abs(out) ** 2)

    cal = calibrate_input_phases_genetic(
        N=2,
        M=2,
        W=10.0e-6,
        n_core=2.0458,
        delta_n=0.0958,
        wavelength=1.55e-6,
        input_amplitudes=np.array([1.0, 1.0], dtype=complex),
        bright_output_idx=0,
        num_modes=20,
        num_z_steps=20,
        epsilon=1e-2,
        verbose=False,
    )
    print("phase calibration OK", cal["best_metric"])

    ncore = calibrate_n_core_and_phases(
        N=2,
        M=2,
        W=10.0e-6,
        n_core_initial=2.0458,
        delta_n=0.0958,
        wavelength=1.55e-6,
        input_amplitudes=np.array([1.0, 1.0], dtype=complex),
        bright_output_idx=0,
        num_modes=20,
        num_z_steps=20,
        n_core_steps_coarse=4,
        gradient_convergence_threshold=5e-3,
        epsilon=1e-2,
        verbose=False,
    )
    print("n_core calibration OK", ncore["best_n_core"], ncore["best_metric"])


if __name__ == "__main__":
    main()
