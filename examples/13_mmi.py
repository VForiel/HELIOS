"""
13_mmi.py

Demonstrates `simulate_contributions` for two photonic MMI layouts:
1. A 2x2 Bracewell-style nuller
2. A 4x4 kernel nuller

If the environment variable `HELIOS_SAVE_PLOTS` is set to "true", animations are
written under `generated/examples/`. Otherwise they are saved next to this script.
"""
import os
import numpy as np

from helios.sim.mmi import simulate_contributions


def _output_path(filename: str) -> str:
    """Resolve where to save the animation file."""
    base_dir = os.path.dirname(__file__)
    if os.environ.get("HELIOS_SAVE_PLOTS") == "true":
        target_dir = os.path.abspath(os.path.join(base_dir, "../generated/examples"))
    else:
        target_dir = base_dir

    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, filename)


def _run_case(label: str, **kwargs) -> None:
    """Run one simulation case and print progress."""
    output_path = kwargs.get("output_file")
    print(f"▶ {label} ...")
    simulate_contributions(**kwargs, verbose=True)
    if output_path:
        print(f"Saved animation to {output_path}")
    print("-" * 60)


def run_demo() -> None:
    """Execute both MMI contribution demonstrations."""
    two_by_two_path = _output_path("../generated/examples/13_mmi_2x2_nuller_mmi.mp4")
    four_by_four_path = _output_path("../generated/examples/13_mmi_4x4_kernel_nuller_mmi.mp4")

    cases = [
        (
            "2x2 Bracewell nuller",
            dict(
                N=2,
                M=2,
                L=100e-6,
                W=10.0e-6,
                Din=5.0e-6,
                Dout=5.0e-6,
                Sin=2.5e-6,
                Sout=2.5e-6,
                n_core=2.0458,
                delta_n=0.0958,
                wavelength=1.55e-6,
                input_amplitudes=np.sqrt(0.5) * np.array([1, 1j], dtype=complex),
                num_modes=50,
                z_resolution=1.0e-6,
                output_file=two_by_two_path,
            ),
        ),
        (
            "4x4 kernel nuller",
            dict(
                N=4,
                M=4,
                L=400e-6,
                W=20e-6,
                Din=4.0e-6,
                Dout=4.0e-6,
                Sin=5.0e-6,
                Sout=5.0e-6,
                n_core=2.0458,
                delta_n=0.0958,
                wavelength=1.55e-6,
                input_amplitudes=np.sqrt(0.25) * np.array([1, 1j, 1, 1j], dtype=complex),
                num_modes=50,
                z_resolution=1.0e-6,
                output_file=four_by_four_path,
            ),
        ),
    ]

    for label, params in cases:
        _run_case(label, **params)

    print("MMI contribution demos complete.")


if __name__ == "__main__":
    run_demo()
