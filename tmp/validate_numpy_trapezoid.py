import inspect
import numpy as np

from helios.sim import mmi


def main() -> None:
    print(f"numpy_version={np.__version__}")
    print(f"mmi_module={inspect.getfile(mmi)}")

    output = mmi.simulate(
        N=4,
        M=4,
        W=20.0e-6,
        L=440.0e-6,
        n_core=2.0458,
        delta_n=0.1,
        wavelength=1.55e-6,
        input_amplitudes=np.ones(4, dtype=complex) / np.sqrt(4),
        num_modes=20,
        num_z_steps=80,
        verbose=False,
    )

    print(f"output_shape={output.shape}")
    print(f"output_dtype={output.dtype}")
    print(f"power_sum={np.sum(np.abs(output) ** 2):.6f}")


if __name__ == "__main__":
    main()
