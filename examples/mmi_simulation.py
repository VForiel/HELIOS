import numpy as np
from helios.sim import mmi_contributions

mmi_contributions(
    N=2,
    M=2,
    L=None,
    W=None,
    n_eff=2.0458,
    wavelength=1.55e-6,
    input_amplitudes=np.sqrt(1/2)*np.array([1, 1j], dtype=complex),
    num_modes=50,
    # Using default z_resolution = lambda/30 (~0.05 um)
    # output_file will have frames corresponding to lambda/30 steps.
    output_file="2x2_nuller_mmi.mp4",
    verbose=True
)

mmi_contributions(
    N=4,
    M=4,
    L=None,
    W=20e-6,
    n_eff=2.0458,
    wavelength=1.55e-6,
    input_amplitudes=np.sqrt(1/4)*np.array([1, 1j, 1, 1j], dtype=complex),
    num_modes=50,
    # z_resolution=1.0e-6, # Removed to use default high-res (lambda/30)
    output_file="4x4_kernel_nuller_mmi.mp4", 
    verbose=True
)