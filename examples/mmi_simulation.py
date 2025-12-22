import numpy as np
from helios.sim import mmi as simulate_mmi

simulate_mmi(N=2, M=2, L=None, W=None, n_eff=2.0458, wavelength=1.55e-6, input_amplitudes=np.sqrt(1/2)*np.array([1, 1j], dtype=complex), num_modes=50, num_z_steps=200, output_file="2x2_nuller_mmi.mp4")

simulate_mmi(N=4, M=4, L=None, W=20e-6, n_eff=2.0458, wavelength=1.55e-6, input_amplitudes=np.sqrt(1/4)*np.array([1, 1j, -1, -1j], dtype=complex), num_modes=50, num_z_steps=200, output_file="4x4_kernel_nuller_mmi.mp4")