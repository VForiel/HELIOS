import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

from helios.sim import lp_modes
from helios.sim import mmi

# Ensure suppression works
print("Before suppression:")
I = lp_modes.compute_lp_mode_profile(
    x_grid=np.linspace(-5e-6, 5e-6, 1001),
    center=0.0,
    core_diameter=6e-6,
    wavelength=1.55e-6,
    n_core=2.05,
    n_cladding=1.95,
    l=0,
    m=2,
)

lp_modes.set_lp_warning_suppression(True)
print("During suppression:")
I2 = lp_modes.compute_lp_mode_profile(
    x_grid=np.linspace(-5e-6, 5e-6, 1001),
    center=0.0,
    core_diameter=6e-6,
    wavelength=1.55e-6,
    n_core=2.05,
    n_cladding=1.95,
    l=0,
    m=2,
)

lp_modes.set_lp_warning_suppression(False)
print("Running calibrations under suppression (via context):")
res = mmi.calibrate_input_phases_genetic(N=2, M=2, W=10e-6, n_core=2.05, delta_n=0.1, wavelength=1.55e-6, verbose=False)
print("OK")
