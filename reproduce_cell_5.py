
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

# 5. Photonic Integrated Circuit Demo
print("Setting up Photonic Integrated Circuit...")

import helios.components.photonics as photonics
import helios.components.fibers as fibers
from helios.core.simulation import Wavefront
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np

# Create a Photonic Chip
chip = photonics.PhotonicChip(inputs=1, lambda0=1.55*u.um)

# Define components
fiber_in = fibers.FiberIn(modes=1)
splitter = photonics.YSplitter()
tops_top = photonics.TOPS(phase=0.0) # Top arm
tops_bot = photonics.TOPS(phase=np.pi/2) # Bottom arm
# 2x2 MMI for recombination
mmi_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
mmi = photonics.MMI(matrix=mmi_matrix)
fiber_out_1 = fibers.FiberOut()
fiber_out_2 = fibers.FiberOut()

# Input Wavefront
wf_in = Wavefront(wavelength=1.55*u.um, size=128)
# Gaussian beam
x = np.linspace(-5, 5, 128)
X, Y = np.meshgrid(x, x)
R = np.sqrt(X**2 + Y**2)
# FIX: Explicit cast to complex128 to allow phase modulation
wf_in.field = np.exp(-R**2).astype(np.complex128)

# 1. Fiber Coupling
wf_coupled = fiber_in.process(wf_in, None)

# 2. Splitter
wf_split = splitter.process(wf_coupled, None) # Returns [top, bot]

# 3. Phase Shifters
wf_top = tops_top.process(wf_split[0], None)
wf_bot = tops_bot.process(wf_split[1], None)

# 4. Recombination (MMI)
wf_out_mmi = mmi.process([wf_top, wf_bot], None)

# 5. Fiber Output
output_1 = fiber_out_1.process(wf_out_mmi[0], None)
output_2 = fiber_out_2.process(wf_out_mmi[1], None)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(np.abs(wf_in.field)**2, cmap='inferno')
axes[0].set_title("Input Field Intensity")
axes[1].imshow(np.abs(output_1.field)**2, cmap='inferno')
axes[1].set_title("Output Port 1")
axes[2].imshow(np.abs(output_2.field)**2, cmap='inferno')
axes[2].set_title("Output Port 2")
plt.show()
print("Cell 5 execution finished successfully")
