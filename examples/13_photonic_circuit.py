"""
10_photonic_circuit.py

Demonstrates a simple photonic integrated circuit simulation.
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import helios
import helios.components.photonics as photonics
import helios.components.photonics.fibers as fibers
from helios.core.wavefront import Wavefront

def run_demo():
    # Components
    fiber_in = fibers.FiberIn(modes=1)
    splitter = photonics.YSplitter()
    tops_top = photonics.TOPS(phase=0.0)
    tops_bot = photonics.TOPS(phase=np.pi/2)
    mmi_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    mmi = photonics.MMI(matrix=mmi_matrix)
    fiber_out_1 = fibers.FiberOut()
    fiber_out_2 = fibers.FiberOut()

    # Input Wavefront
    wf_in = Wavefront(wavelength=1.55*u.um, npix=128)
    x = np.linspace(-5, 5, 128)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    wf_in.field = np.exp(-R**2).astype(np.complex128)

    # Process
    print("Processing through circuit...")
    wf_coupled = fiber_in.process(wf_in)
    wf_split = splitter.process(wf_coupled)
    wf_top = tops_top.process(wf_split[0])
    wf_bot = tops_bot.process(wf_split[1])
    wf_out_mmi = mmi.process([wf_top, wf_bot])
    output_1 = fiber_out_1.process(wf_out_mmi[0])
    output_2 = fiber_out_2.process(wf_out_mmi[1])

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Squeeze to 2D if needed (wavefront has shape (samples, h, w))
    wf_in_2d = wf_in.value[0] if wf_in.value.ndim == 3 else wf_in.value
    out1_2d = output_1.value[0] if output_1.value.ndim == 3 else output_1.value
    out2_2d = output_2.value[0] if output_2.value.ndim == 3 else output_2.value
    
    axes[0].imshow(np.abs(wf_in_2d)**2, cmap='inferno')
    axes[0].set_title("Input Field Intensity")
    axes[1].imshow(np.abs(out1_2d)**2, cmap='inferno')
    axes[1].set_title("Output Port 1")
    axes[2].imshow(np.abs(out2_2d)**2, cmap='inferno')
    axes[2].set_title("Output Port 2")
    
    if os.environ.get("HELIOS_SAVE_PLOTS") == "true":
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated/examples'))
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(__file__).replace('.py', '.png')
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    run_demo()
