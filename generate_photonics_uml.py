
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

# Ensure we can import helios from src
sys.path.insert(0, os.path.abspath('./src'))

import helios
import helios.components.photonics as photonics
import helios.components.fibers as fibers

def generate_uml():
    print("Generating Photonics UML Diagram with Telescope Input...")

    # 1. Create Context
    ctx = helios.Context()

    # 2. Define Components
    
    # Scene
    scene = helios.Scene(distance=10*u.pc, name="Target System")
    scene.add(helios.Star(temperature=5700*u.K, magnitude=5))
    scene.add(helios.Planet(mass=1*u.M_jup, position=(1*u.AU, 0*u.AU)))
    
    # Telescopes (2 collectors)
    # We use a TelescopeArray to represent the collection stage
    telescopes = helios.TelescopeArray(name="Interferometer Array")
    pupil = helios.Pupil(8*u.m)
    telescopes.add_collector(pupil=pupil, position=(-10, 0), size=8*u.m)
    telescopes.add_collector(pupil=pupil, position=(10, 0), size=8*u.m)
    
    # Input coupling (2 fibers for 2 telescopes)
    fiber_in_1 = fibers.FiberIn(modes=1, name="Fiber 1")
    fiber_in_2 = fibers.FiberIn(modes=1, name="Fiber 2")
    
    # Phase Shifters (Parallel arms)
    tops_1 = photonics.TOPS(phase=0.0, name="Phase Shifter 1")
    tops_2 = photonics.TOPS(phase=np.pi/2, name="Phase Shifter 2")
    
    # Recombination (2 -> 2)
    mmi_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    mmi = photonics.MMI(matrix=mmi_matrix, name="Beam Combiner (MMI)")
    
    # Output coupling
    fiber_out_1 = fibers.FiberOut(name="Detector Port 1")
    fiber_out_2 = fibers.FiberOut(name="Detector Port 2")
    
    # Create Photonic Chip container to group elements
    # This ensures they are visualized as a single circuit
    chip = photonics.PhotonicChip(inputs=2, lambda0=1.55*u.um, name="Beam Combiner Chip")
    
    # Manually link elements to chip (simulating adding them to chip)
    # In a real workflow, we might use chip.add_element(), but here we construct the pipeline manually
    for elem in [fiber_in_1, fiber_in_2, tops_1, tops_2, mmi, fiber_out_1, fiber_out_2]:
        elem.layer = chip
    
    # Detectors
    cam1 = helios.Camera(pixels=(1,1), name="Photodiode 1")
    cam2 = helios.Camera(pixels=(1,1), name="Photodiode 2")

    # 3. Build Pipeline
    # Layer 1: Scene
    ctx.add_layer(scene)
    
    # Layer 2: Telescopes
    ctx.add_layer(telescopes)
    
    # Layer 3: Coupling into fibers (Parallel)
    ctx.add_layer([fiber_in_1, fiber_in_2])
    
    # Layer 4: Phase Shifters
    ctx.add_layer([tops_1, tops_2])
    
    # Layer 5: MMI Recombiner
    ctx.add_layer(mmi)
    
    # Layer 6: Fiber Outputs
    ctx.add_layer([fiber_out_1, fiber_out_2])
    
    # Layer 7: Detectors
    ctx.add_layer([cam1, cam2])

    # 4. Generate and Show Diagram
    fig = ctx.plot_uml_diagram(figsize=(16, 8), layer_spacing=2.5)
    plt.title("Interferometric Beam Combiner - UML Diagram", fontsize=16, fontweight='bold', pad=20)
    plt.show()
    
    print("Diagram generated successfully.")

if __name__ == "__main__":
    generate_uml()
