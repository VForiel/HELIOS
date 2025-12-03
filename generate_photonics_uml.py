
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from copy import deepcopy as copy

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

    # Atmosphere
    atmosphere = helios.Atmosphere(name="Atmosphere")
    
    # Telescopes (2 collectors)
    # We use a TelescopeArray to represent the collection stage
    telescopes = helios.TelescopeArray(name="Interferometer Array")
    pupil = helios.Pupil(8*u.m)
    telescopes.add_collector(pupil=pupil, position=(-10, -10), size=8*u.m)
    telescopes.add_collector(pupil=pupil, position=(10, -10), size=8*u.m)
    telescopes.add_collector(pupil=pupil, position=(10, 10), size=8*u.m)
    telescopes.add_collector(pupil=pupil, position=(-10, 10), size=8*u.m)

    # Adaptive Optics (optional)
    ao_system = helios.AdaptiveOptics(name="AO System")
    
    # Input coupling (2 fibers for 2 telescopes)
    fiber_in = fibers.FiberIn(modes=1, name="Fiber Injection")
    
    # Phase Shifters (Parallel arms)
    tops = photonics.TOPS(phase=0.0, name="Phase Shifter")
    
    # Recombination (2 -> 2)
    mmi_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    mmi = photonics.MMI(matrix=mmi_matrix, name="Beam Combiner (MMI)")
    
    # Output coupling
    fiber_out = fibers.FiberOut(name="Detector Port")
    
    # Detectors
    cam = helios.Camera(pixels=(1,1), name="Photodiode")

    # 3. Build Pipeline
    # Layer 1: Scene
    ctx.add_layer(scene)

    ctx.add_layer(atmosphere)
    
    # Layer 2: Telescopes
    ctx.add_layer(telescopes)

    ctx.add_layer([copy(ao_system) for _ in range(4)])

    # Layer 3: Coupling into fibers (Parallel)
    ctx.add_layer([copy(fiber_in) for _ in range(4)])
    
    # Layer 4: Phase Shifters
    ctx.add_layer([copy(tops) for _ in range(4)])
    
    # Layer 5: MMI Recombiner
    ctx.add_layer([copy(mmi) for _ in range(2)])

    # Layer 5: MMI Recombiner
    ctx.add_layer([copy(mmi) for _ in range(2)])
    
    # Layer 6: Fiber Outputs
    ctx.add_layer([copy(fiber_out) for _ in range(4)])
    
    # Layer 7: Detectors
    ctx.add_layer([copy(cam) for _ in range(4)])

    # Create Photonic Chip container to group elements
    # This ensures they are visualized as a single circuit
    chip = photonics.PhotonicChip(inputs=2, lambda0=1.55*u.um, name="Beam Combiner Chip")
    
    # Manually link elements to chip (simulating adding them to chip)
    # In a real workflow, we might use chip.add_element(), but here we construct the pipeline manually
    for elem in [fiber_in, tops, mmi, fiber_out, fiber_out]:
        elem.layer = chip

    # 4. Generate and Show Diagram
    fig = ctx.plot_uml_diagram(figsize=(20, 12), layer_spacing=2.5)
    plt.title("Interferometric Beam Combiner - UML Diagram", fontsize=16, fontweight='bold', pad=20)
    # plt.show()
    plt.savefig('uml_test.png')
    print("Diagram saved to uml_test.png")
    
    print("Diagram generated successfully.")

if __name__ == "__main__":
    generate_uml()
