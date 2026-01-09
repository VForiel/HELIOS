"""
15_uml_visualization.py

Demonstrates how to generate UML diagrams of the optical pipeline.
Includes two examples:
1. A standard exoplanet detection system (ELT + AO + Coronagraph).
2. A complex interferometric beam combiner with photonic circuits.
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from copy import deepcopy as copy

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import helios
import helios.components.photonics as photonics
import helios.components.photonics.fibers as fibers

def run_demo():
    # --- Example 1: Exoplanet Detection System ---
    print("1. Generating Exoplanet Detection System UML...")
    
    exo_scene = helios.Scene(distance=10*u.pc)
    exo_scene.add(helios.Star(temperature=5700*u.K, magnitude=5, position=(0, 0)))
    exo_scene.add(helios.Planet(temperature=300*u.K, magnitude=22, position=(100*u.mas, 0*u.mas)))

    atmosphere = helios.Atmosphere(rms=200*u.nm, wind_speed=8*u.m/u.s, seed=42)
    elt = helios.TelescopeArray(pupil=helios.Pupil.elt(), size=39*u.m, name="ELT")
    elt.add_position(0, 0)
    ao = helios.AdaptiveOptics(coeffs={(1, 1): 0.15, (2, 0): 0.08})
    coronagraph = helios.Coronagraph(phase_mask='4quadrants')
    bs = helios.BeamSplitter(cutoff=0.5)
    camera1 = helios.Camera(pixels=(512, 512))
    camera2 = helios.Camera(pixels=(256, 256))

    exo_ctx = helios.Context()
    exo_ctx.add_layer(exo_scene)
    exo_ctx.add_layer(atmosphere)
    exo_ctx.add_layer(elt)
    exo_ctx.add_layer(ao)
    exo_ctx.add_layer(coronagraph)
    exo_ctx.add_layer(bs)
    exo_ctx.add_layer([camera1, camera2])

    exo_diagram_img = exo_ctx.plot_uml_diagram(return_type='image', figsize=(18, 8))

    plt.figure(figsize=(18, 8))
    plt.imshow(exo_diagram_img)
    plt.axis('off')
    plt.title('1. Complete Exoplanet Detection System', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    if os.environ.get("HELIOS_SAVE_PLOTS") == "true":
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated/examples'))
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(__file__).replace('.py', '_1.png')
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    # --- Example 2: Interferometric Beam Combiner ---
    print("2. Generating Interferometric Beam Combiner UML...")

    ctx = helios.Context()

    # Scene
    scene = helios.Scene(distance=10*u.pc, name="Target System")
    scene.add(helios.Star(temperature=5700*u.K, magnitude=5))
    scene.add(helios.Planet(mass=1*u.M_jup, position=(1*u.AU, 0*u.AU)))

    # Atmosphere
    atmosphere = helios.Atmosphere(name="Atmosphere")
    
    # Telescopes (4 collectors)
    pupil = helios.Pupil(8*u.m)
    telescopes = helios.TelescopeArray(pupil=pupil, size=8*u.m, name="Interferometer Array")
    telescopes.add_position(-10, -10)
    telescopes.add_position(10, -10)
    telescopes.add_position(10, 10)
    telescopes.add_position(-10, 10)

    # Adaptive Optics
    ao_system = helios.AdaptiveOptics(name="AO System")
    
    # Input coupling
    fiber_in = fibers.FiberIn(modes=1, name="Fiber Injection")
    
    # Phase Shifters
    tops = photonics.TOPS(phase=0.0, name="Phase Shifter")
    
    # Recombination
    mmi_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    mmi = photonics.MMI(matrix=mmi_matrix, name="Beam Combiner (MMI)")
    
    cross_mmi_matrix = np.array([
            [np.exp(1j*np.pi/4), np.exp(-1j*np.pi/4)],
            [np.exp(-1j*np.pi/4), np.exp(1j*np.pi/4)]
        ]) / np.sqrt(2)
    cross_mmi = photonics.MMI(matrix=cross_mmi_matrix, name="Cross MMI")
    
    # Output coupling
    fiber_out = fibers.FiberOut(name="Detector Port")
    
    # Detectors
    cam = helios.Camera(pixels=(1,1), name="Photodiode")
    
    swap1 = photonics.Swap(mapping=[0, 2, 1, 3], name="Router")
    swap2 = photonics.Swap(mapping=[0, 1, 3, 2, 5, 4, 6], name="Router")
    y_splitter = photonics.YSplitter(name="Splitter")

    # Build Pipeline
    ctx.add_layer(scene)
    ctx.add_layer(atmosphere)
    ctx.add_layer(telescopes)
    ctx.add_layer([copy(ao_system) for _ in range(4)])
    ctx.add_layer([copy(fiber_in) for _ in range(4)])
    ctx.add_layer([copy(tops) for _ in range(4)])
    ctx.add_layer([copy(mmi) for _ in range(2)])
    ctx.add_layer(swap1)
    ctx.add_layer([copy(tops) for _ in range(4)])
    ctx.add_layer([copy(mmi) for _ in range(2)])
    ctx.add_layer([None] + [copy(y_splitter) for _ in range(3)])
    ctx.add_layer(swap2)
    ctx.add_layer([None] + [copy(tops) for _ in range(6)])
    ctx.add_layer([None] + [copy(cross_mmi) for _ in range(3)])
    ctx.add_layer([copy(fiber_out) for _ in range(7)])
    ctx.add_layer([copy(cam) for _ in range(7)])

    # Chip container
    chip = photonics.PhotonicChip(inputs=2, lambda0=1.55*u.um, name="Beam Combiner Chip")
    for elem in [fiber_in, tops, mmi, fiber_out, fiber_out]:
        elem.layer = chip

    # Generate and Show Diagram
    fig = ctx.plot_uml_diagram(figsize=(20, 12), layer_spacing=2.5)
    plt.title("2. Interferometric Beam Combiner - UML Diagram", fontsize=16, fontweight='bold', pad=20)
    
    output_file = 'uml_complex_test.png'
    
    if os.environ.get("HELIOS_SAVE_PLOTS") == "true":
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated/examples'))
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(__file__).replace('.py', '_2.png')
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        
        # Also save to local file as originally intended
        plt.savefig(output_file)
        print(f"Complex diagram saved to {output_file}")
    else:
        plt.savefig(output_file)
        print(f"Complex diagram saved to {output_file}")
        plt.show()

if __name__ == "__main__":
    run_demo()
