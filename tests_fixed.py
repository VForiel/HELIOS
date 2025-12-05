
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import helios

def run_simulation():
    print("--- Starting Fixed Simulation ---")

    # 1. Setup Scene (Fixed API)
    star = helios.Star(magnitude=5, temperature=5778*u.K)
    planet = helios.Planet(magnitude=5, separation=0.1*u.arcsec, position_angle=0*u.deg)
    
    scene = helios.Scene(distance=10*u.pc)
    scene.add(star)
    scene.add(planet)
    
    print(f"Scene created. Star mag: {star.magnitude}")

    # 2. Setup Telescope (Fixed API)
    pupil1 = helios.Pupil(diameter=1*u.m)
    pupil1.add_spiders(width=0.05*u.m, arms=4)
    pupil1.add_central_obscuration(diameter=0.4*u.m)
    
    pupil2 = helios.Pupil(diameter=1*u.m)
    pupil2.add_spiders(width=0.05*u.m, arms=4)
    pupil2.add_central_obscuration(diameter=0.4*u.m)
    
    collectors = [
        helios.Collector(pupil=pupil1, position=[-5, 0]*u.m),
        helios.Collector(pupil=pupil2, position=[5, 0]*u.m)
    ]
    
    interferometer = helios.TelescopeArray()
    for collector in collectors:
        interferometer.add_element(collector)
    
    print(f"Interferometer created with {len(collectors)} collectors.")

    # 3. Create Context and Get Input Wavefront
    ctx = helios.Context()
    ctx.add_layer(scene)
    ctx.add_layer(interferometer)
    
    wavelength = 1e-6 * u.m
    
    print(f"Generating input wavefront for wavelength: {wavelength}")
    wf_array = ctx.get_input_wavefront(wavelength=wavelength, size=1024, angular_samples=1)
    
    # 4. Propagate with Padding (Fix for Undersampling)
    print("Propagating with padding=4...")
    # This requires the fix in helios.core.simulation.WavefrontArray.propagate
    wf_array.propagate(distance=10*u.m, padding=4)
    
    # 5. Plot
    print("Plotting results...")
    wf_array.plot(title="Fixed Simulation (Padding=4)", log_scale=True)
    plt.savefig("tests_fixed_output.png")
    print("Saved tests_fixed_output.png")

if __name__ == "__main__":
    run_simulation()
