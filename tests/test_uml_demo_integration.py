"""
Demonstration/Test of UML diagram generation for HELIOS optical systems.

This script creates various optical configurations and generates their
UML diagrams for visual inspection.
"""
import pytest
import matplotlib
import matplotlib.pyplot as plt
import helios
from astropy import units as u

matplotlib.use('Agg')  # Non-interactive backend

def test_exoplanet_detection_system():
    """
    Complete exoplanet detection system with:
    - Turbulent atmosphere
    - Large telescope (ELT)
    - Adaptive optics correction
    - Coronagraph for contrast
    - High-resolution camera
    """
    print("Creating exoplanet detection system...")
    
    # Scene: Sun-like star with exoplanet
    scene = helios.Scene(distance=10*u.pc)
    scene.add(helios.Star(temperature=5700*u.K, magnitude=5, position=(0, 0)))
    scene.add(helios.Planet(temperature=300*u.K, magnitude=22, position=(100*u.mas, 0*u.mas)))
    
    # Atmospheric turbulence (moderate seeing)
    atmosphere = helios.Atmosphere(rms=200*u.nm, wind_speed=8*u.m/u.s, seed=42)
    
    # ELT with segmented primary
    # Updated API: Pass pupil, size, and positions directly
    telescope = helios.TelescopeArray(
        name="ELT",
        pupil=helios.Pupil.elt(),
        size=39*u.m,
        positions=[(0, 0)]
    )
    
    # Adaptive optics (correcting tip/tilt and defocus)
    ao = helios.AdaptiveOptics(coeffs={(1, 1): 0.15, (1, -1): 0.12, (2, 0): 0.08})
    
    # Four-quadrant phase mask coronagraph
    coronagraph = helios.Coronagraph(phase_mask='4quadrants')
    
    # Science camera
    camera = helios.Camera(pixels=(1024, 1024))
    
    # Build pipeline
    ctx = helios.Context()
    ctx.add_layer(scene)
    ctx.add_layer(atmosphere)
    ctx.add_layer(telescope)
    ctx.add_layer(ao)
    ctx.add_layer(coronagraph)
    ctx.add_layer(camera)
    
    # Generate diagram
    fig = ctx.plot_uml_diagram(figsize=(18, 8), save_path='tests/generated/demo_exoplanet_system.png')
    plt.close(fig)

def test_dual_channel_spectrograph():
    """
    Dual-channel spectrograph with beam splitter:
    - Single telescope
    - Beam splitter
    - Two cameras (e.g., red and blue channels)
    """
    print("Creating dual-channel spectrograph...")
    
    scene = helios.Scene(distance=50*u.pc)
    scene.add(helios.Star(temperature=4500*u.K, magnitude=8))
    
    telescope = helios.TelescopeArray(
        name="VLT-UT4",
        pupil=helios.Pupil.vlt(),
        size=8.2*u.m,
        positions=[(0, 0)]
    )
    
    # Dichroic beam splitter
    beam_splitter = helios.BeamSplitter(cutoff=0.5)
    
    # Red and blue channel cameras
    red_camera = helios.Camera(pixels=(2048, 2048))
    blue_camera = helios.Camera(pixels=(2048, 2048))
    
    ctx = helios.Context()
    ctx.add_layer(scene)
    ctx.add_layer(telescope)
    ctx.add_layer(beam_splitter)
    ctx.add_layer([red_camera, blue_camera])
    
    fig = ctx.plot_uml_diagram(figsize=(14, 10), save_path='tests/generated/demo_dual_spectrograph.png')
    plt.close(fig)

def test_interferometer():
    """
    Three-telescope interferometer (VLTI configuration):
    - Scene with binary star
    - 3-telescope interferometer
    - Fringe detector
    """
    print("Creating interferometer system...")
    
    scene = helios.Scene(distance=100*u.pc)
    scene.add(helios.Star(temperature=6000*u.K, magnitude=6, position=(0, 0)))
    scene.add(helios.Star(temperature=5500*u.K, magnitude=7, position=(50*u.mas, 30*u.mas)))
    
    # VLTI-style configuration
    # 3 positions
    positions = [
        (0.0, 0.0),
        (60.0, 0.0),
        (30.0, 52.0)
    ]
    interferometer = helios.TelescopeArray(
        name="VLTI",
        pupil=helios.Pupil.vlt(),
        size=8.2*u.m,
        positions=positions
    )
    
    camera = helios.Camera(pixels=(256, 256))
    
    ctx = helios.Context()
    ctx.add_layer(scene)
    ctx.add_layer(interferometer)
    ctx.add_layer(camera)
    
    fig = ctx.plot_uml_diagram(figsize=(12, 8), save_path='tests/generated/demo_interferometer.png')
    plt.close(fig)

def test_fiber_fed_spectrograph():
    """
    Fiber-fed spectrograph with photonic processing:
    - Telescope
    - Fiber input (single-mode coupling)
    - Photonic chip (waveguide processing)
    - Fiber output
    - Detector
    """
    print("Creating fiber-fed spectrograph...")
    
    scene = helios.Scene(distance=25*u.pc)
    scene.add(helios.Star(temperature=5200*u.K, magnitude=7))
    
    telescope = helios.TelescopeArray(
        name="Gemini-South",
        pupil=helios.Pupil(),
        size=8*u.m,
        positions=[(0, 0)]
    )
    
    # Fiber input coupling
    fiber_in = helios.FiberIn(mode_field_diameter=10*u.um)
    
    # Photonic chip for dispersion/filtering
    photonic_chip = helios.PhotonicChip(inputs=2, lambda0=1.55*u.um)
    
    # Fiber output
    fiber_out = helios.FiberOut(mode_field_diameter=10*u.um)
    
    # Detector
    camera = helios.Camera(pixels=(512, 512))
    
    ctx = helios.Context()
    ctx.add_layer(scene)
    ctx.add_layer(telescope)
    ctx.add_layer(fiber_in)
    ctx.add_layer(photonic_chip)
    ctx.add_layer(fiber_out)
    ctx.add_layer(camera)
    
    fig = ctx.plot_uml_diagram(figsize=(16, 8), save_path='tests/generated/demo_fiber_spectrograph.png')
    plt.close(fig)

if __name__ == "__main__":
    # Allow running as script too
    import os
    os.makedirs('tests/generated', exist_ok=True)
    test_exoplanet_detection_system()
    test_dual_channel_spectrograph()
    test_interferometer()
    test_fiber_fed_spectrograph()
    print("All tests passed!")
