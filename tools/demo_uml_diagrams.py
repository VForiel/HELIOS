"""
Demonstration of UML diagram generation for HELIOS optical systems.

This script creates various optical configurations and generates their
UML diagrams for visual inspection.
"""
import sys
sys.path.insert(0, '../src')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import helios
from astropy import units as u


def demo_exoplanet_detection_system():
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
    telescope = helios.TelescopeArray(name="ELT")
    telescope.add_collector(pupil=helios.Pupil.elt(), position=(0, 0), size=39*u.m)
    
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
    fig = ctx.plot_uml_diagram(figsize=(18, 8), save_path='demo_exoplanet_system.png')
    plt.savefig('demo_exoplanet_system_hires.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: demo_exoplanet_system.png")
    plt.close(fig)


def demo_dual_channel_spectrograph():
    """
    Dual-channel spectrograph with beam splitter:
    - Single telescope
    - Beam splitter
    - Two cameras (e.g., red and blue channels)
    """
    print("Creating dual-channel spectrograph...")
    
    scene = helios.Scene(distance=50*u.pc)
    scene.add(helios.Star(temperature=4500*u.K, magnitude=8))
    
    telescope = helios.TelescopeArray(name="VLT-UT4")
    telescope.add_collector(pupil=helios.Pupil.vlt(), position=(0, 0), size=8.2*u.m)
    
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
    
    fig = ctx.plot_uml_diagram(figsize=(14, 10), save_path='demo_dual_spectrograph.png')
    plt.savefig('demo_dual_spectrograph_hires.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: demo_dual_spectrograph.png")
    plt.close(fig)


def demo_interferometer():
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
    interferometer = helios.Interferometer(name="VLTI")
    interferometer.add_collector(pupil=helios.Pupil.vlt(), position=(0*u.m, 0*u.m), size=8.2*u.m)
    interferometer.add_collector(pupil=helios.Pupil.vlt(), position=(60*u.m, 0*u.m), size=8.2*u.m)
    interferometer.add_collector(pupil=helios.Pupil.vlt(), position=(30*u.m, 52*u.m), size=8.2*u.m)
    
    camera = helios.Camera(pixels=(256, 256))
    
    ctx = helios.Context()
    ctx.add_layer(scene)
    ctx.add_layer(interferometer)
    ctx.add_layer(camera)
    
    fig = ctx.plot_uml_diagram(figsize=(12, 8), save_path='demo_interferometer.png')
    plt.savefig('demo_interferometer_hires.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: demo_interferometer.png")
    plt.close(fig)


def demo_fiber_fed_spectrograph():
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
    
    telescope = helios.TelescopeArray(name="Gemini-South")
    telescope.add_collector(pupil=helios.Pupil(), position=(0, 0), size=8*u.m)
    
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
    
    fig = ctx.plot_uml_diagram(figsize=(16, 8), save_path='demo_fiber_spectrograph.png')
    plt.savefig('demo_fiber_spectrograph_hires.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: demo_fiber_spectrograph.png")
    plt.close(fig)


if __name__ == "__main__":
    print("HELIOS UML Diagram Demonstration")
    print("=" * 50)
    print()
    
    demo_exoplanet_detection_system()
    demo_dual_channel_spectrograph()
    demo_interferometer()
    demo_fiber_fed_spectrograph()
    
    print()
    print("=" * 50)
    print("✓ All demonstration diagrams generated!")
    print()
    print("Generated files:")
    print("  - demo_exoplanet_system.png")
    print("  - demo_dual_spectrograph.png")
    print("  - demo_interferometer.png")
    print("  - demo_fiber_spectrograph.png")
    print()
    print("High-resolution versions (300 DPI) also saved.")
