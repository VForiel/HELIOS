"""
Demonstration of full description feature.

Shows how description(full=True) provides comprehensive parameter information
for complete observation setup documentation and debugging.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import helios
from astropy import units as u


def demo_exoplanet_observation():
    """Demonstrate full description for exoplanet direct imaging setup."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "Exoplanet Direct Imaging Setup" + " " * 23 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")
    
    # Create context with observation parameters
    ctx = helios.Context(
        date="2025-06-15 22:30:00",
        declination=-40.5*u.deg
    )
    
    # Build scene: star + hot Jupiter
    scene = helios.Scene(distance=12.5*u.pc, name="HD 189733")
    scene.add(helios.Star(
        temperature=5040*u.K,
        magnitude=7.67,
        mass=0.82*u.M_sun,
        name="HD 189733 A"
    ))
    scene.add(helios.Planet(
        mass=1.13*u.M_jup,
        radius=1.14*u.R_jup,
        temperature=1200*u.K,
        albedo=0.15,
        position=(0.31*u.arcsec, 0*u.arcsec),
        name="HD 189733 b (hot Jupiter)"
    ))
    
    # Build telescope: VLT UT3 (Melipal)
    telescope = helios.TelescopeArray(
        name="VLT UT3 (Melipal)",
        latitude=-24.6275*u.deg,
        longitude=-70.4044*u.deg,
        altitude=2635*u.m
    )
    pupil = helios.Pupil.vlt()
    telescope.add_collector(
        pupil=pupil,
        position=(0, 0),
        size=8.2*u.m,
        name="UT3"
    )
    
    # Build detector: SPHERE/IRDIS
    camera = helios.Camera(
        pixels=(2048, 2048),
        dark_current=0.006*u.electron/u.s,
        read_noise=1.5*u.electron,
        integration_time=60*u.s,
        quantum_efficiency=0.85,
        thermal_background_temp=273*u.K,
        name="SPHERE/IRDIS"
    )
    
    # Assemble observation
    ctx.add_layer(scene)
    ctx.add_layer(telescope)
    ctx.add_layer(camera)
    
    # Show basic description
    print("=" * 70)
    print("BASIC DESCRIPTION (summary)")
    print("=" * 70)
    print(ctx.description(full=False))
    
    # Show full description
    print("\n")
    print("=" * 70)
    print("FULL DESCRIPTION (all parameters)")
    print("=" * 70)
    print(ctx.description(full=True))


def demo_interferometer_observation():
    """Demonstrate full description for interferometric observation."""
    print("\n\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "VLTI Interferometric Imaging" + " " * 22 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")
    
    # Create context
    ctx = helios.Context(
        date="2025-08-20 03:00:00",
        declination=-29.5*u.deg
    )
    
    # Build scene: binary star system
    scene = helios.Scene(distance=140*u.pc, name="α Centauri")
    scene.add(helios.Star(
        temperature=5790*u.K,
        magnitude=0.0,
        mass=1.1*u.M_sun,
        position=(-3.5*u.mas, 0*u.mas),
        name="α Cen A"
    ))
    scene.add(helios.Star(
        temperature=5260*u.K,
        magnitude=1.33,
        mass=0.907*u.M_sun,
        position=(3.5*u.mas, 0*u.mas),
        name="α Cen B"
    ))
    
    # Build interferometer: VLTI with 4 UTs
    vlti = helios.TelescopeArray.vlti(uts=True)  # Use preset configuration
    
    # Build beam combiner and detector
    camera = helios.Camera(
        pixels=(512, 512),
        dark_current=0.001*u.electron/u.s,
        read_noise=0.8*u.electron,
        integration_time=300*u.s,
        quantum_efficiency=0.75,
        name="GRAVITY"
    )
    
    # Assemble observation
    ctx.add_layer(scene)
    ctx.add_layer(vlti)
    ctx.add_layer(camera)
    
    # Show full description
    print("=" * 70)
    print("FULL DESCRIPTION")
    print("=" * 70)
    print(ctx.description(full=True))


def demo_dual_channel_observation():
    """Demonstrate full description with beam splitting."""
    print("\n\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 17 + "Dual-Channel Coronagraphic Imaging" + " " * 17 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")
    
    # Create context
    ctx = helios.Context()
    
    # Build scene
    scene = helios.Scene(distance=50*u.pc, name="Debris Disk System")
    scene.add(helios.Star(
        temperature=6200*u.K,
        magnitude=8.2,
        mass=1.4*u.M_sun,
        name="Host Star"
    ))
    scene.add(helios.Planet(
        mass=0.8*u.M_jup,
        temperature=400*u.K,
        albedo=0.25,
        position=(0.5*u.arcsec, 0.2*u.arcsec),
        name="Jovian Planet"
    ))
    
    # Build telescope
    telescope = helios.TelescopeArray(name="Gemini North")
    pupil = helios.Pupil(diameter=8.1*u.m)
    pupil.add_disk(radius=4.05*u.m)
    telescope.add_collector(pupil=pupil, position=(0, 0), size=8.1*u.m)
    
    # Beam splitter for dual channels
    beam_splitter = helios.BeamSplitter(cutoff=0.6, name="Dichroic (600nm)")
    
    # Two cameras for different bands
    camera_blue = helios.Camera(
        pixels=(1024, 1024),
        dark_current=0.005*u.electron/u.s,
        read_noise=2.5*u.electron,
        integration_time=120*u.s,
        quantum_efficiency=0.92,
        name="Blue Channel (400-600nm)"
    )
    camera_red = helios.Camera(
        pixels=(1024, 1024),
        dark_current=0.008*u.electron/u.s,
        read_noise=2.0*u.electron,
        integration_time=180*u.s,
        quantum_efficiency=0.88,
        name="Red Channel (600-900nm)"
    )
    
    # Assemble observation
    ctx.add_layer(scene)
    ctx.add_layer(telescope)
    ctx.add_layer(beam_splitter)
    ctx.add_layer([camera_blue, camera_red])
    
    # Show full description
    print("=" * 70)
    print("FULL DESCRIPTION")
    print("=" * 70)
    print(ctx.description(full=True))


def demo_comparison():
    """Show side-by-side comparison of full=False vs full=True."""
    print("\n\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "Comparison: Basic vs Full" + " " * 23 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")
    
    # Simple setup
    ctx = helios.Context()
    scene = helios.Scene(distance=10*u.pc)
    scene.add(helios.Star(temperature=5800*u.K, magnitude=5.0))
    
    telescope = helios.TelescopeArray(name="Example Telescope")
    pupil = helios.Pupil.vlt()
    telescope.add_collector(pupil=pupil, position=(0, 0), size=8*u.m)
    
    camera = helios.Camera(
        pixels=(512, 512),
        integration_time=30*u.s,
        name="Example Camera"
    )
    
    ctx.add_layer(scene)
    ctx.add_layer(telescope)
    ctx.add_layer(camera)
    
    print("┌─ full=False (compact summary) " + "─" * 37 + "┐")
    print(ctx.description(full=False))
    print("└" + "─" * 68 + "┘")
    
    print("\n")
    print("┌─ full=True (detailed parameters) " + "─" * 33 + "┐")
    print(ctx.description(full=True))
    print("└" + "─" * 68 + "┘")


if __name__ == "__main__":
    demo_exoplanet_observation()
    demo_interferometer_observation()
    demo_dual_channel_observation()
    demo_comparison()
    
    print("\n\n")
    print("=" * 70)
    print("Demonstrations completed!")
    print("=" * 70)
    print("\nThe full=True parameter provides comprehensive documentation of:")
    print("  • Context parameters (date, declination, etc.)")
    print("  • Scene configuration (distance, number of objects)")
    print("  • Celestial body properties (temperature, mass, magnitude, etc.)")
    print("  • Telescope configuration (collectors, baselines, location)")
    print("  • Detector specifications (pixels, noise, integration time, QE)")
    print("  • Optical components (beam splitter ratios, etc.)")
    print("\nUse description(full=True) for:")
    print("  ✓ Complete observation setup documentation")
    print("  ✓ Debugging simulation configurations")
    print("  ✓ Generating observation logs")
    print("  ✓ Verifying parameter values")
    print("=" * 70)
