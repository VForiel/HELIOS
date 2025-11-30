"""
Demonstration of parent references and description methods.

This script showcases the new architecture where:
- Layers have 'context' references to their parent Context
- Elements have 'layer' and 'context' references
- All classes have description() methods for hierarchical text output
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import helios
from astropy import units as u


def demo_simple_pipeline():
    """Demonstrate description of a simple observation pipeline."""
    print("=" * 70)
    print("DEMO 1: Simple Observation Pipeline")
    print("=" * 70)
    
    # Create context
    ctx = helios.Context()
    
    # Build scene
    scene = helios.Scene(distance=10*u.pc, name="HD 189733")
    scene.add(helios.Star(temperature=5700*u.K, magnitude=7.67))
    scene.add(helios.Planet(
        temperature=1200*u.K,
        radius=1.2*u.R_jup,
        angular_position=(0.1*u.arcsec, 0),
        name="HD 189733 b"
    ))
    
    # Build telescope
    telescope = helios.TelescopeArray(name="VLT")
    pupil = helios.Pupil.vlt()
    telescope.add_collector(pupil=pupil, position=(0, 0), size=8.2*u.m)
    
    # Build camera
    camera = helios.Camera(pixels=(512, 512), name="SPHERE")
    
    # Assemble pipeline
    ctx.add_layer(scene)
    ctx.add_layer(telescope)
    ctx.add_layer(camera)
    
    # Display complete structure
    print(ctx.description())
    print()


def demo_interferometer():
    """Demonstrate description of an interferometer setup."""
    print("=" * 70)
    print("DEMO 2: Interferometer Array")
    print("=" * 70)
    
    # Create context
    ctx = helios.Context()
    
    # Build scene
    scene = helios.Scene(distance=150*u.pc, name="Binary System")
    scene.add(helios.Star(temperature=6000*u.K, magnitude=4, name="Star A"))
    scene.add(helios.Star(
        temperature=4000*u.K,
        magnitude=5,
        angular_position=(0.05*u.arcsec, 0),
        name="Star B"
    ))
    
    # Build interferometer
    interferometer = helios.Interferometer(name="VLTI")
    pupil = helios.Pupil.vlt()
    
    # Add multiple collectors
    interferometer.add_collector(pupil=pupil, position=(0, 0), size=8.2*u.m)
    interferometer.add_collector(pupil=pupil, position=(47*u.m, 0), size=8.2*u.m)
    interferometer.add_collector(pupil=pupil, position=(24*u.m, 48*u.m), size=8.2*u.m)
    interferometer.add_collector(pupil=pupil, position=(71*u.m, 48*u.m), size=8.2*u.m)
    
    # Build camera
    camera = helios.Camera(pixels=(256, 256), name="PIONIER")
    
    # Assemble pipeline
    ctx.add_layer(scene)
    ctx.add_layer(interferometer)
    ctx.add_layer(camera)
    
    # Display complete structure
    print(ctx.description())
    print()


def demo_beam_splitting():
    """Demonstrate description with beam splitting."""
    print("=" * 70)
    print("DEMO 3: Beam Splitting (Dual-Channel Observation)")
    print("=" * 70)
    
    # Create context
    ctx = helios.Context()
    
    # Build scene
    scene = helios.Scene(distance=50*u.pc, name="Exoplanet System")
    scene.add(helios.Star(temperature=5200*u.K, magnitude=8))
    scene.add(helios.Planet(
        temperature=800*u.K,
        radius=0.8*u.R_jup,
        angular_position=(0.2*u.arcsec, 0.1*u.arcsec)
    ))
    
    # Build telescope
    telescope = helios.TelescopeArray(name="Subaru")
    pupil = helios.Pupil(diameter=8.2*u.m)
    pupil.add_disk(radius=4.1*u.m)
    telescope.add_collector(pupil=pupil, position=(0, 0), size=8.2*u.m)
    
    # Beam splitter
    beam_splitter = helios.BeamSplitter()
    
    # Two cameras for different wavelength bands
    camera_visible = helios.Camera(pixels=(512, 512), name="Visible Channel")
    camera_nir = helios.Camera(pixels=(512, 512), name="Near-IR Channel")
    
    # Assemble pipeline
    ctx.add_layer(scene)
    ctx.add_layer(telescope)
    ctx.add_layer(beam_splitter)
    ctx.add_layer([camera_visible, camera_nir])  # Parallel layers
    
    # Display complete structure
    print(ctx.description())
    print()


def demo_parent_references():
    """Demonstrate parent reference navigation."""
    print("=" * 70)
    print("DEMO 4: Parent Reference Navigation")
    print("=" * 70)
    
    # Create context
    ctx = helios.Context()
    
    # Build telescope
    telescope = helios.TelescopeArray(name="Observatory")
    pupil = helios.Pupil.vlt()
    telescope.add_collector(pupil=pupil, position=(0, 0), size=8*u.m)
    telescope.add_collector(pupil=pupil, position=(100, 0)*u.m, size=8*u.m)
    
    # Add to context
    ctx.add_layer(telescope)
    
    # Navigate references
    print("Reference chain demonstration:")
    print(f"  Context: {ctx}")
    print(f"  Layer: {telescope} (name: '{telescope.name}')")
    print(f"  Number of elements: {len(telescope.elements)}")
    
    # Access first collector
    collector = telescope.elements[0]
    print(f"\nFirst collector:")
    print(f"  Element: {collector}")
    print(f"  Element name: '{collector.name}'")
    print(f"  Element.layer is telescope: {collector.layer is telescope}")
    print(f"  Element.context is ctx: {collector.context is ctx}")
    print(f"  Element.context == Element.layer.context: {collector.context == collector.layer.context}")
    
    # Demonstrate shortcut utility
    print(f"\nShortcut utility:")
    print(f"  collector.context (shortcut): {collector.context}")
    print(f"  collector.layer.context (full path): {collector.layer.context}")
    print(f"  Both reference the same object: {collector.context is collector.layer.context}")
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "HELIOS Parent References & Description Demo" + " " * 14 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")
    
    demo_simple_pipeline()
    demo_interferometer()
    demo_beam_splitting()
    demo_parent_references()
    
    print("=" * 70)
    print("All demonstrations completed successfully!")
    print("=" * 70)
