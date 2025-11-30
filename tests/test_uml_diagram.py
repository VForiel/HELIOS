"""
Test UML diagram generation.

This test validates the Context.plot_uml_diagram() method for various
optical configurations.
"""
import sys
sys.path.insert(0, '../src')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

import helios
from astropy import units as u


def test_simple_pipeline():
    """Test UML diagram for simple scene → telescope → camera pipeline."""
    # Create simple pipeline
    scene = helios.Scene(distance=10*u.pc)
    scene.add(helios.Star(temperature=5700*u.K, magnitude=5))
    
    telescope = helios.TelescopeArray(name="VLT")
    telescope.add_collector(pupil=helios.Pupil.vlt(), position=(0, 0), size=8*u.m)
    
    camera = helios.Camera(pixels=(512, 512))
    
    # Build context
    ctx = helios.Context()
    ctx.add_layer(scene)
    ctx.add_layer(telescope)
    ctx.add_layer(camera)
    
    # Test return_type='figure' (default)
    fig = ctx.plot_uml_diagram(save_path='test_simple_pipeline.png')
    assert fig is not None
    print("✓ Simple pipeline diagram generated (figure)")
    plt.close(fig)
    
    # Test return_type='image'
    img = ctx.plot_uml_diagram(return_type='image')
    assert img is not None
    assert img.ndim == 3
    assert img.shape[2] == 3  # RGB
    print("✓ Simple pipeline diagram generated (image array)")
    
    # Test return_type='both'
    fig, img = ctx.plot_uml_diagram(return_type='both')
    assert fig is not None
    assert img is not None
    print("✓ Simple pipeline diagram generated (both)")
    plt.close(fig)


def test_beam_splitter_pipeline():
    """Test UML diagram with beam splitter creating parallel paths."""
    # Create pipeline with beam splitter
    scene = helios.Scene(distance=10*u.pc)
    scene.add(helios.Star(temperature=5700*u.K, magnitude=5))
    
    telescope = helios.TelescopeArray(name="Observatory")
    telescope.add_collector(pupil=helios.Pupil.vlt(), position=(0, 0), size=8*u.m)
    
    # Beam splitter
    bs = helios.BeamSplitter(cutoff=0.5)
    
    # Two cameras
    camera1 = helios.Camera(pixels=(512, 512))
    camera2 = helios.Camera(pixels=(256, 256))
    
    # Build context
    ctx = helios.Context()
    ctx.add_layer(scene)
    ctx.add_layer(telescope)
    ctx.add_layer(bs)
    ctx.add_layer([camera1, camera2])  # Parallel paths
    
    # Generate diagram
    fig = ctx.plot_uml_diagram(save_path='test_beam_splitter.png')
    
    # Validate
    assert fig is not None
    print("✓ Beam splitter diagram generated")
    plt.close(fig)


def test_complex_pipeline():
    """Test UML diagram for complex pipeline with atmosphere, AO, coronagraph."""
    # Create complex pipeline
    scene = helios.Scene(distance=10*u.pc)
    scene.add(helios.Star(temperature=5700*u.K, magnitude=5, position=(0, 0)))
    scene.add(helios.Planet(temperature=300*u.K, magnitude=20, position=(100*u.mas, 0*u.mas)))
    
    telescope = helios.TelescopeArray(name="ELT")
    telescope.add_collector(pupil=helios.Pupil.elt(), position=(0, 0), size=39*u.m)
    
    atmosphere = helios.Atmosphere(rms=500*u.nm, seed=42)
    ao = helios.AdaptiveOptics(coeffs={(1, 1): 0.1, (2, 0): 0.05})
    coronagraph = helios.Coronagraph(phase_mask='4quadrants')
    camera = helios.Camera(pixels=(512, 512))
    
    # Build context
    ctx = helios.Context()
    ctx.add_layer(scene)
    ctx.add_layer(atmosphere)
    ctx.add_layer(telescope)
    ctx.add_layer(ao)
    ctx.add_layer(coronagraph)
    ctx.add_layer(camera)
    
    # Generate diagram
    fig = ctx.plot_uml_diagram(figsize=(18, 8), save_path='test_complex_pipeline.png')
    
    # Validate
    assert fig is not None
    print("✓ Complex pipeline diagram generated")
    plt.close(fig)


def test_interferometer_pipeline():
    """Test UML diagram for interferometer configuration."""
    # Create interferometer
    scene = helios.Scene(distance=10*u.pc)
    scene.add(helios.Star(temperature=5700*u.K, magnitude=5))
    
    # Interferometer with 3 telescopes
    interferometer = helios.Interferometer(name="VLTI")
    interferometer.add_collector(pupil=helios.Pupil.vlt(), position=(0*u.m, 0*u.m), size=8*u.m)
    interferometer.add_collector(pupil=helios.Pupil.vlt(), position=(50*u.m, 0*u.m), size=8*u.m)
    interferometer.add_collector(pupil=helios.Pupil.vlt(), position=(25*u.m, 43*u.m), size=8*u.m)
    
    camera = helios.Camera(pixels=(128, 128))
    
    # Build context
    ctx = helios.Context()
    ctx.add_layer(scene)
    ctx.add_layer(interferometer)
    ctx.add_layer(camera)
    
    # Generate diagram
    fig = ctx.plot_uml_diagram(save_path='test_interferometer.png')
    
    # Validate
    assert fig is not None
    print("✓ Interferometer diagram generated")
    plt.close(fig)


if __name__ == "__main__":
    print("Testing UML diagram generation...")
    test_simple_pipeline()
    test_beam_splitter_pipeline()
    test_complex_pipeline()
    test_interferometer_pipeline()
    print("\n✓ All UML diagram tests passed!")
