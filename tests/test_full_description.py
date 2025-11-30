"""
Test script for the full description feature.

Validates that description(full=True) provides detailed parameter information
for all components in the simulation pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import helios
from astropy import units as u


def test_context_basic_description():
    """Test basic (non-full) description."""
    print("\n=== Test 1: Basic Context Description ===")
    
    ctx = helios.Context()
    scene = helios.Scene(distance=10*u.pc, name="Test System")
    telescope = helios.TelescopeArray(name="VLT")
    camera = helios.Camera(pixels=(512, 512), name="Detector")
    
    ctx.add_layer(scene)
    ctx.add_layer(telescope)
    ctx.add_layer(camera)
    
    desc = ctx.description(full=False)
    print(desc)
    
    assert "HELIOS Simulation Context" in desc
    assert "Test System" in desc
    assert "VLT" in desc
    assert "Detector" in desc
    
    print("✓ Basic description works")


def test_context_full_description():
    """Test full description with all details."""
    print("\n=== Test 2: Full Context Description ===")
    
    ctx = helios.Context(date="2025-01-01", declination=10*u.deg)
    
    # Create scene with celestial objects
    scene = helios.Scene(distance=10*u.pc, name="Target System")
    scene.add(helios.Star(temperature=5700*u.K, magnitude=5.0, name="Host Star"))
    scene.add(helios.Planet(
        mass=1*u.M_jup,
        temperature=800*u.K,
        radius=1.2*u.R_jup,
        albedo=0.3,
        position=(0.1*u.arcsec, 0),
        name="Planet b"
    ))
    
    # Create telescope array
    telescope = helios.TelescopeArray(
        name="VLT",
        latitude=-24.6*u.deg,
        longitude=-70.4*u.deg,
        altitude=2635*u.m
    )
    pupil = helios.Pupil.vlt()
    telescope.add_collector(pupil=pupil, position=(0, 0), size=8.2*u.m)
    
    # Create camera
    camera = helios.Camera(
        pixels=(1024, 1024),
        dark_current=0.01*u.electron/u.s,
        read_noise=3*u.electron,
        integration_time=10*u.s,
        quantum_efficiency=0.9,
        name="SPHERE"
    )
    
    # Build context
    ctx.add_layer(scene)
    ctx.add_layer(telescope)
    ctx.add_layer(camera)
    
    # Get full description
    desc_full = ctx.description(full=True)
    print(desc_full)
    
    # Verify context parameters are included
    assert "date: 2025-01-01" in desc_full
    assert "declination" in desc_full
    
    # Verify scene details
    assert "distance: 10.0 pc" in desc_full or "10.0 pc" in desc_full
    assert "num_objects: 2" in desc_full
    
    # Verify star details
    assert "5700" in desc_full  # Temperature
    assert "5.00" in desc_full  # Magnitude
    
    # Verify planet details
    assert "800" in desc_full  # Temperature
    assert "0.30" in desc_full  # Albedo
    
    # Verify telescope details
    assert "num_collectors" in desc_full
    assert "latitude" in desc_full or "-24.6" in desc_full
    
    # Verify camera details
    assert "1024" in desc_full  # Pixels
    assert "0.01" in desc_full or "0.010" in desc_full  # Dark current
    assert "3.0 e-" in desc_full or "read_noise" in desc_full
    assert "90" in desc_full  # QE percentage
    
    print("✓ Full description includes all expected details")


def test_element_full_description():
    """Test full description for individual elements."""
    print("\n=== Test 3: Element Full Description ===")
    
    # Test Star
    star = helios.Star(temperature=6000*u.K, magnitude=4.5, mass=1.2*u.M_sun, name="Alpha Cen A")
    star_desc = star.description(full=True)
    print(f"\nStar description:\n{star_desc}")
    
    assert "Star" in star_desc
    assert "6000" in star_desc
    assert "4.50" in star_desc
    
    # Test Planet
    planet = helios.Planet(
        mass=2*u.M_jup,
        temperature=1200*u.K,
        albedo=0.2,
        name="Hot Jupiter"
    )
    planet_desc = planet.description(full=True)
    print(f"\nPlanet description:\n{planet_desc}")
    
    assert "Planet" in planet_desc
    assert "1200" in planet_desc
    assert "0.20" in planet_desc
    
    # Test Camera
    camera = helios.Camera(
        pixels=(2048, 2048),
        dark_current=0.001*u.electron/u.s,
        read_noise=2*u.electron,
        name="Science Camera"
    )
    camera_desc = camera.description(full=True)
    print(f"\nCamera description:\n{camera_desc}")
    
    assert "Camera" in camera_desc
    assert "2048" in camera_desc
    assert "0.001" in camera_desc
    
    print("✓ Element full descriptions work correctly")


def test_layer_full_description():
    """Test full description for layers with elements."""
    print("\n=== Test 4: Layer Full Description ===")
    
    # Create scene with multiple objects
    scene = helios.Scene(distance=50*u.pc, name="Multi-planet System")
    scene.add(helios.Star(temperature=5500*u.K, magnitude=6.0))
    scene.add(helios.Planet(mass=1*u.M_jup, temperature=600*u.K))
    scene.add(helios.Planet(mass=0.5*u.M_jup, temperature=300*u.K))
    
    scene_desc = scene.description(full=True)
    print(f"\nScene description:\n{scene_desc}")
    
    assert "Scene" in scene_desc
    assert "50.0 pc" in scene_desc or "50 pc" in scene_desc
    assert "num_objects: 3" in scene_desc
    assert "Star" in scene_desc
    assert "Planet" in scene_desc
    
    # Verify hierarchical structure
    assert "├" in scene_desc or "└" in scene_desc  # Tree connectors
    
    print("✓ Layer full description includes elements")


def test_telescope_array_full_description():
    """Test full description for telescope arrays."""
    print("\n=== Test 5: Telescope Array Full Description ===")
    
    # Single telescope
    single = helios.TelescopeArray(name="Single VLT UT")
    pupil = helios.Pupil.vlt()
    single.add_collector(pupil=pupil, position=(0, 0), size=8.2*u.m)
    
    single_desc = single.description(full=True)
    print(f"\nSingle telescope:\n{single_desc}")
    
    assert "num_collectors: 1" in single_desc
    assert "Single telescope" in single_desc
    
    # Interferometer
    interferometer = helios.TelescopeArray(name="VLTI")
    for i, pos in enumerate([(0, 0), (47, 0), (47, 47), (0, 47)]):
        interferometer.add_collector(pupil=pupil, position=pos, size=8.2*u.m, name=f"UT{i+1}")
    
    inter_desc = interferometer.description(full=True)
    print(f"\nInterferometer:\n{inter_desc}")
    
    assert "num_collectors: 4" in inter_desc
    assert "Interferometric" in inter_desc
    assert "max_baseline" in inter_desc
    
    print("✓ Telescope array descriptions distinguish single vs interferometric")


def test_beam_splitter_full_description():
    """Test beam splitter full description."""
    print("\n=== Test 6: Beam Splitter Full Description ===")
    
    bs = helios.BeamSplitter(cutoff=0.7, name="Dichroic")
    bs_desc = bs.description(full=True)
    print(f"\nBeam splitter:\n{bs_desc}")
    
    assert "BeamSplitter" in bs_desc
    assert "70" in bs_desc  # 70% transmission
    assert "30" in bs_desc  # 30% reflection
    
    print("✓ Beam splitter description includes split ratio")


def test_parallel_layers_full_description():
    """Test full description with parallel layers."""
    print("\n=== Test 7: Parallel Layers Full Description ===")
    
    ctx = helios.Context()
    
    scene = helios.Scene(distance=20*u.pc)
    scene.add(helios.Star(temperature=5800*u.K))
    
    bs = helios.BeamSplitter(cutoff=0.5)
    
    camera1 = helios.Camera(pixels=(512, 512), name="Visible", integration_time=5*u.s)
    camera2 = helios.Camera(pixels=(512, 512), name="IR", integration_time=10*u.s)
    
    ctx.add_layer(scene)
    ctx.add_layer(bs)
    ctx.add_layer([camera1, camera2])
    
    desc = ctx.description(full=True)
    print(f"\nParallel layers:\n{desc}")
    
    assert "Parallel" in desc
    assert "Branch 1" in desc
    assert "Branch 2" in desc
    assert "Visible" in desc
    assert "IR" in desc
    
    # Should show different integration times
    assert "5.0 s" in desc or "5 s" in desc
    assert "10.0 s" in desc or "10 s" in desc
    
    print("✓ Parallel layers full description works")


if __name__ == "__main__":
    print("Testing full description feature")
    print("=" * 70)
    
    try:
        test_context_basic_description()
        test_context_full_description()
        test_element_full_description()
        test_layer_full_description()
        test_telescope_array_full_description()
        test_beam_splitter_full_description()
        test_parallel_layers_full_description()
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
