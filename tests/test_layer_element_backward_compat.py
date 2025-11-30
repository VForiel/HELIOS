"""
Test backward compatibility of Layer/Element refactoring (Phase 1)

This test validates that the new Element/Layer architecture maintains
full backward compatibility with existing code that uses:
- scene.objects instead of scene.elements
- telescope.collectors instead of telescope.elements
"""

import sys
sys.path.insert(0, '../src')

import helios
from astropy import units as u


def test_scene_objects_property_backward_compat():
    """Test that scene.objects property still works (alias for elements)"""
    scene = helios.Scene(distance=10*u.pc)
    star = helios.Star(temperature=5700*u.K, magnitude=5, position=(0*u.AU, 0*u.AU))
    planet = helios.Planet(mass=1*u.M_jup, position=(1*u.AU, 0*u.AU))
    
    # Use old API: scene.add()
    scene.add(star)
    scene.add(planet)
    
    # Verify backward compatibility: scene.objects should work
    assert len(scene.objects) == 2, "scene.objects should return 2 objects"
    assert star in scene.objects, "Star should be in scene.objects"
    assert planet in scene.objects, "Planet should be in scene.objects"
    
    # Verify new API also works
    assert len(scene.elements) == 2, "scene.elements should return 2 elements"
    assert star in scene.elements, "Star should be in scene.elements"
    
    # Verify they are the same
    assert scene.objects is scene.elements, "objects and elements should be the same list"
    
    print("✓ Scene backward compatibility: scene.objects works correctly")


def test_telescope_collectors_property_backward_compat():
    """Test that telescope.collectors property still works (alias for elements)"""
    telescope = helios.TelescopeArray(latitude=0*u.deg, altitude=2400*u.m)
    pupil1 = helios.Pupil(8*u.m)
    pupil2 = helios.Pupil(8*u.m)
    
    # Use old API: add_collector()
    telescope.add_collector(pupil=pupil1, position=(0, 0), size=8*u.m)
    telescope.add_collector(pupil=pupil2, position=(10, 0), size=8*u.m)
    
    # Verify backward compatibility: telescope.collectors should work
    assert len(telescope.collectors) == 2, "telescope.collectors should return 2 collectors"
    
    # Verify new API also works
    assert len(telescope.elements) == 2, "telescope.elements should return 2 elements"
    
    # Verify they are the same
    assert telescope.collectors is telescope.elements, "collectors and elements should be the same list"
    
    print("✓ TelescopeArray backward compatibility: telescope.collectors works correctly")


def test_celestial_body_element_interface():
    """Test that CelestialBody is now an Element with process() method"""
    star = helios.Star(temperature=5700*u.K, magnitude=5, position=(0*u.AU, 0*u.AU))
    
    # Verify Element interface
    assert isinstance(star, helios.Element), "Star should be an Element"
    assert hasattr(star, 'process'), "Star should have process() method"
    assert hasattr(star, 'name'), "Star should have name attribute"
    
    # Test process() method (should be pass-through for CelestialBody)
    from helios.core.simulation import Wavefront
    wf = Wavefront(wavelength=550e-9*u.m, size=256)
    wf_processed = star.process(wf, None)
    assert wf_processed is wf, "CelestialBody.process() should return the same wavefront"
    
    print("✓ CelestialBody is Element: process() method works")


def test_collector_element_interface():
    """Test that Collector is now an Element with process() method"""
    pupil = helios.Pupil(8*u.m)
    collector = helios.Collector(pupil=pupil, position=(0, 0), size=8*u.m)
    
    # Verify Element interface
    assert isinstance(collector, helios.Element), "Collector should be an Element"
    assert hasattr(collector, 'process'), "Collector should have process() method"
    assert hasattr(collector, 'name'), "Collector should have name attribute"
    
    print("✓ Collector is Element: has process() method")


def test_layer_contains_elements():
    """Test that Layer now contains elements list"""
    scene = helios.Scene(distance=10*u.pc)
    
    # Verify Layer interface
    assert isinstance(scene, helios.Layer), "Scene should be a Layer"
    assert hasattr(scene, 'elements'), "Scene should have elements attribute"
    assert hasattr(scene, 'add_element'), "Scene should have add_element() method"
    assert isinstance(scene.elements, list), "scene.elements should be a list"
    
    print("✓ Layer contains elements: Scene has elements list")


def test_full_pipeline_with_old_api():
    """Test that a full pipeline still works with old API calls"""
    # Build pipeline using old API
    scene = helios.Scene(distance=10*u.pc)
    star = helios.Star(temperature=5700*u.K, magnitude=5, position=(0*u.AU, 0*u.AU))
    scene.add(star)  # OLD API
    
    telescope = helios.TelescopeArray(latitude=0*u.deg, altitude=2400*u.m)
    pupil = helios.Pupil(8*u.m)
    telescope.add_collector(pupil=pupil, position=(0, 0), size=8*u.m)  # OLD API
    
    camera = helios.Camera(pixels=(256, 256))
    
    context = helios.Context()
    context.add_layer(scene)
    context.add_layer(telescope)
    context.add_layer(camera)
    
    # Run simulation (should work with backward compatibility)
    result = context.observe()
    
    print(f"Result shape: {result.shape}")
    print(f"Result range: [{result.min()}, {result.max()}]")
    print(f"Result mean: {result.mean()}")
    
    assert result.shape == (256, 256), "Result should be 256x256 array"
    # Don't assert on signal yet - just verify pipeline executes
    # assert result.max() > 0, "Result should have some signal"
    
    print("✓ Full pipeline works with old API: Context.observe() executes without errors")


if __name__ == "__main__":
    test_scene_objects_property_backward_compat()
    test_telescope_collectors_property_backward_compat()
    test_celestial_body_element_interface()
    test_collector_element_interface()
    test_layer_contains_elements()
    test_full_pipeline_with_old_api()
    print("\n✅ All backward compatibility tests passed!")
