"""
Test script to validate layer/element parent references and description() methods.

This validates the new architecture where:
- Each Layer has a 'context' attribute referencing the parent Context
- Each Element has a 'layer' attribute referencing the parent Layer
- Each Element has a 'context' shortcut (equivalent to element.layer.context)
- All classes have a description() method for hierarchical text output
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import helios
from astropy import units as u


def test_layer_context_reference():
    """Test that layers have correct context references."""
    print("\n=== Test 1: Layer context references ===")
    
    # Create context
    ctx = helios.Context()
    
    # Create layers
    scene = helios.Scene(distance=10*u.pc, name="Test Scene")
    telescope = helios.TelescopeArray(name="Test Telescope")
    camera = helios.Camera(pixels=(512, 512), name="Test Camera")
    
    # Add layers
    ctx.add_layer(scene)
    ctx.add_layer(telescope)
    ctx.add_layer(camera)
    
    # Verify context references
    assert scene.context is ctx, "Scene context reference not set"
    assert telescope.context is ctx, "Telescope context reference not set"
    assert camera.context is ctx, "Camera context reference not set"
    
    print("✓ All layers have correct context references")


def test_element_layer_reference():
    """Test that elements have correct layer references."""
    print("\n=== Test 2: Element layer and context references ===")
    
    # Create context and telescope
    ctx = helios.Context()
    telescope = helios.TelescopeArray(name="Test Array")
    
    # Add collector (element)
    pupil = helios.Pupil.vlt()
    telescope.add_collector(pupil=pupil, position=(0, 0), size=8*u.m)
    
    # Add telescope to context
    ctx.add_layer(telescope)
    
    # Verify references
    assert len(telescope.elements) > 0, "No elements in telescope"
    
    for element in telescope.elements:
        assert element.layer is telescope, f"Element layer reference not set: {element}"
        assert element.context is ctx, f"Element context reference not set: {element}"
        # Verify shortcut
        assert element.context == element.layer.context, "Context shortcut mismatch"
    
    print(f"✓ All {len(telescope.elements)} telescope elements have correct references")


def test_parallel_layers_references():
    """Test that parallel layers have correct references."""
    print("\n=== Test 3: Parallel layers context references ===")
    
    # Create context
    ctx = helios.Context()
    
    # Create parallel layers
    camera1 = helios.Camera(pixels=(256, 256), name="Camera 1")
    camera2 = helios.Camera(pixels=(256, 256), name="Camera 2")
    
    # Add as parallel layers
    ctx.add_layer([camera1, camera2])
    
    # Verify both cameras have context reference
    assert camera1.context is ctx, "Camera 1 context reference not set"
    assert camera2.context is ctx, "Camera 2 context reference not set"
    
    print("✓ Parallel layers have correct context references")


def test_element_description():
    """Test Element description() method."""
    print("\n=== Test 4: Element description() method ===")
    
    # Create a simple element (Camera)
    camera = helios.Camera(pixels=(512, 512), name="Science Camera")
    
    desc = camera.description()
    print(f"Camera description:\n{desc}")
    
    assert "Camera" in desc, "Description should contain class name"
    assert "Science Camera" in desc, "Description should contain element name"
    
    print("✓ Element description() works correctly")


def test_layer_description():
    """Test Layer description() method."""
    print("\n=== Test 5: Layer description() method ===")
    
    # Create telescope with multiple collectors
    telescope = helios.TelescopeArray(name="VLT Array")
    
    pupil = helios.Pupil.vlt()
    telescope.add_collector(pupil=pupil, position=(0, 0), size=8*u.m)
    telescope.add_collector(pupil=pupil, position=(100, 0)*u.m, size=8*u.m)
    
    desc = telescope.description()
    print(f"\nTelescope description:\n{desc}")
    
    assert "TelescopeArray" in desc, "Description should contain class name"
    assert "VLT Array" in desc, "Description should contain layer name"
    
    # Should show elements if they exist
    if telescope.elements:
        assert "Collector" in desc or "└" in desc or "├" in desc, \
            "Description should show tree structure for elements"
    
    print("✓ Layer description() works correctly")


def test_context_description():
    """Test Context description() method."""
    print("\n=== Test 6: Context description() method ===")
    
    # Create complete simulation
    ctx = helios.Context()
    
    # Scene
    scene = helios.Scene(distance=10*u.pc, name="Target System")
    scene.add(helios.Star(temperature=5700*u.K, magnitude=5))
    
    # Telescope
    telescope = helios.TelescopeArray(name="Observatory")
    pupil = helios.Pupil.vlt()
    telescope.add_collector(pupil=pupil, position=(0, 0), size=8*u.m)
    
    # Camera
    camera = helios.Camera(pixels=(512, 512), name="Detector")
    
    # Build context
    ctx.add_layer(scene)
    ctx.add_layer(telescope)
    ctx.add_layer(camera)
    
    # Get full description
    desc = ctx.description()
    print(f"\nComplete context description:\n{desc}")
    
    assert "HELIOS" in desc, "Description should have HELIOS header"
    assert "Scene" in desc, "Description should list Scene layer"
    assert "TelescopeArray" in desc or "Observatory" in desc, "Description should list Telescope"
    assert "Camera" in desc or "Detector" in desc, "Description should list Camera"
    
    print("✓ Context description() works correctly")


def test_parallel_layers_description():
    """Test description with parallel layers."""
    print("\n=== Test 7: Description with parallel layers ===")
    
    # Create context with beam splitting
    ctx = helios.Context()
    
    scene = helios.Scene(distance=10*u.pc)
    beam_splitter = helios.BeamSplitter()
    camera1 = helios.Camera(pixels=(256, 256), name="Science")
    camera2 = helios.Camera(pixels=(256, 256), name="Reference")
    
    ctx.add_layer(scene)
    ctx.add_layer(beam_splitter)
    ctx.add_layer([camera1, camera2])
    
    desc = ctx.description()
    print(f"\nContext with parallel layers:\n{desc}")
    
    assert "Parallel" in desc or "Branch" in desc, \
        "Description should indicate parallel layers"
    assert "Science" in desc and "Reference" in desc, \
        "Description should show both camera names"
    
    print("✓ Parallel layer description works correctly")


def test_reference_propagation():
    """Test that references are properly propagated when elements are added."""
    print("\n=== Test 8: Reference propagation on element addition ===")
    
    # Create context and telescope
    ctx = helios.Context()
    telescope = helios.TelescopeArray(name="Array")
    ctx.add_layer(telescope)
    
    # Add element AFTER telescope is added to context
    pupil = helios.Pupil.vlt()
    telescope.add_collector(pupil=pupil, position=(0, 0), size=8*u.m)
    
    # Verify new element has references
    assert len(telescope.elements) > 0, "Element not added"
    element = telescope.elements[-1]  # Last added element
    
    assert element.layer is telescope, "Element layer reference not set after addition"
    assert element.context is ctx, "Element context reference not set after addition"
    
    print("✓ References properly propagated when elements added after layer in context")


if __name__ == "__main__":
    print("Testing Layer/Element parent references and description() methods")
    print("=" * 70)
    
    try:
        test_layer_context_reference()
        test_element_layer_reference()
        test_parallel_layers_references()
        test_element_description()
        test_layer_description()
        test_context_description()
        test_parallel_layers_description()
        test_reference_propagation()
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
