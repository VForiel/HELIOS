
import pytest
import astropy.units as u
import numpy as np
from helios.core.pipeline import Pipeline, Layer
from helios.components import Scene
from helios.components import TelescopeArray, Collector
from helios.components import Pupil
from helios.core.wavefront import Wavefront, WavefrontArray

def test_pipeline_empty_error():
    """Test that get_input_wavefront raises ValueError when Pipeline is empty."""
    pipe = Pipeline()
    with pytest.raises(ValueError, match="Pipeline must contain at least a Scene or a TelescopeArray"):
        pipe.get_input_wavefront()

def test_pipeline_auto_detect_telescope_array():
    """Test that get_input_wavefront automatically detects TelescopeArray and returns WavefrontArray."""
    pipe = Pipeline()
    
    # Create a TelescopeArray
    ta = TelescopeArray(name="TestArray")
    pupil = Pupil(diameter=1*u.m)
    pupil.add_disk(radius=0.5*u.m)
    
    # Add two collectors
    ta.add_collector(pupil=pupil, position=(0, 0), size=1*u.m, name="C1")
    ta.add_collector(pupil=pupil, position=(10, 0), size=1*u.m, name="C2")
    
    # Add to pipeline
    pipe.add_layer(ta)
    
    # Call get_input_wavefront without collectors argument
    # Note: No Scene added, so it should use default source but still return WavefrontArray because TA is present
    wf = pipe.get_input_wavefront()
    
    assert isinstance(wf, WavefrontArray)
    assert len(wf) == 2
    assert wf[0].pixel_scale is not None
    assert wf[1].pixel_scale is not None

def test_pipeline_scene_only():
    """Test that get_input_wavefront returns single Wavefront when only Scene is present."""
    pipe = Pipeline()
    scene = Scene()
    pipe.add_layer(scene)
    
    wf = pipe.get_input_wavefront()
    
    assert isinstance(wf, Wavefront)
    assert not isinstance(wf, WavefrontArray)

if __name__ == "__main__":
    # Manually run tests if executed as script
    try:
        test_context_empty_error()
        print("test_context_empty_error passed")
        test_context_auto_detect_telescope_array()
        print("test_context_auto_detect_telescope_array passed")
        test_context_scene_only()
        print("test_context_scene_only passed")
    except Exception as e:
        print(f"Test failed: {e}")
        raise
