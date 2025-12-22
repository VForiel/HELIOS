import sys
import os
import pytest
from unittest.mock import MagicMock

# Add paths
sys.path.insert(0, os.path.abspath("d:/HELIOS/src"))
sys.path.append(os.path.abspath("d:/HELIOS"))
sys.path.append(os.path.abspath("d:/HELIOS/web/backend"))

# Import app module
# We need to mock 'fastapi' if not installed in the environment running this test?
# Assuming environment has dependencies.
try:
    from app import (
        LensPayload, BeamSplitterPayload, CoronagraphPayload, FiberPayload, PhotonicPayload,
        create_lens, create_beam_splitter, create_coronagraph, create_fiber, create_photonic
    )
    import helios.components as components
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_create_lens():
    payload = LensPayload(focal_length=2.5)
    lens = create_lens(payload)
    assert lens is not None
    # Check property (Lens implementation dependant)
    # Assuming standard lens
    pass

def test_create_beam_splitter():
    payload = BeamSplitterPayload(split_ratio=0.7)
    bs = create_beam_splitter(payload)
    assert bs is not None
    # assert bs.split_ratio == 0.7

def test_create_coronagraph():
    payload = CoronagraphPayload(type="vortex")
    coro = create_coronagraph(payload)
    assert coro is not None

def test_create_fiber():
    p_in = FiberPayload(modes=3, name="In1")
    fib_in = create_fiber(p_in, is_input=True)
    assert fib_in.name == "In1"
    
    p_out = FiberPayload(name="Out1")
    fib_out = create_fiber(p_out, is_input=False)
    assert fib_out.name == "Out1"

def test_create_photonic_tops():
    payload = PhotonicPayload(type='tops', phase=1.57, name="P1")
    tops = create_photonic(payload)
    assert isinstance(tops, components.TOPS)
    assert tops.name == "P1"

def test_create_photonic_mmi():
    payload = PhotonicPayload(type='mmi', matrix_preset='hadamard', name="M1")
    mmi = create_photonic(payload)
    assert isinstance(mmi, components.MMI)

if __name__ == "__main__":
    # Manually run if pytest not available
    test_create_lens()
    test_create_beam_splitter()
    test_create_coronagraph()
    test_create_fiber()
    test_create_photonic_tops()
    test_create_photonic_mmi()
    print("All tests passed!")
