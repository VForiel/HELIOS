import pytest
import numpy as np
from astropy import units as u
from helios.utils.serialization import serialize_value, deserialize_value
from helios.utils.plotting import get_smart_extent, format_coord
import helios.components.scene
import helios.components.pupil

def test_serialization():
    # Test Quantity serialization
    q = 10 * u.m
    serialized = serialize_value(q)
    assert serialized == {"value": 10.0, "unit": "m"}
    deserialized = deserialize_value(serialized)
    assert deserialized == q

    # Test dict serialization
    d = {"lens": 5 * u.mm, "name": "test"}
    s = serialize_value(d)
    assert s["lens"]["value"] == 5.0
    ds = deserialize_value(s)
    assert ds["lens"] == 5 * u.mm
    assert ds["name"] == "test"

def test_plotting_utils():
    # Test get_smart_extent
    shape = (100, 100)
    pixel_scale = 1 * u.cm
    extent, xlabel, ylabel = get_smart_extent(shape, pixel_scale)
    # 100pix * 1cm = 100cm = 1m. Extent should be +/- 0.5m
    assert extent is not None
    assert xlabel == "x [m]"
    
    # Test format_coord
    c = (1 * u.mm, 1 * u.mm)
    fmt = format_coord(c)
    assert "mm" in fmt

def test_imports_integrity():
    # Verify that components can be imported (meaning they found the moved functions)
    from helios.components.scene import Scene
    s = Scene(distance=10*u.pc)
    assert s.distance == 10*u.pc
    
    from helios.components.pupil import Pupil
    p = Pupil(diameter=8*u.m)
    assert p.diameter == 8*u.m
