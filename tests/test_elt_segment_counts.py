from astropy import units as u
from helios.components.pupil import Pupil

def test_elt_segment_counts():
    p = Pupil.elt()
    transmissive = 0
    occluding = 0
    for e in p.elements:
        if e.get('type') == 'hex':
            if e.get('value', 1.0) > 0:
                transmissive += 1
            else:
                occluding += 1
    assert transmissive == 798, f"Expected 798 transmissive segments, got {transmissive}"
    assert occluding == 61, f"Expected 61 occluding (secondary) hexes, got {occluding}"