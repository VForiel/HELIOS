"""Test telescope array preset configurations (VLTI, LIFE)."""
import sys
sys.path.insert(0, '../src')

import numpy as np
from astropy import units as u
import helios


def test_vlti_uts_configuration():
    """Test VLTI Unit Telescopes preset configuration."""
    vlti = helios.TelescopeArray.vlti(uts=True)
    
    # Check basic properties
    assert vlti.name == "VLTI-UTs"
    assert len(vlti.collectors) == 4
    assert vlti.is_interferometric()
    
    # Check location (Paranal Observatory)
    assert abs(vlti.latitude.to(u.deg).value - (-24.627)) < 0.01
    assert abs(vlti.longitude.to(u.deg).value - (-70.404)) < 0.01
    assert abs(vlti.altitude.to(u.m).value - 2635) < 1
    
    # Check telescope sizes
    for collector in vlti.collectors:
        assert abs(collector.size.to(u.m).value - 8.2) < 0.01
    
    # Check baseline array shape
    baselines = vlti.get_baseline_array()
    assert baselines.shape == (4, 2)
    
    # Check that UT2 is at reference (0, 0)
    assert np.allclose(baselines[1], [0, 0])
    
    # Check baseline magnitudes are realistic (tens to ~100m)
    for i in range(len(baselines)):
        for j in range(i+1, len(baselines)):
            dist = np.linalg.norm(baselines[i] - baselines[j])
            assert 10 < dist < 150, f"Baseline {i}-{j} = {dist:.1f}m (expected 10-150m)"
    
    print("✓ VLTI UTs configuration validated")


def test_vlti_ats_configuration():
    """Test VLTI Auxiliary Telescopes preset configuration."""
    vlti = helios.TelescopeArray.vlti(uts=False)
    
    # Check basic properties
    assert vlti.name == "VLTI-ATs"
    assert len(vlti.collectors) == 4
    assert vlti.is_interferometric()
    
    # Check telescope sizes (1.8m for ATs)
    for collector in vlti.collectors:
        assert abs(collector.size.to(u.m).value - 1.8) < 0.01
    
    # Check baseline array shape
    baselines = vlti.get_baseline_array()
    assert baselines.shape == (4, 2)
    
    print("✓ VLTI ATs configuration validated")


def test_life_configuration():
    """Test LIFE space interferometer preset configuration."""
    life = helios.TelescopeArray.life()
    
    # Check basic properties
    assert life.name == "LIFE"
    assert len(life.collectors) == 4
    assert life.is_interferometric()
    
    # Check location (North Pole for space configuration)
    assert abs(life.latitude.to(u.deg).value - 90.0) < 0.01
    assert abs(life.longitude.to(u.deg).value - 0.0) < 0.01
    
    # Check telescope sizes (2m collectors)
    for collector in life.collectors:
        assert abs(collector.size.to(u.m).value - 2.0) < 0.01
    
    # Check baseline array shape
    baselines = life.get_baseline_array()
    assert baselines.shape == (4, 2)
    
    # Check that array is centered at (0, 0)
    centroid_x = baselines[:, 0].mean()
    centroid_y = baselines[:, 1].mean()
    assert abs(centroid_x) < 0.01, f"X centroid = {centroid_x} (should be ~0)"
    assert abs(centroid_y) < 0.01, f"Y centroid = {centroid_y} (should be ~0)"
    
    # Check that all collectors are equidistant from center
    distances = [np.linalg.norm(pos) for pos in baselines]
    mean_dist = np.mean(distances)
    for dist in distances:
        assert abs(dist - mean_dist) < 1.0, f"Collector not equidistant: {dist:.1f}m vs {mean_dist:.1f}m"
    
    # Check maximum baseline is in expected range (100-700m)
    max_baseline = 0
    for i in range(len(baselines)):
        for j in range(i+1, len(baselines)):
            dist = np.linalg.norm(baselines[i] - baselines[j])
            max_baseline = max(max_baseline, dist)
    
    assert 100 < max_baseline < 700, f"Max baseline = {max_baseline:.1f}m (expected 100-700m)"
    
    print("✓ LIFE configuration validated")


def test_preset_collectors_have_pupils():
    """Test that all preset collectors have valid pupil objects."""
    vlti = helios.TelescopeArray.vlti(uts=True)
    life = helios.TelescopeArray.life()
    
    for array in [vlti, life]:
        for collector in array.collectors:
            assert collector.pupil is not None
            assert hasattr(collector.pupil, 'get_array')
            
            # Test pupil rendering
            pupil_arr = collector.pupil.get_array(npix=128)
            assert pupil_arr.shape == (128, 128)
            assert 0 <= pupil_arr.min() <= pupil_arr.max() <= 1
    
    print("✓ All preset collectors have valid pupils")


def test_baseline_array_consistency():
    """Test that baseline arrays are consistent with collector positions."""
    vlti = helios.TelescopeArray.vlti(uts=True)
    
    baselines = vlti.get_baseline_array()
    
    for i, collector in enumerate(vlti.collectors):
        expected_pos = np.array(collector.position)
        actual_pos = baselines[i]
        assert np.allclose(expected_pos, actual_pos), \
            f"Collector {i} position mismatch: {expected_pos} vs {actual_pos}"
    
    print("✓ Baseline arrays consistent with collector positions")


if __name__ == "__main__":
    test_vlti_uts_configuration()
    test_vlti_ats_configuration()
    test_life_configuration()
    test_preset_collectors_have_pupils()
    test_baseline_array_consistency()
    print("\n✅ All telescope preset tests passed!")
