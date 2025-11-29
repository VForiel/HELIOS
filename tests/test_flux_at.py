"""
Test flux_at() method for celestial objects.

This test validates that the flux_at() method correctly computes spectral
flux at specific wavelengths by comparing with direct SED evaluation.
"""
import sys
sys.path.insert(0, '../src')

import numpy as np
from astropy import units as u
import helios


def test_flux_at_star():
    """Test flux_at() for Star objects."""
    star = helios.Star(temperature=5700*u.K, magnitude=5, mass=1*u.M_sun)
    
    # Test at visible wavelength (550 nm)
    wavelength = 550 * u.nm
    flux = star.flux_at(wavelength)
    
    # Verify units
    assert flux.unit == u.W / (u.m**2 * u.m * u.sr), f"Wrong units: {flux.unit}"
    
    # Verify flux is positive
    assert flux.value > 0, "Flux should be positive"
    
    # Compare with direct SED evaluation at the same wavelength
    wl_grid = np.linspace(500, 600, 100) * u.nm
    wl_array, sed_array = star.sed(wavelengths=wl_grid)
    
    # Find closest wavelength in grid
    idx = np.argmin(np.abs(wl_array - wavelength))
    flux_direct = sed_array[idx]
    
    # Should be within 1% (interpolation accuracy)
    rel_error = np.abs((flux - flux_direct) / flux_direct).value
    assert rel_error < 0.01, f"Flux mismatch: {rel_error*100:.2f}% error"
    
    print(f"✓ Star flux at {wavelength}: {flux:.3e}")
    print(f"  Direct SED value: {flux_direct:.3e}")
    print(f"  Relative error: {rel_error*100:.4f}%")


def test_flux_at_planet():
    """Test flux_at() for Planet objects."""
    planet = helios.Planet(mass=1*u.M_jup)
    
    # Test at infrared wavelength (10 μm, thermal emission peak for cool planets)
    wavelength = 10 * u.um
    flux = planet.flux_at(wavelength)
    
    # Verify units and positivity
    assert flux.unit == u.W / (u.m**2 * u.m * u.sr)
    assert flux.value > 0
    
    print(f"✓ Planet flux at {wavelength}: {flux:.3e}")


def test_flux_at_zodiacal():
    """Test flux_at() for Zodiacal dust."""
    zodi = helios.Zodiacal(brightness=1.0)
    
    # Test at mid-IR (25 μm, thermal emission)
    wavelength = 25 * u.um
    flux = zodi.flux_at(wavelength)
    
    assert flux.unit == u.W / (u.m**2 * u.m * u.sr)
    assert flux.value > 0
    
    print(f"✓ Zodiacal flux at {wavelength}: {flux:.3e}")


def test_flux_at_exozodiacal():
    """Test flux_at() for ExoZodiacal dust."""
    exozodi = helios.ExoZodiacal(brightness=0.5)
    
    # Test at near-IR (3 μm)
    wavelength = 3 * u.um
    flux = exozodi.flux_at(wavelength)
    
    assert flux.unit == u.W / (u.m**2 * u.m * u.sr)
    assert flux.value > 0
    
    print(f"✓ ExoZodiacal flux at {wavelength}: {flux:.3e}")


def test_flux_at_temperature_override():
    """Test that temperature parameter can be overridden."""
    planet = helios.Planet(mass=1*u.M_jup)  # Default 300 K
    
    wavelength = 10 * u.um
    
    # Default temperature (300 K)
    flux_default = planet.flux_at(wavelength)
    
    # Hot planet (600 K)
    flux_hot = planet.flux_at(wavelength, temperature=600*u.K)
    
    # Hot planet should have higher flux
    assert flux_hot > flux_default, "Hot planet should emit more"
    
    print(f"✓ Temperature override works:")
    print(f"  Default (300K): {flux_default:.3e}")
    print(f"  Hot (600K): {flux_hot:.3e}")
    print(f"  Ratio: {(flux_hot/flux_default).value:.2f}x")


def test_flux_at_wavelength_conversion():
    """Test that wavelength units are properly handled."""
    star = helios.Star(temperature=5700*u.K)
    
    # Same wavelength in different units
    flux_nm = star.flux_at(550 * u.nm)
    flux_um = star.flux_at(0.55 * u.um)
    flux_m = star.flux_at(550e-9 * u.m)
    
    # Should all give the same result (within numerical precision)
    assert np.allclose(flux_nm.value, flux_um.value, rtol=1e-6)
    assert np.allclose(flux_nm.value, flux_m.value, rtol=1e-6)
    
    print(f"✓ Wavelength unit conversion works:")
    print(f"  550 nm: {flux_nm:.3e}")
    print(f"  0.55 μm: {flux_um:.3e}")
    print(f"  550e-9 m: {flux_m:.3e}")


def test_flux_at_physical_correctness():
    """Test physical coherence: B_λ peak for blackbodies."""
    star = helios.Star(temperature=5700*u.K)
    
    # For B_λ (spectral radiance per unit wavelength), the peak is at λ_max ≈ 2898 μm·K / (5T)
    # This is different from Wien's displacement for B_ν!
    # For T=5700K, B_λ peaks around ~410 nm
    
    # Sample flux at different wavelengths
    wavelengths = [300, 400, 500, 600, 800, 1000] * u.nm
    fluxes = [star.flux_at(wl) for wl in wavelengths]
    
    # Find peak in our samples
    peak_idx = np.argmax([f.value for f in fluxes])
    
    print(f"✓ Physical coherence check:")
    for wl, flux in zip(wavelengths, fluxes):
        marker = " ← peak" if wl == wavelengths[peak_idx] else ""
        print(f"  {wl}: {flux:.3e}{marker}")
    
    # Peak for B_λ should be in blue-violet range for solar-type star
    # (shorter than Wien's displacement peak for B_ν)
    assert wavelengths[peak_idx].value in [300, 400, 500], "Peak should be in UV-blue range for B_λ"


if __name__ == "__main__":
    print("Testing flux_at() method for celestial objects\n")
    print("=" * 60)
    
    test_flux_at_star()
    print()
    
    test_flux_at_planet()
    print()
    
    test_flux_at_zodiacal()
    print()
    
    test_flux_at_exozodiacal()
    print()
    
    test_flux_at_temperature_override()
    print()
    
    test_flux_at_wavelength_conversion()
    print()
    
    test_flux_at_physical_correctness()
    print()
    
    print("=" * 60)
    print("✓ All tests passed!")
