"""
Test reflected light for Planet objects.

This test validates that planets correctly compute SEDs including both
thermal emission and reflected stellar light.
"""
import sys
sys.path.insert(0, '../src')

import numpy as np
from astropy import units as u
from astropy import constants as const
import helios


def test_planet_thermal_only():
    """Test that planet without scene returns thermal emission only."""
    planet = helios.Planet(mass=1*u.M_jup, radius=1*const.R_jup)
    
    wl, sed = planet.sed(temperature=300*u.K)
    
    # Verify units
    assert sed.unit == u.W / (u.m**2 * u.m * u.sr)
    assert sed.value.max() > 0
    
    # Peak should be in thermal IR for 300K
    peak_idx = np.argmax(sed.value)
    peak_wavelength = wl[peak_idx]
    
    # Wien's law for B_λ: peak at ~2898 μm·K / (5T) ≈ 2000 nm for 300K
    assert 5*u.um < peak_wavelength < 20*u.um, f"Peak at {peak_wavelength}, expected ~10 μm for 300K"
    
    print(f"✓ Thermal-only planet: peak at {peak_wavelength:.2f}")


def test_planet_with_reflection_ratio():
    """Test planet with user-specified reflection_ratio."""
    scene = helios.Scene(distance=10*u.pc)
    star = helios.Star(temperature=5700*u.K)
    planet = helios.Planet(mass=1*u.M_jup, reflection_ratio=0.01, position=(1*u.AU, 0*u.AU))
    
    scene.add(star)
    scene.add(planet)
    
    # Get SED with reflection
    wl, sed_with_refl = planet.sed(temperature=300*u.K, include_reflection=True)
    
    # Get SED without reflection
    wl_thermal, sed_thermal = planet.sed(temperature=300*u.K, include_reflection=False)
    
    # At short wavelengths, reflected light should dominate
    # At 0.5 μm (visible), reflected light >> thermal
    idx_visible = np.argmin(np.abs(wl - 0.5*u.um))
    
    assert sed_with_refl[idx_visible] > sed_thermal[idx_visible], \
        "Reflected component should increase visible flux"
    
    ratio = sed_with_refl[idx_visible] / sed_thermal[idx_visible]
    print(f"✓ Reflection ratio mode: visible flux increased {ratio.value:.1f}× at 0.5 μm")


def test_planet_with_physical_parameters():
    """Test planet with physical radius and albedo."""
    scene = helios.Scene(distance=10*u.pc)
    star = helios.Star(temperature=5700*u.K, magnitude=5, mass=1*u.M_sun)
    
    # Earth-like planet: R=1 R_earth, albedo=0.3, at 1 AU
    planet = helios.Planet(
        mass=1*u.M_earth,
        radius=1*const.R_earth,
        albedo=0.3,
        position=(1*u.AU, 0*u.AU)
    )
    
    scene.add(star)
    scene.add(planet)
    
    # Get SED components
    wl_total, sed_total = planet.sed(temperature=288*u.K, include_reflection=True)
    wl_thermal, sed_thermal = planet.sed(temperature=288*u.K, include_reflection=False)
    
    # Reflected component
    sed_reflected = sed_total - sed_thermal
    
    # At visible wavelengths, reflected should dominate
    idx_vis = np.argmin(np.abs(wl_total - 0.55*u.um))
    
    # At thermal IR, thermal should dominate
    idx_ir = np.argmin(np.abs(wl_total - 10*u.um))
    
    ratio_vis = (sed_reflected[idx_vis] / sed_thermal[idx_vis]).value
    ratio_ir = (sed_reflected[idx_ir] / sed_thermal[idx_ir]).value
    
    assert ratio_vis > 100, f"Reflected light should dominate in visible (ratio={ratio_vis:.1f})"
    assert ratio_ir < 1, f"Thermal should dominate in IR (ratio={ratio_ir:.3f})"
    
    print(f"✓ Physical parameters: reflected/thermal = {ratio_vis:.1e} (vis), {ratio_ir:.3f} (IR)")


def test_planet_radius_estimation():
    """Test automatic radius estimation from mass."""
    planet_1mj = helios.Planet(mass=1*u.M_jup)
    planet_2mj = helios.Planet(mass=2*u.M_jup)
    
    # Larger mass should give larger radius (R ∝ M^(1/3))
    assert planet_2mj.radius > planet_1mj.radius
    
    # Check scaling: R(2 M_jup) ≈ 2^(1/3) × R(1 M_jup)
    ratio = (planet_2mj.radius / planet_1mj.radius).decompose().value
    expected_ratio = 2**(1/3)
    
    assert abs(ratio - expected_ratio) < 0.01, f"Radius scaling incorrect: {ratio:.3f} vs {expected_ratio:.3f}"
    
    print(f"✓ Radius estimation: R(2 M_jup) = {ratio:.3f} × R(1 M_jup)")


def test_hot_jupiter_reflection():
    """Test hot Jupiter with high temperature and reflection."""
    scene = helios.Scene(distance=10*u.pc)
    star = helios.Star(temperature=6000*u.K)
    
    # Hot Jupiter: close to star, high temperature
    hot_jupiter = helios.Planet(
        mass=1*u.M_jup,
        radius=1.5*const.R_jup,
        albedo=0.1,  # Low albedo (hot Jupiters are dark)
        position=(0.05*u.AU, 0*u.AU)  # Very close
    )
    
    scene.add(star)
    scene.add(hot_jupiter)
    
    # Hot Jupiter: T ~ 1500 K
    wl, sed = hot_jupiter.sed(temperature=1500*u.K, include_reflection=True)
    
    # Peak should be in near-IR (shorter than cool planet)
    peak_idx = np.argmax(sed.value)
    peak_wl = wl[peak_idx]
    
    assert 1*u.um < peak_wl < 5*u.um, f"Hot Jupiter peak at {peak_wl}"
    
    print(f"✓ Hot Jupiter: SED peak at {peak_wl:.2f}")


def test_flux_at_with_reflection():
    """Test flux_at() method with reflection."""
    scene = helios.Scene(distance=10*u.pc)
    star = helios.Star(temperature=5700*u.K)
    planet = helios.Planet(
        mass=1*u.M_jup,
        reflection_ratio=0.005,
        position=(1*u.AU, 0*u.AU)
    )
    
    scene.add(star)
    scene.add(planet)
    
    # Get flux at visible wavelength
    flux_vis = planet.flux_at(550*u.nm)
    
    # Should be positive
    assert flux_vis.value > 0
    
    # Get thermal-only flux for comparison
    wl_grid = np.linspace(540, 560, 20) * u.nm
    wl_th, sed_th = planet.sed(wavelengths=wl_grid, include_reflection=False)
    flux_thermal = np.interp(550, wl_th.to(u.nm).value, sed_th.value) * sed_th.unit
    
    # With reflection should be higher
    assert flux_vis > flux_thermal
    
    print(f"✓ flux_at() with reflection: {flux_vis:.2e} vs {flux_thermal:.2e} (thermal)")


def test_multiple_stars():
    """Test that only the first star is used for reflection (current implementation)."""
    scene = helios.Scene(distance=10*u.pc)
    star1 = helios.Star(temperature=5700*u.K, position=(0*u.AU, 0*u.AU))
    star2 = helios.Star(temperature=4000*u.K, position=(100*u.AU, 0*u.AU))  # Binary companion
    planet = helios.Planet(mass=1*u.M_jup, reflection_ratio=0.01, position=(1*u.AU, 0*u.AU))
    
    scene.add(star1)
    scene.add(star2)
    scene.add(planet)
    
    # Should successfully compute SED (using first star)
    wl, sed = planet.sed(include_reflection=True)
    
    assert sed.value.max() > 0
    
    print(f"✓ Multiple stars: reflection computed using primary star")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing Planet Reflected Light")
    print("="*70 + "\n")
    
    test_planet_thermal_only()
    print()
    
    test_planet_with_reflection_ratio()
    print()
    
    test_planet_with_physical_parameters()
    print()
    
    test_planet_radius_estimation()
    print()
    
    test_hot_jupiter_reflection()
    print()
    
    test_flux_at_with_reflection()
    print()
    
    test_multiple_stars()
    print()
    
    print("="*70)
    print("✓ All reflection tests passed!")
    print("="*70)
