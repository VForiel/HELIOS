"""
Demonstration of the flux_at() method for celestial objects.

This script shows how to use the new flux_at() method to compute
spectral flux at specific wavelengths without generating the full SED.
"""
import sys
sys.path.insert(0, '../src')

import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
import helios


def demo_flux_at_basic():
    """Basic usage of flux_at() method."""
    print("=" * 70)
    print("Basic flux_at() usage")
    print("=" * 70)
    
    # Create celestial objects
    star = helios.Star(temperature=5700*u.K, magnitude=5, mass=1*u.M_sun)
    planet = helios.Planet(mass=1*u.M_jup)
    zodi = helios.Zodiacal(brightness=1.0)
    
    # Compute flux at specific wavelengths
    wl_visible = 550 * u.nm
    wl_ir = 10 * u.um
    
    star_flux_vis = star.flux_at(wl_visible)
    planet_flux_ir = planet.flux_at(wl_ir)
    zodi_flux_ir = zodi.flux_at(wl_ir)
    
    print(f"\nStar (T={star.temperature}):")
    print(f"  Flux at {wl_visible}: {star_flux_vis:.3e}")
    
    print(f"\nPlanet (default T=300K):")
    print(f"  Flux at {wl_ir}: {planet_flux_ir:.3e}")
    
    print(f"\nZodiacal dust (T=270K):")
    print(f"  Flux at {wl_ir}: {zodi_flux_ir:.3e}")
    
    print(f"\nRatio (planet/zodiacal): {(planet_flux_ir/zodi_flux_ir).value:.2f}×")
    print()


def demo_flux_at_temperature():
    """Demonstrate temperature dependence."""
    print("=" * 70)
    print("Temperature dependence of thermal emission")
    print("=" * 70)
    
    planet = helios.Planet(mass=1*u.M_jup)
    wavelength = 10 * u.um
    
    temperatures = [200, 300, 400, 600, 800, 1000] * u.K
    fluxes = [planet.flux_at(wavelength, temperature=T) for T in temperatures]
    
    print(f"\nFlux at {wavelength} for different planet temperatures:")
    for T, flux in zip(temperatures, fluxes):
        print(f"  T = {T:4.0f}: {flux:.3e}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    temps_val = [T.value for T in temperatures]
    fluxes_val = [f.value for f in fluxes]
    
    ax.semilogy(temps_val, fluxes_val, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel(f'Spectral flux at {wavelength}\n({fluxes[0].unit})', fontsize=12)
    ax.set_title('Thermal emission vs temperature', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('flux_vs_temperature.png', dpi=150)
    print(f"\n✓ Saved plot to flux_vs_temperature.png")
    plt.show()
    print()


def demo_flux_at_wavelength_scan():
    """Scan flux across wavelengths and compare with full SED."""
    print("=" * 70)
    print("Wavelength scan: flux_at() vs full SED")
    print("=" * 70)
    
    star = helios.Star(temperature=5700*u.K)
    
    # Sample wavelengths
    wavelengths = np.logspace(np.log10(300), np.log10(3000), 50) * u.nm
    
    # Method 1: flux_at() for each wavelength
    fluxes_at = [star.flux_at(wl) for wl in wavelengths]
    
    # Method 2: Full SED generation
    wl_sed, flux_sed = star.sed(wavelengths=wavelengths)
    
    # Compare
    print(f"\nComparing two methods at {len(wavelengths)} wavelengths:")
    print(f"  Max relative difference: {np.max(np.abs((np.array([f.value for f in fluxes_at]) - flux_sed.value) / flux_sed.value)):.2e}")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Main plot
    wl_nm = [w.to(u.nm).value for w in wavelengths]
    flux_at_vals = [f.value for f in fluxes_at]
    flux_sed_vals = flux_sed.value
    
    ax1.loglog(wl_nm, flux_at_vals, 'o', label='flux_at()', markersize=4, alpha=0.7)
    ax1.loglog(wl_nm, flux_sed_vals, '-', label='sed()', linewidth=1, alpha=0.9)
    ax1.set_ylabel(f'Spectral flux ({fluxes_at[0].unit})', fontsize=11)
    ax1.set_title(f'Star SED (T={star.temperature})', fontsize=13)
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)
    
    # Residuals
    rel_diff = (np.array(flux_at_vals) - flux_sed_vals) / flux_sed_vals
    ax2.semilogx(wl_nm, rel_diff * 100, 'o-', markersize=3)
    ax2.set_xlabel('Wavelength (nm)', fontsize=11)
    ax2.set_ylabel('Relative difference (%)', fontsize=11)
    ax2.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax2.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('flux_at_comparison.png', dpi=150)
    print(f"✓ Saved plot to flux_at_comparison.png")
    plt.show()
    print()


def demo_flux_at_multiobject():
    """Compare fluxes from multiple objects."""
    print("=" * 70)
    print("Multi-object flux comparison")
    print("=" * 70)
    
    # Create objects
    star = helios.Star(temperature=5700*u.K, magnitude=5, mass=1*u.M_sun)
    planet = helios.Planet(mass=1*u.M_jup)  # Cool
    hot_planet = helios.Planet(mass=1*u.M_jup)  # Hot Jupiter
    exozodi = helios.ExoZodiacal(brightness=1.0)
    
    # Wavelengths of interest
    wavelengths = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0] * u.um
    
    print(f"\nFlux comparison at different wavelengths:")
    print(f"{'Wavelength':<12} {'Star':<12} {'Planet':<12} {'Hot Planet':<12} {'ExoZodi':<12}")
    print("-" * 70)
    
    for wl in wavelengths:
        f_star = star.flux_at(wl).value
        f_planet = planet.flux_at(wl, temperature=300*u.K).value
        f_hot = hot_planet.flux_at(wl, temperature=1500*u.K).value
        f_exozodi = exozodi.flux_at(wl).value
        
        print(f"{wl.value:5.1f} μm    {f_star:.2e}  {f_planet:.2e}  {f_hot:.2e}  {f_exozodi:.2e}")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    wl_vals = [w.to(u.um).value for w in wavelengths]
    
    objects = [
        (star, "Star (5700K)", 'gold'),
        (planet, "Planet (300K)", 'blue'),
        (hot_planet, "Hot Jupiter (1500K)", 'red'),
        (exozodi, "ExoZodi (270K)", 'orange')
    ]
    
    for obj, label, color in objects:
        if "Hot" in label:
            fluxes = [obj.flux_at(wl, temperature=1500*u.K).value for wl in wavelengths]
        elif "Planet" in label:
            fluxes = [obj.flux_at(wl, temperature=300*u.K).value for wl in wavelengths]
        else:
            fluxes = [obj.flux_at(wl).value for wl in wavelengths]
        
        ax.loglog(wl_vals, fluxes, 'o-', label=label, color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('Wavelength (μm)', fontsize=12)
    ax.set_ylabel(f'Spectral flux ({star.flux_at(1*u.um).unit})', fontsize=12)
    ax.set_title('Multi-object flux comparison', fontsize=14)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multiobject_flux_comparison.png', dpi=150)
    print(f"\n✓ Saved plot to multiobject_flux_comparison.png")
    plt.show()
    print()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("HELIOS flux_at() Method Demonstration")
    print("="*70 + "\n")
    
    demo_flux_at_basic()
    demo_flux_at_temperature()
    demo_flux_at_wavelength_scan()
    demo_flux_at_multiobject()
    
    print("="*70)
    print("✓ All demonstrations complete!")
    print("="*70)
