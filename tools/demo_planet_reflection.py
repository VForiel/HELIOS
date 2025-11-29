"""
Demonstration of planet reflected light capabilities.

This script shows how to model planets with both thermal emission
and reflected stellar light, using either physical parameters or
simple reflection ratios.
"""
import sys
sys.path.insert(0, '../src')

import numpy as np
from astropy import units as u
from astropy import constants as const
import matplotlib.pyplot as plt
import helios


def demo_thermal_vs_reflected():
    """Compare thermal-only vs thermal+reflected planet SEDs."""
    print("=" * 70)
    print("Demo 1: Thermal vs Reflected+Thermal")
    print("=" * 70)
    
    # Create scene with star
    scene = helios.Scene(distance=10*u.pc)
    star = helios.Star(temperature=5700*u.K, magnitude=5, mass=1*u.M_sun)
    
    # Jupiter-like planet at 1 AU
    planet = helios.Planet(
        mass=1*u.M_jup,
        radius=1*const.R_jup,
        albedo=0.5,  # Jupiter's actual albedo ~ 0.5
        position=(1*u.AU, 0*u.AU)
    )
    
    scene.add(star)
    scene.add(planet)
    
    # Wavelength grid
    wavelengths = np.logspace(np.log10(0.3), np.log10(100), 200) * u.um
    
    # Get SEDs
    wl_thermal, sed_thermal = planet.sed(wavelengths=wavelengths, 
                                         temperature=124*u.K,  # Jupiter's equilibrium temp
                                         include_reflection=False)
    wl_total, sed_total = planet.sed(wavelengths=wavelengths,
                                     temperature=124*u.K,
                                     include_reflection=True)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Main plot
    ax1.loglog(wl_thermal.to(u.um).value, sed_thermal.value, 
               label='Thermal only', linewidth=2, alpha=0.7)
    ax1.loglog(wl_total.to(u.um).value, sed_total.value,
               label='Thermal + Reflected', linewidth=2)
    ax1.set_xlabel('Wavelength (μm)', fontsize=12)
    ax1.set_ylabel(f'Spectral radiance ({sed_thermal.unit})', fontsize=12)
    ax1.set_title('Jupiter-like Planet SED at 1 AU', fontsize=14)
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)
    
    # Ratio plot
    ratio = (sed_total / sed_thermal).value
    ax2.loglog(wl_thermal.to(u.um).value, ratio, linewidth=2, color='purple')
    ax2.set_xlabel('Wavelength (μm)', fontsize=12)
    ax2.set_ylabel('Flux ratio (total/thermal)', fontsize=12)
    ax2.axhline(1, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(True, which='both', alpha=0.3)
    
    # Add annotation
    ax2.text(0.5, 100, 'Reflected light\ndominates', fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.text(20, 1.5, 'Thermal\ndominates', fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('planet_thermal_vs_reflected.png', dpi=150)
    print("✓ Saved plot: planet_thermal_vs_reflected.png")
    plt.show()
    print()


def demo_albedo_effect():
    """Show effect of albedo on reflected light."""
    print("=" * 70)
    print("Demo 2: Albedo Effect")
    print("=" * 70)
    
    scene = helios.Scene(distance=10*u.pc)
    star = helios.Star(temperature=5700*u.K)
    scene.add(star)
    
    # Create planets with different albedos
    albedos = [0.0, 0.3, 0.6, 0.9]
    colors = ['darkred', 'orange', 'gold', 'lightblue']
    
    wavelengths = np.logspace(np.log10(0.3), np.log10(50), 150) * u.um
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for albedo, color in zip(albedos, colors):
        planet = helios.Planet(
            mass=1*u.M_jup,
            radius=1*const.R_jup,
            albedo=albedo,
            position=(1*u.AU, 0*u.AU)
        )
        # Manually set scene (since we're creating planets in loop)
        planet.scene = scene
        
        wl, sed = planet.sed(wavelengths=wavelengths, temperature=300*u.K, 
                            include_reflection=True)
        
        ax.loglog(wl.to(u.um).value, sed.value, 
                 label=f'Albedo = {albedo:.1f}', linewidth=2, color=color)
    
    ax.set_xlabel('Wavelength (μm)', fontsize=12)
    ax.set_ylabel('Spectral radiance (W m⁻² μm⁻¹ sr⁻¹)', fontsize=12)
    ax.set_title('Effect of Albedo on Planet SED', fontsize=14)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('planet_albedo_effect.png', dpi=150)
    print("✓ Saved plot: planet_albedo_effect.png")
    plt.show()
    print()


def demo_separation_effect():
    """Show effect of planet-star separation."""
    print("=" * 70)
    print("Demo 3: Orbital Separation Effect")
    print("=" * 70)
    
    scene = helios.Scene(distance=10*u.pc)
    star = helios.Star(temperature=5700*u.K)
    scene.add(star)
    
    # Planets at different separations
    separations = [0.1, 0.5, 1.0, 5.0] * u.AU
    colors = ['red', 'orange', 'green', 'blue']
    
    wavelengths = np.logspace(np.log10(0.3), np.log10(20), 100) * u.um
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for sep, color in zip(separations, colors):
        planet = helios.Planet(
            mass=1*u.M_jup,
            radius=1*const.R_jup,
            albedo=0.5,
            position=(sep, 0*u.AU)
        )
        planet.scene = scene
        
        wl, sed = planet.sed(wavelengths=wavelengths, temperature=300*u.K,
                            include_reflection=True)
        
        ax.loglog(wl.to(u.um).value, sed.value,
                 label=f'{sep.value:.1f} AU', linewidth=2, color=color)
    
    ax.set_xlabel('Wavelength (μm)', fontsize=12)
    ax.set_ylabel('Spectral radiance (W m⁻² μm⁻¹ sr⁻¹)', fontsize=12)
    ax.set_title('Effect of Orbital Separation on Planet SED', fontsize=14)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    # Add annotation
    ax.text(0.5, 1e15, 'Closer planets\nreflect more light',
           fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('planet_separation_effect.png', dpi=150)
    print("✓ Saved plot: planet_separation_effect.png")
    plt.show()
    print()


def demo_reflection_ratio_mode():
    """Demonstrate simple reflection_ratio mode."""
    print("=" * 70)
    print("Demo 4: Reflection Ratio Mode (Simple)")
    print("=" * 70)
    
    scene = helios.Scene(distance=10*u.pc)
    star = helios.Star(temperature=5700*u.K)
    scene.add(star)
    
    # Use reflection_ratio for quick tuning
    ratios = [0.0, 0.001, 0.01, 0.1]
    colors = ['black', 'blue', 'orange', 'red']
    
    wavelengths = np.logspace(np.log10(0.3), np.log10(30), 100) * u.um
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for ratio, color in zip(ratios, colors):
        planet = helios.Planet(
            mass=1*u.M_jup,
            reflection_ratio=ratio,
            position=(1*u.AU, 0*u.AU)
        )
        planet.scene = scene
        
        wl, sed = planet.sed(wavelengths=wavelengths, temperature=300*u.K,
                            include_reflection=True)
        
        label = 'Thermal only' if ratio == 0 else f'Reflection ratio = {ratio}'
        ax.loglog(wl.to(u.um).value, sed.value, label=label, linewidth=2, color=color)
    
    ax.set_xlabel('Wavelength (μm)', fontsize=12)
    ax.set_ylabel('Spectral radiance (W m⁻² μm⁻¹ sr⁻¹)', fontsize=12)
    ax.set_title('Reflection Ratio Parameter (Simple Mode)', fontsize=14)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('planet_reflection_ratio.png', dpi=150)
    print("✓ Saved plot: planet_reflection_ratio.png")
    plt.show()
    print()


def demo_hot_vs_cold_jupiter():
    """Compare hot Jupiter vs cold Jupiter."""
    print("=" * 70)
    print("Demo 5: Hot Jupiter vs Cold Jupiter")
    print("=" * 70)
    
    scene = helios.Scene(distance=10*u.pc)
    star = helios.Star(temperature=6000*u.K)
    scene.add(star)
    
    # Hot Jupiter: 0.05 AU, T=1500K, low albedo
    hot_jupiter = helios.Planet(
        mass=1*u.M_jup,
        radius=1.3*const.R_jup,
        albedo=0.1,
        position=(0.05*u.AU, 0*u.AU)
    )
    hot_jupiter.scene = scene
    
    # Cold Jupiter: 5 AU, T=100K, high albedo
    cold_jupiter = helios.Planet(
        mass=1*u.M_jup,
        radius=1*const.R_jup,
        albedo=0.6,
        position=(5*u.AU, 0*u.AU)
    )
    cold_jupiter.scene = scene
    
    wavelengths = np.logspace(np.log10(0.3), np.log10(100), 200) * u.um
    
    wl_hot, sed_hot = hot_jupiter.sed(wavelengths=wavelengths, temperature=1500*u.K,
                                     include_reflection=True)
    wl_cold, sed_cold = cold_jupiter.sed(wavelengths=wavelengths, temperature=100*u.K,
                                        include_reflection=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(wl_hot.to(u.um).value, sed_hot.value,
             label='Hot Jupiter (0.05 AU, 1500K)', linewidth=2, color='red')
    ax.loglog(wl_cold.to(u.um).value, sed_cold.value,
             label='Cold Jupiter (5 AU, 100K)', linewidth=2, color='blue')
    
    ax.set_xlabel('Wavelength (μm)', fontsize=12)
    ax.set_ylabel('Spectral radiance (W m⁻² μm⁻¹ sr⁻¹)', fontsize=12)
    ax.set_title('Hot vs Cold Jupiter Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    # Annotations
    ax.annotate('Hot Jupiter:\nthermal peak', xy=(2, 3e13), xytext=(0.5, 1e13),
               fontsize=9, arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.annotate('Cold Jupiter:\nreflected peak', xy=(0.5, 1e11), xytext=(1, 1e9),
               fontsize=9, arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    
    plt.tight_layout()
    plt.savefig('hot_vs_cold_jupiter.png', dpi=150)
    print("✓ Saved plot: hot_vs_cold_jupiter.png")
    plt.show()
    print()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("HELIOS Planet Reflected Light Demonstration")
    print("="*70 + "\n")
    
    demo_thermal_vs_reflected()
    demo_albedo_effect()
    demo_separation_effect()
    demo_reflection_ratio_mode()
    demo_hot_vs_cold_jupiter()
    
    print("="*70)
    print("✓ All demonstrations complete!")
    print("="*70)
