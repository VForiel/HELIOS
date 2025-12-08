
import matplotlib.pyplot as plt
from helios import Scene, Star, Planet, Zodiacal
import astropy.units as u

def verify_sed_plot():
    # Create scene with Star, Planet, Zodiacal
    scene = Scene(name="Test Scene")
    
    star = Star(temperature=5000*u.K, magnitude=5)
    scene.add(star)
    
    planet = Planet(mass=1*u.M_jup, orbit_radius=1*u.AU)
    scene.add(planet)
    
    zodi = Zodiacal(brightness=1.0)
    scene.add(zodi)
    
    # Calculate and Print SED stats
    wavelengths = None # Default
    for obj in scene.objects:
        try:
            wl, sed = obj.sed(wavelengths=wavelengths)
            print(f"Stats for {type(obj).__name__}: Min={sed.value.min():.2e}, Max={sed.value.max():.2e} {sed.unit}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"FAILED for {type(obj).__name__}: {e}")

    # Plot SED
    fig, ax = scene.plot_sed()
    
    # Check Legend
    legend = ax.get_legend()

    labels = [t.get_text() for t in legend.get_texts()]
    
    print(f"Legend Labels: {labels}")
    
    # Needs: Star, Planet, Zodiacal
    # Must NOT have: Total Scene
    
    checks = {
        "Star": any("Star" in l or "Sun" in l for l in labels), # Star name defaults to Star or Sun-like
        "Planet": any("Planet" in l for l in labels),
        "Zodiacal": any("Zodiacal" in l for l in labels),
        "No Total": not any("Total Scene" in l for l in labels)
    }
    
    if all(checks.values()):
        print("SUCCESS: Legend contains correct items and no Total.")
    else:
        print(f"FAILURE: Checks failed: {checks}")

    plt.savefig("verify_sed_plot_output.png")
    print("Saved to verify_sed_plot_output.png")

if __name__ == "__main__":
    verify_sed_plot()
