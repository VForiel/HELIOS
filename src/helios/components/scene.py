import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Optional, Tuple
from ..core.context import Layer, Context
from ..core.simulation import Wavefront
from astropy import constants as const

class CelestialBody:
    """
    Base class for all celestial objects.
    """
    def __init__(self, position: Tuple[u.Quantity, u.Quantity] = (0*u.arcsec, 0*u.arcsec), **kwargs):
        """
        position: (x, y) coordinates relative to the scene center.
                  Can be angular (arcsec) or physical (AU) if scene distance is defined.
        """
        self.position = position
        self.kwargs = kwargs

    def sed(self,
            wavelengths: Optional[u.Quantity] = None,
            wav_min: u.Quantity = 0.1 * u.um,
            wav_max: u.Quantity = 100 * u.um,
            nwaves: int = 200,
            temperature: Optional[u.Quantity] = None,
            beta: float = 1.0,
            lambda0: u.Quantity = 100 * u.um,
            norm: Optional[float] = None):
        """
        Return a modified blackbody SED for this object.

        Parameters
        - wavelengths: optional astropy Quantity array of wavelengths. If None, a log-spaced
          array between `wav_min` and `wav_max` is created.
        - temperature: astropy Quantity (K). If not provided, subclasses may supply defaults.
        - beta: emissivity spectral index for modified blackbody (default 1.0).
        - lambda0: reference wavelength for the emissivity power law.
        - norm: optional multiplicative normalization factor.

        Returns (wavelengths, sed_values) where sed_values are spectral radiance in
        units W / (m2 m sr) (i.e., per unit wavelength).
        """
        # Create wavelength grid if not provided
        if wavelengths is None:
            wavelengths = np.logspace(np.log10(wav_min.to(u.m).value), np.log10(wav_max.to(u.m).value), nwaves) * u.m

        # Default temperature fallback
        if temperature is None:
            temperature = 300 * u.K

        return modified_blackbody(wavelengths, temperature, beta=beta, lambda0=lambda0, norm=norm)

    def plot_sed(self,
                 wavelengths: Optional[u.Quantity] = None,
                 ax=None,
                 label: Optional[str] = None,
                 color: Optional[str] = None,
                 wl_unit: u.Unit = u.um,
                 loglog: bool = True,
                 **sed_kwargs):
        """
        Plot the object's SED using the `sed()` method.

        Parameters
        - wavelengths: optional wavelength grid (astropy.Quantity)
        - ax: matplotlib Axes to plot into (created if None)
        - label: legend label (defaults to class name)
        - color: matplotlib color
        - wl_unit: wavelength unit for x-axis (default `u.um`)
        - loglog: if True use log-log plot, otherwise linear
        - sed_kwargs: forwarded to `sed()` (temperature, beta, etc.)

        Returns the matplotlib Axes.
        """
        wl, sed = self.sed(wavelengths=wavelengths, **sed_kwargs)

        # Convert for plotting: wavelength in wl_unit, sed per wl_unit
        wl_plot = wl.to(wl_unit).value
        sed_unit = u.W / (u.m ** 2 * wl_unit * u.sr)
        sed_plot = sed.to(sed_unit).value

        import matplotlib.pyplot as _plt
        if ax is None:
            fig, ax = _plt.subplots(figsize=(6, 4))

        lbl = label if label is not None else type(self).__name__
        if loglog:
            ax.loglog(wl_plot, sed_plot, label=lbl, color=color)
        else:
            ax.plot(wl_plot, sed_plot, label=lbl, color=color)

        ax.set_xlabel(f'Wavelength ({wl_unit})')
        ax.set_ylabel(f'Spectral radiance (W m-2 {wl_unit}^-1 sr-1)')
        ax.grid(True, which='both', ls='--', alpha=0.5)
        ax.legend()
        return ax

class Star(CelestialBody):
    def __init__(self, temperature: u.Quantity = 5778*u.K, magnitude: float = 4.83, mass: u.Quantity = 1*u.M_sun, **kwargs):
        self.temperature = temperature
        self.magnitude = magnitude
        self.mass = mass
        super().__init__(**kwargs)

    def sed(self, wavelengths: Optional[u.Quantity] = None, **kwargs):
        # Use star temperature by default
        return super().sed(wavelengths=wavelengths, temperature=self.temperature, **kwargs)

class Planet(CelestialBody):
    def __init__(self, mass: u.Quantity = 1*u.M_jup, **kwargs):
        self.mass = mass
        super().__init__(**kwargs)

    def sed(self, wavelengths: Optional[u.Quantity] = None, temperature: Optional[u.Quantity] = None, **kwargs):
        # Planets are approximated as cool blackbodies; default ~300 K if not provided
        if temperature is None:
            temperature = 300 * u.K
        return super().sed(wavelengths=wavelengths, temperature=temperature, **kwargs)

class ExoZodiacal(CelestialBody):
    def __init__(self, brightness: float = 1.0, radius: Optional[u.Quantity] = None, **kwargs):
        """
        brightness: relative surface brightness (arbitrary units)
        radius: optional angular radius (e.g., in arcsec) to draw the exozodi disk; if None, fill view
        """
        self.brightness = brightness
        self.radius = radius
        super().__init__(**kwargs)

    def sed(self, wavelengths: Optional[u.Quantity] = None, temperature: Optional[u.Quantity] = None, **kwargs):
        # Exozodi: warm dust, default temperature ~270 K
        if temperature is None:
            temperature = 270 * u.K
        return modified_blackbody(wavelengths if wavelengths is not None else None, temperature, **kwargs)

class Zodiacal(CelestialBody):
    def __init__(self, brightness: float = 1.0, radius: Optional[u.Quantity] = None, **kwargs):
        """
        Local zodiacal light (solar system dust) as a diffuse background.
        brightness: relative surface brightness (arbitrary units)
        radius: optional angular radius to draw the zodiacal component; if None, fill view
        """
        self.brightness = brightness
        self.radius = radius
        super().__init__(**kwargs)

    def sed(self, wavelengths: Optional[u.Quantity] = None, temperature: Optional[u.Quantity] = None, **kwargs):
        # Local zodiacal dust ~270 K by default
        if temperature is None:
            temperature = 270 * u.K
        return modified_blackbody(wavelengths if wavelengths is not None else None, temperature, **kwargs)

class Scene(Layer):
    """
    Represents the astronomical scene containing stars, planets, etc.
    """
    def __init__(self, distance: Optional[u.Quantity] = 10*u.pc):
        self.distance = distance
        self.objects = []
        super().__init__()

    def add(self, obj: CelestialBody):
        self.objects.append(obj)

    def process(self, wavefront: None, context: Context) -> Wavefront:
        """
        Generates the initial wavefront from the scene.
        """
        # Placeholder: Create a wavefront
        wf = Wavefront(wavelength=1.0*u.um, size=512) 
        return wf




def modified_blackbody(wavelengths: Optional[u.Quantity], temperature: u.Quantity, beta: float = 1.0,
                       lambda0: u.Quantity = 100 * u.um, norm: Optional[float] = None):
    """
    Compute a modified blackbody spectrum B_lambda(λ, T) * (λ / lambda0)^{-beta}.

    wavelengths: astropy Quantity array (if None, a default grid is created)
    temperature: astropy Quantity (K)
    Returns (wavelengths, sed) with sed in W / (m2 m sr)
    """
    # If wavelengths not provided, create a default grid
    if wavelengths is None:
        wavelengths = np.logspace(np.log10((0.1 * u.um).to(u.m).value), np.log10((100 * u.um).to(u.m).value), 200) * u.m

    # Ensure proper units
    wavelengths = wavelengths.to(u.m)
    T = temperature.to(u.K)

    h = const.h
    c = const.c
    kB = const.k_B

    wl = wavelengths
    # Planck function per unit wavelength B_lambda(λ, T)
    exponent = (h * c) / (wl * kB * T)
    # Avoid overflow warnings by using np.exp with values
    B = (2 * h * c ** 2) / (wl ** 5) / (np.expm1(exponent))
    # Ensure B has radiance units (per steradian)
    B = B * (1.0 / u.sr)

    # Modified emissivity
    emissivity = (wl / lambda0.to(u.m)) ** (-beta)

    sed = (B * emissivity).to(u.W / (u.m ** 2 * u.m * u.sr))

    if norm is not None:
        sed = sed * float(norm)

    return wavelengths, sed

    def plot(self):
        """
        Plot the scene.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.set_xlabel('RA offset [arcsec]')
        ax.set_ylabel('Dec offset [arcsec]')
        ax.invert_xaxis() # RA increases to the left
        # Ensure grid is drawn below patches/points
        ax.set_axisbelow(True)
        # Collect positions (in arcsec) so we can center the view on (0,0)
        xs = []
        ys = []
        # Collect diffuse disk-like components (zodiacal/exozodiacal) to draw after limits are known
        disks = []

        for obj in self.objects:
            # If object is a diffuse zodiacal/exozodiacal component, register for later drawing
            if isinstance(obj, (Zodiacal, ExoZodiacal)):
                disks.append(obj)
                # do not scatter-plot a point for diffuse background
                continue
            # Determine marker size, style, and position
            if isinstance(obj, Star):
                marker = '*'
                color = 'gold'
                # Log scale size based on mass (arbitrary scaling for viz)
                size = 200 * np.log10(obj.mass.to(u.M_sun).value + 1) + 50
                label = f'Star ({obj.magnitude} mag)'
            elif isinstance(obj, Planet):
                marker = 'o'
                color = 'blue'
                size = 100 * np.log10(obj.mass.to(u.M_jup).value + 1) + 20
                label = f'Planet ({obj.mass})'
            else:
                marker = '.'
                color = 'gray'
                size = 10
                label = 'Object'

            # Handle position units for ALL objects
            px, py = obj.position
            # Per spec: Stars are treated as being at the scene center (0,0)
            if isinstance(obj, Star):
                px, py = (0*u.arcsec, 0*u.arcsec)
            if px.unit.is_equivalent(u.m) and self.distance is not None:
                x = (px / self.distance).to(u.arcsec, equivalencies=u.dimensionless_angles())
                y = (py / self.distance).to(u.arcsec, equivalencies=u.dimensionless_angles())
            elif px.unit.is_equivalent(u.deg):
                x = px.to(u.arcsec)
                y = py.to(u.arcsec)
            else:
                    # Fallback or error handling if needed, but for now assuming one of the above
                x = px.to(u.arcsec)
                y = py.to(u.arcsec)

            ax.scatter(x.value, y.value, s=size, marker=marker, c=color, label=label)

            # store numeric arcsec values for later range computation
            try:
                xs.append(float(x.to(u.arcsec).value))
                ys.append(float(y.to(u.arcsec).value))
            except Exception:
                xs.append(float(x.value))
                ys.append(float(y.value))

        # Determine symmetric limits around (0,0)
        if len(xs) == 0 or len(ys) == 0:
            # No objects: choose a sensible default extent (1 arcsec)
            lim = 1.0
        else:
            max_abs = max(max(np.abs(xs)), max(np.abs(ys)))
            # Scale to be only 10% larger than the furthest object
            lim = float(max_abs) * 1.1
            # If all objects are exactly at the center (max_abs == 0), fall back to 1 arcsec
            if lim == 0:
                lim = 1.0

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        # Draw diffuse zodiacal/exozodiacal components as filled, low-alpha disks
        for disk in disks:
            # determine radius in arcsec: use provided radius or fill the view
            if disk.radius is not None:
                try:
                    r_arcsec = float(disk.radius.to(u.arcsec).value)
                except Exception:
                    r_arcsec = float(disk.radius)
            else:
                r_arcsec = lim

            # distinct colors for zodi vs exozodi
            if isinstance(disk, Zodiacal):
                color = 'sandybrown'
                label = 'Zodiacal'
            else:
                color = 'lightsteelblue'
                label = 'ExoZodiacal'

            # brightness -> alpha mapping (very rough, clamp)
            alpha = float(np.clip(disk.brightness * 0.2, 0.03, 0.8))
            circ = Circle((0.0, 0.0), r_arcsec, color=color, alpha=alpha, zorder=0, label=label)
            ax.add_patch(circ)

        # Handle legend (remove duplicates) AFTER drawing disks so they appear in legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.title(f"Scene (Distance: {self.distance})")
        return fig, ax


# Attach a module-level implementation of plot to ensure the method exists
def _scene_plot(self):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlabel('RA offset [arcsec]')
    ax.set_ylabel('Dec offset [arcsec]')
    ax.invert_xaxis()
    ax.set_axisbelow(True)
    xs = []
    ys = []
    disks = []

    for obj in self.objects:
        if isinstance(obj, (Zodiacal, ExoZodiacal)):
            disks.append(obj)
            continue

        if isinstance(obj, Star):
            marker = '*'
            color = 'gold'
            size = 200 * np.log10(obj.mass.to(u.M_sun).value + 1) + 50
            label = f'Star ({obj.magnitude} mag)'
        elif isinstance(obj, Planet):
            marker = 'o'
            color = 'blue'
            size = 100 * np.log10(obj.mass.to(u.M_jup).value + 1) + 20
            label = f'Planet ({obj.mass})'
        else:
            marker = '.'
            color = 'gray'
            size = 10
            label = 'Object'

        px, py = obj.position
        if isinstance(obj, Star):
            px, py = (0*u.arcsec, 0*u.arcsec)
        if px.unit.is_equivalent(u.m) and self.distance is not None:
            x = (px / self.distance).to(u.arcsec, equivalencies=u.dimensionless_angles())
            y = (py / self.distance).to(u.arcsec, equivalencies=u.dimensionless_angles())
        elif px.unit.is_equivalent(u.deg):
            x = px.to(u.arcsec)
            y = py.to(u.arcsec)
        else:
            x = px.to(u.arcsec)
            y = py.to(u.arcsec)

        ax.scatter(x.value, y.value, s=size, marker=marker, c=color, label=label)

        try:
            xs.append(float(x.to(u.arcsec).value))
            ys.append(float(y.to(u.arcsec).value))
        except Exception:
            xs.append(float(x.value))
            ys.append(float(y.value))

    if len(xs) == 0 or len(ys) == 0:
        lim = 1.0
    else:
        max_abs = max(max(np.abs(xs)), max(np.abs(ys)))
        lim = float(max_abs) * 1.1
        if lim == 0:
            lim = 1.0

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    for disk in disks:
        if disk.radius is not None:
            try:
                r_arcsec = float(disk.radius.to(u.arcsec).value)
            except Exception:
                r_arcsec = float(disk.radius)
        else:
            r_arcsec = lim

        if isinstance(disk, Zodiacal):
            color = 'sandybrown'
            label = 'Zodiacal'
        else:
            color = 'lightsteelblue'
            label = 'ExoZodiacal'

        alpha = float(np.clip(disk.brightness * 0.2, 0.03, 0.8))
        circ = Circle((0.0, 0.0), r_arcsec, color=color, alpha=alpha, zorder=0, label=label)
        ax.add_patch(circ)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title(f"Scene (Distance: {self.distance})")
    return fig, ax


# Bind to class
Scene.plot = _scene_plot

def test_scene_creation():
    scene = Scene(distance=10*u.pc)
    star = Star(temperature=5000*u.K, magnitude=5, position=(0*u.AU, 0*u.AU))
    planet = Planet(mass=1*u.M_jup, position=(1*u.AU, 0*u.AU))
    scene.add(star)
    scene.add(planet)
    assert len(scene.objects) == 2
    assert scene.distance == 10*u.pc
    # Test defaults
    default_star = Star()
    assert default_star.temperature == 5778*u.K
    assert default_star.magnitude == 4.83
    assert default_star.mass == 1*u.M_sun

    default_planet = Planet()
    assert default_planet.mass == 1*u.M_jup

    default_scene = Scene()
    assert default_scene.distance == 10*u.pc

    # scene.plot() # Uncomment to test plotting visually

if __name__ == "__main__":
    test_scene_creation()
    print("Scene tests passed.")
