import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Optional, Tuple, Any
from ..core.context import Layer, Element, Context
from ..core.simulation import Wavefront
from astropy import constants as const

class CelestialBody(Element):
    """
    Base class for all celestial objects (stars, planets, zodiacal light, etc.).
    
    CelestialBody inherits from Element, making each celestial object an independent
    processing unit that can be combined in parallel within a Scene layer.
    
    Parameters
    ----------
    position : Tuple[astropy.Quantity, astropy.Quantity], optional
        (x, y) coordinates relative to the scene center.
        Can be angular (arcsec, mas) or physical (AU, m) if scene distance is defined.
        Default: (0 arcsec, 0 arcsec)
    name : str, optional
        Descriptive name for this celestial body.
    **kwargs : dict
        Additional parameters for specialized celestial body types.
    """
    def __init__(self, position: Tuple[u.Quantity, u.Quantity] = (0*u.arcsec, 0*u.arcsec), 
                 name: Optional[str] = None, **kwargs):
        super().__init__(name=name)
        self.position = position
        self.kwargs = kwargs

    def process(self, wavefront: Any, context: Context) -> Any:
        """
        Process the wavefront through this celestial body.
        
        For celestial bodies, processing typically means contributing to the
        scene's emission/reflection. This default implementation passes through
        the wavefront unchanged. Subclasses can override to add specific behaviors.
        
        Parameters
        ----------
        wavefront : Wavefront or None
            Input wavefront (may be None for scene initialization).
        context : Context
            Simulation context.
        
        Returns
        -------
        wavefront : Wavefront
            Processed wavefront.
        """
        # Default: pass-through (Scene layer handles combination)
        return wavefront

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
        ----------
        wavelengths : astropy.Quantity, optional
            Array of wavelengths. If None, a log-spaced array between wav_min and wav_max is created.
        temperature : astropy.Quantity, optional
            Temperature in Kelvin. If not provided, subclasses may supply defaults.
        beta : float
            Emissivity spectral index for modified blackbody (default 1.0).
        lambda0 : astropy.Quantity
            Reference wavelength for the emissivity power law.
        norm : float, optional
            Multiplicative normalization factor.

        Returns
        -------
        tuple
            (wavelengths, sed_values) where sed_values are spectral radiance in W/(m² m sr).
        """
        # Create wavelength grid if not provided
        if wavelengths is None:
            wavelengths = np.logspace(np.log10(wav_min.to(u.m).value), np.log10(wav_max.to(u.m).value), nwaves) * u.m

        # Default temperature fallback
        if temperature is None:
            temperature = 300 * u.K

        return modified_blackbody(wavelengths, temperature, beta=beta, lambda0=lambda0, norm=norm)

    def flux_at(self, wavelength: u.Quantity, **kwargs) -> u.Quantity:
        """
        Compute the spectral flux at a specific wavelength.

        This method evaluates the spectral energy distribution (SED) at a single
        wavelength by computing the SED on a fine grid around the target wavelength
        and interpolating to the exact value.

        Parameters
        ----------
        wavelength : astropy.Quantity
            Target wavelength (must be convertible to meters).
        **kwargs : dict
            Additional parameters passed to the sed() method (e.g., temperature, beta).

        Returns
        -------
        astropy.Quantity
            Spectral radiance at the given wavelength in W/(m² m sr).

        Examples
        --------
        >>> star = Star(temperature=5700*u.K)
        >>> flux = star.flux_at(550*u.nm)
        >>> print(flux)
        <Quantity ... W / (m2 m sr)>
        """
        # Convert wavelength to meters
        wl_target = wavelength.to(u.m).value
        
        # Create a fine wavelength grid around the target (±10% range, 100 points)
        wl_min = wl_target * 0.9
        wl_max = wl_target * 1.1
        wl_grid = np.linspace(wl_min, wl_max, 100) * u.m
        
        # Compute SED on this grid
        wl_array, sed_array = self.sed(wavelengths=wl_grid, **kwargs)
        
        # Interpolate to exact wavelength
        wl_values = wl_array.to(u.m).value
        sed_values = sed_array.to(u.W / (u.m ** 2 * u.m * u.sr)).value
        
        flux_interp = np.interp(wl_target, wl_values, sed_values)
        
        return flux_interp * (u.W / (u.m ** 2 * u.m * u.sr))

    def plot_sed(self,
                 wavelengths: Optional[u.Quantity] = None,
                 ax=None,
                 label: Optional[str] = None,
                 color: Optional[str] = None,
                 wl_unit: u.Unit = u.um,
                 loglog: bool = True,
                 **kwargs):
        """
        Plot the object's SED using the sed() method.

        Parameters
        ----------
        wavelengths : astropy.Quantity, optional
            Wavelength grid for plotting.
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes to plot into (created if None).
        label : str, optional
            Legend label (defaults to class name).
        color : str, optional
            Matplotlib color.
        wl_unit : astropy.units.Unit
            Wavelength unit for x-axis (default u.um).
        loglog : bool
            If True use log-log plot, otherwise linear.
        kwargs : dict
            Additional parameters forwarded to matplotlib plot() (alpha, linewidth, etc.)
            OR to sed() if they are SED parameters (temperature, beta, etc.).

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib Axes object.
        """
        # Separate SED kwargs from matplotlib kwargs
        sed_params = {'temperature', 'beta'}  # Known SED parameters
        sed_kwargs = {k: v for k, v in kwargs.items() if k in sed_params}
        plot_kwargs = {k: v for k, v in kwargs.items() if k not in sed_params}
        
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
            ax.loglog(wl_plot, sed_plot, label=lbl, color=color, **plot_kwargs)
        else:
            ax.plot(wl_plot, sed_plot, label=lbl, color=color, **plot_kwargs)

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
        """Return star SED using stellar temperature.
        
        Uses the star's temperature attribute by default.
        """
        return super().sed(wavelengths=wavelengths, temperature=self.temperature, **kwargs)

class Planet(CelestialBody):
    def __init__(self, 
                 mass: u.Quantity = 1*u.M_jup, 
                 radius: Optional[u.Quantity] = None,
                 temperature: u.Quantity = 300*u.K,
                 albedo: float = 0.3,
                 reflection_ratio: Optional[float] = None,
                 scene: Optional['Scene'] = None,
                 **kwargs):
        """
        Initialize a Planet object.
        
        Parameters
        ----------
        mass : astropy.Quantity
            Planet mass (default: 1 Jupiter mass).
        radius : astropy.Quantity, optional
            Planet radius. If None, estimated from mass assuming Jupiter-like density.
        temperature : astropy.Quantity
            Planet effective temperature for thermal emission (default: 300 K).
        albedo : float
            Geometric albedo for reflected light (default: 0.3, Earth-like).
        reflection_ratio : float, optional
            Direct override for reflected/emitted light ratio. If provided, this takes
            precedence over radius/albedo calculation. Useful for quick tuning without
            physical assumptions.
        scene : Scene, optional
            Parent scene containing stellar sources. Needed for reflected light calculation.
        **kwargs
            Additional arguments passed to CelestialBody (position, etc.).
        """
        self.mass = mass
        self.temperature = temperature
        self.albedo = albedo
        self.reflection_ratio = reflection_ratio
        self.scene = scene
        
        # Estimate radius from mass if not provided (assuming Jupiter-like density)
        if radius is None:
            # R ∝ M^(1/3) for gas giants (rough approximation)
            mass_ratio = (mass / const.M_jup).decompose().value
            self.radius = const.R_jup * (mass_ratio ** (1/3))
        else:
            self.radius = radius
            
        super().__init__(**kwargs)

    def sed(self, 
            wavelengths: Optional[u.Quantity] = None, 
            temperature: Optional[u.Quantity] = None,
            include_reflection: bool = True,
            **kwargs):
        """
        Return planet SED including thermal emission and reflected stellar light.
        
        The total SED is the sum of:
        1. Thermal emission: Blackbody at planet temperature (default 300 K)
        2. Reflected stellar light: Star SED × geometric albedo × (R_planet/d_star)²
        
        Parameters
        ----------
        wavelengths : astropy.Quantity, optional
            Wavelength grid for SED calculation.
        temperature : astropy.Quantity, optional
            Planet effective temperature (default: 300 K).
        include_reflection : bool
            If True and scene is available, add reflected stellar light (default: True).
        **kwargs
            Additional parameters for blackbody calculation (beta, lambda0, norm).
            
        Returns
        -------
        tuple
            (wavelengths, sed) where sed includes both thermal and reflected components.
        """
        # 1. Thermal emission component
        if temperature is None:
            temperature = self.temperature
        wl_thermal, sed_thermal = super().sed(wavelengths=wavelengths, temperature=temperature, **kwargs)
        
        # 2. Reflected light component
        if include_reflection and self.scene is not None:
            # Find stars in the scene
            stars = [obj for obj in self.scene.objects if isinstance(obj, Star)]
            
            if stars:
                # For simplicity, use the first (brightest) star
                # In a real system, you'd sum contributions from all stars
                star = stars[0]
                
                # Get stellar SED at same wavelengths
                wl_star, sed_star = star.sed(wavelengths=wl_thermal)
                
                # Calculate reflection scaling
                if self.reflection_ratio is not None:
                    # User-provided ratio (simple mode)
                    reflection_scale = self.reflection_ratio
                else:
                    # Physical calculation: geometric albedo × solid angle factor
                    # The reflected flux scales as: albedo × (R_planet / distance_to_star)²
                    # For now, use a geometric approximation
                    
                    # Get separation from star (assume star at origin)
                    px, py = self.position
                    if isinstance(px, u.Quantity) and px.unit.is_equivalent(u.m):
                        separation = np.sqrt((px**2 + py**2).to(u.m**2)).to(u.AU)
                    else:
                        # If position is angular, convert using scene distance
                        if self.scene.distance is not None:
                            sep_ang = np.sqrt(px**2 + py**2)
                            separation = (sep_ang * self.scene.distance).to(u.AU)
                        else:
                            separation = 1 * u.AU  # Fallback
                    
                    # Geometric albedo × (R_planet / separation)²
                    reflection_scale = self.albedo * (self.radius / separation)**2
                    reflection_scale = reflection_scale.decompose().value
                
                # Add reflected component
                sed_reflected = sed_star * reflection_scale
                sed_total = sed_thermal + sed_reflected
                
                return wl_thermal, sed_total
        
        # No reflection: return thermal only
        return wl_thermal, sed_thermal

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
        """Return exozodiacal dust SED as warm blackbody.
        
        Exozodiacal dust is modeled as warm dust with default temperature of 270 K.
        """
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
        """Return zodiacal dust SED as warm blackbody.
        
        Local zodiacal dust is modeled with default temperature of 270 K.
        """
        if temperature is None:
            temperature = 270 * u.K
        return modified_blackbody(wavelengths if wavelengths is not None else None, temperature, **kwargs)

class Scene(Layer):
    """
    Represents the astronomical scene containing stars, planets, zodiacal light, etc.
    
    Scene is a Layer that contains multiple CelestialBody elements. Each celestial
    body contributes to the scene's total emission/reflection independently.
    
    Parameters
    ----------
    distance : astropy.Quantity, optional
        Distance to the scene. Default: 10 pc. Used to convert between physical
        positions (AU) and angular positions (arcsec).
    name : str, optional
        Name of the scene for identification in diagrams.
    
    Attributes
    ----------
    elements : List[CelestialBody]
        List of celestial bodies in this scene (inherited from Layer).
    distance : astropy.Quantity
        Distance to the scene.
    
    Examples
    --------
    >>> scene = Scene(distance=10*u.pc, name="Proxima System")
    >>> scene.add(Star(temperature=5700*u.K, name="Proxima Centauri"))
    >>> scene.add(Planet(temperature=300*u.K, position=(1*u.AU, 0*u.AU), name="Proxima b"))
    """
    def __init__(self, distance: Optional[u.Quantity] = 10*u.pc, name: Optional[str] = None):
        super().__init__(name=name or "Scene")
        self.distance = distance
        # Note: self.elements is inherited from Layer

    def add(self, obj: CelestialBody):
        """
        Add a celestial body to the scene.
        
        Parameters
        ----------
        obj : CelestialBody
            The celestial object to add (Star, Planet, Zodiacal, etc.).
        
        Examples
        --------
        >>> scene = Scene(distance=10*u.pc)
        >>> scene.add(Star(temperature=5700*u.K))
        >>> scene.add(Planet(mass=1*u.M_jup, position=(1*u.AU, 0*u.AU)))
        """
        self.add_element(obj)
        # Automatically link planets to this scene for reflection calculation
        if isinstance(obj, Planet) and obj.scene is None:
            obj.scene = self
    
    @property
    def objects(self):
        """Backward compatibility: alias for elements."""
        return self.elements

    def process(self, wavefront: None, context: Context) -> Wavefront:
        """
        Generates the initial wavefront from the scene.
        
        The wavefront amplitude is scaled by:
        1. Distance: flux decreases as (d/d_ref)^-2 where d_ref = 10 pc
        2. Magnitude: flux decreases as 10^(-0.4 * magnitude)
        
        For reference: magnitude 0 star at 10 pc gives ~1e10 photons/s/m²/nm
        """
        # Create initial wavefront
        wf = Wavefront(wavelength=1.0*u.um, size=512)
        
        # Calculate total flux scaling from all objects
        flux_scaling = 0.0
        
        for obj in self.objects:
            if isinstance(obj, (Star, Planet)):
                # Distance scaling: flux ∝ 1/d²
                distance_factor = (10*u.pc / self.distance)**2  # Normalized to 10 pc
                
                # Magnitude scaling: flux ∝ 10^(-0.4*m)
                # Reference: m=0 at 10pc gives flux=1
                magnitude = getattr(obj, 'magnitude', 0.0)
                magnitude_factor = 10**(-0.4 * magnitude)
                
                # Combined scaling
                flux_scaling += float(distance_factor.value) * magnitude_factor
        
        # Apply flux scaling to wavefront field (amplitude scales as sqrt(flux))
        if flux_scaling > 0:
            wf.field = wf.field * np.sqrt(flux_scaling)
        
        return wf
    def render(self, npix: int = 256, fov: u.Quantity = 1.0 * u.arcsec, return_coords: bool = False):
        """Render the scene to a 2D intensity array in angular units and centered on (0,0).

        - `npix`: output image size (square)
        - `fov`: field of view (astropy Quantity, e.g., in arcsec)
        - `return_coords`: if True, return a tuple `(img, x, y)` where `x` and `y`
          are 1D `astropy.Quantity` arrays (arcsec) giving the pixel centers and
          the image is centered on (0,0).

        Returns either:
        - `img` (np.ndarray) when `return_coords` is False (legacy behavior), or
        - `(img, xq, yq)` when `return_coords` is True.

        The renderer places the origin (0,0) at the central pixel and uses
        arcsecond units for coordinates.
        """
        # prepare grid in arcsec (pixel centers)
        fov_val = float(fov.to(u.arcsec).value)
        xs = np.linspace(-fov_val / 2.0, fov_val / 2.0, npix)
        ys = xs.copy()
        xg, yg = np.meshgrid(xs, ys)
        img = np.zeros_like(xg, dtype=float)

        # helper to convert position to arcsec
        def pos_to_arcsec(pos):
            px, py = pos
            if isinstance(px, u.Quantity) and px.unit.is_equivalent(u.m) and self.distance is not None:
                x = (px / self.distance).to(u.arcsec, equivalencies=u.dimensionless_angles()).value
                y = (py / self.distance).to(u.arcsec, equivalencies=u.dimensionless_angles()).value
            elif isinstance(px, u.Quantity) and px.unit.is_equivalent(u.deg):
                x = px.to(u.arcsec).value
                y = py.to(u.arcsec).value
            elif isinstance(px, u.Quantity):
                x = px.to(u.arcsec).value
                y = py.to(u.arcsec).value
            else:
                # assume raw numbers are arcsec
                x = float(px)
                y = float(py)
            return x, y

        for obj in self.objects:
            # star at center
            if isinstance(obj, Star):
                x_arc, y_arc = 0.0, 0.0
                intensity = 1.0
            else:
                x_arc, y_arc = pos_to_arcsec(obj.position)
                intensity = getattr(obj, 'brightness', 0.1)

            # simple point rendering as Gaussian with sigma = 0.02 arcsec
            sigma = 0.02
            gauss = intensity * np.exp(-(((xg - x_arc) ** 2 + (yg - y_arc) ** 2) / (2 * sigma ** 2)))
            img += gauss

        # zodiacal/exozodiacal backgrounds
        for obj in self.objects:
            if isinstance(obj, (Zodiacal, ExoZodiacal)):
                b = getattr(obj, 'brightness', 0.1)
                if obj.radius is None:
                    # fill whole field
                    img += b * 0.1
                else:
                    try:
                        r_arc = float(obj.radius.to(u.arcsec).value)
                    except Exception:
                        r_arc = float(obj.radius)
                    mask = (xg ** 2 + yg ** 2) <= (r_arc ** 2)
                    img[mask] += b * 0.1

        # normalize
        m = img.max()
        if m > 0:
            img = img / float(m)

        if return_coords:
            # return image and coordinate axes as astropy Quantities in arcsec
            xq = (xs * u.arcsec)
            yq = (ys * u.arcsec)
            return img, xq, yq
        return img




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
