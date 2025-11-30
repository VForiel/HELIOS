"""Telescope array classes for single and interferometric observations.

This module provides classes for managing telescope configurations:
- Collector: Single telescope with pupil geometry and position (Element subclass)
- TelescopeArray: Array of one or more collectors with spatial positioning (Layer subclass)

TelescopeArray unifies single-telescope and interferometric observations:
- Single telescope: Add one collector at position (0, 0)
- Interferometer: Add multiple collectors at different baseline positions
"""
import numpy as np
from astropy import units as u
from typing import Tuple, Optional, Any
import matplotlib.pyplot as _plt

from ..core.context import Layer, Element, Context
from ..core.simulation import Wavefront
from .pupil import Pupil


class Collector(Element):
    """Represents a single telescope/collector with pupil geometry and position.
    
    A Collector is an Element that encapsulates the properties of an individual
    telescope aperture, including its pupil geometry (transmission pattern), 
    physical size, and position in the aperture plane (for interferometric arrays).
    
    Collectors are grouped within a TelescopeArray layer for parallel processing
    in interferometric configurations or co-located single telescope observations.
    
    Parameters
    ----------
    pupil : Pupil
        Pupil geometry defining the aperture transmission pattern.
    position : Tuple[float, float]
        (x, y) position in the aperture plane (meters). For single telescopes,
        use (0, 0). For interferometric arrays, specify baseline coordinates.
    size : astropy.Quantity, optional
        Diameter of the collector aperture. If None, inferred from pupil.diameter.
    name : str, optional
        Descriptive name for this collector (e.g., "UT1", "AT3").
    **metadata
        Additional metadata (e.g., mount type, coating, location).
    
    Attributes
    ----------
    pupil : Pupil
        The pupil geometry object.
    position : Tuple[float, float]
        Baseline coordinates in meters.
    size : astropy.Quantity
        Aperture diameter in meters.
    name : str
        Collector identifier (inherited from Element).
    metadata : dict
        Additional properties.
    
    Examples
    --------
    >>> # Create a VLT UT collector
    >>> pupil_vlt = Pupil.like('VLT')
    >>> ut1 = Collector(pupil=pupil_vlt, position=(0, 0), size=8.2*u.m, name="UT1")
    >>> print(ut1.name, ut1.size)
    UT1 8.2 m
    """
    def __init__(self, pupil: Pupil, position: Tuple[float, float] = (0, 0),
                 size: Optional[u.Quantity] = None, name: Optional[str] = None,
                 **metadata):
        # Initialize Element with name
        default_name = f"Collector@({position[0]:.1f},{position[1]:.1f})"
        super().__init__(name=name or default_name)
        
        self.pupil = pupil
        self.position = tuple(position)
        
        # Infer size from pupil if not provided
        if size is None:
            if hasattr(pupil, 'diameter'):
                size = pupil.diameter * u.m  # pupil.diameter stored as float in meters
            else:
                size = 1.0 * u.m
        self.size = size
        
        self.metadata = metadata
    
    def process(self, wavefront: Any, context: Context) -> Any:
        """
        Process the wavefront through this collector's pupil.
        
        Applies the pupil transmission pattern to the wavefront field.
        
        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront to process.
        context : Context
            Simulation context.
        
        Returns
        -------
        wavefront : Wavefront
            Wavefront with pupil mask applied.
        """
        if wavefront is None or not hasattr(wavefront, 'field'):
            return wavefront
        
        try:
            N = wavefront.field.shape[0]
            mask = self.pupil.get_array(npix=N, soft=True)
            wavefront.field = wavefront.field * mask.astype(wavefront.field.dtype)
        except Exception:
            # Fallback: try without soft edges
            try:
                mask = self.pupil.get_array(npix=N, soft=False)
                wavefront.field = wavefront.field * mask.astype(wavefront.field.dtype)
            except Exception:
                pass  # Skip if pupil can't be rendered
        
        return wavefront
    
    def __repr__(self):
        return f"Collector(name='{self.name}', position={self.position}, size={self.size})"
    
    def _get_detailed_attributes(self) -> dict:
        """Return detailed attributes for Collector."""
        attrs = {}
        attrs['position'] = f"({self.position[0]:.2f}, {self.position[1]:.2f}) m"
        attrs['size'] = str(self.size)
        if hasattr(self.pupil, 'diameter'):
            attrs['pupil_diameter'] = f"{self.pupil.diameter:.2f} m"
        return attrs


class TelescopeArray(Layer):
    """Array of one or more telescopes with pupil geometries and spatial positioning.
    
    This class unifies single-telescope and interferometric observations by managing
    an array of Collector elements with arbitrary spatial positions. It handles both:
    
    **Single telescope**: Add one collector at position (0, 0)
        - Used for conventional single-aperture observations
        - The pupil mask is applied at the center of the wavefront
        
    **Interferometer**: Add multiple collectors at different positions
        - Used for interferometric imaging with spatially separated apertures
        - Each pupil is positioned at its baseline coordinates (u,v plane)
        - Enables aperture synthesis and high angular resolution
    
    The spatial positioning is automatically handled: collectors at (0,0) are 
    treated as co-located, while non-zero positions create a dilute aperture array.
    
    Parameters
    ----------
    name : str, optional
        Name of the telescope configuration (e.g., "VLT-UT4", "VLTI", "CHARA").
    latitude : astropy.Quantity, optional
        Geographic latitude of the observatory (degrees).
    longitude : astropy.Quantity, optional
        Geographic longitude of the observatory (degrees).
    altitude : astropy.Quantity, optional
        Altitude above sea level (meters).
    
    Attributes
    ----------
    elements : List[Collector]
        List of Collector elements, each with pupil, position, and metadata (inherited from Layer).
    name : str
        Configuration name (inherited from Layer).
    latitude, longitude, altitude : astropy.Quantity
        Observatory geographic coordinates.
    
    Examples
    --------
    >>> # Single telescope (VLT UT4)
    >>> vlt = TelescopeArray(name="VLT-UT4", latitude=-24.6*u.deg, altitude=2635*u.m)
    >>> pupil_vlt = helios.Pupil.vlt()
    >>> vlt.add_collector(pupil=pupil_vlt, position=(0, 0), size=8.2*u.m)
    >>> 
    >>> # Interferometer (VLTI with 4 UTs)
    >>> vlti = TelescopeArray(name="VLTI")
    >>> for i, pos in enumerate([(0,0), (47,0), (47,47), (0,47)]):
    >>>     vlti.add_collector(pupil=pupil_vlt, position=pos, size=8.2*u.m, name=f"UT{i+1}")
    >>> 
    >>> # Check if this is interferometric (multiple non-colocated apertures)
    >>> print(f"Interferometric: {vlti.is_interferometric()}")
    """
    
    def __init__(self, name: Optional[str] = None,
                 latitude: u.Quantity = 0*u.deg, 
                 longitude: u.Quantity = 0*u.deg, 
                 altitude: u.Quantity = 0*u.m):
        super().__init__(name=name or "TelescopeArray")
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        # Note: self.elements is inherited from Layer
    
    @property
    def collectors(self):
        """Backward compatibility: alias for elements."""
        return self.elements
    
    def add_collector(self, pupil: Pupil, position: Tuple[float, float] = (0, 0), 
                     size: Optional[u.Quantity] = None, name: Optional[str] = None, **kwargs):
        """Add a collector to the telescope array.
        
        Parameters
        ----------
        pupil : Pupil
            Pupil geometry for this collector (defines aperture shape).
        position : Tuple[float, float], optional
            (x, y) baseline coordinates in meters. Default (0, 0) for single telescope.
            For interferometers, specify spatial separation between apertures.
        size : astropy.Quantity, optional
            Diameter of the collector. If None, inferred from pupil.diameter.
        name : str, optional
            Descriptive name for this collector (e.g., "UT1", "AT2").
        **kwargs
            Additional metadata (e.g., mount type, coating).
        
        Examples
        --------
        >>> array = TelescopeArray(name="CHARA")
        >>> pupil = helios.Pupil(diameter=1*u.m)
        >>> array.add_collector(pupil, position=(0, 0), size=1*u.m, name="S1")
        >>> array.add_collector(pupil, position=(100, 0), size=1*u.m, name="S2")
        """
        collector = Collector(pupil=pupil, position=position, size=size, name=name, **kwargs)
        self.add_element(collector)
    
    def is_interferometric(self) -> bool:
        """Check if this array has multiple non-colocated apertures (interferometric).
        
        Returns True if there are multiple collectors at different positions,
        False for single telescope or all collectors at (0, 0).
        """
        if len(self.collectors) <= 1:
            return False
        positions = {c.position for c in self.collectors}
        return len(positions) > 1
    
    def get_baseline_array(self) -> np.ndarray:
        """Return array of baseline vectors (u,v coordinates) in meters.
        
        Returns
        -------
        baselines : ndarray
            Array of shape (N, 2) where N is the number of collectors.
            Each row is (x, y) position in meters.
        """
        return np.array([c.position for c in self.collectors], dtype=float)
    
    def _get_detailed_attributes(self) -> dict:
        """Return detailed attributes for TelescopeArray."""
        attrs = {}
        attrs['num_collectors'] = len(self.elements)
        if self.is_interferometric():
            attrs['configuration'] = "Interferometric"
            baselines = self.get_baseline_array()
            if len(baselines) > 1:
                max_baseline = np.max(np.linalg.norm(baselines - baselines[0], axis=1))
                attrs['max_baseline'] = f"{max_baseline:.2f} m"
        else:
            attrs['configuration'] = "Single telescope"
        if self.latitude != 0*u.deg or self.longitude != 0*u.deg:
            attrs['latitude'] = str(self.latitude)
            attrs['longitude'] = str(self.longitude)
        if self.altitude != 0*u.m:
            attrs['altitude'] = str(self.altitude)
        return attrs
    
    @classmethod
    def vlti(cls, uts: bool = True) -> 'TelescopeArray':
        """Create a VLTI (Very Large Telescope Interferometer) configuration.
        
        The VLTI at ESO Paranal Observatory consists of 4 Unit Telescopes (UTs) 
        or up to 4 Auxiliary Telescopes (ATs) that can be positioned on various 
        stations. This method creates a realistic configuration based on GPS 
        coordinates converted to baseline positions via tangent plane projection.
        
        Parameters
        ----------
        uts : bool, optional
            If True (default), create configuration with 4 Unit Telescopes (8.2m).
            If False, create configuration with 4 Auxiliary Telescopes (1.8m).
        
        Returns
        -------
        vlti : TelescopeArray
            VLTI interferometric array with 4 collectors.
        
        Notes
        -----
        The baseline positions are derived from actual GPS coordinates of the 
        telescopes at Paranal:
        - GPS coordinates: longitude/latitude in degrees
        - Conversion: tangent plane projection with Earth radius + elevation (2635m)
        - Reference: PHISE project (https://github.com/VForiel/PHISE)
        
        UT configuration:
        - 4 Unit Telescopes of 8.2m diameter
        - Baselines ranging from ~47m to ~130m
        - Used for high-resolution interferometry
        
        AT configuration:
        - 4 Auxiliary Telescopes of 1.8m diameter  
        - Relocatable on a grid of stations
        - For this preset, we use a representative compact configuration
        
        Examples
        --------
        >>> # Create VLTI with Unit Telescopes
        >>> vlti_ut = TelescopeArray.vlti(uts=True)
        >>> print(f"VLTI UTs: {len(vlti_ut.collectors)} telescopes")
        >>> print(vlti_ut.get_baseline_array())
        
        >>> # Create VLTI with Auxiliary Telescopes
        >>> vlti_at = TelescopeArray.vlti(uts=False)
        >>> vlti_at.plot_array(show_pupils=True)
        """
        if uts:
            # VLTI Unit Telescopes (8.2m diameter)
            # Real baseline positions from GPS coordinates
            # Source: PHISE project telescope.py get_UT_telescopes()
            vlti = cls(name="VLTI-UTs", latitude=-24.627*u.deg, 
                      longitude=-70.404*u.deg, altitude=2635*u.m)
            pupil = Pupil.like('VLT')
            diameter = 8.2 * u.m
            
            # Baseline positions (GPS → tangent plane projection)
            positions = [
                (-16.14, 62.74),   # UT1
                (0.00, 0.00),      # UT2 (reference)
                (63.03, 53.37),    # UT3
                (101.99, 34.54)    # UT4
            ]
            
            for i, pos in enumerate(positions, 1):
                vlti.add_collector(pupil=pupil, position=pos, size=diameter, 
                                  name=f"UT{i}")
        else:
            # VLTI Auxiliary Telescopes (1.8m diameter)
            # Representative compact configuration
            vlti = cls(name="VLTI-ATs", latitude=-24.627*u.deg, 
                      longitude=-70.404*u.deg, altitude=2635*u.m)
            
            # Simple circular pupil for ATs
            pupil_at = Pupil(diameter=1.8*u.m)
            pupil_at.add_disk(radius=0.9*u.m)
            pupil_at.add_central_obscuration(diameter=0.2*u.m)
            diameter = 1.8 * u.m
            
            # Compact baseline configuration (example positions)
            positions = [
                (0.00, 0.00),      # AT1 (reference)
                (32.00, 0.00),     # AT2
                (16.00, 27.71),    # AT3
                (16.00, -27.71)    # AT4
            ]
            
            for i, pos in enumerate(positions, 1):
                vlti.add_collector(pupil=pupil_at, position=pos, size=diameter, 
                                  name=f"AT{i}")
        
        return vlti
    
    @classmethod
    def life(cls) -> 'TelescopeArray':
        """Create a LIFE (Large Interferometer For Exoplanets) configuration.
        
        LIFE is a proposed space-based nulling interferometer mission concept 
        for direct detection and characterization of exoplanets. It consists 
        of 4 free-flying collector spacecraft arranged in a planar formation.
        
        Returns
        -------
        life : TelescopeArray
            LIFE interferometric array with 4 collectors in space.
        
        Notes
        -----
        Since LIFE operates in space, we model it as being at the North Pole 
        (latitude=90°) looking vertically upward. This configuration ensures:
        - Perfect rotation of the array as Earth rotates
        - No atmospheric turbulence
        - Continuous observation geometry
        
        The baseline configuration is based on the LIFE mission concept with:
        - 4 collectors of 2m diameter each
        - Baselines: 100m to 608m (rectangular configuration)
        - **Centered array**: all collectors orbit around the central point (0,0)
        - All collectors are equidistant (~304m) from the array center
        - Planar arrangement in the XY plane
        
        Reference: PHISE project (https://github.com/VForiel/PHISE)
        
        Examples
        --------
        >>> # Create LIFE array
        >>> life = TelescopeArray.life()
        >>> print(f"LIFE: {len(life.collectors)} collectors")
        >>> life.plot_array(show_pupils=True)
        
        >>> # Check it's interferometric
        >>> print(f"Interferometric: {life.is_interferometric()}")
        """
        # Space-based: North Pole configuration for perfect Earth rotation tracking
        life = cls(name="LIFE", latitude=90*u.deg, longitude=0*u.deg, 
                  altitude=0*u.m)  # altitude=0 for space (not applicable)
        
        # Simple circular pupil for LIFE collectors (2m diameter)
        pupil_life = Pupil(diameter=2.0*u.m)
        pupil_life.add_disk(radius=1.0*u.m)
        diameter = 2.0 * u.m
        
        # LIFE baseline configuration (from PHISE get_LIFE_telescopes)
        # Centered configuration: all telescopes orbit around central point (0,0)
        # Original PHISE positions centered to ensure (0,0) is the array center
        positions_original = [
            (0, 0),        # Collector 1
            (100, 0),      # Collector 2
            (0, 600),      # Collector 3
            (100, 600)     # Collector 4
        ]
        
        # Center the array: compute centroid and shift all positions
        centroid_x = sum(p[0] for p in positions_original) / len(positions_original)
        centroid_y = sum(p[1] for p in positions_original) / len(positions_original)
        
        positions = [
            (x - centroid_x, y - centroid_y) for x, y in positions_original
        ]
        
        for i, pos in enumerate(positions, 1):
            life.add_collector(pupil=pupil_life, position=pos, size=diameter, 
                              name=f"LIFE-{i}")
        
        return life
    
    def plot_array(self, ax: Optional[_plt.Axes] = None, show_pupils: bool = True,
                  pupil_scale: float = 1.0) -> _plt.Axes:
        """Plot the telescope array configuration.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        show_pupils : bool
            If True, render individual pupil shapes at each baseline position.
        pupil_scale : float
            Scale factor for pupil rendering (1.0 = actual size).
        
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plot.
        """
        if ax is None:
            fig, ax = _plt.subplots(figsize=(8, 8))
        
        baselines = self.get_baseline_array()
        
        if show_pupils and len(self.collectors) > 0:
            # Determine plot extent and resolution
            max_extent = 0
            for collector in self.collectors:
                x, y = collector.position
                size = collector.size.to(u.m).value if hasattr(collector.size, 'to') else float(collector.size)
                max_extent = max(max_extent, abs(x) + size, abs(y) + size)
            
            # Create a large canvas to hold all pupils
            margin = max_extent * 0.15  # 15% margin
            canvas_size = 2 * (max_extent + margin)
            npix_canvas = int(canvas_size * 10)  # 10 pixels/meter
            canvas = np.zeros((npix_canvas, npix_canvas), dtype=float)
            
            # Pixel scale (meters per pixel)
            pixel_scale = canvas_size / npix_canvas
            
            # Render each pupil onto the canvas
            for collector in self.collectors:
                pupil = collector.pupil
                x_pos, y_pos = collector.position
                
                # Render pupil at appropriate resolution
                diam = pupil.diameter * pupil_scale
                npix_pupil = int(diam / pixel_scale)
                npix_pupil = max(32, min(npix_pupil, 256))  # Clamp to reasonable range
                pupil_arr = pupil.get_array(npix=npix_pupil, soft=True)
                
                # Calculate pixel position on canvas
                center_m = max_extent + margin
                x_pix = int((x_pos + center_m) / pixel_scale)
                y_pix = int((y_pos + center_m) / pixel_scale)
                
                # Half-size in pixels
                half_npix = npix_pupil // 2
                
                # Insert pupil into canvas
                x_start = max(0, x_pix - half_npix)
                x_end = min(npix_canvas, x_pix + half_npix)
                y_start = max(0, y_pix - half_npix)
                y_end = min(npix_canvas, y_pix + half_npix)
                
                # Corresponding slice in pupil array
                px_start = max(0, -(x_pix - half_npix))
                px_end = px_start + (x_end - x_start)
                py_start = max(0, -(y_pix - half_npix))
                py_end = py_start + (y_end - y_start)
                
                # Overlay pupil (take maximum to avoid overwriting)
                canvas[y_start:y_end, x_start:x_end] = np.maximum(
                    canvas[y_start:y_end, x_start:x_end],
                    pupil_arr[py_start:py_end, px_start:px_end]
                )
            
            # Display the full canvas
            extent = [-center_m, center_m, -center_m, center_m]
            ax.imshow(canvas, origin='lower', cmap='gray', extent=extent, alpha=0.9)
        else:
            # Simple scatter plot
            ax.scatter(baselines[:, 0], baselines[:, 1], s=100, c='blue', 
                      marker='o', edgecolors='black', linewidth=1.5)
        
        ax.set_xlabel('Baseline x (m)')
        ax.set_ylabel('Baseline y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Title indicates if interferometric
        mode = "Interferometric" if self.is_interferometric() else "Single Telescope"
        ax.set_title(f'{self.name} - {mode} ({len(self.collectors)} collector{"s" if len(self.collectors) > 1 else ""})')
        
        return ax
    
    def process(self, wavefront: Wavefront, context: Context) -> Wavefront:
        """Apply telescope array aperture mask to wavefront.
        
        This method overrides the default Layer.process() to implement custom
        combination logic for telescope collectors:
        - If all collectors are at (0, 0): Pupils are combined multiplicatively (co-phased)
        - If collectors have different positions: Pupils are placed at their baseline positions
        
        For true interferometric fringe formation and beam combination, use dedicated
        photonics layers (PhotonicChip, TOPS, etc.) after this layer.
        
        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront to process.
        context : Context
            Simulation context (unused in this implementation).
        
        Returns
        -------
        wavefront : Wavefront
            Wavefront with aperture mask applied.
        """
        try:
            N = wavefront.field.shape[0]
        except Exception:
            return wavefront
        
        if len(self.elements) == 0:  # Use self.elements instead of self.collectors
            return wavefront
        
        # Determine if we need spatial positioning
        is_interferometric = self.is_interferometric()
        
        if not is_interferometric:
            # Single telescope or co-located: combine multiplicatively at center
            total_mask = np.ones((N, N), dtype=float)
            for collector in self.elements:  # Use self.elements
                if isinstance(collector.pupil, Pupil):
                    try:
                        mask = collector.pupil.get_array(npix=N, soft=True)
                    except Exception:
                        mask = collector.pupil.get_array(npix=N, soft=False)
                    total_mask = total_mask * mask
            wavefront.field = wavefront.field * total_mask.astype(wavefront.field.dtype)
        else:
            # Interferometric: position each pupil at its baseline coordinates
            combined_mask = np.zeros((N, N), dtype=float)
            
            # Determine array extent
            baselines = self.get_baseline_array()
            max_extent = np.max(np.abs(baselines)) if len(baselines) > 0 else 1.0
            for collector in self.elements:  # Use self.elements
                size_m = collector.size.to(u.m).value if hasattr(collector.size, 'to') else float(collector.size)
                max_extent = max(max_extent, size_m)
            
            # Pixel scale: meters per pixel
            pixel_scale = 2.0 * max_extent / float(N)
            
            # Render each collector pupil at its position
            for collector in self.elements:  # Use self.elements
                pupil = collector.pupil
                x, y = collector.position
                
                # Render pupil
                pupil_arr = pupil.get_array(npix=N, soft=True)
                
                # Compute shift in pixels
                shift_x = int(x / pixel_scale)
                shift_y = int(y / pixel_scale)
                
                # Shift pupil to baseline position
                shifted_pupil = np.roll(pupil_arr, shift=(shift_y, shift_x), axis=(0, 1))
                
                # Add to combined mask
                combined_mask = np.maximum(combined_mask, shifted_pupil)
            
            # Apply to wavefront
            wavefront.field = wavefront.field * combined_mask.astype(wavefront.field.dtype)
        
        return wavefront


# Legacy aliases for backward compatibility
Telescope = TelescopeArray  # Single telescope is just TelescopeArray with one collector at (0,0)
Interferometer = TelescopeArray  # Interferometer is just TelescopeArray with multiple collectors at different positions
