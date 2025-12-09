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
from typing import Tuple, Optional, Any, Union
import matplotlib.pyplot as _plt

from ..core.context import Layer, Element, Context, serialize_value, deserialize_value
from ..core.simulation import Wavefront, WavefrontArray
from .pupil import Pupil


class Collector(Element):
    __slots__ = ("pupil", "position", "size", "name")
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
    position : Tuple[float, float] or Tuple[astropy.Quantity, astropy.Quantity]
        (x, y) position in the aperture plane. If floats, assumed to be meters.
        Can also be astropy Quantities. For single telescopes, use (0, 0).
        For interferometric arrays, specify baseline coordinates.
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
        # Handle position units (convert to meters if Quantity)
        x, y = position
        x = x.to(u.m).value if hasattr(x, 'to') else float(x)
        y = y.to(u.m).value if hasattr(y, 'to') else float(y)
        self.position = (x, y)

        # Initialize Element with name
        default_name = f"Collector@({self.position[0]:.1f},{self.position[1]:.1f})"
        super().__init__(name=name or default_name)
        
        self.pupil = pupil
        
        # Infer size from pupil if not provided
        if size is None:
            if hasattr(pupil, 'diameter'):
                if isinstance(pupil.diameter, u.Quantity):
                    size = pupil.diameter
                else:
                    size = pupil.diameter * u.m
            else:
                size = 1.0 * u.m
        self.size = size
        
        self.metadata = metadata
        
    def to_dict(self) -> dict:
        """Serialize collector."""
        data = super().to_dict()
        data.update({
            "pupil": self.pupil.to_dict(),
            "position": serialize_value(self.position),
            "size": serialize_value(self.size),
            "metadata": serialize_value(self.metadata)
        })
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'Collector':
        """Create collector from dict."""
        name = data.get("name")
        pupil_data = data.get("pupil")
        pupil = Pupil.from_dict(pupil_data) if pupil_data else None
        
        # Position can be list/tuple from JSON
        pos_raw = deserialize_value(data.get("position", (0,0)))
        # Ensure tuple
        if isinstance(pos_raw, list):
            pos_raw = tuple(pos_raw)
            
        size = deserialize_value(data.get("size"))
        metadata = deserialize_value(data.get("metadata", {}))
        
        return cls(pupil=pupil, position=pos_raw, size=size, name=name, **metadata)
    
    def process(self, wavefront: Wavefront, auto_magnify: Optional[bool] = None) -> Any:
        """
        Process the wavefront through this collector's pupil.
        
        Applies the pupil transmission pattern to the wavefront field.
        Also updates the wavefront's pixel scale to match the collector size.
        
        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront to process.
        auto_magnify : bool, optional
            If True, resize wavefront to match collector size.
            If False, crop wavefront to collector size.
            If None, check sizes and warn if mismatch.
        
        Returns
        -------
        wavefront : Wavefront
            Wavefront with pupil mask applied and pixel scale updated.
        """
        
        # Handle auto_magnify logic
        if self.size is not None:
            # Ensure size is in meters
            collector_size = self.size
            if not isinstance(collector_size, u.Quantity):
                collector_size = collector_size * u.m
                
            wf_size = wavefront.width
            if not isinstance(wf_size, u.Quantity):
                wf_size = wf_size * u.m
            
            # Check if sizes match (with some tolerance)
            sizes_match = np.isclose(collector_size.to(u.m).value, wf_size.to(u.m).value, rtol=1e-5)
            
            if auto_magnify is None:
                if not sizes_match:
                    import warnings
                    warnings.warn(f"Wavefront size ({wf_size}) does not match Collector size ({collector_size}). "
                                  f"Resizing wavefront metadata to match collector (auto_magnify=True).")
                    auto_magnify = True
                else:
                    auto_magnify = False
            
            if auto_magnify:
                # Modify wavefront size metadata
                wavefront.width = collector_size
                wavefront.pixel_scale = (collector_size / wavefront.npix).to(u.m)
            else:
                # Crop wavefront
                wavefront = wavefront.crop(new_size=collector_size, center=(0*u.m, 0*u.m))
        
        # Use the last dimension for spatial size (assuming square)
        # field shape is typically (samples, height, width) or just (height, width)
        N = wavefront.shape[-1]
        
        # Update pixel scale based on collector size
        # The wavefront now represents the field at the pupil plane of this collector
        if self.size is not None:
            # Ensure size is in meters
            size_m = self.size.to(u.m).value if hasattr(self.size, 'to') else float(self.size)
            wavefront.pixel_scale = (size_m / N) * u.m

        mask = self.pupil.get_array(npix=N, soft=True)
        wavefront[:] = wavefront * mask

        wavefront._last_focal_length_m = float(self.pupil.focal_length.to(u.m).value)

        return wavefront

        wavefront._last_focal_length_m = float(self.pupil.focal_length.to(u.m).value)

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

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            "latitude": serialize_value(self.latitude),
            "longitude": serialize_value(self.longitude),
            "altitude": serialize_value(self.altitude)
        })
        return data

    @classmethod
    def from_dict(cls, data: dict, context: Optional[Context] = None) -> 'TelescopeArray':
        name = data.get("name")
        latitude = deserialize_value(data.get("latitude"))
        longitude = deserialize_value(data.get("longitude"))
        altitude = deserialize_value(data.get("altitude"))
        
        array = cls(name=name, latitude=latitude, longitude=longitude, altitude=altitude)
        
        # Restore collectors (elements)
        elements_data = data.get("elements", [])
        for elem_data in elements_data:
            type_name = elem_data.get("type", "Collector")
            # We assume it's a Collector if part of TelescopeArray
            if "Collector" in type_name: 
                try:
                    collector = Collector.from_dict(elem_data)
                    array.add_element(collector)
                except Exception as e:
                    print(f"Error restoring collector: {e}")
            else:
                 print(f"Unknown TelescopeArray element type: {type_name}")
                 
        return array
    
    @property
    def num_outputs(self) -> int:
        """Number of outputs produced by this array (one per collector)."""
        return len(self.elements)
    
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
        
        # Handle empty collector list gracefully
        if len(self.collectors) == 0:
            ax.text(0.5, 0.5, "No Collectors Defined", ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{self.name} - Empty Configuration')
            return ax

        baselines = self.get_baseline_array()
        
        if show_pupils:
            # Determine bounding box of all collectors
            min_x, max_x = float('inf'), float('-inf')
            min_y, max_y = float('inf'), float('-inf')
            
            for collector in self.collectors:
                x, y = collector.position
                size = collector.size.to(u.m).value if hasattr(collector.size, 'to') else float(collector.size)
                radius = size / 2.0
                min_x = min(min_x, x - radius)
                max_x = max(max_x, x + radius)
                min_y = min(min_y, y - radius)
                max_y = max(max_y, y + radius)
                
            # If no collectors or single point, handle defaults
            if min_x == float('inf'):
                cx, cy = 0.0, 0.0
                span = 10.0 # Default span
            else:
                cx = (min_x + max_x) / 2.0
                cy = (min_y + max_y) / 2.0
                span_x = max_x - min_x
                span_y = max_y - min_y
                span = max(span_x, span_y)
                if span == 0: span = collector.size.to(u.m).value * 2.0 # Fallback for single point
            
            # Add margin
            margin = span * 0.15
            canvas_span = span + 2 * margin
            
            # Setup canvas
            npix_canvas = int(canvas_span * 10) # 10 pixels/meter resolution base
            npix_canvas = max(512, min(npix_canvas, 2048)) # Clamp resolution
            
            # Pixel scale
            pixel_scale = canvas_span / npix_canvas
            
            canvas = np.zeros((npix_canvas, npix_canvas), dtype=float)
            
            # Canvas extent
            extent = [cx - canvas_span/2, cx + canvas_span/2, cy - canvas_span/2, cy + canvas_span/2]
            
            # Render each pupil onto the canvas
            for collector in self.collectors:
                pupil = collector.pupil
                x_pos, y_pos = collector.position
                
                # Render pupil
                diam_m = pupil.diameter.to(u.m).value if isinstance(pupil.diameter, u.Quantity) else float(pupil.diameter)
                diam = diam_m * pupil_scale 
                npix_pupil = int(diam / pixel_scale)
                npix_pupil = max(32, min(npix_pupil, 256))
                pupil_arr = pupil.get_array(npix=npix_pupil, soft=True)
                
                # Calculate pixel position on canvas relative to bottom-left corner of extent
                # content_x = x_pos - extent[0]
                
                x_pix = int((x_pos - extent[0]) / pixel_scale)
                y_pix = int((y_pos - extent[2]) / pixel_scale)
                
                half_npix = npix_pupil // 2
                
                # Insert pupil
                x_start_can = max(0, x_pix - half_npix)
                x_end_can = min(npix_canvas, x_pix + half_npix)
                y_start_can = max(0, y_pix - half_npix)
                y_end_can = min(npix_canvas, y_pix + half_npix)
                
                # Slices in pupil array
                # If canvas start > intended start, we cropped left/bottom side of pupil
                x_crop_start = max(0, -(x_pix - half_npix)) 
                y_crop_start = max(0, -(y_pix - half_npix))
                
                # Length to copy
                len_x = x_end_can - x_start_can
                len_y = y_end_can - y_start_can
                
                px_start = x_crop_start
                px_end = px_start + len_x
                py_start = y_crop_start
                py_end = py_start + len_y
                
                if len_x > 0 and len_y > 0:
                    canvas[y_start_can:y_end_can, x_start_can:x_end_can] = np.maximum(
                        canvas[y_start_can:y_end_can, x_start_can:x_end_can],
                        pupil_arr[py_start:py_end, px_start:px_end]
                    )

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
    
    def process(self, wavefront: Any) -> Any:
        """Apply telescope array aperture mask to wavefront.
        
        This method overrides the default Layer.process() to implement custom
        combination logic for telescope collectors.
        
        It supports two modes of operation:

        1. **Single Wavefront Input**:
            - The input wavefront is broadcasted to all collectors (copied).
            - Each collector applies its pupil mask to its copy.
            - Returns a WavefrontArray containing one wavefront per collector.

        2. **WavefrontArray/List Input** (Optimization Mode):
            - If input is a list of wavefronts (one per collector), each collector's
              pupil is applied to the corresponding wavefront.
            - This allows simulating large arrays without huge wavefront arrays,
              by processing each pupil in its own local coordinate system.
        
        Parameters
        ----------
        wavefront : Wavefront or WavefrontArray or List[Wavefront]
            Input wavefront(s) to process.
        
        Returns
        -------
        wavefront : WavefrontArray
            Wavefronts with aperture mask applied (one per collector).
        """

        # Check for list/WavefrontArray input
        is_list_input = isinstance(wavefront, list) or (hasattr(wavefront, '__iter__') and not hasattr(wavefront, 'field'))
        
        if not is_list_input:
            # If no input provided, generate input wavefronts from context
            if wavefront is None:
                wf_generated = self.context.get_input_wavefront(collectors=self.elements)
                # wf_generated is WavefrontArray; convert to list
                wf_list = [wf_generated[i] for i in range(len(self.elements))]
            else:
                # Broadcast single wavefront to all collectors
                wf_list = [wavefront.copy() for _ in range(len(self.elements))]
        else:
            wf_list = list(wavefront)
            # Handle mismatch length if needed (broadcast if len=1)
            if len(wf_list) == 1 and len(self.elements) > 1:
                 wf_list = [wf_list[0].copy() for _ in range(len(self.elements))]
        
        output_list = []
        locations = []
        for i, collector in enumerate(self.elements):
            if i < len(wf_list):
                wf = wf_list[i]
                # Ensure 3D field shape for downstream components even with single sample
                if wf.ndim == 2:
                    wf = wf[np.newaxis, ...]

                # Apply geometric phase (piston + tilt) for off-axis sources when input provided
                try:
                    wavelength = wf.wavelength if hasattr(wf, 'wavelength') else None
                except Exception:
                    wavelength = None
                if wavelength is not None and hasattr(wf, 'source_directions') and wf.source_directions is not None:
                    cx, cy = getattr(collector, 'position', (0.0, 0.0))
                    print(f"DEBUG: Applying phase shift. cx={cx}, cy={cy}, dirs={wf.source_directions}")
                    k = 2 * np.pi / wavelength.to(u.m).value
                    # Build local coordinate grid from wavefront size
                    try:
                        size_m = wf.width.to(u.m).value
                    except Exception:
                        size_m = float(wf.width)
                    npix = wf.npix
                    u_vec = np.linspace(-size_m/2, size_m/2, npix)
                    v_vec = np.linspace(-size_m/2, size_m/2, npix)
                    U, V = np.meshgrid(u_vec, v_vec)
                    dirs = wf.source_directions
                    for s in range(wf.shape[0]):
                        tx = u.Quantity(dirs[s][0], u.rad).to(u.rad).value
                        ty = u.Quantity(dirs[s][1], u.rad).to(u.rad).value
                        piston = k * (cx * tx + cy * ty)
                        tilt = k * (U * tx + V * ty)
                        phasor = np.exp(1j * (piston + tilt))
                        wf[s] *= phasor
                
                # Phase shift is now handled in Context.get_input_wavefront if collectors were passed.
                # If a generic wavefront was passed, we assume it's already correct or we might need
                # to re-implement piston here if needed for external wavefronts.
                # But for now, we assume Context handles it.
                
                # Apply collector pupil
                wf_processed = collector.process(wf)
                output_list.append(wf_processed)
                locations.append(collector.position)
        
        # If single collector, return single Wavefront for backward compatibility
        if len(output_list) == 1:
            return output_list[0]
        return WavefrontArray(output_list, locations=locations)


# Legacy aliases for backward compatibility
Telescope = TelescopeArray  # Single telescope is just TelescopeArray with one collector at (0,0)
Interferometer = TelescopeArray  # Interferometer is just TelescopeArray with multiple collectors at different positions
