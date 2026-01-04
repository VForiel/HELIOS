"""Telescope component representing a single aperture."""
import numpy as np
from astropy import units as u
from typing import Tuple, Optional, Any
import matplotlib.pyplot as _plt

from ...core.component import SamplingComponent
from ...core.pipeline import Pipeline
from ...utils.serialization import serialize_value, deserialize_value
from ...core.wavefront import Wavefront
from .pupil import Pupil


class Telescope(Pupil):
    __slots__ = ("position", "size", "name")
    """Represents a single telescope with pupil geometry and position.
    
    A Telescope is a Pupil that encapsulates the properties of an individual
    telescope aperture, including its pupil geometry (transmission pattern), 
    physical size, and position in the aperture plane (for interferometric arrays).
    
    Telescopes are grouped within a TelescopeArray component for parallel processing
    in interferometric configurations or co-located single telescope observations.
    
    Parameters
    ----------
    pupil : Pupil
        Source pupil geometry defining the aperture transmission pattern.
    position : Tuple[float, float] or Tuple[astropy.Quantity, astropy.Quantity]
        (x, y) position in the aperture plane. If floats, assumed to be meters.
        Can also be astropy Quantities. For single telescopes, use (0, 0).
        For interferometric arrays, specify baseline coordinates.
    size : astropy.Quantity, optional
        Diameter of the telescope aperture. If None, inferred from pupil.diameter.
    name : str, optional
        Descriptive name for this telescope (e.g., "UT1", "AT3").
    **metadata
        Additional metadata (e.g., mount type, coating, location).

    Examples
    --------
    >>> # Create a VLT UT telescope
    >>> pupil_vlt = Pupil.like('VLT')
    >>> ut1 = Telescope(pupil=pupil_vlt, position=(0, 0), size=8.2*u.m, name="UT1")
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

        # Initialize Pupil (parent)
        default_name = f"Telescope@({self.position[0]:.1f},{self.position[1]:.1f})"
        
        # Inherit properties from the source pupil
        super().__init__(diameter=pupil.diameter, focal_length=pupil.focal_length, name=name or default_name)
        
        # Copy elements (geometry)
        from copy import deepcopy
        self.elements = deepcopy(pupil.elements)
        
        # Infer size from pupil if not provided
        if size is None:
            if hasattr(pupil, 'diameter'):
                if isinstance(pupil.diameter, u.Quantity):
                    size = pupil.diameter
            else:
                size = 1.0 * u.m
        self.size = size
        
        self.metadata = metadata
        
    def to_dict(self) -> dict:
        """Serialize telescope."""
        # Use super().to_dict() from Pupil (which uses SamplingComponent/Component)
        # But Pupil.to_dict only saves diameter/focal_length/elements.
        # We need to add telescope specific data.
        data = super().to_dict()
        data.update({
            "position": serialize_value(self.position),
            "size": serialize_value(self.size),
            "metadata": serialize_value(self.metadata)
        })
        # Note: 'pupil' key is no longer needed since self IS the pupil
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'Telescope':
        """Create telescope from dict."""
        name = data.get("name")
        
        # Reconstruct base pupil properties
        # In current design, we need a 'pupil' object to passed to __init__
        # But we are deserializing a Telescope directly.
        # We can create a temporary Pupil object from the data to pass to __init__.
        # Or we can support creating Telescope without a source pupil in __init__?
        # The current __init__ REQUIRES a pupil object. This is a constraint.
        # Let's create a proxy Pupil from the dictionary data.
        
        diameter = deserialize_value(data.get("diameter", 1.0*u.m))
        focal_length = deserialize_value(data.get("focal_length", 1.0*u.m))
        proxy_pupil = Pupil(diameter=diameter, focal_length=focal_length)
        
        elements = data.get("elements", [])
        for e in elements:
             proxy_pupil.elements.append(deserialize_value(e))
        
        # Position
        pos_raw = deserialize_value(data.get("position", (0,0)))
        if isinstance(pos_raw, list):
            pos_raw = tuple(pos_raw)
            
        size = deserialize_value(data.get("size"))
        metadata = deserialize_value(data.get("metadata", {}))
        
        return cls(pupil=proxy_pupil, position=pos_raw, size=size, name=name, **metadata)
    
    def process(self, wavefront: Wavefront, pipeline: Optional['Pipeline'] = None, auto_magnify: Optional[bool] = None) -> Any:
        """
        Process the wavefront through this telescope's pupil.
        
        Applies the pupil transmission pattern to the wavefront field.
        Also updates the wavefront's pixel scale to match the telescope size.
        
        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront to process.
        pipeline : Pipeline, optional
            The simulation pipeline (unused in Telescope but required by Layer protocol).
        auto_magnify : bool, optional
            If True, resize wavefront to match telescope size.
            If False, crop wavefront to telescope size.
            If None, check sizes and warn if mismatch.
        
        Returns
        -------
        wavefront : Wavefront
            Wavefront with pupil mask applied and pixel scale updated.
        """
        # Backwards compatibility: if pipeline is passed as boolean, treat it as auto_magnify
        if isinstance(pipeline, bool) and auto_magnify is None:
            auto_magnify = pipeline
            pipeline = None
        
        # Handle auto_magnify logic
        if self.size is not None:
            # Ensure size is in meters
            telescope_size = self.size
            if not isinstance(telescope_size, u.Quantity):
                telescope_size = telescope_size * u.m
                
            wf_size = wavefront.width
            if not isinstance(wf_size, u.Quantity):
                wf_size = wf_size * u.m
            
            # Check if sizes match (with some tolerance)
            sizes_match = np.isclose(telescope_size.to(u.m).value, wf_size.to(u.m).value, rtol=1e-5)
            
            if auto_magnify is None:
                if not sizes_match:
                    import warnings
                    warnings.warn(f"Wavefront size ({wf_size}) does not match Telescope size ({telescope_size}). "
                                  f"Resizing wavefront metadata to match telescope (auto_magnify=True).")
                    auto_magnify = True
                else:
                    auto_magnify = False
            
            if auto_magnify:
                # Modify wavefront size metadata
                wavefront.width = telescope_size
                wavefront.pixel_scale = (telescope_size / wavefront.npix).to(u.m)
            else:
                # Crop wavefront
                wavefront = wavefront.crop(new_size=telescope_size, center=(0*u.m, 0*u.m))
        
        # Use the last dimension for spatial size (assuming square)
        # field shape is typically (samples, height, width) or just (height, width)
        N = wavefront.shape[-1]
        
        # Update pixel scale based on telescope size
        # The wavefront now represents the field at the pupil plane of this telescope
        if self.size is not None:
            # Ensure size is in meters
            size_m = self.size.to(u.m).value if hasattr(self.size, 'to') else float(self.size)
            wavefront.pixel_scale = (size_m / N) * u.m

        mask = self.get_array(npix=N, soft=True)
        wavefront[:] = wavefront * mask

        wavefront._last_focal_length_m = float(self.focal_length.to(u.m).value)

        return wavefront
    
    def __repr__(self):
        return f"Telescope(name='{self.name}', position={self.position}, size={self.size})"
    
    def _get_detailed_attributes(self) -> dict:
        """Return detailed attributes for Telescope."""
        attrs = {}
        attrs['position'] = f"({self.position[0]:.2f}, {self.position[1]:.2f}) m"
        attrs['size'] = str(self.size)
        attrs['pupil_diameter'] = f"{self.diameter:.2f} m"
        return attrs

    def plot(self, ax: Optional[_plt.Axes] = None, title: Optional[str] = None) -> _plt.Axes:
        """Plot the telescope pupil.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        title : str, optional
            Plot title.
            
        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = _plt.subplots(figsize=(6, 6))
            
        # Get pupil array
        N = 512
        pupil_arr = self.get_array(npix=N, soft=True)
        
        # Extent in meters
        size_m = self.size.to(u.m).value if hasattr(self.size, 'to') else float(self.size)
        extent = [-size_m/2, size_m/2, -size_m/2, size_m/2]
        
        ax.imshow(pupil_arr, origin='lower', extent=extent, cmap='gray', vmin=0, vmax=1)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(title or f"Pupil: {self.name} ({size_m:.1f}m)")
        ax.grid(False)
        return ax
