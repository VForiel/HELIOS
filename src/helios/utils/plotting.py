import numpy as np
from astropy import units as u
from typing import Tuple, Union, List

def get_smart_extent(shape: Tuple[int, ...], pixel_scale: u.Quantity):
    """
    Determine plot extent and axis labels with appropriate units.
    
    Parameters
    ----------
    shape : tuple
        (height, width) of the array.
    pixel_scale : astropy.Quantity
        Physical or angular size of one pixel.
        
    Returns
    -------
    extent : list
        [xmin, xmax, ymin, ymax]
    xlabel : str
        Label for x axis with unit.
    ylabel : str
        Label for y axis with unit.
    """
    if len(shape) == 3:
        H, W = shape[1], shape[2]
    else:
        H, W = shape[0], shape[1]
        
    extent = None
    xlabel = 'x (pix)'
    ylabel = 'y (pix)'
    
    if pixel_scale is not None and isinstance(pixel_scale, u.Quantity):
        ps = pixel_scale
        # Determine best unit based on total field of view
        total_width = W * ps
        unit = ps.unit
        
        if unit.is_equivalent(u.m):
            if total_width < 100 * u.um:
                unit = u.um
            elif total_width < 1 * u.m:
                unit = u.mm
            else:
                unit = u.m
        elif unit.is_equivalent(u.rad):
            if total_width < 1 * u.arcsec:
                unit = u.mas
            elif total_width < 2 * u.deg:
                unit = u.arcsec
            else:
                unit = u.deg
                
        # Calculate extent
        half_x = (W / 2) * ps.to(unit).value
        half_y = (H / 2) * ps.to(unit).value
        extent = [-half_x, half_x, -half_y, half_y]
        xlabel = f"x [{unit}]"
        ylabel = f"y [{unit}]"
        
    return extent, xlabel, ylabel

def format_coord(coord: Union[u.Quantity, Tuple, List]) -> str:
    """
    Format a coordinate (tuple or Quantity) into a readable string with integer values if possible.
    Tries to find a unit (deg, arcmin, arcsec, mas, uas) where values are in [0, 999].
    """
    if isinstance(coord, (tuple, list)):
        # Check if elements are quantities
        if all(isinstance(c, u.Quantity) for c in coord):
            # Convert to array quantity
            try:
                coord = u.Quantity(coord)
            except:
                pass # Mixed units?
        else:
            # Plain numbers, assume radians if small? Or just print as is.
            # User said "si c'est des tuples ... affiche les en int en convertissant"
            # If plain floats, we don't know unit. Just format nicely.
            try:
                return f"({coord[0]:.2e}, {coord[1]:.2e})"
            except:
                return str(coord)

    if isinstance(coord, u.Quantity):
        # Flatten if needed
        vals = coord.flatten()
        if vals.size != 2:
            return str(coord)
        
        # Try units from smallest to largest to find best fit
        # Default for 0 is mas
        if np.max(np.abs(vals.value)) == 0:
             return "0, 0 mas"

        for unit in [u.uas, u.mas, u.arcsec, u.arcmin, u.deg]:
            try:
                v = vals.to(unit).value
                max_val = np.max(np.abs(v))
                
                if 0.1 <= max_val < 1000:
                    # Good range. Format as int if close to int, else float
                    # User asked for "affiche les en int"
                    return f"{int(round(v[0]))}, {int(round(v[1]))} {unit}"
            except:
                continue
        
        # Fallback
        return f"{vals[0].value:.2e}, {vals[1].value:.2e} {vals.unit}"

    return str(coord)
