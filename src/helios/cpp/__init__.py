import numpy as np

def compute_diffraction_pattern(size, wavelength, distance, pixel_scale):
    """
    Compute the diffraction pattern phase factor using a spherical approximation.
    
    Args:
        size (int): Size of the grid in pixels.
        wavelength (float): Wavelength of light in meters.
        distance (float): Propagation distance in meters.
        pixel_scale (float): Size of one pixel in meters.
        
    Returns:
        numpy.ndarray: Complex array representing the phase factor.
    """
    k = 2 * np.pi / wavelength
    center = size / 2.0
    
    y, x = np.ogrid[:size, :size]
    dx = (x - center) * pixel_scale
    dy = (y - center) * pixel_scale
    r2 = dx**2 + dy**2
    
    # Simple spherical phase factor approximation
    phase = k * r2 / (2 * distance)
    return np.exp(1j * phase)
