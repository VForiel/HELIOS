import numpy as np
from astropy import units as u

class Wavefront:
    """
    Represents the electromagnetic field (complex amplitude).
    
    A wavefront describes the spatial distribution of light at a given wavelength.
    The complex field contains both amplitude and phase information, enabling
    simulation of interference, diffraction, and aberrations.
    
    Parameters
    ----------
    wavelength : Quantity
        Wavelength of the light (e.g., 550*u.nm, 1.6*u.um)
    size : int
        Number of pixels in the field array (creates size × size array)
    
    Attributes
    ----------
    wavelength : Quantity
        Wavelength of the electromagnetic radiation
    field : ndarray of complex128
        Complex amplitude array representing the electric field.
        Shape is (size, size). Amplitude = abs(field), phase = angle(field)
    pixel_scale : Quantity
        Physical size per pixel in meters (for pupil plane) or angular
        size per pixel (for image plane)
    
    Examples
    --------
    Create a wavefront and apply a phase aberration:
    
    >>> import numpy as np
    >>> from astropy import units as u
    >>> 
    >>> wf = Wavefront(wavelength=550*u.nm, size=256)
    >>> # Apply pupil amplitude
    >>> pupil = helios.Pupil.like('JWST')
    >>> wf.field = pupil.get_array(256).astype(np.complex128)
    >>> # Add phase aberration
    >>> phase = np.random.randn(256, 256) * 0.5  # radians
    >>> wf.field *= np.exp(1j * phase)
    
    Notes
    -----
    The field is typically initialized to uniform amplitude (ones) and then
    modified by layers to include pupil masks, phase aberrations, etc.
    
    The complex field enables coherent propagation:
    - Fourier transform for Fraunhofer diffraction
    - Fresnel propagation for arbitrary distances
    - Interference between multiple beams
    
    See Also
    --------
    Layer : Components that transform wavefronts
    """
    def __init__(self, wavelength: u.Quantity, size: int):
        self.wavelength = wavelength
        self.field = np.ones((size, size), dtype=np.complex128)
        self.pixel_scale = 1.0 * u.m # Placeholder

    def propagate(self, distance: u.Quantity):
        """
        Propagate the wavefront by a certain distance.
        
        Parameters
        ----------
        distance : Quantity
            Propagation distance (e.g., 10*u.m, 1*u.km)
        
        Notes
        -----
        This is a placeholder for future Fresnel/Fraunhofer propagation
        implementation. Current version does not modify the field.
        """
        # Placeholder for Fresnel/Fraunhofer propagation
        pass

class Simulation:
    """
    Helper class for running specific simulation types if needed.
    
    This class provides utilities for common simulation workflows and may
    be extended in the future for specialized observation modes.
    
    Notes
    -----
    Most simulations should use Context directly. This class is reserved
    for future specialized simulation types or batch processing workflows.
    
    See Also
    --------
    Context : Main simulation orchestrator
    """
    pass

def test_wavefront_init():
    wf = Wavefront(wavelength=600*u.nm, size=128)
    assert wf.field.shape == (128, 128)
    assert wf.wavelength == 600 * u.nm

if __name__ == "__main__":
    test_wavefront_init()
    print("Simulation tests passed.")
