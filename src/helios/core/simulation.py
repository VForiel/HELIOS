import numpy as np
from astropy import units as u
from typing import Optional
import matplotlib.pyplot as plt
import copy

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
    >>> # Visualize
    >>> wf.plot(title="Aberrated Wavefront")
    
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
        self.max_modes: Optional[int] = None  # None for free-space, int for guided modes

    def copy(self) -> 'Wavefront':
        """
        Return a deep copy of the wavefront.
        
        Returns
        -------
        Wavefront
            A new Wavefront instance with independent field array.
        """
        new_wf = copy.copy(self)
        new_wf.field = self.field.copy()
        return new_wf

    def plot(self, title: Optional[str] = None, figsize: tuple = (12, 5), 
             show: bool = True):
        """
        Plot the wavefront amplitude and phase side by side.
        
        Parameters
        ----------
        title : str, optional
            Super title for the figure.
        figsize : tuple, optional
            Figure size (width, height). Default (12, 5).
        show : bool, optional
            If True, call plt.show(). Default True.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : list of matplotlib.axes.Axes
            The axes objects (ax1, ax2).
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Amplitude
        im1 = ax1.imshow(np.abs(self.field), cmap='inferno', origin='lower')
        ax1.set_title("Amplitude")
        plt.colorbar(im1, ax=ax1)
        
        # Phase
        im2 = ax2.imshow(np.angle(self.field), cmap='twilight', vmin=-np.pi, vmax=np.pi, origin='lower')
        ax2.set_title("Phase (rad)")
        plt.colorbar(im2, ax=ax2)
        
        if title:
            fig.suptitle(title)
            
        plt.tight_layout()
        
        if show:
            plt.show()
            
        return fig, (ax1, ax2)

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
