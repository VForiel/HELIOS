import numpy as np
from typing import List, Tuple, Optional
from astropy import units as u

from .wavefront import Wavefront

class Spectrum:
    """
    Polychromatic wavefront from a single coherent source.
    
    A Spectrum contains multiple Wavefront objects at different wavelengths,
    representing a single source with spectral bandwidth. This is used for
    simulating chromatic effects like dispersion and chromatic aberration.
    
    Parameters
    ----------
    wavefronts : list of Wavefront
        List of monochromatic wavefronts at different wavelengths.
    weights : ndarray, optional
        Spectral weights for each wavelength (for integration).
        If None, uniform weights are used.
    
    Attributes
    ----------
    wavefronts : list of Wavefront
        The constituent monochromatic wavefronts.
    weights : ndarray
        Normalized spectral weights.
    
    Examples
    --------
    Create a polychromatic spectrum:
    
    >>> wavelengths = np.linspace(500, 600, 10) * u.nm
    >>> wfs = [Wavefront(wavelength=λ, size=1*u.m, npix=256) for λ in wavelengths]
    >>> spectrum = Spectrum(wfs)
    >>> spectrum_focal = spectrum.propagate(distance=100*u.m, focal_length=100*u.m)
    >>> image = spectrum.integrate()
    
    See Also
    --------
    Wavefront : Monochromatic wavefront
    OpticalScene : Multi-source wavefront
    """
    def __init__(self, wavefronts: List[Wavefront], weights: Optional[np.ndarray] = None):
        self.wavefronts = wavefronts
        
        if weights is None:
            self.weights = np.ones(len(wavefronts))
        else:
            self.weights = np.asarray(weights)
        
        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)
    
    @property
    def wavelengths(self) -> u.Quantity:
        """Array of wavelengths in this spectrum."""
        return u.Quantity([wf.wavelength for wf in self.wavefronts])
    
    def propagate(self, distance: u.Quantity, output_size: Optional[u.Quantity] = None,
                  output_npix: Optional[int] = None, focal_length: Optional[u.Quantity] = None,
                  regime: Optional[str] = None) -> 'Spectrum':
        """
        Propagate each wavelength independently.
        
        Parameters
        ----------
        distance : Quantity
            Propagation distance.
        output_size : Quantity, optional
            Physical size of output grid.
        output_npix : int, optional
            Resolution of output grid.
        focal_length : Quantity, optional
            Focal length for Fraunhofer detection.
        regime : str, optional
            Force 'fraunhofer' or 'fresnel'.
        
        Returns
        -------
        Spectrum
            Propagated spectrum with all wavelengths propagated.
        """
        propagated_wfs = [wf.propagate(distance, output_size, output_npix, focal_length, regime)
                         for wf in self.wavefronts]
        return Spectrum(propagated_wfs, weights=self.weights)
    
    def integrate(self) -> np.ndarray:
        """
        Integrate spectrally (weighted sum of intensities).
        
        Returns
        -------
        ndarray
            Spectrally-integrated intensity image.
        """
        intensities = [w * np.abs(wf.value)**2 for w, wf in zip(self.weights, self.wavefronts)]
        return np.sum(intensities, axis=0)


class OpticalScene:
    """
    Multi-source polychromatic wavefront.
    
    A OpticalScene contains multiple Spectrum objects at different angular positions,
    representing multiple incoherent sources. This is used for simulating
    complex astronomical scenes with stars, planets, and other objects.
    
    Note: This is different from PlanetarySystem, which is a GenerationLayer
    that creates the initial wavefront. OpticalScene is a container for propagating
    multiple incoherent sources through an optical system.
    
    Parameters
    ----------
    sources : list of tuple
        List of (Spectrum, (theta_x, theta_y)) tuples, where theta_x and theta_y
        are angular offsets in radians or angular units.
    
    Attributes
    ----------
    sources : list of tuple
        The constituent sources with their positions.
    
    Examples
    --------
    Create a scene with star and planet:
    
    >>> # Star at center
    >>> star_wfs = [Wavefront(wavelength=λ, size=1*u.m, npix=256) 
    ...             for λ in np.linspace(500, 600, 10)*u.nm]
    >>> star_spectrum = Spectrum(star_wfs)
    >>> 
    >>> # Planet offset by 0.5 arcsec
    >>> planet_wfs = [Wavefront(wavelength=λ, size=1*u.m, npix=256) 
    ...               for λ in np.linspace(500, 600, 10)*u.nm]
    >>> planet_spectrum = Spectrum(planet_wfs)
    >>> 
    >>> scene = OpticalScene([
    ...     (star_spectrum, (0*u.arcsec, 0*u.arcsec)),
    ...     (planet_spectrum, (0.5*u.arcsec, 0*u.arcsec))
    ... ])
    >>> scene_focal = scene.propagate(distance=100*u.m, focal_length=100*u.m)
    >>> image = scene.render()
    
    See Also
    --------
    Wavefront : Monochromatic wavefront
    Spectrum : Polychromatic wavefront
    PlanetarySystem : Generation layer for creating astronomical scenes
    """
    def __init__(self, sources: List[Tuple[Spectrum, Tuple[u.Quantity, u.Quantity]]]):
        self.sources = sources
    
    def propagate(self, distance: u.Quantity, output_size: Optional[u.Quantity] = None,
                  output_npix: Optional[int] = None, focal_length: Optional[u.Quantity] = None,
                  regime: Optional[str] = None) -> 'OpticalScene':
        """
        Propagate each source independently.
        
        Parameters
        ----------
        distance : Quantity
            Propagation distance.
        output_size : Quantity, optional
            Physical size of output grid.
        output_npix : int, optional
            Resolution of output grid.
        focal_length : Quantity, optional
            Focal length for Fraunhofer detection.
        regime : str, optional
            Force 'fraunhofer' or 'fresnel'.
        
        Returns
        -------
        OpticalScene
            Propagated scene with all sources propagated.
        """
        propagated_sources = [(spec.propagate(distance, output_size, output_npix, focal_length, regime), pos)
                             for spec, pos in self.sources]
        return OpticalScene(propagated_sources)
    
    def render(self) -> np.ndarray:
        """
        Render the scene (incoherent sum of all sources).
        
        This method integrates each source spectrally and applies angular
        offsets (tilts) before summing incoherently.
        
        Returns
        -------
        ndarray
            Final rendered image with all sources.
        
        Notes
        -----
        Currently, angular offsets are not yet implemented. This will be
        added in a future update to apply phase tilts based on source positions.
        """
        images = []
        for spectrum, (theta_x, theta_y) in self.sources:
            # Integrate spectrally
            img = spectrum.integrate()
            
            # TODO: Apply angular offset (tilt) based on theta_x, theta_y
            # This requires applying a phase ramp in the pupil plane or
            # shifting in the image plane
            
            images.append(img)
        
        return np.sum(images, axis=0)
