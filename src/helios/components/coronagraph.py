"""Coronagraph focal-plane masks for high-contrast imaging.

This module provides the Coronagraph class for suppressing on-axis starlight
to enable detection of faint companions (exoplanets, circumstellar disks).
"""
import numpy as np
from astropy import units as u
from typing import Optional
import matplotlib.pyplot as _plt

from ..core.context import Layer, Context
from ..core.simulation import Wavefront


class Coronagraph(Layer):
    """Coronagraph focal-plane mask layer.
    
    Applies phase and/or amplitude masks in the focal plane to suppress
    on-axis stellar light, enabling high-contrast imaging of faint companions.
    
    The coronagraph works by:
    1. FFT of input field → focal plane complex field
    2. Multiply by coronagraph mask (phase/amplitude pattern)
    3. Inverse FFT → output field with suppressed starlight
    
    Parameters
    ----------
    phase_mask : str
        Type of coronagraph mask. Options:
        - '4quadrants' or '4q': Four-quadrant phase mask (π phase shifts)
        - 'vortex': Optical vortex coronagraph (charge-2 by default)
        Default: '4quadrants'.
    
    Attributes
    ----------
    phase_mask : str
        The selected mask type.
    
    Examples
    --------
    >>> # 4-quadrant phase mask
    >>> coro = Coronagraph(phase_mask='4quadrants')
    >>> 
    >>> # Vortex coronagraph
    >>> coro_vortex = Coronagraph(phase_mask='vortex')
    >>> 
    >>> # Plot mask
    >>> coro.plot_mask(npix=512, charge=2)
    """
    def __init__(self, phase_mask: str = '4quadrants'):
        self.phase_mask = phase_mask
        super().__init__()

    def process(self, wavefront: Wavefront, context: Context) -> Wavefront:
        """Apply coronagraph mask to wavefront.
        
        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront (pupil plane)
        context : Context
            Simulation context
        
        Returns
        -------
        Wavefront
            Output wavefront after coronagraph suppression
        
        Notes
        -----
        Uses Fraunhofer approximation (monochromatic):
        1. FFT(wavefront.field) → focal plane field
        2. Multiply by focal-plane mask (phase and/or amplitude)
        3. Inverse FFT → back to pupil/image plane
        """
        try:
            field = wavefront.field
            N = field.shape[0]
        except Exception:
            return wavefront

        # Focal-plane field
        ffield = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))

        # Build mask
        mask = self.mask_array(npix=N)

        # Apply mask in focal plane
        ffield_masked = ffield * mask

        # Back to pupil/image plane
        field_after = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ffield_masked)))
        wavefront.field = field_after.astype(wavefront.field.dtype)
        return wavefront

    def mask_array(self, npix: int = 512, kind: Optional[str] = None, charge: int = 2,
                   lam: Optional[u.Quantity] = None,
                   diameter: Optional[u.Quantity] = None,
                   fov: Optional[u.Quantity] = None) -> np.ndarray:
        """Generate complex focal-plane mask array.

        Parameters
        ----------
        npix : int
            Array size (pixels). Default: 512.
        kind : str, optional
            Mask type (overrides self.phase_mask if provided).
        charge : int
            Topological charge for vortex mask. Default: 2.
        lam : astropy.Quantity, optional
            Wavelength for physical scaling (angular units).
        diameter : astropy.Quantity, optional
            Aperture diameter for physical scaling.
        fov : astropy.Quantity, optional
            Total field-of-view (angular, e.g., 4*u.arcsec).
        
        Returns
        -------
        mask : ndarray
            Complex focal-plane mask, shape (npix, npix).
        
        Notes
        -----
        If lam, diameter, and fov are provided, the mask is generated with
        physical angular scaling (λ/D units). Otherwise, uses normalized [-1,1] grid.
        
        Supported masks:
        - '4quadrants' or '4q': π phase shifts in alternating quadrants
        - 'vortex': Optical vortex exp(i*charge*θ)
        """
        k = kind or self.phase_mask or '4quadrants'

        # Build coordinate grid
        if (lam is not None) and (diameter is not None) and (fov is not None):
            # Physical angular scaling
            lam = lam.to(u.m)
            diameter = diameter.to(u.m)
            fov_angle = fov.to(u.rad).value
            
            # Angular resolution per pixel (radians)
            xs = np.linspace(-fov_angle / 2.0, fov_angle / 2.0, npix)
            ys = xs.copy()
            xg_ang, yg_ang = np.meshgrid(xs, ys)
            
            # Lambda/D in radians
            lam_over_D = (lam / diameter).decompose().value
            
            # Coordinates in units of lambda/D
            xg = xg_ang / lam_over_D
            yg = yg_ang / lam_over_D
        else:
            # Normalized grid [-1, 1]
            xs = np.linspace(-1.0, 1.0, npix)
            ys = xs.copy()
            xg, yg = np.meshgrid(xs, ys)

        # Generate mask
        if k.lower() in ('4quadrants', '4q'):
            # 4QPM: alternate quadrants have π phase (i.e., -1 complex multiplier)
            mask = np.ones((npix, npix), dtype=np.complex64)
            mask[(xg < 0) & (yg > 0)] = -1.0
            mask[(xg > 0) & (yg < 0)] = -1.0
            return mask
        elif k.lower() in ('vortex',):
            # Optical vortex: exp(i * charge * θ)
            theta = np.arctan2(yg, xg)
            phase = np.exp(1j * charge * theta)
            return phase.astype(np.complex64)
        else:
            # Identity (no mask)
            return np.ones((npix, npix), dtype=np.complex64)

    def plot_mask(self, npix: int = 512, kind: Optional[str] = None, charge: int = 2,
                  lam: Optional[u.Quantity] = None,
                  diameter: Optional[u.Quantity] = None,
                  fov: Optional[u.Quantity] = None,
                  ax: Optional[_plt.Axes] = None,
                  cmap: str = 'gray',
                  vmin: Optional[float] = None,
                  vmax: Optional[float] = None,
                  display: Optional[str] = None) -> _plt.Axes:
        """Plot coronagraph focal-plane mask phase.

        Parameters
        ----------
        npix : int
            Array size. Default: 512.
        kind : str, optional
            Mask type (overrides self.phase_mask).
        charge : int
            Vortex topological charge. Default: 2.
        lam, diameter, fov : astropy.Quantity, optional
            Physical scaling parameters.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        cmap : str
            Colormap. Default: 'gray'.
        vmin, vmax : float, optional
            Color limits.
        display : str, optional
            Display units: 'lambda/D', 'arcsec', 'rad', or None (pixels).
        
        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes with mask plot.
        
        Notes
        -----
        Plots the phase angle np.angle(mask) in radians [-π, π].
        For real masks (4-quadrant), values are 0 or π.
        """
        mask = self.mask_array(npix=npix, kind=kind, charge=charge, lam=lam, diameter=diameter, fov=fov)
        
        # Compute phase in range [-π, π]
        phase = np.angle(mask)
        
        if ax is None:
            fig, ax = _plt.subplots()
        
        # Determine extent
        extent = None
        xlabel = 'Focal plane x (pixels)'
        ylabel = 'Focal plane y (pixels)'
        
        if (lam is not None) and (diameter is not None) and (fov is not None):
            # Physical scaling available
            display_mode = (display or 'lambda/D').lower()
            fov_rad = float(fov.to(u.rad).value)
            lam_over_D = float((lam / diameter).to(u.dimensionless_unscaled).value)
            
            if display_mode in ('lambda/d', 'lambdad', 'lambda/d'):
                # Extent in units of lambda/D
                half = (fov_rad / 2.0) / lam_over_D
                extent = [-half, half, -half, half]
                xlabel = 'Focal plane x (λ/D)'
                ylabel = 'Focal plane y (λ/D)'
            elif display_mode in ('arcsec', 'arcseconds'):
                half = float(fov.to(u.arcsec).value) / 2.0
                extent = [-half, half, -half, half]
                xlabel = 'Focal plane x (arcsec)'
                ylabel = 'Focal plane y (arcsec)'
            elif display_mode in ('rad', 'radians'):
                half = fov_rad / 2.0
                extent = [-half, half, -half, half]
                xlabel = 'Focal plane x (rad)'
                ylabel = 'Focal plane y (rad)'
            else:
                extent = [-npix/2.0, npix/2.0, -npix/2.0, npix/2.0]
        else:
            extent = [-npix/2.0, npix/2.0, -npix/2.0, npix/2.0]

        im = ax.imshow(phase, origin='lower', cmap=cmap, extent=extent)
        
        if vmin is not None or vmax is not None:
            im.set_clim(vmin=vmin if vmin is not None else phase.min(), 
                       vmax=vmax if vmax is not None else phase.max())
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')
        _plt.colorbar(im, ax=ax, label='Phase (rad)')
        
        return ax

    def image_from_scene(self, scene_array: np.ndarray, soft: bool = True, oversample: int = 4, 
                        normalize: bool = True,
                        lam: Optional[u.Quantity] = None, 
                        diameter: Optional[u.Quantity] = None, 
                        fov: Optional[u.Quantity] = None) -> np.ndarray:
        """Compute coronagraphic image from scene.

        Parameters
        ----------
        scene_array : ndarray
            Input scene (2D square array).
        soft : bool
            Unused (for API compatibility).
        oversample : int
            Unused (for API compatibility).
        normalize : bool
            Normalize output to peak = 1.
        lam, diameter, fov : astropy.Quantity, optional
            Physical scaling parameters.
        
        Returns
        -------
        intensity : ndarray
            Coronagraphic image (intensity).
        
        Notes
        -----
        Simplified pipeline (monochromatic, Fraunhofer):
        1. FFT(scene_array) → focal plane field
        2. Apply coronagraph mask
        3. Inverse FFT → image field, return intensity
        """
        arr = np.asarray(scene_array, dtype=np.complex64)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError('scene_array must be a square 2D array')
        
        N = arr.shape[0]
        
        # Field in focal plane
        ffield = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arr)))
        
        # Build mask
        mask = self.mask_array(npix=N, lam=lam, diameter=diameter, fov=fov)
        ffield_masked = ffield * mask
        
        # Back to image plane
        img_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ffield_masked)))
        intensity = np.abs(img_field) ** 2
        
        if normalize:
            m = intensity.max()
            if m > 0:
                intensity = intensity / float(m)
        
        return intensity

    def plot_image_from_scene(self, scene_array: np.ndarray, 
                             ax: Optional[_plt.Axes] = None, 
                             cmap: str = 'inferno', 
                             normalize: bool = True, 
                             log: bool = False) -> _plt.Axes:
        """Compute and plot coronagraphic image.

        Parameters
        ----------
        scene_array : ndarray
            Input scene.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        cmap : str
            Colormap. Default: 'inferno'.
        normalize : bool
            Normalize to peak = 1.
        log : bool
            Use log scale.
        
        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes with image plot.
        """
        intensity = self.image_from_scene(scene_array, normalize=normalize)
        disp = intensity
        
        if log:
            disp = np.log10(disp + 1e-12)
        
        if ax is None:
            fig, ax = _plt.subplots()
        
        im = ax.imshow(disp, origin='lower', cmap=cmap)
        ax.set_xlabel('Image x (pixels)')
        ax.set_ylabel('Image y (pixels)')
        ax.set_aspect('equal')
        _plt.colorbar(im, ax=ax)
        
        return ax
