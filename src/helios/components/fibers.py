"""Fiber coupling components for integrated photonics.

This module provides FiberIn and FiberOut classes for modeling single-mode
and multi-mode fiber coupling in photonic integrated circuits.
"""
from typing import Optional
from ..core.pipeline import Element, Pipeline, OpticalLayer
from ..core.simulation import Wavefront


class FiberIn(OpticalLayer):
    """Fiber input coupler.
    
    Models coupling of light from free-space optics into optical fiber(s).
    
    Parameters
    ----------
    modes : int
        Number of fiber modes (1 = single-mode, >1 = multi-mode).
        Default: 1 (single-mode fiber).
    **kwargs
        Additional fiber parameters (e.g., core diameter, numerical aperture).
    
    Examples
    --------
    >>> # Single-mode fiber
    >>> fiber = FiberIn(modes=1)
    >>> 
    >>> # Multi-mode fiber with 5 modes
    >>> fiber_mm = FiberIn(modes=5)
    """
    def __init__(self, modes: int = 1, name: str = None, **kwargs):
        self.modes = modes
        super().__init__(name=name)
        self.num_inputs = 1
    
    def process(self, wavefront: Wavefront, context: Optional['Context'] = None) -> Wavefront:
        """Couple wavefront into fiber.
        
        Parameters
        ----------
        wavefront : Wavefront
            Input free-space wavefront
        context : Context, optional
            Simulation context.
        
        Returns
        -------
        Wavefront
            Fiber-coupled wavefront
        
        Notes
        -----
        Current implementation is a placeholder. Full implementation would:
        - Compute overlap integral with fiber mode(s)
        - Apply coupling efficiency losses
        - Model modal decomposition for multi-mode fibers
        """
        # Set max_modes to simulate coupling into fiber
        wavefront.max_modes = self.modes
        return wavefront


class FiberOut(OpticalLayer):
    """Fiber output coupler.
    
    Models light exiting from optical fiber back into free-space optics.
    
    Parameters
    ----------
    **kwargs
        Fiber output parameters (e.g., beam divergence, mode field diameter).
    
    Examples
    --------
    >>> fiber_out = FiberOut()
    """
    def __init__(self, name: str = None, **kwargs):
        super().__init__(name=name)
        self.num_inputs = 1

    def process(self, wavefront: Wavefront, context: Optional['Context'] = None) -> Wavefront:
        """Output light from fiber.
        
        Parameters
        ----------
        wavefront : Wavefront
            Input fiber-guided wavefront
        context : Context, optional
            Simulation context.
        
        Returns
        -------
        Wavefront
            Free-space wavefront after fiber exit
        
        Notes
        -----
        Current implementation is a placeholder. Full implementation would:
        - Apply fiber mode field pattern
        - Model beam divergence
        - Account for Gaussian beam propagation
        """
        # Reset max_modes to None (free space)
        wavefront.max_modes = None
        return wavefront
