"""Beam splitter for dividing optical paths.

This module provides the BeamSplitter class for splitting wavefronts into multiple paths.
"""
from typing import List, Optional
from ...core.component import Component, OpticalComponent
from ...core.layer import Layer, OpticalLayer
from ...core.pipeline import Pipeline
from ...core.wavefront import Wavefront


class BeamSplitter(OpticalComponent):
    """Optical beam splitter component.
    
    Splits an incoming wavefront into two or more output wavefronts.
    
    Parameters
    ----------
    cutoff : float
        Transmission coefficient (0 to 1). Default: 0.5 (50/50 split).
    name : str, optional
        Name of the beam splitter for identification in diagrams
    
    Examples
    --------
    >>> bs = BeamSplitter(cutoff=0.5)
    >>> wf_out = bs.process(wf_in)  # Returns list of 2 wavefronts
    """
    def __init__(self, cutoff: float = 0.5, name: Optional[str] = None):
        super().__init__(name=name or "BeamSplitter")
        self.cutoff = cutoff

    def process(self, wavefront: Wavefront, pipeline: Optional['Pipeline'] = None) -> List[Wavefront]:
        """Split wavefront into two paths.
        
        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront
        
        Returns
        -------
        List[Wavefront]
            Two wavefront copies (placeholder implementation)
        """
        # TODO: Implement proper amplitude splitting with cutoff ratio
        # For now, returns two identical copies
        return [wavefront, wavefront]
    
    def _get_detailed_attributes(self) -> dict:
        """Return detailed attributes for BeamSplitter."""
        attrs = {}
        attrs['transmission'] = f"{self.cutoff:.2%}"
        attrs['reflection'] = f"{1 - self.cutoff:.2%}"
        return attrs
