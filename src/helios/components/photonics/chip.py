"""Photonic Integrated Circuit (PIC) container."""
import numpy as np
from astropy import units as u
from typing import List, Union, Tuple, Optional
from ...core.pipeline import Layer, Component, OpticalComponent, Pipeline, OpticalLayer
from ...core.simulation import Wavefront
import copy


class PhotonicChip(OpticalLayer):
    __slots__ = ("inputs", "lambda0", "num_inputs", "name")
    """
    Container for photonic elements.
    
    This class can be used to group photonic elements, but elements can also
    be added directly to the Pipeline.
    """
    def __init__(self, inputs: int, lambda0: u.Quantity, **kwargs):
        self.inputs = inputs
        self.lambda0 = lambda0
        super().__init__()
        self.num_inputs = inputs

    def add_element(self, component: Component):
        """Add a component to the chip and link it."""
        super().add_element(component)
        # We can also explicitly set a property on the element if needed,
        # but accessing via self.layer (which is this chip) is cleaner.
        # For convenience, we can check if the element has a set_chip method or similar.
        pass

    def process(self, wavefronts: Union[Wavefront, List[Wavefront]], pipeline: Optional['Pipeline'] = None) -> Union[Wavefront, List[Wavefront]]:
        """
        Process light through the chip's internal layers.

        Parameters
        ----------
        wavefronts : Wavefront or list of Wavefront
            Input signal(s).
        pipeline : Pipeline, optional
            Simulation pipeline.

        Returns
        -------
        Wavefront or list of Wavefront
            Processed signal(s).
        """
        # Process light through the chip's internal layers
        # This acts as a mini-pipeline
        current_signal = wavefronts
        for element in self.elements:
            # This simple loop doesn't support the complex routing of Pipeline.observe
            # It assumes a linear chain or simple parallel processing
            # For complex routing, use Pipeline directly
            
            # Ensure element has access to chip properties if needed
            # (Already handled by parent link in Component)
            
            current_signal = element.process(current_signal)
        return current_signal
