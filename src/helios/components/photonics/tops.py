"""Thermo-Optic Phase Shifter (TOPS) component."""
import numpy as np
from typing import Optional
from ...core.component import OpticalComponent
from ...core.pipeline import Pipeline
from ...core.wavefront import Wavefront
import copy


class ThermoOpticPhaseShifter(OpticalComponent):
    __slots__ = ("phase", "num_inputs", "name")
    """
    Thermo-Optic Phase Shifter (TOPS).
    
    Applies a phase shift to the input wavefront.
    """
    def __init__(self, phase: float = 0.0, name: Optional[str] = None):
        super().__init__(name=name or "TOPS")
        self.num_inputs = 1
        self.phase = phase # Phase shift in radians
        
    def set_phase(self, phase: float):
        """Set the phase shift in radians."""
        self.phase = phase
        
    def process(self, wavefront: Wavefront, pipeline: Optional['Pipeline'] = None) -> Wavefront:
        """
        Apply phase shift to the wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront.
        pipeline : Pipeline, optional
            Simulation pipeline.

        Returns
        -------
        Wavefront
            Phase-shifted wavefront.
        """
        # Apply phase shift
        wf_out = copy.deepcopy(wavefront)
        wf_out *= np.exp(1j * self.phase)
        return wf_out

# Alias for convenience
TOPS = ThermoOpticPhaseShifter
