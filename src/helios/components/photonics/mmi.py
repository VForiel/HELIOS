"""Multi-Mode Interferometer (MMI) component."""
import numpy as np
from typing import List, Union, Optional
from ...core.component import OpticalComponent
from ...core.pipeline import Pipeline
from ...core.wavefront import Wavefront
import copy


class MultiModeInterferometer(OpticalComponent):
    __slots__ = ("matrix", "num_outputs", "num_inputs", "name")
    """
    Multi-Mode Interferometer (MMI).
    
    General N x M coupler defined by a transfer matrix.
    """
    def __init__(self, matrix: Union[List[List[complex]], np.ndarray], name: Optional[str] = None):
        super().__init__(name=name or "MMI")
        self.matrix = np.array(matrix, dtype=np.complex128)
        # Matrix shape: (M_outputs, N_inputs)
        self.num_outputs, self.num_inputs = self.matrix.shape
        
    def process(self, wavefronts: Union[Wavefront, List[Wavefront]], pipeline: Optional['Pipeline'] = None) -> List[Wavefront]:
        """
        Mix inputs according to the transfer matrix.

        Parameters
        ----------
        wavefronts : Wavefront or list of Wavefront
            Input signal(s).
        pipeline : Pipeline, optional
            Simulation pipeline.

        Returns
        -------
        list of Wavefront
            Output signals corresponding to matrix rows.
        """
        # Handle input: can be single Wavefront or list
        if isinstance(wavefronts, Wavefront):
            inputs = [wavefronts]
        else:
            inputs = wavefronts
            
        if len(inputs) != self.num_inputs:
            # Gracefully handle fewer inputs by zero-padding missing ports
            if len(inputs) < self.num_inputs:
                # Create zero-field copies to pad up to required inputs
                template = inputs[0] if len(inputs) > 0 else None
                if template is None:
                    raise ValueError("MMI received no inputs to infer grid for padding")
                padded = [copy.deepcopy(template) for _ in range(self.num_inputs - len(inputs))]
                for wf in padded:
                    wf[:] = 0
                inputs = inputs + padded
            else:
                # Truncate extra inputs to expected number
                inputs = inputs[:self.num_inputs]
        
        # We assume all inputs have same wavelength/grid
        # We perform matrix multiplication on the fields
        
        # Stack input fields: (N_inputs, nsource, Size, Size)
        input_fields = np.stack([wf for wf in inputs])
        
        # Matrix multiplication
        # Matrix: (M_out, N_in)
        # Input: (N_in, nsource, Size, Size)
        # Output: (M_out, nsource, Size, Size)
        # We can use einsum: 'mn,nsij->msij'
        output_fields = np.einsum('mn,nsij->msij', self.matrix, input_fields)
        
        # Create output wavefronts
        outputs = []
        base_wf = inputs[0] # Use first input as template
        
        for i in range(self.num_outputs):
            new_wf = copy.deepcopy(base_wf)
            new_wf[:] = output_fields[i]
            outputs.append(new_wf)
            
        return outputs

# Alias for convenience
MMI = MultiModeInterferometer
