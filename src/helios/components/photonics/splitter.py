"""Waveguide splitters and routing components."""
import numpy as np
from typing import List, Union, Optional
from ...core.pipeline import OpticalComponent, OpticalLayer, Pipeline
from ...core.simulation import Wavefront
import copy


class YSplitter(OpticalComponent):
    __slots__ = ("num_inputs", "num_outputs", "name")
    """
    Y-Junction Beam Splitter.
    
    Splits 1 input into 2 outputs with equal amplitude (50/50 split).
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name or "YSplitter")
        self.num_inputs = 1
        self.num_outputs = 2
        
    def process(self, wavefront: Wavefront, pipeline: Optional['Pipeline'] = None) -> List[Wavefront]:
        """
        Split input wavefront into two output paths.

        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront.
        pipeline : Pipeline, optional
            Simulation pipeline.

        Returns
        -------
        list of Wavefront
            Two wavefronts with amplitude scaled by 1/sqrt(2).
        """
        # Split input into 2 identical copies (amplitude division)
        # Energy conservation: amplitude / sqrt(2) -> intensity / 2
        
        # Create copies
        out1 = copy.deepcopy(wavefront)
        out2 = copy.deepcopy(wavefront)
        
        # Apply splitting loss/factor
        # 1/sqrt(2) for amplitude
        out1 *= 1/np.sqrt(2)
        out2 *= 1/np.sqrt(2)
        
        return [out1, out2]


class Swap(OpticalLayer):
    """
    A layer that permutes the order of optical paths.
    
    This component is used to reorder wavefronts between layers, for example
    to implement waveguide crossings or specific routing topologies.
    It does not modify the wavefronts themselves, only their order in the list.
    
    Parameters
    ----------
    mapping : List[int]
        Permutation indices. The i-th output will be the mapping[i]-th input.
        Example: [0, 2, 1, 3] means:
        - Output 0 comes from Input 0
        - Output 1 comes from Input 2
        - Output 2 comes from Input 1
        - Output 3 comes from Input 3
    name : str, optional
        Name of the component.
    """
    def __init__(self, mapping: List[int], name: Optional[str] = None):
        super().__init__(name=name or "Swap")
        self.mapping = mapping
        self.num_inputs = len(mapping)
        self.num_outputs = len(mapping)
        
    def process(self, wavefronts: Union[Wavefront, List[Wavefront]], pipeline: Optional['Pipeline'] = None) -> List[Wavefront]:
        # Ensure input is a list
        if not isinstance(wavefronts, list):
            inputs = [wavefronts]
        else:
            inputs = wavefronts
            
        if len(inputs) != self.num_inputs:
            # If we have fewer inputs than expected, we can't map correctly
            # But maybe we just map what we have?
            # Strict check is safer
            if len(inputs) < self.num_inputs:
                 raise ValueError(f"Swap expects {self.num_inputs} inputs, got {len(inputs)}")
        
        # Apply permutation
        # Output[i] = Input[mapping[i]]
        outputs = []
        for idx in self.mapping:
            if idx < len(inputs):
                outputs.append(inputs[idx])
            else:
                outputs.append(None) # Should not happen if check passes
                
        return outputs
