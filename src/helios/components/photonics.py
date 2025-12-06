import numpy as np
from astropy import units as u
from typing import List, Union, Tuple, Optional
from ..core.context import Layer, Element, Context
from ..core.simulation import Wavefront
import copy

class PhotonicChip(Layer):
    __slots__ = ("inputs", "lambda0", "num_inputs", "name")
    """
    Container for photonic elements.
    
    This class can be used to group photonic elements, but elements can also
    be added directly to the Context.
    """
    def __init__(self, inputs: int, lambda0: u.Quantity, **kwargs):
        self.inputs = inputs
        self.lambda0 = lambda0
        super().__init__()
        self.num_inputs = inputs

    def add_element(self, element: Element):
        """Add an element to the chip and link it."""
        super().add_element(element)
        # We can also explicitly set a property on the element if needed,
        # but accessing via self.layer (which is this chip) is cleaner.
        # For convenience, we can check if the element has a set_chip method or similar.
        pass

    def process(self, wavefronts: Union[Wavefront, List[Wavefront]], context: Context) -> Union[Wavefront, List[Wavefront]]:
        # Process light through the chip's internal layers
        # This acts as a mini-context
        current_signal = wavefronts
        for element in self.elements:
            # This simple loop doesn't support the complex routing of Context.observe
            # It assumes a linear chain or simple parallel processing
            # For complex routing, use Context directly
            
            # Ensure element has access to chip properties if needed
            # (Already handled by parent link in Element)
            
            current_signal = element.process(current_signal, context)
        return current_signal

class YSplitter(Element):
    __slots__ = ("num_inputs", "num_outputs", "name")
    """
    Y-Junction Beam Splitter.
    
    Splits 1 input into 2 outputs with equal amplitude (50/50 split).
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name or "YSplitter")
        self.num_inputs = 1
        self.num_outputs = 2
        
    def process(self, wavefront: Wavefront, context: Context) -> List[Wavefront]:
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

class ThermoOpticPhaseShifter(Element):
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
        
    def process(self, wavefront: Wavefront, context: Context) -> Wavefront:
        # Apply phase shift
        wf_out = copy.deepcopy(wavefront)
        wf_out *= np.exp(1j * self.phase)
        return wf_out

# Alias for convenience
TOPS = ThermoOpticPhaseShifter

class MultiModeInterferometer(Element):
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
        
    def process(self, wavefronts: Union[Wavefront, List[Wavefront]], context: Context) -> List[Wavefront]:
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

class Swap(Layer):
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
        
    def process(self, wavefronts: Union[Wavefront, List[Wavefront]], context: Context) -> List[Wavefront]:
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

def test_photonics():
    # Test YSplitter
    ys = YSplitter()
    wf = Wavefront(wavelength=1.55*u.um, size=10)
    outs = ys.process(wf, None)
    assert len(outs) == 2
    assert np.allclose(np.abs(outs[0])**2 + np.abs(outs[1])**2, np.abs(wf)**2)
    
    # Test TOPS
    tops = TOPS(phase=np.pi)
    out_tops = tops.process(wf, None)
    assert np.allclose(out_tops, -wf)
    
    # Test MMI 2x2 (Hadamard-like)
    mat = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    mmi = MMI(matrix=mat)
    assert mmi.num_inputs == 2
    assert mmi.num_outputs == 2
    
    ins = [wf, wf] # Constructive interference on port 0, destructive on port 1
    mmi_outs = mmi.process(ins, None)
    # Port 0: (1+1)/sqrt(2) = sqrt(2) -> Intensity 2
    # Port 1: (1-1)/sqrt(2) = 0 -> Intensity 0
    # Input intensity sum: 1+1 = 2
    assert np.allclose(np.abs(mmi_outs[0])**2, 2)
    assert np.allclose(np.abs(mmi_outs[1])**2, 0)

if __name__ == "__main__":
    test_photonics()
    print("Photonics tests passed.")
