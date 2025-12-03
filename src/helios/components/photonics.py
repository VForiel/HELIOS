import numpy as np
from astropy import units as u
from typing import List, Union, Tuple, Optional
from ..core.context import Layer, Element, Context
from ..core.simulation import Wavefront
import copy

class PhotonicChip(Layer):
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
        out1.field *= 1/np.sqrt(2)
        out2.field *= 1/np.sqrt(2)
        
        return [out1, out2]

class ThermoOpticPhaseShifter(Element):
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
        wf_out.field *= np.exp(1j * self.phase)
        return wf_out

# Alias for convenience
TOPS = ThermoOpticPhaseShifter

class MultiModeInterferometer(Element):
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
            # If mismatch, we might need to handle it. 
            # For now, assume correct number of inputs or pad/truncate?
            # Strict check for now
            if len(inputs) < self.num_inputs:
                 raise ValueError(f"MMI expects {self.num_inputs} inputs, got {len(inputs)}")
        
        # We assume all inputs have same wavelength/grid
        # We perform matrix multiplication on the fields
        
        # Stack input fields: (N_inputs, Size, Size)
        input_fields = np.stack([wf.field for wf in inputs])
        
        # Matrix multiplication
        # Matrix: (M_out, N_in)
        # Input: (N_in, Size, Size)
        # Output: (M_out, Size, Size)
        # We can use einsum: 'mn,nij->mij'
        output_fields = np.einsum('mn,nij->mij', self.matrix, input_fields)
        
        # Create output wavefronts
        outputs = []
        base_wf = inputs[0] # Use first input as template
        
        for i in range(self.num_outputs):
            new_wf = copy.deepcopy(base_wf)
            new_wf.field = output_fields[i]
            outputs.append(new_wf)
            
        return outputs

# Alias for convenience
MMI = MultiModeInterferometer

def test_photonics():
    # Test YSplitter
    ys = YSplitter()
    wf = Wavefront(wavelength=1.55*u.um, size=10)
    outs = ys.process(wf, None)
    assert len(outs) == 2
    assert np.allclose(np.abs(outs[0].field)**2 + np.abs(outs[1].field)**2, np.abs(wf.field)**2)
    
    # Test TOPS
    tops = TOPS(phase=np.pi)
    out_tops = tops.process(wf, None)
    assert np.allclose(out_tops.field, -wf.field)
    
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
    assert np.allclose(np.abs(mmi_outs[0].field)**2, 2)
    assert np.allclose(np.abs(mmi_outs[1].field)**2, 0)

if __name__ == "__main__":
    test_photonics()
    print("Photonics tests passed.")
