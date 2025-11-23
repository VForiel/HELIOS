import numpy as np
from astropy import units as u
from typing import List, Union, Optional, Any
import copy

class Layer:
    """
    Base class for all simulation layers.
    """
    def __init__(self):
        pass

    def process(self, wavefront: Any, context: 'Context') -> Any:
        """
        Process the incoming wavefront/signal and return the result.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement process()")

class Context:
    """
    Main simulation context managing layers and execution.
    """
    def __init__(self, date: Any = None, declination: Any = None, **kwargs):
        self.date = date
        self.declination = declination
        self.layers: List[Union[Layer, List[Layer]]] = []
        self.results = {}

    def add_layer(self, layer: Union[Layer, List[Layer]]):
        """
        Add a layer or a list of parallel layers to the simulation.
        """
        self.layers.append(layer)

    def observe(self) -> Any:
        """
        Run the simulation through all layers.
        """
        # Initial wavefront/signal (starts as None or empty)
        current_signal = None

        for i, layer in enumerate(self.layers):
            if isinstance(layer, list):
                # Parallel processing (e.g., splitting paths)
                # If current_signal is a list, we assume 1-to-1 mapping or broadcasting
                # For now, let's assume the previous layer returned a list of signals 
                # OR we broadcast the single signal to all parallel layers
                
                outputs = []
                if isinstance(current_signal, list):
                    if len(current_signal) != len(layer):
                         # Try broadcasting if single signal, else error
                         pass # TODO: Handle mismatch
                    
                    for sig, sub_layer in zip(current_signal, layer):
                        outputs.append(sub_layer.process(sig, self))
                else:
                    # Broadcast
                    for sub_layer in layer:
                        # We might need to copy the signal if it's mutable
                        outputs.append(sub_layer.process(copy.deepcopy(current_signal), self))
                
                current_signal = outputs

            else:
                # Single layer
                # If current_signal is a list, this layer might merge them or process them individually
                # For now, let's assume if it receives a list, it processes the list (merging or keeping as list)
                # But typically a single layer after a split might be a detector array or a combiner.
                
                # Let's let the layer handle the input type
                current_signal = layer.process(current_signal, self)

        return current_signal

    def get_output_intensities(self):
        # Placeholder for interferometry output
        pass

def test_context_initialization():
    ctx = Context(date="2025-01-01", declination=10)
    assert ctx.date == "2025-01-01"
    assert ctx.declination == 10
    assert len(ctx.layers) == 0

def test_context_add_layer():
    ctx = Context()
    class MockLayer(Layer):
        def process(self, wf, ctx): return "processed"
    
    l1 = MockLayer()
    ctx.add_layer(l1)
    assert len(ctx.layers) == 1
    assert ctx.layers[0] == l1

if __name__ == "__main__":
    import pytest
    # Run internal tests
    # pytest.main([__file__])
    test_context_initialization()
    test_context_add_layer()
    print("Context tests passed.")
