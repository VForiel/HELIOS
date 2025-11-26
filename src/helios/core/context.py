import numpy as np
from astropy import units as u
from typing import List, Union, Optional, Any
import copy

class Layer:
    """
    Base class for all simulation layers.
    
    All components in HELIOS inherit from this class and implement the `process()` method
    to transform wavefronts or signals as they propagate through the optical system.
    
    The layer abstraction enables flexible composition of simulation pipelines:
    - Layers are processed sequentially by the Context
    - Multiple layers can be combined in parallel for beam splitting
    - Each layer receives a wavefront and returns a transformed wavefront
    
    Examples
    --------
    >>> class CustomLayer(Layer):
    ...     def process(self, wavefront, context):
    ...         # Apply custom transformation
    ...         wavefront.field *= np.exp(1j * phase_pattern)
    ...         return wavefront
    
    See Also
    --------
    Context : Orchestrates layer execution
    """
    def __init__(self):
        pass

    def process(self, wavefront: Any, context: 'Context') -> Any:
        """
        Process the incoming wavefront/signal and return the result.
        
        This method must be implemented by all subclasses. It defines how
        the layer transforms the electromagnetic field or signal.
        
        Parameters
        ----------
        wavefront : Wavefront or list of Wavefront
            The input electromagnetic field(s) to process. For parallel layers,
            this may be a list of wavefronts.
        context : Context
            The simulation context providing global parameters (time, observation
            conditions, etc.)
        
        Returns
        -------
        wavefront : Wavefront or list of Wavefront or ndarray
            The transformed wavefront(s). Terminal layers (e.g., Camera) may
            return numpy arrays instead of Wavefront objects.
        
        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement process()")

class Context:
    """
    Main simulation context managing layers and execution.
    
    The Context orchestrates the simulation pipeline by sequentially processing
    layers. It maintains global simulation parameters and executes the observation
    workflow from scene generation through optical propagation to detector output.
    
    Parameters
    ----------
    date : str or datetime, optional
        Observation date/time for astronomical calculations
    declination : Quantity, optional
        Target declination for coordinate transformations
    **kwargs : dict
        Additional context parameters
    
    Attributes
    ----------
    layers : list of Layer or list of list of Layer
        Ordered sequence of simulation layers. Single layers process sequentially,
        lists of layers process in parallel (beam splitting)
    results : dict
        Dictionary to store intermediate or final results
    
    Examples
    --------
    Build a complete observation pipeline:
    
    >>> import helios
    >>> from astropy import units as u
    >>> 
    >>> # Create scene
    >>> scene = helios.Scene(distance=10*u.pc)
    >>> scene.add(helios.Star(temperature=5700*u.K, magnitude=5))
    >>> 
    >>> # Create optical system
    >>> collectors = helios.Collectors()
    >>> collectors.add(size=8*u.m, shape=helios.Pupil.like('VLT'))
    >>> 
    >>> # Create detector
    >>> camera = helios.Camera(pixels=(512, 512))
    >>> 
    >>> # Build context and run
    >>> ctx = Context()
    >>> ctx.add_layer(scene)
    >>> ctx.add_layer(collectors)
    >>> ctx.add_layer(camera)
    >>> image = ctx.observe()
    
    See Also
    --------
    Layer : Base class for all simulation components
    """
    def __init__(self, date: Any = None, declination: Any = None, **kwargs):
        self.date = date
        self.declination = declination
        self.layers: List[Union[Layer, List[Layer]]] = []
        self.results = {}

    def add_layer(self, layer: Union[Layer, List[Layer]]):
        """
        Add a layer or a list of parallel layers to the simulation.
        
        Layers are executed in the order they are added. To create parallel
        processing (e.g., beam splitting), pass a list of layers.
        
        Parameters
        ----------
        layer : Layer or list of Layer
            Single layer for sequential processing, or list of layers for
            parallel processing (e.g., splitting to multiple detectors)
        
        Examples
        --------
        Sequential layers:
        
        >>> ctx.add_layer(scene)
        >>> ctx.add_layer(atmosphere)
        >>> ctx.add_layer(camera)
        
        Parallel layers (beam splitting):
        
        >>> ctx.add_layer(beam_splitter)
        >>> ctx.add_layer([camera1, camera2])  # Both receive split beams
        """
        self.layers.append(layer)

    def observe(self) -> Any:
        """
        Run the simulation through all layers.
        
        Executes the complete observation pipeline by sequentially processing
        each layer. The output of one layer becomes the input to the next.
        
        Returns
        -------
        output : ndarray or Wavefront or list
            The final output from the last layer. Typically a numpy array
            from a Camera detector, but may be a Wavefront or list of outputs
            from other terminal layers.
        
        Examples
        --------
        >>> ctx = Context()
        >>> ctx.add_layer(scene)
        >>> ctx.add_layer(collectors)
        >>> ctx.add_layer(camera)
        >>> image = ctx.observe()  # Returns 2D numpy array
        >>> print(image.shape)  # (512, 512)
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
