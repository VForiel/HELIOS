import numpy as np
from astropy import units as u
from typing import List, Union, Optional, Any, Tuple
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MPath
from pathlib import Path
import os
import xml.etree.ElementTree as ET
import re
import json
from .simulation import Wavefront, WavefrontArray

# Serialization Helpers
def serialize_value(value: Any) -> Any:
    """Recursively serialize values to JSON-friendly types."""
    if isinstance(value, u.Quantity):
        return {"value": float(value.value), "unit": str(value.unit)}
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    elif hasattr(value, 'to_dict'):
        return value.to_dict()
    return value

def deserialize_value(value: Any) -> Any:
    """Recursively deserialize values from JSON types."""
    if isinstance(value, dict):
        if "value" in value and "unit" in value and len(value) == 2:
            try:
                return value["value"] * u.Unit(value["unit"])
            except Exception:
                pass # Not a quantity dict
        return {k: deserialize_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [deserialize_value(v) for v in value]
    return value


class Element:
    """
    Base class for all simulation elements (physical components).
    
    An Element represents a physical component in the optical system that can
    process wavefronts independently. Elements are grouped within Layers for
    parallel processing.
    
    Parameters
    ----------
    name : str, optional
        Descriptive name for this element (used in diagrams and logging)
    
    Attributes
    ----------
    name : str
        Descriptive name for this element
    layer : Layer
        Reference to the parent layer containing this element
    pipeline : Pipeline
        Shortcut to access the parent pipeline (equivalent to self.layer.pipeline)
    
    Examples
    --------
    >>> class CustomElement(Element):
    ...     def __init__(self, parameter, name=None):
    ...         super().__init__(name=name or "CustomElement")
    ...         self.parameter = parameter
    ...
    ...     def process(self, wavefront, pipeline):
    ...         # Apply custom transformation
    ...         wavefront.field *= self.parameter
    ...         return wavefront
    """
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.layer: Optional['Layer'] = None
        self.pipeline: Optional['Pipeline'] = None
        self.num_inputs: int = 1  # Number of inputs this element consumes
        self.num_outputs: int = 1 # Number of outputs this element produces
        self.metadata: dict = {}  # Store for UI/Application specific data

    def twin(self) -> 'Element':
        """
        Create a twin copy of this element.
        
        A twin is a deep copy of the element that preserves the reference to its
        parent layer (if any). This is useful for creating multiple instances
        of the same component type that share the same physical container (e.g.,
        multiple MMI components on the same PhotonicChip).
        
        Returns
        -------
        Element
            A new instance of the element with identical attributes but sharing
            the parent layer reference.
        """
        # Create a deep copy of the element
        # We need to temporarily detach the layer to avoid deep copying the parent
        parent_layer = self.layer
        self.layer = None
        
        try:
            new_element = copy.deepcopy(self)
        finally:
            # Restore the layer reference on the original object
            self.layer = parent_layer
            
        # Restore the layer reference on the new object
        new_element.layer = parent_layer
        
        return new_element

    def description(self, indent: int = 0, full: bool = False) -> str:
        """
        Generate a text description of this element.
        
        Parameters
        ----------
        indent : int, optional
            Number of spaces to indent the description (for hierarchical display)
        full : bool, optional
            If True, include detailed parameters and attributes (default: False)
        
        Returns
        -------
        str
            Formatted description of the element
        
        Examples
        --------
        >>> element = CustomElement()
        >>> print(element.description())
        CustomElement
        >>> print(element.description(full=True))
        CustomElement
        >>>   - parameter: value
        """
        prefix = " " * indent
        class_name = self.__class__.__name__
        name_str = f" '{self.name}'" if self.name else ""
        
        result = f"{prefix}{class_name}{name_str}"
        
        if full:
            # Add detailed attributes (subclasses should override this)
            details = self._get_detailed_attributes()
            if details:
                for key, value in details.items():
                    result += f"\n{prefix}  • {key}: {value}"
        
        return result
    
    def _get_detailed_attributes(self) -> dict:
        """
        Return a dictionary of detailed attributes for full description.
        
        Subclasses should override this method to provide specific parameters.
        
        Returns
        -------
        dict
            Dictionary of attribute names and their string representations
        """
        return {}

    def process(self, wavefront: Any) -> Any:
        """
        Process the incoming wavefront/signal and return the result.
        
        This method must be implemented by all subclasses. It defines how
        the element transforms the electromagnetic field or signal.
        
        Parameters
        ----------
        wavefront : Wavefront or list of Wavefront
            The input electromagnetic field(s) to process
        
        Returns
        -------
        wavefront : Wavefront or list of Wavefront or ndarray
            The transformed wavefront(s). Terminal elements (e.g., Camera) may
            return numpy arrays instead of Wavefront objects.
        
        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement process()")

    def to_dict(self) -> dict:
        """
        Serialize element configuration to dictionary.
        
        Returns
        -------
        dict
            Dictionary containing class info and attributes.
        """
        return {
            "type": self.__class__.__name__,
            "module": self.__class__.__module__,
            "type": self.__class__.__name__,
            "module": self.__class__.__module__,
            "name": self.name,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Element':
        """
        Create element instance from dictionary.
        
        Parameters
        ----------
        data : dict
            Dictionary containing element configuration
            
        Returns
        -------
        Element
            New element instance
        """
        name = data.get("name")
        name = data.get("name")
        elem = cls(name=name)
        elem.metadata = data.get("metadata", {})
        return elem


class Layer:
    """
    Base class for all simulation layers (logical grouping of elements).
    
    A Layer represents a logical stage in the simulation pipeline and contains
    one or more Elements that process wavefronts in parallel.
    
    The layer abstraction enables flexible composition of simulation pipelines:
    - Layers are processed sequentially by the Context
    - Multiple layers can be combined in parallel for beam splitting
    - Each layer receives a wavefront and returns a transformed wavefront
    
    Parameters
    ----------
    name : str, optional
        Descriptive name for this layer (used in diagrams and logging)
    
    Attributes
    ----------
    elements : list of Element
        Physical components contained in this layer
    pipeline : Pipeline
        Reference to the parent pipeline managing this layer
    
    Examples
    --------
    >>> class CustomLayer(Layer):
    ...     def __init__(self, name=None):
    ...         super().__init__(name=name or "CustomLayer")
    ...
    ...     def process(self, wavefront, pipeline):
    ...         # Apply custom transformation
    ...         wavefront.field *= np.exp(1j * phase_pattern)
    ...         return wavefront
    
    See Also
    --------
    Pipeline : Orchestrates layer execution
    Element : Physical components within layers
    """
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.elements: List[Element] = []
        self.pipeline: Optional['Pipeline'] = None
        self.metadata: dict = {} # Store for UI/Application specific data
        self.num_inputs: int = 1  # Number of inputs this layer consumes (if single layer)
        
        # Caching
        self._cached_input: Any = None
        self._cached_output: Any = None
    
    num_outputs: int = 1 # Default number of outputs

    def invalidate_cache(self):
        """
        Invalidate the cache of this layer and trigger propagation to downstream layers.
        """
        self._cached_input = None
        self._cached_output = None
        if self.pipeline:
            self.pipeline.invalidate_downstream_cache(self)

    def get_input_wavefront(self) -> Any:
        """
        Retrieve the input wavefront for this layer, efficiently using cache or pulling from previous layer.
        """
        if self._cached_input is not None:
            return self._cached_input
            
        if self.pipeline is None:
            return None

        # Logic to find previous output
        prev_output = self.pipeline.get_previous_layer_output(self)
        self._cached_input = prev_output
        return prev_output

    def get_output_wavefront(self) -> Any:
        """
        Retrieve the output wavefront of this layer, processing if not cached.
        """
        if self._cached_output is not None:
            return self._cached_output
        
        # Pull input
        input_wf = self.get_input_wavefront()
        
        # Process (Process is responsible for handling None inputs if strictly needed, 
        # but generally get_input_wavefront should provide it)
        # However, for GenerationLayer, input might be None or implicit.
        
        # If input is None and we are NOT a GenerationLayer, we might have an issue 
        # unless the layer can generate data (Source).
        
        result = self.process(input_wf)
        self._cached_output = result
        return result
    
    def twin(self) -> 'Layer':
        """
        Create a twin copy of this layer.
        
        A twin is a deep copy of the layer that preserves the reference to its
        parent pipeline (if any) and potentially other shared resources.
        
        Returns
        -------
        Layer
            A new instance of the layer with identical attributes.
        """
        # Similar logic to Element.twin() if needed, but Layer usually doesn't have a 'parent' 
        # in the same way Element has 'layer'. Layer has 'pipeline'.
        # However, the user asked for "elements" to have twin().
        # Since Layer is also used as a component (e.g. MMI inherits from Layer),
        # we should implement it here too or ensure MMI inherits from Element?
        # Wait, in HELIOS, components like MMI inherit from Layer, not Element?
        # Let's check photonics.py.
        
        # If MMI inherits from Layer, then we need twin() on Layer.
        # But Layer has 'pipeline', not 'layer'.
        # The user said "conservent leurs éléments parents".
        # If MMI is a Layer, does it have a parent?
        # In generate_photonics_uml.py:
        # chip = photonics.PhotonicChip(...)
        # mmi.layer = chip
        # So MMI *does* have a .layer attribute if it's part of a chip.
        # But Layer class definition doesn't have .layer attribute by default.
        # It seems components might dynamically get .layer attribute or inherit from something else?
        
        # Let's check if Layer has .layer attribute.
        # In the provided code for Layer class:
        # def __init__(self, name=None):
        #     self.name = name
        #     self.elements = []
        #     self.pipeline = None
        
        # It does NOT have self.layer.
        # But in generate_photonics_uml.py:
        # for elem in [fiber_in, tops, mmi, cross, fiber_out, fiber_out]:
        #     elem.layer = chip
        
        # So it's added dynamically.
        
        # So we should implement twin() on Layer as well, handling .layer if it exists.
        
        parent_layer = getattr(self, 'layer', None)
        if parent_layer is not None:
            self.layer = None
            
        try:
            new_layer = copy.deepcopy(self)
        finally:
            if parent_layer is not None:
                self.layer = parent_layer
                
        if parent_layer is not None:
            new_layer.layer = parent_layer
            
        return new_layer

    def add_element(self, element: Element):
        """
        Add an element to this layer.
        
        Automatically sets the element's layer and pipeline references.
        
        Parameters
        ----------
        element : Element
            The element to add to this layer
        """
        self.elements.append(element)
        element.layer = self
        # Set pipeline if the layer is already attached to a pipeline
        if self.pipeline is not None:
            element.pipeline = self.pipeline

    def description(self, indent: int = 0, full: bool = False) -> str:
        """
        Generate a text description of this layer and all its elements.
        
        Parameters
        ----------
        indent : int, optional
            Number of spaces to indent the description (for hierarchical display)
        full : bool, optional
            If True, include detailed parameters and attributes (default: False)
        
        Returns
        -------
        str
            Formatted description of the layer and all sub-elements
        
        Examples
        --------
        >>> layer = CustomLayer()
        >>> layer.add_element(CustomElement())
        >>> print(layer.description())
        CustomLayer
        >>>   └─ CustomElement
        >>> print(layer.description(full=True))
        CustomLayer
        >>>   • parameter: value
        >>>   └─ CustomElement
        >>>     • element_param: value
        """
        prefix = " " * indent
        class_name = self.__class__.__name__
        name_str = f" '{self.name}'" if self.name else ""
        
        lines = [f"{prefix}{class_name}{name_str}"]
        
        # Add detailed attributes if full mode
        if full:
            details = self._get_detailed_attributes()
            if details:
                for key, value in details.items():
                    lines.append(f"{prefix}  • {key}: {value}")
        
        # Add elements if any
        if self.elements:
            for i, element in enumerate(self.elements):
                is_last = (i == len(self.elements) - 1)
                connector = "└─" if is_last else "├─"
                elem_desc = element.description(0, full=full)
                # Indent multi-line descriptions properly
                elem_lines = elem_desc.split('\n')
                lines.append(f"{prefix}  {connector} {elem_lines[0]}")
                if len(elem_lines) > 1:
                    continuation = "  " if is_last else "│ "
                    for line in elem_lines[1:]:
                        lines.append(f"{prefix}  {continuation} {line}")
        
        return "\n".join(lines)
    
    def _get_detailed_attributes(self) -> dict:
        """
        Return a dictionary of detailed attributes for full description.
        
        Subclasses should override this method to provide specific parameters.
        
        Returns
        -------
        dict
            Dictionary of attribute names and their string representations
        """
        return {}

    def process(self, wavefront: Any) -> Any:
        """
        Process the incoming wavefront/signal and return the result.
        
        This method must be implemented by all subclasses. It defines how
        the layer transforms the electromagnetic field or signal.
        
        Parameters
        ----------
        wavefront : Wavefront or list of Wavefront
            The input electromagnetic field(s) to process. For parallel layers,
            this may be a list of wavefronts.
        
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

    def to_dict(self) -> dict:
        """
        Serialize layer configuration to dictionary.
        """
        return {
            "type": self.__class__.__name__,
            "module": self.__class__.__module__,
            "name": self.name,
            "metadata": self.metadata,
            "elements": [e.to_dict() for e in self.elements]
        }
    
    @classmethod
    def from_dict(cls, data: dict, context: Optional['Context'] = None) -> 'Layer':
        """
        Create layer instance from dictionary.
        """
        name = data.get("name")
        name = data.get("name")
        layer = cls(name=name)
        layer.metadata = data.get("metadata", {})
        return layer

# =============================================================================
# LAYER TYPES
# =============================================================================

class GenerationLayer(Layer):
    """
    Layer generating the continuous electromagnetic field (Scene, Atmosphere).
    Output: Single continuous Wavefront/Field.
    """
    pass

class SamplingLayer(Layer):
    """
    Layer sampling the continuous field into discrete optical paths (TelescopeArray).
    Input: Single continuous Wavefront/Field.
    Output: Array of discrete Wavefronts (Optical Beams).
    """
    pass

class OpticalLayer(Layer):
    """
    Layer propagating/modifying optical beams (Lenses, Mirrors, BeamSplitters).
    Input: Optical Beam(s).
    Output: Optical Beam(s).
    """
    pass

class DetectionLayer(Layer):
    """
    Layer converting photons to digital data (Camera, Detector).
    Input: Optical Beam(s).
    Output: Data Array.
    """
    pass

class DataLayer(Layer):
    """
    Layer processing digital data (Algorithms, Stackers).
    Input: Data Array.
    Output: Data Array.
    """
    pass

class Pipeline:
    """
    Main simulation pipeline managing layers and execution.
    
    The Pipeline orchestrates the simulation by sequentially processing
    layers. It maintains global simulation parameters and executes the observation
    workflow from scene generation through optical propagation to detector output.
    
    Parameters
    ----------
    date : str or datetime, optional
        Observation date/time for astronomical calculations
    declination : Quantity, optional
        Target declination for coordinate transformations
    kwargs : dict
        Additional pipeline parameters
    
    Attributes
    ----------
    layers : list of Layer or list of list of Layer
        Ordered sequence of simulation layers. Single layers process sequentially,
        lists of layers process in parallel (beam splitting)
    results : dict
        Dictionary to store intermediate or final results
    """
    def __init__(self, date: Any = None, declination: Any = None, layers: Optional[List[Union[Layer, List[Layer]]]] = None, **kwargs):
        self.date = date
        self.declination = declination
        self.kwargs = kwargs
        self.layers: List[Union[Layer, List[Layer]]] = []
        self.results = {}
        
        if layers:
            for layer in layers:
                self.add_layer(layer)

    def invalidate_downstream_cache(self, start_layer: Layer):
        """
        Invalidate cache for all layers downstream of the given layer.
        """
        start_idx = -1
        # Find index
        for i, l_item in enumerate(self.layers):
            if isinstance(l_item, list):
                if start_layer in l_item:
                    start_idx = i
                    break
            else:
                if l_item is start_layer:
                    start_idx = i
                    break
        
        if start_idx == -1:
            return 
            
        # Invalidate all subsequent layers
        for i in range(start_idx + 1, len(self.layers)):
            l_item = self.layers[i]
            if isinstance(l_item, list):
                for sub_l in l_item:
                    if sub_l:
                        # Clear cache effectively
                        sub_l._cached_input = None
                        sub_l._cached_output = None
            else:
                l_item._cached_input = None
                l_item._cached_output = None

    def get_previous_layer_output(self, current_layer: Layer) -> Any:
        """
        Get the output from the layer immediately preceding the current_layer.
        """
        curr_idx = -1
        for i, l_item in enumerate(self.layers):
            if isinstance(l_item, list):
                if current_layer in l_item:
                    curr_idx = i
                    break
            else:
                if l_item is current_layer:
                    curr_idx = i
                    break
        
        if curr_idx <= 0:
            return None # No previous layer
            
        prev_item = self.layers[curr_idx - 1]
        
        # If previous is a list (Parallel), we need to handle merging or specific routing
        # For simplicity in this logic, we assume we want the output of that "stage".
        # If the current layer is inside a parallel block, getting input is complex (routing).
        # We rely on existing logic where "Parallel blocks" usually output a flat list or specific struct.
        
        # Simplification: If previous is single layer, call get_output_wavefront
        if not isinstance(prev_item, list):
            return prev_item.get_output_wavefront()
            
        # If previous is parallel, we collect outputs
        outputs = []
        for sub in prev_item:
            if sub:
                outputs.append(sub.get_output_wavefront())
        return outputs

    def add_layer(self, layer: Union[Layer, List[Layer]]):
        """
        Add a layer or a list of parallel layers to the simulation.
        
        Layers are executed in the order they are added. To create parallel
        processing (e.g., beam splitting), pass a list of layers.
        
        Automatically sets the layer's pipeline reference and propagates to elements.
        
        Parameters
        ----------
        layer : Layer or list of Layer
            Single layer for sequential processing, or list of layers for
            parallel processing (e.g., splitting to multiple detectors)
        
        Examples
        --------
        Sequential layers:
        
        >>> pipe.add_layer(scene)
        >>> pipe.add_layer(atmosphere)
        >>> pipe.add_layer(camera)
        
        Parallel layers (beam splitting):
        
        >>> pipe.add_layer(beam_splitter)
        >>> pipe.add_layer([camera1, camera2])  # Both receive split beams
        """
        self.layers.append(layer)
        
        # Set pipeline reference for layer(s)
        if isinstance(layer, list):
            for l in layer:
                if l is not None:
                    if l.pipeline is not None and l.pipeline is not self:
                        import warnings
                        warnings.warn(f"Layer {l} is being moved from Pipeline {l.pipeline} to {self}.")
                    l.pipeline = self
                    # Propagate to elements if layer has them
                    if hasattr(l, 'elements') and l.elements:
                        for element in l.elements:
                            element.pipeline = self
        else:
            if layer.pipeline is not None and layer.pipeline is not self:
                import warnings
                warnings.warn(f"Layer {layer} is being moved from Pipeline {layer.pipeline} to {self}.")
            layer.pipeline = self
            # Propagate to elements if layer has them
            if hasattr(layer, 'elements') and layer.elements:
                for element in layer.elements:
                    element.pipeline = self

    def description(self, full: bool = False) -> str:
        """
        Generate a complete text description of the entire simulation setup.
        
        Parameters
        ----------
        full : bool, optional
            If True, include detailed parameters and attributes for all components (default: False)
        
        Returns
        -------
        str
            Formatted description of all layers and elements in the pipeline
        
        Examples
        --------
        >>> pipe = Pipeline()
        >>> pipe.add_layer(scene)
        >>> pipe.add_layer(telescope)
        >>> pipe.add_layer(camera)
        >>> print(pipe.description())
        HELIOS Simulation Pipeline
        ========================
        Layer 1: Scene
        Layer 2: TelescopeArray
        >>>   └─ Collector 1
        Layer 3: Camera
        
        >>> print(pipe.description(full=True))
        HELIOS Simulation Pipeline
        ========================
        Pipeline Parameters:
        >>>   • date: 2025-01-01
        >>>   • declination: 10.0 deg
        
        Layer 1: Scene 'Target'
        >>>   • distance: 10.0 pc
        >>>   └─ Star
        >>>     • temperature: 5700 K
        >>>     • magnitude: 5.0
        ...
        """
        lines = ["HELIOS Simulation Pipeline", "=" * 50, ""]
        
        # Add pipeline parameters if full mode
        if full:
            pipe_params = []
            if self.date is not None:
                pipe_params.append(f"  • date: {self.date}")
            if self.declination is not None:
                pipe_params.append(f"  • declination: {self.declination}")
            if pipe_params:
                lines.append("Pipeline Parameters:")
                lines.extend(pipe_params)
                lines.append("")
        
        for i, layer_item in enumerate(self.layers, 1):
            if isinstance(layer_item, list):
                # Parallel layers
                lines.append(f"Layer {i}: [Parallel Layers]")
                for j, layer in enumerate(layer_item, 1):
                    lines.append(f"  Branch {j}:")
                    if layer is None:
                        lines.append("    [Pass-through]")
                    else:
                        layer_desc = layer.description(indent=4, full=full)
                        lines.append(layer_desc)
            else:
                # Single layer
                lines.append(f"Layer {i}: {layer_item.description(full=full)}")
            lines.append("")  # Empty line between layers
        
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        Serialize complete pipeline to dictionary.
        
        Returns
        -------
        dict
            Complete simulation state found in JSON-compatible format.
        """
        layers_data = []
        for layer_item in self.layers:
            if isinstance(layer_item, list):
                # Parallel layers
                layers_data.append([
                    l.to_dict() if l is not None else None 
                    for l in layer_item
                ])
            else:
                layers_data.append(layer_item.to_dict())
                
        return {
            "date": str(self.date) if self.date else None,
            "declination": serialize_value(self.declination),
            "kwargs": serialize_value(self.kwargs),
            "layers": layers_data
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Pipeline':
        """
        Reconstruct pipeline from dictionary.
        """
        # Basic pipeline params
        date = data.get("date")
        declination = deserialize_value(data.get("declination"))
        kwargs = deserialize_value(data.get("kwargs", {}))
        
        pipe = cls(date=date, declination=declination, **kwargs)
        
        # Reconstruct layers
        from ..components import scene, atmosphere, collector, detectors
        from .context import Layer # import self for check? No need.
        
        # Mapping of type names to classes
        type_map = {
            'Scene': scene.Scene,
            'Atmosphere': atmosphere.Atmosphere,
            'TelescopeArray': collector.TelescopeArray,
            'Camera': detectors.Camera
        }
        
        def restore_layer(l_data):
            if l_data is None: return None
            
            type_name = l_data.get("type")
            if type_name in type_map:
                try:
                    return type_map[type_name].from_dict(l_data)
                except Exception as e:
                    print(f"Error restoring layer {type_name}: {e}")
                    return None
            else:
                print(f"Unknown layer type: {type_name}")
                return None

        layers_data = data.get("layers", [])
        for l_item in layers_data:
            if isinstance(l_item, list):
                # Parallel
                parallel_layers = [restore_layer(ld) for ld in l_item]
                ctx.add_layer(parallel_layers)
            else:
                layer = restore_layer(l_item)
                if layer:
                    ctx.add_layer(layer)
                    
        return ctx

    def save(self, filename: Union[str, Path]):
        """Save context to a JSON file."""
        data = self.to_dict()
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load(cls, filename: Union[str, Path]) -> 'Context':
        """Load context from a JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_input_wavefront(self, wavelength: Optional[u.Quantity] = None, 
                            size: Optional[Union[int, u.Quantity]] = None,
                            npix: Optional[int] = None,
                            angular_samples: int = 1,
                            coherent_sources: bool = True,
                            collectors: Optional[List[Any]] = None) -> Union[Wavefront, WavefrontArray]:
        """
        Generate the input wavefront(s) from Scene and Atmosphere in the context.
        
        Retrieves simulation parameters (wavelength, npix) from context.kwargs,
        creates a new Wavefront, and applies Scene flux and Atmosphere phase.

        Parameters
        ----------
        wavelength : astropy.Quantity, optional
            Wavelength of the wavefront. If None, uses context.kwargs['wavelength'] or default 550nm.
        size : astropy.Quantity or int, optional
            Physical size of the wavefront (e.g. 10*u.m). 
            If int is provided, it is treated as npix (pixels) for backward compatibility.
        npix : int, optional
            Number of pixels. If None, uses context.kwargs['npix'] or default 512.
        angular_samples : int, optional
            Number of angular samples along one dimension for extended sources (default: 1).
            Total samples = angular_samples^2 if coherent_sources=False.
        coherent_sources : bool, optional
            If True, creates one wavefront sample per discrete source (Star, Planet).
            If False, creates a grid of angular_samples^2 wavefronts sampling the scene.
            Default: True.
        collectors : list of Collector, optional
            If provided, generates a WavefrontArray with one wavefront per collector,
            including geometric phase shifts (piston + tilt) corresponding to each collector's position.
        """
        # Handle backward compatibility for size (int -> npix)
        if isinstance(size, int):
            if npix is None:
                npix = size
            size = None # Reset size so we use default physical size later
            
        # Determine simulation parameters
        if wavelength is None:
            wavelength = self.kwargs.get('wavelength', 550 * u.nm)
        if npix is None:
            npix = self.kwargs.get('npix', 512)
            
        # Physical size default
        if size is None:
             size = self.kwargs.get('diameter', 10.0 * u.m)
            
        # Find Scene
        scene = None
        # DEBUG
        with open("debug_log.txt", "a") as f:
            f.write(f"Context layers: {len(self.layers)}\n")
            for l in self.layers:
                f.write(f"  - {type(l).__name__} ({type(l)})\n")

        for layer in self.layers:
            if type(layer).__name__ == 'Scene':
                scene = layer
            elif type(layer).__name__ == 'TelescopeArray' and collectors is None:
                # Auto-detect TelescopeArray if collectors not explicitly provided
                collectors = layer.elements
        
        # Error handling: Need at least a Scene or a TelescopeArray (to define collectors)
        if scene is None and collectors is None:
             raise ValueError("Context must contain at least a Scene or a TelescopeArray to generate input wavefront.")
        
        # Determine samples and directions
        samples = 1
        directions = [(0.0, 0.0)] # (theta_x, theta_y) in radians
        amplitudes = [1.0]
        sources_list = ["Default Source"]
        
        if scene:
            # DEBUG
            with open("debug_log.txt", "a") as f:
                f.write(f"Scene found. Coherent: {coherent_sources}\n")
                f.write(f"Scene objects: {len(scene.objects)}\n")
                for o in scene.objects:
                    f.write(f"  - {type(o).__name__}\n")

            if coherent_sources:
                # One sample per source
                # We look for objects that are likely point sources or discrete bodies
                scene_objects = [obj for obj in scene.objects if type(obj).__name__ in ['Star', 'Planet']]
                
                if not scene_objects and len(scene.objects) > 0:
                     # Fallback if no Star/Planet but other objects exist
                     scene_objects = scene.objects
                
                if scene_objects:
                    samples = len(scene_objects)
                    directions = []
                    amplitudes = []
                    sources_list = []
                    
                    # Distance for angular conversion
                    dist = getattr(scene, 'distance', None)
                    
                    for obj in scene_objects:
                        # Position
                        px, py = 0.0 * u.rad, 0.0 * u.rad
                        if hasattr(obj, 'position'):
                            pos = obj.position
                            if len(pos) == 2:
                                px, py = pos
                        
                        # Convert to radians
                        tx, ty = 0.0, 0.0
                        
                        # DEBUG
                        with open("debug_log.txt", "a") as f:
                            f.write(f"Object: {type(obj).__name__}, Position: {px}, {py}\n")
                            if hasattr(px, 'unit'):
                                f.write(f"  Unit: {px.unit}, Equiv rad: {px.unit.is_equivalent(u.rad)}\n")
                        
                        if hasattr(px, 'unit'):
                            if px.unit.is_equivalent(u.m) and dist is not None:
                                tx = (px / dist).to(u.rad, equivalencies=u.dimensionless_angles()).value
                                ty = (py / dist).to(u.rad, equivalencies=u.dimensionless_angles()).value
                            elif px.unit.is_equivalent(u.rad):
                                tx = px.to(u.rad).value
                                ty = py.to(u.rad).value
                        
                        directions.append((tx, ty))
                        
                        # Amplitude (Flux)
                        # Fallback to magnitude scaling as in Scene.get_flux_scaling
                        d_factor = 1.0
                        if dist is not None:
                            d_ref = 10 * u.pc
                            d_factor = (d_ref / dist).to(u.dimensionless_unscaled).value**2
                            
                        mag = getattr(obj, 'magnitude', 0.0)
                        mag_factor = 10**(-0.4 * mag)
                        
                        flux = d_factor * mag_factor
                        amplitudes.append(np.sqrt(flux))
                        
                        # Source name
                        name = getattr(obj, 'name', None)
                        if not name:
                            name = type(obj).__name__
                        sources_list.append(name)
            else:
                # Grid sampling (Extended source mode)
                samples = angular_samples ** 2
                
                # Render scene to get spatial distribution
                fov = 2.0 * u.arcsec # Default
                
                if hasattr(scene, 'render'):
                    try:
                        img, x, y = scene.render(npix=angular_samples, fov=fov, return_coords=True)
                        # img is intensity map, x, y are 1D arrays of coordinates in arcsec
                        
                        # Flatten
                        amplitudes = np.sqrt(img.flatten())
                        
                        # Directions
                        xg, yg = np.meshgrid(x, y)
                        tx = xg.flatten().to(u.rad).value
                        ty = yg.flatten().to(u.rad).value
                        directions = list(zip(tx, ty))
                        # Store sources as Quantities for better formatting later
                        sources_list = [np.array([txi, tyi]) * u.rad for txi, tyi in zip(tx, ty)]
                        
                    except Exception as e:
                        print(f"Warning: Scene rendering failed: {e}. Using default source.")
                        samples = 1
                        directions = [(0.0, 0.0)]
                        amplitudes = [1.0]
                        sources_list = ["Default Source"]
                else:
                    samples = 1
                    directions = [(0.0, 0.0)]
                    amplitudes = [1.0]
                    sources_list = ["Default Source"]

        # If collectors are provided, generate WavefrontArray
        if collectors is not None:
            wf_list = []
            locations = []
            
            # Wavenumber
            k = 2 * np.pi / wavelength.to(u.m).value
            
            for collector in collectors:
                # Determine collector size for local grid
                if hasattr(collector, 'size') and collector.size is not None:
                    diameter = collector.size
                else:
                    diameter = 1.0 * u.m # Default

                # Create wavefront for this collector
                wf = Wavefront(wavelength=wavelength, size=diameter, npix=npix, nsource=samples)
                # Ensure 3D field shape even for single sample to satisfy downstream expectations
                if samples == 1 and wf.ndim == 2:
                    wf = wf[np.newaxis, ...]
                wf.sources = sources_list
                wf.source_directions = np.array(directions) * u.rad
                
                # DEBUG
                # print(f"Collector: {collector.name}, Size: {diameter}, Type: {type(diameter)}")
                
                # wf.pixel_scale is already set by Wavefront constructor
                try:
                    size_m = diameter.to(u.m).value
                except AttributeError:
                    print(f"ERROR: diameter has no .to() method. Type: {type(diameter)}, Value: {diameter}")
                    # Fallback if it's a float (assume meters)
                    size_m = float(diameter)
                    diameter = size_m * u.m
                    # Update pixel scale if diameter changed type
                    wf.pixel_scale = (diameter / npix)
                
                # Create local grid (u, v)
                u_vec = np.linspace(-size_m/2, size_m/2, npix)
                v_vec = np.linspace(-size_m/2, size_m/2, npix)
                U, V = np.meshgrid(u_vec, v_vec)
                
                # Collector position for piston
                if hasattr(collector, 'position'):
                    cx, cy = collector.position
                else:
                    cx, cy = 0.0, 0.0
                
                # Apply phase shifts
                for s in range(samples):
                    if s < len(directions):
                        tx, ty = directions[s] # radians

                        piston = k * (cx * tx + cy * ty)
                        tilt = k * (U * tx + V * ty)
                        total_phase = piston + tilt

                        phase_factor = np.exp(1j * total_phase)
                        # Support 2D fields when samples == 1
                        if wf.ndim == 3:
                            wf[s] *= phase_factor
                        else:
                            wf *= phase_factor
                # Set amplitudes
                for i in range(samples):
                    if i < len(amplitudes):
                        wf[i] *= amplitudes[i]
                
                wf_list.append(wf)
                locations.append((cx, cy))
            
            return WavefrontArray(wf_list, locations=locations)

        # Default single wavefront behavior (if no collectors provided)
        # Default diameter 10m if not specified
        # size variable already holds the physical size (defaulted to 10m if not provided)
        
        # Create wavefront with samples
        wf = Wavefront(wavelength=wavelength, size=size, npix=npix, nsource=samples)
        if samples == 1 and wf.ndim == 2:
            wf = wf[np.newaxis, ...]
        wf.sources = sources_list
        
        # Set directions
        wf.source_directions = np.array(directions) * u.rad
        
        # Apply phase tilt for off-axis sources
        # k = 2pi / lambda
        k = 2 * np.pi / wavelength.to(u.m).value
        
        # Create grid (u, v) in meters
        # centered on 0
        size_m = size.to(u.m).value
        u_vec = np.linspace(-size_m/2, size_m/2, npix)
        v_vec = np.linspace(-size_m/2, size_m/2, npix)
        U, V = np.meshgrid(u_vec, v_vec)
        
        for s in range(samples):
            if s < len(directions):
                tx, ty = directions[s] # radians

                tilt = k * (U * tx + V * ty)
                phase_factor = np.exp(1j * tilt)
                if wf.ndim == 3:
                    wf[s] *= phase_factor
                else:
                    wf *= phase_factor

        # Set amplitudes
        # wf is (samples, size, size)
        # We broadcast amplitude to (size, size)
        for i in range(samples):
            if i < len(amplitudes):
                wf[i] *= amplitudes[i]
        
        # Find Atmosphere and apply phase
        atmosphere = None
        for layer in self.layers:
            if type(layer).__name__ == 'Atmosphere':
                atmosphere = layer
                break
        
        if atmosphere:
            # Atmosphere.process might return Wavefront or WavefrontArray
            # It will detect this TelescopeArray in context and optimize (split) if needed
            wf = atmosphere.process(wf, self)
            
        return wf

    def propagate_until(self, target_layer: Any) -> Any:
        """
        Run the simulation pipeline until reaching the target layer.
        
        Returns the input signal destined for the target layer.
        """
        # Initial wavefront/signal
        current_signal = None

        for i, layer in enumerate(self.layers):
            # Check if this single layer is the target
            if layer is target_layer:
                return current_signal

            if isinstance(layer, list):
                # Check if target is inside this parallel block
                if target_layer in layer:
                    # Parallel processing logic to find input for target
                    # Ensure current_signal is a list
                    if not isinstance(current_signal, list):
                        current_signal = [current_signal] if current_signal is not None else []
                    
                    input_idx = 0
                    for sub_layer in layer:
                        # Determine how many inputs this element consumes
                        if sub_layer is None:
                            num_inputs = 1
                        elif hasattr(sub_layer, 'num_inputs'):
                            num_inputs = sub_layer.num_inputs
                        else:
                            num_inputs = 1
                        
                        # Gather inputs for this element
                        if input_idx + num_inputs > len(current_signal):
                            inputs = current_signal[input_idx:]
                        else:
                            inputs = current_signal[input_idx : input_idx + num_inputs]
                        
                        # If this is the target, return its inputs
                        if sub_layer is target_layer:
                             if num_inputs == 1 and len(inputs) == 1:
                                return inputs[0]
                             else:
                                return inputs
                        
                        input_idx += num_inputs
                    
                    # If target in list but not found (shouldn't happen), assume processed?
                    return None

                # Parallel processing (N-to-M routing) - Copied from observe
                outputs = []
                
                if not isinstance(current_signal, list):
                    current_signal = [current_signal] if current_signal is not None else []
                
                input_idx = 0
                for sub_layer in layer:
                    if sub_layer is None:
                        num_inputs = 1
                    elif hasattr(sub_layer, 'num_inputs'):
                        num_inputs = sub_layer.num_inputs
                    else:
                        num_inputs = 1
                    
                    if input_idx + num_inputs > len(current_signal):
                        inputs = current_signal[input_idx:]
                    else:
                        inputs = current_signal[input_idx : input_idx + num_inputs]
                    
                    input_idx += num_inputs
                    
                    if sub_layer is None:
                        outputs.extend(inputs)
                    else:
                        if num_inputs == 1 and len(inputs) == 1:
                            proc_input = inputs[0]
                        else:
                            proc_input = inputs
                            
                        result = sub_layer.process(proc_input)
                        
                        if isinstance(result, list):
                            outputs.extend(result)
                        else:
                            outputs.append(result)
                
                current_signal = outputs

            else:
                # Single layer processing
                current_signal = layer.process(current_signal)

        return current_signal

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
                # Parallel processing (N-to-M routing)
                outputs = []
                
                # Ensure current_signal is a list for consistent processing
                if not isinstance(current_signal, list):
                    current_signal = [current_signal] if current_signal is not None else []
                
                input_idx = 0
                for sub_layer in layer:
                    # Determine how many inputs this element consumes
                    if sub_layer is None:
                        num_inputs = 1
                    elif hasattr(sub_layer, 'num_inputs'):
                        num_inputs = sub_layer.num_inputs
                    else:
                        num_inputs = 1
                    
                    # Gather inputs for this element
                    if input_idx + num_inputs > len(current_signal):
                        # Not enough inputs available - this might be a configuration error
                        # or we might need to recycle inputs (broadcasting)
                        # For now, let's raise a warning or error, but strictly following
                        # the user request, we assume the user configures it correctly.
                        # Fallback: take what's left or None
                        inputs = current_signal[input_idx:]
                    else:
                        inputs = current_signal[input_idx : input_idx + num_inputs]
                    
                    input_idx += num_inputs
                    
                    # Process
                    if sub_layer is None:
                        # Pass-through
                        outputs.extend(inputs)
                    else:
                        # If the element expects a single input but we have a list of 1, unwrap it
                        # If it expects multiple, pass the list
                        if num_inputs == 1 and len(inputs) == 1:
                            proc_input = inputs[0]
                        else:
                            proc_input = inputs
                            
                        result = sub_layer.process(proc_input)
                        
                        # Result handling: always extend the outputs list
                        if isinstance(result, list):
                            outputs.extend(result)
                        else:
                            outputs.append(result)
                
                current_signal = outputs

            else:
                # Single layer
                # If current_signal is a list, this layer might merge them or process them individually
                # For now, let's assume if it receives a list, it processes the list (merging or keeping as list)
                # But typically a single layer after a split might be a detector array or a combiner.
                
                # Let's let the layer handle the input type
                current_signal = layer.process(current_signal)

        return current_signal

    def get_output_intensities(self):
        # Placeholder for interferometry output
        pass

    def validate_architecture(self):
        """
        Validate the simulation architecture.
        
        Checks if the number of outputs from each layer matches the number of
        inputs expected by the next layer.
        
        Raises
        ------
        ValueError
            If a mismatch is detected.
        """
        current_ports = 1 # Start with 1 (Scene)
        
        for i, layer in enumerate(self.layers):
            # Determine inputs expected by this layer
            if isinstance(layer, list):
                # Parallel layer
                expected_inputs = 0
                for elem in layer:
                    if hasattr(elem, 'num_inputs'):
                        expected_inputs += elem.num_inputs
                    else:
                        expected_inputs += 1
            else:
                # Single layer
                if hasattr(layer, 'num_inputs'):
                    expected_inputs = layer.num_inputs
                else:
                    expected_inputs = 1
            
            # Special case: TelescopeArray (Collector)
            # Can take 1 input (Scene) and produce N outputs
            # We skip input check if it's a TelescopeArray/Collector layer receiving from Scene/Atmosphere
            is_collector = False
            if isinstance(layer, list):
                if len(layer) > 0 and type(layer[0]).__name__ == 'Collector':
                    is_collector = True
            elif type(layer).__name__ == 'TelescopeArray':
                is_collector = True
                
            # Check inputs
            if not is_collector:
                # If mismatch, raise error
                # But allow broadcasting (1 -> N)
                if current_ports != expected_inputs and current_ports != 1:
                     print(f"Warning: Layer {i+1} ({self._get_display_name(layer) if not isinstance(layer, list) else 'Parallel'}) expects {expected_inputs} inputs but previous layer provides {current_ports} outputs.")
            
            # Determine outputs produced by this layer
            if isinstance(layer, list):
                current_ports = 0
                for elem in layer:
                    if hasattr(elem, 'num_outputs'):
                        current_ports += elem.num_outputs
                    else:
                        current_ports += 1
            else:
                if hasattr(layer, 'num_outputs'):
                    current_ports = layer.num_outputs
                elif hasattr(layer, 'elements') and len(layer.elements) > 0:
                    # TelescopeArray or similar container
                    # Assume it produces one output per element if it's a TelescopeArray
                    if type(layer).__name__ == 'TelescopeArray':
                        current_ports = len(layer.elements)
                    else:
                        current_ports = 1
                else:
                    current_ports = 1

        # Strict Type Validation
        for i in range(len(self.layers) - 1):
            curr = self.layers[i]
            next_l = self.layers[i+1]
            
            # Helper to get type(s)
            def get_types(item):
                if isinstance(item, list):
                    return {type(sub) for sub in item if sub}
                return {type(item)}
            
            curr_types = get_types(curr)
            next_types = get_types(next_l)
            
            # Check transitions
            for t_curr in curr_types:
                for t_next in next_types:
                    # Generation -> Generation (OK)
                    # Generation -> Sampling (OK)
                    # Sampling -> Optical (OK)
                    # Optical -> Optical (OK)
                    # Optical -> Detection (OK)
                    # Detection -> Data (OK)
                    # Data -> Data (OK)
                    
                    is_valid = False
                    
                    if issubclass(t_curr, GenerationLayer):
                        if issubclass(t_next, (GenerationLayer, SamplingLayer)): is_valid = True
                    elif issubclass(t_curr, SamplingLayer):
                        if issubclass(t_next, (OpticalLayer, DetectionLayer)): is_valid = True
                    elif issubclass(t_curr, OpticalLayer):
                        if issubclass(t_next, (OpticalLayer, DetectionLayer)): is_valid = True
                    elif issubclass(t_curr, DetectionLayer):
                        if issubclass(t_next, DataLayer): is_valid = True
                    elif issubclass(t_curr, DataLayer):
                        if issubclass(t_next, DataLayer): is_valid = True
                    # Fallback for legacy generic Layer
                    elif t_curr == Layer or t_next == Layer:
                         is_valid = True
                    # Fallback if users haven't updated their classes yet
                    else:
                         is_valid = True
                    
                    if not is_valid:
                         print(f"Warning: Invalid architecture transition from {t_curr.__name__} to {t_next.__name__}")


    def plot_uml_diagram(self, figsize: Tuple[float, float] = (16, 10), 
                         layer_spacing: float = 2.0,
                         save_path: Optional[str] = None,
                         return_type: str = 'figure') -> Union[plt.Figure, np.ndarray]:
        """
        Generate a UML-style diagram of the complete optical setup.
        
        This function creates a visual representation of the simulation pipeline,
        showing all layers from scene (left) to camera (right). Beam splitters
        create parallel paths that are displayed vertically.

        This visualization is particularly useful for:
        - **Documentation**: Quickly document complex optical systems
        - **Debugging**: Verify pipeline structure before running simulations
        - **Communication**: Share system designs with collaborators
        - **Teaching**: Explain optical concepts with clear visual diagrams
        
        Parameters
        ----------
        figsize : tuple of float, optional
            Figure size as (width, height) in inches. Default: (16, 10)
        layer_spacing : float, optional
            Horizontal distance between layers. Default: 2.0
        save_path : str, optional
            If provided, save the figure to this path
        return_type : str, optional
            Type of return value:
            - 'figure': Return matplotlib Figure object (default)
            - 'image': Return diagram as numpy array (RGB image)
            - 'both': Return tuple (figure, image_array)
        
        Returns
        -------
        fig : matplotlib.figure.Figure or ndarray or tuple
            Depending on return_type:
            - 'figure': The matplotlib Figure object
            - 'image': RGB numpy array of shape (H, W, 3) with values in [0, 255]
            - 'both': Tuple of (figure, image_array)

        Notes
        -----
        **Visual Features:**
        
        - **Left-to-Right Layout**: Diagrams are laid out from left (scene) to right (detector), 
          matching the physical light propagation path.
        - **Schematic Icons**: Each component is represented by a schematic icon:
            - **Scene**: Star with planets
            - **Telescope**: Circular aperture with spider vanes
            - **Atmosphere**: Wavy turbulence patterns
            - **Adaptive Optics**: Deformable mirror with actuators
            - **Coronagraph**: Focal plane mask
            - **Beam Splitter**: Diagonal mirror splitting beam
            - **Fibers**: Input/output coupling
            - **Photonics**: Integrated waveguide circuits
            - **Camera**: Detector array with pixels
            - **Interferometer**: Multiple telescopes with combiner
        - **Parallel Paths**: When beam splitters create multiple paths, they are displayed vertically.
        - **Component Labels**: Each component is labeled with its class name and custom name (if provided).
        - **Signal Flow**: Red arrows show the signal flow between components.
        
        Examples
        --------
        >>> import helios
        >>> from astropy import units as u
        >>> import matplotlib.pyplot as plt
        
        >>> # Create a simple pipeline
        >>> scene = helios.Scene(distance=10*u.pc)
        >>> scene.add(helios.Star(temperature=5700*u.K, magnitude=5))
        >>> telescope = helios.TelescopeArray(name="VLT")
        >>> telescope.add_collector(pupil=helios.Pupil.vlt(), position=(0, 0), size=8*u.m)
        >>> camera = helios.Camera(pixels=(512, 512))
        
        >>> # Build pipeline
        >>> pipeline = helios.Pipeline()
        >>> pipeline.add_layer(scene)
        >>> pipeline.add_layer(telescope)
        >>> pipeline.add_layer(camera)
        
        >>> # Generate diagram
        >>> fig = pipeline.plot_uml_diagram()
        >>> plt.show()
        
        >>> # Or save to file
        >>> pipeline.plot_uml_diagram(save_path='my_optical_system.png')

        >>> # Create dual-channel system with BeamSplitter
        >>> pipeline = helios.Pipeline()
        >>> pipeline.add_layer(scene)
        >>> pipeline.add_layer(telescope)
        >>> pipeline.add_layer(helios.BeamSplitter(cutoff=0.5))
        >>> pipeline.add_layer([camera, camera])  # Parallel paths shown vertically
        >>> fig = pipeline.plot_uml_diagram()
        """
        # Validate architecture first
        self.validate_architecture()
        
        fig, ax = plt.subplots(figsize=figsize)
        # ax.set_xlim will be set later or auto-scaled
        
        # Get asset directory
        asset_dir = Path(__file__).parent.parent / "assets"
        
        # Build layer tree structure to handle beam splitting
        layer_tree = self._build_layer_tree()
        max_paths = self._count_max_parallel_paths(layer_tree)
        
        # Set y-limits based on number of parallel paths
        y_margin = 1.0
        ax.set_ylim(-y_margin, max_paths + y_margin)
        
        # Draw each layer
        self._draw_layer_tree(ax, layer_tree, layer_spacing, asset_dir)
        
        # Configure axes
        ax.set_aspect('equal', adjustable='datalim')
        ax.axis('off')
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.set_title('HELIOS Optical System Diagram', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # Handle return type
        if return_type == 'figure':
            return fig
        elif return_type == 'image':
            # Convert figure to numpy array
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())
            image = image[:, :, :3]  # Keep only RGB channels
            plt.close(fig)
            return image
        elif return_type == 'both':
            # Return both figure and image
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())
            image = image[:, :, :3]  # Keep only RGB channels
            return fig, image
        else:
            raise ValueError(f"Invalid return_type: {return_type}. Must be 'figure', 'image', or 'both'")
    
    def _build_layer_tree(self) -> List[dict]:
        """
        Build a tree structure representing layer organization.
        
        Returns
        -------
        list of dict
            Each dict has 'layer' (Layer or list), 'x' (position), 'paths' (list of path indices)
        """
        tree = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, list):
                # Check if purely Swap/None layer
                is_pure_swap = True
                has_swap = False
                for elem in layer:
                    if elem is not None:
                        if type(elem).__name__ != 'Swap':
                            is_pure_swap = False
                            break
                        else:
                            has_swap = True
                
                if is_pure_swap and has_swap and len(layer) > 0:
                    # Calculate global mapping
                    global_mapping = []
                    current_in_offset = 0
                    
                    # We need a Swap class to instantiate. 
                    # Since we can't easily import it, we'll use the class of the first Swap found.
                    swap_class = None
                    
                    for elem in layer:
                        if elem is None:
                            # Identity for 1 path
                            global_mapping.append(current_in_offset)
                            current_in_offset += 1
                        else:
                            # Swap component
                            if swap_class is None:
                                swap_class = elem.__class__
                                
                            # elem.mapping contains local indices
                            for local_in_idx in elem.mapping:
                                global_mapping.append(current_in_offset + local_in_idx)
                            current_in_offset += len(elem.mapping)
                    
                    if swap_class:
                        virtual_swap = swap_class(mapping=global_mapping, name="Combined Swap")
                        tree.append({
                            'layer': virtual_swap,
                            'x': i,
                            'is_parallel': False, # Treat as single layer!
                            'num_branches': 1
                        })
                        continue # Skip the standard parallel handling

                # Parallel layers - create branching
                tree.append({
                    'layer': layer,
                    'x': i,
                    'is_parallel': True,
                    'num_branches': len(layer)
                })
            elif type(layer).__name__ == 'TelescopeArray' and hasattr(layer, 'elements') and len(layer.elements) > 1:
                # Explode TelescopeArray into parallel collectors for visualization
                tree.append({
                    'layer': layer.elements,
                    'x': i,
                    'is_parallel': True,
                    'num_branches': len(layer.elements)
                })
            else:
                tree.append({
                    'layer': layer,
                    'x': i,
                    'is_parallel': False,
                    'num_branches': 1
                })
        return tree
    
    def _count_max_parallel_paths(self, tree: List[dict]) -> int:
        """Count maximum number of parallel paths at any point."""
        max_paths = 1
        current_paths = 1
        
        for node in tree:
            if node['is_parallel']:
                current_paths = max(current_paths, node['num_branches'])
                max_paths = max(max_paths, current_paths)
        
        return max_paths
    
    def _draw_layer_tree(self, ax: plt.Axes, tree: List[dict], 
                        spacing: float, asset_dir: Path):
        """
        Draw the complete layer tree with icons and connections.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to draw on
        tree : list of dict
            Layer tree structure
        spacing : float
            Horizontal spacing between layers
        asset_dir : Path
        """
        # Track active paths (y-positions)
        active_paths = [0.5]  # Start with single path at center
        
        # Pre-calculate x-positions to handle Swap spacing
        # Swap should not consume a full spacing slot
        x_coords = []
        current_x = 0.0
        for node in tree:
            if not node['is_parallel'] and type(node['layer']).__name__ == 'Swap':
                # Permutator sits "between" layers, effectively at the same X as previous?
                # Or we just don't increment for it.
                # Let's assign it the current X, but NOT increment for the next layer.
                # This means Permutator and Next Layer share the same X?
                # No, that would overlap.
                # We want: Prev(X) -> Perm -> Next(X+spacing).
                # So Perm is effectively at X (or X+0.5 spacing).
                # If we assign Perm = X, and Next = X+spacing.
                # Then Perm draws from Prev(X-spacing) to Next(X+spacing).
                # Wait, if Prev is at X-spacing.
                # We want Prev(0) -> Next(2).
                # Perm is at index i. Prev at i-1. Next at i+1.
                # x_coords[i-1] = 0.
                # x_coords[i+1] = 2.
                # x_coords[i] = ? (Doesn't matter for drawing, but matters for loop).
                x_coords.append(current_x)
            else:
                x_coords.append(current_x)
                current_x += spacing

        # Track photonic components for background rectangles
        # Dictionary mapping chip_id (or 'default') to list of coordinates
        photonic_groups = {}
        photonic_types = {'FiberIn', 'FiberOut', 'PhotonicChip', 'YSplitter', 
                         'TOPS', 'ThermoOpticPhaseShifter', 'MMI', 
                         'MultiModeInterferometer', 'Waveguide'}
        
        for i, node in enumerate(tree):
            x_pos = x_coords[i]
            
            if node['is_parallel']:
                # Beam splitter creates multiple paths
                layer_list = node['layer']
                num_branches = len(layer_list)
                
                # Calculate y-positions for branches
                y_positions = self._calculate_branch_positions(num_branches)
                
                # Draw each branch
                for j, (layer, y_pos) in enumerate(zip(layer_list, y_positions)):
                    # Handle different layer types
                    if layer is None:
                        # Draw pass-through line
                        ax.plot([x_pos - 0.4, x_pos + 0.4], [y_pos, y_pos], 
                               color='#E74C3C', linestyle='-', linewidth=2, zorder=1)
                        
                    elif type(layer).__name__ == 'Swap':
                        # Draw as a standard block when in parallel mode
                        self._draw_layer_icon(ax, layer, x_pos, y_pos, asset_dir, 
                                            layer_index=i, element_index=j)

                    else:
                        # Draw standard layer icon
                        self._draw_layer_icon(ax, layer, x_pos, y_pos, asset_dir, 
                                            layer_index=i, element_index=j)
                    
                    # Track photonic components
                    if layer is not None and type(layer).__name__ in photonic_types:
                        # Determine group (chip)
                        chip_id = 'default'
                        if hasattr(layer, 'layer') and layer.layer is not None:
                            # If element belongs to a PhotonicChip layer
                            if type(layer.layer).__name__ == 'PhotonicChip':
                                chip_id = id(layer.layer)
                        
                        if chip_id not in photonic_groups:
                            photonic_groups[chip_id] = []
                        photonic_groups[chip_id].append((x_pos, y_pos))
                    
                    # Draw connection from previous layer(s)
                    if i > 0:
                        # Check if previous layer was Swap
                        prev_node = tree[i-1]
                        is_prev_permutator = False
                        if not prev_node['is_parallel']:
                            if type(prev_node['layer']).__name__ == 'Swap':
                                is_prev_permutator = True

                        # Intelligent connection routing
                        
                        # Determine arrow style based on destination
                        arrow_style = '-' if layer is None else '-|>'

                        # Calculate total inputs expected by current layer
                        expected_inputs = []
                        for elem in layer_list:
                            if elem is not None and hasattr(elem, 'num_inputs'):
                                expected_inputs.append(elem.num_inputs)
                            else:
                                expected_inputs.append(1)
                        
                        total_expected = sum(expected_inputs)
                        
                        if total_expected == len(active_paths):
                            # Perfect match! Route sequentially (Grouped routing)
                            # We need to find which inputs belong to this element (j-th element)
                            
                            # Calculate start index for this element
                            start_idx = sum(expected_inputs[:j])
                            n_in = expected_inputs[j]
                            
                            for k in range(n_in):
                                if start_idx + k < len(active_paths):
                                    prev_y = active_paths[start_idx + k]
                                    if is_prev_permutator:
                                        # Connection already drawn by Swap
                                        pass
                                    else:
                                        self._draw_arrow(ax, x_coords[i-1] + 0.4, prev_y, 
                                                   x_pos - 0.4, y_pos, arrowstyle=arrow_style)
                                               
                        elif len(active_paths) == num_branches:
                            # 1-to-1 connection (Parallel -> Parallel) fallback
                            prev_y = active_paths[j]
                            if is_prev_permutator:
                                # Connection already drawn by Swap
                                pass
                            else:
                                self._draw_arrow(ax, x_coords[i-1] + 0.4, prev_y, 
                                               x_pos - 0.4, y_pos, arrowstyle=arrow_style)
                        else:
                            # All-to-All connection (Split or Combine)
                            for prev_y in active_paths:
                                if is_prev_permutator:
                                    # Connection already drawn by Swap
                                    pass
                                else:
                                    self._draw_arrow(ax, x_coords[i-1] + 0.4, prev_y, 
                                               x_pos - 0.4, y_pos, arrowstyle=arrow_style)
                
                # Update active paths
                # We need to calculate output paths based on num_outputs of each element
                new_active_paths = []
                for j, (layer, y_pos) in enumerate(zip(layer_list, y_positions)):
                    n_out = 1
                    if layer is not None and hasattr(layer, 'num_outputs'):
                        n_out = layer.num_outputs
                    
                    # If n_out > 1, we should probably spread them around y_pos?
                    # For now, let's just keep y_pos if n_out=1, or duplicate if n_out > 1
                    # But visually, the box is at y_pos.
                    # If we have multiple outputs, they should emerge from y_pos.
                    # But for the NEXT layer, we need distinct y positions if they are to be routed separately.
                    # This is getting complex for visualization.
                    # Simplified: All outputs from this element start at y_pos.
                    for _ in range(n_out):
                        new_active_paths.append(y_pos)
                
                active_paths = new_active_paths
                
            else:
                # Single layer
                layer = node['layer']
                
                if type(layer).__name__ == 'Swap':
                    # Special handling for Swap (CrossSection)
                    # No icon, just crossed connections
                    
                    # Calculate new spread-out positions for the outputs of the permutator
                    # This ensures connections fan out to match the next layer's spacing
                    num_paths = len(active_paths)
                    new_y_positions = self._calculate_branch_positions(num_paths)
                    
                    if i > 0:
                        mapping = layer.mapping
                        # Draw crossed arrows from prev layer to this layer
                        for dest_idx, src_idx in enumerate(mapping):
                            if src_idx < len(active_paths) and dest_idx < len(new_y_positions):
                                y_src = active_paths[src_idx]
                                y_dest = new_y_positions[dest_idx]
                                
                                # Check if next layer exists and if the destination is None
                                arrow_style = '-|>'
                                if i + 1 < len(tree):
                                    next_node = tree[i+1]
                                    if next_node['is_parallel']:
                                        # If next layer is parallel, check if the corresponding element is None
                                        # We need to map dest_idx to the element in next layer
                                        # Assuming 1-to-1 mapping for now or simple sequential
                                        # This is tricky because we don't know exactly which element consumes which output
                                        # But if we assume 1 output per path...
                                        
                                        # Let's try to find the element at dest_idx
                                        # We need to count inputs of elements in next layer
                                        next_layer_list = next_node['layer']
                                        current_input_idx = 0
                                        target_element = None
                                        
                                        for elem in next_layer_list:
                                            n_in = 1
                                            if elem is not None and hasattr(elem, 'num_inputs'):
                                                n_in = elem.num_inputs
                                            
                                            if current_input_idx <= dest_idx < current_input_idx + n_in:
                                                target_element = elem
                                                break
                                            current_input_idx += n_in
                                            
                                        if target_element is None:
                                            arrow_style = '-'
                                    else:
                                        # Next layer is single
                                        if next_node['layer'] is None:
                                            arrow_style = '-'

                                # Arrow from prev layer output directly to next layer input
                                # Spanning across the permutator layer
                                # Use x_coords to get correct positions
                                self._draw_arrow(ax, x_coords[i-1] + 0.4, y_src, 
                                               x_coords[i+1] - 0.4, y_dest, arrowstyle=arrow_style)
                    
                    # Update active paths to the new spread-out positions
                    active_paths = new_y_positions
                    
                else:
                    # Standard Single Layer Logic
                    
                    # Draw at center of active paths
                    y_pos = sum(active_paths) / len(active_paths)
                    
                    # Draw layer icon
                    self._draw_layer_icon(ax, layer, x_pos, y_pos, asset_dir, 
                                        layer_index=i)
                
                    # Track photonic components
                    if type(layer).__name__ in photonic_types:
                        # Determine group (chip)
                        chip_id = 'default'
                        if hasattr(layer, 'layer') and layer.layer is not None:
                            if type(layer.layer).__name__ == 'PhotonicChip':
                                chip_id = id(layer.layer)
                        
                        if chip_id not in photonic_groups:
                            photonic_groups[chip_id] = []
                        photonic_groups[chip_id].append((x_pos, y_pos))
                
                    # Draw connections from all active paths
                    if i > 0:
                        # Check if previous layer was Swap
                        prev_node = tree[i-1]
                        is_prev_permutator = False
                        if not prev_node['is_parallel']:
                            if type(prev_node['layer']).__name__ == 'Swap':
                                is_prev_permutator = True
                        
                        for prev_y in active_paths:
                            if is_prev_permutator:
                                # Connection already drawn by Swap
                                pass
                            else:
                                self._draw_arrow(ax, x_coords[i-1] + 0.4, prev_y,
                                               x_pos - 0.4, y_pos)
                    
                    # Update active paths
                    # Single output path (or multiple if single layer produces multiple)
                    # If TelescopeArray, it produces N outputs (visually)
                    if type(layer).__name__ == 'TelescopeArray':
                         # This case is actually handled by the "explode" logic in _build_layer_tree
                         # So we shouldn't reach here for TelescopeArray unless it has 1 element
                         pass
                
                    n_out = 1
                    if hasattr(layer, 'num_outputs'):
                        n_out = layer.num_outputs
                
                    # For single layer, we usually collapse to 1 path unless it's a splitter
                    # But if it's a splitter (YSplitter), it should probably be in a parallel list?
                    # Or if it's a single YSplitter layer, it produces 2 outputs.
                    # If it produces 2 outputs, we should probably split the active path?
                
                    if n_out > 1:
                        # Split active paths
                        # We need to generate n_out new y positions centered around y_pos
                        # But we don't have a good way to space them without knowing global context
                        # For now, just replicate y_pos
                        active_paths = [y_pos] * n_out
                    else:
                        active_paths = [y_pos]
        
        # Draw background rectangles for photonic circuits
        for chip_id, coords in photonic_groups.items():
            if not coords:
                continue
                
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # Add padding
            pad_x = 0.8
            pad_y = 0.8
            
            rect = patches.Rectangle(
                (min_x - pad_x, min_y - pad_y),
                (max_x - min_x) + 2*pad_x,
                (max_y - min_y) + 2*pad_y,
                linewidth=1,
                edgecolor='#BDC3C7',
                facecolor='#ECF0F1',
                alpha=0.5,
                zorder=0,
                linestyle='--'
            )
            ax.add_patch(rect)
            
            # Add label "Photonic Circuit"
            label = "Photonic Circuit"
            if chip_id != 'default':
                # Try to get chip name if possible, but we only have ID here
                # We could store the chip object instead of ID
                pass
                
            ax.text((min_x + max_x)/2, max_y + pad_y, label,
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   color='#7F8C8D')
    
    def _calculate_branch_positions(self, num_branches: int) -> List[float]:
        """Calculate y-positions for parallel branches."""
        if num_branches == 1:
            return [0.5]
        
        # Spread branches vertically
        spacing = 1.5
        total_height = (num_branches - 1) * spacing
        start_y = 0.5 - total_height / 2
        
        # Return positions in reverse order (Top to Bottom)
        # Index 0 (low index) -> Highest Y
        # Index N (high index) -> Lowest Y
        return [start_y + (num_branches - 1 - i) * spacing for i in range(num_branches)]
    
    def _draw_layer_icon(self, ax: plt.Axes, layer: Layer, 
                        x: float, y: float, asset_dir: Path, 
                        layer_index: Optional[int] = None, 
                        element_index: Optional[int] = None):
        """
        Draw a layer icon with label.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to draw on
        layer : Layer
            The layer to represent
        x : float
            X-position
        y : float
            Y-position
        asset_dir : Path
            Path to assets directory
        layer_index : int, optional
            Index of the layer (1-based)
        element_index : int, optional
            Index of the element in a parallel layer (1-based)
        """
        # Get layer type name
        layer_name = type(layer).__name__
        
        # Map layer types to icon files and markers
        # (icon_file, marker_style, color)
        icon_map = {
            'Scene': ('scene.svg', '*', '#F1C40F'),
            'Star': ('scene.svg', '*', '#F1C40F'),
            'Planet': ('scene.svg', 'o', '#E67E22'),
            'Telescope': ('telescope.svg', 'o', '#3498DB'),
            'TelescopeArray': ('telescope.svg', 'o', '#3498DB'),
            'Collector': ('telescope.svg', 'o', '#3498DB'),
            'Interferometer': ('interferometer.svg', 'D', '#9B59B6'),
            'Atmosphere': ('atmosphere.svg', 'H', '#95A5A6'),
            'AdaptiveOptics': ('adaptive_optics.svg', 's', '#2ECC71'),
            'Coronagraph': ('coronagraph.svg', '8', '#34495E'),
            'BeamSplitter': ('beam_splitter.svg', 'D', '#E74C3C'),
            'FiberIn': ('fiber_in.svg', 'h', '#1ABC9C'),
            'FiberOut': ('fiber_out.svg', 'h', '#1ABC9C'),
            'PhotonicChip': ('photonic_chip.svg', 's', '#34495E'),
            'YSplitter': ('splitter.svg', 'v', '#E74C3C'),
            'TOPS': ('phase_shifter.svg', 's', '#E67E22'),
            'ThermoOpticPhaseShifter': ('phase_shifter.svg', 's', '#E67E22'),
            'MMI': ('mmi.svg', 's', '#8E44AD'),
            'MultiModeInterferometer': ('mmi.svg', 's', '#8E44AD'),
            'Swap': ('swap.svg', 's', '#7F8C8D'),
            'Camera': ('camera.svg', 's', '#2C3E50')
        }
        
        icon_info = icon_map.get(layer_name, ('telescope.svg', 'o', '#95A5A6'))
        icon_file, marker, color = icon_info
        
        # Draw box for component
        box_width = 0.6
        box_height = 0.6
        
        # Use fancy box with rounded corners
        box = FancyBboxPatch(
            (x - box_width/2, y - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.05",
            edgecolor='#2C3E50',
            facecolor='#ECF0F1',
            linewidth=2,
            zorder=2
        )
        ax.add_patch(box)
        
        # Try to render SVG icon
        icon_path = asset_dir / icon_file
        if icon_path.exists():
            try:
                self._render_svg_icon(ax, icon_path, x, y, box_width*0.8)
            except Exception as e:
                # Fallback to marker if SVG rendering fails
                # print(f"SVG render failed for {icon_file}: {e}")
                ax.plot(x, y, marker, markersize=15, color=color, zorder=3, alpha=0.8)
        else:
            # Fallback to marker
            ax.plot(x, y, marker, markersize=15, color=color, zorder=3, alpha=0.8)
        
        # Construct label
        display_name = self._get_display_name(layer)
            
        # Add label below box
        ax.text(x, y - box_height/2 - 0.15, display_name,
               ha='center', va='top', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='none', alpha=0.8))
        
        # Add type in parentheses (gray, smaller)
        type_text = f"({layer_name})"
        ax.text(x, y - box_height/2 - 0.4, type_text,
               ha='center', va='top', fontsize=7, color='#7F8C8D')
               
        # Add indices in code format [i] or [i,j]
        if layer_index is not None:
            if element_index is not None:
                idx_text = f"[{layer_index},{element_index}]"
            else:
                idx_text = f"[{layer_index}]"
            
            ax.text(x, y - box_height/2 - 0.55, idx_text,
                   ha='center', va='top', fontsize=7, family='monospace', color='#2C3E50')
    
    def _render_svg_icon(self, ax: plt.Axes, svg_path: Path, center_x: float, center_y: float, size: float):
        """
        Render a simple SVG icon onto the axes.
        
        Supports basic SVG elements: path (M, L, C, Z), rect, circle.
        Assumes SVG viewBox is 0 0 100 100.
        """
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Namespace handling
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        
        # Scale factor (SVG 100x100 -> size x size)
        scale = size / 100.0
        offset_x = center_x - size/2
        offset_y = center_y - size/2
        
        # Helper to transform coordinates
        def trans_x(val): return offset_x + float(val) * scale
        def trans_y(val): return offset_y + (100 - float(val)) * scale # Flip Y for MPL
        def trans_len(val): return float(val) * scale
        
        # Helper to parse style attributes
        def get_style(elem):
            color = elem.get('stroke', 'none')
            if color == 'none': color = None
            
            fill = elem.get('fill', 'none')
            if fill == 'none': fill = None
            
            lw = float(elem.get('stroke-width', 1)) * 0.5
            
            alpha = float(elem.get('opacity', 1.0))
            
            ls = '-'
            if elem.get('stroke-dasharray'):
                ls = '--'
                
            return color, fill, lw, alpha, ls
        
        # Iterate elements (ignoring namespace for simplicity in tag check)
        for elem in root.iter():
            tag = elem.tag.split('}')[-1]
            
            if tag == 'path':
                d = elem.get('d')
                if d:
                    color, fill, lw, alpha, ls = get_style(elem)
                    self._draw_svg_path(ax, d, color, fill, lw, alpha, ls, trans_x, trans_y)
            
            elif tag == 'rect':
                x = float(elem.get('x', 0))
                y = float(elem.get('y', 0))
                w = float(elem.get('width', 0))
                h = float(elem.get('height', 0))
                
                mpl_x = trans_x(x)
                mpl_y = trans_y(y + h)
                mpl_w = trans_len(w)
                mpl_h = trans_len(h)
                
                color, fill, lw, alpha, ls = get_style(elem)
                
                rect = patches.Rectangle((mpl_x, mpl_y), mpl_w, mpl_h, 
                                       linewidth=lw, edgecolor=color, facecolor=fill, 
                                       alpha=alpha, linestyle=ls, zorder=3)
                ax.add_patch(rect)
                
            elif tag == 'circle':
                cx = float(elem.get('cx', 0))
                cy = float(elem.get('cy', 0))
                r = float(elem.get('r', 0))
                
                mpl_cx = trans_x(cx)
                mpl_cy = trans_y(cy)
                mpl_r = trans_len(r)
                
                color, fill, lw, alpha, ls = get_style(elem)
                
                circ = patches.Circle((mpl_cx, mpl_cy), mpl_r, 
                                    linewidth=lw, edgecolor=color, facecolor=fill, 
                                    alpha=alpha, linestyle=ls, zorder=3)
                ax.add_patch(circ)
                
            elif tag == 'text':
                # Skip text as requested
                pass

    def _draw_svg_path(self, ax, d, color, fill, lw, alpha, ls, tx, ty):
        """Parse simple SVG path d string and draw PathPatch."""
        # Regex to tokenize path data: commands (letters) and numbers
        tokens = re.findall(r'([a-zA-Z])|([-+]?\d*\.?\d+)', d)
        tokens = [t[0] or t[1] for t in tokens]
        
        verts = []
        codes = []
        
        i = 0
        current_pos = (0, 0)
        
        while i < len(tokens):
            cmd = tokens[i]
            i += 1
            
            if cmd == 'M': # Move to x,y
                x = float(tokens[i]); y = float(tokens[i+1])
                verts.append((tx(x), ty(y)))
                codes.append(MPath.MOVETO)
                current_pos = (x, y)
                i += 2
            elif cmd == 'L': # Line to x,y
                x = float(tokens[i]); y = float(tokens[i+1])
                verts.append((tx(x), ty(y)))
                codes.append(MPath.LINETO)
                current_pos = (x, y)
                i += 2
            elif cmd == 'C': # Cubic Bezier (x1 y1 x2 y2 x y)
                x1 = float(tokens[i]); y1 = float(tokens[i+1])
                x2 = float(tokens[i+2]); y2 = float(tokens[i+3])
                x = float(tokens[i+4]); y = float(tokens[i+5])
                
                verts.append((tx(x1), ty(y1)))
                verts.append((tx(x2), ty(y2)))
                verts.append((tx(x), ty(y)))
                
                codes.append(MPath.CURVE4)
                codes.append(MPath.CURVE4)
                codes.append(MPath.CURVE4)
                
                current_pos = (x, y)
                i += 6
            elif cmd == 'Q': # Quadratic Bezier (x1 y1 x y)
                x1 = float(tokens[i]); y1 = float(tokens[i+1])
                x = float(tokens[i+2]); y = float(tokens[i+3])
                
                verts.append((tx(x1), ty(y1)))
                verts.append((tx(x), ty(y)))
                
                codes.append(MPath.CURVE3)
                codes.append(MPath.CURVE3)
                
                current_pos = (x, y)
                i += 4
            elif cmd == 'Z': # Close path
                verts.append((0,0)) # Ignored
                codes.append(MPath.CLOSEPOLY)
            # Add more commands (T, etc.) if needed
            
        if verts:
            path = MPath(verts, codes)
            
            patch = PathPatch(path, facecolor=fill, edgecolor=color, linewidth=lw, 
                            alpha=alpha, linestyle=ls, zorder=3)
            ax.add_patch(patch)

    def _get_display_name(self, layer: Layer) -> str:
        """Get display name for a layer."""
        layer_name = type(layer).__name__
        
        if layer_name == 'Swap':
            return f"Swap: {layer.mapping}"
        
        # Check for name attribute (TelescopeArray, Scene, etc.)
        if hasattr(layer, 'name') and layer.name:
            return layer.name
        
        # Use class name
        return layer_name
    
    def _draw_arrow(self, ax: plt.Axes, x1: float, y1: float, 
                   x2: float, y2: float, arrowstyle: str = '-|>'):
        """
        Draw an arrow representing signal flow.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to draw on
        x1, y1 : float
            Start position
        x2, y2 : float
            End position
        arrowstyle : str, optional
            Style of the arrow (default: '-|>')
        """
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=arrowstyle,
            color='#E74C3C',
            linewidth=2,
            mutation_scale=20,
            zorder=1
        )
        ax.add_patch(arrow)

def test_pipeline_initialization():
    pipe = Pipeline(date="2025-01-01", declination=10)
    assert pipe.date == "2025-01-01"
    assert pipe.declination == 10
    assert len(pipe.layers) == 0

def test_pipeline_add_layer():
    pipe = Pipeline()
    class MockLayer(Layer):
        def process(self, wf, pipe): return "processed"
    
    l1 = MockLayer()
    pipe.add_layer(l1)
    assert len(pipe.layers) == 1
    assert pipe.layers[0] == l1

if __name__ == "__main__":
    import pytest
    # Run internal tests
    # pytest.main([__file__])
    test_pipeline_initialization()
    test_pipeline_add_layer()
    print("Pipeline tests passed.")
