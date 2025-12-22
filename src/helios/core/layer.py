from typing import Optional, List, Any, TYPE_CHECKING
import copy
from .component import Component, GenerationComponent, SamplingComponent, OpticalComponent, DetectionComponent, DataComponent

if TYPE_CHECKING:
    from helios.core.pipeline import Pipeline

class Layer:
    """
    Base class for all simulation layers (logical grouping of components).
    
    A Layer represents a logical stage in the simulation pipeline and contains
    one or more Components that process wavefronts in parallel.
    """
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.elements: List[Component] = []  # List of components
        self.pipeline: Optional['Pipeline'] = None
        self.metadata: dict = {}
        self.num_inputs: int = 1
        
        # Caching
        self._cached_input: Any = None
        self._cached_output: Any = None
    
    num_outputs: int = 1

    def invalidate_cache(self):
        """Invalidate the cache of this layer and trigger propagation."""
        self._cached_input = None
        self._cached_output = None
        if self.pipeline:
            self.pipeline.invalidate_downstream_cache(self)

    def get_input_wavefront(self) -> Any:
        """Retrieve the input wavefront for this layer."""
        if self._cached_input is not None:
            return self._cached_input
            
        if self.pipeline is None:
            return None

        prev_output = self.pipeline.get_previous_layer_output(self)
        self._cached_input = prev_output
        return prev_output

    def get_output_wavefront(self) -> Any:
        """Retrieve the output wavefront of this layer."""
        if self._cached_output is not None:
            return self._cached_output
        
        input_wf = self.get_input_wavefront()
        result = self.process(input_wf)
        self._cached_output = result
        return result
    
    def twin(self) -> 'Layer':
        """Create a twin copy of this layer."""
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

    def add_element(self, component: Component):
        """Add a component to this layer."""
        self.elements.append(component)
        component.layer = self
        if self.pipeline is not None:
            component.pipeline = self.pipeline
    
    def add_component(self, component: Component):
        """Add a component to this layer (alias)."""
        self.add_element(component)
    
    @property
    def components(self) -> List[Component]:
        """Alias for elements attribute."""
        return self.elements

    def description(self, indent: int = 0, full: bool = False) -> str:
        """Generate a text description of this layer and all its components."""
        prefix = " " * indent
        class_name = self.__class__.__name__
        name_str = f" '{self.name}'" if self.name else ""
        
        lines = [f"{prefix}{class_name}{name_str}"]
        
        if full:
            details = self._get_detailed_attributes()
            if details:
                for key, value in details.items():
                    lines.append(f"{prefix}  • {key}: {value}")
        
        if self.elements:
            for i, element in enumerate(self.elements):
                is_last = (i == len(self.elements) - 1)
                connector = "└─" if is_last else "├─"
                elem_desc = element.description(0, full=full)
                elem_lines = elem_desc.split('\n')
                lines.append(f"{prefix}  {connector} {elem_lines[0]}")
                if len(elem_lines) > 1:
                    continuation = "  " if is_last else "│ "
                    for line in elem_lines[1:]:
                        lines.append(f"{prefix}  {continuation} {line}")
        
        return "\n".join(lines)
    
    def _get_detailed_attributes(self) -> dict:
        return {}

    def process(self, wavefront: Any) -> Any:
        raise NotImplementedError("Subclasses must implement process()")

    def to_dict(self) -> dict:
        """Serialize layer configuration to dictionary."""
        return {
            "type": self.__class__.__name__,
            "module": self.__class__.__module__,
            "name": self.name,
            "metadata": self.metadata,
            "elements": [e.to_dict() for e in self.elements]
        }
    
    @classmethod
    def from_dict(cls, data: dict, context: Optional['Pipeline'] = None) -> 'Layer':
        """Create layer instance from dictionary."""
        name = data.get("name")
        layer = cls(name=name)
        layer.metadata = data.get("metadata", {})
        return layer

# =============================================================================
# LAYER TYPES
# =============================================================================

class GenerationLayer(Layer):
    """Layer generating the continuous electromagnetic field (Scene, Atmosphere)."""
    pass

class SamplingLayer(Layer):
    """Layer sampling the continuous field into discrete optical paths (TelescopeArray)."""
    pass

class OpticalLayer(Layer):
    """Layer propagating/modifying optical beams."""
    pass

class DetectionLayer(Layer):
    """Layer converting photons to digital data."""
    pass

class DataLayer(Layer):
    """Layer processing digital data."""
    pass
