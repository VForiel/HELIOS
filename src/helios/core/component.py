from typing import Optional, TYPE_CHECKING, List, Any, Tuple
import copy
from ..utils.serialization import serialize_value, deserialize_value

if TYPE_CHECKING:
    from helios.core.pipeline import Pipeline
    from helios.core.layer import Layer

class Component:
    """
    Base class for all simulation components (physical elements).
    
    A Component represents a physical element in the optical system that can
    process wavefronts independently. Components are grouped within Layers for
    parallel processing.
    
    Parameters
    ----------
    name : str, optional
        Descriptive name for this component (used in diagrams and logging)
    
    Attributes
    ----------
    name : str
        Descriptive name for this component
    layer : Layer
        Reference to the parent layer containing this component
    pipeline : Pipeline
        Shortcut to access the parent pipeline (equivalent to self.layer.pipeline)
    """
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.layer: Optional['Layer'] = None
        self.pipeline: Optional['Pipeline'] = None
        self.num_inputs: int = 1  # Number of inputs this component consumes
        self.num_outputs: int = 1 # Number of outputs this component produces
        
    def get_index(self) -> Tuple[int, int]:
        """
        Get the (layer_index, component_index) of this component.
        
        Returns
        -------
        tuple (int, int)
        
        Raises
        ------
        RuntimeError
            If not attached to a layer or pipeline.
        """
        if self.layer is None:
            raise RuntimeError("Component is not attached to a layer.")
            
        try:
            l_idx = self.layer.get_index()
        except RuntimeError:
            raise RuntimeError("Parent layer is not attached to a pipeline.")
            
        try:
            c_idx = self.layer.elements.index(self)
        except ValueError:
            raise RuntimeError("Component not found in its parent layer elements.")
            
        return (l_idx, c_idx)

    def next(self) -> 'Component':
        """
        Get the next component in the pipeline execution order.
        
        If this is the last component in a layer, returns the first component
        of the next layer.
        
        Returns
        -------
        Component
        
        Raises
        ------
        IndexError
            If this is the last component in the pipeline.
        """
        if self.layer is None:
            raise RuntimeError("Not attached to layer.")
            
        # Check if there is a next component in the same layer
        my_idx = self.layer.elements.index(self)
        if my_idx < len(self.layer.elements) - 1:
            return self.layer.elements[my_idx + 1]
            
        # Otherwise, go to next layer
        next_layer = self.layer.next()
        
        # Handle parallel layers (List[Layer])
        if isinstance(next_layer, list):
            # Ambiguity: which branch? Default to first valid branch's first component
            for sub in next_layer:
                if sub and sub.elements:
                    return sub.elements[0]
            raise IndexError("Next layer group has no components.")
        else:
            if not next_layer.elements:
                raise IndexError("Next layer is empty.")
            return next_layer.elements[0]

    def previous(self) -> 'Component':
        """
        Get the previous component in the pipeline execution order.
        
        Returns
        -------
        Component
        
        Raises
        ------
        IndexError
            If this is the first component in the pipeline.
        """
        if self.layer is None:
            raise RuntimeError("Not attached to layer.")
            
        my_idx = self.layer.elements.index(self)
        if my_idx > 0:
            return self.layer.elements[my_idx - 1]
            
        prev_layer = self.layer.previous()
        
        if isinstance(prev_layer, list):
            # Default to first valid branch's last component
            for sub in prev_layer:
                if sub and sub.elements:
                    return sub.elements[-1]
            raise IndexError("Previous layer group has no components.")
        else:
             if not prev_layer.elements:
                 raise IndexError("Previous layer is empty.")
             return prev_layer.elements[-1]

    def twin(self) -> 'Component':
        """
        Create a twin copy of this component.
        """
        parent_layer = self.layer
        self.layer = None
        
        try:
            new_component = copy.deepcopy(self)
        finally:
            self.layer = parent_layer
            
        new_component.layer = parent_layer
        return new_component

    def description(self, indent: int = 0, full: bool = False) -> str:
        """Generate a text description of this component."""
        prefix = " " * indent
        class_name = self.__class__.__name__
        name_str = f" '{self.name}'" if self.name else ""
        
        result = f"{prefix}{class_name}{name_str}"
        
        if full:
            details = self._get_detailed_attributes()
            if details:
                for key, value in details.items():
                    result += f"\n{prefix}  • {key}: {value}"
        
        return result
    
    def _get_detailed_attributes(self) -> dict:
        """Return a dictionary of detailed attributes for full description."""
        return {}

    def process(self, wavefront: Any) -> Any:
        """Process the incoming wavefront/signal and return the result."""
        raise NotImplementedError("Subclasses must implement process()")

    def to_dict(self) -> dict:
        """Serialize component configuration to dictionary."""
        return {
            "type": self.__class__.__name__,
            "module": self.__class__.__module__,
            "name": self.name,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Component':
        """Create component instance from dictionary."""
        name = data.get("name")
        comp = cls(name=name)
        comp.metadata = data.get("metadata", {})
        return comp

# =============================================================================
# COMPONENT TYPES
# =============================================================================

class GenerationComponent(Component):
    """Component that generates electromagnetic fields (Scene elements)."""
    pass

class SamplingComponent(Component):
    """Component that samples continuous fields into discrete optical paths."""
    pass

class OpticalComponent(Component):
    """Component that propagates or modifies optical beams."""
    pass

class DetectionComponent(Component):
    """Component that converts photons to digital data."""
    pass

class DataComponent(Component):
    """Component that processes digital data."""
    pass
