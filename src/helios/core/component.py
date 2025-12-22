from typing import Optional, TYPE_CHECKING, List, Any
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
        self.metadata: dict = {}  # Store for UI/Application specific data

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
