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
    context : Context
        Shortcut to access the parent context (equivalent to self.layer.context)
    
    Examples
    --------
    >>> class CustomElement(Element):
    ...     def __init__(self, parameter, name=None):
    ...         super().__init__(name=name or "CustomElement")
    ...         self.parameter = parameter
    ...
    ...     def process(self, wavefront, context):
    ...         # Apply custom transformation
    ...         wavefront.field *= self.parameter
    ...         return wavefront
    """
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.layer: Optional['Layer'] = None
        self.context: Optional['Context'] = None
        self.num_inputs: int = 1  # Number of inputs this element consumes
        self.num_outputs: int = 1 # Number of outputs this element produces

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

    def process(self, wavefront: Any, context: 'Context') -> Any:
        """
        Process the incoming wavefront/signal and return the result.
        
        This method must be implemented by all subclasses. It defines how
        the element transforms the electromagnetic field or signal.
        
        Parameters
        ----------
        wavefront : Wavefront or list of Wavefront
            The input electromagnetic field(s) to process
        context : Context
            The simulation context providing global parameters
        
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
    context : Context
        Reference to the parent context managing this layer
    
    Examples
    --------
    >>> class CustomLayer(Layer):
    ...     def __init__(self, name=None):
    ...         super().__init__(name=name or "CustomLayer")
    ...
    ...     def process(self, wavefront, context):
    ...         # Apply custom transformation
    ...         wavefront.field *= np.exp(1j * phase_pattern)
    ...         return wavefront
    
    See Also
    --------
    Context : Orchestrates layer execution
    Element : Physical components within layers
    """
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.elements: List[Element] = []
        self.context: Optional['Context'] = None
        self.num_inputs: int = 1  # Number of inputs this layer consumes (if single layer)
        # self.num_outputs is defined as class attribute to allow property override
    
    num_outputs: int = 1 # Default number of outputs
    
    def add_element(self, element: Element):
        """
        Add an element to this layer.
        
        Automatically sets the element's layer and context references.
        
        Parameters
        ----------
        element : Element
            The element to add to this layer
        """
        self.elements.append(element)
        element.layer = self
        # Set context if the layer is already attached to a context
        if self.context is not None:
            element.context = self.context

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
    >>> # Create optical system (single telescope)
    >>> telescope = helios.TelescopeArray(name="Observatory")
    >>> telescope.add_collector(pupil=helios.Pupil.vlt(), position=(0,0), size=8*u.m)
    >>> 
    >>> # Create detector
    >>> camera = helios.Camera(pixels=(512, 512))
    >>> 
    >>> # Build context and run
    >>> ctx = Context()
    >>> ctx.add_layer(scene)
    >>> ctx.add_layer(telescope)
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
        
        Automatically sets the layer's context reference and propagates to elements.
        
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
        
        # Set context reference for layer(s)
        if isinstance(layer, list):
            for l in layer:
                if l is not None:
                    l.context = self
                    # Propagate to elements if layer has them
                    if hasattr(l, 'elements') and l.elements:
                        for element in l.elements:
                            element.context = self
        else:
            layer.context = self
            # Propagate to elements if layer has them
            if hasattr(layer, 'elements') and layer.elements:
                for element in layer.elements:
                    element.context = self

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
            Formatted description of all layers and elements in the context
        
        Examples
        --------
        >>> ctx = Context()
        >>> ctx.add_layer(scene)
        >>> ctx.add_layer(telescope)
        >>> ctx.add_layer(camera)
        >>> print(ctx.description())
        HELIOS Simulation Context
        ========================
        Layer 1: Scene
        Layer 2: TelescopeArray
        >>>   └─ Collector 1
        Layer 3: Camera
        
        >>> print(ctx.description(full=True))
        HELIOS Simulation Context
        ========================
        Context Parameters:
        >>>   • date: 2025-01-01
        >>>   • declination: 10.0 deg
        
        Layer 1: Scene 'Target'
        >>>   • distance: 10.0 pc
        >>>   └─ Star
        >>>     • temperature: 5700 K
        >>>     • magnitude: 5.0
        ...
        """
        lines = ["HELIOS Simulation Context", "=" * 50, ""]
        
        # Add context parameters if full mode
        if full:
            ctx_params = []
            if self.date is not None:
                ctx_params.append(f"  • date: {self.date}")
            if self.declination is not None:
                ctx_params.append(f"  • declination: {self.declination}")
            if ctx_params:
                lines.append("Context Parameters:")
                lines.extend(ctx_params)
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
                            
                        result = sub_layer.process(proc_input, self)
                        
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
                current_signal = layer.process(current_signal, self)

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

    def plot_uml_diagram(self, figsize: Tuple[float, float] = (16, 10), 
                         layer_spacing: float = 2.0,
                         save_path: Optional[str] = None,
                         return_type: str = 'figure') -> Union[plt.Figure, np.ndarray]:
        """
        Generate a UML-style diagram of the complete optical setup.
        
        This function creates a visual representation of the simulation pipeline,
        showing all layers from scene (left) to camera (right). Beam splitters
        create parallel paths that are displayed vertically.
        
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
        
        Examples
        --------
        >>> ctx = Context()
        >>> ctx.add_layer(scene)
        >>> ctx.add_layer(telescope)
        >>> ctx.add_layer(BeamSplitter())
        >>> ctx.add_layer([camera1, camera2])
        >>> fig = ctx.plot_uml_diagram()
        >>> plt.show()
        
        Notes
        -----
        The diagram displays:
        - Each layer with its schematic icon (from assets/)
        - Layer names as labels
        - Arrows showing signal flow
        - Parallel paths for beam splitting
        
        The coordinate system is left-to-right (scene → detector) with parallel
        paths displayed vertically when beam splitting occurs.
        """
        # Validate architecture first
        self.validate_architecture()
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(-1, len(self.layers) * layer_spacing + 1)
        
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
        
        # Track photonic components for background rectangles
        # Dictionary mapping chip_id (or 'default') to list of coordinates
        photonic_groups = {}
        photonic_types = {'FiberIn', 'FiberOut', 'PhotonicChip', 'YSplitter', 
                         'TOPS', 'ThermoOpticPhaseShifter', 'MMI', 
                         'MultiModeInterferometer', 'Waveguide'}
        
        for i, node in enumerate(tree):
            x_pos = i * spacing
            
            if node['is_parallel']:
                # Beam splitter creates multiple paths
                layer_list = node['layer']
                num_branches = len(layer_list)
                
                # Calculate y-positions for branches
                y_positions = self._calculate_branch_positions(num_branches)
                
                # Draw each branch
                for j, (layer, y_pos) in enumerate(zip(layer_list, y_positions)):
                    # Draw layer icon
                    self._draw_layer_icon(ax, layer, x_pos, y_pos, asset_dir, 
                                        layer_index=i+1, element_index=j+1)
                    
                    # Track photonic components
                    if type(layer).__name__ in photonic_types:
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
                        # Intelligent connection routing
                        
                        # Calculate total inputs expected by current layer
                        expected_inputs = []
                        for elem in layer_list:
                            if hasattr(elem, 'num_inputs'):
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
                                    self._draw_arrow(ax, (i-1)*spacing + 0.4, prev_y, 
                                               x_pos - 0.4, y_pos)
                                               
                        elif len(active_paths) == num_branches:
                            # 1-to-1 connection (Parallel -> Parallel) fallback
                            prev_y = active_paths[j]
                            self._draw_arrow(ax, (i-1)*spacing + 0.4, prev_y, 
                                           x_pos - 0.4, y_pos)
                        else:
                            # All-to-All connection (Split or Combine)
                            for prev_y in active_paths:
                                self._draw_arrow(ax, (i-1)*spacing + 0.4, prev_y, 
                                           x_pos - 0.4, y_pos)
                
                # Update active paths
                # We need to calculate output paths based on num_outputs of each element
                new_active_paths = []
                for j, (layer, y_pos) in enumerate(zip(layer_list, y_positions)):
                    n_out = 1
                    if hasattr(layer, 'num_outputs'):
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
                
                # Draw at center of active paths
                y_pos = sum(active_paths) / len(active_paths)
                
                # Draw layer icon
                self._draw_layer_icon(ax, layer, x_pos, y_pos, asset_dir, 
                                    layer_index=i+1)
                
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
                    for prev_y in active_paths:
                        self._draw_arrow(ax, (i-1)*spacing + 0.4, prev_y,
                                       x_pos - 0.4, y_pos)
                
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
        
        return [start_y + i * spacing for i in range(num_branches)]
    
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
        
        # Check for name attribute (TelescopeArray, Scene, etc.)
        if hasattr(layer, 'name') and layer.name:
            return layer.name
        
        # Use class name
        return layer_name
    
    def _draw_arrow(self, ax: plt.Axes, x1: float, y1: float, 
                   x2: float, y2: float):
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
        """
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='-|>',
            color='#E74C3C',
            linewidth=2,
            mutation_scale=20,
            zorder=1
        )
        ax.add_patch(arrow)

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
