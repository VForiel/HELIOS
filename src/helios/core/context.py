import numpy as np
from astropy import units as u
from typing import List, Union, Optional, Any, Tuple
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
import os

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
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return image
        elif return_type == 'both':
            # Return both figure and image
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
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
            Path to assets directory
        """
        # Track active paths (y-positions)
        active_paths = [0.5]  # Start with single path at center
        
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
                    self._draw_layer_icon(ax, layer, x_pos, y_pos, asset_dir)
                    
                    # Draw connection from previous layer(s)
                    if i > 0:
                        for prev_y in active_paths:
                            self._draw_arrow(ax, (i-1)*spacing + 0.4, prev_y, 
                                           x_pos - 0.4, y_pos)
                
                # Update active paths
                active_paths = y_positions
                
            else:
                # Single layer
                layer = node['layer']
                
                # Draw at center of active paths
                y_pos = sum(active_paths) / len(active_paths)
                
                # Draw layer icon
                self._draw_layer_icon(ax, layer, x_pos, y_pos, asset_dir)
                
                # Draw connections from all active paths
                if i > 0:
                    for prev_y in active_paths:
                        self._draw_arrow(ax, (i-1)*spacing + 0.4, prev_y,
                                       x_pos - 0.4, y_pos)
                
                # Single output path
                active_paths = [y_pos]
    
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
                        x: float, y: float, asset_dir: Path):
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
        """
        # Get layer type name
        layer_name = type(layer).__name__
        
        # Map layer types to icon files
        icon_map = {
            'Scene': 'scene.svg',
            'Star': 'scene.svg',
            'Planet': 'scene.svg',
            'Telescope': 'telescope.svg',
            'TelescopeArray': 'telescope.svg',
            'Collector': 'telescope.svg',
            'Interferometer': 'interferometer.svg',
            'Atmosphere': 'atmosphere.svg',
            'AdaptiveOptics': 'adaptive_optics.svg',
            'Coronagraph': 'coronagraph.svg',
            'BeamSplitter': 'beam_splitter.svg',
            'FiberIn': 'fiber_in.svg',
            'FiberOut': 'fiber_out.svg',
            'PhotonicChip': 'photonic_chip.svg',
            'TOPS': 'photonic_chip.svg',
            'MMI': 'photonic_chip.svg',
            'Camera': 'camera.svg'
        }
        
        icon_file = icon_map.get(layer_name, 'telescope.svg')  # Default to telescope
        icon_path = asset_dir / icon_file
        
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
        
        # Add icon if SVG exists
        if icon_path.exists():
            # For now, just indicate icon presence with a marker
            # Full SVG rendering would require additional library (e.g., svgpath2mpl)
            ax.plot(x, y, 'o', markersize=15, color='#3498DB', zorder=3, alpha=0.3)
        
        # Add label below box
        display_name = self._get_display_name(layer)
        ax.text(x, y - box_height/2 - 0.15, display_name,
               ha='center', va='top', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='none', alpha=0.8))
    
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
