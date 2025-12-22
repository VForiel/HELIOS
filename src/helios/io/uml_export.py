import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MPath
from pathlib import Path
import xml.etree.ElementTree as ET
import re
from typing import Tuple, Union, Optional, List, Any

# We use Any for Pipeline/Layer to avoid circular imports if strictly type checking in this file
# but we can try to import if TYPE_CHECKING
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.pipeline import Pipeline
    from ..core.layer import Layer

def plot_uml_diagram(pipeline: 'Pipeline', 
                     figsize: Tuple[float, float] = (16, 10), 
                     layer_spacing: float = 2.0,
                     save_path: Optional[str] = None,
                     return_type: str = 'figure') -> Union[plt.Figure, np.ndarray, Tuple[plt.Figure, np.ndarray]]:
    """
    Generate a UML-style diagram of the complete optical setup.
    
    This function creates a visual representation of the simulation pipeline.
    """
    # Validate architecture first
    pipeline.validate_architecture()
    
    fig, ax = plt.subplots(figsize=figsize)
    # ax.set_xlim will be set later or auto-scaled
    
    # Get asset directory
    # src/helios/io/uml_export.py -> src/helios/io -> src/helios -> assets
    asset_dir = Path(__file__).parent.parent / "assets"
    
    # Build layer tree structure to handle beam splitting
    layer_tree = _build_layer_tree(pipeline.layers)
    max_paths = _count_max_parallel_paths(layer_tree)
    
    # Set y-limits based on number of parallel paths
    y_margin = 1.0
    ax.set_ylim(-y_margin, max_paths + y_margin)
    
    # Draw each layer
    _draw_layer_tree(ax, layer_tree, layer_spacing, asset_dir)
    
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
        try:
            image = np.asarray(fig.canvas.buffer_rgba())
        except AttributeError:
            image = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            # Convert ARGB to RGBA if needed or handle buffer_rgba which is standard in newer mpl
        
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

def _build_layer_tree(layers: List[Union['Layer', List['Layer']]]) -> List[dict]:
    """Build a tree structure representing layer organization."""
    tree = []
    for i, layer in enumerate(layers):
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

def _count_max_parallel_paths(tree: List[dict]) -> int:
    """Count maximum number of parallel paths at any point."""
    max_paths = 1
    current_paths = 1
    
    for node in tree:
        if node['is_parallel']:
            current_paths = max(current_paths, node['num_branches'])
            max_paths = max(max_paths, current_paths)
    
    return max_paths

def _draw_layer_tree(ax: plt.Axes, tree: List[dict], 
                    spacing: float, asset_dir: Path):
    """Draw the complete layer tree with icons and connections."""
    # Track active paths (y-positions)
    active_paths = [0.5]  # Start with single path at center
    
    # Pre-calculate x-positions to handle Swap spacing
    x_coords = []
    current_x = 0.0
    for node in tree:
        if not node['is_parallel'] and type(node['layer']).__name__ == 'Swap':
            x_coords.append(current_x)
        else:
            x_coords.append(current_x)
            current_x += spacing

    # Track photonic components for background rectangles
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
            y_positions = _calculate_branch_positions(num_branches)
            
            # Draw each branch
            for j, (layer, y_pos) in enumerate(zip(layer_list, y_positions)):
                # Handle different layer types
                if layer is None:
                    # Draw pass-through line
                    ax.plot([x_pos - 0.4, x_pos + 0.4], [y_pos, y_pos], 
                           color='#E74C3C', linestyle='-', linewidth=2, zorder=1)
                    
                elif type(layer).__name__ == 'Swap':
                    # Draw as a standard block when in parallel mode
                    _draw_layer_icon(ax, layer, x_pos, y_pos, asset_dir, 
                                        layer_index=i, element_index=j)

                else:
                    # Draw standard layer icon
                    _draw_layer_icon(ax, layer, x_pos, y_pos, asset_dir, 
                                        layer_index=i, element_index=j)
                
                # Track photonic components
                if layer is not None and type(layer).__name__ in photonic_types:
                    chip_id = 'default'
                    if hasattr(layer, 'layer') and layer.layer is not None:
                        if type(layer.layer).__name__ == 'PhotonicChip':
                            chip_id = id(layer.layer)
                    
                    if chip_id not in photonic_groups:
                        photonic_groups[chip_id] = []
                    photonic_groups[chip_id].append((x_pos, y_pos))
                
                # Draw connection from previous layer(s)
                if i > 0:
                    prev_node = tree[i-1]
                    is_prev_permutator = False
                    if not prev_node['is_parallel']:
                        if type(prev_node['layer']).__name__ == 'Swap':
                            is_prev_permutator = True

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
                        # Perfect match! Route sequentially
                        start_idx = sum(expected_inputs[:j])
                        n_in = expected_inputs[j]
                        
                        for k in range(n_in):
                            if start_idx + k < len(active_paths):
                                prev_y = active_paths[start_idx + k]
                                if not is_prev_permutator:
                                    _draw_arrow(ax, x_coords[i-1] + 0.4, prev_y, 
                                               x_pos - 0.4, y_pos, arrowstyle=arrow_style)
                                           
                    elif len(active_paths) == num_branches:
                        # 1-to-1 connection
                        prev_y = active_paths[j]
                        if not is_prev_permutator:
                            _draw_arrow(ax, x_coords[i-1] + 0.4, prev_y, 
                                           x_pos - 0.4, y_pos, arrowstyle=arrow_style)
                    else:
                        # All-to-All connection
                        for prev_y in active_paths:
                            if not is_prev_permutator:
                                _draw_arrow(ax, x_coords[i-1] + 0.4, prev_y, 
                                           x_pos - 0.4, y_pos, arrowstyle=arrow_style)
            
            # Update active paths
            new_active_paths = []
            for j, (layer, y_pos) in enumerate(zip(layer_list, y_positions)):
                n_out = 1
                if layer is not None and hasattr(layer, 'num_outputs'):
                    n_out = layer.num_outputs
                
                for _ in range(n_out):
                    new_active_paths.append(y_pos)
            
            active_paths = new_active_paths
            
        else:
            # Single layer
            layer = node['layer']
            
            if type(layer).__name__ == 'Swap':
                # Special handling for Swap (CrossSection)
                num_paths = len(active_paths)
                new_y_positions = _calculate_branch_positions(num_paths)
                
                if i > 0:
                    mapping = layer.mapping
                    for dest_idx, src_idx in enumerate(mapping):
                        if src_idx < len(active_paths) and dest_idx < len(new_y_positions):
                            y_src = active_paths[src_idx]
                            y_dest = new_y_positions[dest_idx]
                            
                            arrow_style = '-|>'
                            if i + 1 < len(tree):
                                next_node = tree[i+1]
                                if next_node['is_parallel']:
                                    # Tricky next layer logic simplified:
                                    pass # Assuming OK for visualisation
                                else:
                                    if next_node['layer'] is None:
                                        arrow_style = '-'

                            _draw_arrow(ax, x_coords[i-1] + 0.4, y_src, 
                                           x_coords[i+1] - 0.4, y_dest, arrowstyle=arrow_style)
                
                active_paths = new_y_positions
                
            else:
                y_pos = sum(active_paths) / len(active_paths)
                _draw_layer_icon(ax, layer, x_pos, y_pos, asset_dir, 
                                    layer_index=i)
            
                if type(layer).__name__ in photonic_types:
                    chip_id = 'default'
                    if hasattr(layer, 'layer') and layer.layer is not None:
                        if type(layer.layer).__name__ == 'PhotonicChip':
                            chip_id = id(layer.layer)
                    
                    if chip_id not in photonic_groups:
                        photonic_groups[chip_id] = []
                    photonic_groups[chip_id].append((x_pos, y_pos))
            
                if i > 0:
                    prev_node = tree[i-1]
                    is_prev_permutator = False
                    if not prev_node['is_parallel']:
                        if type(prev_node['layer']).__name__ == 'Swap':
                            is_prev_permutator = True
                    
                    for prev_y in active_paths:
                        if not is_prev_permutator:
                            _draw_arrow(ax, x_coords[i-1] + 0.4, prev_y,
                                           x_pos - 0.4, y_pos)
                
                n_out = 1
                if hasattr(layer, 'num_outputs'):
                    n_out = layer.num_outputs
            
                if n_out > 1:
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
        
        label = "Photonic Circuit"
        ax.text((min_x + max_x)/2, max_y + pad_y, label,
               ha='center', va='bottom', fontsize=10, fontweight='bold',
               color='#7F8C8D')

def _calculate_branch_positions(num_branches: int) -> List[float]:
    """Calculate y-positions for parallel branches."""
    if num_branches == 1:
        return [0.5]
    
    spacing = 1.5
    total_height = (num_branches - 1) * spacing
    start_y = 0.5 - total_height / 2
    
    return [start_y + (num_branches - 1 - i) * spacing for i in range(num_branches)]

def _draw_layer_icon(ax: plt.Axes, layer: 'Layer', 
                    x: float, y: float, asset_dir: Path, 
                    layer_index: Optional[int] = None, 
                    element_index: Optional[int] = None):
    """Draw a layer icon with label."""
    layer_name = type(layer).__name__
    
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
    
    box_width = 0.6
    box_height = 0.6
    
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
    
    icon_path = asset_dir / icon_file
    if icon_path.exists():
        try:
            _render_svg_icon(ax, icon_path, x, y, box_width*0.8)
        except Exception as e:
            ax.plot(x, y, marker, markersize=15, color=color, zorder=3, alpha=0.8)
    else:
        ax.plot(x, y, marker, markersize=15, color=color, zorder=3, alpha=0.8)
    
    display_name = _get_display_name(layer)
        
    ax.text(x, y - box_height/2 - 0.15, display_name,
           ha='center', va='top', fontsize=8, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                    edgecolor='none', alpha=0.8))
    
    type_text = f"({layer_name})"
    ax.text(x, y - box_height/2 - 0.4, type_text,
           ha='center', va='top', fontsize=7, color='#7F8C8D')
           
    if layer_index is not None:
        if element_index is not None:
            idx_text = f"[{layer_index},{element_index}]"
        else:
            idx_text = f"[{layer_index}]"
        
        ax.text(x, y - box_height/2 - 0.55, idx_text,
               ha='center', va='top', fontsize=7, family='monospace', color='#2C3E50')

def _render_svg_icon(ax: plt.Axes, svg_path: Path, center_x: float, center_y: float, size: float):
    """Render a simple SVG icon onto the axes."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    scale = size / 100.0
    offset_x = center_x - size/2
    offset_y = center_y - size/2
    
    def trans_x(val): return offset_x + float(val) * scale
    def trans_y(val): return offset_y + (100 - float(val)) * scale
    def trans_len(val): return float(val) * scale
    
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
    
    for elem in root.iter():
        tag = elem.tag.split('}')[-1]
        
        if tag == 'path':
            d = elem.get('d')
            if d:
                color, fill, lw, alpha, ls = get_style(elem)
                _draw_svg_path(ax, d, color, fill, lw, alpha, ls, trans_x, trans_y)
        
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

def _draw_svg_path(ax, d, color, fill, lw, alpha, ls, tx, ty):
    """Parse simple SVG path d string and draw PathPatch."""
    tokens = re.findall(r'([a-zA-Z])|([-+]?\d*\.?\d+)', d)
    tokens = [t[0] or t[1] for t in tokens]
    
    verts = []
    codes = []
    
    i = 0
    current_pos = (0, 0)
    
    while i < len(tokens):
        cmd = tokens[i]
        i += 1
        
        if cmd == 'M': 
            x = float(tokens[i]); y = float(tokens[i+1])
            verts.append((tx(x), ty(y)))
            codes.append(MPath.MOVETO)
            current_pos = (x, y)
            i += 2
        elif cmd == 'L':
            x = float(tokens[i]); y = float(tokens[i+1])
            verts.append((tx(x), ty(y)))
            codes.append(MPath.LINETO)
            current_pos = (x, y)
            i += 2
        elif cmd == 'C':
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
        elif cmd == 'Q':
            x1 = float(tokens[i]); y1 = float(tokens[i+1])
            x = float(tokens[i+2]); y = float(tokens[i+3])
            
            verts.append((tx(x1), ty(y1)))
            verts.append((tx(x), ty(y)))
            
            codes.append(MPath.CURVE3)
            codes.append(MPath.CURVE3)
            
            current_pos = (x, y)
            i += 4
        elif cmd == 'Z':
            verts.append((0,0))
            codes.append(MPath.CLOSEPOLY)
        
    if verts:
        path = MPath(verts, codes)
        patch = PathPatch(path, facecolor=fill, edgecolor=color, linewidth=lw, 
                        alpha=alpha, linestyle=ls, zorder=3)
        ax.add_patch(patch)

def _get_display_name(layer: 'Layer') -> str:
    """Get display name for a layer."""
    layer_name = type(layer).__name__
    
    if layer_name == 'Swap':
        return f"Swap: {layer.mapping}"
    
    if hasattr(layer, 'name') and layer.name:
        return layer.name
    
    return layer_name

def _draw_arrow(ax: plt.Axes, x1: float, y1: float, 
               x2: float, y2: float, arrowstyle: str = '-|>'):
    """Draw an arrow representing signal flow."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=arrowstyle,
        color='#E74C3C',
        linewidth=2,
        mutation_scale=20,
        zorder=1
    )
    ax.add_patch(arrow)
