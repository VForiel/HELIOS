# UML Diagram Visualization

HELIOS provides automatic UML-style diagram generation for visualizing complete optical simulation pipelines.

## Overview

The `Context.plot_uml_diagram()` method generates a visual representation of your optical setup, displaying all layers from scene (left) to detector (right). This is particularly useful for:

- **Documentation**: Quickly document complex optical systems
- **Debugging**: Verify pipeline structure before running simulations
- **Communication**: Share system designs with collaborators
- **Teaching**: Explain optical concepts with clear visual diagrams

## Basic Usage

```python
import helios
from astropy import units as u
import matplotlib.pyplot as plt

# Create a simple pipeline
scene = helios.Scene(distance=10*u.pc)
scene.add(helios.Star(temperature=5700*u.K, magnitude=5))

telescope = helios.TelescopeArray(name="VLT")
telescope.add_collector(pupil=helios.Pupil.vlt(), position=(0, 0), size=8*u.m)

camera = helios.Camera(pixels=(512, 512))

# Build context
ctx = helios.Context()
ctx.add_layer(scene)
ctx.add_layer(telescope)
ctx.add_layer(camera)

# Generate diagram
fig = ctx.plot_uml_diagram()
plt.show()

# Or save to file
ctx.plot_uml_diagram(save_path='my_optical_system.png')
```

## Features

### Left-to-Right Layout

Diagrams are laid out from left (scene) to right (detector), matching the physical light propagation path.

### Schematic Icons

Each component is represented by a schematic icon:
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

### Parallel Paths (Beam Splitting)

When beam splitters create multiple paths, they are displayed vertically:

```python
# Create dual-channel system
ctx = helios.Context()
ctx.add_layer(scene)
ctx.add_layer(telescope)
ctx.add_layer(helios.BeamSplitter(cutoff=0.5))
ctx.add_layer([camera1, camera2])  # Parallel paths shown vertically

fig = ctx.plot_uml_diagram()
plt.show()
```

### Component Labels

Each component is labeled with:
- Its **class name** (e.g., "Camera", "Atmosphere")
- Its **custom name** if provided (e.g., "VLT", "ELT", "VLTI")

### Signal Flow Arrows

Red arrows show the signal flow between components, making it easy to trace the light path through the system.

## Advanced Examples

### Complete Exoplanet Detection System

```python
# Scene with star and planet
scene = helios.Scene(distance=10*u.pc)
scene.add(helios.Star(temperature=5700*u.K, magnitude=5, position=(0, 0)))
scene.add(helios.Planet(temperature=300*u.K, magnitude=22, position=(100*u.mas, 0*u.mas)))

# Atmospheric turbulence
atmosphere = helios.Atmosphere(rms=200*u.nm, wind_speed=8*u.m/u.s, seed=42)

# Large telescope
telescope = helios.TelescopeArray(name="ELT")
telescope.add_collector(pupil=helios.Pupil.elt(), position=(0, 0), size=39*u.m)

# Adaptive optics correction
ao = helios.AdaptiveOptics(coeffs={(1, 1): 0.15, (2, 0): 0.08})

# Coronagraph for high contrast
coronagraph = helios.Coronagraph(phase_mask='4quadrants')

# Science camera
camera = helios.Camera(pixels=(1024, 1024))

# Build and visualize
ctx = helios.Context()
ctx.add_layer(scene)
ctx.add_layer(atmosphere)
ctx.add_layer(telescope)
ctx.add_layer(ao)
ctx.add_layer(coronagraph)
ctx.add_layer(camera)

fig = ctx.plot_uml_diagram(figsize=(18, 8))
plt.savefig('exoplanet_system.png', dpi=300)
plt.show()
```

### Interferometer Configuration

```python
# Binary star system
scene = helios.Scene(distance=100*u.pc)
scene.add(helios.Star(temperature=6000*u.K, magnitude=6, position=(0, 0)))
scene.add(helios.Star(temperature=5500*u.K, magnitude=7, position=(50*u.mas, 30*u.mas)))

# Three-telescope interferometer
interferometer = helios.Interferometer(name="VLTI")
interferometer.add_collector(pupil=helios.Pupil.vlt(), position=(0*u.m, 0*u.m), size=8.2*u.m)
interferometer.add_collector(pupil=helios.Pupil.vlt(), position=(60*u.m, 0*u.m), size=8.2*u.m)
interferometer.add_collector(pupil=helios.Pupil.vlt(), position=(30*u.m, 52*u.m), size=8.2*u.m)

camera = helios.Camera(pixels=(256, 256))

ctx = helios.Context()
ctx.add_layer(scene)
ctx.add_layer(interferometer)
ctx.add_layer(camera)

fig = ctx.plot_uml_diagram()
plt.show()
```

### Fiber-Fed Spectrograph

```python
# Star to observe
scene = helios.Scene(distance=25*u.pc)
scene.add(helios.Star(temperature=5200*u.K, magnitude=7))

# Telescope
telescope = helios.TelescopeArray(name="Gemini-South")
telescope.add_collector(pupil=helios.Pupil(), position=(0, 0), size=8*u.m)

# Fiber coupling and photonic processing
fiber_in = helios.FiberIn(mode_field_diameter=10*u.um)
photonic_chip = helios.PhotonicChip(inputs=2, lambda0=1.55*u.um)
fiber_out = helios.FiberOut(mode_field_diameter=10*u.um)

# Detector
camera = helios.Camera(pixels=(512, 512))

ctx = helios.Context()
ctx.add_layer(scene)
ctx.add_layer(telescope)
ctx.add_layer(fiber_in)
ctx.add_layer(photonic_chip)
ctx.add_layer(fiber_out)
ctx.add_layer(camera)

fig = ctx.plot_uml_diagram(figsize=(16, 8))
plt.show()
```

## Customization Options

### Figure Size

Adjust the figure size for different layouts:

```python
# Wide layout for many components
fig = ctx.plot_uml_diagram(figsize=(20, 8))

# Tall layout for many parallel paths
fig = ctx.plot_uml_diagram(figsize=(12, 14))
```

### Layer Spacing

Control horizontal distance between components:

```python
# Compact layout
fig = ctx.plot_uml_diagram(layer_spacing=1.5)

# Spread out layout
fig = ctx.plot_uml_diagram(layer_spacing=3.0)
```

### Saving Output

Save diagrams in various formats:

```python
# PNG for presentations
ctx.plot_uml_diagram(save_path='system.png')

# High-resolution for publications
fig = ctx.plot_uml_diagram()
fig.savefig('system_hires.png', dpi=300, bbox_inches='tight')

# PDF for vector graphics
fig.savefig('system.pdf', bbox_inches='tight')
```

## API Reference

### `Context.plot_uml_diagram()`

```python
def plot_uml_diagram(
    self,
    figsize: Tuple[float, float] = (16, 10),
    layer_spacing: float = 2.0,
    save_path: Optional[str] = None
) -> plt.Figure
```

**Parameters:**

- `figsize` : tuple of float, optional  
  Figure size as (width, height) in inches. Default: (16, 10)

- `layer_spacing` : float, optional  
  Horizontal distance between layers. Default: 2.0

- `save_path` : str, optional  
  If provided, save the figure to this path

**Returns:**

- `fig` : matplotlib.figure.Figure  
  The generated figure

**Examples:**

```python
# Basic usage
fig = ctx.plot_uml_diagram()

# Custom size and spacing
fig = ctx.plot_uml_diagram(figsize=(18, 10), layer_spacing=2.5)

# Auto-save
ctx.plot_uml_diagram(save_path='my_system.png')
```

## Technical Details

### Icon Assets

Icons are stored as SVG files in `src/helios/assets/`. Each component type is mapped to a specific icon file.

### Layout Algorithm

The layout algorithm:
1. **Builds layer tree**: Analyzes the pipeline structure to identify parallel paths
2. **Counts parallel paths**: Determines maximum vertical extent
3. **Positions components**: Places each layer with appropriate spacing
4. **Draws connections**: Creates arrows showing signal flow between layers
5. **Renders icons**: Displays schematic representation of each component

### Beam Splitter Handling

When a beam splitter is encountered:
1. Subsequent parallel layers are positioned vertically
2. Connections are drawn from previous layer(s) to all branches
3. Vertical spacing is calculated automatically based on number of branches
4. Each branch maintains its own path through subsequent layers

## Tips and Best Practices

### 1. Name Your Components

Give descriptive names to major components for clearer diagrams:

```python
telescope = helios.TelescopeArray(name="VLT UT4")
interferometer = helios.Interferometer(name="CHARA Array")
```

### 2. Use Appropriate Figure Sizes

- **Simple pipelines**: (12, 8)
- **Complex pipelines**: (18, 10)
- **Many parallel paths**: (16, 12)

### 3. Combine with Documentation

Include diagrams in your documentation:

```python
# Generate diagram for README
ctx.plot_uml_diagram(save_path='docs/system_diagram.png')
```

### 4. Verify Before Simulation

Always check the diagram before running expensive simulations:

```python
# Quick visual check
ctx.plot_uml_diagram()
plt.show()

# If correct, run simulation
result = ctx.observe()
```

## Limitations

- **Icon rendering**: Currently uses placeholder markers instead of full SVG rendering
- **Complex branching**: Very complex beam splitter networks may require manual layout adjustment
- **3D information**: Spatial layouts (e.g., interferometer baselines) are not shown in 3D

## Future Enhancements

Planned improvements include:
- Full SVG icon rendering
- Interactive diagrams with parameter tooltips
- 3D visualization for interferometer arrays
- Export to various vector formats
- Custom icon support
- Automatic layout optimization

## See Also

- [Context API Documentation](api/core.md#context)
- [Component Gallery](components/index.md)
- [Demo Notebooks](../demo.ipynb)
