# HELIOS Component Icons

This directory contains SVG icons for visualizing HELIOS optical components in UML diagrams.

## Icon Files

| File | Component | Description |
|------|-----------|-------------|
| `scene.svg` | Scene | Star system with planets and rays |
| `telescope.svg` | Telescope, TelescopeArray, Collector | Circular aperture with spider vanes and central obscuration |
| `interferometer.svg` | Interferometer | Multiple telescopes with beam combiner |
| `atmosphere.svg` | Atmosphere | Wavy lines representing turbulence with wind arrow |
| `adaptive_optics.svg` | AdaptiveOptics | Deformable mirror with actuator grid |
| `coronagraph.svg` | Coronagraph | Focal plane mask blocking starlight |
| `beam_splitter.svg` | BeamSplitter | Diagonal mirror splitting beam into two paths |
| `fiber_in.svg` | FiberIn | Coupling lens and fiber input |
| `fiber_out.svg` | FiberOut | Fiber output with diverging beam |
| `photonic_chip.svg` | PhotonicChip, TOPS, MMI | Integrated waveguide circuit |
| `camera.svg` | Camera | Detector array with pixel grid |

## Icon Specifications

- **Format**: SVG (Scalable Vector Graphics)
- **Dimensions**: 80×80 pixels
- **Color scheme**: Professional scientific palette
  - Blues: #4682B4, #87CEEB, #0066CC
  - Grays: #2C3E50, #696969, #C0C0C0
  - Accent: #FDB813 (star), #4169E1 (planet)

## Usage

Icons are automatically loaded by `Context.plot_uml_diagram()` method. The mapping between component classes and icon files is defined in `src/helios/core/context.py`.

```python
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
```

## Adding New Icons

To add an icon for a new component:

1. Create an 80×80px SVG file following the existing style
2. Save it in this directory
3. Update the `icon_map` in `context.py`
4. Test with `Context.plot_uml_diagram()`

### Design Guidelines

- Use simple, recognizable shapes
- Limit colors to 2-3 main colors
- Ensure visibility at small sizes
- Match scientific/technical aesthetic
- Include identifying features (e.g., spider vanes for telescopes)

## License

These icons are part of HELIOS and follow the same license as the main project.
