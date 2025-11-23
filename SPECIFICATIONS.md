# HELIOS - Project Specifications for maintainers and AI Agents

## Project Overview
**HELIOS** (Hierarchical End-to-end Lightpath & Instrumental response Simulation) is a Python module for end-to-end simulation of astronomical observations, designed for high-performance scientific computing.

## Core Objectives
1. **End-to-end simulation**: From astronomical scene to detector output
2. **Modular architecture**: Layered design allowing flexible composition
3. **High performance**: C++ optimized extensions for computationally intensive operations
4. **Scientific rigor**: Full support for physical units via `astropy.units`
5. **Extensibility**: Easy addition of new components and instruments

## Architecture Constraints

### Layered System
- **Context**: Main simulation orchestrator managing layers
- **Layers**: Sequential or parallel processing stages
- **Components**: Individual optical/photonic/detector elements
- Each layer processes wavefront/signal and passes to next layer

### Data Flow
1. **Scene** (first layer): Generates initial wavefront from celestial objects
2. **Collectors** (second layer): Telescope/collector array
3. **Optical/Photonic components**: Beam manipulation, coronagraphy, interferometry
4. **Detectors** (terminal layers): Convert light to digital signal

### Unit Handling
- **Interface**: All user-facing parameters MUST use `astropy.Quantity`
- **Internal storage**: Convert to native Python types (`int`, `float`, `complex`, `numpy.ndarray`)
- **Rationale**: Performance optimization while maintaining scientific correctness

## Technical Requirements

### Code Quality
- **Language**: English for all code, comments, and documentation
- **Documentation**: Sphinx autodoc from docstrings for all user-facing functions
- **Testing**: Unit tests embedded in each module file
- **Readability**: Clear, well-commented code

### Performance
- **C++ Extensions**: Use `pybind11` for performance-critical functions
- **Build System**: `scikit-build-core` for seamless C++/Python integration
- **Target operations**: Wavefront propagation, diffraction calculations, matrix operations

### Deployment
- **PyPI**: Automated publishing via GitHub Actions
- **ReadTheDocs**: Automatic documentation generation
- **Examples**: Jupyter notebooks demonstrating usage

## Component Specifications

### Scene Component
- **Distance**: Defined at Scene level (distance to observer)
- **Celestial Bodies**: Position relative to scene barycenter
  - `Star`: temperature, magnitude, mass, position
  - `Planet`: mass, position
  - `ExoZodiacal`, `Galaxy`, `Nebula`, `AsteroidBelt`: extensible

#### Zodiacal Light and ExoZodiacal
- **Zodiacal**: Local zodiacal light produced by interplanetary dust in the Solar System. Treated as a diffuse background component.
  - Attributes: `brightness` (relative surface brightness, float), optional `radius` (angular size as `astropy.Quantity`, e.g., `arcsec`).
  - Behavior: drawn as a faint, filled disk centered on the scene origin in visualizations. If `radius` is omitted the component fills the plotted view; `brightness` controls rendering alpha.

- **ExoZodiacal**: Analogous diffuse dust disk around the target system (exozodiacal light).
  - Attributes: `brightness` (relative surface brightness), optional `radius` (angular extent in `astropy.Quantity`).
  - Behavior: rendered like `Zodiacal` but with distinct default styling; spatially centered on the target star (scene origin). Useful to model astrophysical background and contrast limits.

Notes:
- All user-facing parameters for zodiacal/exozodiacal components MUST be provided as `astropy.Quantity` where applicable (e.g., `radius=1*u.arcsec`).
- Visualization: `scene.plot()` will render these components as semi-transparent disks before plotting point-like objects (stars/planets are shown on top).
- Implementation detail: implementations may map `brightness` to a rendering alpha in the range ~0.03–0.8; this is a visualization aid only and not a physical radiometric model.
- **Visualization**: `scene.plot()` method showing spatial arrangement in arcsec/mas

### Optical Components
- **Collectors**: Telescope array with geographic location
- **Pupil**: Aperture shape (segments, spider, secondary obstruction)
- **Coronagraph**: Phase masks (4-quadrant, vortex, etc.)
- **BeamSplitter**: Wavelength-dependent splitting
- **Adaptive Optics**: Deformable mirrors
- **Fibers**: Single/multi-mode coupling

### Photonic Components
- **PhotonicChip**: Integrated photonic circuits
- **TOPS**: Thermo-Optic Phase Shifters
- **MMI**: Multi-Mode Interference couplers (matrix operations)

### Detectors
- **Camera**: CCD/CMOS with noise models (dark current, read noise)
- **Spectrometer**: Wavelength-resolved detection

## Usage Patterns

### Coronagraphy Example
```python
scene = helios.Scene(distance=10*u.pc)
scene.add(Star(...))
scene.add(Planet(...))

collectors = helios.Collectors(...)
coronograph = helios.components.Coronagraph(phase_mask='4quadrants')
camera = helios.Camera(pixels=(1024,1024))

context = helios.Context()
context.add_layer(scene)
context.add_layer(collectors)
context.add_layer(coronograph)
context.add_layer(camera)

result = context.observe()
```

### Interferometry Example
```python
collectors = helios.Collectors(...)
collectors.add(size=8*u.m, position=(x1,y1))
collectors.add(size=8*u.m, position=(x2,y2))

context.add_layer(scene)
context.add_layer(collectors)
context.add_layer([helios.components.FiberIn() for _ in range(4)])

chip = helios.components.PhotonicChip(inputs=4)
chip.add_layer(helios.components.TOPS())
chip.add_layer(helios.components.MMI(matrix=...))

context.add_layer(chip)
context.add_layer([helios.Camera(...) for _ in range(4)])

result = context.observe()
```

## Design Principles

### Parallel Processing
- List of layers = parallel paths (e.g., after beam splitter)
- Single layer = sequential processing
- Context handles signal routing automatically

### Preset Support
- Common configurations accessible via `.like()` methods
- Example: `Pupil.like('JWST')`, `PhaseMask.like('4quadrants')`

### Extensibility
- New components inherit from `Layer` base class
- Implement `process(wavefront, context)` method
- Register in appropriate module `__init__.py`

## Notebook development guidance

- When adding or modifying notebook cells that create objects or compute data (e.g., SEDs), first run the same code in a plain Python script to validate data shapes, units, and types before adding plotting commands.
- Example workflow:
  1. Create a small script `tools/run_demo_seds.py` reproducing the notebook cells that build `Scene`, `Star`, `Planet`, etc.
 2. Print summaries of arrays (length, min/max, sample values) and `astropy.Quantity` units.
 3. Once the outputs look correct, copy the plotting code into the notebook cell.
- Rationale: plotting functions may hide unit/shape issues; running a headless script first makes debugging easier and is reproducible in CI.

## Current Implementation Status
This is **v0.1** - a structural foundation. The architecture (Context, Layers, C++ hooks) is established, but physics implementation is minimal. Future iterations will add:
- Actual wavefront propagation (Fresnel/Fraunhofer)
- Physical optics (diffraction, aberrations)
- Realistic detector models
- Atmospheric effects
- More celestial body types
- Expanded photonic components

## Development Workflow
1. Plan changes in `implementation_plan.md`
2. Update `task.md` checklist
3. Implement with embedded unit tests
4. Update examples and documentation
5. Verify with integration tests
6. Update `walkthrough.md`

## References
- Inspiration: PHISE project (https://github.com/VForiel/PHISE)
- Build system: scikit-build-core
- Bindings: pybind11
- Documentation: Sphinx + ReadTheDocs
- Units: astropy.units
