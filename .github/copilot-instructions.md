# HELIOS AI Coding Agent Instructions

## Project Overview
HELIOS (Hierarchical End-to-end Lightpath & Instrumental response Simulation) is a Python astronomical simulation framework for modeling observations from scene → optics → detector. Built for scientific computing with C++ optimization via pybind11.

## Architecture Fundamentals

### Layered Processing Model
The core abstraction is a **Layer** pipeline orchestrated by **Context**:
- All components inherit from `helios.core.context.Layer` and implement `process(wavefront, context)`
- `Context.observe()` sequentially processes layers: Scene → Collectors → Optics/Photonics → Detectors
- Layers can be parallel (list of layers) or sequential (single layer)
- Signal flow: `Scene` generates initial `Wavefront`, each layer transforms it, `Camera` produces final array

**Key files:** `src/helios/core/context.py`, `src/helios/core/simulation.py`

### Component Categories
1. **Scene** (`components/scene.py`): First layer, generates celestial objects (Star, Planet, ExoZodiacal, Zodiacal)
2. **Optics** (`components/optics.py`): Collectors, Pupil, Coronagraph, BeamSplitter, FiberIn/Out, Atmosphere, AdaptiveOptics
3. **Photonics** (`components/photonics.py`): PhotonicChip, TOPS, MMI for integrated photonic circuits
4. **Detectors** (`components/detectors.py`): Camera (terminal layer, returns numpy array)

## Critical Conventions

### Units: astropy.Quantity at API Boundaries
**MANDATORY PATTERN**: All user-facing parameters use `astropy.units.Quantity`. Internal storage converts to native Python types for performance.

```python
# ✅ Correct - API accepts Quantity
def __init__(self, temperature: u.Quantity = 5778*u.K, mass: u.Quantity = 1*u.M_sun):
    self.temperature = temperature  # Store as Quantity OR...
    self.mass_kg = mass.to(u.kg).value  # Convert to float internally

# ❌ Wrong - never accept raw floats for physical quantities
def __init__(self, temperature: float, mass: float):
```

**When adding new components**: Always use `u.Quantity` for distances (u.m, u.AU, u.arcsec), masses (u.M_sun, u.M_jup), temperatures (u.K), wavelengths (u.nm, u.um).

### Pupil Geometry System
`Pupil` class builds aperture masks via primitives (disk, hexagon, spiders, segments):
- **Coordinate system**: Pupil diameter in meters, elements positioned relative to center
- **Segmented primaries**: Use `add_segmented_primary(seg_flat, rings, gap)` with flat-to-flat segment size
- **Anti-aliasing**: Use `get_array(npix, soft=True, oversample=4)` for smooth edges
- **Presets exist**: `Pupil.jwst()`, `Pupil.vlt()`, `Pupil.elt()` - use these as references

**Example from `optics.py` lines 470-485**: ELT uses hexagonal tiling with gap=0.004m, 5 rings → 91 segments total (minus central)

### Tests Embedded in Modules
**PROJECT SPECIFIC**: Test functions live at the bottom of implementation files, not just in `/tests`:
```python
# At end of component files
def test_camera():
    cam = Camera(pixels=(100, 100))
    assert cam.pixels == (100, 100)

if __name__ == "__main__":
    test_camera()
```

Main test suite is in `/tests` with pytest, but inline tests validate module behavior during development.

## Build & Development Workflow

### Installation & Testing
```powershell
# Install in development mode
pip install .

# Run tests (pytest discovers tests/ directory)
pytest

# Run specific test file
pytest tests/test_helios.py

# Execute demo notebook (integration test)
python tools/execute_demo_notebook.py
```

### Package Structure
- **Source**: `src/helios/` (installed package)
- **Build artifacts**: `build/lib/helios/` (transient, gitignored)
- **C++ extensions**: `src/helios/cpp/` (pybind11 bindings, not yet implemented)

**Important**: Tests import via `sys.path.insert(0, '../src')` to use local source, not installed package.

## Common Patterns

### Adding New Celestial Bodies
1. Inherit from `CelestialBody` in `components/scene.py`
2. Accept `position: Tuple[u.Quantity, u.Quantity]` (angular or physical coords)
3. Implement `sed(wavelengths, temperature, **kwargs)` returning modified blackbody
4. Override defaults (e.g., `Planet` defaults to 300K, `ExoZodiacal` to 270K)

### Adding New Optical Layers
1. Inherit from `Layer` in `core/context.py`
2. Implement `process(self, wavefront: Wavefront, context: Context) -> Wavefront`
3. Convert input `u.Quantity` parameters to native types in `__init__`
4. Register in `components/__init__.py` for import

### Visualization Methods
Scene and optical components implement `.plot()` methods:
- `Scene.plot()`: Shows celestial objects in angular coordinates (arcsec/mas)
- `Pupil.plot()`: Displays aperture mask
- Use matplotlib, return Axes for chaining

## Key File References

- **Entry point**: `src/helios/__init__.py` exports `Context`, `components`
- **Test example**: `tests/test_helios.py` shows full Scene→Collectors→Camera flow
- **Pupil geometry**: `tests/test_pupil_geometry.py` validates segment counts using flood-fill
- **Specifications**: `SPECIFICATIONS.md` contains detailed component specs and zodiacal light behavior

## Documentation & CI

- **Docs**: Sphinx with MyST (markdown), built via `.github/workflows/docs.yml`
- **Publishing**: Automated PyPI via `.github/workflows/publish.yml`
- **Docstrings**: All public APIs must have numpy-style docstrings for Sphinx autodoc

## Code Quality Requirements (CRITICAL)

### Every Function Must Have
1. **English docstring** (numpy-style for Sphinx autodoc)
2. **Clear English comments** explaining non-obvious logic
3. **Unit test** validating both correctness AND physical coherence
4. **Documentation generation** - ensure Sphinx autodoc can process it

### Testing Philosophy
Tests must verify:
- ✅ Code executes without errors
- ✅ **Physical results are coherent** (units, magnitudes, conservation laws)
- ✅ Edge cases and boundary conditions

Example from existing codebase (`components/scene.py`):
```python
def sed(self, wavelengths: Optional[u.Quantity] = None, ...):
    """
    Return a modified blackbody SED for this object.
    
    Parameters
    ----------
    wavelengths : astropy.Quantity, optional
        Array of wavelengths. If None, creates log-spaced grid.
    temperature : astropy.Quantity
        Temperature in Kelvin.
    
    Returns
    -------
    tuple
        (wavelengths, sed_values) in W/(m² m sr)
    """
    # Implementation with physical validation
```

### Documentation Maintenance
- **Auto-generated docs**: All public APIs appear in `docs/` via Sphinx
- **Static docs**: Add markdown files in `docs/` for architectural explanations
- **Keep synchronized**: Update docstrings when changing function signatures

## Agent Modification Logs (MANDATORY)

**Every modification session must create a log file** in `agent-logs/` with format:
```
agent-logs/YYYY.MM.DD-NN_<topic>.md
```

Where:
- `YYYY.MM.DD`: Date (e.g., 2025.11.25)
- `NN`: Sequential number if multiple sessions same day (01, 02, ...)
- `<topic>`: 1-2 word summary (e.g., `pupil-geometry`, `camera-noise`)

**Log content should include**:
- Summary of changes made
- Files modified
- New functions/classes added
- Tests added/updated
- Any breaking changes or migration notes

Example: `agent-logs/2025.11.25-01_copilot-instructions.md`

## What NOT to Do
- ❌ Don't use raw floats for physical quantities in APIs
- ❌ Don't break Layer abstraction (components must implement `process()`)
- ❌ Don't hardcode array sizes - use npix parameters
- ❌ Don't ignore units when reading existing code - maintain consistency
- ❌ Don't commit code without docstrings and unit tests
- ❌ Don't skip the agent modification log
