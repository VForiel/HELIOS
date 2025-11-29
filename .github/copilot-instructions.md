# HELIOS AI Coding Agent Instructions

## Project Overview
HELIOS (Hierarchical End-to-end Lightpath & Instrumental response Simulation) is a Python astronomical simulation framework for modeling observations from scene → optics → detector. Built for scientific computing with C++ optimization via pybind11.

## Python Environment (CRITICAL)

**ALWAYS activate the virtual environment first** before running any Python commands:

```powershell
# Activate venv (PowerShell) - DO THIS FIRST
& .venv\Scripts\Activate.ps1

# Navigate normally after activation
cd tests
pytest

# For documentation
cd docs
.\make.bat html
```

**Why venv activation is required**:
- The venv uses **Python 3.13.9** (required for modern packages)
- Sphinx Breeze theme requires Python ≥3.10
- All project dependencies are installed in venv only
- Documentation builds will **fail** without venv activation

**Proper workflow**:
1. **Activate venv ONCE** at the start: `& .venv\Scripts\Activate.ps1`
2. **Navigate normally** with `cd` commands
3. **Use make.bat** for documentation: `cd docs; .\make.bat html`
4. **Never** combine activation + command in single line (e.g. don't do `& .venv\Scripts\python.exe -m sphinx ...`)

**Build commands**:
- **Documentation**: `cd docs; .\make.bat html` (make.bat auto-detects venv)
- **Tests**: `pytest` or `python -m pytest`
- **Install package**: `pip install -e .`

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

### Performance Architecture
**High-level interface with low-level performance**: The API is designed with two layers:

1. **User-facing API**: Accepts `astropy.Quantity` objects for all physical quantities (distances, masses, temperatures, wavelengths, etc.)
2. **Internal storage**: Converts to native Python types (float, numpy arrays) with fixed or dimensionless units for performance

**Conversion pattern**:
```python
# At API boundary (constructor, setters)
def __init__(self, rms: u.Quantity = 100*u.nm):
    self.rms = rms  # High-level API

@property
def rms(self):
    return self._rms_internal * u.m  # Return as Quantity

@rms.setter
def rms(self, value: u.Quantity):
    self._rms_internal = value.to(u.m).value  # Convert on set
```

**Performance optimization priorities**:
- Use **numba JIT compilation** for computationally intensive loops
- Implement performance-critical code in **C++** via pybind11 bindings (`src/helios/cpp/`)
- Prefer numpy vectorized operations over Python loops
- Cache expensive computations when possible (e.g., frozen turbulence screens)

The goal is to provide a user-friendly, physically intuitive API while achieving the best possible computational performance for scientific simulations.

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

### Educational Philosophy
**HELIOS is a scientific project with educational clarity** - the code must be rigorous and scientifically accurate, but explained so that any scientist can understand it, even if they are not experts in the specific field. This means:

- **Explain the physics**: Every optical/astronomical concept must be explained in docstrings and comments
- **Provide context**: Don't assume users know why a particular algorithm or formula is used
- **Use clear variable names**: Prefer descriptive names like `phase_rms` over `phi_rms`, `optical_path_difference` over `opd` (units are already in `astropy.Quantity` objects)
- **Add educational comments**: Explain the "why" not just the "what"
- **Include references**: When implementing published algorithms, cite the paper/textbook
- **Validate physically**: Tests should verify that results make physical sense, not just that code runs

### Every Function Must Have
1. **English docstring** (numpy-style for Sphinx autodoc)
   - Explain the physical concept being modeled
   - Define all parameters with units and physical meaning
   - Include mathematical formulas when relevant (using LaTeX in docstrings)
2. **Clear English comments** explaining non-obvious logic
   - Explain physical reasoning behind algorithmic choices
   - Clarify coordinate systems, sign conventions, normalizations
3. **Unit test** validating both correctness AND physical coherence
   - Test edge cases and boundary conditions
   - Verify conservation laws (energy, flux, etc.)
   - Check dimensional analysis (units consistency)
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

## Notebook Validation Protocol (CRITICAL)

**When modifying code that impacts notebook cells**, you MUST validate the changes by executing the affected code:

### Validation Methods (in order of preference)

1. **Direct execution via `run_notebook_cell` tool** (preferred when available)
   - Execute the modified notebook cells directly
   - Verify outputs match expectations
   - Check that no errors are raised

2. **Standalone Python script** (when direct execution not available)
   - Extract the relevant cell code into a temporary Python script
   - Add necessary imports and setup code
   - Execute via `run_in_terminal` to validate correctness

3. **Data validation without plots** (most reliable)
   - **PREFERRED**: Skip matplotlib visualization, validate data directly with `print()` statements
   - Check array shapes, value ranges, statistical properties (mean, std, min, max)
   - Verify physical coherence (units, magnitudes, conservation laws)
   - Example:
     ```python
     print(f"PSF shape: {psf.shape}")
     print(f"Peak value: {psf.max():.2e}")
     print(f"Total flux: {psf.sum():.2e}")
     print(f"Strehl ratio: {strehl:.3f}")
     ```

4. **File-based visualization** (only if visual inspection required)
   - If plots are necessary, save to files with `plt.savefig('test_output.png')`
   - Use descriptive filenames indicating what is being tested
   - Analyze saved images to verify correctness
   - Clean up test output files after validation

### What to Validate
- ✅ Code executes without errors
- ✅ Output shapes and types are correct
- ✅ Numerical values are in expected ranges
- ✅ Physical quantities have correct units and magnitudes
- ✅ Visualizations (if needed) display expected features

### Example Validation Pattern
```python
# GOOD: Validate data directly
import sys
sys.path.insert(0, '../src')
import helios
from astropy import units as u
import numpy as np

# Test atmospheric phase screen generation
atm = helios.Atmosphere(rms=0.5*u.rad, seed=42)
wf = helios.Wavefront(wavelength=550e-9*u.m, size=512)
wf.field = np.ones((512, 512), dtype=np.complex128)
wf_atm = atm.process(wf, None)
phase = np.angle(wf_atm.field)

# Validate without plotting
print(f"Phase shape: {phase.shape}")  # Expected: (512, 512)
print(f"Phase range: [{phase.min():.3f}, {phase.max():.3f}] rad")  # Expected: [-π, π]
print(f"Phase RMS: {np.std(phase):.3f} rad")  # Expected: ~0.5
assert phase.shape == (512, 512), "Incorrect shape"
assert -np.pi <= phase.min() <= phase.max() <= np.pi, "Phase out of range"
print("✓ Validation passed")
```

**Always validate changes before finalizing** - this catches bugs early and ensures physical coherence.

## What NOT to Do
- ❌ Don't use raw floats for physical quantities in APIs
- ❌ Don't break Layer abstraction (components must implement `process()`)
- ❌ Don't hardcode array sizes - use npix parameters
- ❌ Don't ignore units when reading existing code - maintain consistency
- ❌ Don't commit code without docstrings and unit tests
- ❌ Don't skip the agent modification log
- ❌ Don't modify notebook cells without validating the changes execute correctly
