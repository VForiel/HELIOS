# ⚙️ Architecture

HELIOS uses a **Layered Architecture** with a **Pull-Based Execution Model** to simulate optical pipelines.

## 1. Layer Types
The simulation pipeline is structured into 5 strict types of layers, representing the physical flow of information.

### 🔷 GenerationLayer (Field Definition)
*   **Role**: Defines the continuous electromagnetic field (The "World").
*   **Input**: None (or previous Generation layer).
*   **Output**: Full continuous Wavefront/Field.
*   **Constraint**: Unique path (Single Link).
*   **Components**: `PlanetarySystem`.

### 🌫️ EnvironmentLayer (Field Modifier)
*   **Role**: Modifies the continuous electromagnetic field before sampling (e.g. atmosphere, dust).
*   **Input**: `GenerationLayer` or `EnvironmentLayer`.
*   **Output**: Full continuous Wavefront/Field.
*   **Constraint**: Unique path.
*   **Components**: `Atmosphere`, `DustCloud`.

### 🔶 SamplingLayer (Field Discrete Sampling)
*   **Role**: Interfaces the continuous world with the discrete instrument. Samples the field at specific aperture positions.
*   **Input**: `GenerationLayer` (The Field).
*   **Output**: Array of discrete Wavefronts (Optical Beams).
*   **Constraint**: 1 Input (Field) -> N Outputs (Beams). Use this for broadcasting.
*   **Components**: `TelescopeArray`.

### 🟣 OpticalLayer (Beam Propagation)
*   **Role**: Transports and modifies optical beams within the instrument.
*   **Input**: Optical Beam(s).
*   **Output**: Optical Beam(s).
*   **Constraint**: Can be 1-to-1, N-to-N, or N-to-M (Beam Splitters).
*   **Components**: `Lens`, `Mirror`, `BeamSplitter`, `Fiber`, `Coronagraph`.

### 🔴 DetectionLayer (Photon-to-Data)
*   **Role**: Converts optical energy into digital data. The "Sink" of the optical simulation.
*   **Input**: Optical Beam(s).
*   **Output**: Data Array (Pixels).
*   **Components**: `Camera`, `Detector`.

### ⚪ DataLayer (Post-Processing)
*   **Role**: Processes digital data after acquisition.
*   **Input**: Data Array.
*   **Output**: Data Array.
*   **Components**: `ImageStacker`, `NullingCombiner` (Algorithm).

---

## 2. Pull Model (Lazy Evaluation)
HELIOS does not "push" photons from the start. Instead, the final detector "pulls" the information it needs.

### Logic Flow
1.  User calls `pipeline.observe()` (or `Camera.get_output()`).
2.  `Camera` asks: "Do I have an input?" -> No.
3.  `Camera` calls `previous_layer.get_output_wavefront()`.
4.  Recursively, this climbs back up to the `Scene`.
5.  `Scene` generates the initial field and returns it.
6.  Each layer processes the data on the way back down.

### Why?
*   **Efficiency**: We only calculate what is connected to the detector.
*   **State Inspection**: Since every layer computes its output on demand, we can inspect the state *between* any two layers by asking the previous one for its output.

## 3. Caching & Invalidation
To optimise performance, every layer has a **Cache**:
*   `_cached_input`: usage internal
*   `_cached_output`: result of the last process()

### Invalidation Strategy
If a parameter changes in a Layer (e.g., changing Telescope diameter):
1.  **Local**: The layer invalidates its own cache.
2.  **Propagation**: The Pipeline is notified and invalidates **ALL downstream layers**.
3.  **Result**: The next `observe()` will re-compute only the dirty part of the pipeline.

---

## Detailed Class Architecture

This section presents the detailed architecture of the HELIOS package in the form of a UML class diagram, showing all classes, their attributes, methods, and interactions.

### Architecture Overview

HELIOS is organized into two main modules:
- **`core`**: Base classes defining the simulation structure
- **`components`**: Specific optical and astronomical components

```{mermaid}
classDiagram
    %% ============================================
    %% CORE CLASSES - Base Architecture
    %% ============================================
    
    class Wavefront {
        +Quantity wavelength
        +ndarray field
        +Quantity pixel_scale
        +__init__(wavelength, size)
        +propagate(distance)
    }
    
    class Component {
        <<abstract>>
        +str name
        +Layer layer
        +Pipeline pipeline
        +__init__(name)
        +description(indent, full) str
        +_get_detailed_attributes() dict
        +process(wavefront, pipeline)* Wavefront
    }
    
    class Layer {
        <<abstract>>
        +str name
        +List~Component~ elements
        +Pipeline pipeline
        +__init__(name)
        +add_element(element)
        +description(indent, full) str
        +_get_detailed_attributes() dict
        +process(wavefront, pipeline)* Wavefront
    }
    
    class Pipeline {
        +List layers
        +Any date
        +Any declination
        +Quantity time
        +__init__(date, declination, **kwargs)
        +add_layer(layer)
        +observe(wavelength, size, **kwargs) Any
        +description(full) str
        +plot_architecture(filename, show_elements)
        +_build_graphviz_tree()
    }
    
    class Simulation {
        <<utility>>
    }
    
    %% Relationships - Core
    Pipeline "1" *-- "0..*" Layer : contains
    Layer "1" *-- "0..*" Component : contains
    Component ..> Wavefront : processes
    Layer ..> Wavefront : processes
    Component --> Layer : belongs to
    Component --> Pipeline : references
    Layer --> Pipeline : belongs to
    
    %% ============================================
    %% SCENE COMPONENTS - Celestial Objects
    %% ============================================
    
    class CelestialBody {
        <<abstract>>
        +Tuple position
        +__init__(position, name, **kwargs)
        +process(wavefront, pipeline) Wavefront
        +sed(wavelengths, **kwargs) Tuple
        +flux_at(wavelength, **kwargs) Quantity
        +plot_sed(wavelengths, ax, label, color, **kwargs)
    }
    
    class Star {
        +Quantity temperature
        +float magnitude
        +Quantity mass
        +__init__(temperature, magnitude, mass, **kwargs)
        +sed(wavelengths, **kwargs) Tuple
        +_get_detailed_attributes() dict
    }
    
    class Planet {
        +Quantity mass
        +Quantity radius
        +Quantity temperature
        +float albedo
        +float reflection_ratio
        +PlanetarySystem scene
        +__init__(mass, radius, temperature, albedo, **kwargs)
        +sed(wavelengths, temperature, include_reflection, **kwargs) Tuple
        +_get_detailed_attributes() dict
    }
    
    class ZodiacalLight {
        +Quantity temperature
        +float brightness
        +__init__(temperature, brightness, **kwargs)
        +sed(wavelengths, **kwargs) Tuple
        +_get_detailed_attributes() dict
    }
    
    class LocalZodi {
        +Quantity temperature
        +float brightness
        +__init__(temperature, brightness, **kwargs)
        +sed(wavelengths, **kwargs) Tuple
        +_get_detailed_attributes() dict
    }
    
    class ExoZodi {
        +Quantity temperature
        +float brightness
        +__init__(temperature, brightness, **kwargs)
        +sed(wavelengths, **kwargs) Tuple
        +_get_detailed_attributes() dict
    }
    
    class PlanetarySystem {
        +Quantity distance
        +List~CelestialBody~ bodies
        +__init__(distance, name)
        +add_body(body)
        +add_star(**kwargs) Star
        +add_planet(**kwargs) Planet
        +add_local_zodi(**kwargs) LocalZodi
        +add_exo_zodi(**kwargs) ExoZodi
        +plot(ax, unit, center_on_star, **kwargs)
        +process(wavefront, pipeline) Wavefront
        +_get_detailed_attributes() dict
    }
    
    %% Relationships - Scene
    CelestialBody --|> Component
    Star --|> CelestialBody
    Planet --|> CelestialBody
    ZodiacalLight --|> CelestialBody
    LocalZodi --|> ZodiacalLight
    ExoZodi --|> ZodiacalLight
    PlanetarySystem --|> Layer
    PlanetarySystem "1" *-- "0..*" CelestialBody : contains
    Planet --> PlanetarySystem : references
    
    %% ============================================
    %% COLLECTOR COMPONENTS - Telescopes
    %% ============================================
    
    class Pupil {
        +Quantity diameter
        +List primitives
        +__init__(diameter)
        +add_disk(radius, center, value)
        +add_hexagon(radius, center, value, rotation)
        +add_central_obscuration(diameter)
        +add_spiders(arms, width, angle, angles)
        +add_segmented_primary(seg_flat, rings, rotation, gap)
        +get_array(npix, soft, oversample) ndarray
        +plot(npix, soft, oversample, ax, cmap)
        +diffraction_pattern(npix, soft, oversample, wavelength) ndarray
        +plot_diffraction_pattern(npix, soft, oversample, ax, **kwargs)
        +image_through_pupil(scene_array, soft, oversample, normalize) ndarray
        +plot_image_through_pupil(scene_array, soft, oversample, ax, **kwargs)
        +jwst()$ Pupil
        +vlt()$ Pupil
        +elt()$ Pupil
        +like(name)$ Pupil
    }
    
    class Collector {
        +Pupil pupil
        +Tuple position
        +Quantity size
        +dict metadata
        +__init__(pupil, position, size, name, **metadata)
        +process(wavefront, pipeline) Wavefront
        +_get_detailed_attributes() dict
    }
    
    class TelescopeArray {
        +Quantity latitude
        +Quantity longitude
        +Quantity altitude
        +List~Collector~ elements
        +__init__(name, latitude, longitude, altitude)
        +collectors() List~Collector~
        +add_collector(pupil, position, size, name, **kwargs)
        +is_interferometric() bool
        +get_baseline_array() ndarray
        +plot_array(ax, show_pupils, pupil_scale)
        +vlti(uts)$ TelescopeArray
        +life()$ TelescopeArray
        +_get_detailed_attributes() dict
    }
    
    %% Relationships - Collectors
    Collector --|> Component
    TelescopeArray --|> Layer
    Collector "1" *-- "1" Pupil : has
    TelescopeArray "1" *-- "1..*" Collector : contains
    
    %% ============================================
    %% ATMOSPHERE & ADAPTIVE OPTICS
    %% ============================================
    
    class Atmosphere {
        %% EnvironmentLayer component
        +Quantity rms
        +Quantity wind_speed
        +float wind_direction
        +int seed
        +Quantity inner_scale
        +Quantity outer_scale
        +ndarray _frozen_screen
        +int _screen_size
        +__init__(rms, wind_speed, wind_direction, seed, **kwargs)
        +_generate_frozen_screen(N, oversample) ndarray
        +_extract_screen_at_time(time, N) ndarray
        +process(wavefront, pipeline) Wavefront
        +plot_screen_animation(collectors, wavelength, **kwargs)
        +plot_animation(collectors, wavelength, **kwargs)
    }
    
    class AdaptiveOptics {
        +dict coeffs
        +bool normalize
        +__init__(coeffs, normalize, name)
        +noll_to_nm(j)$ Tuple
        +_radial_polynomial(n, m, r) ndarray
        +_zernike_nm(n, m, rho, theta) ndarray
        +process(wavefront, pipeline) Wavefront
    }
    
    %% Relationships - Atmosphere
    Atmosphere --|> Component
    AdaptiveOptics --|> Component
    
    %% ============================================
    %% CORONAGRAPH
    %% ============================================
    
    class Coronagraph {
        +str phase_mask
        +__init__(phase_mask, name)
        +process(wavefront, pipeline) Wavefront
        +mask_array(npix, kind, charge, lam, diameter, fov) ndarray
        +plot_mask(npix, kind, charge, ax, **kwargs)
        +image_from_scene(scene_array, soft, oversample, **kwargs) ndarray
        +plot_image_from_scene(scene_array, ax, **kwargs)
    }
    
    %% Relationships - Coronagraph
    Coronagraph --|> Component
    
    %% ============================================
    %% DETECTOR
    %% ============================================
    
    class Camera {
        +Tuple pixels
        +Quantity dark_current
        +Quantity read_noise
        +Quantity integration_time
        +float quantum_efficiency
        +float gain
        +Quantity thermal_background
        +Quantity thermal_background_temp
        +__init__(pixels, dark_current, read_noise, **kwargs)
        +get_raw_image(wavefront, pipeline) ndarray
        +get_dark() ndarray
        +get_image(wavefront, pipeline, subtract_dark) ndarray
        +process(wavefront, pipeline) ndarray
        +_get_detailed_attributes() dict
    }
    
    %% Relationships - Camera
    Camera --|> Component
    
    %% ============================================
    %% BEAM SPLITTING
    %% ============================================
    
    class BeamSplitter {
        +float cutoff
        +__init__(cutoff, name)
        +process(wavefront, pipeline) List~Wavefront~
        +_get_detailed_attributes() dict
    }
    
    %% Relationships - BeamSplitter
    BeamSplitter --|> Component
    
    %% ============================================
    %% FIBER OPTICS
    %% ============================================
    
    class FiberIn {
        +int modes
        +__init__(modes, **kwargs)
        +process(wavefront, pipeline) Wavefront
    }
    
    class FiberOut {
        +__init__(**kwargs)
        +process(wavefront, pipeline) Wavefront
    }
    
    %% Relationships - Fibers
    FiberIn --|> Layer
    FiberOut --|> Layer
    
    %% ============================================
    %% PHOTONIC INTEGRATED CIRCUITS
    %% ============================================
    
    class PhotonicChip {
        +int inputs
        +Quantity lambda0
        +List~Layer~ layers
        +__init__(inputs, lambda0, **kwargs)
        +add_layer(layer)
        +process(wavefronts, pipeline) List~Wavefront~
    }
    
    class TOPS {
        +Union on_paths
        +__init__(on_paths)
        +process(wavefronts, pipeline) List~Wavefront~
    }
    
    class MMI {
        +ndarray matrix
        +__init__(matrix)
        +process(wavefronts, pipeline) List~Wavefront~
    }
    
    %% Relationships - Photonics
    PhotonicChip --|> Layer
    TOPS --|> Layer
    MMI --|> Layer
    PhotonicChip "1" *-- "0..*" Layer : contains
```

### Module Descriptions

#### `core` Module

##### `Wavefront`
Represents the complex electromagnetic field (amplitude and phase) at a given wavelength. It is the main object that flows through the simulation chain.

**Key Attributes:**
- `wavelength`: Light wavelength
- `field`: Complex 2D array representing amplitude and phase
- `pixel_scale`: Physical scale per pixel

**Core Method: `propagate()`**

The `Wavefront.propagate()` method is **the heart of HELIOS**. All optical simulations fundamentally rely on this method to propagate electromagnetic wavefronts through space.

**Conceptual Model:**
- **Input**: Wavefront $\psi_0$ represented by a grid of physical size $L_0$ with resolution $N_0$ pixels (complex amplitude distribution) and a wavelength $\lambda$.
- **Propagation**: Simulates propagation over distance $d$ using physically realistic diffraction.
- **Output**: Wavefront $\psi_f$ that can be imaged on a detector grid of size $L_f$ with resolution $N_f$ pixels.

The goal of this high-level method is to encapsulate several propagation methods (Fraunhofer, Fresnel, etc.) and automatically select the appropriate one based on the input parameters.

**Underlying Physics & Libraries**

This complex physics engine relies on established methods. Below is a comparison of relevant libraries that inform or can be integrated into HELIOS's approach:

| Library | Language / Backend | Propagation Type | Strengths | Weaknesses | Main Limitations | Link |
|:---|:---|:---|:---|:---|:---|:---|
| **POPPY** | Python (NumPy) | Fresnel / Fraunhofer | Astro standard (JWST), excellent doc, Astropy units | CPU only, rigid architecture | Not flexible outside astro, FFT-centric | [Link](https://poppy-optics.readthedocs.io) |
| **HCIPy** | Python (NumPy/C++) | FFT, Fresnel, MFT | Very complete (AO, polarization), end-to-end | Complex API, steep learning curve | Scalar by default, tricky sampling | [Link](https://hcipy.readthedocs.io) |
| **PROPER** | IDL / Python / Matlab | Fresnel (FFT) | NASA/JPL reference, very robust | Aging API, not very Pythonic | Strictly propagation | [Link](https://proper-library.sourceforge.net) |
| **dLux** | Python (JAX, XLA) | Differentiable Fourier optics | Autodiff, GPU/TPU, inverse calibration | Small community, JAX mindset | GPU VRAM usage, young project | [Link](https://github.com/LouisDesdoigts/dLux) |
| **Diffractio** | Python | RS, BPM, Fresnel, Vector | Educational, vector support, X-ray | Modest performance | Not scalable for heavy pipelines | [Link](https://diffractio.readthedocs.io) |
| **waveprop** | Python (NumPy / PyTorch) | Angular Spectrum, Fresnel | Simple, GPU possible, clear | Few optical elements | Low level | [Link](https://github.com/ebezzam/waveprop) |
| **LightPipes** | C++ / Python | Scalar FFT | Fast, laser modes, cavities | Not astro-focused, weak unit support | Older model | [Link](https://opticspy.github.io/lightpipes) |
| **PyOptica** | Python | Scalar diffraction | Lightweight, readable | Small community | Few advanced elements | [Link](https://pypi.org/project/pyoptica) |
| **PyNX** | Python (CUDA/OpenCL) | Fresnel / FFT | Very fast, GPU, HPC | Not generalist | X-ray oriented | [Link](https://pynx.esrf.fr) |

**Note**: While the `Wavefront` class encapsulates complex logic beyond a simple grid (multiple sources, metadata, history tracking), the fundamental principle is that any wavefront can be imaged with a grid of arbitrary size $L$ and resolution $N$ at any location in the optical system. This flexibility is what makes `propagate()` the foundation of all optical simulations in HELIOS.

##### `Component`
Abstract base class for all individual physical components. Each element can process a wavefront independently.

**Key Methods:**
- `process(wavefront, pipeline)`: Abstract method to be implemented by subclasses
- `description()`: Generates a text description of the element

##### `Layer`
Abstract base class for logical groups of elements. A layer can contain multiple elements that process wavefronts in parallel.

**Key Methods:**
- `add_element(element)`: Adds an element to the layer
- `process(wavefront, pipeline)`: Processes the wavefront through all elements

##### `Pipeline`
Main simulation orchestrator. Manages the sequence of layers and the execution of the observation.

**Key Methods:**
- `add_layer(layer)`: Adds a layer to the simulation
- `observe(wavelength, size)`: Executes the full simulation
- `plot_architecture()`: Visualizes the simulation architecture

#### `components` Module

##### Astronomical Scene

**`PlanetarySystem`**: Layer containing all celestial objects
- Manages distance to the star system
- Contains stars, planets, and zodiacal light

**`CelestialBody`**: Base class for all celestial objects
- `sed()`: Calculates Spectral Energy Distribution
- `flux_at()`: Calculates flux at a specific wavelength

**`Star`**: Star with blackbody spectrum
- Temperature, magnitude, mass

**`Planet`**: Planet with thermal emission and stellar reflection
- Mass, radius, temperature, albedo
- Automatic calculation of stellar light reflection

**`ZodiacalLight`**, **`LocalZodi`**, **`ExoZodi`**: Zodiacal light (local and exo-zodiacal)

##### Light Collectors

**`Pupil`**: Optical aperture geometry
- Construction via primitives (disks, hexagons, spiders)
- Diffraction pattern calculation
- Presets: JWST, VLT, ELT

**`Collector`**: Individual telescope
- Associates a pupil with a spatial position
- Applies the pupil mask to the wavefront

**`TelescopeArray`**: Telescope array
- Interferometric configuration management
- Presets: VLTI, LIFE

##### Atmosphere and Adaptive Optics

**`Atmosphere`**: Kolmogorov atmospheric turbulence
- Phase screens with temporal evolution (frozen-flow)
- Parameters: RMS, wind speed, inner/outer scales

**`AdaptiveOptics`**: Adaptive optics correction
- Correction based on Zernike polynomials
- Zernike coefficients (n,m) or Noll indices

##### High Contrast Imaging

**`Coronagraph`**: Coronagraphic masks
- Types: 4 quadrants, vortex, Lyot
- On-axis starlight suppression

##### Detection

**`Camera`**: Detector with realistic noise
- Dark current, read noise
- Thermal background, quantum efficiency
- Methods: `get_raw_image()`, `get_dark()`, `get_image()`

##### Beam Splitting

**`BeamSplitter`**: Optical beam splitter
- Splits a wavefront into multiple paths
- Transmission/reflection parameter

##### Integrated Photonics

**`FiberIn`** / **`FiberOut`**: Fiber coupling
- Optical fiber input/output
- Single or multiple modes

**`PhotonicChip`**: Integrated photonic circuit
- Contains layers of photonic components
- **`TOPS`**: Thermo-optic phase shifter
- **`MMI`**: Multi-mode interference coupler (coupling matrix)

### Relationships and Data Flow

#### Inheritance Hierarchy

```
Component (abstract)
├── CelestialBody (abstract)
│   ├── Star
│   ├── Planet
│   └── ZodiacalLight
│       ├── LocalZodi
│       └── ExoZodi
├── Collector
├── Atmosphere
├── AdaptiveOptics
├── Coronagraph
├── Camera
└── BeamSplitter

Layer (abstract)
├── PlanetarySystem
├── TelescopeArray
├── FiberIn
├── FiberOut
├── PhotonicChip
├── TOPS
└── MMI
```

#### Composition

- **Pipeline** contains **Layers**
- **Layer** contains **Components**
- **PlanetarySystem** contains **CelestialBodies**
- **TelescopeArray** contains **Collectors**
- **Collector** contains a **Pupil**
- **PhotonicChip** contains photonic **Layers**

#### Processing Flow

1. **Pipeline.observe()** initializes a **Wavefront**
2. The **Wavefront** passes sequentially through each **Layer**
3. Each **Layer** applies its **Components** to the **Wavefront**
4. The final result is returned (image, intensity, etc.)

#### Typical Chain Example

```
PlanetarySystem → TelescopeArray → Atmosphere → AdaptiveOptics → Coronagraph → Camera
```

Each component transforms the wavefront according to its physical properties, enabling realistic end-to-end simulation of astronomical observations.

### Implementation Notes

- Abstract classes (`Component`, `Layer`) define the common interface
- All classes inheriting from `Component` or `Layer` must implement `process()`
- Bidirectional references (`Component.layer`, `Layer.pipeline`) allow access to the global pipeline
- `_get_detailed_attributes()` methods allow generation of detailed descriptions
- Static methods (marked `$`) are factory methods for creating predefined configurations
