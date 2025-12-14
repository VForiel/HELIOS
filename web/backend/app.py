import io
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Tuple, Literal, Union, Dict, Any
from datetime import datetime
from astropy import units as u

import helios
from helios.components import Zodiacal, Atmosphere, Pupil
import helios.components.photonics as photonics
import helios.components.fibers as fibers
import helios.components.lens as lens_comp
import helios.components.beam_splitter as bs_comp
import helios.components.coronagraph as corona_comp

app = FastAPI(title="Helios Web API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"]
)

# ... (skip to preview_layer)

# --- Pydantic Models for Component Payloads ---
# These match the structures used in the previous Config models but are reused dynamically.

class StarData(BaseModel):
    temperature: float = 5778
    magnitude: float = 4.83
    x_arcsec: float = 0.0
    y_arcsec: float = 0.0

class PlanetData(BaseModel):
    mass: float = 1.0
    separation: float = 1.0
    angle: float = 0.0
    radius: Optional[float] = None
    x_arcsec: Optional[float] = None
    y_arcsec: Optional[float] = None

class ZodiacalData(BaseModel):
    enabled: bool = False
    brightness: float = 1.0
    radius: Optional[float] = None

class ScenePayload(BaseModel):
    model_config = ConfigDict(extra='allow')  # Allow view_mode and figsize from frontend
    stars: List[StarData] = []
    planets: List[PlanetData] = []
    zodiacal: ZodiacalData = ZodiacalData()
    view_mode: str = 'geometry'

class AtmospherePayload(BaseModel):
    enabled: bool = True
    rms_nm: float = 100.0
    wind_speed: float = 5.0
    seed: Optional[int] = None  # Random seed for reproducible turbulence

class CollectorData(BaseModel):
    id: Optional[str] = None
    x: float = 0
    y: float = 0
    diameter: float = 8.0
    pupil_type: str = "Circular"
    central_obstruction: float = 0
    spiders: int = 0

class TelescopePayload(BaseModel):
    preset: str = "Single"
    diameter: Optional[float] = 8.0
    pupil_type: str = "Circular"
    central_obstruction: float = 0.0
    spiders: int = 0
    collectors: List[CollectorData] = []

class CameraPayload(BaseModel):
    model_config = ConfigDict(extra='allow')  # Allow view_mode and figsize from frontend
    wavelength: float = 1.0
    exposure: float = 0.1

class LensPayload(BaseModel):
    focal_length: float = 1.0

class BeamSplitterPayload(BaseModel):
    split_ratio: float = 0.5

class CoronagraphPayload(BaseModel):
    type: str = "4quadrants" # 4quadrants, vortex, etc.

class FiberPayload(BaseModel):
    modes: int = 1
    name: Optional[str] = None

class PhotonicPayload(BaseModel):
    type: Literal['y_splitter', 'tops', 'mmi', 'swap']
    phase: Optional[float] = 0.0
    matrix_preset: Optional[str] = "hadamard"
    mapping: Optional[List[int]] = None
    name: Optional[str] = None

# Generic Layer Wrapper
class LayerConfig(BaseModel):
    type: Literal['scene', 'atmosphere', 'telescope', 'camera', 'lens', 'beam_splitter', 'coronagraph', 'fiber_in', 'fiber_out', 'photonic']
    config: Union[ScenePayload, AtmospherePayload, TelescopePayload, CameraPayload, LensPayload, BeamSplitterPayload, CoronagraphPayload, FiberPayload, PhotonicPayload, Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class PipelineRequest(BaseModel):
    mode: Literal['pipeline'] = 'pipeline'
    layers: List[Union[LayerConfig, List[LayerConfig]]]

# --- Helper Functions ---

def get_layer_type(element_type: str) -> str:
    """
    Map element type to layer type class name for validation.
    
    This mapping enforces the architectural rules:
    - GenerationLayer: Scene, Atmosphere (generate electromagnetic fields)
    - SamplingLayer: Telescope (samples continuous field into discrete beams)
    - OpticalLayer: Lenses, mirrors, fibers, photonics (propagate/modify beams)
    - DetectionLayer: Camera (converts photons to digital data)
    - DataLayer: Data processing algorithms (not yet implemented)
    """
    mapping = {
        'scene': 'GenerationLayer',
        'atmosphere': 'GenerationLayer',
        'telescope': 'SamplingLayer',
        'lens': 'OpticalLayer',
        'beam_splitter': 'OpticalLayer',
        'coronagraph': 'OpticalLayer',
        'fiber_in': 'OpticalLayer',
        'fiber_out': 'OpticalLayer',
        'photonic': 'OpticalLayer',
        'mmi': 'OpticalLayer',
        'camera': 'DetectionLayer',
        # DataLayer types will be added when implemented
    }
    return mapping.get(element_type, 'Layer')  # Fallback to generic Layer


def create_scene(config: ScenePayload):
    # Fixed distance for conversion scaling (10 pc -> 1 AU = 0.1 arcsec)
    sys_distance = 10 * u.pc 
    scene = helios.Scene(distance=sys_distance)
    
    # Stars
    if not config.stars:
        config.stars.append(StarData()) # Default star

    for s in config.stars:
        star = helios.Star(temperature=s.temperature * u.K, magnitude=s.magnitude)
        star.position = (s.x_arcsec * u.arcsec, s.y_arcsec * u.arcsec)
        scene.add(star)
    
    # Planets
    for p in config.planets:
        planet = helios.Planet(
            mass=p.mass * u.M_jup,
            radius=p.radius * u.R_jup if p.radius else None,
            orbit_radius=p.separation * u.AU
        )
        # Position logic
        dist_pc = sys_distance.to(u.pc).value
        sep_arcsec = p.separation / dist_pc if dist_pc > 0 else 0
        
        if p.x_arcsec is not None and p.y_arcsec is not None:
            planet.position = (p.x_arcsec * u.arcsec, p.y_arcsec * u.arcsec)
        else:
            angle_rad = np.radians(p.angle)
            x = sep_arcsec * np.cos(angle_rad)
            y = sep_arcsec * np.sin(angle_rad)
            planet.position = (x * u.arcsec, y * u.arcsec)
        scene.add(planet)

    # Zodiacal
    if config.zodiacal.enabled:
        zodi = helios.Zodiacal(
            brightness=config.zodiacal.brightness,
            radius=config.zodiacal.radius * u.arcsec if config.zodiacal.radius else None
        )
        scene.add(zodi)
            
    return scene

def create_atmosphere(config: AtmospherePayload):
    if not config.enabled:
        return None # Should we skip? The visual node exists, so maybe user wants it. 
        # But if enabled flag allows toggling without removing node, verify.
        # If disabled, return None or Identity? Context.add_layer handles None? No.
        # We will filter out None later.
    return helios.Atmosphere(
        rms=config.rms_nm * u.nm,
        wind_speed=config.wind_speed * u.m / u.s,
        seed=None
    )

def create_telescope(config: TelescopePayload):
    if config.preset == "VLTI-UT":
        return helios.TelescopeArray.vlti(uts=True)
    elif config.preset == "VLTI-AT":
        return helios.TelescopeArray.vlti(uts=False)
    elif config.preset == "LIFE":
        return helios.TelescopeArray.life()
    elif config.preset == "Single":
         telescope = helios.TelescopeArray(name="Single Telescope")
         diam = config.diameter if config.diameter else 8.0
         
         if config.pupil_type == "VLT":
             pupil = helios.Pupil.vlt()
         elif config.pupil_type == "JWST":
             pupil = helios.Pupil.jwst()
         elif config.pupil_type == "Obstructed":
             pupil = helios.Pupil(diameter=diam * u.m)
             pupil.add_disk(radius=diam/2 * u.m)
             pupil.add_central_obscuration(diameter=diam * config.central_obstruction * u.m)
             if config.spiders > 0:
                 pupil.add_spiders(arms=config.spiders, width=0.02 * diam * u.m)
         else:
             pupil = helios.Pupil(diameter=diam * u.m)
             pupil.add_disk(radius=diam/2 * u.m)
             
         telescope.add_collector(pupil=pupil, position=(0,0), size=diam*u.m)
         return telescope
    else:
        # Custom
        telescope = helios.TelescopeArray(name="Custom Array")
        for i, col in enumerate(config.collectors):
            d = col.diameter * u.m
            # Simplified pupil creation logic for brevity
            # (Matches previous implementation logic)
            if col.pupil_type == "VLT":
                p = helios.Pupil.vlt()
            elif col.pupil_type == "JWST":
                p = helios.Pupil.jwst()
            elif col.pupil_type == "Obstructed":
                p = helios.Pupil(diameter=col.diameter * u.m)
                p.add_disk(radius=col.diameter/2 * u.m) 
                p.add_central_obscuration(diameter=col.diameter * col.central_obstruction * u.m)
                if col.spiders > 0: p.add_spiders(arms=col.spiders, width=0.02 * col.diameter * u.m)
            else: # Circular
                p = helios.Pupil(diameter=col.diameter * u.m)
                p.add_disk(radius=col.diameter/2 * u.m)
            
            telescope.add_collector(pupil=p, position=(col.x * u.m, col.y * u.m), size=col.diameter * u.m, name=f"T{i+1}")
        return telescope


def create_lens(config: LensPayload):
    return helios.Lens(focal_length=config.focal_length * u.m)

def create_beam_splitter(config: BeamSplitterPayload):
    return helios.BeamSplitter(cutoff=config.split_ratio)

def create_coronagraph(config: CoronagraphPayload):
    # Map type string to implementation params if needed
    # Currently Coronagraph just takes type string e.g. '4quadrants'
    return helios.Coronagraph(phase_mask=config.type)

def create_fiber(config: FiberPayload, is_input: bool):
    if is_input:
        return fibers.FiberIn(modes=config.modes, name=config.name)
    else:
        return fibers.FiberOut(name=config.name)

def create_photonic(config: PhotonicPayload):
    if config.type == 'y_splitter':
        return photonics.YSplitter(name=config.name)
    elif config.type == 'tops':
        return photonics.TOPS(phase=config.phase if config.phase is not None else 0.0, name=config.name)
    elif config.type == 'mmi':
        # Simple preset handling
        if config.matrix_preset == 'hadamard':
            mat = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        elif config.matrix_preset == 'cross':
             mat = np.array([
                [np.exp(1j*np.pi/4), np.exp(-1j*np.pi/4)],
                [np.exp(-1j*np.pi/4), np.exp(1j*np.pi/4)]
            ]) / np.sqrt(2)
        else:
             mat = np.eye(2) # Fallback identity
        return photonics.MMI(matrix=mat, name=config.name)
    elif config.type == 'swap':
        mapping = config.mapping if config.mapping else [0, 1]
        return photonics.Swap(mapping=mapping, name=config.name)
    return None

def create_camera(config: CameraPayload, pipeline):
    # Camera needs pixel scale. We can try to infer from pipeline or defaults.
    # The previous logic calculated FOV from scene planets.
    # But now Scene and Camera are decoupled in the list.
    # We need to peek at the Scene layer if possible, or use a default FOV.
    
    # We can try to find the Scene layer in the pipeline history?
    # Pipeline layers are stored.
    
    # Heuristic: FOV = 2 arcsec default.
    fov = 2.0
    
    # Try to find planets in existing scene layers
    for layer in pipeline.layers:
        if isinstance(layer, helios.Scene):
            # Inspect elements
            # Accessing private or internal lists might be needed depending on Scene implementation
            # helios.Scene inherits Layer? It manages Elements.
            # Assuming we can't easily introspect without robust API.
            pass
            
    return helios.Camera(pixels=(256, 256), pixel_scale=(fov/256)*u.arcsec)


def get_config_dict(config_obj):
    if isinstance(config_obj, dict):
        return config_obj
    if hasattr(config_obj, 'model_dump'):
        return config_obj.model_dump()
    return config_obj

# --- Converters (Context -> Payload) ---

def scene_to_payload(scene: helios.Scene) -> ScenePayload:
    stars_data = []
    planets_data = []
    zodiacal_data = ZodiacalData(enabled=False)
    
    for elem in scene.elements:
        if isinstance(elem, helios.Star):
            x_as = 0.0
            y_as = 0.0
            if hasattr(elem, 'position'):
                try:
                    pos = elem.position
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        x_as = u.Quantity(pos[0], u.arcsec).to(u.arcsec).value
                        y_as = u.Quantity(pos[1], u.arcsec).to(u.arcsec).value
                except: pass
            
            s = StarData(
                temperature=elem.temperature.to(u.K).value if hasattr(elem.temperature, 'to') else float(elem.temperature),
                magnitude=float(elem.magnitude),
                x_arcsec=float(x_as),
                y_arcsec=float(y_as)
            )
            stars_data.append(s)
            
        elif isinstance(elem, helios.Planet):
            dist_pc = scene.distance.to(u.pc).value if hasattr(scene, 'distance') and scene.distance is not None else 10.0
            
            x_as = 0.0
            y_as = 0.0
            sep_au = 1.0
            if hasattr(elem, 'position'):
                 try:
                    pos = elem.position
                    x_len = u.Quantity(pos[0], u.m)
                    y_len = u.Quantity(pos[1], u.m)
                    
                    x_as = (x_len / (dist_pc * u.pc)).to(u.dimensionless_unscaled).value * 206265
                    y_as = (y_len / (dist_pc * u.pc)).to(u.dimensionless_unscaled).value * 206265
                    
                    sep_au = np.hypot(x_len.to(u.au).value, y_len.to(u.au).value)
                 except: pass

            p = PlanetData(
                mass=elem.mass.to(u.M_jup).value if hasattr(elem.mass, 'to') else float(elem.mass),
                radius=elem.radius.to(u.R_jup).value if hasattr(elem, 'radius') and elem.radius is not None else 1.0,
                separation=float(sep_au),
                x_arcsec=float(x_as),
                y_arcsec=float(y_as),
                angle=0.0
            )
            planets_data.append(p)
            
        elif isinstance(elem, helios.Zodiacal):
            zodiacal_data = ZodiacalData(
                enabled=True,
                brightness=float(elem.brightness),
                radius=None
            )
            
    return ScenePayload(stars=stars_data, planets=planets_data, zodiacal=zodiacal_data)

def atmosphere_to_payload(atm: helios.Atmosphere) -> AtmospherePayload:
    speed = np.linalg.norm(atm.wind_velocity)
    return AtmospherePayload(
        enabled=True,
        rms_nm=float(u.Quantity(atm.rms, u.m).to(u.nm).value),
        wind_speed=float(u.Quantity(speed, u.m/u.s).to(u.m/u.s).value)
    )

def telescope_to_payload(tel: helios.TelescopeArray) -> TelescopePayload:
    collectors = []
    max_diam = 8.0
    for i, col in enumerate(tel.collectors):
        x = col.position[0]
        y = col.position[1]
        
        diam = 8.0
        if col.size is not None:
             diam = u.Quantity(col.size, u.m).to(u.m).value
        max_diam = max(max_diam, diam)
        
        p_type = "Circular"
        if hasattr(col.pupil, 'elements') and len(col.pupil.elements) > 2:
                  pass
        
        collectors.append(CollectorData(
            id=f"c{i}", x=float(x), y=float(y), diameter=float(diam),
            pupil_type=p_type
        ))
        
    return TelescopePayload(
        preset="Custom",
        diameter=float(max_diam),
        collectors=collectors
    )

def camera_to_payload(cam: helios.Camera) -> CameraPayload:
    exp = 0.1
    if hasattr(cam, 'integration_time'):
        exp = u.Quantity(cam.integration_time, u.s).to(u.s).value
    return CameraPayload(exposure=float(exp), wavelength=1.0)

# --- Converters (Context -> Payload) ---

def scene_to_payload(scene: helios.Scene) -> ScenePayload:
    stars_data = []
    planets_data = []
    zodiacal_data = ZodiacalData(enabled=False)
    
    for elem in scene.elements:
        if isinstance(elem, helios.Star):
            # Convert Star
            # Pos is (ra, dec) or (x, y). Assumed x,y in arcsec for this simple UI
            x_as = 0.0
            y_as = 0.0
            if hasattr(elem, 'position'):
                try:
                    pos = elem.position
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        x_as = u.Quantity(pos[0], u.arcsec).to(u.arcsec).value
                        y_as = u.Quantity(pos[1], u.arcsec).to(u.arcsec).value
                except: pass
            
            s = StarData(
                temperature=elem.temperature.to(u.K).value if hasattr(elem.temperature, 'to') else float(elem.temperature),
                magnitude=float(elem.magnitude),
                x_arcsec=float(x_as),
                y_arcsec=float(y_as)
            )
            stars_data.append(s)
            
        elif isinstance(elem, helios.Planet):
            # Convert Planet
            dist_pc = scene.distance.to(u.pc).value if hasattr(scene, 'distance') and scene.distance is not None else 10.0
            
            x_as = 0.0
            y_as = 0.0
            sep_au = 1.0
            if hasattr(elem, 'position'):
                 try:
                    pos = elem.position
                    # If pos is in length (m/au), convert to arcsec via distance
                    # separation ~ sqrt(x^2 + y^2)
                    # angle ~ atan2(y, x)
                    # But UI supports x/y arcsec directly
                    
                    # Assume stored as length (e.g. AU) in simulation
                    # Convert to arcsec: theta = r / d
                    x_len = u.Quantity(pos[0], u.m)
                    y_len = u.Quantity(pos[1], u.m)
                    
                    x_as = (x_len / (dist_pc * u.pc)).to(u.dimensionless_unscaled).value * 206265
                    y_as = (y_len / (dist_pc * u.pc)).to(u.dimensionless_unscaled).value * 206265
                    
                    sep_au = np.hypot(x_len.to(u.au).value, y_len.to(u.au).value)
                 except: pass

            p = PlanetData(
                mass=elem.mass.to(u.M_jup).value if hasattr(elem.mass, 'to') else float(elem.mass),
                radius=elem.radius.to(u.R_jup).value if hasattr(elem, 'radius') and elem.radius is not None else 1.0,
                separation=float(sep_au),
                x_arcsec=float(x_as),
                y_arcsec=float(y_as),
                angle=0.0 # derived from x/y if needed, but x/y is sufficient
            )
            planets_data.append(p)
            
        elif isinstance(elem, helios.Zodiacal):
            zodiacal_data = ZodiacalData(
                enabled=True,
                brightness=float(elem.brightness),
                radius=None # TODO if needed
            )
            
    return ScenePayload(stars=stars_data, planets=planets_data, zodiacal=zodiacal_data)

def atmosphere_to_payload(atm: helios.Atmosphere) -> AtmospherePayload:
    # Estimate wind speed magnitude
    speed = np.linalg.norm(atm.wind_velocity)
    return AtmospherePayload(
        enabled=True,
        rms_nm=float(u.Quantity(atm.rms, u.m).to(u.nm).value),
        wind_speed=float(u.Quantity(speed, u.m/u.s).to(u.m/u.s).value)
    )

def telescope_to_payload(tel: helios.TelescopeArray) -> TelescopePayload:
    # Check if generic preset
    # For now, always return Custom to be safe
    collectors = []
    max_diam = 8.0
    for i, col in enumerate(tel.collectors):
        x = col.position[0]
        y = col.position[1]
        
        diam = 8.0
        if col.size is not None:
             diam = u.Quantity(col.size, u.m).to(u.m).value
        max_diam = max(max_diam, diam)
        
        # Infer pupil type
        p_type = "Circular"
        if hasattr(col.pupil, 'elements'):
             # Very rough heuristic
             if len(col.pupil.elements) > 2: #Likely complex
                  pass
        
        collectors.append(CollectorData(
            id=f"c{i}", x=float(x), y=float(y), diameter=float(diam),
            pupil_type=p_type
        ))
        
    return TelescopePayload(
        preset="Custom",
        diameter=float(max_diam),
        collectors=collectors
    )

def camera_to_payload(cam: helios.Camera) -> CameraPayload:
    # exposure
    exp = 0.1
    if hasattr(cam, 'integration_time'):
        exp = u.Quantity(cam.integration_time, u.s).to(u.s).value
    return CameraPayload(exposure=float(exp), wavelength=1.0) # wavelength dummy


# --- Converters (Context -> Payload) ---

def scene_to_payload(scene: helios.Scene) -> ScenePayload:
    stars_data = []
    planets_data = []
    zodiacal_data = ZodiacalData(enabled=False)
    
    for elem in scene.elements:
        if isinstance(elem, helios.Star):
            x_as = 0.0
            y_as = 0.0
            if hasattr(elem, 'position'):
                try:
                    pos = elem.position
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        x_as = u.Quantity(pos[0], u.arcsec).to(u.arcsec).value
                        y_as = u.Quantity(pos[1], u.arcsec).to(u.arcsec).value
                except: pass
            
            s = StarData(
                temperature=elem.temperature.to(u.K).value if hasattr(elem.temperature, 'to') else float(elem.temperature),
                magnitude=float(elem.magnitude),
                x_arcsec=float(x_as),
                y_arcsec=float(y_as)
            )
            stars_data.append(s)
            
        elif isinstance(elem, helios.Planet):
            dist_pc = scene.distance.to(u.pc).value if hasattr(scene, 'distance') and scene.distance is not None else 10.0
            
            x_as = 0.0
            y_as = 0.0
            sep_au = 1.0
            if hasattr(elem, 'position'):
                 try:
                    pos = elem.position
                    x_len = u.Quantity(pos[0], u.m)
                    y_len = u.Quantity(pos[1], u.m)
                    
                    x_as = (x_len / (dist_pc * u.pc)).to(u.dimensionless_unscaled).value * 206265
                    y_as = (y_len / (dist_pc * u.pc)).to(u.dimensionless_unscaled).value * 206265
                    
                    sep_au = np.hypot(x_len.to(u.au).value, y_len.to(u.au).value)
                 except: pass

            p = PlanetData(
                mass=elem.mass.to(u.M_jup).value if hasattr(elem.mass, 'to') else float(elem.mass),
                radius=elem.radius.to(u.R_jup).value if hasattr(elem, 'radius') and elem.radius is not None else 1.0,
                separation=float(sep_au),
                x_arcsec=float(x_as),
                y_arcsec=float(y_as),
                angle=0.0
            )
            planets_data.append(p)
            
        elif isinstance(elem, helios.Zodiacal):
            zodiacal_data = ZodiacalData(
                enabled=True,
                brightness=float(elem.brightness),
                radius=None
            )
            
    return ScenePayload(stars=stars_data, planets=planets_data, zodiacal=zodiacal_data)

def atmosphere_to_payload(atm: helios.Atmosphere) -> AtmospherePayload:
    speed = np.linalg.norm(atm.wind_velocity)
    return AtmospherePayload(
        enabled=True,
        rms_nm=float(u.Quantity(atm.rms, u.m).to(u.nm).value),
        wind_speed=float(u.Quantity(speed, u.m/u.s).to(u.m/u.s).value)
    )

def telescope_to_payload(tel: helios.TelescopeArray) -> TelescopePayload:
    collectors = []
    max_diam = 8.0
    for i, col in enumerate(tel.collectors):
        x = col.position[0]
        y = col.position[1]
        
        diam = 8.0
        if col.size is not None:
             diam = u.Quantity(col.size, u.m).to(u.m).value
        max_diam = max(max_diam, diam)
        
        p_type = "Circular"
        if hasattr(col.pupil, 'elements') and len(col.pupil.elements) > 2:
                  pass
        
        collectors.append(CollectorData(
            id=f"c{i}", x=float(x), y=float(y), diameter=float(diam),
            pupil_type=p_type
        ))
        
    return TelescopePayload(
        preset="Custom",
        diameter=float(max_diam),
        collectors=collectors
    )

def camera_to_payload(cam: helios.Camera) -> CameraPayload:
    exp = 0.1
    if hasattr(cam, 'integration_time'):
        exp = u.Quantity(cam.integration_time, u.s).to(u.s).value
    return CameraPayload(exposure=float(exp), wavelength=1.0)

# --- Endpoint ---

@app.post("/api/pipeline/export_file")
def export_pipeline_file(request: PipelineRequest):
    """Export current pipeline configuration as a library-compatible JSON context file."""
    try:
        print(f"DEBUG EXPORT: Layers Count = {len(request.layers)}")
        for i, item in enumerate(request.layers):
            print(f"  Item {i} Type: {type(item)}")
            if isinstance(item, list):
                print(f"    List length: {len(item)}")
                for j, sub in enumerate(item):
                     print(f"      Sub {j} Type: {type(sub)}")
            else:
                print(f"    Val: {item}")

        # 1. Build Pipeline from request
        pipeline = helios.Pipeline()
        for item in request.layers:
            if isinstance(item, list):
                # Parallel branch
                parallel_layers = []
                for sub_conf in item:
                    layer_obj = None
                    # ... Copy creation logic or refactor to helper ...
                    # Replicating logic for robust fix without massive refactor:
                    if sub_conf.type == 'scene': layer_obj = create_scene(ScenePayload(**get_config_dict(sub_conf.config)))
                    elif sub_conf.type == 'atmosphere': layer_obj = create_atmosphere(AtmospherePayload(**get_config_dict(sub_conf.config)))
                    elif sub_conf.type == 'telescope': layer_obj = create_telescope(TelescopePayload(**get_config_dict(sub_conf.config)))
                    elif sub_conf.type == 'camera': layer_obj = create_camera(CameraPayload(**get_config_dict(sub_conf.config)), pipeline)
                    elif sub_conf.type == 'lens': layer_obj = create_lens(LensPayload(**get_config_dict(sub_conf.config)))
                    elif sub_conf.type == 'beam_splitter': layer_obj = create_beam_splitter(BeamSplitterPayload(**get_config_dict(sub_conf.config)))
                    elif sub_conf.type == 'coronagraph': layer_obj = create_coronagraph(CoronagraphPayload(**get_config_dict(sub_conf.config)))
                    elif sub_conf.type == 'fiber_in': layer_obj = create_fiber(FiberPayload(**get_config_dict(sub_conf.config)), is_input=True)
                    elif sub_conf.type == 'fiber_out': layer_obj = create_fiber(FiberPayload(**get_config_dict(sub_conf.config)), is_input=False)
                    elif sub_conf.type == 'photonic': layer_obj = create_photonic(PhotonicPayload(**get_config_dict(sub_conf.config)))
                    
                    if layer_obj:
                        layer_obj.metadata = sub_conf.metadata
                        parallel_layers.append(layer_obj)
                pipeline.add_layer(parallel_layers)
            else:
                # Single layer
                layer_conf = item
                layer_obj = None
                if layer_conf.type == 'scene':
                     data = ScenePayload(**get_config_dict(layer_conf.config))
                     layer_obj = create_scene(data)
                elif layer_conf.type == 'atmosphere':
                     data = AtmospherePayload(**get_config_dict(layer_conf.config))
                     layer_obj = create_atmosphere(data)
                elif layer_conf.type == 'telescope':
                     data = TelescopePayload(**get_config_dict(layer_conf.config))
                     layer_obj = create_telescope(data)
                elif layer_conf.type == 'camera':
                     data = CameraPayload(**get_config_dict(layer_conf.config))
                     layer_obj = create_camera(data, pipeline)
                elif layer_conf.type == 'lens':
                     data = LensPayload(**get_config_dict(layer_conf.config))
                     layer_obj = create_lens(data)
                elif layer_conf.type == 'beam_splitter':
                     data = BeamSplitterPayload(**get_config_dict(layer_conf.config))
                     layer_obj = create_beam_splitter(data)
                elif layer_conf.type == 'coronagraph':
                     data = CoronagraphPayload(**get_config_dict(layer_conf.config))
                     layer_obj = create_coronagraph(data)
                elif layer_conf.type == 'fiber_in':
                     data = FiberPayload(**get_config_dict(layer_conf.config))
                     layer_obj = create_fiber(data, is_input=True)
                elif layer_conf.type == 'fiber_out':
                     data = FiberPayload(**get_config_dict(layer_conf.config))
                     layer_obj = create_fiber(data, is_input=False)
                elif layer_conf.type == 'photonic':
                     data = PhotonicPayload(**get_config_dict(layer_conf.config))
                     layer_obj = create_photonic(data)
                
                if layer_obj:
                    layer_obj.metadata = layer_conf.metadata
                    context.add_layer(layer_obj)
        
        # 2. Serialize
        data_dict = context.to_dict()
        
        # 3. Return as file
        import json
        json_str = json.dumps(data_dict, indent=2)
        return Response(
            content=json_str, 
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="helios_context.json"'}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def _help_convert(layer):
    l_type = None
    l_config = None
    if isinstance(layer, helios.Scene):
         l_type = 'scene'
         l_config = scene_to_payload(layer)
    elif isinstance(layer, helios.Atmosphere):
         l_type = 'atmosphere'
         l_config = atmosphere_to_payload(layer)
    elif isinstance(layer, helios.TelescopeArray):
         l_type = 'telescope'
         l_config = telescope_to_payload(layer)
    elif isinstance(layer, helios.Camera):
         l_type = 'camera'
         l_config = camera_to_payload(layer)
    # Add other types checks if needed (lens, etc) - assuming conversion functions exist or fallback
    # For now, only core types have converters defined in this file. 
    # Generics? We need generic converters or update component converters.
    # Current codebase only has scene/atm/tel/cam converters.
    # We should add Generic handling or assume custom components use a Generic payload if possible?
    # Actually, the user code earlier defined generic payloads but not converters FROM object TO payload.
    # We need to implement those or fallback to dict.
    
    # Quick fix: if no converter, try to use object's to_dict and map to Generic Dict
    if not l_type:
        # Infer type from class name
        # Mapping reverse?
        name = layer.__class__.__name__
        if 'Lens' in name: 
            l_type = 'lens'
            l_config = LensPayload(focal_length=layer.focal_length.to(u.m).value if hasattr(layer,'focal_length') else 1.0)
        elif 'BeamSplitter' in name:
            l_type = 'beam_splitter'
            l_config = BeamSplitterPayload(split_ratio=layer.cutoff if hasattr(layer, 'cutoff') else 0.5)
        elif 'Coronagraph' in name:
            l_type = 'coronagraph'
            l_config = CoronagraphPayload(type=layer.phase_mask if hasattr(layer, 'phase_mask') else '4quadrants')
        elif 'FiberIn' in name:
            l_type = 'fiber_in'
            l_config = FiberPayload(modes=layer.modes if hasattr(layer, 'modes') else 1, name=layer.name)
        elif 'FiberOut' in name:
            l_type = 'fiber_out'
            l_config = FiberPayload(name=layer.name)
        elif 'YSplitter' in name:
             l_type = 'photonic'
             l_config = PhotonicPayload(type='y_splitter', name=layer.name)
        else:
             # Fallback
             pass

    return l_type, l_config

@app.post("/api/context/import_file")
def import_pipeline_file(file_data: Dict[str, Any]):
    """Import a library JSON context file and convert it to pipeline configuration."""
    try:
        # file_data is the JSON dict parsed by FastAPI from body
        # 1. Load Pipeline
        pipeline = helios.Pipeline.from_dict(file_data)
        
        # 2. Convert to PipelineRequest layers
        layers_config = []
        
        for layer in pipeline.layers:
             if isinstance(layer, list):
                 # Parallel branch
                 sub_layers = []
                 for sub in layer:
                     l_type, l_config = _help_convert(sub)
                     if l_type:
                         sub_layers.append(LayerConfig(type=l_type, config=l_config, metadata=sub.metadata))
                 if sub_layers:
                     layers_config.append(sub_layers)
             else:
                 l_type, l_config = _help_convert(layer)
                 if l_type and l_config:
                     layers_config.append(LayerConfig(type=l_type, config=l_config, metadata=layer.metadata))
                 
        return PipelineRequest(mode='pipeline', layers=layers_config)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulate")
def run_pipeline(request: PipelineRequest):
    try:
        pipeline = helios.Pipeline()
        
        for layer_conf in request.layers:
            layer_obj = None
            
            if layer_conf.type == 'scene':
                if isinstance(layer_conf.config, ScenePayload):
                    data = layer_conf.config
                elif isinstance(layer_conf.config, dict):
                    data = ScenePayload(**layer_conf.config)
                else:
                    data = layer_conf.config
                layer_obj = create_scene(data)
                
            elif layer_conf.type == 'atmosphere':
                if isinstance(layer_conf.config, AtmospherePayload):
                    data = layer_conf.config
                elif isinstance(layer_conf.config, dict):
                    data = AtmospherePayload(**layer_conf.config)
                else:
                    data = layer_conf.config
                layer_obj = create_atmosphere(data)
                
            elif layer_conf.type == 'telescope':
                if isinstance(layer_conf.config, TelescopePayload):
                    data = layer_conf.config
                elif isinstance(layer_conf.config, dict):
                    data = TelescopePayload(**layer_conf.config)
                else:
                    data = layer_conf.config
                layer_obj = create_telescope(data)
                
            elif layer_conf.type == 'camera':
                if isinstance(layer_conf.config, CameraPayload):
                    data = layer_conf.config
                elif isinstance(layer_conf.config, dict):
                    data = CameraPayload(**layer_conf.config)
                else:
                    data = layer_conf.config
                layer_obj = create_camera(data, pipeline)
            
            elif layer_conf.type == 'lens':
                data = LensPayload(**get_config_dict(layer_conf.config))
                layer_obj = create_lens(data)
            elif layer_conf.type == 'beam_splitter':
                data = BeamSplitterPayload(**get_config_dict(layer_conf.config))
                layer_obj = create_beam_splitter(data)
            elif layer_conf.type == 'coronagraph':
                data = CoronagraphPayload(**get_config_dict(layer_conf.config))
                layer_obj = create_coronagraph(data)
            elif layer_conf.type == 'fiber_in':
                data = FiberPayload(**get_config_dict(layer_conf.config))
                layer_obj = create_fiber(data, is_input=True)
            elif layer_conf.type == 'fiber_out':
                data = FiberPayload(**get_config_dict(layer_conf.config))
                layer_obj = create_fiber(data, is_input=False)
            elif layer_conf.type == 'photonic':
                data = PhotonicPayload(**get_config_dict(layer_conf.config))
                layer_obj = create_photonic(data)
            
            if layer_obj:
                pipeline.add_layer(layer_obj)
            


        # Run Observation
        # Wavelength? Handled by camera or observe parameters.
        # previous code: pipeline.observe()
        
        result = pipeline.observe()
        
        image_data = result
        if hasattr(result, 'value'): 
            image_data = result.value
        
        if image_data.max() > 0:
            image_data = image_data / image_data.max()
            image_data = np.power(image_data, 0.5)
        
        plt.figure(figsize=(6, 6), dpi=100)
        plt.imshow(image_data, cmap='inferno', origin='lower')
        plt.axis('off')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        
        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate_code")
def generate_code(request: PipelineRequest):
    code = []
    code.append("import numpy as np")
    code.append("import matplotlib.pyplot as plt")
    code.append("from astropy import units as u")
    code.append("import helios")
    code.append("from helios.components import *")
    code.append("import helios.components.photonics as photonics")
    code.append("import helios.components.fibers as fibers")
    code.append("")
    code.append("# Initialize Pipeline")
    code.append("pipeline = helios.Pipeline()")
    code.append("")

    def format_layer(layer_conf, var_name="layer"):
        lines = []
        
        # Helper to get config dict/obj
        if isinstance(layer_conf.config, dict):
            conf = layer_conf.config
        else:
            conf = layer_conf.config.dict() if hasattr(layer_conf.config, 'dict') else layer_conf.config.__dict__

        l_type = layer_conf.type
        
        if l_type == 'scene':
            lines.append(f"# Scene Layer")
            lines.append(f"{var_name} = helios.Scene(distance=10*u.pc)")
            
            stars = conf.get('stars', [])
            for i, s in enumerate(stars):
                lines.append(f"star_{i} = helios.Star(temperature={s.get('temperature', 5778)}*u.K, magnitude={s.get('magnitude', 4.83)})")
                lines.append(f"star_{i}.position = ({s.get('x_arcsec', 0)}*u.arcsec, {s.get('y_arcsec', 0)}*u.arcsec)")
                lines.append(f"{var_name}.add(star_{i})")
            
            planets = conf.get('planets', [])
            for i, p in enumerate(planets):
                lines.append(f"planet_{i} = helios.Planet(mass={p.get('mass', 1.0)}*u.M_jup, orbit_radius={p.get('separation', 1.0)}*u.AU)")
                if p.get('x_arcsec') is not None:
                    lines.append(f"planet_{i}.position = ({p.get('x_arcsec')}*u.arcsec, {p.get('y_arcsec')}*u.arcsec)")
                else:
                    angle = np.radians(p.get('angle', 0))
                    dist_pc = 10
                    sep_arcsec = p.get('separation', 1.0) / dist_pc
                    x = sep_arcsec * np.cos(angle)
                    y = sep_arcsec * np.sin(angle)
                    lines.append(f"planet_{i}.position = ({x:.4f}*u.arcsec, {y:.4f}*u.arcsec)")
                lines.append(f"{var_name}.add(planet_{i})")

            zodi = conf.get('zodiacal', {})
            if zodi.get('enabled'):
                lines.append(f"zodi = helios.Zodiacal(brightness={zodi.get('brightness', 1.0)})")
                lines.append(f"{var_name}.add(zodi)")

        elif l_type == 'atmosphere':
            if not conf.get('enabled', True):
                lines.append(f"# Atmosphere (Disabled)")
                return []
            lines.append(f"# Atmosphere Layer")
            lines.append(f"{var_name} = helios.Atmosphere(rms={conf.get('rms_nm', 100)}*u.nm, wind_speed={conf.get('wind_speed', 5.0)}*u.m/u.s)")

        elif l_type == 'telescope':
            lines.append(f"# Telescope Layer")
            preset = conf.get('preset', 'Single')
            if preset == 'VLTI-UT':
                lines.append(f"{var_name} = helios.TelescopeArray.vlti(uts=True)")
            elif preset == 'VLTI-AT':
                lines.append(f"{var_name} = helios.TelescopeArray.vlti(uts=False)")
            elif preset == 'LIFE':
                lines.append(f"{var_name} = helios.TelescopeArray.life()")
            else:
                lines.append(f"{var_name} = helios.TelescopeArray(name='Custom Array')")
                collectors = conf.get('collectors', [])
                for i, col in enumerate(collectors):
                    diam = col.get('diameter', 8.0)
                    ptype = col.get('pupil_type', 'Circular')
                    
                    lines.append(f"# Collector {i+1}")
                    if ptype == 'VLT':
                        lines.append(f"pupil_{i} = helios.Pupil.vlt()")
                    elif ptype == 'JWST':
                        lines.append(f"pupil_{i} = helios.Pupil.jwst()")
                    elif ptype == 'Obstructed':
                        lines.append(f"pupil_{i} = helios.Pupil(diameter={diam}*u.m)")
                        lines.append(f"pupil_{i}.add_disk(radius={diam/2}*u.m)")
                        obs = col.get('central_obstruction', 0)
                        lines.append(f"pupil_{i}.add_central_obscuration(diameter={diam*obs}*u.m)")
                        spiders = col.get('spiders', 0)
                        if spiders > 0:
                            lines.append(f"pupil_{i}.add_spiders(arms={spiders}, width={0.02*diam}*u.m)")
                    else:
                        lines.append(f"pupil_{i} = helios.Pupil(diameter={diam}*u.m)")
                        lines.append(f"pupil_{i}.add_disk(radius={diam/2}*u.m)")
                    
                    lines.append(f"{var_name}.add_collector(pupil=pupil_{i}, position=({col.get('x', 0)}*u.m, {col.get('y', 0)}*u.m), size={diam}*u.m)")

        elif l_type == 'camera':
            lines.append(f"# Camera Layer")
            lines.append(f"{var_name} = helios.Camera(pixels=(256, 256), pixel_scale=(2.0/256)*u.arcsec)")
            lines.append(f"{var_name}.exposure = {conf.get('exposure', 0.1)}*u.s")

        elif l_type == 'lens':
            lines.append(f"{var_name} = helios.Lens(focal_length={conf.get('focal_length', 1.0)}*u.m)")
        
        elif l_type == 'beam_splitter':
            lines.append(f"{var_name} = helios.BeamSplitter(cutoff={conf.get('split_ratio', 0.5)})")
            
        elif l_type == 'coronagraph':
            lines.append(f"{var_name} = helios.Coronagraph(phase_mask='{conf.get('type', '4quadrants')}')")
            
        elif l_type == 'fiber_in':
            lines.append(f"{var_name} = fibers.FiberIn(modes={conf.get('modes', 1)}, name='{conf.get('name', 'FiberIn')}')")
            
        elif l_type == 'fiber_out':
            lines.append(f"{var_name} = fibers.FiberOut(name='{conf.get('name', 'FiberOut')}')")
            
        elif l_type == 'photonic':
            ptype = conf.get('type')
            name = conf.get('name', 'Photonic')
            if ptype == 'y_splitter':
                lines.append(f"{var_name} = photonics.YSplitter(name='{name}')")
            elif ptype == 'tops':
                lines.append(f"{var_name} = photonics.TOPS(phase={conf.get('phase', 0.0)}, name='{name}')")
            elif ptype == 'mmi':
                preset = conf.get('matrix_preset', 'default')
                if preset == 'hadamard':
                    lines.append(f"mat = np.array([[1, 1], [1, -1]]) / np.sqrt(2)")
                elif preset == 'cross':
                    lines.append(f"mat = np.array([[np.exp(1j*np.pi/4), np.exp(-1j*np.pi/4)], [np.exp(-1j*np.pi/4), np.exp(1j*np.pi/4)]]) / np.sqrt(2)")
                else:
                    lines.append(f"mat = np.eye(2)")
                lines.append(f"{var_name} = photonics.MMI(matrix=mat, name='{name}')")
            elif ptype == 'swap':
                mapping = conf.get('mapping', [0, 1])
                lines.append(f"{var_name} = photonics.Swap(mapping={mapping}, name='{name}')")

        return lines

    for i, layer in enumerate(request.layers):
        if isinstance(layer, list):
            code.append(f"# Parallel Block {i}")
            branch_vars = []
            for j, sub_layer in enumerate(layer):
                var_name = f"layer_{i}_{j}"
                lines = format_layer(sub_layer, var_name)
                if lines:
                    code.extend(lines)
                    branch_vars.append(var_name)
            code.append(f"pipeline.add_layer([{', '.join(branch_vars)}])")
        else:
            var_name = f"layer_{i}"
            lines = format_layer(layer, var_name)
            if lines:
                code.extend(lines)
                code.append(f"pipeline.add_layer({var_name})")
        code.append("")

    code.append("# Run Observation")
    code.append("result = pipeline.observe()")
    code.append("")
    code.append("# Plot Result")
    code.append("if hasattr(result, 'value'):")
    code.append("    data = result.value")
    code.append("else:")
    code.append("    data = result")
    code.append("")
    code.append("if data.max() > 0:")
    code.append("    data = data / data.max()")
    code.append("    data = np.power(data, 0.5)")
    code.append("")
    code.append("plt.figure(figsize=(10, 10))")
    code.append("plt.imshow(data, cmap='inferno', origin='lower')")
    code.append("plt.colorbar()")
    code.append("plt.show()")

    return {"code": "\n".join(code)}


@app.post("/api/preview_layer")
def preview_layer(layer_conf: LayerConfig):
    try:
        buf = io.BytesIO()
        filename = f"{layer_conf.type}_preview.png"
        
        # Determine figsize
        # Default to 6x6
        figsize = (6, 6)
        
        # Check config for figsize override
        config_dict = None
        if isinstance(layer_conf.config, dict):
            config_dict = layer_conf.config
        elif hasattr(layer_conf.config, 'dict'):
            config_dict = layer_conf.config.dict()
            
        if config_dict:
             sz = config_dict.get('figsize', None)
             if sz:
                 try:
                     if isinstance(sz, (list, tuple)):
                         figsize = tuple(map(float, sz))
                     else:
                         val = float(sz)
                         figsize = (val, val)
                 except:
                     pass # Fallback to default

        if layer_conf.type == 'scene':
            if isinstance(layer_conf.config, dict):
                data = ScenePayload(**layer_conf.config)
            else:
                data = layer_conf.config

            view_mode = data.view_mode
            scene = create_scene(data)
            
            # Create figure with determined figsize
            fig, ax = plt.subplots(figsize=figsize)
            
            if view_mode == 'sed':
                scene.plot_sed(ax=ax)
                filename = "scene_sed.png"
            else:
                # scene.plot currently creates its own figure if we don't handle it.
                # We need to update scene.py to accept ax. 
                # For now, let's assume we'll update Scene.plot to take ax.
                if hasattr(scene, 'plot') and 'ax' in scene.plot.__code__.co_varnames:
                     scene.plot(ax=ax)
                else: 
                     # Fallback if I haven't updated scene.py yet (I will in next step)
                     plt.close(fig) 
                     fig, ax = scene.plot() # Uses default inside scene.py
                     fig.set_size_inches(figsize) # Force resize
                     
                filename = "scene_geometry.png"
                
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            
        elif layer_conf.type == 'atmosphere':
            # ... (atmosphere logic unchanged)
            fig = plt.figure(figsize=figsize)
            plt.text(0.5, 0.5, "Atmosphere Preview\n(Phase Screen - TODO)", ha='center')
            plt.xlim(0, 1); plt.ylim(0, 1); plt.axis('off')
            plt.savefig(buf, format='png')
            plt.close()
            filename = "atmosphere_preview.png"
            
        elif layer_conf.type == 'telescope':
            if isinstance(layer_conf.config, dict):
                data = TelescopePayload(**layer_conf.config)
            else:
                data = layer_conf.config
            telescope = create_telescope(data)
            fig, ax = plt.subplots(figsize=figsize)
            if hasattr(telescope, 'plot_array'):
               telescope.plot_array(ax=ax) 
            else:
               ax.text(0.5, 0.5, "No plot method", ha='center')
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            filename = "telescope_preview.png"
            
        elif layer_conf.type == 'camera':
            # Visualize Camera (Processed / Raw / Dark Frame)
            try:
                # Extract view_mode FIRST (before creating CameraPayload which consumes the dict)
                view_mode = 'processed'
                if isinstance(layer_conf.config, dict):
                    view_mode = layer_conf.config.get('view_mode', 'processed')
                    print(f"[DEBUG] Camera config dict: {layer_conf.config}")
                    print(f"[DEBUG] Extracted view_mode: {view_mode}")
                elif hasattr(layer_conf.config, 'view_mode'):
                    view_mode = layer_conf.config.view_mode
                    print(f"[DEBUG] view_mode from object: {view_mode}")
                
                # Now create the CameraPayload (filter out view_mode and figsize which are not CameraPayload fields)
                if isinstance(layer_conf.config, dict):
                    # Create a copy without view_mode and figsize
                    camera_config = {k: v for k, v in layer_conf.config.items() if k not in ['view_mode', 'figsize']}
                    config = CameraPayload(**camera_config)
                else:
                    config = layer_conf.config
                    
                camera = helios.Camera(
                    pixels=(256, 256), 
                    integration_time=float(config.exposure) * u.s,
                    wavelength=float(config.wavelength) * u.um
                )
                
                print(f"[DEBUG] About to generate image with view_mode: {view_mode}")
                
                # Get the appropriate image based on view_mode
                if view_mode == 'processed':
                    image_data = camera.get_image(wavefront=None)
                    title = "Camera - Processed Image (Dark Subtracted)"
                    filename = "camera_processed.png"
                elif view_mode == 'raw':
                    image_data = camera.get_raw_image(wavefront=None)
                    title = "Camera - Raw Image (Signal + Dark + Noise)"
                    filename = "camera_raw.png"
                else:  # 'dark'
                    image_data = camera.get_dark()
                    title = "Camera - Dark Frame"
                    filename = "camera_dark.png"
                
                # Create figure and plot
                fig, ax = plt.subplots(figsize=figsize)
                im = ax.imshow(image_data, origin='lower', cmap='inferno')
                
                # Colorbar
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Counts (e-)')
                
                # Labels and Title
                ax.set_xlabel('Pixel X')
                ax.set_ylabel('Pixel Y')
                ax.set_title(title)
                
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close(fig)
                
            except Exception as e:
                print(f"Error previewing camera: {e}")
                import traceback
                traceback.print_exc()
                fig, ax = plt.subplots(figsize=(4, 1))
                ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                ax.axis('off')
                fig.savefig(buf, format='png')
                plt.close(fig)
                filename = "camera_error.png"

        buf.seek(0)
        return Response(
            content=buf.getvalue(), 
            media_type="image/png",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/presets/{preset_name}")
def get_preset(preset_name: str):
    """Get configuration for a standard telescope preset."""
    try:
        telescope = None
        if preset_name == "VLTI-UT":
            telescope = helios.TelescopeArray.vlti(uts=True)
        elif preset_name == "VLTI-AT":
            telescope = helios.TelescopeArray.vlti(uts=False)
        elif preset_name == "LIFE":
            telescope = helios.TelescopeArray.life()
        else:
            raise HTTPException(status_code=404, detail="Preset not found")
            
        collectors_data = []
        for col in telescope.collectors:
            # Extract data. Position is (x, y) in meters.
            x, y = col.position
            # Size
            if hasattr(col.size, 'to'):
                diameter = col.size.to(u.m).value
            else:
                diameter = float(col.size)
            
            # Pupil inference (simplified)
            # We need to map back to our simple frontend types if possible, or just default to Custom/Circular
            # VLT and LIFE have specific pupil classes.
            # We can try to guess based on name or diameter, or just send "Circular"/generic params.
            
            # Default
            pupil_type = "Circular"
            central_obstruction = 0.0
            spiders = 0
            
            # Heuristics
            if "UT" in str(col.name) or "VLT" in str(col.name):
                pupil_type = "VLT"
            elif "LIFE" in str(col.name):
                # LIFE pupil is obstructed
                pupil_type = "Obstructed"
                central_obstruction = 0.5 # Default life obs
                # Actually checking the pupil object would be better if we exposed attributes
                # But for now, hardcoded mapping is safer than introspection of complex objects
    
            collectors_data.append({
                "x": float(x),
                "y": float(y),
                "diameter": float(diameter),
                "pupil_type": pupil_type,
                "central_obstruction": central_obstruction,
                "spiders": spiders
            })
            
        return collectors_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def read_root():
    return {"message": "Helios Web API (Pipeline Mode) is running"}

import os
from fastapi.staticfiles import StaticFiles

# Serve static files if build directory exists (Production mode)
# Adjust path relative to this file: ../frontend/dist
# But inside Docker, we might copy it to /app/static
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    # Fallback for local dev if not built
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
@app.post("/api/inspect_node")
def inspect_node(request: PipelineRequest, target_index: int = 0):
    """
    Inspect the output wavefront of a specific layer in the pipeline.
    Reconstructs the context statelessly and uses the Pull Model to retrieve
    the specific layer's output.
    """
    try:
        # 1. Build Context (same logic as run_pipeline)
        context = helios.Context()
        
        # Flattened list of layers for indexing logic match with frontend
        # The frontend likely sends a linear list. If it sends nested lists (parallel),
        # we need to be careful about "target_index".
        # Assuming linear sequence for now or that target_index refers to the top-level list index.
        
        # We need to rebuild the context exactly as run_pipeline does to ensure connections work.
        flat_layers = [] 
        
        for layer_conf in request.layers:
            # Similar creation logic to run_pipeline
            # We can refactor `run_pipeline` creation logic into a helper `build_context`
            # to avoid code duplication, but for safety I will duplicate/adapt here.
            
            layer_obj = None
            if layer_conf.type == 'scene':
                 layer_obj = create_scene(ScenePayload(**get_config_dict(layer_conf.config)))
            elif layer_conf.type == 'atmosphere':
                 layer_obj = create_atmosphere(AtmospherePayload(**get_config_dict(layer_conf.config)))
            elif layer_conf.type == 'telescope':
                 layer_obj = create_telescope(TelescopePayload(**get_config_dict(layer_conf.config)))
            elif layer_conf.type == 'camera':
                 layer_obj = create_camera(CameraPayload(**get_config_dict(layer_conf.config)), context)
            elif layer_conf.type == 'lens':
                 layer_obj = create_lens(LensPayload(**get_config_dict(layer_conf.config)))
            elif layer_conf.type == 'beam_splitter':
                 layer_obj = create_beam_splitter(BeamSplitterPayload(**get_config_dict(layer_conf.config)))
            elif layer_conf.type == 'coronagraph':
                 layer_obj = create_coronagraph(CoronagraphPayload(**get_config_dict(layer_conf.config)))
            elif layer_conf.type == 'fiber_in':
                 layer_obj = create_fiber(FiberPayload(**get_config_dict(layer_conf.config)), is_input=True)
            elif layer_conf.type == 'fiber_out':
                 layer_obj = create_fiber(FiberPayload(**get_config_dict(layer_conf.config)), is_input=False)
            elif layer_conf.type == 'photonic':
                 layer_obj = create_photonic(PhotonicPayload(**get_config_dict(layer_conf.config)))
            
            if layer_obj:
                # Add metadata if present
                if layer_conf.metadata:
                    layer_obj.metadata = layer_conf.metadata
                context.add_layer(layer_obj)
                flat_layers.append(layer_obj)

        if not 0 <= target_index < len(flat_layers):
             raise HTTPException(status_code=400, detail=f"Target index {target_index} out of range (0-{len(flat_layers)-1})")

        target_layer = flat_layers[target_index]
        print(f"Inspecting Layer {target_index}: {target_layer.name} ({type(target_layer).__name__})")

        # 2. Get Output Wavefront (Triggers Pull)
        # We use the new Layer architecture method!
        if hasattr(target_layer, 'get_output_wavefront'):
             # This automatically pulls from upstream!
             output = target_layer.get_output_wavefront()
        else:
             # Fallback? Should not happen if all inherit from Layer
             output = None
        
        # 3. Visualize Output
        buf = io.BytesIO()
        
        if output is None:
             # Maybe it's a layer that produces no output or failed?
             fig, ax = plt.subplots(figsize=(4,1))
             ax.text(0.5, 0.5, "No Wavefront Data Available", ha='center')
             ax.axis('off')
             fig.savefig(buf, format='png')
             plt.close(fig)
        
        elif isinstance(output, helios.Wavefront) or isinstance(output, helios.WavefrontArray):
             # Plot Wavefront
             # Wavefront.plot() usually creates a figure.
             # We want to force it to a buffer.
             # Check if Wavefront has a `plot` method that accepts `show=False`.
             
             # If WavefrontArray, we might have multiple. Plot the first or intensity sum?
             # For inspection, let's plot intensity.
             
             try:
                 # Attempt to use built-in plot if available and robust
                 # Or manually plot intensity
                 if isinstance(output, helios.WavefrontArray):
                     # Flatten/Stack for visualization
                     # Just plot the first one for now or grid?
                     # A simple grid of intensities is nice.
                     n = len(output)
                     if n == 1:
                         data = np.abs(output[0].value)**2
                     else:
                         # Sum of intensities (incoherent sum approximation for viewing)
                         # OR plot grid.
                         # Let's do max projection or sum.
                         data = np.sum([np.abs(w.value)**2 for w in output], axis=0) # Sum
                 else:
                     data = np.abs(output.value)**2
                     # Handle Multi-spectral/source (nsource, npix, npix)
                     if data.ndim == 3:
                         data = np.sum(data, axis=0)
                 
                 # Normalize
                 if data.max() > 0:
                     data = data / data.max()
                     data = np.power(data, 0.5) # Gamma correction
                 
                 plt.figure(figsize=(6, 6), dpi=100)
                 plt.imshow(data, cmap='inferno', origin='lower')
                 plt.colorbar(label='Normalized Intensity')
                 plt.title(f"Output of {target_layer.name}")
                 plt.tight_layout()
                 plt.savefig(buf, format='png')
                 plt.close()

             except Exception as plot_err:
                 print(f"Plotting error: {plot_err}")
                 import traceback
                 traceback.print_exc()
                 fig, ax = plt.subplots()
                 ax.text(0.5, 0.5, f"Plot Error: {plot_err}")
                 fig.savefig(buf, format='png')
                 plt.close(fig)

        elif isinstance(output, np.ndarray):
             # Data Layer output (e.g. Camera image)
             plt.figure(figsize=(6, 6))
             
             disp = output
             if disp.max() > 0:
                 disp = disp / disp.max()
                 disp = np.power(disp, 0.5)

             plt.imshow(disp, cmap='gray', origin='lower')
             plt.title(f"Data Output: {target_layer.name}")
             plt.axis('off')
             plt.savefig(buf, format='png')
             plt.close()
             
        else:
             # Unknown type
             fig, ax = plt.subplots()
             ax.text(0.5, 0.5, f"Unknown Output Type: {type(output)}")
             fig.savefig(buf, format='png')
             plt.close(fig)

        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
