import io
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Literal, Union, Dict, Any
from astropy import units as u

import helios
from helios.components import Zodiacal, Atmosphere, Pupil

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
    stars: List[StarData] = []
    planets: List[PlanetData] = []
    zodiacal: ZodiacalData = ZodiacalData()
    view_mode: str = 'geometry'

class AtmospherePayload(BaseModel):
    enabled: bool = True
    rms_nm: float = 100.0
    wind_speed: float = 5.0

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
    wavelength: float = 1.0
    exposure: float = 0.1

# Generic Layer Wrapper
class LayerConfig(BaseModel):
    type: Literal['scene', 'atmosphere', 'telescope', 'camera']
    config: Union[ScenePayload, AtmospherePayload, TelescopePayload, CameraPayload, Dict[str, Any]]

class PipelineRequest(BaseModel):
    mode: Literal['pipeline'] = 'pipeline'
    layers: List[LayerConfig]

# --- Helper Functions ---

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

def create_camera(config: CameraPayload, context):
    # Camera needs pixel scale. We can try to infer from context or defaults.
    # The previous logic calculated FOV from scene planets.
    # But now Scene and Camera are decoupled in the list.
    # We need to peek at the Scene layer if possible, or use a default FOV.
    
    # We can try to find the Scene layer in the context history?
    # Context layers are stored.
    
    # Heuristic: FOV = 2 arcsec default.
    fov = 2.0
    
    # Try to find planets in existing scene layers
    for layer in context.layers:
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

@app.post("/api/context/export_file")
def export_context_file(request: PipelineRequest):
    """Export current pipeline configuration as a library-compatible JSON context file."""
    try:
        # 1. Build Context from request
        context = helios.Context()
        for layer_conf in request.layers:
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
                 layer_obj = create_camera(data, context)
            
            if layer_obj:
                context.add_layer(layer_obj)
        
        # 2. Serialize
        data_dict = context.to_dict()
        
        # 3. Return as file
        import json
        json_str = json.dumps(data_dict, indent=2)
        return Response(
            content=json_str, 
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=helios_context.json"}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/context/import_file")
def import_context_file(file_data: Dict[str, Any]):
    """Import a library JSON context file and convert it to pipeline configuration."""
    try:
        # file_data is the JSON dict parsed by FastAPI from body
        # 1. Load Context
        context = helios.Context.from_dict(file_data)
        
        # 2. Convert to PipelineRequest layers
        layers_config = []
        
        for layer in context.layers:
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
             
             if l_type and l_config:
                 layers_config.append(LayerConfig(type=l_type, config=l_config))
                 
        return PipelineRequest(mode='pipeline', layers=layers_config)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulate")
def run_pipeline(request: PipelineRequest):
    try:
        context = helios.Context()
        
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
                layer_obj = create_camera(data, context)
            
            if layer_obj:
                context.add_layer(layer_obj)
            


        # Run Observation
        # Wavelength? Handled by camera or observe parameters.
        # previous code: context.observe()
        
        result = context.observe()
        
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
            # Visualize Camera (Dark Frame / Noise)
            try:
                if isinstance(layer_conf.config, dict):
                    config = CameraPayload(**layer_conf.config)
                else:
                    config = layer_conf.config
                    
                camera = helios.Camera(
                    pixels=(256, 256), 
                    integration_time=float(config.exposure) * u.s,
                    wavelength=float(config.wavelength) * u.um
                )
                
                # plot creates its own figure, let's try to control it if possible
                # Camera.plot returns ax
                # We can resize the figure controls
                ax = camera.plot(wavefront=None, show=False, title="Detector Dark Frame Preview")
                fig = ax.figure
                fig.set_size_inches(figsize)
                
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close(fig)
                filename = "camera_preview.png"
                
            except Exception as e:
                print(f"Error previewing camera: {e}")
                fig, ax = plt.subplots(figsize=(4, 1))
                ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                ax.axis('off')
                fig.savefig(buf, format='png')
                plt.close(fig)

        buf.seek(0)
        return Response(
            content=buf.getvalue(), 
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
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
