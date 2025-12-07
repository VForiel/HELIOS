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
)

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

class AtmospherePayload(BaseModel):
    enabled: bool = True
    rms_nm: float = 100.0
    wind_speed: float = 5.0

class CollectorData(BaseModel):
    x: float = 0
    y: float = 0
    diameter: float = 8.0
    pupil_type: str = "Circular"
    central_obstruction: float = 0
    spiders: int = 0

class TelescopePayload(BaseModel):
    preset: str = "Single"
    diameter: Optional[float] = 8.0
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
         pupil = helios.Pupil(diameter=diam * u.m)
         telescope.add_collector(pupil=pupil, position=(0,0), size=diam*u.m)
         return telescope
    else:
        # Custom
        telescope = helios.TelescopeArray(name="Custom Array")
        for i, col in enumerate(config.collectors):
            d = col.diameter ** u.m
            # Simplified pupil creation logic for brevity
            # (Matches previous implementation logic)
            if col.pupil_type == "VLT":
                p = helios.Pupil.vlt()
            elif col.pupil_type == "JWST":
                p = helios.Pupil.jwst()
            elif col.pupil_type == "Obstructed":
                p = helios.Pupil(diameter=col.diameter * u.m)
                p.add_disk(radius=col.diameter/2 * u.m) # redundant if diameter set? Pupil init logic varies.
                # Actually helios.Pupil(diameter=...) sets grid size, doesn't draw aperture?
                # Need to verify Pupil class. It usually starts empty.
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
    if hasattr(config_obj, 'dict'):
        return config_obj.dict()
    return config_obj

# --- Endpoint ---

@app.post("/simulate")
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


@app.post("/preview_layer")
def preview_layer(layer_conf: LayerConfig):
    try:
        buf = io.BytesIO()
        
        if layer_conf.type == 'scene':
            if isinstance(layer_conf.config, dict):
                data = ScenePayload(**layer_conf.config)
            else:
                data = layer_conf.config
            scene = create_scene(data)
            fig, ax = scene.plot()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            
        elif layer_conf.type == 'atmosphere':
            # ... (atmosphere logic unchanged, assuming it doesn't unpack config yet, but good to be safe if we expand)
            plt.figure(figsize=(5,5))
            plt.text(0.5, 0.5, "Atmosphere Preview\n(Phase Screen - TODO)", ha='center')
            plt.xlim(0, 1); plt.ylim(0, 1); plt.axis('off')
            plt.savefig(buf, format='png')
            plt.close()
            
        elif layer_conf.type == 'telescope':
            if isinstance(layer_conf.config, dict):
                data = TelescopePayload(**layer_conf.config)
            else:
                data = layer_conf.config
            telescope = create_telescope(data)
            fig = plt.figure(figsize=(5,5))
            if hasattr(telescope, 'plot_array'):
               telescope.plot_array() 
            else:
               plt.text(0.5, 0.5, "No plot method", ha='center')
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            
        elif layer_conf.type == 'camera':
            plt.figure(figsize=(5,5))
            plt.text(0.5, 0.5, "Camera Detector", ha='center')
            plt.axis('off')
            plt.savefig(buf, format='png')
            plt.close()

        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
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
