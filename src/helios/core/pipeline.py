import numpy as np
from astropy import units as u
from typing import List, Union, Optional, Any, Tuple
import copy
import importlib
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings

from helios.core.wavefront import Wavefront
from helios.core.optical_scene import OpticalScene, Spectrum
from helios.core.layer import Layer, GenerationLayer, SamplingLayer, OpticalLayer, DetectionLayer, DataLayer
from helios.core.component import Component
from helios.utils.serialization import serialize_value, deserialize_value

class Pipeline:
    """
    Main simulation pipeline managing layers and execution.
    
    The Pipeline orchestrates the simulation by sequentially processing
    layers. It maintains global simulation parameters and executes the observation
    workflow from scene generation through optical propagation to detector output.
    """
    def __init__(self, date: Any = None, declination: Any = None, layers: Optional[List[Union[Layer, List[Layer]]]] = None, **kwargs):
        self.date = date
        self.declination = declination
        self.kwargs = kwargs
        self.layers: List[Union[Layer, List[Layer]]] = []
        self.results = {}
        
        if layers:
            for layer in layers:
                self.add_layer(layer)

    def invalidate_downstream_cache(self, start_layer: Layer):
        """Invalidate cache for all layers downstream of the given layer."""
        start_idx = -1
        for i, l_item in enumerate(self.layers):
            if isinstance(l_item, list):
                if start_layer in l_item:
                    start_idx = i
                    break
            else:
                if l_item is start_layer:
                    start_idx = i
                    break
        
        if start_idx == -1:
            return 
            
        for i in range(start_idx + 1, len(self.layers)):
            l_item = self.layers[i]
            if isinstance(l_item, list):
                for sub_l in l_item:
                    if sub_l:
                        sub_l._cached_input = None
                        sub_l._cached_output = None
            else:
                l_item._cached_input = None
                l_item._cached_output = None

    def get_previous_layer_output(self, current_layer: Layer) -> Any:
        """Get the output from the layer immediately preceding the current_layer."""
        curr_idx = -1
        for i, l_item in enumerate(self.layers):
            if isinstance(l_item, list):
                if current_layer in l_item:
                    curr_idx = i
                    break
            else:
                if l_item is current_layer:
                    curr_idx = i
                    break
        
        if curr_idx <= 0:
            return None
            
        prev_item = self.layers[curr_idx - 1]
        
        if not isinstance(prev_item, list):
            return prev_item.get_output_wavefront()
            
        outputs = []
        for sub in prev_item:
            if sub:
                outputs.append(sub.get_output_wavefront())
        return outputs

    def add_layer(self, layer: Union[Layer, Component, List[Union[Layer, Component]]]):
        """
        Add a layer or component to the pipeline.

        If a component is provided, it is wrapped in a generic Layer.
        If a list (of components) is provided, a single Layer is created containing all of them.
        """
        
        # Helper to ensure item is a Component or Layer
        def bind_pipeline(item):
            if hasattr(item, 'pipeline'):
                if item.pipeline is not None and item.pipeline is not self:
                    warnings.warn(f"{item} is being moved from Pipeline {item.pipeline} to {self}.")
                item.pipeline = self
            if hasattr(item, 'elements'):
                for e in item.elements:
                    bind_pipeline(e)

        if isinstance(layer, list):
            # Add list directly as a parallel layer group
            # Filter None values first
            clean_list = [item for item in layer if item is not None]
            
            # Validate items in list
            for item in clean_list:
                if isinstance(item, (Component, Layer)):
                    bind_pipeline(item)
                else:
                    raise TypeError(f"Invalid item type in list: {type(item)}")
            
            # Only append if valid items exist
            if clean_list:
                self.layers.append(clean_list)
                
        elif isinstance(layer, Component):
            new_layer = Layer(name=layer.name)
            new_layer.add_component(layer)
            self.layers.append(new_layer)
            bind_pipeline(new_layer)
            
        elif isinstance(layer, Layer):
            self.layers.append(layer)
            bind_pipeline(layer)
        else:
            raise TypeError(f"Invalid layer type: {type(layer)}")

    def description(self, full: bool = False) -> str:
        """Generate a complete text description of the entire simulation setup."""
        lines = ["HELIOS Simulation Pipeline", "=" * 50, ""]
        
        if full:
            pipe_params = []
            if self.date is not None:
                pipe_params.append(f"  • date: {self.date}")
            if self.declination is not None:
                pipe_params.append(f"  • declination: {self.declination}")
            if pipe_params:
                lines.append("Pipeline Parameters:")
                lines.extend(pipe_params)
                lines.append("")
        
        for i, layer in enumerate(self.layers, 1):
            lines.append(f"Layer {i}: {layer.description(full=full)}")
            lines.append("")
        
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize complete pipeline to dictionary."""
        layers_data = [
            [sub_l.to_dict() for sub_l in l] if isinstance(l, list) else l.to_dict()
            for l in self.layers
        ]
        return {
            "date": str(self.date) if self.date else None,
            "declination": serialize_value(self.declination),
            "kwargs": serialize_value(self.kwargs),
            "layers": layers_data
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Pipeline':
        """Reconstruct pipeline from dictionary."""
        date = data.get("date")
        declination = deserialize_value(data.get("declination"))
        kwargs = deserialize_value(data.get("kwargs", {}))
        
        pipe = cls(date=date, declination=declination, **kwargs)
        
        from helios import components
        from helios.core.layer import Layer, GenerationLayer, SamplingLayer, OpticalLayer, DetectionLayer, DataLayer
        
        type_map = {
            name: getattr(components, name)
            for name in getattr(components, "__all__", [])
            if hasattr(components, name)
        }
        type_map.update({
            "Layer": Layer,
            "GenerationLayer": GenerationLayer,
            "SamplingLayer": SamplingLayer,
            "OpticalLayer": OpticalLayer,
            "DetectionLayer": DetectionLayer,
            "DataLayer": DataLayer,
            "Scene": components.Scene,
        })

        def resolve_type(item_data):
            type_name = item_data.get("type")
            if type_name in type_map:
                return type_map[type_name]

            module_name = item_data.get("module")
            if module_name:
                try:
                    module = importlib.import_module(module_name)
                    return getattr(module, type_name)
                except (ImportError, AttributeError, TypeError):
                    pass

            return None

        def restore_item(item_data):
            if item_data is None:
                return None

            cls_obj = resolve_type(item_data)
            if cls_obj is None:
                return Layer.from_dict(item_data)

            try:
                if hasattr(cls_obj, "from_dict"):
                    return cls_obj.from_dict(item_data)
                return cls_obj(name=item_data.get("name"))
            except Exception as e:
                print(f"Error restoring {item_data.get('type')}: {e}")
                return None
        
        def restore_layer(l_data):
            if l_data is None:
                return None
            
            restored = restore_item(l_data)
            if restored is None:
                return None

            if hasattr(restored, "elements") and "elements" in l_data:
                existing = list(getattr(restored, "elements", []))
                if not existing:
                    for element_data in l_data.get("elements", []):
                        element = restore_item(element_data)
                        if element is not None:
                            restored.add_component(element)

            return restored

        layers_data = data.get("layers", [])
        for l_item in layers_data:
            if isinstance(l_item, list):
                layers = [restore_layer(sub) for sub in l_item]
                layers = [layer for layer in layers if layer is not None]
                if layers:
                    pipe.add_layer(layers)
            else:
                layer = restore_layer(l_item)
                if layer:
                    pipe.add_layer(layer)
                    
        return pipe

    def __getitem__(self, key: Union[int, Tuple[int, int]]) -> Union[Layer, Component]:
        """
        Get a layer or component by index.
        
        Parameters
        ----------
        key : int or tuple
            If int: returns the layer at that index.
            If tuple (n, m): returns the m-th component of the n-th layer.
            
        Returns
        -------
        Layer or Component
        """
        if isinstance(key, int):
            return self.layers[key]
        elif isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Pipeline index tuple must be (layer_index, component_index).")
            
            n, m = key
            layer = self.layers[n]
            
            if not hasattr(layer, 'elements'):
                 raise IndexError(f"Layer at index {n} has no elements/components.")
                 
            return layer.elements[m]
        else:
            raise TypeError("Pipeline indices must be integers or tuples of integers.")

    def save(self, filename: Union[str, Path]):
        """Save pipeline to a JSON file (Proxy to helios.io.json)."""
        from ..io.json import save_pipeline
        save_pipeline(self, filename)
            
    @classmethod
    def load(cls, filename: Union[str, Path]) -> 'Pipeline':
        """Load pipeline from a JSON file (Proxy to helios.io.json)."""
        from ..io.json import load_pipeline
        return load_pipeline(filename)

    def get_input_wavefront(self, wavelength: Optional[u.Quantity] = None, 
                            size: Optional[Union[int, u.Quantity]] = None,
                            npix: Optional[int] = None,
                            angular_samples: int = 1,
                            coherent_sources: bool = True,
                            collectors: Optional[List[Any]] = None) -> Union[Wavefront, List[Wavefront]]:
        """Generate the input wavefront(s) from Scene and Atmosphere."""
        if isinstance(size, int):
            if npix is None:
                npix = size
            size = None 
            
        if wavelength is None:
            wavelength = self.kwargs.get('wavelength', 550 * u.nm)
        if npix is None:
            npix = self.kwargs.get('npix', 512)
            
        if size is None:
             size = self.kwargs.get('diameter', 10.0 * u.m)
            
        scene = None
        for layer in self.layers:
            # Flatten search since layers are simpler now
            # But we might need to check inside layer elements?
            # Scene behaves as a Component? or Layer?
            # Scene seems to be a Layer in current code (based on imports GenerationLayer).
            # But if we refactored, it might be Component? (Atmosphere is GenerationComponent).
            # Scene implementation wasn't refactored in this session, assume still Layer-like or wrapper?
            # Actually, check type(layer).__name__
            if type(layer).__name__ in ['Scene', 'PlanetarySystem']:
                scene = layer
                break
            # If Scene is inside a generic Layer (as Component)
            for elem in layer.elements:
                 if type(elem).__name__ in ['Scene', 'PlanetarySystem']:
                     scene = elem 
                     break
            if scene: break
        
        # Checking for collectors similarly
        if collectors is None:
             for layer in self.layers:
                 if type(layer).__name__ == 'TelescopeArray':
                     collectors = layer.elements
                     break
                 for elem in layer.elements:
                     if type(elem).__name__ == 'TelescopeArray':
                         collectors = elem.elements
                         break
                 if collectors: break
        
        if scene is None and collectors is None:
             raise ValueError("Context must contain at least a Scene or a TelescopeArray to generate input wavefront.")
        
        samples = 1
        directions = [(0.0, 0.0)]
        amplitudes = [1.0]
        sources_list = ["Default Source"]
        
        if scene:
            if coherent_sources:
                scene_objects = [obj for obj in scene.objects if type(obj).__name__ in ['Star', 'Planet']]
                if not scene_objects and len(scene.objects) > 0:
                     scene_objects = scene.objects
                
                if scene_objects:
                    samples = len(scene_objects)
                    directions = []
                    amplitudes = []
                    sources_list = []
                    
                    dist = getattr(scene, 'distance', None)
                    
                    for obj in scene_objects:
                        px, py = 0.0 * u.rad, 0.0 * u.rad
                        if hasattr(obj, 'position'):
                            pos = obj.position
                            if len(pos) == 2:
                                px, py = pos
                        
                        tx, ty = 0.0, 0.0
                        if hasattr(px, 'unit'):
                            if px.unit.is_equivalent(u.m) and dist is not None:
                                tx = (px / dist).to(u.rad, equivalencies=u.dimensionless_angles()).value
                                ty = (py / dist).to(u.rad, equivalencies=u.dimensionless_angles()).value
                            elif px.unit.is_equivalent(u.rad):
                                tx = px.to(u.rad).value
                                ty = py.to(u.rad).value
                        
                        directions.append((tx, ty))
                        
                        d_factor = 1.0
                        if dist is not None:
                            d_ref = 10 * u.pc
                            d_factor = (d_ref / dist).to(u.dimensionless_unscaled).value**2
                            
                        mag = getattr(obj, 'magnitude', 0.0)
                        mag_factor = 10**(-0.4 * mag)
                        
                        flux = d_factor * mag_factor
                        amplitudes.append(np.sqrt(flux))
                        
                        name = getattr(obj, 'name', None)
                        if not name:
                            name = type(obj).__name__
                        sources_list.append(name)
            else:
                samples = angular_samples ** 2
                fov = 2.0 * u.arcsec
                if hasattr(scene, 'render'):
                    try:
                        img, x, y = scene.render(npix=angular_samples, fov=fov, return_coords=True)
                        amplitudes = np.sqrt(img.flatten())
                        xg, yg = np.meshgrid(x, y)
                        tx = xg.flatten().to(u.rad).value
                        ty = yg.flatten().to(u.rad).value
                        directions = list(zip(tx, ty))
                        sources_list = [np.array([txi, tyi]) * u.rad for txi, tyi in zip(tx, ty)]
                    except Exception as e:
                        print(f"Warning: Scene rendering failed: {e}. Using default source.")
                        samples = 1
                        directions = [(0.0, 0.0)]
                        amplitudes = [1.0]
                        sources_list = ["Default Source"]
                else:
                    samples = 1
                    directions = [(0.0, 0.0)]
                    amplitudes = [1.0]
                    sources_list = ["Default Source"]

        if collectors is not None:
            wf_list = []
            locations = []
            k = 2 * np.pi / wavelength.to(u.m).value
            
            for collector in collectors:
                if hasattr(collector, 'size') and collector.size is not None:
                    diameter = collector.size
                else:
                    diameter = 1.0 * u.m 

                wf = Wavefront(wavelength=wavelength, size=diameter, npix=npix, nsource=samples)
                if samples == 1 and wf.ndim == 2:
                    wf = wf[np.newaxis, ...]
                wf.sources = sources_list
                wf.source_directions = np.array(directions) * u.rad
                
                try:
                    size_m = diameter.to(u.m).value
                except AttributeError:
                    size_m = float(diameter)
                    diameter = size_m * u.m
                    wf.pixel_scale = (diameter / npix)
                
                u_vec = np.linspace(-size_m/2, size_m/2, npix)
                v_vec = np.linspace(-size_m/2, size_m/2, npix)
                U, V = np.meshgrid(u_vec, v_vec)
                
                if hasattr(collector, 'position'):
                    cx, cy = collector.position
                else:
                    cx, cy = 0.0, 0.0
                
                for s in range(samples):
                    if s < len(directions):
                        tx, ty = directions[s]
                        piston = k * (cx * tx + cy * ty)
                        tilt = k * (U * tx + V * ty)
                        total_phase = piston + tilt
                        phase_factor = np.exp(1j * total_phase)
                        if wf.ndim == 3:
                            wf[s] *= phase_factor
                        else:
                            wf *= phase_factor
                for i in range(samples):
                    if i < len(amplitudes):
                        wf[i] *= amplitudes[i]
                
                wf_list.append(wf)
                locations.append((cx, cy))
            
            return wf_list

        wf = Wavefront(wavelength=wavelength, size=size, npix=npix, nsource=samples)
        if samples == 1 and wf.ndim == 2:
            wf = wf[np.newaxis, ...]
        wf.sources = sources_list
        wf.source_directions = np.array(directions) * u.rad
        
        k = 2 * np.pi / wavelength.to(u.m).value
        size_m = size.to(u.m).value
        u_vec = np.linspace(-size_m/2, size_m/2, npix)
        v_vec = np.linspace(-size_m/2, size_m/2, npix)
        U, V = np.meshgrid(u_vec, v_vec)
        
        for s in range(samples):
            if s < len(directions):
                tx, ty = directions[s]
                tilt = k * (U * tx + V * ty)
                phase_factor = np.exp(1j * tilt)
                if wf.ndim == 3:
                    wf[s] *= phase_factor
                else:
                    wf *= phase_factor

        for i in range(samples):
            if i < len(amplitudes):
                wf[i] *= amplitudes[i]
        
        atmosphere = None
        for layer in self.layers:
            if type(layer).__name__ == 'Atmosphere':
                atmosphere = layer
                break
            for elem in layer.elements:
                if type(elem).__name__ == 'Atmosphere':
                    atmosphere = elem
                    break
            if atmosphere: break
        
        if atmosphere:
            wf = atmosphere.process(wf)
            
        return wf

    def propagate_until(self, target_layer: Union[Layer, Component]) -> Any:
        """
        Run the simulation pipeline until reaching the target layer or component.
        
        If target is a Component, propagation stops just *before* processing that component.
        Returns the input signal to the target component.
        """
        current_signal = None

        for layer in self.layers:
            # Check if this layer IS the target
            if layer is target_layer:
                return current_signal
            
            # Check if target is inside this layer
            if hasattr(layer, 'elements') and target_layer in layer.elements:
                # Target is inside this layer.
                # We need to process elements up to the target.
                
                # Retrieve layer input (current_signal)
                # Assuming layer.process logic: sequential processing of elements?
                # The current Layer.process (in layer.py) typically iterates elements.
                # Use a partial process helper or replicate logic here?
                # Replicating logic is safer to avoid modifying Layer interface recursively.
                
                input_signal = current_signal
                
                for element in layer.elements:
                    if element is target_layer:
                        return input_signal
                    
                    # Process intermediate element
                    input_signal = element.process(input_signal)
                
                # Should have found it
                return input_signal

            # Otherwise, process full layer and continue
            current_signal = layer.process(current_signal)

        return current_signal

    def observe(self) -> Any:
        """Run the simulation through all layers."""
        current_signal = None

        for layer in self.layers:
            current_signal = layer.process(current_signal)

        return current_signal

    def get_output_intensities(self):
        pass

    def validate_architecture(self):
        """Validate the simulation architecture."""
        current_ports = 1
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, list):
                expected_inputs = 0
                for elem in layer:
                    if hasattr(elem, 'num_inputs'):
                        expected_inputs += elem.num_inputs
                    else:
                        expected_inputs += 1
            else:
                if hasattr(layer, 'num_inputs'):
                    expected_inputs = layer.num_inputs
                else:
                    expected_inputs = 1
            
            is_collector = False
            if isinstance(layer, list):
                if len(layer) > 0 and type(layer[0]).__name__ == 'Collector':
                    is_collector = True
            elif type(layer).__name__ == 'TelescopeArray':
                is_collector = True
                
            if not is_collector:
                if current_ports != expected_inputs and current_ports != 1:
                     print(f"Warning: Layer {i+1} expects {expected_inputs} inputs but previous layer provides {current_ports} outputs.")
            
            if isinstance(layer, list):
                current_ports = 0
                for elem in layer:
                    if hasattr(elem, 'num_outputs'):
                        current_ports += elem.num_outputs
                    else:
                        current_ports += 1
            else:
                if hasattr(layer, 'num_outputs'):
                    current_ports = layer.num_outputs
                elif hasattr(layer, 'elements') and len(layer.elements) > 0:
                    if type(layer).__name__ == 'TelescopeArray':
                        current_ports = len(layer.elements)
                    else:
                        current_ports = 1
                else:
                    current_ports = 1

        for i in range(len(self.layers) - 1):
            curr = self.layers[i]
            next_l = self.layers[i+1]
            
            def get_types(item):
                if isinstance(item, list):
                    return {type(sub) for sub in item if sub}
                return {type(item)}
            
            curr_types = get_types(curr)
            next_types = get_types(next_l)
            
            for t_curr in curr_types:
                for t_next in next_types:
                    is_valid = False
                    error_msg = None
                    
                    if issubclass(t_curr, GenerationLayer):
                        if issubclass(t_next, (GenerationLayer, SamplingLayer)): 
                            is_valid = True
                        else:
                            error_msg = f"{t_curr.__name__} can only connect to GenerationLayer or SamplingLayer, not {t_next.__name__}"
                    elif issubclass(t_curr, SamplingLayer):
                        if issubclass(t_next, (OpticalLayer, DetectionLayer)): 
                            is_valid = True
                        else:
                            error_msg = f"{t_curr.__name__} can only connect to OpticalLayer or DetectionLayer, not {t_next.__name__}"
                    elif issubclass(t_curr, OpticalLayer):
                        if issubclass(t_next, (OpticalLayer, DetectionLayer)): 
                            is_valid = True
                        else:
                            error_msg = f"{t_curr.__name__} can only connect to OpticalLayer or DetectionLayer, not {t_next.__name__}"
                    elif issubclass(t_curr, DetectionLayer):
                        if issubclass(t_next, DataLayer): 
                            is_valid = True
                        else:
                            error_msg = f"{t_curr.__name__} can only connect to DataLayer, not {t_next.__name__}"
                    elif issubclass(t_curr, DataLayer):
                        if issubclass(t_next, DataLayer): 
                            is_valid = True
                        else:
                            error_msg = f"{t_curr.__name__} can only connect to DataLayer, not {t_next.__name__}"
                    elif t_curr == Layer or t_next == Layer:
                         is_valid = True
                    else:
                         is_valid = True
                    
                    if not is_valid and error_msg:
                         print(f"Warning: Invalid architecture transition - {error_msg}")

    def plot_uml_diagram(self, **kwargs):
        """Generate a UML-style diagram (Proxy to helios.io.uml_export)."""
        from ..io.uml_export import plot_uml_diagram
        return plot_uml_diagram(self, **kwargs)

# Backward compatibility alias
Context = Pipeline
