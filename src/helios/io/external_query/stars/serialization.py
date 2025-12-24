
import json
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

def serialize_star_data(data):
    """
    Recursively converts star_data dictionary into a JSON-serializable format.
    Handles Astropy Quantities and Numpy arrays.
    """
    if isinstance(data, dict):
        return {k: serialize_star_data(v) for k, v in data.items()}
    
    elif isinstance(data, (list, tuple)):
        return [serialize_star_data(v) for v in data]
        
    elif isinstance(data, u.Quantity):
        # Convert to dict {value, unit}
        val = data.value
        # If array, convert to list
        if isinstance(val, np.ndarray):
            val = val.tolist()
        return {
            "__type__": "Quantity",
            "value": val,
            "unit": str(data.unit)
        }
    
    elif isinstance(data, np.ndarray):
        return data.tolist()
    
    elif isinstance(data, (np.int64, np.int32)):
        return int(data)
        
    elif isinstance(data, (np.float64, np.float32)):
        return float(data)
        
    return data

def deserialize_star_data(data):
    """
    Recursively reconstructs star_data dictionary from JSON.
    Restores Astropy Quantities.
    """
    if isinstance(data, dict):
        if "__type__" in data and data["__type__"] == "Quantity":
            val = data["value"]
            unit = u.Unit(data["unit"])
            return val * unit
        else:
            return {k: deserialize_star_data(v) for k, v in data.items()}
            
    elif isinstance(data, list):
        return [deserialize_star_data(v) for v in data]
        
    return data
