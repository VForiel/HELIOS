
import json
import numpy as np
from astropy import units as u

def serialize_exo_data(data):
    if isinstance(data, dict):
        return {k: serialize_exo_data(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [serialize_exo_data(v) for v in data]
    elif isinstance(data, u.Quantity):
        val = data.value
        if isinstance(val, np.ndarray): val = val.tolist()
        return {"__type__": "Quantity", "value": val, "unit": str(data.unit)}
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.float64, np.float32)):
        return float(data)
    elif isinstance(data, (np.int64, np.int32)):
        return int(data)
    return data

def deserialize_exo_data(data):
    if isinstance(data, dict):
        if "__type__" in data and data["__type__"] == "Quantity":
            return data["value"] * u.Unit(data["unit"])
        else:
            return {k: deserialize_exo_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [deserialize_exo_data(v) for v in data]
    return data
