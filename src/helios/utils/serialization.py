import numpy as np
from astropy import units as u
from typing import Any

def serialize_value(value: Any) -> Any:
    """Recursively serialize values to JSON-friendly types."""
    if isinstance(value, u.Quantity):
        return {"value": float(value.value), "unit": str(value.unit)}
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    elif hasattr(value, 'to_dict'):
        return value.to_dict()
    return value

def deserialize_value(value: Any) -> Any:
    """Recursively deserialize values from JSON types."""
    if isinstance(value, dict):
        if "value" in value and "unit" in value and len(value) == 2:
            try:
                return value["value"] * u.Unit(value["unit"])
            except Exception:
                pass # Not a quantity dict
        return {k: deserialize_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [deserialize_value(v) for v in value]
    return value
