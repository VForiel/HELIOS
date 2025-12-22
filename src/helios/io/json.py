import json
from pathlib import Path
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.pipeline import Pipeline

def save_pipeline(pipeline: 'Pipeline', filename: Union[str, Path]):
    """Save pipeline to a JSON file."""
    data = pipeline.to_dict()
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def load_pipeline(filename: Union[str, Path]) -> 'Pipeline':
    """Load pipeline from a JSON file."""
    # Local import to avoid circular dependency
    from ..core.pipeline import Pipeline
    
    with open(filename, 'r') as f:
        data = json.load(f)
    return Pipeline.from_dict(data)
