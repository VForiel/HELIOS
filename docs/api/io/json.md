# JSON Serialization

HELIOS provides functions to save and load simulation pipelines using JSON format.

## Overview

The JSON module enables:
- Saving complete pipeline configurations to disk
- Loading pipelines from saved files
- Serializing complex objects (Wait, serialization helpers are internal usually, but `save_pipeline` handles it).

## API Reference

```{eval-rst}
.. automodule:: helios.io.json
   :members:
   :undoc-members:
   :show-inheritance:
```

## Example Usage

```python
import helios
from helios.io.json import save_pipeline, load_pipeline

# Create a pipeline
pipeline = helios.Pipeline()
# ... add layers ...

# Save to file
save_pipeline(pipeline, "my_pipeline.json")

# Load from file
loaded_pipeline = load_pipeline("my_pipeline.json")

# Verify
print(loaded_pipeline.description())
```
