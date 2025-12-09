
from web.backend.app import PipelineRequest, export_context_file
import helios.core.context as helios
from fastapi import HTTPException
import json

# Mock data simulating Frontend Payload
payload = {
    "mode": "pipeline",
    "layers": [
        {
            "type": "scene",
            "config": {
                "stars": [],
                "planets": [],
                "zodiacal": False
            },
            "metadata": {"position": {"x": 50, "y": 100}}
        },
        {
            "type": "telescope",
            "config": {
                "input_diameter": 1.0,
                "collectors": []
            },
            "metadata": {"position": {"x": 500, "y": 100}}
        },
        [
            {
                "type": "camera",
                "config": {
                     "resolution": [128, 128],
                     "pixel_scale": 10
                },
                "metadata": {"position": {"x": 950, "y": 100}}
            },
            {
                "type": "camera",
                "config": {
                     "resolution": [128, 128],
                     "pixel_scale": 10
                },
                "metadata": {"position": {"x": 950, "y": 200}}
            }
        ]
    ]
}

print("Attempting to parse PipelineRequest...")
try:
    req = PipelineRequest(**payload)
    print("PipelineRequest parsed successfully.")
except Exception as e:
    print(f"PipelineRequest Parse Error: {e}")
    exit(1)

print("Attempting to run export_context_file logic...")
try:
    # We can't call the endpoint directly because it returns a Response object/Blob usually, 
    # but the function itself does logic + return FileResponse.
    # Let's just run the logic inside export_context_file normally.
    # But wait, export_context_file expects a PipelineRequest object.
    
    # Copying relevant logic from app.py to pinpoint error without full server
    context = helios.Context()
    # Mocking create functions? simpler to import them or rely on app.py import
    # app.py imports create_scene etc.
    # But they might fail if dependencies missing?
    # Let's try calling the real function first.
    
    # We need to mock 'export_context_file' slightly if it relies on DB? It doesn't.
    # It relies on helios.Context.
    
    # The endpoint function:
    # export_context_file(request: PipelineRequest)
    # It returns a FileResponse.
    
    # Let's try running it.
    res = export_context_file(req)
    print("Export successful!")
    print(res)

except Exception as e:
    print(f"Export Logic Error: {e}")
    import traceback
    traceback.print_exc()
