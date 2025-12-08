
import sys
import os
import json
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath("d:/HELIOS"))

from web.backend.app import (
    export_context_file, import_context_file, 
    PipelineRequest, LayerConfig, ScenePayload, TelescopePayload
)
from helios.core.context import Context

def test_backend_api():
    print("Testing Backend API Endpoints...")
    
    # 1. Create a export request
    print("1. Testing Export...")
    req = PipelineRequest(
        mode='pipeline',
        layers=[
            LayerConfig(
                type='scene',
                config=ScenePayload(
                    stars=[{"temperature": 5000, "magnitude": 4.0}],
                    planets=[{"mass": 1.0, "separation": 1.0, "x_arcsec": 0.1, "y_arcsec": 0.0}]
                )
            ),
            LayerConfig(
                type='telescope',
                config= TelescopePayload(preset="VLTI-UT")
            )
        ]
    )
    
    # Call export endpoint
    response = export_context_file(req)
    content = response.body
    json_str = content.decode('utf-8')
    data = json.loads(json_str)
    
    print("Export received JSON keys:", data.keys())
    assert "layers" in data
    assert len(data["layers"]) >= 2
    
    # 2. Testing Import
    print("\n2. Testing Import...")
    # Feed back the data we just exported
    result_pipeline = import_context_file(data)
    
    print("Imported Pipeline Layers:", len(result_pipeline.layers))
    assert len(result_pipeline.layers) == len(req.layers)
    
    # Check types
    types = [l.type for l in result_pipeline.layers]
    print("Layer types:", types)
    assert 'scene' in types
    assert 'telescope' in types
    
    # Check values
    scene_conf = next(l.config for l in result_pipeline.layers if l.type == 'scene')
    print("Scene stars:", len(scene_conf.stars))
    assert len(scene_conf.stars) == 1
    assert scene_conf.stars[0].temperature == 5000.0
    
    print("\nBackend API Verification Successful!")

if __name__ == "__main__":
    try:
        test_backend_api()
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
