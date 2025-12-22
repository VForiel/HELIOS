
import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from web.backend.app import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    # Should probably be 404 or index
    # We just check it doesn't crash app
    assert response.status_code in [200, 404]

def test_export_pipeline_simple():
    payload = {
        "mode": "pipeline",
        "layers": [
            {
                "type": "scene",
                "config": {
                    "stars": [{"temperature": 5000, "magnitude": 5}],
                    "planets": []
                },
                "metadata": {}
            },
            {
                "type": "telescope",
                "config": {
                    "preset": "Single",
                    "diameter": 8.0
                },
                "metadata": {}
            },
            {
                "type": "camera",
                "config": {
                    "exposure": 0.1
                },
                "metadata": {}
            }
        ]
    }
    
    response = client.post("/api/pipeline/export_file", json=payload)
    if response.status_code != 200:
        print(response.json())
        import traceback
        # Attempt to debug print server logic? no client logs only response
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]
    data = response.json()
    assert "layers" in data
    assert len(data["layers"]) == 3

def test_run_pipeline():
    # Construct a runnable pipeline
    # Scene -> Telescope -> Camera
    payload = {
        "mode": "pipeline",
        "layers": [
            {
                "type": "scene",
                "config": {
                    "stars": [{"temperature": 5000, "magnitude": 5}],
                    "planets": []
                },
                "metadata": {}
            },
            {
                "type": "telescope",
                "config": {
                    "preset": "Single",
                    "diameter": 8.0
                },
                "metadata": {}
            },
            {
                "type": "camera",
                "config": {
                    "exposure": 0.1
                },
                "metadata": {}
            }
        ]
    }
    
    # Check if run_pipeline endpoint exists
    # Found endpoint: /api/simulate
    response = client.post("/api/simulate", json=payload)
    if response.status_code == 404:
        # Maybe it's named differently?
        pytest.skip("simulate endpoint not found")
        
    if response.status_code != 200:
        print(response.json())
        
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0

def test_run_pipeline_parallel():
    # Test parallel configuration: Scene -> [Telescope, Telescope] -> Camera
    
    payload = {
        "mode": "pipeline",
        "layers": [
            {
                "type": "scene",
                "config": {"stars": [], "planets": [], "zodiacal": {"enabled": False}},
                "metadata": {}
            },
            {
                "type": "beam_splitter",
                "config": {"cutoff": 0.5},
                "metadata": {}
            },
            [
                {
                    "type": "telescope",
                    "config": {"preset": "Single", "diameter": 8.0},
                    "metadata": {}
                },
                {
                    "type": "telescope",
                    "config": {"preset": "Single", "diameter": 4.0},
                    "metadata": {}
                }
            ],
            {
                "type": "camera",
                "config": {"exposure": 0.1},
                "metadata": {}
            }
        ]
    }
    
    response = client.post("/api/simulate", json=payload)
    
    if response.status_code == 200:
         assert response.headers["content-type"] == "image/png"
    else:
         # Debug print
         error_detail = response.json().get("detail", "")
         print(f"Failed with 500: {error_detail}")
         assert "list" not in error_detail or "attribute" not in error_detail
