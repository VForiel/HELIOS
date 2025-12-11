import pytest
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from web.backend.app import app

client = TestClient(app)

def test_inspect_node_scene():
    """Test inspecting the Scene layer (index 0)."""
    payload = {
        "mode": "pipeline",
        "layers": [
            {
                "type": "scene",
                "config": {
                    "stars": [{"temperature": 5778, "magnitude": 0, "x_arcsec": 0, "y_arcsec": 0}],
                    "planets": [],
                    "zodiacal": {"enabled": False}
                }
            },
            {
                "type": "telescope",
                "config": {
                    "preset": "Custom",
                    "diameter": 8.0,
                    "collectors": [{"x": 0, "y": 0, "diameter": 8.0, "pupil_type": "Circular"}]
                }
            }
        ]
    }
    
    print("Testing Scene Inspection...")
    # Inspect Node 0 (Scene)
    response = client.post("/api/inspect_node?target_index=0", json=payload)
    if response.status_code != 200:
        print(f"FAILED: {response.text}")
        sys.exit(1)
    
    assert response.headers["content-type"] == "image/png"
    print("✓ Scene Inspection Passed")
    
    print("Testing Telescope Inspection...")
    # Inspect Node 1 (Telescope) - Should trigger Pull Model from Scene
    response = client.post("/api/inspect_node?target_index=1", json=payload)
    if response.status_code != 200:
        print(f"FAILED: {response.text}")
        sys.exit(1)
        
    assert response.headers["content-type"] == "image/png"
    print("✓ Telescope Inspection Passed")

def test_inspect_node_out_of_bounds():
    print("Testing Out of Bounds...")
    payload = {
        "mode": "pipeline",
        "layers": [{"type": "scene", "config": {}}]
    }
    response = client.post("/api/inspect_node?target_index=5", json=payload)
    assert response.status_code == 400
    print("✓ Out of Bounds Check Passed")

if __name__ == "__main__":
    try:
        test_inspect_node_scene()
        test_inspect_node_out_of_bounds()
        print("\nAll Verification Tests Passed!")
    except Exception as e:
        print(f"\nVerification Failed: {e}")
        sys.exit(1)
