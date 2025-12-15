import requests
import json
import time

payload = {
    "mode": "inspection",
    "layers": [
        {
            "type": "scene",
            "config": {
                "stars": [{"temperature": 5778, "magnitude": 4.83, "x_arcsec": 0, "y_arcsec": 0}],
                "planets": [],
                "zodiacal": {"enabled": False, "brightness": 1.0},
                "view_mode": "geometry"
            },
            "metadata": {"id": "scene-1", "position": {"x": 0, "y": 0}}
        },
        {
            "type": "telescope",
            "config": {
                "preset": "Single",
                "diameter": 8.0,
                "pupil_type": "Circular",
                "central_obstruction": 0.0,
                "spiders": 0,
                "positions": [{"id": "h1", "x": 0, "y": 0}]
            },
            "metadata": {"id": "telescope-1", "position": {"x": 200, "y": 0}}
        },
        {
            "type": "camera",
            "config": {
                "pixels": [512, 512],
                "integration_time": 0.1,
                "read_noise": 5,
                "dark_current": 0.1
            },
            "metadata": {"id": "camera-1", "position": {"x": 400, "y": 0}}
        }
    ]
}

# Inspect Nodes
nodes_to_inspect = ["scene-1", "telescope-1", "camera-1"]

for node_id in nodes_to_inspect:
    print(f"\nInspecting {node_id}...")
    try:
        # Test with custom params
        query_params = "?width=10&height=3&style=xkcd" 
        response = requests.post(f"http://localhost:8001/api/inspect_node{query_params}&target_id={node_id}", json=payload)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            plots = data.get("plots", [])
            print(f"Received {len(plots)} plots")
            for i, plot in enumerate(plots):
                 title = plot.get("title", "Untitled")
                 print(f"  - Plot {i+1}: {title} (Len: {len(plot['image'])})")
                 # Option: Decode and save
                 
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
