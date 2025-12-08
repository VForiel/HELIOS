
import requests
import json

def test_api():
    url = "http://localhost:8001/api/preview_layer"
    payload = {
        "type": "scene",
        "config": {
            "stars": [{"temperature": 5778, "magnitude": 4.83}],
            "planets": [{"mass": 1.0, "separation": 1.0}],
            "view_mode": "geometry"
        }
    }
    
    try:
        print(f"Sending request to {url}...")
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Success! Image received.")
            print(f"Content Type: {response.headers.get('content-type')}")
            print(f"Response Size: {len(response.content)} bytes")
        else:
            print("Error!")
            print(response.text)
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_api()
