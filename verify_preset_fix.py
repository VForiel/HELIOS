import requests
import json

try:
    print("Testing /api/presets/VLTI-UT...")
    response = requests.get("http://127.0.0.1:8001/api/presets/VLTI-UT")
    
    if response.status_code == 200:
        data = response.json()
        print("Success! Response:")
        print(json.dumps(data, indent=2))
        
        # Validate data structure
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if "x" in first and "y" in first and "id" in first:
                print("Data structure valid.")
            else:
                print("FAILED: Invalid data structure.")
        else:
             print("FAILED: Empty or invalid list.")
    else:
        print(f"FAILED: Status Code {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"FAILED: Exception {e}")
