
import requests
import json
import traceback

BASE_URL = "http://localhost:8002"

def test_preset_api(preset_name, expected_count):
    print(f"Testing preset: {preset_name}...")
    try:
        response = requests.get(f"{BASE_URL}/api/presets/{preset_name}")
        if response.status_code == 200:
            data = response.json()
            print(f"SUCCESS: Got {len(data)} collectors.")
            if len(data) == expected_count:
                print("Count matches expectation.")
                # Print first collector sample
                print("Sample:", json.dumps(data[0], indent=2))
            else:
                print(f"WARNING: Expected {expected_count} but got {len(data)}")
        else:
            print(f"FAILED: Status {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_preset_api("VLTI-UT", 4)
    test_preset_api("LIFE", 4)
    test_preset_api("INVALID", 0) # Should fail
