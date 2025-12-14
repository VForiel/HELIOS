import sys
import os
sys.path.insert(0, os.path.abspath('src'))
try:
    import helios
    print("Helios imported successfully")
    print("Attributes:", dir(helios))
    if hasattr(helios, 'Context'):
        print("helios.Context exists")
    else:
        print("helios.Context MISSING")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
