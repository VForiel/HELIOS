
import sys
import os
import importlib.util

# Add demo/scripts to path logic is tricky with numbers.
# We'll just load by path.

script_path = os.path.abspath("d:/HELIOS/demo/scripts/07_interferometry_arrays.py")
module_name = "demo_07"
spec = importlib.util.spec_from_file_location(module_name, script_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

print("Running 07 with save=True...")
try:
    module.run_demo(save=True)
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
