
import sys
import os
import importlib.util

script_path = os.path.abspath("d:/HELIOS/demo/scripts/14_uml_visualization.py")
module_name = "demo_14"
spec = importlib.util.spec_from_file_location(module_name, script_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

print("Running 14 with save=True...")
try:
    module.run_demo(save=True)
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
