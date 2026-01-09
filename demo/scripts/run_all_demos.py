"""
run_all_demos.py

Executes all numbered demo scripts in the examples directory sequentially.
"""
import os
import sys
import glob
import importlib.util

def run_all_demos():
    # Get the directory of the current script
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add examples dir to sys.path to allow importing modules
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)

    # Find all python scripts starting with two digits
    pattern = os.path.join(examples_dir, "[0-9][0-9]_*.py")
    scripts = glob.glob(pattern)
    scripts.sort()
    
    if not scripts:
        print("No numbered demo scripts found in examples directory.")
        return

    # Check for --save argument
    save_plots = "--save" in sys.argv
    if save_plots:
        print("Running in SAVE mode. Plots will be saved to 'generated/examples/' instead of shown.")
    else:
        print("Running in INTERACTIVE mode. Plots will be shown.")
    
    print(f"Found {len(scripts)} demo scripts to execute.")
    
    for i, script_path in enumerate(scripts):
        script_name = os.path.basename(script_path)
        module_name = os.path.splitext(script_name)[0]
        
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(scripts)}] Running {script_name}...")
        print(f"{'='*60}\n")
        
        try:
            # Import module dynamically
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            if spec is None or spec.loader is None:
                print(f"❌ Could not load spec for {script_path}")
                continue
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for run_demo function
            if hasattr(module, "run_demo"):
                # Call run_demo with save argument
                module.run_demo(save=save_plots)
            else:
                print(f"⚠️  No 'run_demo' function found in {script_name}")

        except Exception as e:
            print(f"\n❌ Error running {script_name}: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing to next demo...")
            
    print(f"\n{'='*60}")
    print("All demos finished.")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_all_demos()
