"""
run_all_demos.py

Executes all numbered demo scripts in the examples directory sequentially.
Note: You will need to close the matplotlib windows to proceed to the next demo.
"""
import os
import subprocess
import sys
import glob

def run_all_demos():
    # Get the directory of the current script
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all python scripts starting with two digits
    # Pattern matches 01_..., 02_..., etc.
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
        os.environ["HELIOS_SAVE_PLOTS"] = "true"
        # Create output directory
        output_dir = os.path.join(examples_dir, "../generated/examples")
        os.makedirs(output_dir, exist_ok=True)
    else:
        print("Running in INTERACTIVE mode. Plots will be shown.")
        print("NOTE: Close the plot windows to proceed to the next demo.")
        print("Tip: Run with '--save' to save plots instead of showing them.")
    
    print(f"Found {len(scripts)} demo scripts to execute.")
    
    for i, script in enumerate(scripts):
        script_name = os.path.basename(script)
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(scripts)}] Running {script_name}...")
        print(f"{'='*60}\n")
        
        try:
            # Run the script using the same python interpreter
            # check=True raises CalledProcessError on non-zero exit code
            subprocess.run([sys.executable, script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error running {script_name}: {e}")
            print("Continuing to next demo...")
        except KeyboardInterrupt:
            print("\n🛑 Execution interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            
    print(f"\n{'='*60}")
    print("All demos finished.")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_all_demos()
