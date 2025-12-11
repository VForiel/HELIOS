import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import sys

def execute_notebook(notebook_path):
    print(f"Executing {notebook_path}...")
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
        
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    try:
        # Set path to project root for imports
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ep.preprocess(nb, {'metadata': {'path': root_path}})
        print("Notebook executed successfully.")
    except Exception as e:
        print(f"Error executing the notebook: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Assuming demo.ipynb is in the root or examples folder
    # Based on file structure, likely in root or examples.
    # checking repo structure via list_dir earlier didn't show it in root?
    # Let's assume it's in the root as 'demo.ipynb' or check later.
    # The test calls this script from repo_root.
    
    notebook_filename = 'demo.ipynb'
    # Try to find it
    potential_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'demo.ipynb'),
        os.path.join(os.path.dirname(__file__), '..', 'examples', 'demo.ipynb')
    ]
    
    target_path = None
    for p in potential_paths:
        if os.path.exists(p):
            target_path = p
            break
            
    if not target_path:
        print("demo.ipynb not found.")
        sys.exit(1)
        
    execute_notebook(target_path)
