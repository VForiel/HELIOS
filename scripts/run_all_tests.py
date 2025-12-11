import pytest
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_tests():
    print("Running all tests...")
    # Run pytest on the 'tests' directory
    # -v: verbose
    # -x: stop on first failure (optional, but good for quick feedback)
    exit_code = pytest.main(["-v", "tests"])
    
    if exit_code == 0:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed.")
        sys.exit(exit_code)

if __name__ == "__main__":
    run_tests()
