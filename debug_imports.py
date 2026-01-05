
import sys
import site
import os

print(f"Python Executable: {sys.executable}")
print(f"Sys Path: {sys.path}")

try:
    import poppy
    print(f"Poppy: {poppy.__file__}")
except ImportError as e:
    print(f"Poppy Import Failed: {e}")

try:
    import dlux
    print(f"dLux: {dlux.__file__}")
except ImportError as e:
    print(f"dLux Import Failed: {e}")
    
try:
    import jax
    print(f"JAX: {jax.__file__}")
except ImportError as e:
    print(f"JAX Import Failed: {e}")
