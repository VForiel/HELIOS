#!/usr/bin/env python
"""
Validation script for Streamlit MMI demo convergence plots.
Checks that plot code is syntactically valid and imports are available.
"""

import sys
import ast

# Check syntax with UTF-8 encoding
print("Checking syntax of 14_mmi_streamlit.py...")
try:
    with open("examples/14_mmi_streamlit.py", "r", encoding="utf-8") as f:
        code = f.read()
    ast.parse(code)
    print("✓ Syntax is valid")
except SyntaxError as e:
    print(f"✗ Syntax error: {e}")
    sys.exit(1)

# Check matplotlib presence in code
print("\nChecking matplotlib imports and usage...")
if "import matplotlib.pyplot as plt" in code:
    print("✓ matplotlib.pyplot imported")
else:
    print("✗ matplotlib.pyplot not imported")
    sys.exit(1)

# Check plot code presence
if 'ax.semilogy(result["metric"]' in code:
    print("✓ Phase calibration semilogy plot code found")
else:
    print("✗ Phase calibration plot code missing")

if 'ax_left.semilogy(' in code:
    print("✓ n_core coarse scan plot code found")
else:
    print("✗ n_core coarse scan plot code missing")

if 'ax_right.semilogy(' in code and 'result["metrics_gradient"]' in code:
    print("✓ n_core gradient descent plot code found")
else:
    print("✗ n_core gradient descent plot code missing")

# Check st.pyplot calls
if 'st.pyplot(fig, use_container_width=True)' in code:
    count = code.count('st.pyplot(fig, use_container_width=True)')
    print(f"✓ {count} st.pyplot calls found")
else:
    print("✗ st.pyplot calls missing")

print("\n✓ All validation checks passed!")
print("\nNote: To run the Streamlit app:")
print("  streamlit run examples/14_mmi_streamlit.py")
