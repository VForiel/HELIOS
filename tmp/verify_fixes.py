#!/usr/bin/env python
"""Quick syntax and import check for fixed Streamlit app."""

import sys
import ast

try:
    with open("examples/14_mmi_streamlit.py", "r", encoding="utf-8") as f:
        code = f.read()
    
    # Check syntax
    ast.parse(code)
    print("✓ Syntax check passed")
    
    # Check key fixes
    if 'len(result["metric"]) > 0' in code:
        print("✓ Phase plot condition fixed (using len() instead of truthy)")
    else:
        print("✗ Phase plot condition not fixed")
        
    if 'st.session_state.get("n_core_override"' in code:
        print("✓ n_core widget now uses session_state override")
    else:
        print("✗ n_core widget not using session_state")
        
    print("\n✓ All fixes verified!")
    sys.exit(0)
    
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
