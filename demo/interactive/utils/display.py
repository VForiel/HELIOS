import streamlit as st
import inspect
from pathlib import Path

def display_code(file_path: str | Path):
    """
    Display the content of a file in an expendable block.
    
    Args:
        file_path: Path to the file to display.
    """
    path = Path(file_path)
    if not path.exists():
        st.error(f"File not found: {path}")
        return

    with st.expander("📝 Show Source Code", expanded=False):
        st.code(path.read_text(encoding="utf-8"), language="python")

def show_source(obj):
    """
    Display the source code of a python object (function/class).
    """
    try:
        lines = inspect.getsource(obj)
        with st.expander(f"📝 Show Source: {obj.__name__}", expanded=False):
            st.code(lines, language="python")
    except OSError:
        st.error(f"Could not retrieve source for {obj}")
