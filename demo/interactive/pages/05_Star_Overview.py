import streamlit as st
from pathlib import Path
import sys

# --- Path Setup ---
ROOT = Path(__file__).parent.parent.parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
    
# Import utils
UTILS = Path(__file__).parent.parent / "utils"
if str(UTILS.parent) not in sys.path:
    sys.path.insert(0, str(UTILS.parent))

from utils.display import display_code
from helios.io.external_query.stars.query_all import get_star_properties

# --- Page Config ---
st.set_page_config(
    page_title="Star Overview",
    page_icon="⭐",
    layout="wide"
)

st.title("Star Overview ⭐")
st.markdown("""
Get a comprehensive overview of a star's properties by querying SIMBAD and Gaia catalogs.
Values are merged and units are handled automatically.
""")

# --- Show Code ---
EXAMPLE_PATH = ROOT / "demo" / "scripts" / "05_star_overview.py"
display_code(EXAMPLE_PATH)

st.divider()

# --- Interactive Demo ---

col1, col2 = st.columns([1, 2])

with col1:
    star_name = st.text_input("Star Name", value="Vega")
    fetch_btn = st.button("Fetch Data", type="primary")

if fetch_btn and star_name:
    with st.spinner(f"Querying data for {star_name}..."):
        # We set plot=False to purely get data
        data = get_star_properties(star_name, complete_data=False, plot=False)
        
        if data:
            st.success(f"Data retrieved for {star_name}")
            
            # Use JSON explorer for the dictionary
            st.write("### Raw Data Structure")
            st.json(data)
            
            # We could also pretty print specific sections if we wanted
            # e.g. Physics
            if 'physics' in data:
                st.write("### Physics Properties")
                st.write(data['physics'])
        else:
            st.error(f"Could not retrieve data for {star_name}")
