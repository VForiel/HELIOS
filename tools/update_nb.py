
import json

def update_notebook():
    nb_path = r"d:\HELIOS\examples\mmi.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Walk through cells
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            new_source = []
            for line in cell["source"]:
                # Update imports
                if "from helios.sim.multi_mode_interferometer" in line:
                    line = line.replace("helios.sim.multi_mode_interferometer", "helios.sim.mmi")
                    line = line.replace("calculate_mmi_contrib_data", "compute_contributions")
                if "from helios.sim import mmi_contributions" in line:
                     # This might be tricky if it spans lines, but looking at previous view it was one line
                     # or close. 
                     # "from helios.sim import mmi_contributions"
                     # changed to "from helios.sim.mmi import simulate_contributions" (or similar)
                     pass
                
                # Let's do string replacements which are safer for simple renames
                line = line.replace("calculate_mmi_contrib_data", "compute_contributions")
                line = line.replace("mmi_contributions", "simulate_contributions")
                
                # Fix the specific import line manually if needed
                # Original: "from helios.sim.multi_mode_interferometer import calculate_mmi_contrib_data\n"
                # New: "from helios.sim.mmi import compute_contributions\n"
                
                # Original: "from helios.sim import mmi_contributions"
                # New: "from helios.sim.mmi import simulate_contributions"
                
                # Let's just handle the two specific import blocks we saw in step 74
                if "from helios.sim.multi_mode_interferometer import calculate_mmi_contrib_data" in line:
                     line = "from helios.sim.mmi import compute_contributions\n"
                if "from helios.sim import mmi_contributions" in line:
                     line = "from helios.sim.mmi import simulate_contributions\n"
                
                new_source.append(line)
            cell["source"] = new_source

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=4) # indent=1 is usually closer to default or whatever

    print("Notebook updated.")

if __name__ == "__main__":
    update_notebook()
