import json

# Load notebook
with open('examples/demo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Original cells: {len(cells)}\n")

# Reorganize cells
# Section 1: Scene (cells 0-7)
# Section 2: Pupilles (cells 18-21) - MOVED HERE
# Section 3: Atmosphère (cells 8-17) - MOVED HERE  
# Section 4: Coronagraphes (cells 22-26)
# Section 5: Advanced (cells 27-28)
# Section 6: End-to-end (cells 29-30)

new_cells = []

# Section 1: Scene (cells 0-7, indices include markdown+code)
new_cells.extend(cells[0:8])

# Section 2: Pupilles/Interferometry - create new section header + cells 18-21
pupil_section_header = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2. Telescope Pupil & Interferometry\n",
        "The telescope pupil defines the aperture geometry that collects light. HELIOS provides tools to construct custom pupils or use presets.\n",
        "\n",
        "You can build pupils manually with geometric primitives (disks, spiders, segments) or load telescope presets (JWST, VLT, ELT)."
    ]
}
new_cells.append(pupil_section_header)
new_cells.extend(cells[19:22])  # Pupil cells (skip old section header at 18)

# Section 3: Atmosphère - update section header + cells 9-17
atm_section_header = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 3. Atmospheric Phase Screen (Chromatic Aberrations)\n",
        "The atmosphere introduces turbulent optical path difference (OPD) errors that are **chromatic** - the phase shift depends on wavelength: φ = 2π × OPD / λ.\n",
        "\n",
        "This means shorter wavelengths (blue) experience larger phase aberrations than longer wavelengths (infrared) for the same atmospheric turbulence.\n",
        "\n",
        "Temporal evolution is modeled via **frozen-flow turbulence** (Taylor hypothesis): turbulent screens drift at constant wind velocity."
    ]
}
new_cells.append(atm_section_header)
new_cells.extend(cells[9:18])  # Atmosphere cells (skip old section header at 8)

# Section 4: Coronagraphes - update to section 4
coro_section_header = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4. Coronagraphic Phase Masks\n",
        "Coronagraphs use phase masks to suppress starlight and enhance the detection of faint companions. HELIOS supports vortex and 4-quadrant phase masks."
    ]
}
new_cells.append(coro_section_header)
new_cells.extend(cells[23:27])  # Coronagraph cells (skip old section header at 22)

# Section 5: Advanced - update to section 5
adv_section_header = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5. Advanced: Atmospheric Degradation\n",
        "\n",
        "**Chromatic PSF degradation**\n",
        "\n",
        "The atmosphere degrades the PSF through chromatic phase aberrations. We compare PSFs at different wavelengths and atmospheric conditions to visualize the chromatic nature of seeing.\n",
        "\n",
        "All PSFs are normalized to the ideal (no-atmosphere) peak to visualize Strehl ratio: Strehl = peak_degraded / peak_ideal."
    ]
}
new_cells.append(adv_section_header)
new_cells.extend(cells[28:29])  # Advanced cells (skip old section header at 27)

# Section 6: End-to-end - update to section 6
e2e_section_header = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6. Full End-to-End Simulation\n",
        "Putting it all together: Scene → Collectors → Camera."
    ]
}
new_cells.append(e2e_section_header)
new_cells.extend(cells[30:31])  # E2E cells (skip old section header at 29)

# Update notebook cells
nb['cells'] = new_cells

# Update title to reflect new organization
nb['cells'][0]['source'] = [
    "# HELIOS Demo\n",
    "This notebook demonstrates the basic usage of the HELIOS simulation package.\n",
    "\n",
    "The demo follows the optical path of a simulation:\n",
    "1. **Scene observation** - Define astronomical objects and their spectral energy distributions\n",
    "2. **Telescope pupil & Interferometry** - Configure aperture geometry and baseline arrays\n",
    "3. **Atmospheric effects** - Visualize and control atmospheric phase screens\n",
    "4. **Coronagraphic masks** - Apply phase masks for high-contrast imaging"
]

# Save reorganized notebook
with open('examples/demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Reorganized cells: {len(new_cells)}")
print("✓ Notebook reorganized successfully!")
