# HELIOS AI Coding Agent Instructions

## 🚨 CRITICAL FIRST STEP 🚨

**Before writing any code**, you **MUST** read the following documentation files to understand the project architecture, developer guidelines, and contribution standards:

1.  **`README.md`** and **`docs/index.md`**: General project overview.
2.  **`docs/architecture.md`**: Understanding the layered optical simulation engine and propagation logic.
3.  **`docs/contribute.md`**: Strict coding conventions, unit standards, and "Full Agent" development strategy.

**Failure to read these documents will result in incorrect architectural decisions.**

---

## 🤖 Full Agent Development Mode

You are working in **"Full Agent"** mode. You are the primary developer.
*   **Write code for AI readability**: Explain *why* you do things.
*   **Log every session**: You MUST create a modification log in `.github/agent-logs/` (see `docs/contribute.md` for format).

## ⚡ Quick Constraints Checklist

*   **Virtual Env**: Always ensure the virtual environment is active.
*   **Units**: ALL physical parameters MUST use `astropy.units`. Never use raw floats for physical quantities.
*   **File Structure**:
    *   `src/helios/`: Source code ONLY.
    *   `tests/`: Tests ONLY.
    *   `examples/`: User scripts.
    *   `tmp/`: YOUR scratchpad. Use this for temporary scripts. **NEVER** create files at root.
*   **Atomic Commands**: Run terminal commands one by one. Do NOT chain with `&` or `;`.
*   **Do not run scripts in terminal**: Avoid `python -c "..."`. Create a temporary script instead and then run it.

## 🧪 Testing

*   **Physical Coherence**: Tests must verify that results make physical sense (units, conservation of energy), not just that they run.
*   **Run Tests**: Always run tests before finishing a task.

---

## 🍎 Physics Sources and Verification

**ALWAYS** check on internet to ensure your physical reasoning and/or implementation is correct.

For light propagation, refer to the following sources:

| Bibliothèque | Langage / Backend | Type de propagation | Points forts | Points faibles | Limitations principales | Liens |
|-------------|------------------|---------------------|--------------|----------------|-------------------------|-------|
| **POPPY** | Python (NumPy) | Fresnel / Fraunhofer | Standard astro (JWST), excellente doc, unités Astropy | CPU only, architecture rigide | Peu flexible hors astro, FFT-centric | https://poppy-optics.readthedocs.io |
| **HCIPy** | Python (NumPy/C++) | FFT, Fresnel, MFT | Très complet (AO, polarisation, segments), end-to-end | API complexe, courbe d’apprentissage | Scalaire par défaut, sampling délicat | https://hcipy.readthedocs.io |
| **PROPER** | IDL / Python / Matlab | Fresnel (FFT) | Référence NASA/JPL, très robuste | API vieillissante, peu pythonique | Strictement propagation | https://proper-library.sourceforge.net |
| **dLux** | Python (JAX, XLA) | Fourier optics différentiable | Autodiff, GPU/TPU, calibration inverse | Communauté réduite, JAX mindset | VRAM GPU, encore jeune | https://github.com/LouisDesdoigts/dLux |
| **Diffractio** | Python | RS, BPM, Fresnel, vectoriel | Très pédagogique, vectoriel, X-ray | Performances modestes | Peu scalable pour pipelines lourds | https://diffractio.readthedocs.io |
| **waveprop** | Python (NumPy / PyTorch) | Angular Spectrum, Fresnel | Simple, GPU possible, clair | Peu d’éléments optiques | Bas niveau | https://github.com/ebezzam/waveprop |
| **LightPipes** | C++ / Python | FFT scalaire | Rapide, cavités, modes laser | Peu astro, unités faibles | Modèle ancien | https://opticspy.github.io/lightpipes |
| **PyOptica** | Python | Diffraction scalaire | Léger, lisible | Petite communauté | Peu d’éléments avancés | https://pypi.org/project/pyoptica |
| **PyNX (wavefront)** | Python (CUDA/OpenCL) | Fresnel / FFT | Très rapide, GPU, HPC | Peu généraliste | Orienté X-ray | https://pynx.esrf.fr |
| **WPG** | Python / C++ | Cohérent & partiellement cohérent | Gestion avancée de cohérence | Courbe d’apprentissage | Peu astro visible | https://wpg.readthedocs.io |
| **TorchOptics** | Python (PyTorch) | Fourier optics diff. | Autodiff, GPU, ML-ready | Jeune, API instable | Physique simplifiée | https://github.com/matthewfilipovich/torchoptics |
| **AOtools** | Python | FFT paraxial | Turbulence, AO, PSF | Pas propagation pure | AO-centric | https://github.com/AOtools/aotools |
| **HoloPy** | Python / Fortran | Diffraction scalaire | Propagation inverse, holographie | Cas d’usage étroit | Peu flexible | https://holopy.readthedocs.io |
| **Wavesim** | Python / Matlab | Helmholtz (MBS) | Haute précision, sans FFT | Très coûteux numériquement | Champs limités | https://www.wavesim.org |
| **Finesse 3** | C / Python | Modale / fréquentielle | Référence LIGO/Virgo, bruits quantiques | Pas image directe | Pas pixel-based | https://finesse.ifosim.org |
| **Meep** | C++ / Python | EM complet (FDTD) | Physique complète | Lent, lourd | Pas Fourier optics | https://meep.readthedocs.io |
| **FourierOpticsToolBox** | Matlab | Fourier optics | Clair, pédagogique | Matlab-only | Peu performant | https://github.com/USNavalResearchLaboratory/FourierOpticsToolBox |

---

**For all detailed rules, refer to `docs/contribute.md`.**