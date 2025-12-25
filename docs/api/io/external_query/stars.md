# Stars

Query logic for stellar objects.

```{eval-rst}
.. autofunction:: helios.io.get_star_properties
   :noindex:
```

## Retrieval Process

1.  **Identifier Resolution:**
    *   Queries **Simbad** to resolve the star name to ICRS coordinates and retrieve basic photometry (V-band).
    *   Retrieves parallax for distance calculation.

2.  **Spectral Data Search (Prioritized):**
    The system searches multiple archives for flux-calibrated spectra. The search stops as soon as a high-quality spectrum is found.
    *   **Priority 1: CALSPEC (STScI):** High-fidelity standards (UV-NIR).
    *   **Priority 2: ESO Phase 3:** Reduced 1D spectra (X-Shooter, UVES) via TAP query.
    *   **Priority 3: Vizier Catalogs:**
        *   Standard Stars (Morel et al. 2011)
        *   ESO Stellar Library (STELIB)
        *   IRTF Spectral Library (Rayner 2009) - Crucial for cool stars in IR.
        *   Burnashev Spectrophotometry

3.  **Fallback to Models:**
    If no observed spectrum (>10 points) is found, the system falls back to theoretical models.
    *   **Pickles (1998):** Matches Spectral Type (from Simbad) or Effective Temperature.
    *   The model flux is scaled to match the V-band photometry (or J/K if V is missing/unreliable for the object type).

## Supported Catalogs

| Tier | Catalog | Reference | Description |
| :--- | :--- | :--- | :--- |
| **Primary** | **CALSPEC** | [STScI](https://archive.stsci.edu/hlsps/reference_atlases/cdbs/calspec/) | HST Flux Standards. |
| | **ESO Phase 3** | [ESO](http://archive.eso.org/scienceportal/home) | Generic reduced spectra. |
| **Secondary** | **STELIB** | [Le Borgne+ 2003](https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=III/232) | Optical (3200-9500 $\mathring{A}$). |
| | **IRTF** | [Rayner+ 2009](https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=J/ApJS/185/289) | Near-IR ($0.8-5.0 \mu m$). |
| | **Morel Standards** | [Morel+ 2011](https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=J/A%2BA/529/A75) | Optical standards. |
| | **Burnashev** | [Burnashev 1985](https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=III/202) | Spectrophotometry. |
| **Model** | **Pickles** | [Pickles 1998](https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=J/PASP/110/863) | Flux library. |

## Limitations

*   **Variability:** Queries return a single snapshot or average.
*   **Completeness:** Gaps may exist between instrument ranges.
