# External Query Module

## Philosophy: Real Data Only
This module is responsible for retrieving, processing, and caching astronomical data from external sources (NASA, ESO, JPL, etc.).

**Crucially, the cache (`cache/`) must ONLY contain data derived from real observations or official standard references (like ASTM E-490 for the Sun).**

### Rules
1.  **No Synthetic Generation**: We do NOT generate synthetic spectra (e.g., Blackbody curves) to fill gaps in the cache. If real data is unavailable for an object, the cache remains empty or partial for that specific property.
2.  **Traceability**: Every cached file should ideally have a `source` field in its metadata indicating origin (e.g., "ASTM E-490", "JPL Horizons", "VPL").
3.  **Processing**: "Real Data" can be processed (units converted, interpolated to a standard grid) to fit the Helios data model, but it must be based on actual measurements.

## Data Pipeline
1.  **Query**: Request data from an external API or download a specific catalog file.
2.  **Process**:
    *   Parse the raw format (CSV, FITS, JSON).
    *   Convert units to Helios standards (e.g., `um` for wavelength, `Jy` for flux density).
    *   Clean/Filter invalid data points.
3.  **Cache**: Save the processed Result to a JSON file in the local `cache/` directory.
4.  **Load**: Subsequent requests load directly from JSON.

## Supported Objects & Sources
*   **Sun**: ASTM E-490 Standard Extraterrestrial Spectrum (NREL).
*   **Planets**:
    *   *Earth*: Earthshine / VPL Data (Implementation in progress).
    *   *Jupiter*: Karkoschka (1994) Albedo (Implementation in progress).
    *   *Others*: Currently no spectral data cached (User must provide or we return empty SED).
*   **Stars**: SIMBAD / VizieR Photometry.
