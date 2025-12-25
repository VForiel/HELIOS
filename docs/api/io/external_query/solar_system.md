# Solar System

Query logic for Solar System objects.

```{eval-rst}
.. autofunction:: helios.io.get_solar_system_properties
   :noindex:
```

## Data Sources

*   **Ephemerides:** **JPL Horizons** (via `astroquery.jplhorizons`) provides precise positions (RA/Dec) and distances (Delta, r) for any given epoch.
*   **Solar Flux:** **CALSPEC** Sun reference (`sun_reference_stis_002.fits`) is used to model the reflected light from planets/moons (using proper albedo and phase functions).
