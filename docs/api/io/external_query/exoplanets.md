# Exoplanets

Query logic for exoplanetary systems.

```{eval-rst}
.. autofunction:: helios.io.get_exoplanet_properties
   :noindex:
```

## Data Sources

*   **NASA Exoplanet Archive:** Accessed via TAP/VO to retrieve:
    *   Planet Radius ($R_p$)
    *   Planet Mass ($M_p$)
    *   Equilibrium Temperature ($T_{eq}$)
    *   Orbital Semi-major Axis ($a$)
    *   Orbital Period ($P$)

## Note
Spectra for exoplanets are currently generated synthetically (Blackbody or Reflective Model) in the `Sim` layer, as few observed reflection spectra are available in public standardized archives.
