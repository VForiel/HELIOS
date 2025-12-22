# Splitters and Routing

Waveguide splitters and routing components.

```{eval-rst}
.. autoclass:: helios.components.YSplitter
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: helios.components.Swap
   :members:
   :undoc-members:
   :show-inheritance:
```

## Description

**YSplitter**: A standard Y-junction 1x2 splitter that divides the input amplitude equally (50/50 power split).

**Swap**: A logical component that reorders wavefronts in a multi-wavefront pipeline (e.g. waveguide crossings). It permutes the list of wavefronts according to a provided mapping.