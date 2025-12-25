
"""
HELIOS I/O Module
=================

Provides simplified access to external query functions for astronomical objects.
"""

from helios.io.external_query.stars.query_all import get_star_properties
from helios.io.external_query.exoplanets.query_all import get_exoplanet_properties
from helios.io.external_query.solar_system.query_all import get_solar_system_properties

__all__ = [
    "get_star_properties",
    "get_exoplanet_properties",
    "get_solar_system_properties"
]
