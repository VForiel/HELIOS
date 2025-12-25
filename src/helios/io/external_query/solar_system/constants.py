
from astropy import units as u

# Static Properties for Major Bodies (Fallback/Reference)
# Added Albedo for reflection modeling
SOLAR_SYSTEM_DATA = {
    'Sun': {'id': '10', 'mass': 1.989e30 * u.kg, 'radius': 696340 * u.km, 'teff': 5778 * u.K, 'albedo': 0.0},
    'Mercury': {'id': '199', 'mass': 3.301e23 * u.kg, 'radius': 2439.7 * u.km, 'albedo': 0.142, 'teff': 440 * u.K}, # Hot side approx
    'Venus': {'id': '299', 'mass': 4.867e24 * u.kg, 'radius': 6051.8 * u.km, 'albedo': 0.77, 'teff': 737 * u.K},
    'Earth': {'id': '399', 'mass': 5.972e24 * u.kg, 'radius': 6371.0 * u.km, 'albedo': 0.306, 'teff': 288 * u.K},
    'Mars': {'id': '499', 'mass': 6.417e23 * u.kg, 'radius': 3389.5 * u.km, 'albedo': 0.25, 'teff': 210 * u.K},
    'Jupiter': {'id': '599', 'mass': 1.898e27 * u.kg, 'radius': 69911 * u.km, 'albedo': 0.52, 'teff': 165 * u.K}, # Internal heat significant
    'Saturn': {'id': '699', 'mass': 5.683e26 * u.kg, 'radius': 58232 * u.km, 'albedo': 0.47, 'teff': 134 * u.K},
    'Uranus': {'id': '799', 'mass': 8.681e25 * u.kg, 'radius': 25362 * u.km, 'albedo': 0.51, 'teff': 76 * u.K},
    'Neptune': {'id': '899', 'mass': 1.024e26 * u.kg, 'radius': 24622 * u.km, 'albedo': 0.41, 'teff': 72 * u.K},
}
