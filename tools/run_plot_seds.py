import sys
sys.path.insert(0, 'src')
from astropy import units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from helios.components import Scene, Star, Planet

scene = Scene(distance=10*u.pc)
star = Star(temperature=5700*u.K, magnitude=5, mass=1*u.M_sun, position=(0*u.AU, 0*u.AU))
planet = Planet(mass=1*u.M_jup, position=(1*u.AU, 0*u.AU))
scene.add(star)
scene.add(planet)

# Use plot_sed but do not show; inspect ax lines
ax = star.plot_sed(color='gold', label='Star')
ax = planet.plot_sed(ax=ax, color='blue', label='Planet')

print('Axes lines count:', len(ax.lines))
for line in ax.lines:
    print('Line label:', line.get_label(), 'num points:', len(line.get_xdata()))

# Save figure to file to ensure no display required
plt.savefig('g:/HELIOS/tools/test_sed_plot.png')
print('Saved test plot to tools/test_sed_plot.png')
