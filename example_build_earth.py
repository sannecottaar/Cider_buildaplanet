# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.


'''
example_build_planet
--------------------

For Earth we have well-constrained one-dimensional density models.  This allows us to
calculate pressure as a funcion of depth.  Furthermore, petrologic data and assumptions
regarding the convective state of the planet allow us to estimate the temperature.

For planets other than Earth we have much less information, and in particular we
know almost nothing about the pressure and temperature in the interior.  Instead, we tend
to have measurements of things like mass, radius, and moment-of-inertia.  We would like
to be able to make a model of the planet's interior that is consistent with those
measurements.

However, there is a difficulty with this.  In order to know the density of the planetary
material, we need to know the pressure and temperature.  In order to know the pressure,
we need to know the gravity profile.  And in order to the the gravity profile, we need
to know the density.  This is a nonlinear problem which requires us to iterate to find
a self-consistent solution.

Here we show an example that does this, using the planet Mercury as motivation.


*Uses:*

* :doc:`mineral_database`
* :class:`burnman.composite.Composite`
* :func:`burnman.material.Material.evaluate`
'''
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('burnman-0.9.0'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

import burnman
import burnman.minerals as minerals
from build_planet import *


# Here we actually do the interation.  We make an instance
# of our Earth planet, then call generate_profiles.
# Emprically, 300 slices and 5 iterations seem to do
# a good job of converging on the correct profiles.
n_slices = 300
n_iterations = 5
earth = Earth(n_slices)
earth.generate_profiles(n_iterations)

# These are the actual observables
# from the model, that is to say,
# the total mass of the planet and
# the moment of inertia factor,
# or C/MR^2
observed_mass = 5.95e24
observed_moment = 0.33  # From Margot. et al, 2012

print(("Total mass of the planet: %.2e, or %.0f%% of the observed mass" %
      (earth.mass, earth.mass / observed_mass * 100.)))
print(("Moment of inertia factor of the planet: %.3g, or %0.f%% of the observed factor" %
      (earth.moment_of_inertia_factor, earth.moment_of_inertia_factor / observed_moment * 100.)))

# As we can see by running this, the calculated mass of the planet is much too large.
# One could do a better job of fitting this by using a more complicated interior model,
# with a liquid outer core, light alloying elements in the core, and a more realistic
# temperature profile.  That, however, is outside of the scope of this
# example.

import matplotlib.gridspec as gridspec

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\usepackage{relsize}'
plt.rc('font', family='sans-serif')

# Come up with axes for the final plot
figure = plt.figure(figsize=(12, 10))
ax1 = plt.subplot2grid((5, 3), (0, 0), colspan=3, rowspan=3)
ax2 = plt.subplot2grid((5, 3), (3, 0), colspan=3, rowspan=1)
ax3 = plt.subplot2grid((5, 3), (4, 0), colspan=3, rowspan=1)

# Plot density, vphi, and vs for the planet.
ax1.plot(earth.radii / 1.e3, earth.densities /
         1.e3, label=r'$\rho$', linewidth=2.)
ax1.plot(earth.radii / 1.e3, earth.bulk_sound_speed /
         1.e3, label=r'$V_\phi$', linewidth=2.)
ax1.plot(earth.radii / 1.e3, earth.shear_velocity /
         1.e3, label=r'$V_S$', linewidth=2.)

# Also plot a black line for the CMB
ylimits = [3., 13.]
ax1.plot([earth.cmb / 1.e3, earth.cmb / 1.e3], ylimits, 'k', linewidth=6.)

ax1.legend()
ax1.set_ylabel("Velocities (km/s) and Density (kg/m$^3$)")

# Make a subplot showing the calculated pressure profile
ax2.plot(earth.radii / 1.e3, earth.pressures / 1.e9, 'k', linewidth=2.)
ax2.set_ylabel("Pressure (GPa)")

# Make a subplot showing the calculated gravity profile
ax3.plot(earth.radii / 1.e3, earth.gravity, 'k', linewidth=2.)
ax3.set_ylabel("Gravity (m/s$^2)$")
ax3.set_xlabel("Radius (km)")

plt.show()
