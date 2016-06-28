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
sys.path.insert(1, os.path.abspath('burnman-0.9.0'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

import burnman
import burnman.minerals as minerals


# gravitational constant
G = 6.67e-11


## For temperature the Anderson geotherm is used. There are different option in BurnMan, but this is a entire Earth one. Try to add a temperature anomaly, and look at the effect. 
table_anderson = burnman.tools.read_table("input_geotherm/anderson_82.txt")
table_anderson_depth = np.array(table_anderson)[:, 0]
table_anderson_temperature = np.array(table_anderson)[:, 1]
prem = burnman.seismic.PREM()
def anderson(pressure):
    """
    Geotherm from :cite:`anderson1982earth`.

    Parameters
    ----------
    pressure : list of floats
        The list of pressures at which to evaluate the geotherm. :math:`[Pa]`

    Returns
    -------
    temperature : list of floats
        The list of temperatures for each of the pressures. :math:`[K]`
    """
    temperature = np.empty_like(pressure)
    for i in range(len(pressure)):
        try:
            depth = prem.depth(pressure[i])
        except:
            depth = 6371.e3 # in case pressure are higher than the center of the earth. This can occur during the iterations
        temperature[i] = burnman.tools.lookup_and_interpolate(
            table_anderson_depth, table_anderson_temperature, depth)
    return temperature



# Create a class for Planet that will do the heavy lifting.
class Planet(object):

    def __init__(self, n_slices):
        # The constructor takes the number of depth slices which will
        # be used for the calculation.  More slices will generate more
        # accurate profiles, but it will take longer.

        self.cmb = None
        self.outer_radius = None

        self.radii = np.linspace(
            0.e3, self.outer_radius, n_slices)  # Radius list
        self.pressures = np.linspace(
            35.0e9, 0.0, n_slices)  # initial guess at pressure profile
        self.temperatures = np.ones_like(
            self.pressures) * 1000.  # Assume an isothermal interior (not a great assumption, but we do this for simplicity's sake).

        self.upper_mantle = None
        self.lower_mantle = None
        self.core = None

    def generate_profiles(self, n_iterations):
        # Generate the density, gravity, pressure, vphi, and vs profiles for the planet.
        # The n_iterations parameter sets how many times to iterate over density, pressure,
        # and gravity.  Empirically, five-ish iterations is adequate.  After the iterations,
        # this also calculates mass and moment of inertia of the planet.

        for i in range(n_iterations):
            self.temperatures = anderson(self.pressures )
            self.densities, self.bulk_sound_speed, self.shear_velocity = self._evaluate_eos(
                self.pressures, self.temperatures, self.radii)
            self.gravity = self._compute_gravity(
                self.densities, self.radii)
            self.pressures = self._compute_pressure(
                self.densities, self.gravity, self.radii)

        self.mass = self._compute_mass(self.densities, self.radii)
        self.moment_of_inertia = self._compute_moment_of_inertia(
            self.densities, self.radii)
        self.moment_of_inertia_factor = self.moment_of_inertia / \
            self.mass / self.outer_radius / self.outer_radius

    def _evaluate_eos(self, pressures, temperatures, radii):
        # Evaluates the equation of state for each radius slice of the model.
        # Returns density, bulk sound speed, and shear speed.

        rho = np.empty_like(radii)
        bulk_sound_speed = np.empty_like(radii)
        shear_velocity = np.empty_like(radii)



        for i in range(len(radii)):
            density = vs = vphi = 0.
            if radii[i] > self.cmb:
                
                if radii[i] > self.tz:
                    density, vs, vphi = self.upper_mantle.evaluate(
                                                 ['density', 'v_s', 'v_phi'], [pressures[i]], [temperatures[i]])
                else:
                    density, vs, vphi = self.lower_mantle.evaluate(
                                                ['density', 'v_s', 'v_phi'], [pressures[i]], [temperatures[i]])
            else:
                density = self.core.density(6371.e3-radii[i]) # taken from PREM
                vphi = self.core.v_phi(6371.e3-radii[i]) # taken from PREM
                vs = self.core.v_s(6371.e3-radii[i]) # taken from PREM
            rho[i] = density
            bulk_sound_speed[i] = vphi
            shear_velocity[i] = vs

        return rho, bulk_sound_speed, shear_velocity

    def _compute_gravity(self, density, radii):
        # Calculate the gravity of the planet, based on a density profile.  This integrates
        # Poisson's equation in radius, under the assumption that the planet is laterally
        # homogeneous.

        # Create a spline fit of density as a function of radius
        rhofunc = UnivariateSpline(radii, density)

        # Numerically integrate Poisson's equation
        poisson = lambda p, x: 4.0 * np.pi * G * rhofunc(x) * x * x
        grav = np.ravel(odeint(poisson, 0.0, radii))
        grav[1:] = grav[1:] / radii[1:] / radii[1:]
        grav[
            0] = 0.0  # Set it to zero a the center, since radius = 0 there we cannot divide by r^2
        return grav

    def _compute_pressure(self, density, gravity, radii):
        # Calculate the pressure profile based on density and gravity.  This integrates
        # the equation for hydrostatic equilibrium  P = rho g z.

        # convert radii to depths
        depth = radii[-1] - radii

        # Make a spline fit of density as a function of depth
        rhofunc = UnivariateSpline(depth[::-1], density[::-1])
        # Make a spline fit of gravity as a function of depth
        gfunc = UnivariateSpline(depth[::-1], gravity[::-1])

        # integrate the hydrostatic equation
        pressure = np.ravel(
            odeint((lambda p, x: gfunc(x) * rhofunc(x)), 0.0, depth[::-1]))
        return pressure[::-1]

    def _compute_mass(self, density, radii):
        # Returns a list of moments of inertia of the planet [kg m^2]
        rhofunc = UnivariateSpline(radii, density)
        mass = quad(lambda r: 4 * np.pi * rhofunc(r) * r * r,
                    radii[0], radii[-1])[0]
        return mass

    def _compute_moment_of_inertia(self, density, radii):
        # Returns the moment of inertia of the planet [kg m^2]

        rhofunc = UnivariateSpline(radii, density)
        moment = quad(
            lambda r: 8.0 / 3.0 * np.pi * rhofunc(r) * r * r * r * r,
            radii[0], radii[-1])[0]
        return moment


