"""
Optical calculations.
"""

import numpy as np

from .constants import PLANCK
from .constants import LIGHT_SPEED
from .constants import NM
from .constants import AVOGADRO

def photon_energy(wavelength_nm):
    """
    Photon energy.

    Parameters
    ----------
    wavelength_nm : float or ndarray

    Returns
    -------
    Joule
    """

    wavelength = wavelength_nm * NM

    return PLANCK * LIGHT_SPEED / wavelength

def photons_per_joule(wavelength_nm):

    energy = photon_energy(wavelength_nm)

    return 1.0 / energy

def umol_per_joule(wavelength_nm):

    photons = photons_per_joule(wavelength_nm)

    return photons / AVOGADRO * 1e6

def beer_lambert(surface_irradiance,
                 kd,
                 depth):

    return surface_irradiance * np.exp(-kd * depth)
