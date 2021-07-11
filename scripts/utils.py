import numpy as np
from astropy.cosmology import Planck18
from astropy.constants import M_sun
import astropy.units as u

critical_density = Planck18.critical_density0.value * 1000


def mass_at_radius(
    r,
    gamma,
    phi0,
    full=False,
):
    """Calculate the mass at a given radius given some posterior samples with keys `p_gamma`
    (gamma, the power-law slope of the gravitational potential) and `p_phi0` (phi0, the scale
    factor of the gravitational potential). If full=True, then an array of masses, one for
    each posterior draw, is calculated. Otherwise, various quantiles (2.5%, 12.5%, 25%, 50%,
    75%, 87.5%, and 97.5%) are returned.

    Input:
        r: float
        samples: a dictionary-like object with keys `p_gamma` and `p_phi0`

    Output:
        if full=True, then an array of masses at the given radius
        if full=False, then an array of various quantiles of the mass at the given radius
    """
    masses = phi0 * gamma * 2.325e-3 * r ** (1 - gamma)
    if full:
        return masses
    return np.quantile(masses, [0.025, 0.125, 0.25, 0.5, 0.75, 0.875, 0.975])


def virial_radius(gamma: float, phi0: float) -> float:
    """Calculates the radius (in kpc) for which a sphere has a mean matter density 200 times the critical density."""
    return (
        4
        / 3
        * np.pi
        * (critical_density * 200)
        * (u.kpc.to(u.m)) ** 3
        / (phi0 * gamma * 2.325e-3 * 1e12 * M_sun.value)
    ) ** (1 / (-gamma - 2))
