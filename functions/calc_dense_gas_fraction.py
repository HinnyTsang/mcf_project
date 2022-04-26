"""
    Calculate some physics quantities
    
    Author: Hinny Tsang
    Last Edit: 2022-04-06
"""

import copy
import numpy as np
import h5py
from typing import List, Tuple
import functions.calc_unit_conversion as uc


def calc_den_weighted_mean_b(
    den: np.ndarray, bx: np.ndarray, by: np.ndarray, bz: np.ndarray
) -> Tuple[float, ...]:
    """
        :param den: density.
        :param bx: magnetic field x component.
        :param by: magnetic field y component.
        :param bz: magnetic field z component.
        :return: tubple of (bx, by, bz, b)
    """
    bx_sum = np.sum(den*bx)
    by_sum = np.sum(den*by)
    bz_sum = np.sum(den*bz)

    den_sum = np.sum(den)

    bx_sum /= den_sum
    by_sum /= den_sum
    bz_sum /= den_sum

    return bx_sum, by_sum, bz_sum, (bx_sum**2 + by_sum**2 + bz_sum**2)**0.5


def calc_magnetic_critial_den(b: float) -> float:
    """
        :param b: magnetic field in muG
        :return: density in Msun.pc-2
    """

    # critical density in NH
    crit_den = b/1.1e-21

    # convert to Msun.pc-2
    crit_den = uc.number_den_H2_per_cm2_to_column_den_Msun_per_pc2(
        crit_den, 1.37)

    return crit_den
