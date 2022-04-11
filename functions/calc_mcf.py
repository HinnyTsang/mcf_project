"""
    Calculate MCF from a given sample
    
    Author: Hinny Tsang
    Last Edit: 2022-04-11
"""

from multiprocessing.dummy import Array
import numpy as np
import scipy.stats as st
from typing import Union, List, Tuple, TypeVar


# Calculate the MCF
def calc_mcf(density: np.ndarray, bins: np.ndarray, threshold: float, dx: float = 10/480) -> Tuple[np.ndarray]:
    """
        Calculate MCF of the density profile

    """

    binsMark = 0.5*(bins[1:] + bins[:-1])

    binData = st.binned_statistic(density, density, statistic='sum', bins=bins)

    # Mass distribution function
    MDF = binData[0] * dx**2

    MDFCutOff = MDF[binsMark >= threshold]
    binsCutOff = binsMark[binsMark >= threshold]

    # Accumulate the MDF -> MCF
    MCF = np.add.accumulate(MDFCutOff)
    MCF = np.max(MCF)-MCF

    # Normalize
    MCF = MCF/np.max(MCF)

    return MCF, binsCutOff


def calc_mcf_slope_and_area(mcf: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculate the Slope of the first 90% of the MCF
    """
    n = len(bins)

    # TODO 1 Calculate the column density at MCF=0.1
    i = 0
    while i < n:

        # 0.1 = 90% cutoff line
        if mcf[i] > 0.1 and mcf[i+1] < 0.1:
            # TODO linear polation to calculate den = 0.1 Max
            # Slope
            M = (mcf[i]-mcf[i+1])/(bins[i]-bins[i+1])
            # y - y0         0.1 - y0
            # ------  = M -> --------- = M -> x = (0.1 - y0) / M + x0
            # x - x0           x - x0
            den10 = (0.1 - mcf[i])/M + bins[i]
            break

        i += 1

    # TODO 2 Calculate the slope of the MCF
    mcf_slope = np.abs(1/(den10-np.min(bins[mcf > 0])))

    # TODO 3 calculate the area inside the given range
    dbins = bins[1] - bins[0]
    area = np.sum(np.log10(mcf[mcf >= 0.1])-np.log10(0.1)) * dbins

    return mcf_slope, area, den10
