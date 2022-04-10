"""
    Calculate orientation of the given sample.
    
    Author: Hinny Tsang
    Last Edit: 2022-04-11
"""

import numpy as np
import math
import scipy.stats as st
from typing import Union, List, Tuple, TypeVar

T = TypeVar('T')
Array = Union[List[T], np.ndarray]


# SVD
def cov(x, y, p, pTot):
    EX = np.sum(x*p)/pTot
    EY = np.sum(y*p)/pTot
    covXY = np.sum(p*(x-EX)*(y-EY))/pTot
    return covXY


def var(x, p, pTot):
    EX = np.sum(x*p)/pTot
    varX = np.sum(p*(x-EX)**2)/pTot
    return varX


def cov_matrix(x, y, p):
    pTot = np.sum(p)
    varX = var(x, p, pTot)
    varY = var(y, p, pTot)
    covXY = cov(x, y, p, pTot)
    return np.array([[varX, covXY], [covXY, varY]])


def calc_weighted_orientation_axis(x: np.ndarray, y: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray]:
    """
        princinple components weighted by density

        :param x: x data
        :param y: y data
        :param p: probability
        Returns:
        [variance1, variance2], [vector1[2],vector2[2]]
    """
    covMatrix = cov_matrix(x, y, p)
    val, vec = np.linalg.eig(covMatrix)
    vec1 = vec[:, 0]
    vec2 = vec[:, 1]

    if val[0] >= val[1]:
        return val, [vec1, vec2]
    else:
        return val[::-1], [vec2, vec1]
