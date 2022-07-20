# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 19:20:19 2021

@author: hinhi
"""

import numpy as np
from scipy import signal


def GreenFunction(size: int, cell_size: float) -> np.ndarray:
    """
        written by Cao Zhuo 
    """
    xx, yy, zz = np.mgrid[-size[0]+1:size[0], -
                          size[1]+1:size[1], -size[2]+1:size[2]]*cell_size
    r = np.sqrt(xx**2+yy**2+zz**2)
    g = np.divide(1, r, out=np.zeros_like(r), where=r != 0)

    return g


def CalGPE(density: np.ndarray, cell_size: float, G: float = 1) -> np.ndarray:
    """
        written by Cao Zhuo 
    """
    coeff = -cell_size ** 3
    size = density.shape

    GFunc = GreenFunction(size, cell_size)
    phi = coeff*signal.fftconvolve(density, GFunc, mode='same')

    # becareful of the value of G
    return G * phi
