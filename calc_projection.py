"""
    Calculate projections of a given sample
    
    Author: Hinny Tsang
    Last Edit: 2022-04-11
"""

import numpy as np
import math
import scipy.stats as st
from typing import Union, List


# Calculate uniform distribution by Fibonacci Sphere
def fibonacci_sphere(n: int) -> tuple(np.ndarray, np.ndarray, np.ndarray):
    """
        Generate xyz points with fibonacci sphere algorithm
        :param n: number of samples
        :return: tuple of three components of vectors
    """
    X = []
    Y = []
    Z = []

    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        X = X+[x]
        Y = Y+[y]
        Z = Z+[z]

    return np.array(X), np.array(Y), np.array(Z)


# Calculate random unit vector in 3D.
def random_unit_vector(n: int) -> tuple(np.ndarray, np.ndarray, np.ndarray):
    """
        Muller, Marsaglia ('Normalised Gaussians')
        :param n: number of samples
        :return: tuple of three components of vectors
    """
    u = np.random.normal(0, 1, n)
    v = np.random.normal(0, 1, n)
    w = np.random.normal(0, 1, n)
    norm = (u*u + v*v + w*w)**(0.5)
    u /= norm
    v /= norm
    w /= norm

    return u, v, w


# rotation
def rotational_matrix(axis: Union(List[float], np.ndarray), theta: float) -> np.ndarray:
    """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        ER- formula

        :param axis: rotational axis
        :param theta: rotational angle along the axis
        :return: tuple of three components of vectors
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotate_3d(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, rot_x: float, rot_y: float, rot_z: float
) -> tuple(np.ndarray, np.ndarray, np.ndarray):
    """
        Rotate data x, y, z with the identical rotational matrix for rot_i rotated into the z axis.
    """
    z_axis = [0, 0, 1]

    axis = np.cross([rot_x, rot_y, rot_z], z_axis)  # Axis of rotation

    if (np.sqrt(np.sum(axis**2)) == 0):
        # No rotation needed of los is identical to z_axis
        return x, y, z

    # angle of rotation
    theta = np.arccos(np.dot(z_axis, [rot_x, rot_y, rot_z]))

    fac_x, fac_y, fac_z = rotational_matrix(axis, theta)

    return fac_x[0]*x + fac_x[1]*y + fac_x[2]*z,\
        fac_y[0]*x + fac_y[1]*y + fac_y[2]*z,\
        fac_z[0]*x + fac_z[1]*y + fac_z[2]*z


def cart_to_sph(x, y, z):

    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)

    return theta, phi


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


def calcComponentsWeighted(x, y, p):
    """
        SVD

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


# Project sample in the given Vector.
def calcProjection(data, projVector):
    """
        Project data to given vector and bin in the 2d plane
    """

    newX, newY, newZ = rotate_3d(data['x'], data['y'], data['z'], *projVector)
    newBx, newBy, newBz = rotate_3d(
        data['bx'], data['by'], data['bz'], *projVector)

    dx = 1
    minCoor = np.min([data['x'], data['y'], data['z']])
    maxCoor = np.max([data['x'], data['y'], data['z']])
    BINS = np.arange(minCoor, maxCoor, int((maxCoor - minCoor)/0.5) + 1)

    binDen = st.binned_statistic_2d(
        newY, newX, data['den'], statistic='sum', bins=[BINS, BINS])[0] * dx
    binBx = st.binned_statistic_2d(
        newY, newX, newBx, statistic='mean', bins=[BINS, BINS])[0]
    binBy = st.binned_statistic_2d(
        newY, newX, newBy, statistic='mean', bins=[BINS, BINS])[0]
    binBz = st.binned_statistic_2d(
        newY, newX, newBz, statistic='mean', bins=[BINS, BINS])[0]

    for binData in [binDen, binBx, binBy, binBz]:
        binData[binData == 0] = np.nan

    return binDen, binBx, binBy, binBz


# Calculate the orientation through stoke parameters.
def calcStoke(vx, vy):
    """
    Calculate Stoke parameters IQU and phi.
    V are assumed to be zero for non circular polarization

    @Arguments:
        vx: x component 
        vy: y component 
        p: probability
    @Returns:
        I, Q, U, phi
    """
    I = vx**2 + vy**2

    # va = vx/np.sqrt(2) + vy/np.sqrt(2)
    # vb = -vx/np.sqrt(2) + vy/np.sqrt(2)

    Q = vx**2 - vy**2
    U = 2*vx*vy  # simplified

    phi = 0.5*np.arctan2(U, Q)
    return I, Q, U, phi


def calcStoke_Phi(U, Q):
    return 0.5*np.arctan2(U, Q)


# get the connected structure in the data
def connectedStructure(data):

    i, j = np.unravel_index(data.argmax(), data.shape)

    indices = np.zeros(data.shape)

    iToCheck = [i]
    jToCheck = [j]

    while len(iToCheck) > 0:
        nIToCheck = []
        nJToCheck = []
        for i, j in zip(iToCheck, jToCheck):
            # if (i < 0 or j < 0 or i >= data.shape[0] or j >= data.shape[0]):
            #     continue
            if indices[i][j] or data[i][j] == 0:
                continue
            else:

                indices[i][j] = 1
                nIToCheck += [i-1, i+1, i, i]
                nJToCheck += [j, j, j-1, j+1]

        iToCheck = nIToCheck
        jToCheck = nJToCheck

    return indices


# Calculate the MCF
def calcMCF(density, bins, threshold, dx=10/480):
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


def calcMCFSlopeAndArea(MCF, bins):
    """
    Calculate the Slope of the first 90% of the MCF
    """
    nBins = len(bins)

    # TODO 1 Calculate the column density at MCF=0.1
    i = 0
    while i < nBins:

        # 0.1 = 90% cutoff line
        if MCF[i] > 0.1 and MCF[i+1] < 0.1:

            # TODO linear polation to calculate den = 0.1 Max
            # Slope
            M = (MCF[i]-MCF[i+1])/(bins[i]-bins[i+1])
            # y - y0         0.1 - y0
            # ------  = M -> --------- = M -> x = (0.1 - y0) / M + x0
            # x - x0           x - x0

            den10 = (0.1 - MCF[i])/M + bins[i]
            break

        i += 1

    # TODO 2 Calculate the slope of the MCF
    MCFSlope = np.abs(1/(den10-np.min(bins[MCF > 0])))

    # TODO 3 calculate the area inside the given range
    dbins = bins[1] - bins[0]
    Area = np.sum(MCF[MCF >= 0.1]-0.1) * dbins

    return MCFSlope, Area
