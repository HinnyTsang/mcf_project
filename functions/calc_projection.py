"""
    Calculate projections of a given sample
    
    Author: Hinny Tsang
    Last Edit: 2022-04-11
"""

import numpy as np
import math
import scipy.stats as st
from typing import Union, List, Tuple, TypeVar

from sklearn.metrics import multilabel_confusion_matrix
import functions.calc_mcf as calc_mcf
import functions.calc_orientation as calc_ori

T = TypeVar('T')
Array = Union[List[T], np.ndarray]


# Calculate uniform distribution by Fibonacci Sphere
def fibonacci_sphere(n: int) -> Tuple[np.ndarray]:
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
def random_unit_vector(n: int) -> Tuple[np.ndarray]:
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
def rotational_matrix(axis: Array[float], theta: float) -> np.ndarray:
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
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    rot_x: float, rot_y: float, rot_z: float
) -> Tuple[np.ndarray]:
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


def reduce_size(data: np.ndarray, n: int) -> np.ndarray:
    """
        smoothing the data buy the mean of n*n grids.
        :param data:  array of data
        :param n:     beam size
        :return:      smoothed data. 
    """
    
    slices = [data[i::n, j::n] for i in range(n) for j in range(n)]
    
    return np.mean([slices[0]], 0)



# Project sample in the given Vector.
def projection(
    den: np.ndarray, bx: np.ndarray, by: np.ndarray, bz: np.ndarray,
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
) -> Tuple:
    """
        Similar code in 'get_projection_mpi.py, but less parameters are calculated

        :param den:    1D array of density
        :param bx:     1D array of magnetic field x
        :param by:     1D array of magnetic field y
        :param bz:     1D array of magnetic field z
        :param x:      1D array of x cooridinates
        :param y:      1D array of y cooridinates
        :param z:      1D array of z cooridinates
        
        :return:       Tuple of physicals paramters: 
                       (
                        cloud_orientation, b_field_orientation,
                        mcf, mcf_bin, mcf_slope, mcf_area,
                        aspect_ratio, cloud_mass, min_den, max_den,
                        binned_den, binned_stoke_Q, binned_stoke_U
                       )
    """

    # Defination of cloud mass > 2Av
    cloud_threshold = 30

    # # TODO 1 Rotate data py lOS ###########################################
    # xRot, yRot, zRot = rotate_3d(x, y, z, *los)
    # BxRot, ByRot, BzRot = rotate_3d(bx, by, bz, *los)
    # #######################################################################

    # TODO Prepare Stoke Parameters for projection ########################
    stoke_I, stoke_Q, stoke_U, stoke_phi = calcStoke(bx, by)
    stoke_phi = None
    # weight the stoke parameters by density
    weighted_stoke_Q = stoke_Q * den
    weighted_stoke_U = stoke_U * den
    #######################################################################

    # TODO Create bins for projections ####################################
    # Create bins data, noteded that the BINS ARE KNOWN!!!.
    # cell center cooridinate.
    box_length = 20
    box_grids = 960
    dx = box_length/box_grids
    x_ = np.linspace(-box_length/2+dx/2, box_length/2-dx/2, box_grids)
    y_ = x_.copy()
    X, Y = np.meshgrid(x_, y_)

    # boundary cooridinates for binning !!!!!!
    binX = np.linspace(-10, 10, 961)
    binY = binX.copy()
    #######################################################################

    # TODO Project the data into 2D #######################################
    # Density projections
    binned_den = st.binned_statistic_2d(
        y, x, den, statistic="sum", bins=[binX, binY])[0]

    # Stokes parameters projections (weighted by density)
    binned_stoke_Q = st.binned_statistic_2d(
        y, x, weighted_stoke_Q, statistic="sum", bins=[binX, binY])[0]
    binned_stoke_U = st.binned_statistic_2d(
        y, x, weighted_stoke_U, statistic="sum", bins=[binX, binY])[0]
    # Divide by the column density.
    binned_stoke_Q = np.divide(binned_stoke_Q, binned_den, out=np.zeros_like(
        binned_stoke_Q), where=binned_den != 0)
    binned_stoke_U = np.divide(binned_stoke_U, binned_den, out=np.zeros_like(
        binned_stoke_U), where=binned_den != 0)

    # Convert into column density
    binned_den *= dx
    #######################################################################

    # TODO Calculate connected structure, not usefull in current state. ###
    # connectedIndex = cp.connectedStructure(binned_den)
    valid_index = np.where(binned_den > 0)
    #######################################################################
    # turn binned data into cooridinates.
    x_points = X[valid_index]
    y_points = Y[valid_index]
    den_points = binned_den[valid_index]

    stoke_U_points = binned_stoke_U[valid_index]
    stoke_Q_points = binned_stoke_Q[valid_index]
    #######################################################################

    # TODO Calculate cloud mass ###########################################
    # Calculate mass above star formation threshold
    cloud_mass = np.sum(binned_den[binned_den >= cloud_threshold])*dx**2
    #######################################################################

    # TODO Density weighted stoke parameters ##############################
    stoke_phi_weighted_2D = calcStoke_Phi(np.sum(stoke_U_points*den_points)/np.sum(den_points),
                                          np.sum(stoke_Q_points*den_points)/np.sum(den_points))

    b_field_orientation = stoke_phi_weighted_2D
    #######################################################################

    # TODO PCA cloud orientation. #########################################
    val, vec = calc_ori.calc_weighted_orientation_axis(
        x_points, y_points, den_points)
    majVec = vec[0]*np.sqrt(val[0])  # major axis
    # minVec = vec[1]*np.sqrt(val[1]) # minor axis (not useful)
    aspect_ratio = np.sqrt(val[0]/val[1])
    cloud_orientation = calcStoke(*majVec)[3]
    #######################################################################

    # TODO MCF calculation ################################################
    bins = np.linspace(0, 150000, int(10e5)+1) * dx
    mcf, mcf_bin = calc_mcf.calc_mcf(den_points, bins, 0)
    mcf_slope, mcf_area, den_10 = calc_mcf.calc_mcf_slope_and_area(mcf, mcf_bin)
    #######################################################################

    # TODO Maximum and minimum of column density ##########################
    min_den = np.min(den_points)
    max_den = np.max(den_points)
    #######################################################################

    return cloud_orientation, b_field_orientation, mcf, mcf_bin, mcf_slope, mcf_area, \
        aspect_ratio, cloud_mass, min_den, max_den, binned_den, binned_stoke_Q, binned_stoke_U


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
