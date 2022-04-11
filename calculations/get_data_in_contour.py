"""
    Calculate the maximum contour from the scorpio output file.
    
    Author: Hinny Tsang
    Last Edit: 2022-04-10
"""
import copy
import numpy as np
import h5py
from typing import List


def calc_connected_structure_3d(
    data: np.ndarray,
    start_i: int, start_j: int, start_k: int, threshold: float,
    indices: any, min_threshold: float
) -> tuple(List[int], List(int), List(int), np.ndarray):
    """

        given bondory index, threshold, connected indices.
        return new bondoury.

        :param data: 2d array
        :param start_i: start i index
        :param start_j: start j index
        :param threshold: contour value
        :indices: bit mask.
    """

    i_bfs = copy.deepcopy(start_i)
    j_bfs = copy.deepcopy(start_j)
    k_bfs = copy.deepcopy(start_k)

    temp = indices.copy()

    # indices for bondoury
    i_bdry, j_bdry, k_bdry = [], [], []

    # index for neibhour
    neib = [-1, 0, 0, 1, 0, 0, -1, 0]

    # bfs-find the connected structure in the current threshold.
    while len(i_bfs) > 0:

        # next iteration
        i_bfs_next = []
        j_bfs_next = []
        k_bfs_next = []

        # loop through all elements in bfs
        for i, j, k in zip(i_bfs, j_bfs, k_bfs):

            # counter for neibhour that is within threshold
            neib_in_threshold = 0

            # neibhour
            for l in range(6):

                # index of each neibhoud
                a = i + neib[l]
                b = j + neib[l+1]
                c = k + neib[l+2]

                # boundary conditions
                if 0 <= a < data.shape[0] and 0 <= b < data.shape[1] and 0 <= c < data.shape[2]:

                    # check if it is allowed in the current bondory
                    if data[a, b, c] < threshold:
                        continue

                    # current neibour is in the connected structure.
                    neib_in_threshold += 1

                    # already checked.
                    if temp[a, b, c]:
                        continue

                    # new index finded, add to the next iteration and update the mask.
                    temp[a, b, c] = 1
                    i_bfs_next += [a]
                    j_bfs_next += [b]
                    k_bfs_next += [c]

            # current nodes is in boundary
            if neib_in_threshold < 6:

                # if all neibhour is below the minimum threshold, don't need to included in the bondary.
                # neibhour
                neib_below_min = 0
                for l in range(6):

                    # index of each neibhoud
                    a = i + neib[l]
                    b = j + neib[l+1]
                    c = k + neib[l+2]

                    # boundary conditions
                    if 0 <= a < data.shape[0] and 0 <= b < data.shape[1] and 0 <= c < data.shape[2]:

                        # check if it is allowed in the current bondory
                        if data[a, b, c] < min_threshold:
                            neib_below_min += 1

                # there are some hopeful neibhour.
                if neib_in_threshold < 6:
                    i_bdry += [i]
                    j_bdry += [j]
                    k_bdry += [k]

        # update for next iteration of bfs.
        i_bfs = i_bfs_next
        j_bfs = j_bfs_next
        k_bfs = k_bfs_next

    return i_bdry, j_bdry, k_bdry, temp


def get_boundary_index_3d(
    data: np.ndarray, threshold: float
) -> tuple(np.ndarray, float):
    """
        Naively calculate the contour by the maximum boundary value (maximum value on the cube surface):

        :param data: 3d array
        :param threshold: contour value
        :return: (indices, threshold)
    """

    indices = np.where(data > threshold)

    return indices, threshold


def binary_search_bondory_3d(
    data: np.ndarray, min_threshold: float, max_threshold: float
) -> tuple(np.ndarray, float):
    """
    Binary search for the density threshold.

    :param data: 3d array with density
    :param threshold: initial value for checking
    :return: mask for the initial value.
    """

    ############################################################
    # index of maximum value (starting index of the core)
    # Change here if you need others starting point.
    i, j, k = np.unravel_index(data.argmax(), data.shape)
    ############################################################

    # binary search for the threshold.
    threshold = max_threshold

    # output bit map
    indices = np.zeros(data.shape)

    # if no points greater than the threshold value
    if (data.argmax() < threshold):
        return indices

    # bfs indices.
    i_bfs = [i]
    j_bfs = [j]
    k_bfs = [k]
    indices[i, j, k] = 1

    # boundary index only use when threshold update.
    i_bdry, j_bdry, k_bdry, indices = calc_connected_structure_3d(
        data, i_bfs, j_bfs, k_bfs, threshold, indices, min_threshold)

    if np.min(i_bdry) == 0 or np.min(j_bdry) == 0 or np.min(k_bdry) == 0 or\
            np.max(i_bdry) == data.shape[0]-1 or np.max(j_bdry) == data.shape[1]-1 or np.max(k_bdry) == data.shape[0]-1:
        print(f"maximum threshold {threshold} is too low.")
        return

    ##############################################################################
    # so far we calculated the boundary index,  next part doing binary search on the bondoury.
    ##############################################################################

    # for next iteration
    i_bdry_next = []
    j_bdry_next = []
    k_bdry_next = []

    # ending criteria:
    # boundary for two iteration is identical.
    while True:
        # binary search
        threshold = (min_threshold + max_threshold)/2

        print(f"searching for {threshold} ...")

        # bfs for the current threshold.
        i_bdry_next, j_bdry_next, k_bdry_next, indices_next = calc_connected_structure_3d(
            data, i_bdry, j_bdry, k_bdry, threshold, indices, min_threshold)

        if np.min(i_bdry_next) == 0 or np.min(j_bdry_next) == 0 or np.max(i_bdry_next) == data.shape[0]-1 or np.max(j_bdry_next) == data.shape[1]-1 or \
                np.min(k_bdry_next) == 0 or np.max(k_bdry_next) == data.shape[2]-1:
            # the threshold is too low.
            print(f"{threshold} is too low ")
            min_threshold = threshold

        else:
            # ending criteria.
            if i_bdry_next == i_bdry and j_bdry == j_bdry_next and k_bdry == k_bdry_next:
                break

            # update upper bound for binary seaarch.
            max_threshold = threshold

            # update results.
            indices = indices_next
            i_bdry = i_bdry_next
            j_bdry = j_bdry_next
            k_bdry = k_bdry_next

    return indices, threshold


def binary_search_bondory_by_mass_3d(
    data: np.ndarray, min_threshold: float, max_threshold: float, target_mass: float
) -> tuple(np.ndarray, float):
    """
    Binary search for the density threshold by mass.

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!IMPORTANT. the min_threshold must not touching the boundary!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    :param data: 3d array with density
    :param threshold: initial value for checking
    :param target_mass: target mass of the cloud
    :return: mask for the initial value.
    """

    ############################################################
    # index of maximum value (starting index of the core)
    # Change here if you need others starting point.
    i, j, k = np.unravel_index(data.argmax(), data.shape)
    ############################################################

    # binary search for the threshold.
    threshold = max_threshold

    # output bit map
    indices = np.zeros(data.shape)

    # if no points greater than the threshold value
    if (data.argmax() < threshold):
        return indices

    # bfs indices.
    i_bfs = [i]
    j_bfs = [j]
    k_bfs = [k]
    indices[i, j, k] = 1

    # boundary index only use when threshold update.
    i_bdry, j_bdry, k_bdry, indices = calc_connected_structure_3d(
        data, i_bfs, j_bfs, k_bfs, threshold, indices, min_threshold)

    curr_mass = np.sum(data[indices == 1])
    if curr_mass > target_mass:
        print(
            f"maximum threshold gives mass = {curr_mass:2f}, which is too low")
        return

    ##############################################################################
    # so far we calculated the boundary index,  next part doing binary search on the bondoury.
    ##############################################################################

    # for next iteration
    i_bdry_next = []
    j_bdry_next = []
    k_bdry_next = []

    # ending criteria:
    # boundary for two iteration is identical.
    while True:
        # binary search
        threshold = (min_threshold + max_threshold)/2

        print(f"searching for {threshold} ...")

        # bfs for the current threshold.
        i_bdry_next, j_bdry_next, k_bdry_next, indices_next = calc_connected_structure_3d(
            data, i_bdry, j_bdry, k_bdry, threshold, indices, min_threshold)

        # calculate the mass within the current contour
        mass_current = np.sum(data[indices_next == 1])
        print(f"mass for current contour is {mass_current:.2f}")

        # perfect! but impossible to happen
        if (mass_current == target_mass):
            break

        elif mass_current > target_mass:

            # ending criteria.
            if i_bdry_next == i_bdry and j_bdry == j_bdry_next and k_bdry == k_bdry_next:
                break

            # the threshold is too low.
            print(f"{threshold} is too low")
            min_threshold = threshold

            # don't keep the result since it is beyond.
        else:
            # ending criteria.
            if i_bdry_next == i_bdry and j_bdry == j_bdry_next and k_bdry == k_bdry_next:
                break

            # update upper bound for binary seaarch.
            print(f"{threshold} is too high")
            max_threshold = threshold

            # keep it.
            # update results.
            indices = indices_next
            i_bdry = i_bdry_next
            j_bdry = j_bdry_next
            k_bdry = k_bdry_next

    return indices, threshold


if __name__ == "__main__":

    files = ['./h5/g1040_0016.h5', './h5/g1041_9015.h5']

    # previous calculated thresholds are
    thresholds = [15.125809751451015, 32.608720645308495]

    # target mass for ending criteria
    #              886549208.11
    target_mass = [402855975.61866176128387451172]

    for file in files:
        print("Reading file %s" % file)

        # directly read Scorpio output data.
        with h5py.File(file, 'r') as data:
            # data dimension.
            nbuf = data['nbuf'][0].astype(int)
            nx, ny, nz = data['nMesh'][:].astype(int)

            # density
            density = data['den'][nbuf:(nz + nbuf),
                                  nbuf:(ny + nbuf), nbuf:(nx + nbuf)]
            # momentum
            momx = data['momx'][nbuf:(nz + nbuf),
                                nbuf:(ny + nbuf), nbuf:(nx + nbuf)]
            momy = data['momy'][nbuf:(nz + nbuf),
                                nbuf:(ny + nbuf), nbuf:(nx + nbuf)]
            momz = data['momz'][nbuf:(nz + nbuf),
                                nbuf:(ny + nbuf), nbuf:(nx + nbuf)]
            # magnetic field
            bx = 0.5*(data['bxr'][nbuf:(nz + nbuf), nbuf:(ny + nbuf), nbuf:(nx + nbuf)] +
                      data['bxl'][nbuf:(nz + nbuf), nbuf:(ny + nbuf), nbuf:(nx + nbuf)])
            by = 0.5*(data['byr'][nbuf:(nz + nbuf), nbuf:(ny + nbuf), nbuf:(nx + nbuf)] +
                      data['byl'][nbuf:(nz + nbuf), nbuf:(ny + nbuf), nbuf:(nx + nbuf)])
            bz = 0.5*(data['bzr'][nbuf:(nz + nbuf), nbuf:(ny + nbuf), nbuf:(nx + nbuf)] +
                      data['bzl'][nbuf:(nz + nbuf), nbuf:(ny + nbuf), nbuf:(nx + nbuf)])
            # cooridinates
            x = data['xc1'][nbuf:(nx + nbuf)]
            y = data['xc2'][nbuf:(ny + nbuf)]
            z = data['xc3'][nbuf:(nz + nbuf)]
            # time
            t = data['t'][:]

        # calculate cooridinates.
        Y, Z, X = np.meshgrid(y, z, x)
        index: np.ndarray = None
        threshold: float = None

        # # 8-4-2022 ########################################################
        # # start index.
        # i, j, k = np.unravel_index(density.argmax(), density.shape)
        # threshold = thresholds[1]
        # # index for saving the density
        # i_bdry, j_bdry, k_bdry, index = calc_connected_structure_3d(density, [i], [j], [k], threshold=threshold, indices=np.zeros_like(density), min_threshold=threshold)
        # ###################################################################

        # index for saving the density ####################################
        # use this line if needed to find the boundary.
        # index, threshold = binary_search_bondory_3d(density, 0, 129)
        ###################################################################

        # 9-4-2022 serch for threshold that gives target mass #############
        # use this line if needed to find the boundary.
        # index, threshold = binary_search_bondory_by_mass_3d(
        #     density, thresholds[1], 36, target_mass=target_mass[0])
        ###################################################################

        # take values by index.
        x_contour = X[index == 1]
        y_contour = Y[index == 1]
        z_contour = Z[index == 1]

        den_contour = density[index == 1]

        mx_contour = momx[index == 1]
        my_contour = momy[index == 1]
        mz_contour = momz[index == 1]

        bx_contour = bx[index == 1]
        by_contour = by[index == 1]
        bz_contour = bz[index == 1]

        # write output file.
        newFileNAME = file.replace(
            ".h5", "_data_in_contour_new_contour_by_mass.h5")

        with h5py.File(newFileNAME, 'w-') as writeData:

            writeData.create_dataset('x', data=x_contour)
            writeData.create_dataset('y', data=y_contour)
            writeData.create_dataset('z', data=z_contour)
            writeData.create_dataset('density', data=den_contour)
            writeData.create_dataset('momx', data=mx_contour)
            writeData.create_dataset('momy', data=my_contour)
            writeData.create_dataset('momz', data=mz_contour)
            writeData.create_dataset('bx', data=bx_contour)
            writeData.create_dataset('by', data=by_contour)
            writeData.create_dataset('bz', data=bz_contour)
            writeData.create_dataset('threshold', data=threshold)
            writeData.create_dataset('t', data=t)
            writeData.close()
    F
