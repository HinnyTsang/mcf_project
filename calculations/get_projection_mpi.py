from typing import Tuple
from xmlrpc.client import Boolean
from mpi4py import MPI
import h5py
import numpy as np

import sys
sys.path.append('..')
import functions.calc_mcf as calc_mcf
import functions.calc_projection as cp
import functions.calc_unit_conversion as uc
import functions.calc_orientation as calc_ori

from operator import pos
from os.path import exists







# initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


print(f"Number of threads is {size}.")


def read_data(file):
    with h5py.File(file, 'r') as data:
        myData = {key: data[key][()] for key in data.keys()}
    return myData


def main(file: str, n_rand: int, method: str) -> None:
    """
        Do projection for the given file.

        The file should be the output of 'get_data_in_contour.py'


        :param file: path to the h5 file.
        :param n_rand: number of projections.
        :param method: either be 'fib', or 'ran', define the way to do the projection.
    """

    # set random seed
    np.random.seed(0)

    if not exists(file):
        print("file {file} not exist.")
        return
    
    if method == "fib":
        print("generating uniform projection")
        x_rand_, y_rand_, z_rand_ = cp.fibonacci_sphere(n_rand)

    elif method == "ran":
        print("generating random projection")
        x_rand_, y_rand_, z_rand_ = cp.random_unit_vector(n_rand)

    else:
        print("arg should be either 'ran' or 'fib'.")
        return

    # number of projection per processor
    nRandPerProc = int(n_rand/size)

    # index of each processor.
    init = rank*nRandPerProc
    term = (rank+1)*nRandPerProc

    # assign los to different threads
    comm.Barrier()
    x_rand = x_rand_[init:term]
    y_rand = y_rand_[init:term]
    z_rand = z_rand_[init:term]

    # Read data

    # MPI read data.
    data = None

    if rank == 0:
        print(rank, f"Reading data from {file}")
        data = read_data(file)
        print(rank, "Finish reading data")
        for key in data.keys():
            print(key, type(data[key]))

    # boardcast data.
    comm.Barrier()
    data = comm.bcast(data, root=0)
    comm.Barrier()

    #         np.array(all_cloud_orientation), np.array(all_b_field_orientation), \
    #         np.array(all_mcf), np.array(all_mcf_bin), np.array(all_mcf_slope), np.array(all_mcf_area), \
    #         np.array(all_aspect_ratio), np.array(all_cloud_masses), \
    #         np.array(all_min_den), np.array(all_max_den), \
    #         np.array(binned_den), np.array(all_stoke_Q_map), np.array(all_stoke_U_map)
    cloud_orientation, b_orientation, \
        mcf, mcf_bin, mcf_slope, mcf_area, \
        aspect_ratio, cloud_mass, min_den, max_den, \
        binned_den, binned_stoke_Q, binned_stoke_U \
        = calc_all_projections(data, x_rand, y_rand, z_rand)


    # LOS 
    xRandRecvbuf = comm.gather(x_rand, root=0)
    yRandRecvbuf = comm.gather(y_rand, root=0)
    zRandRecvbuf = comm.gather(z_rand, root=0)
    # Orientation
    cloudThetaRecvbuf = comm.gather(cloud_orientation, root=0)
    posBThetaRecvbuf = comm.gather(b_orientation, root=0)
    # MCF
    MCFRecvbuf = comm.gather(mcf, root=0)
    MCFBinsRecvbuf = comm.gather(mcf_bin, root=0)
    MCFSlopeRecvbuf = comm.gather(mcf_slope, root=0)
    MCFAreaRecvbuf = comm.gather(mcf_area, root=0)
    # Cloud properties
    cloudAspectRecvbuf = comm.gather(aspect_ratio, root=0)
    cloudMassRecvbuf = comm.gather(cloud_mass, root=0)
    minDenRecvbuf = comm.gather(min_den, root=0)
    maxDenRecvbuf = comm.gather(max_den, root=0)
    # Projection map
    DensityRecvbuf = comm.gather(binned_den, root=0)
    StokeQRecvbuf = comm.gather(binned_stoke_Q, root=0)
    StokeURecvbuf = comm.gather(binned_stoke_Q, root=0)
    
    
    # Create output file.
    if rank == 0:
        # LOS
        xRandRecvbuf = np.reshape(xRandRecvbuf, n_rand)
        yRandRecvbuf = np.reshape(yRandRecvbuf, n_rand)
        zRandRecvbuf = np.reshape(zRandRecvbuf, n_rand)
        # Orientation
        cloudThetaRecvbuf = np.reshape(cloudThetaRecvbuf, n_rand)
        posBThetaRecvbuf = np.reshape(posBThetaRecvbuf, n_rand)
        # MCF 
        MCFRecvbuf = np.reshape(MCFRecvbuf, n_rand)
        MCFBinsRecvbuf = np.reshape(MCFBinsRecvbuf, n_rand)
        MCFSlopeRecvbuf = np.reshape(MCFSlopeRecvbuf, n_rand)
        MCFAreaRecvbuf = np.reshape(MCFAreaRecvbuf, n_rand)
        # Cloud properties
        cloudAspectRecvbuf = np.reshape(cloudAspectRecvbuf, n_rand)
        cloudMassRecvbuf = np.reshape(cloudMassRecvbuf, n_rand)
        minDenRecvbuf = np.reshape(minDenRecvbuf, n_rand)
        maxDenRecvbuf = np.reshape(maxDenRecvbuf, n_rand)
        # Projection map
        DensityRecvbuf = np.reshape(DensityRecvbuf, n_rand)
        StokeQRecvbuf = np.reshape(StokeQRecvbuf, n_rand)
        StokeURecvbuf = np.reshape(StokeURecvbuf, n_rand)

        print("Writing output file ...")

        # First file storing 1d projected data
        with h5py.File(file.replace(".h5", f"_{n_rand:d}_{method}_main.h5").replace('/h5/', '/h5_projected/'), 'w-') as write_data:

            # Parameters TODO
            # LOS
            write_data.create_dataset('los_x', data=xRandRecvbuf)
            write_data.create_dataset('los_y', data=yRandRecvbuf)
            write_data.create_dataset('los_z', data=zRandRecvbuf)
            # Orientation
            write_data.create_dataset('cloud_orientation', data=cloudThetaRecvbuf)
            write_data.create_dataset('b_orientation', data=posBThetaRecvbuf)
            # MCF
            write_data.create_dataset('mcf_slope', data=MCFSlopeRecvbuf)
            write_data.create_dataset('mcf_area', data=MCFAreaRecvbuf)
            # Cloud properties
            write_data.create_dataset('cloud_aspect', data=cloudAspectRecvbuf)
            write_data.create_dataset('cloud_mass', data=cloudMassRecvbuf)
            write_data.create_dataset('min_den', data=minDenRecvbuf)
            write_data.create_dataset('max_den', data=maxDenRecvbuf)
            write_data.close()

        # Second file, storing the projected map.
        with h5py.File(file.replace(".h5", f"_{n_rand:d}_{method}_map.h5").replace('/h5/', '/h5_projected/'), 'w-') as write_data:
            # projection maps
            write_data.create_dataset('den', DensityRecvbuf)
            write_data.create_dataset('stoke_Q', StokeQRecvbuf)
            write_data.create_dataset('stoke_U', StokeURecvbuf)
            write_data.close()
            
        # third file, storing the mcf
        with h5py.File(file.replace(".h5", f"_{n_rand:d}_{method}_mcf.h5").replace('/h5/', '/h5_projected/'), 'w-') as write_data:
            # mcfs.
            write_data.create_dataset('mcf', MCFRecvbuf)
            write_data.create_dataset('mcf_bin', MCFBinsRecvbuf)
            write_data.close()
    
def calc_all_projections(
    data: dict, losX: np.ndarray, losY: np.ndarray, losZ: np.ndarray
) -> Tuple:
    """
        calc projections for a lot of line of sight.

        :param data: dictionary of max_contour output
        :param losX: los vector x components
        :param losY: los vector y components
        :param losZ: los vector z components

        :return:  tubple of the following parameters
                    (
                        np.array(all_cloud_orientation), 
                        np.array(all_b_field_orientation),
                        np.array(all_mcf), 
                        np.array(all_mcf_bin), 
                        np.array(all_mcf_slope), 
                        np.array(all_mcf_area),
                        np.array(all_aspect_ratio), 
                        np.array(all_cloud_masses),
                        np.array(all_min_den), 
                        np.array(all_max_den),
                        np.array(binned_den), 
                        np.array(all_stoke_Q_map), 
                        np.array(all_stoke_U_map)
                    )
    """

    # TODO paramters to store #################
    # Orientation of the cloud
    all_cloud_orientation = []
    all_aspect_ratio = []
    # Orientation of magnetic field.
    all_b_field_orientation = []
    # for mcf
    all_mcf = []
    all_mcf_bin = []
    all_mcf_slope = []
    all_mcf_area = []
    # min and max column density
    all_min_den = []
    all_max_den = []
    # Cloud and dense threshold
    all_cloud_masses = []
    # 2D data map
    all_column_density_map = []
    all_stoke_Q_map = []
    all_stoke_U_map = []
    ###########################################

    cnt = 0
    # Calculate all step for each los
    for los in zip(losX, losY, losZ):

        print(f"rank {rank:2d}, cnt = {cnt:4d}")
        cnt += 1
        theta, phi = uc.cartToSph(*los)
        sys.stdout.flush()

        # TODO 1 Rotate data py lOS ###########################################
        x, y, z = cp.rotate_3d(data['bx'], data['by'], data['bz'], *los)
        bx, by, bz = cp.rotate_3d(data['bx'], data['by'], data['bz'], *los)
        #######################################################################

        cloud_orientation, b_field_orientation,\
            mcf, mcf_bin, mcf_slope, mcf_area, \
            aspect_ratio, cloud_mass, min_den, max_den,\
            binned_den, binned_stoke_Q, binned_stoke_U = \
            cp.projection(data['den'], x, y, z, bx, by, bz, los)

        # TODO paramters to store #################
        # Orientation of the cloud
        all_cloud_orientation += [cloud_orientation]
        all_aspect_ratio += [aspect_ratio]
        # Orientation of magnetic field.
        all_b_field_orientation += [b_field_orientation]
        # for mcf
        all_mcf += [mcf]
        all_mcf_bin += [mcf_bin]
        all_mcf_slope += [mcf_slope]
        all_mcf_area += [mcf_area]
        # min and max column density
        all_min_den += [min_den]
        all_max_den += [max_den]
        # Cloud and dense threshold
        all_cloud_masses += [cloud_mass]
        # 2D data map
        all_column_density_map += [binned_den]
        all_stoke_Q_map += [binned_stoke_Q]
        all_stoke_U_map += [binned_stoke_U]
        ###########################################

    return np.array(all_cloud_orientation), np.array(all_b_field_orientation), \
        np.array(all_mcf), np.array(all_mcf_bin), np.array(all_mcf_slope), np.array(all_mcf_area), \
        np.array(all_aspect_ratio), np.array(all_cloud_masses), \
        np.array(all_min_den), np.array(all_max_den), \
        np.array(binned_den), np.array(
            all_stoke_Q_map), np.array(all_stoke_U_map)


if __name__ == "__main__":

    # 2022-04-11
    main('../h5_max_contour/g1040_0016_binary_search.h5', 100, 'fib')
    main('../h5_max_contour/g1041_9015_by_mass.h5', 100, 'fib')