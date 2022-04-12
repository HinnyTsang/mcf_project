from typing import Tuple
from mpi4py import MPI
import h5py
import numpy as np
import os

import sys
sys.path.append('..')
import functions.calc_orientation as calc_ori
import functions.calc_unit_conversion as uc
import functions.calc_projection as cp
import functions.calc_mcf as calc_mcf


# initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
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

    if not os.path.exists(file):
        if rank == 0:
            print(f"file {file} not exist.")
        return

    if method == "fib":
        if rank == 0:
            print("generating uniform projection")
        x_rand_, y_rand_, z_rand_ = cp.fibonacci_sphere(n_rand)

    elif method == "ran":
        if rank == 0:
            print("generating random projection")
        x_rand_, y_rand_, z_rand_ = cp.random_unit_vector(n_rand)

    else:
        if rank == 0:
            print("arg should be either 'ran' or 'fib'.")
        return

    # TODO Check if output path exist, create if not exist. #####
    path, file_name = os.path.split(file)
    out_path = os.path.splitext(file_name)[0] + f"_{n_rand}_{method}" # output path.
    out_path = os.path.join(path, out_path).replace('/h5_max_contour/', '/h5_projected/')
    
    if os.path.exists(out_path):
        if rank == 0: print(f"output path {out_path} exist")
    else:
        if rank == 0: 
            print(f"output path {out_path} is not exist, create output path.")
            os.mkdir(out_path)
    #############################################################
    
    
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


    # MPI read data.
    data = None

    if rank == 0:
        print(f"Reading data from {file}")
        data = read_data(file)
        print("Finish reading data")

    # boardcast data.
    comm.Barrier()
    data = comm.bcast(data, root=0)
    comm.Barrier()

    # TODO Call the projection function ########################################
    cloud_orientation, b_orientation, \
        mcf_slope, mcf_area, \
        aspect_ratio, cloud_mass, min_den, max_den, \
        = calc_all_projections(data, x_rand, y_rand, z_rand, out_path)
    ############################################################################
    # the last arguement file and method is not necessary.

    # LOS
    xRandRecvbuf = comm.gather(x_rand, root=0)
    yRandRecvbuf = comm.gather(y_rand, root=0)
    zRandRecvbuf = comm.gather(z_rand, root=0)
    # Orientation
    cloudThetaRecvbuf = comm.gather(cloud_orientation, root=0)
    posBThetaRecvbuf = comm.gather(b_orientation, root=0)
    # MCF
    MCFSlopeRecvbuf = comm.gather(mcf_slope, root=0)
    MCFAreaRecvbuf = comm.gather(mcf_area, root=0)
    # Cloud properties
    cloudAspectRecvbuf = comm.gather(aspect_ratio, root=0)
    cloudMassRecvbuf = comm.gather(cloud_mass, root=0)
    minDenRecvbuf = comm.gather(min_den, root=0)
    maxDenRecvbuf = comm.gather(max_den, root=0)


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
        MCFSlopeRecvbuf = np.reshape(MCFSlopeRecvbuf, n_rand)
        MCFAreaRecvbuf = np.reshape(MCFAreaRecvbuf, n_rand)
        # Cloud properties
        cloudAspectRecvbuf = np.reshape(cloudAspectRecvbuf, n_rand)
        cloudMassRecvbuf = np.reshape(cloudMassRecvbuf, n_rand)
        minDenRecvbuf = np.reshape(minDenRecvbuf, n_rand)
        maxDenRecvbuf = np.reshape(maxDenRecvbuf, n_rand)
        # Projection map

        print("Writing output file ...")

        # First file storing 1d projected data
        with h5py.File(os.path.join(out_path, 'main.h5'), 'w-') as write_data:

            # Parameters TODO
            # LOS
            write_data.create_dataset('los_x', data=xRandRecvbuf)
            write_data.create_dataset('los_y', data=yRandRecvbuf)
            write_data.create_dataset('los_z', data=zRandRecvbuf)
            # Orientation
            write_data.create_dataset(
                'cloud_orientation', data=cloudThetaRecvbuf)
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

def calc_all_projections(
    # for file output, not necessary
    data: dict, losX: np.ndarray, losY: np.ndarray, losZ: np.ndarray, out_path: str
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
                        np.array(all_mcf_slope),
                        np.array(all_mcf_area),
                        np.array(all_aspect_ratio),
                        np.array(all_cloud_masses),
                        np.array(all_min_den),
                        np.array(all_max_den)
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

    cnt = 1
    # Calculate all step for each los
    for los in zip(losX, losY, losZ):

        print(f"rank {rank:2d}, cnt = {cnt:4d}")
        theta, phi = cp.cart_to_sph(*los)
        sys.stdout.flush()

        # TODO 1 Rotate data py lOS ###########################################
        x, y, z = cp.rotate_3d(data['x'], data['y'], data['z'], *los)
        bx, by, bz = cp.rotate_3d(data['bx'], data['by'], data['bz'], *los)
        #######################################################################

        cloud_orientation, b_field_orientation,\
            mcf, mcf_bin, mcf_slope, mcf_area, \
            aspect_ratio, cloud_mass, min_den, max_den,\
            binned_den, binned_stoke_Q, binned_stoke_U = \
            cp.projection(data['density'], bx, by, bz, x, y, z)

        # TODO paramters to store #############################################
        # Orientation of the cloud
        all_cloud_orientation += [cloud_orientation]
        all_b_field_orientation += [b_field_orientation]
        # for mcf
        all_mcf += [mcf]
        all_mcf_bin += [mcf_bin]
        all_mcf_slope += [mcf_slope]
        all_mcf_area += [mcf_area]
        # min and max column density
        all_aspect_ratio += [aspect_ratio]
        all_cloud_masses += [cloud_mass]
        all_min_den += [min_den]
        all_max_den += [max_den]
        # 2D data map
        all_column_density_map += [binned_den]
        all_stoke_Q_map += [binned_stoke_Q]
        all_stoke_U_map += [binned_stoke_U]
        ###########################################


        # TODO save map files due to limitation of memories.  #################
        # Create output for every 25 projections.
        if cnt % 25 == 0:
            
            out_file = f"r{rank:02d}_{int(cnt/25):02d}"
            
            # save for density #################################################
            print(f"rank {rank: 2d}, saving output file {out_file}_map.h5")

            with h5py.File(os.path.join(out_path, out_file + '_map.h5'), 'w-') as write_data:
                write_data.create_dataset('los_x', data=losX[cnt-25:cnt])
                write_data.create_dataset('los_y', data=losY[cnt-25:cnt])
                write_data.create_dataset('los_z', data=losZ[cnt-25:cnt])
                write_data.create_dataset('den', data=all_column_density_map)
                write_data.create_dataset('stoke_Q', data=all_stoke_Q_map)
                write_data.create_dataset('stoke_U', data=all_stoke_U_map)

            # free memory.
            del all_column_density_map
            all_column_density_map = []
            del all_stoke_Q_map
            all_stoke_Q_map = []
            del all_stoke_U_map
            all_stoke_U_map = []
            ###################################################################
            
            # save for mcf. ###################################################
            print(f"rank {rank:02d}, saving output file {out_file}_mcf.h5")

            with h5py.File(os.path.join(out_path, out_file + '_mcf.h5'), 'w-') as write_data:
                write_data.create_dataset('los_x', data=losX[cnt-25:cnt])
                write_data.create_dataset('los_y', data=losY[cnt-25:cnt])
                write_data.create_dataset('los_z', data=losZ[cnt-25:cnt])
                write_data.create_dataset('mcf', data=all_mcf)
                write_data.create_dataset('mcf_bin', data=all_mcf_bin)
                
            # free memory.
            del all_mcf
            all_mcf = []
            del all_mcf_bin
            all_mcf_bin = []
        #######################################################################

        # counter
        cnt += 1

    return np.array(all_cloud_orientation), np.array(all_b_field_orientation), \
        np.array(all_mcf_slope), np.array(all_mcf_area), \
        np.array(all_aspect_ratio), np.array(all_cloud_masses), \
        np.array(all_min_den), np.array(all_max_den)



if __name__ == "__main__":

    # 2022-04-11 ######################################################
    main('../h5_max_contour/g1040_0016_binary_search.h5', 100, 'fib')
    main('../h5_max_contour/g1041_9015_by_mass.h5', 100, 'fib')

    # main('../h5_max_contour/g1040_0016_binary_search.h5', 10000, 'fib')
    # main('../h5_max_contour/g1041_9015_by_mass.h5', 10000, 'fib')

    # main('../h5_max_contour/g1040_0016_binary_search.h5', 10000, 'ran')
    # main('../h5_max_contour/g1041_9015_by_mass.h5', 10000, 'ran')
    ###################################################################
