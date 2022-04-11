from operator import pos

import h5py
import numpy as np
import sys
import scipy.stats as st

import calc_unit_conversion as uc
import calc_projection as cp

import math
from mpi4py import MPI





#%% initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


print("size is %d " % size)

def readData(file):
    with h5py.File(file, 'r') as data:
        myData = {key: data[key][()] for key in data.keys()}
    return myData



def main(argv):
    
        
    
    # create projection angles.
    np.random.seed(0)

    n_rand = int(argv[0]) # number of projection directions
    
    if str(argv[1]) == "fib":
        print("generating uniform projection")
        x_rand_, y_rand_, z_rand_ = cp.fibonacci_sphere(n_rand)

    else:
        print("generating random projection")
        x_rand_, y_rand_, z_rand_ = cp.random_unit_vector(n_rand)
    
    nRandPerProc = int(n_rand/size)

    init = rank*nRandPerProc
    term = (rank+1)*nRandPerProc

    print(rank, nRandPerProc)
    comm.Barrier()

    x_rand = x_rand_[init:term]
    y_rand = y_rand_[init:term]
    z_rand = z_rand_[init:term]

    # Read data
    # fileNames = ["./half_scale_restart/g1041_9015_data_in_contour_checkingVersion.h5", "./half_scale_restart/g1040_0016_data_in_contour_checkingVersion.h5"]
    fileNames = ["./half_scale_restart/g1040_0016_data_in_contour_new_contour_32.h5" ]#, "./half_scale_restart/g1041_9015_data_in_contour_new_contour_32.h5"]

    for fullPATH in fileNames:

    
        # MPI read data.
        data = None

        if rank == 0:
            print(rank, "Reading data from " + fullPATH)
            data = readData(fullPATH)
            print(rank, "Finish reading data")
            for key in data.keys():
                print(key, type(data[key]))
        
        comm.Barrier()
        
        # boardcast data.
        data = comm.bcast(data, root = 0)
        
        comm.Barrier()

        cloudOri, bOri, mcfArea, mcfSlope, aspect, min_den, max_den, cloud_masses, dense_masses = calcAll(data, x_rand, y_rand, z_rand)


        xRandRecvbuf = comm.gather(x_rand, root=0)
        yRandRecvbuf = comm.gather(y_rand, root=0)
        zRandRecvbuf = comm.gather(z_rand, root=0)
        cloudAspectRecvbuf = comm.gather(aspect, root=0)
        cloudThetaRecvbuf = comm.gather(cloudOri, root=0)
        posBThetaRecvbuf = comm.gather(bOri, root=0)
        MCFSlopeRecvbuf = comm.gather(mcfSlope, root=0)
        MCFAreaRecvbuf = comm.gather(mcfArea, root=0)
        minDenRecvbuf = comm.gather(min_den, root = 0)
        maxDenRecvbuf = comm.gather(max_den, root = 0)
        cloudMassRecvbuf = comm.gather(cloud_masses, root = 0)
        denseMassRecvbufs = [comm.gather(dense_mass, root = 0) for dense_mass in dense_masses]


        if rank==0:

            xRandRecvbuf = np.reshape(xRandRecvbuf, n_rand)
            yRandRecvbuf = np.reshape(yRandRecvbuf ,n_rand)
            zRandRecvbuf = np.reshape(zRandRecvbuf ,n_rand)
            cloudAspectRecvbuf = np.reshape(cloudAspectRecvbuf,n_rand)
            cloudThetaRecvbuf = np.reshape(cloudThetaRecvbuf ,n_rand)
            posBThetaRecvbuf = np.reshape(posBThetaRecvbuf ,n_rand)
            MCFSlopeRecvbuf = np.reshape(MCFSlopeRecvbuf ,n_rand)
            MCFAreaRecvbuf = np.reshape(MCFAreaRecvbuf, n_rand)
            minDenRecvbuf = np.reshape(minDenRecvbuf, n_rand)
            maxDenRecvbuf = np.reshape(maxDenRecvbuf, n_rand)
            cloudMassRecvbuf = np.reshape(cloudMassRecvbuf, n_rand)
            denseMassRecvbufs = [np.reshape(denseMassRecvbuf, n_rand) for denseMassRecvbuf in denseMassRecvbufs]


            print("Write output file ...")

            with h5py.File(fullPATH.replace(".h5", f"_calculated_fib_with_{n_rand:d}_DR_SF_nT.h5"), 'w-') as write_data:
                    
                # Parameters TODO
                # cloudAspect = []
                # cloudTheta = []
                # losBStr = []
                # posBStr = []
                # posBTheta = []
                # MCFSlope = []

                write_data.create_dataset('losx', data = xRandRecvbuf)
                write_data.create_dataset('losy', data = yRandRecvbuf)
                write_data.create_dataset('losz', data = zRandRecvbuf)
                write_data.create_dataset('cloudAspect', data = cloudAspectRecvbuf)
                write_data.create_dataset('cloudTheta', data = cloudThetaRecvbuf)
                write_data.create_dataset('posBTheta', data = posBThetaRecvbuf)
                write_data.create_dataset('MCFSlope', data = MCFSlopeRecvbuf)
                write_data.create_dataset('MCFArea', data = MCFAreaRecvbuf)
                write_data.create_dataset('minDen', data = minDenRecvbuf)
                write_data.create_dataset('maxDen', data = maxDenRecvbuf)
                write_data.create_dataset('cloud_mass', data = cloudMassRecvbuf)
                dense_thresholds = [number_den_H2_per_cm2_to_column_den_Msun_per_pc2(b_code_to_muG(7.2945)/1.1e-21, mu = 1.37),
                                    220.92, 221.18, 277.60, 294.94, 252.71, 265.24]
                
                for dense_threshold, denseMassRecvbuf in zip(dense_thresholds, denseMassRecvbufs):
                    key = f"dense_mass_{dense_threshold:.2f}"
                    write_data.create_dataset(key, data = denseMassRecvbuf)                
                    
                write_data.close()

def calcAll(data, losX, losY, losZ):

    # Orientation of the cloud
    cloudOri = []

    # Orientation of magnetic field.
    bOri_weighted_2D = []

    mcfSlope = []
    mcfArea = []
    aspectRatio = []

    # min and max column density
    min_den = []
    max_den = []

    # Cloud and dense threshold
    cloud_threshold = 2*15 # extinction_mag_to_column_denisty_Msun_per_pc2(2, mu = 2.3)
    # dense_threshold = 129 # extinction_mag_to_column_denisty_Msun_per_pc2(8, mu = 2.3)
    dense_thresholds = [number_den_H2_per_cm2_to_column_den_Msun_per_pc2(b_code_to_muG(7.2945)/1.1e-21, mu = 1.37),
                    220.92, 221.18, 277.60, 294.94, 252.71, 265.24]
    cloud_masses = []
    dense_masses = [[] for i in dense_thresholds]

    cnt = 0
    # Calculate all step for each los
    for los in zip(losX, losY, losZ):

        print(f"rank {rank:2d}, cnt = {cnt:4d}")
        cnt += 1
        theta, phi = cartToSph(*los)
        sys.stdout.flush()

        # los = (losX[53], losY[53], losZ[53])

        # Rotate data py lOS
        xRot, yRot, zRot = rotate_3d(data['x'], data['y'], data['z'], *los)
        BxRot, ByRot, BzRot = rotate_3d(data['bx'], data['by'], data['bz'], *los)


        ###################################
        ### Magnetic field
        ###################################
        # Calculate stoke parameter of Magnetic field.
        stoke_I, stoke_Q, stoke_U, stoke_phi = calcStoke(BxRot, ByRot)
        stoke_phi = None


        # # weighted stoke parameter
        weighted_stoke_Q = stoke_Q * data['density']
        weighted_stoke_U = stoke_U * data['density']
        # weighted_stoke_I = stoke_I * data['density']


        # Create bins data
        # cell cneter cooridinate.
        dx = 10/480
        x = np.linspace(-10+dx/2, 10-dx/2, 960)
        y = x.copy()
        X, Y = np.meshgrid(x, y)

        # boundary cooridinates
        binX = np.linspace(-10, 10, 961)
        binY = binX.copy()

        # bin projected data into 2d
        # (sum_i den) 
        binsDen = st.binned_statistic_2d(yRot, xRot, data['density'], statistic="sum", bins=[binX, binY])[0] * dx


        # (sum_i phi_i * den_i) 
        binsQ_weighted_2D = st.binned_statistic_2d(yRot, xRot, weighted_stoke_Q, statistic="sum", bins=[binX, binY])[0]
        binsQ_weighted_2D = np.divide(binsQ_weighted_2D, binsDen, out = np.zeros_like(binsQ_weighted_2D), where = binsDen != 0)

        binsU_weighted_2D = st.binned_statistic_2d(yRot, xRot, weighted_stoke_U, statistic="sum", bins=[binX, binY])[0]
        binsU_weighted_2D = np.divide(binsU_weighted_2D, binsDen, out = np.zeros_like(binsU_weighted_2D), where = binsDen != 0)


        # Phi Map
        binsPhi_weighted_2D = 0.5*np.arctan2(binsU_weighted_2D, binsQ_weighted_2D)
        connectedIndex = connectedStructure(binsDen)

        # turn binned data into cooridinates.
        xCoor = X[connectedIndex > 0]
        yCoor = Y[connectedIndex > 0]
        dCoor = binsDen[connectedIndex > 0]

        stoke_phi_weighted_2D_Coor = binsPhi_weighted_2D[connectedIndex > 0]
        stoke_U_weighted_2D_Coor = binsU_weighted_2D[connectedIndex > 0]
        stoke_Q_weighted_2D_Coor = binsQ_weighted_2D[connectedIndex > 0]


        # Calculate mass above star formation threshold
        cloud_mass = np.sum(binsDen[binsDen >= cloud_threshold])*dx**2
        dense_mass = (np.sum(binsDen[binsDen >= dense_threshold])*dx**2 for dense_threshold in dense_thresholds)
        
        cloud_masses += [cloud_mass]
        for dms, dm in zip(dense_masses, dense_mass):
            dms += [dm]
        

        # weighted magnetic field phi
        stoke_phi_weighted_2D = calcStoke_Phi(np.sum(stoke_U_weighted_2D_Coor*dCoor)/np.sum(dCoor),
                                              np.sum(stoke_Q_weighted_2D_Coor*dCoor)/np.sum(dCoor))
        
        bOri_weighted_2D += [stoke_phi_weighted_2D]

        # # B field in given projection
        bPOS_weighted_2D = np.array([np.cos(bOri_weighted_2D[-1]), np.sin(bOri_weighted_2D[-1])])
        
        # # PCA cloud orientation.
        val, vec = calcComponentsWeighted(xCoor, yCoor, dCoor)
        majVec = vec[0]*np.sqrt(val[0])
        minVec = vec[1]*np.sqrt(val[1])
        aspectRatio += [np.sqrt(val[0]/val[1])]

        # Save cloud orientation and dimension
        cloudPhi = calcStoke(*majVec)[3]
        cloudOri += [cloudPhi]

        
        # MCF
        bins = np.linspace(0, 150000, int(10e5)+1) * dx
        
        mcf, mcfBins = calcMCF(dCoor, bins, 0)
        slope, area = calcMCFSlopeAndArea(mcf, mcfBins)
        # print(f"{slope:.7f}, {1/area:.7f}")
        
        mcfSlope += [slope]
        mcfArea += [area]
        min_den += [np.min(dCoor)]
        max_den += [np.max(dCoor)]

    return np.array(cloudOri), np.array(bOri_weighted_2D), \
           np.array(mcfArea), np.array(mcfSlope), np.array(aspectRatio), \
           min_den, max_den, cloud_masses, dense_masses
    

if __name__=="__main__":
    
    main(sys.argv[1:])