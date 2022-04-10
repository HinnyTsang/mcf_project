import copy
import numpy as np
import h5py
from typing import List






if __name__ == "__main__":
    
    
    # output file from scorpio.
    files = [  # '/data/hinny/Scorpio_1.5_result/patch_cluster2/template/h5/half_scale_restart/g1040_0016.h5',
        '/data/hinny/Scorpio_1.5_result/patch_cluster2/template/h5/half_scale_restart/g1041_9015.h5']

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