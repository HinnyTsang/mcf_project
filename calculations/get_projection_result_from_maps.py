"""
    Plot the scatter plot of the projected parameters.
    see check_projection_parameters.ipynb
    
    Author: Hinny Tsang
    Last Edit: 2022-04-21
"""

import h5py
import os
import sys

from matplotlib import pyplot as plt
import numpy as np
sys.path.append('..')
import functions.plot as plot
from typing import List



def main(file_path: str, versions: List[str], n_threads: List[int]) -> None:
    """
        calculate the parameters from the map already projected
    """
    for version, n_thread in zip(versions, n_threads):
        
        # number of projections
        n_proj = int(version[0].split('_')[-2])
        
        # output are store into one file for 25 projections.
        n_folder = int(n_proj/n_thread/25)
        
        dx = 10/480 # since I know, modify if need change.
        
        for cloud_orientation in version:
            
            # values to calculate
            # cloud mass (Av)
            all_mass_2 = []
            all_mass_4 = []
            all_mass_8 = []
            
            # output file path.
            out_path = os.path.join(file_path, cloud_orientation)
            
            for thread in range(n_thread):
                for folder_id in range(1, n_folder+1):
                    
                    file_name = f"r{thread:02d}_{folder_id:02d}_map.h5"
                    full_path = os.path.join(file_path, cloud_orientation, file_name)
                    
                    if os.path.isfile(full_path):
                        
                        # read h5 file
                        with h5py.File(full_path, 'r') as file:
                            data = {key: file[key][()] for key in file.keys()}

                        # calculate dense gas fraction greater that some value
                        for map_id in range(len(data['den'])):
                            
                            # Av = 2:
                            mass_2 = np.sum(data['den'][map_id][data['den'][map_id] > 15*2])
                            # Av = 4:
                            mass_4 = np.sum(data['den'][map_id][data['den'][map_id] > 15*4])
                            # Av = 8
                            mass_8 = np.sum(data['den'][map_id][data['den'][map_id] > 15*8])
                            
                            # store
                            all_mass_2 += [mass_2]
                            all_mass_4 += [mass_4]
                            all_mass_8 += [mass_8]

                    else:
                        print(f"{full_path} doesn't exist")
                        
            # <- carefull about the indent.            
            # save file (since only dgf in this case)
            out_file = os.path.join(out_path, 'sub_dgf.h5')
            
            with h5py.File(out_file, 'w') as write_data:
                
                # Parameters TODO
                write_data.create_dataset('cloud_mass_2', data=np.array(all_mass_2) * dx ** 2)
                write_data.create_dataset('cloud_mass_4', data=np.array(all_mass_4) * dx ** 2)
                write_data.create_dataset('cloud_mass_8', data=np.array(all_mass_8) * dx ** 2)
                
                write_data.close()
            
    pass


if __name__ == "__main__":

    projection_path = '../h5_projected'
    versions = [
        # ['g1040_0016_binary_search_100_fib', 'g1041_9015_by_mass_100_fib'],
        ['g1040_0016_binary_search_10000_fib', 'g1041_9015_by_mass_10000_fib'],
        ['g1040_0016_binary_search_10000_ran', 'g1041_9015_by_mass_10000_ran']
    ]
    n_threads = [
        # 4, 
        25, 
        25]

    main(projection_path, versions, n_threads)
