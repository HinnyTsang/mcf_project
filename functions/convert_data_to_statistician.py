"""
    Convert data to the statistician
    
    Author: Hinny Tsang
    Last Edit: 2022-04-12
"""
import os
import numpy as np
import csv
import sys
sys.path.append("..") 
import functions.data_class as data_class


OUTPATH = "../csv"

def save_data_to_statisian(para_path: str, perp_path: str, date: str):
    """
        :param para_path:   path of the parallel file
        :param perp_path:   path of the perpendicular file
        :param date:        date of today yyyymmdd
        :reture:
    """
    mode1, mode2 = para_path.split('_')[-1], perp_path.split('_')[-1]
    n1, n2 = para_path.split('_')[-2], perp_path.split('_')[-2]

    if mode1 != mode2 or n1 != n2:
        print("two version is not matched")

    # TODO Read h5 file. ##################################
    para = data_class.main_data(os.path.join(para_path, 'main.h5'))
    perp = data_class.main_data(os.path.join(perp_path, 'main.h5'))
    
    # TODO calculate dense gass mass. #####################
    # Parameters calculated in 'check_dense_gas_fraction.ipynb'
    para.calc_dgf(dense_gas=4413.0297186780,
                  total_cloud_mass=11002.1148454371,
                  contour_cloud_mass=3642.722286)
    perp.calc_dgf(dense_gas=950.9536251046,
                  total_cloud_mass=11002.1148454371,
                  contour_cloud_mass=3642.722489)
    #######################################################

    # TODO save data for statistician #####################
    # open the file in the write mode
    out_file = os.path.join(OUTPATH,
        f"simulation_output_{mode1}_{n1}_{date}.csv")

    header = ['mcf_slope', 'mcf_area', 'cloud_field_offset',
              'dense_mass_fraction', 'cloud_type']

    # TODO write the output file
    with open(out_file, 'w', newline='') as f:

        writer = csv.writer(f)  # create the csv writer
        writer.writerow(header)  # write a row to the csv file

        # the parallel cloud data row by row
        for row in zip(
                para['mcf_slope']*15,       # convert to mag-1
                para['mcf_area']/15,        # convert to mag
                para['b_offset']/180*np.pi, # convert to rad
                para['dgf'],
                np.zeros(int(n1), dtype=int)):
            writer.writerow(row)

        # the parallel cloud data row by row
        for row in zip(
                perp['mcf_slope']*15,       # convert to mag-1
                perp['mcf_area']/15,        # convert to mag
                perp['b_offset']/180*np.pi, # convert to rad
                perp['dgf'],
                np.ones(int(n2), dtype=int)):
            writer.writerow(row)


if __name__ == "__main__":

    # 2022-04-12 writing output csv files. #############################
    # para_path = '../h5_projected/g1040_0016_binary_search_10000_ran'
    # perp_path = '../h5_projected/g1041_9015_by_mass_10000_ran'
    # save_data_to_statisian(para_path, perp_path, '20220412')
    # ####################################################################
    # para_path = '../h5_projected/g1040_0016_binary_search_10000_fib'
    # perp_path = '../h5_projected/g1041_9015_by_mass_10000_fib'
    # save_data_to_statisian(para_path, perp_path, '20220412')
    # ####################################################################
    # para_path = '../h5_projected/g1040_0016_binary_search_100_fib'
    # perp_path = '../h5_projected/g1041_9015_by_mass_100_fib'
    # save_data_to_statisian(para_path, perp_path, '20220412')
    ###################################################################
    
    
    
    pass