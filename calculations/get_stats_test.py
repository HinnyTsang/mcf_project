"""
    Calculate statistical test for each set of sample

    Author: Hinny Tsang
    Last Edit: 2022-04-13
"""
import sys
sys.path.append('..')
import os
import h5py
import functions.data_class as data_class
import functions.calc_stat_tests as stat_tests



OUTPATH = "../statistc_test_result/"


def main(para_file: str, perp_file: str, out_folder: str, to_do: list) -> None:
    """
        Do statistical test for the given data set.

        The file should be the output of 'get_projection_mpi.py'

        :param para_file:  path to the parallel cloud h5 file.
        :param perp_file:  path to the perpendicular cloud h5 file.
        :param out_folder: name of output folder
        :param to_do:      list of stat tests that wanted to do.
    """

    # check if input file exist.
    if not os.path.exists(para_file):
        print(f"file {para_file} not exist.")
        exit()
    if not os.path.exists(perp_file):
        print(f"file {perp_file} not exist.")
        exit()

    # TODO Check if output path exist, create if not exist. #####
    # if outpath is not exist, create it.
    out_path = os.path.join(OUTPATH, out_folder)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    #############################################################

    # TODO read input files. ####################################
    para = data_class.main_data(para_file)
    perp = data_class.main_data(perp_file)
    # calculate dgf
    para.calc_dgf(dense_gas=4413.0297186780,
                  total_cloud_mass=11002.1148454371,
                  contour_cloud_mass=3642.722286)
    perp.calc_dgf(dense_gas=950.9536251046,
                  total_cloud_mass=11002.1148454371,
                  contour_cloud_mass=3642.722489)
    #############################################################

    # TODO do all statistical tests. ############################
    for test_args in to_do:
        test_name = test_args['test']
        
        print(f"Try test {test_name} ...")

        # TODO Projection test.
        if test_name == 'projection':

            print("do projection test")
            # TODO decode args.
            n = test_args['n']
            # TODO do statistical test.
            result = stat_tests.projection_test(
                n=n,
                b_offset_para=para['b_offset'],
                b_offset_perp=perp['b_offset'])

            # TODO save output file.
            out_file = os.path.join(out_path, f"proj_test_{n}.h5")
            print(f"writing output to {out_file}")

            with h5py.File(out_file, mode='w') as write_data:
                for key, val in result.items():
                    write_data.create_dataset(key, data=val)

        # TODO Parameter test with r.
        elif test_name == 'parameter' and 'r' in test_args.keys():

            # TODO decode args.
            n = test_args['n']
            r = test_args['n']
            n_sample = test_args['n_sample']
            test_param = test_args['param']

            # TODO do statistical test.
            print(f"do {test_param} test")
            result = stat_tests.parameter_test_given_r(
                n=n, r=r,
                b_offset_para=para['b_offset'],
                b_offset_perp=perp['b_offset'],
                parameter_para=para[test_param],
                parameter_perp=perp[test_param],
                sampling=n_sample
            )
            # TODO save output file.
            out_file = os.path.join(
                out_path, f"{test_param}_test_{n}_{r}.h5")
            print(f"writing output to {out_file}")

            with h5py.File(out_file, mode='w') as write_data:
                for key, val in result.items():
                    write_data.create_dataset(key, data=val)

        # TODO Parameter test.
        elif test_name == 'parameter' and 'r' not in test_args.keys():

            # TODO decode args.
            n = test_args['n']
            n_sample = test_args['n_sample']
            test_param = test_args['param']

            # TODO do statistical test.
            print(f"do {test_param} test")
            result = stat_tests.parameter_test(
                n=n,
                b_offset_para=para['b_offset'],
                b_offset_perp=perp['b_offset'],
                parameter_para=para[test_param],
                parameter_perp=perp[test_param],
                sampling=n_sample
            )
            # TODO save output file.
            out_file = os.path.join(
                out_path, f"{test_param}_test_{n}.h5")
            print(f"writing output to {out_file}")

            with h5py.File(out_file, mode='w') as write_data:
                for key, val in result.items():
                    write_data.create_dataset(key, data=val)

        else:
            print(f"unknown test {test_name}.")
            return


if __name__ == "__main__":

    # 2022-04-13 ######################################################
    test_to_do = [
        {"test": "projection", 'n': 13, },
        {"test": "projection", 'n': 12, },
        {"test": "parameter",
            'param': 'mcf_slope',
            'n': 13,
            'n_sample': 10000},
        {"test": "parameter",
            'param': 'mcf_slope',
            'n': 12,
            'n_sample': 10000},
        {"test": "parameter",
            'param': 'mcf_slope',
            'n': 13, 'r': 5,
            'n_sample': 10000},
        {"test": "parameter",
            'param': 'mcf_slope',
            'n': 12, 'r': 6,
            'n_sample': 10000},
    ]
    test_to_do = [
        {"test": "parameter",
            'param': 'dgf',
            'n': 13, 'r': 5,
            'n_sample': 10000},
        {"test": "parameter",
            'param': 'dgf',
            'n': 12, 'r': 6,
            'n_sample': 10000}
    ]

    # data set 1
    # main(para_file='../h5_projected/g1040_0016_binary_search_100_fib/main.h5',
    #      perp_file='../h5_projected/g1041_9015_by_mass_100_fib/main.h5',
    #      out_folder='fib_100',
    #      to_do=test_to_do)
    # data set 2.
    # main(para_file='../h5_projected/g1040_0016_binary_search_10000_fib/main.h5',
    #      perp_file='../h5_projected/g1041_9015_by_mass_10000_fib/main.h5',
    #      out_folder='fib_10000',
    #      to_do=test_to_do)
    # data set 3.
    main(para_file='../h5_projected/g1040_0016_binary_search_10000_ran/main.h5',
         perp_file='../h5_projected/g1041_9015_by_mass_10000_ran/main.h5',
         out_folder='ran_10000',
         to_do=test_to_do)
    ###################################################################
    pass
