import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd


class main_data():

    def __init__(self, file: str = False):
        """
            :param file: string of the file name
        """
        self.file = file
        self.raw = {}
        self.table = None
        self.obs_set = []

        if not file:
            return

        if file.split('.')[-1] == 'h5':
            self.read_h5()
            self.calc_b_offset()
            # self.calc_dgf()

        elif file.split('.')[-1] == 'txt':
            self.obs = self.read_txt()

        else:
            print(f"Unsupport data type {file.split('.')[-1]}")
            return

    def read_h5(self):
        with h5py.File(self.file, 'r') as data:
            self.raw = {key: data[key][()] for key in data.keys()}

        self.table = pd.DataFrame(
            {
                # Orientation
                'cloud_orientation': self.raw['cloud_orientation']*180/np.pi,
                'b_orientation': self.raw['b_orientation']*180/np.pi,
                # MCF
                'mcf_area': self.raw['mcf_area'],
                'mcf_slope': self.raw['mcf_slope'],
                # Cloud properties
                'cloud_aspect': self.raw['cloud_aspect'],
                'cloud_mass': self.raw['cloud_mass'],
                # Line-of-sight
                'los_x': self.raw['los_x'],
                'los_y': self.raw['los_y'],
                'los_z': self.raw['los_z'],
            }
        )

    def read_txt(self):
        """
            Total three set of observation. 
            set 1 for test 1, 2
            set 2 for test 3 with dgs and ref 3
            set 3 for test 3 with dgs and ref 4
        """
        F = pd.read_csv(self.file, header=0, sep='\t')

        self.raw = {key: np.asarray(val) for key, val in F.iteritems()}
        self.table = F

        ###########################
        # Set 1, test 1 and test 2, only MCF and b-offset
        ###########################

        o1, p1 = main_data(), main_data()
        o1.file = p1.file = 'set 1'

        o1.table = F[~np.isnan(F['mcf_slope']) &
                     ~np.isnan(F['b_dir_optical'])
                     ][
            [
                'cloud_name',
                'mcf_slope',
                'b_dir_optical',
                'cloud_dir'
            ]
        ].copy(deep=True)

        p1.table = F[~np.isnan(F['mcf_slope']) &
                     ~np.isnan(F['b_dir_planck'])
                     ][
            [
                'cloud_name',
                'mcf_slope',
                'b_dir_planck',
                'cloud_dir'
            ]
        ].copy(deep=True)

        o1.table = o1.table.rename(
            columns={'b_dir_optical': 'b_orientation', 'cloud_dir': 'cloud_orientation'})
        p1.table = p1.table.rename(
            columns={'b_dir_planck': 'b_orientation', 'cloud_dir': 'cloud_orientation'})

        o1.calc_b_offset()
        p1.calc_b_offset()

        self.obs_set += [[o1, p1]]

        ###########################
        # Set 2, test 3 b_offset and ref 3
        ###########################
        o2, p2 = main_data(), main_data()
        o2.file = p2.file = 'set 1'

        o2.table = F[~F['cloud_name'].isin(['Orion A', 'Orion B']) &
                     ~np.isnan(F['sfr_3']) &
                     ~np.isnan(F['b_dir_optical'])
                     ][
            [
                'cloud_name',
                'sfr_3',
                'b_dir_optical',
                'cloud_dir'
            ]
        ].copy(deep=True)

        p2.table = F[~np.isnan(F['sfr_3']) &
                     ~np.isnan(F['b_dir_planck'])
                     ][
            [
                'cloud_name',
                'sfr_3',
                'b_dir_planck',
                'cloud_dir'
            ]
        ].copy(deep=True)

        o2.table = o2.table.rename(columns={
                                   'b_dir_optical': 'b_orientation', 'cloud_dir': 'cloud_orientation', 'sfr_3': 'dgf'})
        p2.table = p2.table.rename(columns={
                                   'b_dir_planck': 'b_orientation', 'cloud_dir': 'cloud_orientation', 'sfr_3': 'dgf'})

        o2.calc_b_offset()
        p2.calc_b_offset()

        self.obs_set += [[o2, p2]]

        ###########################
        # Set 3, test 3 b_offset and ref 4
        ###########################
        o3, p3 = main_data(), main_data()
        o3.file = p3.file = 'set 1'

        o3.table = F[~F['cloud_name'].isin(['Orion A', 'Orion B']) &
                     ~np.isnan(F['sfr_4']) &
                     ~np.isnan(F['b_dir_optical'])
                     ][
            [
                'cloud_name',
                'sfr_4',
                'b_dir_optical',
                'cloud_dir'
            ]
        ].copy(deep=True)

        p3.table = F[~np.isnan(F['sfr_4']) &
                     ~np.isnan(F['b_dir_planck'])
                     ][
            [
                'cloud_name',
                'sfr_4',
                'b_dir_planck',
                'cloud_dir'
            ]
        ].copy(deep=True)

        o3.table = o3.table.rename(columns={
                                   'b_dir_optical': 'b_orientation',
                                   'cloud_dir': 'cloud_orientation', 'sfr_4': 'dgf'})
        p3.table = p3.table.rename(columns={
                                   'b_dir_planck': 'b_orientation',
                                   'cloud_dir': 'cloud_orientation', 'sfr_4': 'dgf'})

        o3.calc_b_offset()
        p3.calc_b_offset()

        self.obs_set += [[o3, p3]]

        return

    def calc_b_offset(self):
        """
            calculate the cloud field offset of the current file.
        """
        self['b_offset'] = np.abs(
            self['cloud_orientation'] - self['b_orientation'])
        self['b_offset'][self['b_offset'] > 90] = 180 - \
            self['b_offset'][self['b_offset'] > 90]

    def calc_dgf(self, dense_gas: float, total_cloud_mass: float, contour_cloud_mass: float) -> None:
        """
            calculate the dense gas fraction of the current file

            :param dense_gas_fraction: The dense gas fraction, 
                                        define as dense mass / total cloud mass * contour cloud mass / projected cloud mass
        """
        self.table['dgf'] = dense_gas / total_cloud_mass * contour_cloud_mass / self.table['cloud_mass']

    def __getitem__(self, item):
        return self.table[item]

    def __setitem__(self, key, val):
        self.table[key] = val



def read_data(file: str) -> dict:
    """
        simple way to read h5 file.
        :param file: path to the h5 file.
        "return: dictionary of data.
    """
    with h5py.File(file, 'r') as data:
        my_data = {key: data[key][()] for key in data.keys()}
    return my_data