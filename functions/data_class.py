import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd


class data():
    def __init__(self, file=False):
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
        F = pd.read_csv(self.file, header=0, sep = '\t')

        self.raw = {key: np.asarray(val) for key, val in F.iteritems()}
        self.table = F
        
        ###########################
        # Set 1, test 1 and test 2, only MCF and b-offset
        ###########################
        
        o1, p1 = data(), data()
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
                    ].copy(deep = True)
                    
        p1.table = F[~np.isnan(F['mcf_slope']) &
                     ~np.isnan(F['b_dir_planck'])
                    ][
                        [
                            'cloud_name',
                            'mcf_slope',
                            'b_dir_planck',
                            'cloud_dir'
                        ]
                    ].copy(deep = True)
                    
        o1.table = o1.table.rename(columns={'b_dir_optical': 'b_orientation', 'cloud_dir': 'cloud_orientation'})
        p1.table = p1.table.rename(columns={'b_dir_planck': 'b_orientation', 'cloud_dir': 'cloud_orientation'})
        
        o1.calc_b_offset()
        p1.calc_b_offset()
        
        self.obs_set += [[o1, p1]]


        ###########################
        # Set 2, test 3 b_offset and ref 3
        ###########################
        o2, p2 = data(), data()
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
                    ].copy(deep = True)
                    
        p2.table = F[~np.isnan(F['sfr_3']) &
                     ~np.isnan(F['b_dir_planck'])
                    ][
                        [
                            'cloud_name',
                            'sfr_3',
                            'b_dir_planck',
                            'cloud_dir'
                        ]
                    ].copy(deep = True)
                    
        o2.table = o2.table.rename(columns={'b_dir_optical': 'b_orientation', 'cloud_dir': 'cloud_orientation', 'sfr_3': 'dgf'})
        p2.table = p2.table.rename(columns={'b_dir_planck': 'b_orientation', 'cloud_dir': 'cloud_orientation', 'sfr_3': 'dgf'})
        
        o2.calc_b_offset()
        p2.calc_b_offset()
        
        self.obs_set += [[o2, p2]]
        
        
        
        ###########################
        # Set 3, test 3 b_offset and ref 4
        ###########################
        o3, p3 = data(), data()
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
                    ].copy(deep = True)
                    
        p3.table = F[~np.isnan(F['sfr_4']) &
                     ~np.isnan(F['b_dir_planck'])
                    ][
                        [
                            'cloud_name',
                            'sfr_4',
                            'b_dir_planck',
                            'cloud_dir'
                        ]
                    ].copy(deep = True)
                    
        o3.table = o3.table.rename(columns={'b_dir_optical': 'b_orientation', 'cloud_dir': 'cloud_orientation', 'sfr_4': 'dgf'})
        p3.table = p3.table.rename(columns={'b_dir_planck': 'b_orientation', 'cloud_dir': 'cloud_orientation', 'sfr_4': 'dgf'})
        
        o3.calc_b_offset()
        p3.calc_b_offset()
        
        self.obs_set += [[o3, p3]]
        
        return

        ###########################
        # Set 4, test 1,2,3 b_offset, MCF slope, and ref 3
        ###########################
        o4, p4 = data(), data()
        o4.file = p4.file = 'set 1'
        
        o4.table = F[~F['cloud_name'].isin(['Orion A', 'Orion B']) &
                     ~np.isnan(F['mcf_slope']) &
                     ~np.isnan(F['sfr_3']) &
                     ~np.isnan(F['b_dir_optical'])
                    ][
                        [
                            'cloud_name',
                            'sfr_3',
                            'mcf_slope',
                            'b_dir_optical',
                            'cloud_dir'
                        ]
                    ].copy(deep = True)
                    
        p4.table = F[~np.isnan(F['sfr_3']) &
                     ~np.isnan(F['mcf_slope']) &
                     ~np.isnan(F['b_dir_planck'])
                    ][
                        [
                            'cloud_name',
                            'sfr_3',
                            'mcf_slope',
                            'b_dir_planck',
                            'cloud_dir'
                        ]
                    ].copy(deep = True)
                    
        o4.table = o4.table.rename(columns={'b_dir_optical': 'b_dir'})
        p4.table = p4.table.rename(columns={'b_dir_planck': 'b_dir'})
        
        o4.calc_b_offset()
        p4.calc_b_offset()
        
        self.obs_set += [[o4, p4]]
        
        return
        
        ###########################
        # Set 4, test 1,2,3 b_offset, MCF slope, and ref 3
        ###########################
        # ['Cloud', 'long axes P.A.', 'B-field P.A. PLANCK', 'B P.A. Starlight', 'MCF Slope', 'SFR ref 4',
        # 'SFR ref 3', 'b_offset_planck', 'b_offset_optical']
        set4_planck = data()
        set4_planck.file = 'set4 Planck'
        
        condition1 = np.logical_and(~np.isnan(b_offset_planck), ~np.isnan(self['SFR ref 3']))
        condition2 = np.logical_and(~np.isnan(b_offset_planck), ~np.isnan(self['MCF Slope']))
        condition = np.logical_and(condition1, condition2)
        
        set4_planck.stats.b_offset = b_offset_planck[condition]
        set4_planck.stats.cloud_name = self['Cloud'][condition]
        set4_planck.stats.mcf_slope = self['MCF Slope'][condition]
        set4_planck.stats.mcf_area  = np.empty_like(b_offset_planck[condition])
        set4_planck.stats.mcf_area[:] = np.nan
        set4_planck.stats.dgf       = self['SFR ref 4'][condition]

        
        set4_optical = data()      
        set4_optical.file = 'set4 optical'
        
        condition1 = np.logical_and(self['Cloud'] != 'Orion A', self['Cloud'] != 'Orion B')
        condition2 = np.logical_and(~np.isnan(b_offset_optical), ~np.isnan(self['SFR ref 3']))
        condition3 = np.logical_and(~np.isnan(b_offset_optical), ~np.isnan(self['MCF Slope']))
        condition = np.logical_and(condition1, condition2)
        condition = np.logical_and(condition, condition3)
        
        set4_optical.stats.b_offset = b_offset_optical[condition]
        set4_optical.stats.cloud_name = self['Cloud'][condition]
        set4_optical.stats.mcf_slope = self['MCF Slope'][condition]
        set4_optical.stats.mcf_area  = np.empty_like(b_offset_optical[condition])
        set4_optical.stats.mcf_area[:] = np.nan
        set4_optical.stats.dgf       = self['SFR ref 3'][condition]
        
        
        ###########################
        # Set 5, test 1,2,3 b_offset, MCF slope, and ref 4
        ###########################
        # ['Cloud', 'long axes P.A.', 'B-field P.A. PLANCK', 'B P.A. Starlight', 'MCF Slope', 'SFR ref 4',
        # 'SFR ref 3', 'b_offset_planck', 'b_offset_optical']
        set5_planck = data()
        set5_planck.file = 'set5 Planck'
        
        condition1 = np.logical_and(~np.isnan(b_offset_planck), ~np.isnan(self['SFR ref 4']))
        condition2 = np.logical_and(~np.isnan(b_offset_planck), ~np.isnan(self['MCF Slope']))
        condition = np.logical_and(condition1, condition2)
        
        set5_planck.stats.b_offset = b_offset_planck[condition]
        set5_planck.stats.cloud_name = self['Cloud'][condition]
        set5_planck.stats.mcf_slope = self['MCF Slope'][condition]
        set5_planck.stats.mcf_area  = np.empty_like(b_offset_planck[condition])
        set5_planck.stats.mcf_area[:] = np.nan
        set5_planck.stats.dgf       = self['SFR ref 4'][condition]

        
        set5_optical = data()      
        set5_optical.file = 'set5 optical'
        
        condition1 = np.logical_and(self['Cloud'] != 'Orion A', self['Cloud'] != 'Orion B')
        condition2 = np.logical_and(~np.isnan(b_offset_optical), ~np.isnan(self['SFR ref 4']))
        condition3 = np.logical_and(~np.isnan(b_offset_optical), ~np.isnan(self['MCF Slope']))
        condition = np.logical_and(condition1, condition2)
        condition = np.logical_and(condition, condition3)
        
        set5_optical.stats.b_offset = b_offset_optical[condition]
        set5_optical.stats.cloud_name = self['Cloud'][condition]
        set5_optical.stats.mcf_slope = self['MCF Slope'][condition]
        set5_optical.stats.mcf_area  = np.empty_like(b_offset_optical[condition])
        set5_optical.stats.mcf_area[:] = np.nan
        set5_optical.stats.dgf       = self['SFR ref 4'][condition]        
        
        
        return [[set1_planck, set1_optical], 
                [set2_planck, set2_optical], 
                [set3_planck, set3_optical],
                [set4_planck, set4_optical],
                [set5_planck, set5_optical]]

    def calc_b_offset(self):
        
        self['b_offset'] = np.abs(self['cloud_orientation'] - self['b_orientation'])
        self['b_offset'][self['b_offset'] > 90] = 180 - self['b_offset'][self['b_offset'] > 90]

    def calc_dgf(self):
        self.table['dgf'] = self.table['dense_mass'] / self.table['cloud_mass']

    def quick_view(self, fig = None, ax = None):
        
        if fig == None:
            fig, ax = plt.subplots(1, 3, figsize = (9, 3), dpi = 100)
        
        
        
        t = self.table
        
        
        s = 1000/t.shape[0]
        
        keys = ['mcf_area', 'mcf_slope', 'dgf']
        
        for i, key in enumerate(keys):
            ax[i].set_title(key)
            if key in self.table.keys():
                ax[i].scatter(t['b_offset'], t[key], s = s)            
        
        return fig, ax
    
    
    def __getitem__(self, item):
        return self.table[item]
    
    def __setitem__(self, key, val):
        self.table[key] = val
    