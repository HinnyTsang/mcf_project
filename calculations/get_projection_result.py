"""
    Plot the scatter plot of the projected parameters.
    see check_projection_parameters.ipynb
    
    Author: Hinny Tsang
    Last Edit: 2022-04-21
"""
import os
import sys
sys.path.append('..')
import functions.plot as plot
import functions.data_class as data_class
from typing import List

def main(file_path: str, versions: List[str], out_paths: List[str])->None:
    """
        plot the graphs for different version one-by-ones.
    
    """    
    for version, out_path in zip(versions, out_paths):
        
        para_file = os.path.join(file_path, version[0], 'main.h5')
        perp_file = os.path.join(file_path, version[1], 'main.h5')
        
        para_data = data_class.main_data(para_file)
        perp_data = data_class.main_data(perp_file)
        
        # value from check_dense_gas_fraction.ipynb
        para_data.calc_dgf(dense_gas=4413.0297186780,
                        total_cloud_mass=11002.1148454371,
                        contour_cloud_mass=3642.722286)
        perp_data.calc_dgf(dense_gas=950.9536251046,
                        total_cloud_mass=11002.1148454371,
                        contour_cloud_mass=3642.722489)
        
        # plot mcf slope vs cloud-field offset
        plot.plot_scatter_with_hist(
            x1=para_data['b_offset'], x2=perp_data['b_offset'],
            y1=para_data['mcf_slope'], y2=perp_data['mcf_slope'],
            label1="Parallel cloud", label2="Perpendicular cloud",
            abbr1 = "Para. c.", abbr2= "Perp. c.",
            xlabel = "Cloud-field offset [degree]",
            ylabel = "MCF Slope [column density$^{-1}$]",
            xlim = (0, 90), ylim = (0, 0.005),
            out_file=os.path.join('../images/', out_path, 'mcf_slope_versus_b_offset.png')
        )
        
        # plot dgf vs cloud-field offset
        plot.plot_scatter_with_hist(
            x1=para_data['b_offset'], x2=perp_data['b_offset'],
            y1=para_data['dgf'], y2=perp_data['dgf'],
            label1="Parallel cloud", label2="Perpendicular cloud",
            abbr1 = "Para. c.", abbr2= "Perp. c.",
            xlabel = "Cloud-field offset [degree]",
            ylabel = "Dense gas fraction",
            xlim = (0, 90), ylim = (0, 0.6),
            out_file=os.path.join('../images/', out_path, 'dgf_versus_b_offset.png')
        )
        

if __name__ == "__main__":
    
    projection_path = '../h5_projected'
    versions = [
        ['g1040_0016_binary_search_100_fib', 'g1041_9015_by_mass_100_fib'],
        ['g1040_0016_binary_search_10000_fib', 'g1041_9015_by_mass_10000_fib'],
        ['g1040_0016_binary_search_10000_ran', 'g1041_9015_by_mass_10000_ran']
    ]
    
    out_paths = ['fib_100', 'fib_10000', 'ran_10000']
    
    main(projection_path, versions, out_paths)

