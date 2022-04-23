# MCF Project -- Main structure

Codes are written in python.

## Project Structure
1. Create input parameters for Scorpio. (not yet including here)
2. MHD simulation done by Scorpio.
3. Calculate the maximum contour of Scorpio output file, and store the data.
    - Maximum contour is done in [get_data_in_contour.py](calculations/get_data_in_contour.py)
    - The Dense gas fraction is also calculated using the Scorpio output file. see [check_dense_gas_fraction.ipynb](validation/check_dense_gas_fraction.ipynb)

    <br>
4. Projection of the data within the maximum contour, the following parameters are calculated. see[get_projection_mpi.py](calculations/get_projection_mpi.py)
    1. MCF Slope
    2. Cloud mass, defined by column density $\geq A_v=2(30M_\odot pc^{-2})$.
    3. Cloud orientation and magnetic field orientation.
    - Projected parameters are visualised in [check_projection_parameters.ipynb](validation\check_projection_parameters.ipynb)
    
    <br>
5. Statistical test for the three parameters defined above.
    - Calculated in [get_stats_test.py](calculations\get_stats_test.py)
    - Rough draft can be find in [check_stat_test.ipynb](validation\check_stat_test.ipynb)
    - Result is plotted by [get_stats_test_result.py](calculations\get_stats_test_result.py)
        - Example - [check_stat_test_result.ipynb](validation\check_stat_test_result.ipynb)

