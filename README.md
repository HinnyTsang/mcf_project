# MCF Project -- Main structure

codes written in python.

project structure.

1. Create input parameters for Scorpio.
2. MHD simulation done by Scorpio.
3. Calculate the maximum contour of Scorpio output file, and store the data.
    - Maximum contour is done in [get_data_in_contour.py](get_data_in_contour.py)
    - The Dense gas fraction is also calculated using the Scorpio output file. see [check_dense_gas_fraction.ipynb](validation/check_dense_gas_fraction.ipynb)
    - 
4. Projection of the data within the maximum contour, the following parameters are calculated.
    1. MCF
    2. cloud mass, defined by column density $\geq A_v=2(30M_\odot pc^{-2})$.


<br><br><br>

# Datatype and usage of every function.
1. Scorpio output file reader turns the data in to a `dict`.
2. Python scripts start with `calc` written for run calculation, `get` written for functions that will be import.