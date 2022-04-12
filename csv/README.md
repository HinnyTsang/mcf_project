# Updated 12-4-2022.

Three csv files containing the data.

- simulation_output_fib_100_20220404.csv
    - 'uniform' data with 100 samples.
- simulation_output_fib_10000_20220404.csv 
    - 'uniform' data with 10000 samples.
- simulation_output_ran_10000_20220404.csv 
    - random data with 10000 samples.

Format of the data file:

# column 1 'mcf_slope': 
- slope of the mass cumulative function
- unit is [mag $^{-1}$]

# column 2 'mcf_area':
- area under the mass cumulative function
- unit is [mag]

# column 3 'cloud_field_offset': 
- offset between magentic field and orientation of cloud
- unit is radian.

# column 4 'dense_mass_fraction': 
- dimensionless ratio.
- projected cloud mass is the sum of mass with extinction $\geq 2$ [mag] (column density >= 30 [Msun pc $^{-2}$]) for each projection.
- dense gas mass is the sum of mass with column density $\geq 277.60$ [Msun pc $^{-2}$] for parallel and $\geq 252.71$ for perpendicular cloud [Msun pc $^{-2}$], calculate using the whole simulation cube, the values are calculated by the relation from the density weighted mean magnetic field. 
$$ \bar{B}  = ||\frac{\sum_i^N \rho_iB_{ji}}{\sum_i^N \rho_i} \hat{x}_j ||$$
- $\bar{B} = 28.23\mu$ G for parallel clouds, and $\bar{B} = 25.39 \mu$ G for perpendicular cloud.
- Conversion between magnetic field and critical column density is [ref](https://doi.org/10.1093/mnras/stt1849) eq (II):
$$B=1.1\times 10^{21} N_H cm^{-2}$$ 
- The total cloud mass is $11002.11$ [Msun] for the both cloud.
- The contour cloud mass is $3642.72$ [Msun] for the both cloud (not exactly identical, but equal in this sig. fig.).
- The dense mass ratio is calculated by:
$$ 
    DGF = \frac{\text{dense gas mass}}{\text{total cloud mass}} \frac{\text{contour cloud mass}}{\text{ projected cloud mass}}
$$
- Conversion between $N_H cm^{-2}$ to $M_\odot pc^{-2}$ is.
$$ M_\odot pc^{-2} =  8.70\times 10^{19} N_H cm^{-2} $$

# column 5 'cloud_type':  
- intrinsic orientation of the clouds. (0 = parallel cloud, 1 = perpendicular cloud) 


$$\text{Number of Young Stellar Obj (YSO)} \times \text{typical mass of YSO} (0.5 \text{M}_\odot) / \text{typical age of YSO} (2\text{ Myrs}) \times 100\%$$