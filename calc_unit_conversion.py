"""
    Astronomical unit conversion
    
    Author: Hinny Tsang
    Last Edit: 2022-03-29
    
    [code] units is the units using in Scorpio.
"""

import numpy as np


# Physical constants
SOLARMASS = 1.9e30                   # kg
PC        = 3.086e16                 # m
YEAR      = 3.156e7                  # sec
MU0       = 1.256637062 * 10 ** - 6  # N.A-2 == H.m-1
HMASS     = 1.6735575 * 10 ** -27    # kg
GRAVCONST = 4.3011 * 10 ** -3        # km2.pc.s-2.Msun-1


# Code units: units using in Scorpio code.

# Time 
def time_code_to_Myrs(time):
    """
    convert time from code unit to Myr
    
    
    :param time: time
    :type time: float / numpy array
    :return: time in Myr
    :return type: auto
    """
    time_ = time * 10 ** 6
    time_ /= 9.78e5
    
    return time_

def time_Myrs_to_code(time):

    time_ = time * 9.78e5
    time_ /= 10 ** 6
    
    return time_


# Density
def volume_den_Msun_per_pc3_to_kg_per_m3(density):
    """
    :param density: Density of Hydrogen molecules
    """
    density_  = density * SOLARMASS # kg.pc-3
    density_ /= PC ** 3   # kg.m-3
    return density_

def volume_den_Msun_per_pc3_to_number_den_H2_per_cm3(density, mu = 2.3):
    """
    :param density: Density of Hydrogen molecules
    """
    H2MASS    = HMASS * mu # kg
    density_  = density * SOLARMASS # kg.pc-3
    density_ /= PC ** 3   # kg.m-3
    density_ /= 100 ** 3  # kg.cm-3
    density_ /= H2MASS    # H2.cm-3
    return density_

def number_den_H2_per_cm3_to_volume_den_Msun_per_pc3(density, mu = 2.3):
    """
    :param density: Number density of Hydrogen molecules
    """
    H2MASS    = HMASS * mu # kg
    density_  = density * H2MASS    # kg.cm-3
    density_ /= SOLARMASS # Msun.cm-3
    density_ *= 100 ** 3  # Msun.m-3
    density_ *= PC ** 3   # Msun.pc-3
    return density_    
        
def number_den_H2_per_cm2_to_column_den_Msun_per_pc2(density, mu = 2.3):
    """
    :param density: Number density of Hydrogen molecules
    """
    H2MASS    = HMASS * mu # kg
    density_  = density * H2MASS    # kg.cm-3
    density_ /= SOLARMASS # Msun.cm-3
    density_ *= 100 ** 2  # Msun.m-3
    density_ *= PC ** 2   # Msun.pc-3
    return density_    
    
def column_den_Msun_per_pc2_to_number_den_H2_per_cm2(density, mu = 2.3):
    """
    :param density: Column density of Hydrogen molecules
    """
    H2MASS   = HMASS * mu
    density_ = density * SOLARMASS # kg.pc-2
    density_ /= PC ** 2   # kg.m-2
    density_ /= 100 ** 2  # kg.cm-2
    density_ /= H2MASS    # H2.cm-2
    return density_

# Extinction 
# NH to Av
def column_density_H2_per_cm2_to_extinction_mag(N):
    return N / (1.37e21)

def extinction_mag_to_column_density_H2_per_cm2(A):
    return A * (1.37e21)

# CD to Av
def column_denisty_Msun_per_pc2_to_extinceion_mag(n, mu):
    N = column_den_Msun_per_pc2_to_number_den_H2_per_cm2(n, mu)
    return column_density_ncc_to_extinction_mag(N) 

def extinction_mag_to_column_denisty_Msun_per_pc2(A, mu):
    N = extinction_mag_to_column_density_H2_per_cm2(A)
    return number_den_H2_per_cm2_to_column_den_Msun_per_pc2(N, mu)
    

def code_energy_density_to_joule_per_m3(energy_density):
    """
    Convert code unit: Msun.km^2.s-2.pc-3

    to
    physical unit: J.m-3 == kg.m^2.s-2.m-3

    Equivalent unit of Joule:

        N.m
        kg.m^2.s-2
        Pa.m^3
        W.s
        C.V

    :param energy_density:
    :return: energy_density
    """

    energy_density_ = energy_density * SOLARMASS  # kg.km^2.s-2.pc-3
    energy_density_ *= 1000 ** 2  # kg.m^2.s-2.pc-3
    energy_density_ /= PC ** 3  # kg.m^2.s-2.m-3 == J.m-3

    return energy_density_

def code_b_field_to_code_b_energy_density(mx, my, mz):
    """
    calculate magnetic energy from magnetic field

    Energy density code unit: Msun.km^2.s-2.pc-3
    :param mx:
    :param my:
    :param mz:
    :return:
    """

    return (mx ** 2 + my ** 2 + mz ** 2) / 2

def Tesla_to_muG(b_tesla):
    """
    Tesla to muG

    1 T = 10^4 G
    1 G = 10^6 muG

    :param b_tesla:
    :return:
    """
    return b_tesla * 10 ** 4 * 10 ** 6

def Tesla_to_G(b_tesla):
    """
    Tesla to muG

    1 T = 10^4 G

    :param b_tesla:
    :return:
    """
    return b_tesla * 10 ** 4

def b_phy_to_G(b_code):
    """
    [Bphy]= 1.0 [sqrt(Msun)*km/(s*pc^1.5)] = 8.0405e-7 [G,sqrt(g/cm)/s]

    SOLARMASS = 1.9e30  # kg
    PC = 3.086e16  # m
    YEAR = 3.156e7  # sec
    MU0 = 1.256637062 * 10 ** - 6  # N.A-2 == H.m-1

    :param b_code:   sqrt(g)*km/(s*pc^1.5)
    :return: b_code: G
    """
    b_code_ = b_code * np.sqrt(SOLARMASS * 1000)  # sqrt(g)*km/(s*pc^1.5)
    b_code_ *= 100000  # sqrt(g)*cm/(s*pc^1.5)
    b_code_ /= (PC * 100) ** 1.5  # sqrt(g)*cm/(s*cm^1.5) = sqrt(g/cm)/s

    return b_code_

def b_code_to_phy(b_code):
    """
    Bcode=Bphy/sqrt(4*pi)
    :param b_code:
    :return: b_code
    """
    return b_code * np.sqrt(4 * np.pi)

def b_phy_to_code(b_code):
    """
    Bcode=Bphy/sqrt(4*pi)
    :param b_code:
    :return: b_code
    """
    return b_code / np.sqrt(4 * np.pi)

def b_code_to_muG(b_code):
    """
    :param b_code:
    :return b_muG:
    """
    b_phy = b_code_to_phy(b_code)
    b_G = b_phy_to_G(b_phy)
    b_muG = b_G * 10 ** 6

    return b_muG

def b_G_to_b_phy(b_G):
    """
    [Bphy]= 1.0 [sqrt(Msun)*km/(s*pc^1.5)] = 8.0405e-7 [G,sqrt(g/cm)/s]

    SOLARMASS = 1.9e30  # kg
    PC = 3.086e16  # m
    YEAR = 3.156e7  # sec
    MU0 = 1.256637062 * 10 ** - 6  # N.A-2 == H.m-1

    :param b_code:   sqrt(g)*km/(s*pc^1.5)
    :return: b_code: G
    """
    b_temp = b_G * (PC * 100) ** 1.5
    b_temp /= 100000
    b_temp /= np.sqrt(SOLARMASS * 1000)
    
    return b_temp

def b_muG_to_b_code(b_muG):

    """
    :param b_muG:
    :return b_code:
    """
    b_G = b_muG / 10 ** 6
    b_phy = b_G_to_b_phy(b_G)
    b_code = b_phy_to_code(b_phy)
    
    return b_code
    
def b_code_to_Tesla(b_code):

    b_muG = b_code_to_muG(b_code)
    
    b_Tesla = b_muG / 10 ** 4 / 10 ** 6
    
    return b_Tesla

def b_energy_density_from_b_G(bx, by, bz):
    """

    :param bx_:
    :param by_:
    :param bz_:
    :return:
    """

    # 1: Convert B to T from G
    bx_ = bx / 10 ** 4
    by_ = by / 10 ** 4
    bz_ = bz / 10 ** 4

    # 2: calculate magnetic field energy density
    bsq = bx_ ** 2 + by_ ** 2 + bz_ ** 2
    e_den = bsq / (2 * MU0)

    return e_den


if __name__ == '__main__':
    # Test energy conversion
    b_code = 2.487
    b_phy = b_code_to_phy(b_code)
    b_G = b_phy_to_G(b_phy)

    print(f"B[phy]: {b_phy:.2e} [sqrt(Msun)*km/(s*pc^1.5)] = {b_G:.4e} [G,sqrt(g/cm)/s]")

    ene_den = b_energy_density_from_b_G(b_G, 0, 0)
    print(f"Energy density from b = {b_G:.4e} G is {ene_den:.4e} J.m-3")

    ene_den_code = (b_code) ** 2 / 2
    print(f"                        {b_G:.4e} G is {ene_den_code:.4e} code")

    ene_den_cal = code_energy_density_to_joule_per_m3(ene_den_code)
    print(f"                        {b_G:.4e} G is {ene_den_cal:.4e} J.m-3")