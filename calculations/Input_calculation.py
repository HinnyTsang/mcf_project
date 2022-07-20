# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 19:19:20 2021

@author: hinhi
"""



#%% Import libraries
import sys
sys.path.append(r"C:\Users\hinhi\OneDrive - HKUST Connect\MPhil\Scorpio_tool_codes\Scorpio_1.5\src")
from unit_conversion import b_muG_to_b_code, time_Myrs_to_code
from Tools_Code import *
from GPE_calculation import CalGPE
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
import sympy as sp
from sympy.physics import units
from sympy.physics.units.systems import SI

### use to set front color
sp.init_printing(use_latex='svg', forecolor='black',scale=1)

#%%

from sympy.physics.units import Quantity
def unit(expr):
    return expr.subs({x: 1 for x in expr.args if not x.has(Quantity)})



#%% Constants


### boltzmann_constant
kB = units.boltzmann_constant.convert_to(units.meter **2 * units.kg * units.second ** -2 * units.K ** -1)

### Magnetic constant
mu = units.magnetic_constant.convert_to(units.m * units.kg * units.s** -2 * units.ampere ** -2)

### Solar mass
Msun = units.Quantity(r"Msun")
SI.set_quantity_dimension(Msun, units.mass)
SI.set_quantity_scale_factor(Msun, 1.98847*10**30*units.kg)

### Parsec
pc = units.Quantity("pc")
SI.set_quantity_dimension(pc, units.length)
SI.set_quantity_scale_factor(pc, 3.08567758*10**16*units.meter)


### Gravitational constant
G = units.gravitational_constant.convert_to(units.km**2*pc/(units.s**2*Msun))


### boltzmann_constant
kB = units.boltzmann_constant.convert_to(units.meter **2 * units.kg * units.second ** -2 * units.K ** -1)

### Hydrogen Mass
Hmass = 1.6735575 * 10 ** -24 * units.gram


# print('Hydrogen mass is =', Hmass)

#%% On the paper

gcm3 = 1.4*10**-19 * units.gram*units.cm ** -3

Msunpc3 = units.convert_to(gcm3, [Msun * pc ** -3])

print(Msunpc3)



#%% Variables to solve

### for center density
sym_rho_c = sp.Symbol('rho_c')

### for magnetic field strength
sym_B = sp.Symbol('B')


#%% Density profile
plt.rcParams['figure.figsize'] = [7,7]

def plummer_density(r, a, rho_c):
    
    A = (1 + (r/a) ** 2)
    
    return rho_c / A

r = np.arange(0, 10, 10/1000)

den_test = plummer_density(r, a = 1, rho_c = 1)
plt.plot(r, den_test)

plt.vlines(5, 0.01, 0.2, color = 'black', ls = 'dashed')
plt.hlines(0.05, 0, 5, color = 'black', ls = 'dashed')

plt.title("Radial density profile")

plt.ylabel(r'Relative density [$\rho_c$]')
plt.xlabel(r"Radial distance [pc]")
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
# plt.yscale('log')
# plt.xscale('log')
plt.xlim(0.01, 6)
plt.ylim(0.01, 1.05)

plt.show()



#%% Create sample

def create_unit_sample(nx, ny, nz, Lx, Ly, Lz, L_cloud, r_flat = 1):

    dx = Lx/(nx)
    dy = Ly/(ny)
    dz = Lz/(nz)
    
    x = np.arange(0, Lx, dx) - Lx/2 + dx/2
    y = np.arange(0, Ly, dy) - Ly/2 + dy/2
    z = np.arange(0, Lz, dz) - Lz/2 + dz/2

    X, Y, Z = np.meshgrid(x, y, z)
    
    R = np.sqrt(X ** 2 + Y ** 2)
    
    ### Desnity profile
    density = plummer_density(R, r_flat, 1)
    
    ### maximum density in this domain
    max_den = plummer_density(5, r_flat, 1)
    
    ### Exponential decay in the two end
    density = np.multiply(density, np.where(np.abs(Z) > L_cloud/2, np.exp(-(L_cloud/20)*(abs(Z) - L_cloud/2)), 1))
    
    ### fill minimum value
    density[density < max_den] = max_den
    
    return density

#%%

a = create_unit_sample(nx = 200, ny = 200, nz = 400, Lx = 10, Ly = 10, Lz = 20, L_cloud = 15, r_flat = 1)

    
#%% Input parameters

### nMesh of the domain
N = 128

nx, ny = N/2, N/2
nz = N

### length of the domain [pc]
Lx, Ly = 10, 10 
Lz = 20

### length of the cloud [pc]
L_cloud = 15

dx = Lx/nx 
dy = Ly/ny 
dz = Lz/nz 

dV = dx*dy*dz * pc ** 3

r_flat = 1

# sonic mach number
Mach = 10 

### Temperature
T = (25) * units.kelvin


#%% Calculate density profile
### unit density [no unit]
unit_den = create_unit_sample(nx, ny, nz, Lx, Ly, Lz, L_cloud, r_flat = r_flat)

### unit mass [pc-3]
unit_mass = np.sum(unit_den) * dV 

### total mass [Msun]
mass = unit_mass * sym_rho_c

print("Mass in terms of center density is", mass)



#%% calculate the aspect ratio of the cloud 

def cov(x, y, p):
    EX = np.sum(x*p)/np.sum(p)
    EY = np.sum(y*p)/np.sum(p)
    covXY = np.sum(p*(x-EX)*(y-EY))/np.sum(p)
    return covXY

def var(x,p):
    EX = np.sum(x*p)/np.sum(p)
    varX = np.sum(p*(x-EX)**2)/np.sum(p)
    return varX

def covMat(x,y,z,p):
    varX = var(x,p)
    varY = var(y,p)
    varZ = var(z,p)
    covXY = cov(x,y,p)
    covXZ = cov(x,z,p)
    covYZ = cov(y,z,p)
    return np.array([[varX,covXY,covXZ],[covXY, varY, covYZ], [covXZ,covYZ,varZ]])

def calcComponentsWeighted(x,y,z,p):
    """
    SVD
    Arguments:
    x: x data
    y: y data
    z: z data
    p: probability
    Returns:
    [variance1, variance2], [vector1[2],vector2[2]]
    """
    covMatrix = covMat(x,y,z,p)
    val,vec=np.linalg.eig(covMatrix)

    vec1= vec[:,0]
    vec2= vec[:,1]
    vec3= vec[:,2]

    return val, [vec1, vec2, vec3]

#%%
x = np.linspace(-Lx/2, Lx/2, int(nx)+1)[1:] - dx/2
y = x.copy()
z = np.linspace(-Lz/2, Lz/2, int(nz)+1)[1:] - dz/2

X, Y, Z = np.meshgrid(x, y, z)



thres = 0.2

x = X[unit_den > thres]
y = Y[unit_den > thres]
z = Z[unit_den > thres]
d = unit_den[unit_den > thres]

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_xlim(-10,10)
# ax.set_ylim(-10,10)
# ax.set_zlim(-10,10)
# ax.scatter(x, y, z, c =d )
# plt.show()

val, vec = calcComponentsWeighted(x, y, z, d)

print(f"Aspect ratio is: {np.sqrt(max(val)/min(val)):.2f}")


#%% plot density
plt.rcParams['figure.figsize'] = [15,7]

extent = [[-20, 20, -10, 10], [-20, 20, -10, 10], [-10, 10, -10, 10]]
title = ["x", "y", "z"]

fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 2, 1]})


for los in range(3):
    ax[los].imshow(np.sum(unit_den, los), extent = extent[los])
    # if los < 2:
    #     for i in np.linspace(-5, 5, 10):
    #         ax[los].hlines(i, -10, 10, color = 'white')
    if los == 1:
        for i in np.linspace(-20, 20, 20):
            ax[los].vlines(i, -10, 10, color = 'white')
    elif los == 2:
        for i in np.linspace(-10, 10, 10):
            ax[los].hlines(i, -10, 10, color = 'white')
    ax[los].grid(False)
    ax[los].set_title("LOS = " + title[los])
    ax[los].set_yticks([-10, 0, 10], [-10, 0, 10])
    # ax[los].set_xlabel('pc')
plt.show()



#%% Calculate GPE 

### Calculate the portential field [without Gravitional constant]
GP = CalGPE(unit_den, cell_size = dx)

### Calculate total energy
GPE = 1/2 * G * np.sum(GP*unit_den) * dV * sym_rho_c ** 2  * pc ** 2
print("Total gravitational potential energy is ", GPE)

#%% Speed of sound


print('temperature is =', T)


### sound speed
sound_speed = sp.sqrt(kB * T / (2.3 * Hmass)).n()
sound_speed = units.convert_to(sound_speed, [units.km / units.s])

print('sound speed is =', sound_speed)


#%% Turbulent speed

### Mach number is 10
v = Mach*sound_speed

#%% Solve for center density


### turbulent kinetic energy
KE = 1/2* mass * v ** 2
print("Turbulent kinetic energy is ", KE)
print("Gravitional potential energy is", GPE)


### center density
center_density = sp.solve(KE + GPE, sym_rho_c)[1]
print("Center density is", center_density)

### in units of number density
center_density_npcc = units.convert_to(center_density / Hmass / 2.3, [units.cm ** -3])
print("     in H2.cc-1, ", center_density_npcc)

#%% Total Mass is

Total_mass = mass.subs(sym_rho_c, center_density)
print("Total mass of the cube is", Total_mass)

#%% Calculate magnetic field strength

### Expression of magnetic energy
BE = sym_B ** 2 / 2 / mu  *  Lx*Ly*Lz * pc ** 3
print("Magneic energy is", BE)


### Solve for B
B = sp.solve(BE + GPE.subs(sym_rho_c, center_density), sym_B)[1]
print(B)

B = units.convert_to(B, [units.tesla])

print("Magnetic field strength is", B)
print("Magnetic field strength is", B/units.tesla * 10 ** 10, "muG")

B_code = b_muG_to_b_code(B/units.tesla * 10 ** 10)

print("Magnetic field strength is", B_code, "[code]")

#%% Free fall time

t_ff = sp.sqrt(3 * sp.pi / 32 / G / center_density)
t_ff = units.convert_to(t_ff, units.year)

print("Free fall times is ", time_Myrs_to_code(t_ff.n()/units.year / 10 ** 6), '[code unit]')


#%% TrueLove criteria


### find minimum grid size
dL = np.min([dx, dy, dz]) * pc / 2
dL = 30/1440 * pc
# dL =  * pc

print("Minimum distance is ", dL/pc, "[pc]")


### minimum Jeans length
Jlen_min = dL * 5

print("Minimum Jeans length is ", Jlen_min/pc, "[pc]")


### Calculate maximum density
Jlen_formula = sp.sqrt(sp.pi * sound_speed ** 2/ ( G * sym_rho_c ))

max_rho = sp.solve(Jlen_formula - Jlen_min, sym_rho_c)[0]

print("Maximum density that can be resolved is ", max_rho/Msun*pc**3, "[Msun.pc-3]")
print("Maximum density that can be resolved is ", volume_den_Msun_per_pc3_to_number_den_H2_per_cm3(max_rho/Msun*pc**3, mu = 2.3), "[H2.cc-1]")


#%% RMS density is
RMS_unit_den = np.sqrt(np.mean(unit_den ** 2)) * center_density / Msun * pc ** 3 
RMS_unit_den_cc = volume_den_Msun_per_pc3_to_number_den_H2_per_cm3(RMS_unit_den, mu = 2.3)
print("RMS density is                         \n\t\t%.4f" % (RMS_unit_den_cc), "[H2.cc-1]")
print("\t\t%.4f" % (RMS_unit_den), "[Msun.pc-3]")


#%% Print all result 

print("-"*50)
print("Dimension (X Y Z) of the cube is        \n\t\t(%.2f, %.2f, %.2f)" % (Lx, Ly, Lz))
print("R_flat is                               \n\t\t%.4f" % (r_flat), "pc")
print("-"*50)
print("Length is                               \n\t\t%.4f" % (L_cloud), "pc")

print("The result is")
print("Sound speed is                          \n\t\t%.4f" % (sound_speed/units.km*units.s ), "[km.s-1]")
print("Center density is                       \n\t\t%.4f" % (center_density/Msun * pc**3) , "[Msun.pc-3]")
print("\t\t%.4f" % volume_den_Msun_per_pc3_to_number_den_H2_per_cm3(center_density/Msun * pc**3) , "[H2.cc-1]")
print("Magnetic field strength is              \n\t\t%.4f" % B_code, "[code]")
print("\t\t%.4f" % b_code_to_muG(B_code), "[muG]")
print("Turbulence energy per mass is           \n\t\t%.4f" % (KE.subs(sym_rho_c, center_density)/Total_mass / units.km**2 * units.seconds ** 2 ), "[km2.s-2]")
print("Free fall times is                      \n\t\t%.4f" % (time_Myrs_to_code(t_ff.n()/units.year / 10 ** 6)), '[code unit]')
print("\t\t%.4f" % (t_ff.n()/units.year/ 10 ** 6 ), '[yr]')
print("Maximum density that can be resolved is \n\t\t%.4f" % (max_rho/Msun*pc**3), "[Msun.pc-3]")
print("Turbulent energy per unit mass is       \n\t\t%.4f" % ((Mach*sound_speed/units.km*units.s)**2/2), ["km2.s-2"])

#%%

r_cylinder = 4 * pc
h_cylinder = 18 * pc

# Total Mass
Total_mass = np.sum(unit_den)*center_density*dV
print("Total mass is                           \n\t\t%.4e" % (Total_mass/Msun), "[Msun]")



# Volume
volume_uc = np.pi * r_cylinder ** 2 * h_cylinder
print("Volume is                               \n\t\t%.4e" % (volume_uc/pc**3), "[pc3]")




# Density
density_uc = Total_mass/ volume_uc
print("Mean density                            \n\t\t%.4f" % (density_uc/Msun*pc**3), "[Msun.pc-3]")
print("\t\t%.4f" % volume_den_Msun_per_pc3_to_number_den_H2_per_cm3(density_uc/Msun*pc**3) , "[H2.cc-1]")



t_ff = sp.sqrt(3 * sp.pi / 32 / G / density_uc)
t_ff = units.convert_to(t_ff, units.year)

print("Free fall times is %.4f" % time_Myrs_to_code(t_ff.n()/units.year / 10 ** 6), '[code unit]')



#%%

r = np.arange(0, 5, 0.1) + 0.01
denn = np.ones(r.shape) * density_uc / Msun * pc**3

zz = np.zeros(r.shape)
zz[r>4] = zz[r>4] - 4

denn = denn * np.exp(-zz)

plt.plot(r, denn)




#%%




#%%

den_H2cc    = 100
b_muG       = 100

den_Msunpc3 = number_den_H2_per_cm3_to_volume_den_Msun_per_pc3(den_H2cc, 2.3)
b_code      = b_muG_to_b_code(b_muG)

### Setting using
print(den_Msunpc3)
print(b_code)


#%% Parallel case

dx = Lx/(nx)
dy = Ly/(ny)
dz = Lz/(nz)

x = np.arange(0, Lx, dx) - Lx/2 + dx/2
y = np.arange(0, Ly, dy) - Ly/2 + dy/2
z = np.arange(0, Lz, dz) - Lz/2 + dz/2

X, Y, Z = np.meshgrid(x, y, z)

#%% Create sample
import numpy_indexed as npi

den = unit_den * den_Msunpc3
B   = np.ones(den.shape) * b_code
Rz  = np.sqrt(X ** 2 + Y ** 2)
Rz  = np.mean(Rz, 2)
Z_mod = Z.copy()
Z_mod[Z > 15 / 2] = Z_mod[Z > 15 / 2] - 15/2
Z_mod[Z < -15 / 2] = Z_mod[Z < -15 / 2] + 15/2
Z_mod[np.abs(Z) < 15/2] = 0

Rx  = np.sqrt(Y ** 2 + Z_mod ** 2)
Rx  = np.mean(Rx, 1)

#Bz = np.mean(B, 2)
#Bx = np.mean(B, 1)

Bz = np.sum(B * den, 2) / np.sum(den, 2)
Bx = np.sum(B * den, 1) / np.sum(den, 1)
#%%

Bphy = sp.sqrt(Msun)*units.km/(units.s*pc**1.5)

m2flux_units = Msun / (Bphy * pc ** 2)

crit_m2flux = 1 /(2*np.pi*sp.sqrt(G))

ratio = m2flux_units/crit_m2flux

ratio = float(ratio)

#%%
print("Los = z")
plt.imshow(Rz)
plt.title("Los = z")
plt.show()

den_z = np.sum(den, 2) * dz

Rz_avg, Bz_avg = npi.group_by(Rz.flatten()).sum(Bz.flatten())
Rz_avg, denz_avg = npi.group_by(Rz.flatten()).sum(den_z.flatten())

Mass = np.add.accumulate(denz_avg) * dx * dy

Flux = np.add.accumulate(Bz_avg) * dx * dy

m2flux = Mass / Flux  / np.sqrt(4 * np.pi) /ratio

fig, ax = plt.subplots(3, 1)

ax[0].plot(Rz_avg, m2flux)
ax[0].hlines(np.max(m2flux), 0, 7, label = "Maximum mass to flux ratio is %.2f" % np.max(m2flux))
ax[0].hlines(np.min(m2flux), 0, 7, label = "Minimum mass to flux ratio is %.2f" % np.min(m2flux))
ax[0].hlines(m2flux[np.logical_and(Rz_avg > 2.49, Rz_avg < 2.5)],
             0, 7,label = r"At $r_{flat}$ %.2f" % m2flux[np.logical_and(Rz_avg > 2.49, Rz_avg < 2.5)])
ax[0].set_title("Mass to flux ratio")
ax[0].legend()
# ax[1].plot(r_bins, b_bins, label = "T = %.2f" % data['t'][:])
ax[1].plot(Rz_avg, Flux)
ax[1].set_title("Flux")

ax[2].plot(Rz_avg, Mass)
ax[2].set_title("Mass")

#%%
plt.imshow(Rx)
plt.title("Los = x")
plt.show()

den_x = np.sum(den, 1) * dz

Rx_avg, Bx_avg = npi.group_by(Rx.flatten()).sum(Bx.flatten())
Rx_avg, denx_avg = npi.group_by(Rx.flatten()).sum(den_x.flatten())

Mass = np.add.accumulate(denx_avg) * dx * dy

Flux = np.add.accumulate(Bx_avg) * dx * dy

m2flux = Mass / Flux  / np.sqrt(4 * np.pi) /ratio

fig, ax = plt.subplots(3, 1)

ax[0].plot(Rx_avg, m2flux)
ax[0].hlines(np.max(m2flux), 0, np.max(Rx_avg), label = "Maximum mass to flux ratio is %.2f" % np.max(m2flux))
ax[0].hlines(np.min(m2flux), 0, np.max(Rx_avg), label = "Minimum mass to flux ratio is %.2f" % np.min(m2flux))
ax[0].hlines(m2flux[np.logical_and(Rx_avg > 2.49, Rx_avg < 2.5)],
             0, np.max(Rx_avg),label = r"At $r_{flat}$ %.2f" % m2flux[np.logical_and(Rx_avg > 2.49, Rx_avg < 2.5)])
ax[0].set_title("Mass to flux ratio")
ax[0].legend()
# ax[1].plot(r_bins, b_bins, label = "T = %.2f" % data['t'][:])
ax[1].plot(Rx_avg, Flux)
ax[1].set_title("Flux")

ax[2].plot(Rx_avg, Mass)
ax[2].set_title("Mass")

#%% Calculate GPE

### Calculate the portential field [without Gravitional constant]
GP = CalGPE(den, cell_size = dx)

### Calculate total energy
GPE = 1/2 * G * np.sum(GP*den) * dV * Msun ** 2 / pc ** 6 * pc ** 2

print("Total gravitational potential energy is ", GPE)


#%% Magnetic energy

BE = (b_code_to_Tesla(b_muG_to_b_code(b_muG )) * units.tesla)** 2 / 2 / mu  *  Lx*Ly*Lz * pc ** 3

BE = units.convert_to(BE.n(), [unit(GPE)])
print("Magneic energy is", BE)


#%% Kinetic energy

Mass = np.sum(den) * dV * Msun / pc**3

print("Mass is", Mass)


KE = 1/2 * Mass * (10*sound_speed) ** 2

print("Mass is", KE)




#%% Virial

KEE = 2*KE

TE = KE + BE - GPE

print(r"Fraction of Kinetic Energy is   %6.2f%% " % (KE/TE*100))
print(r"Fraction of Magnetic Energy is  %6.2f%% " % (BE/TE*100))
print(r"Fraction of Potential Energy is %6.2f%% " % (np.abs(GPE/TE*100)))



#%%
TrialKE = -GPE/2
TrialV = sp.sqrt(TrialKE * 2 / Mass)

print(TrialV/sound_speed)

#%%

Volume = np.pi * 4**2 * 18 * pc ** 3 
Density = Mass/Volume

#%%

R  = np.sqrt(Y ** 2 + X ** 2)

new_den = np.ones(R.shape)


# %%

# %%
