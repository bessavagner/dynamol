import numpy as np
import scipy.constants as sc

vol_key = 'molar volume of ideal gas (273.15 K, 101.325 kPa)'
mass_key = 'atomic mass constant'
zero3D = np.array([0, 0, 0])
zero2D = np.array([0, 0])
atomic_mass_constant = sc.physical_constants[mass_key][0]
vol_mol_constant = sc.physical_constants[vol_key][0]

atomic_mass = {
    'argon': 39.948*atomic_mass_constant,
    'neon': 20.18*atomic_mass_constant,
    'xenon': 131.293*atomic_mass_constant,
    'O2': 31.998*atomic_mass_constant,
    'CO2': 43.991*atomic_mass_constant,
    'C6H6': 78.114*atomic_mass_constant
}  # kg
dispersion_energy = {
    'argon': 1.65e-21,
    'neon': 4.91511044e-22,
    'xenon': 3.05123429e-21,
    'O2': 1.6222625750000002e-21,
    'CO2': 2.60942661e-21,
    'C6H6': 2.1612e-21
}  # J
spacing = {
    'argon': 3.4e-10,
    'neon': 2.749e-10,
    'xenon': 4.10e-10,
    'O2': 3.58e-10,
    'CO2': 4.4862e-10,
    'C6H6': 4.782e-10
}  # m
