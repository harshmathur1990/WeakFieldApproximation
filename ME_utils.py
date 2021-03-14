import numpy as np


h = 6.62606957e-34
c = 2.99792458e8
kb = 1.380649e-23


def prepare_temperature_from_intensity(wavelength_in_angstrom):

    f = c / wavelength_in_angstrom

    def get_temperature_from_intensity(intensity_in_si):

        I = intensity_in_si
        return (h * f) / (np.log((2 * h * f**3 / (c**2 * I)) + 1) * kb)

    return get_temperature_from_intensity
