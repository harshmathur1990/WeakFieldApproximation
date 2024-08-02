import netCDF4
import numpy as np
from helita.sim import rh15d
import matplotlib.pyplot as plt


def write_to_ray_input(filename, indices, index500):
    f = open(filename, 'w')  # this will overwrite any existing file!
    f.write('1.00\n')
    output = str(len(indices) + 1)
    for ind in indices:
        output += ' %i' % ind
    output += ' %i\n' % index500
    f.write(output)
    f.close()


def get_indices(wave, wave_start=655, wave_end=657):

    wave.sel(wavelength=500, method='nearest')

    index500 = np.argmin(np.abs(wave.data - 500))

    indices = np.arange(
        len(wave)
    )[(wave > wave_start) & (wave < wave_end)]

    return indices, index500


def return_wavelength_intensity(fdir='.'):

    out = rh15d.Rh15dout(fdir=fdir)

    wave = out.ray.wavelength

    intensity = out.ray.intensity[0][0]

    plt.plot(wave, intensity)

    plt.show()

    return wave, intensity


def combine_indices(*args, **kwargs):

    total_indices = list()

    for arg in args:
        total_indices += list(arg)

    return np.array(total_indices)


if __name__ == '__main__':

    wave = None

    indices_849, _ = get_indices(wave, 849.347, 850.279)

    indices_854, _ = get_indices(wave, 853.327, 855.094)

    indices_866, _ = get_indices(wave, 866.445, 866.961)

    indices_h, _ = get_indices(wave, 652.388, 658.455)

    indices_290, _ = get_indices(wave, 380, 399)

    indices_mg, index500 = get_indices(wave, 292.617, 294.041)

    indices_ca = combine_indices(
        indices_290,
        indices_849,
        indices_854,
        indices_866
    )

    write_to_ray_input('ray_ca.input', indices_ca, index500)

    write_to_ray_input('ray_mg.input', indices_mg, index500)

    write_to_ray_input('ray_h.input', indices_h, index500)
