import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/rh/rhv2src/python')
sys.path.insert(2, '/home/harsh/CourseworkRepo/WFAComparison')
import os
import rh
import numpy as np
import matplotlib.pyplot as plt
import h5py
from stray_light_approximation import *


def make_ray_file():
    out = rh.readOutFiles()

    wave = np.array(out.spect.lambda0)

    indices = list()

    interesting_waves = [121.5668237310, 121.5673644608, 656.275181, 656.290944, 102.572182505, 102.572296565, 656.272483, 656.277153, 	656.270970, 656.285177, 656.286734]

    for w in interesting_waves:
        indices.append(
            np.argmin(np.abs(wave-w))
        )

    f = open('ray.input', 'w')

    f.write('1.00\n')
    f.write(
        '{} {}'.format(
            len(indices),
            ' '.join([str(indice) for indice in indices])
        )
    )
    f.close()


def make_plot(name):
    catalog = np.loadtxt('/home/harsh/CourseworkRepo/WFAComparison/catalog_6563.txt')

    os.chdir("/home/harsh/CourseworkRepo/rh/rhv2src/rhf1d/run")

    out = rh.readOutFiles()

    os.chdir("/home/harsh/Spinor_2008/Ca_x_30_y_18_2_20_250_280/Synthesis")

    wave = np.array(out.spect.lambda0)

    wave *= 10

    intensity = np.array(out.ray.I)

    f = h5py.File('falc_ha_Ca_H_15_He_synthesis.nc', 'r')

    get_indice = prepare_get_indice(wave)

    vec_get_indice = np.vectorize(get_indice)

    atlas_indice = vec_get_indice(f['wav'][1860:])

    norm_line, norm_atlas, atlas_wave = normalise_profiles(
        intensity[atlas_indice], wave[atlas_indice],
        catalog[:, 1], catalog[:, 0],
        wave[atlas_indice][-1]
    )

    f.close()

    plt.close('all')

    plt.clf()

    plt.cla()

    plt.plot(wave[atlas_indice], norm_line, label='synthesis')

    plt.plot(atlas_wave, norm_atlas, label='BASS2000')

    plt.xlabel('Wavelength (Angstrom)')

    plt.ylabel('Normalised Intensity')

    plt.legend()

    fig = plt.gcf()

    fig.set_size_inches(19.2, 10.8, forward=True)

    plt.savefig('{}.eps'.format(name), dpi=300, format='eps')

    plt.savefig('{}.png'.format(name), dpi=300, format='png')

    plt.show()
