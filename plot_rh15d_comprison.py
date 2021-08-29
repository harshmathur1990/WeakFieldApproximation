import sys
import pandas as pd
sys.path.insert(2, '/home/harsh/CourseworkRepo/WFAComparison')
import os
# import rh
import numpy as np
import matplotlib.pyplot as plt
import h5py
from stray_light_approximation import *
from helita.sim import rh15d
from scipy.integrate import cumtrapz


def make_ray_file():
    out = rh15d.Rh15dout(fdir='output')

    wave = out.ray.wavelength.data

    indices= list()

    interesting_waves = [121.56, 102.57, 97.25, 656.28, 486.135, 500]

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

def make_csv_file(filename):

    data = {
        'Height (Km)': [
            2158 ,2157 ,2155 ,2154 ,2152 ,2151 ,2150 ,2149 ,
            2149 ,2148 ,2148 ,2147 ,2147 ,2147 ,2146 ,2146 ,
            2146 ,2146 ,2145 ,2145 ,2145 ,2145 ,2144 ,2144 ,
            2143 ,2143 ,2142 ,2141 ,2139 ,2132 ,2117 ,2085 ,
            2037 ,1982 ,1926 ,1877 ,1815 ,1734 ,1641 ,1556 ,
            1454 ,1349 ,1257 ,1160 ,1069 ,978 ,910 ,851 ,801 ,
            752 ,702 ,649 ,600 ,560 ,525 ,490 ,450 ,401 ,351 ,
            302 ,252 ,203 ,178 ,152 ,127 ,102 ,77 ,52 ,36 ,
            21 ,11 ,1 ,-9 ,-19 ,-29 ,-39 ,-50 ,-60 ,
            -70 ,-80 ,-90 ,-100
        ],
        'Temperature (K)': [
            100000.0, 95600.0, 90820.0, 83890.0, 75930.0,
            71340.0, 66150.0, 60170.0, 53280.0, 49390.0,
            45420.0, 41180.0, 36590.0, 32150.0, 27970.0,
            24060.0, 20420.0, 17930.0, 16280.0, 14520.0,
            13080.0, 12190.0, 11440.0, 10850.0, 10340.0,
            9983.0, 9735.0, 9587.0, 9458.0, 9358.0, 9228.0,
            8988.0, 8635.0, 8273.0, 7970.0, 7780.0, 7600.0,
            7410.0, 7220.0, 7080.0, 6910.0, 6740.0, 6570.0,
            6370.0, 6180.0, 5950.0, 5760.0, 5570.0, 5380.0,
            5160.0, 4900.0, 4680.0, 4560.0, 4520.0, 4500.0,
            4510.0, 4540.0, 4610.0, 4690.0, 4780.0, 4880.0,
            4990.0, 5060.0, 5150.0, 5270.0, 5410.0, 5580.0,
            5790.0, 5980.0, 6180.0, 6340.0, 6520.0, 6720.0,
            6980.0, 7280.0, 7590.0, 7900.0, 8220.0, 8540.0,
            8860.0, 9140.0, 9400.0
        ],
        'Electron Density (1/m^3)': [
            1E+16, 1E+16, 1E+16, 1E+16, 2E+16, 2E+16,
            2E+16, 2E+16, 2E+16, 2E+16, 3E+16, 3E+16,
            3E+16, 3E+16, 4E+16, 4E+16, 5E+16, 5E+16,
            6E+16, 6E+16, 7E+16, 7E+16, 7E+16, 7E+16,
            7E+16, 7E+16, 7E+16, 6E+16, 6E+16, 6E+16,
            6E+16, 7E+16, 7E+16, 8E+16, 8E+16, 8E+16,
            8E+16, 9E+16, 9E+16, 1E+17, 1E+17, 1E+17,
            1E+17, 1E+17, 1E+17, 1E+17, 1E+17, 1E+17,
            1E+17, 8E+16, 8E+16, 9E+16, 1E+17, 2E+17,
            2E+17, 3E+17, 5E+17, 7E+17, 1E+18, 2E+18,
            3E+18, 4E+18, 5E+18, 6E+18, 8E+18, 1E+19,
            1E+19, 2E+19, 3E+19, 4E+19, 5E+19, 8E+19,
            1E+20, 2E+20, 3E+20, 4E+20, 7E+20, 1E+21,
            2E+21, 2E+21, 3E+21, 4E+21
        ],
        'Total H Population (1/m^3)': [
            1E+16, 1E+16, 1E+16, 1E+16, 1E+16, 1E+16,
            2E+16, 2E+16, 2E+16, 2E+16, 2E+16, 2E+16,
            3E+16, 3E+16, 4E+16, 4E+16, 5E+16, 5E+16,
            6E+16, 7E+16, 7E+16, 8E+16, 8E+16, 9E+16,
            9E+16, 1E+17, 1E+17, 1E+17, 1E+17, 1E+17,
            1E+17, 1E+17, 2E+17, 2E+17, 2E+17, 3E+17,
            3E+17, 5E+17, 7E+17, 1E+18, 2E+18, 3E+18,
            5E+18, 1E+19, 2E+19, 4E+19, 6E+19, 1E+20,
            1E+20, 2E+20, 4E+20, 6E+20, 1E+21, 1E+21,
            2E+21, 3E+21, 4E+21, 6E+21, 1E+22, 2E+22,
            2E+22, 3E+22, 4E+22, 5E+22, 6E+22, 7E+22,
            8E+22, 9E+22, 1E+23, 1E+23, 1E+23, 1E+23,
            1E+23, 1E+23, 1E+23, 1E+23, 1E+23, 1E+23,
            1E+23, 1E+23, 1E+23, 1E+23
        ],
        'Velocity_Z (Km/sec)': list(np.zeros(82)),
        'Velocity_Turbulent (Km/sec)': [
            11, 11, 11, 10, 10, 10, 10, 10, 10, 10,
            10, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8,
            8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 6, 6,
            6, 6, 5, 5, 5, 4, 4, 3, 3, 2, 2, 2, 2, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2
        ]
    }

    interesting_waves = [500, 121.56, 102.57, 97.25, 656.28, 486.135]

    interesting_names = [
        'Log Tau at 5000 Angstrom',
        'Log Tau at Lyman Alpha Core',
        'Log Tau at Lyman Beta Core',
        'Log Tau at Lyman Gamma Core',
        'Log Tau at H alpha Core',
        'Log Tau at H Beta Core'
    ]

    os.chdir("/home/harsh/CourseworkRepo/rh/rh/rh15d/run")

    out = rh15d.Rh15dout(fdir='output')

    height = out.atmos.height_scale[0, 0].dropna('height')

    for w, name in zip(interesting_waves, interesting_names):
        index = np.argmin(np.abs(out.ray.wavelength_selected.data - w))
        tau = cumtrapz(out.ray.chi[0, 0, :, index].dropna('height'), x=-height)
        log_tau = np.log(tau)
        if log_tau.size < 82:
            log_tau = (["N/A"] * (82 - log_tau.size)) + list(log_tau)
        data[name] = log_tau

    os.chdir("/home/harsh/CourseworkRepo/WFAComparison")

    df = pd.DataFrame(data)

    df.to_csv(filename)


def make_plot(name):
    catalog = np.loadtxt('/home/harsh/CourseworkRepo/WFAComparison/catalog_6563.txt')

    os.chdir("/home/harsh/CourseworkRepo/rh/rh/rh15d/run")

    out = rh15d.Rh15dout(fdir='output')

    os.chdir("/home/harsh/Spinor_2008/Ca_x_30_y_18_2_20_250_280/Synthesis/")

    wave = np.array(out.ray.wavelength.data)

    wave *= 10

    intensity = out.ray.intensity.data[0, 0]

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

    os.chdir("/home/harsh/CourseworkRepo/WFAComparison")

    plt.savefig('{}.eps'.format(name), dpi=300, format='eps')

    plt.savefig('{}.png'.format(name), dpi=300, format='png')

    plt.show()
