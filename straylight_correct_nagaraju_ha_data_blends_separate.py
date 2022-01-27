import sys

import scipy.ndimage

sys.path.insert(1, '/home/harsh/CourseworkRepo/stic/example')
sys.path.insert(2, '/home/harsh/CourseworkRepo/WFAComparison')
import h5py
import numpy as np
import sunpy.io.fits
from pathlib import Path
from prepare_data import *
from stray_light_approximation import *
import matplotlib.pyplot as plt


base_path = Path(
    '/home/harsh/SpinorNagaraju/'
)

write_path = base_path / 'maps_1/stic/processed_inputs'

falc_file_path = Path(
    '/home/harsh/CourseworkRepo/stic/run/falc_nicole_for_stic.nc'
)

synthesis_file_0p8 = '/home/harsh/SpinorNagaraju/halpha_flat_master_calc/Ha_mu_0.8_1d_syn_with_blends.nc'

synthesis_file_1 = '/home/harsh/SpinorNagaraju/halpha_flat_master_calc/Ha_mu_1.0_1d_syn_with_blends.nc'

catalog_ha_bass_2000 = '/home/harsh/SpinorNagaraju/catalog_ha_bass_2000.nc'

index = '[32:482][::-1][:-6]'

wave_ha = np.array(
    [
        6559.799  , 6559.82145, 6559.8439 , 6559.86635, 6559.8888 ,
        6559.91125, 6559.9337 , 6559.95615, 6559.9786 , 6560.00105,
        6560.0235 , 6560.04595, 6560.0684 , 6560.09085, 6560.1133 ,
        6560.13575, 6560.1582 , 6560.18065, 6560.2031 , 6560.22555,
        6560.248  , 6560.27045, 6560.2929 , 6560.31535, 6560.3378 ,
        6560.36025, 6560.3827 , 6560.40515, 6560.4276 , 6560.45005,
        6560.4725 , 6560.49495, 6560.5174 , 6560.53985, 6560.5623 ,
        6560.58475, 6560.6072 , 6560.62965, 6560.6521 , 6560.67455,
        6560.697  , 6560.71945, 6560.7419 , 6560.76435, 6560.7868 ,
        6560.80925, 6560.8317 , 6560.85415, 6560.8766 , 6560.89905,
        6560.9215 , 6560.94395, 6560.9664 , 6560.98885, 6561.0113 ,
        6561.03375, 6561.0562 , 6561.07865, 6561.1011 , 6561.12355,
        6561.146  , 6561.16845, 6561.1909 , 6561.21335, 6561.2358 ,
        6561.25825, 6561.2807 , 6561.30315, 6561.3256 , 6561.34805,
        6561.3705 , 6561.39295, 6561.4154 , 6561.43785, 6561.4603 ,
        6561.48275, 6561.5052 , 6561.52765, 6561.5501 , 6561.57255,
        6561.595  , 6561.61745, 6561.6399 , 6561.66235, 6561.6848 ,
        6561.70725, 6561.7297 , 6561.75215, 6561.7746 , 6561.79705,
        6561.8195 , 6561.84195, 6561.8644 , 6561.88685, 6561.9093 ,
        6561.93175, 6561.9542 , 6561.97665, 6561.9991 , 6562.02155,
        6562.044  , 6562.06645, 6562.0889 , 6562.11135, 6562.1338 ,
        6562.15625, 6562.1787 , 6562.20115, 6562.2236 , 6562.24605,
        6562.2685 , 6562.29095, 6562.3134 , 6562.33585, 6562.3583 ,
        6562.38075, 6562.4032 , 6562.42565, 6562.4481 , 6562.47055,
        6562.493  , 6562.51545, 6562.5379 , 6562.56035, 6562.5828 ,
        6562.60525, 6562.6277 , 6562.65015, 6562.6726 , 6562.69505,
        6562.7175 , 6562.73995, 6562.7624 , 6562.78485, 6562.8073 ,
        6562.82975, 6562.8522 , 6562.87465, 6562.8971 , 6562.91955,
        6562.942  , 6562.96445, 6562.9869 , 6563.00935, 6563.0318 ,
        6563.05425, 6563.0767 , 6563.09915, 6563.1216 , 6563.14405,
        6563.1665 , 6563.18895, 6563.2114 , 6563.23385, 6563.2563 ,
        6563.27875, 6563.3012 , 6563.32365, 6563.3461 , 6563.36855,
        6563.391  , 6563.41345, 6563.4359 , 6563.45835, 6563.4808 ,
        6563.50325, 6563.5257 , 6563.54815, 6563.5706 , 6563.59305,
        6563.6155 , 6563.63795, 6563.6604 , 6563.68285, 6563.7053 ,
        6563.72775, 6563.7502 , 6563.77265, 6563.7951 , 6563.81755,
        6563.84   , 6563.86245, 6563.8849 , 6563.90735, 6563.9298 ,
        6563.95225, 6563.9747 , 6563.99715, 6564.0196 , 6564.04205,
        6564.0645 , 6564.08695, 6564.1094 , 6564.13185, 6564.1543 ,
        6564.17675, 6564.1992 , 6564.22165, 6564.2441 , 6564.26655,
        6564.289  , 6564.31145, 6564.3339 , 6564.35635, 6564.3788 ,
        6564.40125, 6564.4237 , 6564.44615, 6564.4686 , 6564.49105,
        6564.5135 , 6564.53595, 6564.5584 , 6564.58085, 6564.6033 ,
        6564.62575, 6564.6482 , 6564.67065, 6564.6931 , 6564.71555,
        6564.738  , 6564.76045, 6564.7829 , 6564.80535, 6564.8278 ,
        6564.85025, 6564.8727 , 6564.89515, 6564.9176 , 6564.94005,
        6564.9625 , 6564.98495, 6565.0074 , 6565.02985, 6565.0523 ,
        6565.07475, 6565.0972 , 6565.11965, 6565.1421 , 6565.16455,
        6565.187  , 6565.20945, 6565.2319 , 6565.25435, 6565.2768 ,
        6565.29925, 6565.3217 , 6565.34415, 6565.3666 , 6565.38905,
        6565.4115 , 6565.43395, 6565.4564 , 6565.47885, 6565.5013 ,
        6565.52375, 6565.5462 , 6565.56865, 6565.5911 , 6565.61355,
        6565.636  , 6565.65845, 6565.6809 , 6565.70335, 6565.7258 ,
        6565.74825, 6565.7707 , 6565.79315, 6565.8156 , 6565.83805,
        6565.8605 , 6565.88295, 6565.9054 , 6565.92785, 6565.9503 ,
        6565.97275, 6565.9952 , 6566.01765, 6566.0401 , 6566.06255,
        6566.085  , 6566.10745, 6566.1299 , 6566.15235, 6566.1748 ,
        6566.19725, 6566.2197 , 6566.24215, 6566.2646 , 6566.28705,
        6566.3095 , 6566.33195, 6566.3544 , 6566.37685, 6566.3993 ,
        6566.42175, 6566.4442 , 6566.46665, 6566.4891 , 6566.51155,
        6566.534  , 6566.55645, 6566.5789 , 6566.60135, 6566.6238 ,
        6566.64625, 6566.6687 , 6566.69115, 6566.7136 , 6566.73605,
        6566.7585 , 6566.78095, 6566.8034 , 6566.82585, 6566.8483 ,
        6566.87075, 6566.8932 , 6566.91565, 6566.9381 , 6566.96055,
        6566.983  , 6567.00545, 6567.0279 , 6567.05035, 6567.0728 ,
        6567.09525, 6567.1177 , 6567.14015, 6567.1626 , 6567.18505,
        6567.2075 , 6567.22995, 6567.2524 , 6567.27485, 6567.2973 ,
        6567.31975, 6567.3422 , 6567.36465, 6567.3871 , 6567.40955,
        6567.432  , 6567.45445, 6567.4769 , 6567.49935, 6567.5218 ,
        6567.54425, 6567.5667 , 6567.58915, 6567.6116 , 6567.63405,
        6567.6565 , 6567.67895, 6567.7014 , 6567.72385, 6567.7463 ,
        6567.76875, 6567.7912 , 6567.81365, 6567.8361 , 6567.85855,
        6567.881  , 6567.90345, 6567.9259 , 6567.94835, 6567.9708 ,
        6567.99325, 6568.0157 , 6568.03815, 6568.0606 , 6568.08305,
        6568.1055 , 6568.12795, 6568.1504 , 6568.17285, 6568.1953 ,
        6568.21775, 6568.2402 , 6568.26265, 6568.2851 , 6568.30755,
        6568.33   , 6568.35245, 6568.3749 , 6568.39735, 6568.4198 ,
        6568.44225, 6568.4647 , 6568.48715, 6568.5096 , 6568.53205,
        6568.5545 , 6568.57695, 6568.5994 , 6568.62185, 6568.6443 ,
        6568.66675, 6568.6892 , 6568.71165, 6568.7341 , 6568.75655,
        6568.779  , 6568.80145, 6568.8239 , 6568.84635, 6568.8688 ,
        6568.89125, 6568.9137 , 6568.93615, 6568.9586 , 6568.98105,
        6569.0035 , 6569.02595, 6569.0484 , 6569.07085, 6569.0933 ,
        6569.11575, 6569.1382 , 6569.16065, 6569.1831 , 6569.20555,
        6569.228  , 6569.25045, 6569.2729 , 6569.29535, 6569.3178 ,
        6569.34025, 6569.3627 , 6569.38515, 6569.4076 , 6569.43005,
        6569.4525 , 6569.47495, 6569.4974 , 6569.51985, 6569.5423 ,
        6569.56475, 6569.5872 , 6569.60965, 6569.6321 , 6569.65455,
        6569.677  , 6569.69945, 6569.7219 , 6569.74435
    ]
)

interesting_fov = '[2:, :, 228:288]'

cw = np.asarray([6562.])
cont = []
for ii in cw:
    cont.append(getCont(ii))


def get_catalog_0p8():
    f_s_1 = h5py.File(synthesis_file_1, 'r')

    f_s_0p8 = h5py.File(synthesis_file_0p8, 'r')

    f_c_b = h5py.File(catalog_ha_bass_2000, 'r')

    indd = list()

    for a_wave in wave_ha:
        indd.append(np.argmin(np.abs(f_c_b['wav'][()] - a_wave)))

    indd = np.array(indd)

    catalog_0p8 = f_c_b['profiles'][0, 0, 0, indd, 0] * f_s_0p8['profiles'][0, 0, 0, :, 0] / f_s_1['profiles'][0, 0, 0, :, 0]

    plt.plot(catalog_0p8, label='mu=0.8')

    plt.plot(f_c_b['profiles'][0, 0, 0, indd, 0], label='mu=1')

    plt.legend()

    plt.show()

    f_s_1.close()

    f_s_0p8.close()

    f_c_b.close()

    return catalog_0p8


def get_raw_data(filename):

    raw_data = np.zeros((20, 4, 512, 512), dtype=np.float64)

    for i in range(20):
        data, header = sunpy.io.fits.read(base_path / filename)[i + 1]
        raw_data[i] = data

    raw_data[:, 1:] -= 32768

    return raw_data[:, :, :, 32:482][:, :, :, ::-1][:, :, :, :-6]


def correct_for_straylight(data):
    crop_indice_x = np.arange(4, 17)

    crop_indice_y = np.array(
        list(
            np.arange(203, 250)
        ) +
        list(
            np.arange(280, 370)
        )
    )

    median_profile = np.median(
        data[crop_indice_x, 0, :][:, crop_indice_y], (0, 1)
    )

    catalog_0p8 = get_catalog_0p8()

    norm_line, norm_atlas, atlas_wave = normalise_profiles(
        median_profile,
        wave_ha,
        catalog_0p8,
        wave_ha,
        cont_wave=wave_ha[-1]
    )

    a, b = np.polyfit([0, norm_atlas.size - 1], [norm_atlas[0], norm_atlas[-1]], 1)

    atlas_slope = a * np.arange(norm_atlas.size) + b

    atlas_slope /= atlas_slope.max()

    a, b = np.polyfit([0, norm_line.size - 1], [norm_line[0], norm_line[-1]], 1)

    line_slope = a * np.arange(norm_line.size) + b

    line_slope /= line_slope.max()

    multiplicative_factor = atlas_slope / line_slope

    multiplicative_factor /= multiplicative_factor.max()

    norm_line *= multiplicative_factor

    result, result_atlas, fwhm, sigma, k_values = approximate_stray_light_and_sigma(
        norm_line,
        norm_atlas,
        continuum=1.0,
        indices=None
    )

    f = h5py.File(write_path / 'straylight_ha_using_median_profiles_with_atlas_at_0p8_estimated_profile.h5', 'w')

    f['wave_ha'] = wave_ha

    f['correction_factor'] = multiplicative_factor

    f['atlas_at_0p8'] = catalog_0p8

    f['norm_atlas'] = norm_atlas

    f['median_indice'] = '[4:17, 0, 203:250, 280:270]'

    f['median_profile'] = median_profile

    f['norm_median'] = norm_line

    f['mean_square_error'] = result

    f['result_atlas'] = result_atlas

    f['fwhm'] = fwhm

    f['sigma'] = sigma

    f['k_values'] = k_values

    f['fwhm_in_pixels'] = fwhm[np.unravel_index(np.argmin(result), result.shape)[0]]

    f['sigma_in_pixels'] = sigma[np.unravel_index(np.argmin(result), result.shape)[0]]

    f['straylight_value'] = np.unravel_index(np.argmin(result), result.shape)[1] / 100

    f['broadening_in_km_sec'] = fwhm[np.unravel_index(np.argmin(result), result.shape)[0]] * (
                wave_ha[1] - wave_ha[0]) * 2.99792458e5 / 6562.8

    f.close()

    stray_corrected_data = data.copy()

    stray_corrected_data[:, 0] = stray_corrected_data[:, 0] * multiplicative_factor

    stray_corrected_data[:, 0] = (stray_corrected_data[:, 0] - (
                (np.unravel_index(np.argmin(result), result.shape)[1] / 100) * stray_corrected_data[:, 0, :, 0][:, :,
                                                                               np.newaxis])) / (
                                             1 - (np.unravel_index(np.argmin(result), result.shape)[1] / 100))

    stray_corrected_median = np.median(
        stray_corrected_data[crop_indice_x, 0, :][:, crop_indice_y],
        (0, 1)
    )

    f1 = h5py.File(synthesis_file_0p8, 'r')

    norm_median_stray, norm_atlas, atlas_wave = normalise_profiles(
        stray_corrected_median,
        wave_ha,
        catalog_0p8,
        wave_ha,
        cont_wave=wave_ha[-1]
    )

    stic_cgs_calib_factor = stray_corrected_median[-1] / f1['profiles'][0, 0, 0, -1, 0]

    plt.plot(wave_ha, norm_median_stray, label='Stray Corrected Median')

    plt.plot(wave_ha, scipy.ndimage.gaussian_filter1d(norm_atlas, sigma=fwhm[np.unravel_index(np.argmin(result), result.shape)[0]]/2.355), label='Atlas')

    plt.legend()

    plt.show()

    f1.close()

    return stray_corrected_data, stray_corrected_median, stic_cgs_calib_factor, sigma[np.unravel_index(np.argmin(result), result.shape)[0]]


def generate_stic_input_files(filename):

    filename = Path(filename)

    data = get_raw_data(filename)

    stray_corrected_data, stray_corrected_median, stic_cgs_calib_factor, sigma = correct_for_straylight(data)

    f = h5py.File(
        write_path / '{}_stray_corrected.h5'.format(
            filename.name
        ),
        'w'
    )

    f['stray_corrected_data'] = stray_corrected_data

    f['stray_corrected_median'] = stray_corrected_median

    f['stic_cgs_calib_factor'] = stic_cgs_calib_factor

    f['wave_ha'] = wave_ha

    f.close()

    fov_data = stray_corrected_data[2:, :, 228:288, :]

    wc8, ic8 = findgrid(wave_ha, (wave_ha[10] - wave_ha[9])*0.25, extra=8)

    ha = sp.profile(nx=60, ny=18, ns=4, nw=wc8.size)

    ha.wav[:] = wc8[:]

    ha.dat[0,:,:,ic8,:] = np.transpose(
        fov_data,
        axes=(3, 0, 2, 1)
    ) / stic_cgs_calib_factor

    ha.write(
        write_path / '{}_stic_profiles.nc'.format(
            filename.name
        )
    )

    if wc8.size % 2 == 0:
        kernel_size = wc8.size - 1
    else:
        kernel_size = wc8.size - 2
    rev_kernel = np.zeros(kernel_size)
    rev_kernel[kernel_size // 2] = 1
    kernel = scipy.ndimage.gaussian_filter1d(rev_kernel, sigma=sigma * 4)

    broadening_filename = 'gaussian_broadening_{}_pixel_{}.h5'.format(np.round(sigma * 2.355 * 4, 1), 'Ha')
    f = h5py.File(write_path / broadening_filename, 'w')
    f['iprof'] = kernel
    f['wav'] = np.zeros_like(kernel)
    f.close()

    lab = "region = {0:10.5f}, {1:8.5f}, {2:3d}, {3:e}, {4}"
    print(" ")
    print("Regions information for the input file:" )
    print(lab.format(ha.wav[0], ha.wav[1]-ha.wav[0], ha.wav.size, cont[0],  'none, none'))
    print("(w0, dw, nw, normalization, degradation_type, instrumental_profile file)")
    print(" ")


def generate_input_atmos_file():

    f = h5py.File(falc_file_path, 'r')

    m = sp.model(nx=60, ny=19, nt=1, ndep=150)

    m.ltau[:, :, :] = f['ltau500'][0, 0, 0]

    m.pgas[:, :, :] = 1

    m.temp[:, :, :] = f['temp'][0, 0, 0]

    m.vlos[:, :, :] = f['vlos'][0, 0, 0]

    m.vturb[:, :, :] = f['vturb'][0, 0, 0]

    m.write('falc_60_19.nc')


if __name__ == '__main__':
    # get_catalog_0p8()
    generate_stic_input_files('/home/harsh/SpinorNagaraju//alignedspectra_scan1_map01_Ha.fits')
