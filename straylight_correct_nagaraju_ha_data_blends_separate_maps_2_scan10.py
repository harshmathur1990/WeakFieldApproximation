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

write_path = base_path / 'maps_2_scan10/stic/processed_inputs'

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

wave_ca = np.array(
    [
        8531.96148, 8531.99521, 8532.02894, 8532.06267, 8532.0964 ,
        8532.13013, 8532.16386, 8532.19759, 8532.23132, 8532.26505,
        8532.29878, 8532.33251, 8532.36624, 8532.39997, 8532.4337 ,
        8532.46743, 8532.50116, 8532.53489, 8532.56862, 8532.60235,
        8532.63608, 8532.66981, 8532.70354, 8532.73727, 8532.771  ,
        8532.80473, 8532.83846, 8532.87219, 8532.90592, 8532.93965,
        8532.97338, 8533.00711, 8533.04084, 8533.07457, 8533.1083 ,
        8533.14203, 8533.17576, 8533.20949, 8533.24322, 8533.27695,
        8533.31068, 8533.34441, 8533.37814, 8533.41187, 8533.4456 ,
        8533.47933, 8533.51306, 8533.54679, 8533.58052, 8533.61425,
        8533.64798, 8533.68171, 8533.71544, 8533.74917, 8533.7829 ,
        8533.81663, 8533.85036, 8533.88409, 8533.91782, 8533.95155,
        8533.98528, 8534.01901, 8534.05274, 8534.08647, 8534.1202 ,
        8534.15393, 8534.18766, 8534.22139, 8534.25512, 8534.28885,
        8534.32258, 8534.35631, 8534.39004, 8534.42377, 8534.4575 ,
        8534.49123, 8534.52496, 8534.55869, 8534.59242, 8534.62615,
        8534.65988, 8534.69361, 8534.72734, 8534.76107, 8534.7948 ,
        8534.82853, 8534.86226, 8534.89599, 8534.92972, 8534.96345,
        8534.99718, 8535.03091, 8535.06464, 8535.09837, 8535.1321 ,
        8535.16583, 8535.19956, 8535.23329, 8535.26702, 8535.30075,
        8535.33448, 8535.36821, 8535.40194, 8535.43567, 8535.4694 ,
        8535.50313, 8535.53686, 8535.57059, 8535.60432, 8535.63805,
        8535.67178, 8535.70551, 8535.73924, 8535.77297, 8535.8067 ,
        8535.84043, 8535.87416, 8535.90789, 8535.94162, 8535.97535,
        8536.00908, 8536.04281, 8536.07654, 8536.11027, 8536.144  ,
        8536.17773, 8536.21146, 8536.24519, 8536.27892, 8536.31265,
        8536.34638, 8536.38011, 8536.41384, 8536.44757, 8536.4813 ,
        8536.51503, 8536.54876, 8536.58249, 8536.61622, 8536.64995,
        8536.68368, 8536.71741, 8536.75114, 8536.78487, 8536.8186 ,
        8536.85233, 8536.88606, 8536.91979, 8536.95352, 8536.98725,
        8537.02098, 8537.05471, 8537.08844, 8537.12217, 8537.1559 ,
        8537.18963, 8537.22336, 8537.25709, 8537.29082, 8537.32455,
        8537.35828, 8537.39201, 8537.42574, 8537.45947, 8537.4932 ,
        8537.52693, 8537.56066, 8537.59439, 8537.62812, 8537.66185,
        8537.69558, 8537.72931, 8537.76304, 8537.79677, 8537.8305 ,
        8537.86423, 8537.89796, 8537.93169, 8537.96542, 8537.99915,
        8538.03288, 8538.06661, 8538.10034, 8538.13407, 8538.1678 ,
        8538.20153, 8538.23526, 8538.26899, 8538.30272, 8538.33645,
        8538.37018, 8538.40391, 8538.43764, 8538.47137, 8538.5051 ,
        8538.53883, 8538.57256, 8538.60629, 8538.64002, 8538.67375,
        8538.70748, 8538.74121, 8538.77494, 8538.80867, 8538.8424 ,
        8538.87613, 8538.90986, 8538.94359, 8538.97732, 8539.01105,
        8539.04478, 8539.07851, 8539.11224, 8539.14597, 8539.1797 ,
        8539.21343, 8539.24716, 8539.28089, 8539.31462, 8539.34835,
        8539.38208, 8539.41581, 8539.44954, 8539.48327, 8539.517  ,
        8539.55073, 8539.58446, 8539.61819, 8539.65192, 8539.68565,
        8539.71938, 8539.75311, 8539.78684, 8539.82057, 8539.8543 ,
        8539.88803, 8539.92176, 8539.95549, 8539.98922, 8540.02295,
        8540.05668, 8540.09041, 8540.12414, 8540.15787, 8540.1916 ,
        8540.22533, 8540.25906, 8540.29279, 8540.32652, 8540.36025,
        8540.39398, 8540.42771, 8540.46144, 8540.49517, 8540.5289 ,
        8540.56263, 8540.59636, 8540.63009, 8540.66382, 8540.69755,
        8540.73128, 8540.76501, 8540.79874, 8540.83247, 8540.8662 ,
        8540.89993, 8540.93366, 8540.96739, 8541.00112, 8541.03485,
        8541.06858, 8541.10231, 8541.13604, 8541.16977, 8541.2035 ,
        8541.23723, 8541.27096, 8541.30469, 8541.33842, 8541.37215,
        8541.40588, 8541.43961, 8541.47334, 8541.50707, 8541.5408 ,
        8541.57453, 8541.60826, 8541.64199, 8541.67572, 8541.70945,
        8541.74318, 8541.77691, 8541.81064, 8541.84437, 8541.8781 ,
        8541.91183, 8541.94556, 8541.97929, 8542.01302, 8542.04675,
        8542.08048, 8542.11421, 8542.14794, 8542.18167, 8542.2154 ,
        8542.24913, 8542.28286, 8542.31659, 8542.35032, 8542.38405,
        8542.41778, 8542.45151, 8542.48524, 8542.51897, 8542.5527 ,
        8542.58643, 8542.62016, 8542.65389, 8542.68762, 8542.72135,
        8542.75508, 8542.78881, 8542.82254, 8542.85627, 8542.89   ,
        8542.92373, 8542.95746, 8542.99119, 8543.02492, 8543.05865,
        8543.09238, 8543.12611, 8543.15984, 8543.19357, 8543.2273 ,
        8543.26103, 8543.29476, 8543.32849, 8543.36222, 8543.39595,
        8543.42968, 8543.46341, 8543.49714, 8543.53087, 8543.5646 ,
        8543.59833, 8543.63206, 8543.66579, 8543.69952, 8543.73325,
        8543.76698, 8543.80071, 8543.83444, 8543.86817, 8543.9019 ,
        8543.93563, 8543.96936, 8544.00309, 8544.03682, 8544.07055,
        8544.10428, 8544.13801, 8544.17174, 8544.20547, 8544.2392 ,
        8544.27293, 8544.30666, 8544.34039, 8544.37412, 8544.40785,
        8544.44158, 8544.47531, 8544.50904, 8544.54277, 8544.5765 ,
        8544.61023, 8544.64396, 8544.67769, 8544.71142, 8544.74515,
        8544.77888, 8544.81261, 8544.84634, 8544.88007, 8544.9138 ,
        8544.94753, 8544.98126, 8545.01499, 8545.04872, 8545.08245,
        8545.11618, 8545.14991, 8545.18364, 8545.21737, 8545.2511 ,
        8545.28483, 8545.31856, 8545.35229, 8545.38602, 8545.41975,
        8545.45348, 8545.48721, 8545.52094, 8545.55467, 8545.5884 ,
        8545.62213, 8545.65586, 8545.68959, 8545.72332, 8545.75705,
        8545.79078, 8545.82451, 8545.85824, 8545.89197, 8545.9257 ,
        8545.95943, 8545.99316, 8546.02689, 8546.06062, 8546.09435,
        8546.12808, 8546.16181, 8546.19554, 8546.22927, 8546.263  ,
        8546.29673, 8546.33046, 8546.36419, 8546.39792, 8546.43165,
        8546.46538, 8546.49911, 8546.53284, 8546.56657, 8546.6003 ,
        8546.63403, 8546.66776, 8546.70149, 8546.73522, 8546.76895,
        8546.80268, 8546.83641, 8546.87014, 8546.90387, 8546.9376 ,
        8546.97133, 8547.00506, 8547.03879, 8547.07252, 8547.10625,
        8547.13998, 8547.17371, 8547.20744, 8547.24117, 8547.2749 ,
        8547.30863, 8547.34236, 8547.37609, 8547.40982, 8547.44355,
        8547.47728, 8547.51101, 8547.54474, 8547.57847
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

    plt.plot(wave_ha, norm_median_stray, label='Stray Corrected Median', linewidth=0.5)

    plt.plot(wave_ha, scipy.ndimage.gaussian_filter1d(norm_atlas, sigma=fwhm[np.unravel_index(np.argmin(result), result.shape)[0]]/2.355), label='Atlas', linewidth=0.5)

    plt.gcf().set_size_inches(19.2, 10.8, forward=True)

    plt.legend()

    plt.savefig(write_path / 'Ha_median_comparison.pdf', format='pdf', dpi=300)

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

    fov_data = stray_corrected_data[2:20, :, 228:288, :]

    wc8, ic8 = findgrid(wave_ha, (wave_ha[10] - wave_ha[9])*0.25, extra=8)

    ha = sp.profile(nx=60, ny=18, ns=4, nw=wc8.size)

    ha.wav[:] = wc8[:]

    ha.dat[0,:,:,ic8,:] = np.transpose(
        fov_data,
        axes=(3, 0, 2, 1)
    ) / stic_cgs_calib_factor

    ha.weights[:, :] = 1e16

    ha.weights[ic8, 0] = 0.004

    ha.weights[ic8[18:46], 0] = 0.002

    ha.weights[ic8[69:186], 0] = 0.002

    ha.weights[ic8[405:432], 0] = 0.002

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


def combine_ha_ca_data():
    base_path = Path('/home/harsh/SpinorNagaraju/maps_2_scan10/stic/processed_inputs/')

    ca_file = base_path / 'alignedspectra_scan2_map10_Ca.fits_stic_profiles.nc'

    ha_file = base_path / 'alignedspectra_scan2_map10_Ha.fits_stic_profiles.nc'

    ca_output_file_path = Path('/home/harsh/SpinorNagaraju/maps_2_scan10/stic/fulldata_inversions/alignedspectra_scan1_map01_Ca.fits_stic_profiles_cycle_1_t_6_vl_3_vt_4_blos_3_atmos.nc')

    fha = h5py.File(ha_file, 'r')

    fca = h5py.File(ca_file, 'r')

    ha = sp.profile(nx=60, ny=18, ns=4, nw=fha['wav'][()].size)

    ha.wav[:] = fha['wav'][()]

    ha.dat[0, :, :, :, :] = fha['profiles'][0]

    ind = np.where(ha.dat[0, 0, 0, :, 0] != 0)[0]

    ha.weights[:, :] = 1e16

    ha.weights[ind, 0] = 0.004

    ha.weights[ind[18:46], 0] = 0.002

    ha.weights[ind[69:186], 0] = 0.002

    ha.weights[ind[405:432], 0] = 0.002

    ca = sp.profile(nx=60, ny=18, ns=4, nw=fca['wav'][()].size)

    ca.wav[:] = fca['wav'][()]

    ca.dat[0, :, :, :, :] = fca['profiles'][0, 0:18]

    ind = np.where(ca.dat[0, 0, 0, :, 0] != 0)[0]

    ca.weights[:, :] = 1e16

    ca.weights[ind, 0] = 0.004

    ca.weights[ind[19:36], 0] = 0.002

    ca.weights[ind[76:85], 0] = 0.002

    ca.weights[ind[188:211], 0] = 0.002

    all_profiles = ca + ha

    all_profiles.write(
        write_path / 'aligned_Ca_Ha_stic_profiles.nc'
    )

    fca.close()

    fha.close()

    f = h5py.File(ca_output_file_path, 'r')

    m = sp.model(nx=60, ny=17, nt=1, ndep=150)

    m.ltau[:, :, :] = f['ltau500'][0, 0, 0]

    m.pgas[:, :, :] = 1

    m.temp[:, :, :] = f['temp'][0, 0:17]

    m.vlos[:, :, :] = f['vlos'][0, 0:17]

    m.vturb[:, :, :] = f['vturb'][0, 0:17]

    m.Bln[:, :, :] = f['blong'][0, 0:17]

    m.write(write_path / 'ha_ca_input_atmos_60_17.nc')

    f.close()


if __name__ == '__main__':
    # get_catalog_0p8()
    generate_stic_input_files('/home/harsh/SpinorNagaraju/alignedspectra_scan2_map10_Ha.fits')

    combine_ha_ca_data()