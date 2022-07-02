import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/stic/example')
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.gridspec as gridspec
from prepare_data import *
from scipy.interpolate import CubicSpline


kmeans_output_dir = Path(
    '/home/harsh/SpinorNagaraju/maps_1/stic/kmeans_output'
)

atmos_rp_write_path = Path(
    '/home/harsh/SpinorNagaraju/maps_1/stic/'
)

input_file = Path(
    '/home/harsh/SpinorNagaraju/maps_1/stic/processed_inputs/alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc'
)


input_halpha_file = Path(
    '/home/harsh/SpinorNagaraju/maps_1/stic/processed_inputs/alignedspectra_scan1_map01_Ha.fits_stic_profiles.nc'
)


kmeans_file = Path(
    '/home/harsh/SpinorNagaraju/maps_1/stic/chosen_out_30.h5'
)


rps_plot_write_dir = Path(
    '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/'
)

falc_file_path = Path(
    '/home/harsh/CourseworkRepo/stic/run/falc_nicole_for_stic.nc'
)



cw = np.asarray([8542.])
cont = []
for ii in cw:
    cont.append(getCont(ii))

wave_ha_orig = np.array(
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
wave_8542_orig = np.array(
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


ltau = np.array(
    [
        -8.       , -7.78133  , -7.77448  , -7.76712  , -7.76004  ,
        -7.75249  , -7.74429  , -7.7356   , -7.72638  , -7.71591  ,
        -7.70478  , -7.69357  , -7.68765  , -7.68175  , -7.67589  ,
        -7.66997  , -7.66374  , -7.65712  , -7.64966  , -7.64093  ,
        -7.63093  , -7.6192   , -7.6053   , -7.58877  , -7.56925  ,
        -7.54674  , -7.52177  , -7.49317  , -7.4585   , -7.41659  ,
        -7.36725  , -7.31089  , -7.24834  , -7.18072  , -7.1113   ,
        -7.04138  , -6.97007  , -6.89698  , -6.82299  , -6.74881  ,
        -6.67471  , -6.60046  , -6.52598  , -6.45188  , -6.37933  ,
        -6.30927  , -6.24281  , -6.17928  , -6.11686  , -6.05597  ,
        -5.99747  , -5.94147  , -5.88801  , -5.84684  , -5.81285  ,
        -5.78014  , -5.74854  , -5.71774  , -5.68761  , -5.65825  ,
        -5.6293   , -5.60066  , -5.57245  , -5.54457  , -5.51687  ,
        -5.48932  , -5.46182  , -5.43417  , -5.40623  , -5.37801  ,
        -5.3496   , -5.32111  , -5.29248  , -5.26358  , -5.23413  ,
        -5.20392  , -5.17283  , -5.14073  , -5.1078   , -5.07426  ,
        -5.03999  , -5.00492  , -4.96953  , -4.93406  , -4.89821  ,
        -4.86196  , -4.82534  , -4.78825  , -4.75066  , -4.71243  ,
        -4.67439  , -4.63696  , -4.59945  , -4.5607   , -4.52212  ,
        -4.48434  , -4.44653  , -4.40796  , -4.36863  , -4.32842  ,
        -4.28651  , -4.24205  , -4.19486  , -4.14491  , -4.09187  ,
        -4.03446  , -3.97196  , -3.90451  , -3.83088  , -3.7496   ,
        -3.66     , -3.56112  , -3.4519   , -3.33173  , -3.20394  ,
        -3.07448  , -2.94444  , -2.8139   , -2.68294  , -2.55164  ,
        -2.42002  , -2.28814  , -2.15605  , -2.02377  , -1.89135  ,
        -1.7588   , -1.62613  , -1.49337  , -1.36127  , -1.23139  ,
        -1.10699  , -0.99209  , -0.884893 , -0.782787 , -0.683488 ,
        -0.584996 , -0.485559 , -0.383085 , -0.273456 , -0.152177 ,
        -0.0221309,  0.110786 ,  0.244405 ,  0.378378 ,  0.51182  ,
        0.64474  ,  0.777188 ,  0.909063 ,  1.04044  ,  1.1711
    ]
)


wave_8542 = np.array(
    list(wave_8542_orig[95:153]) + list(wave_8542_orig[155:194]) + list(wave_8542_orig[196:405])
)

wave_ha = wave_ha_orig

def resample_grid(line_center, min_val, max_val, num_points):
    grid_wave = list()

    # grid_wave.append(line_center)

    separation = (max_val - min_val) / num_points

    for w in np.arange(min_val, max_val, separation):
        grid_wave.append(w + line_center)

    if line_center not in grid_wave:
        grid_wave.append(line_center)

    grid_wave.sort()

    return np.array(grid_wave)


def make_rps():

    f = h5py.File(kmeans_file, 'r+')

    fi = h5py.File(input_file, 'r')

    ind = np.where(fi['profiles'][0, 0, 0, :, 0] != 0)[0]

    profiles = fi['profiles'][0, :, :][:, :, ind]

    keys = ['rps', 'final_labels']

    for key in keys:
        if key in list(f.keys()):
            del f[key]

    labels = f['labels_'][()].reshape(19, 60).astype(np.int64)

    f['final_labels'] = labels

    total_labels = labels.max() + 1

    rps = np.zeros(
        (total_labels, ind.size, 4),
        dtype=np.float64
    )

    for i in range(total_labels):
        a, b = np.where(labels == i)
        rps[i] = np.mean(profiles[a, b], axis=0)

    f['rps'] = rps

    fi.close()

    f.close()


def make_halpha_rps():

    f = h5py.File(kmeans_file, 'r+')

    fi = h5py.File(input_halpha_file, 'r')

    ind = np.where(fi['profiles'][0, 0, 0, :, 0] != 0)[0]

    profiles = fi['profiles'][0, :, :][:, :, ind]

    keys = ['halpha_rps']

    for key in keys:
        if key in list(f.keys()):
            del f[key]

    labels = f['labels_'][()].reshape(19, 60)[0:17].astype(np.int64)

    total_labels = labels.max() + 1

    rps = np.zeros(
        (total_labels, ind.size, 4),
        dtype=np.float64
    )

    for i in range(total_labels):
        a, b = np.where(labels == i)
        rps[i] = np.mean(profiles[a, b], axis=0)

    f['halpha_rps'] = rps

    fi.close()

    f.close()


def get_farthest(whole_data, a, center, r):
    all_profiles = whole_data[a, :, r]
    difference = np.sqrt(
        np.sum(
            np.square(
                np.subtract(
                    all_profiles,
                    center
                )
            ),
            axis=1
        )
    )
    index = np.argsort(difference)[-1]
    return all_profiles[index]


def get_max_min(whole_data, a, r):
    all_profiles = whole_data[a, :, r]
    return all_profiles.max(), all_profiles.min()


def get_data(get_data=True, get_labels=True, get_rps=True):

    whole_data, labels, rps = None, None, None

    if get_data:
        f = h5py.File(input_file, 'r')

        ind = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

        whole_data = f['profiles'][0, :, :, ind, :]

        whole_data[:, :, :, 1:4] /= whole_data[:, :, :, 0][:, :, :, np.newaxis]

        whole_data = whole_data.reshape(19 * 60, ind.size, 4)

        f.close()

    f = h5py.File(kmeans_file, 'r')

    if get_labels:
        labels = f['labels_'][()]

    if get_rps:
        rps = f['rps'][()]

    f.close()

    return whole_data, labels, rps


def make_rps_plots(name='RPs'):

    whole_data, labels, rps = get_data()

    k = 0

    color = 'black'

    cm = 'Greys'

    wave = np.array(list(wave_8542))

    wave_x = np.arange(wave.size)

    xticks = list()

    xticks.append(np.argmin(np.abs(wave-6562.8)))

    xticks.append(np.argmin(np.abs(wave - 8542.09)))

    for m in range(2):

        plt.close('all')

        plt.clf()

        plt.cla()

        fig = plt.figure(figsize=(8.27, 11.69))

        subfigs = fig.subfigures(5, 3)

        for i in range(5):

            for j in range(3):

                gs = gridspec.GridSpec(2, 2)

                gs.update(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)

                r = 0

                sys.stdout.write('{}\n'.format(k))

                subfig = subfigs[i][j]

                a = np.where(labels == k)[0]

                for p in range(2):
                    for q in range(2):

                        ax1 = subfig.add_subplot(gs[r])

                        center = rps[k, :, r]

                        # farthest_profile = get_farthest(whole_data, a, center, r)

                        c, f = get_max_min(whole_data, a, r)

                        max_8542, min_8542  = c, f

                        min_8542 = min_8542 * 0.9
                        max_8542 = max_8542 * 1.1

                        in_bins_8542 = np.linspace(min_8542, max_8542, 1000)

                        H1, xedge1, yedge1 = np.histogram2d(
                            np.tile(wave_x, a.shape[0]),
                            whole_data[a, :, r].flatten(),
                            bins=(wave_x, in_bins_8542)
                        )

                        ax1.plot(
                            wave_x,
                            center,
                            color=color,
                            linewidth=0.5,
                            linestyle='solid'
                        )

                        # ax1.plot(
                        #     wave_8542,
                        #     farthest_profile,
                        #     color=color,
                        #     linewidth=0.5,
                        #     linestyle='dotted'
                        # )

                        ymesh = H1.T

                        # ymeshmax = np.max(ymesh, axis=0)

                        ymeshnorm = ymesh / ymesh.max()

                        X1, Y1 = np.meshgrid(xedge1, yedge1)

                        ax1.pcolormesh(X1, Y1, ymeshnorm, cmap=cm)

                        ax1.set_ylim(min_8542, max_8542)

                        if r == 0:
                            ax1.text(
                                0.2,
                                0.6,
                                'n = {} %'.format(
                                    np.round(a.size * 100 / labels.size, 2)
                                ),
                                transform=ax1.transAxes,
                                fontsize=8
                            )

                            ax1.text(
                                0.3,
                                0.8,
                                'RP {}'.format(k),
                                transform=ax1.transAxes,
                                fontsize=8
                            )

                        ax1.set_xticks(xticks)
                        ax1.set_xticklabels([])

                        if r == 0:
                            y_ticks = [
                                np.round(
                                    min_8542 + (max_8542 - min_8542) * 0.1,
                                    2
                                ),
                                np.round(
                                    min_8542 + (max_8542 - min_8542) * 0.8,
                                    2
                                )
                            ]
                        else:
                            y_ticks = [
                                np.round(
                                    min_8542 + (max_8542 - min_8542) * 0.1,
                                    4
                                ),
                                np.round(
                                    min_8542 + (max_8542 - min_8542) * 0.8,
                                    4
                                )
                            ]

                        ax1.set_yticks(y_ticks)
                        ax1.set_yticklabels(y_ticks)

                        ax1.tick_params(axis="y",direction="in", pad=-30)

                        r += 1

                k += 1

        fig.savefig(
            rps_plot_write_dir / 'RPs_{}.png'.format(k),
            format='png',
            dpi=300
        )

        plt.close('all')

        plt.clf()

        plt.cla()


def get_halpha_data(get_data=True, get_labels=True, get_rps=True):

    whole_data, labels, rps = None, None, None

    if get_data:
        f = h5py.File(input_halpha_file, 'r')

        ind = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

        whole_data = f['profiles'][0, :, :, ind, :]

        whole_data[:, :, :, 1:4] /= whole_data[:, :, :, 0][:, :, :, np.newaxis]

        whole_data = whole_data.reshape(17 * 60, ind.size, 4)

        f.close()

    f = h5py.File(kmeans_file, 'r')

    if get_labels:
        labels = f['final_labels'][0:17].reshape(17 * 60)

    if get_rps:
        rps = f['halpha_rps'][()]

    f.close()

    return whole_data, labels, rps


def make_halpha_rps_plots(name='RPs'):

    whole_data, labels, rps = get_halpha_data()

    k = 0

    color = 'black'

    cm = 'Greys'

    wave = np.array(list(wave_ha))

    wave_x = np.arange(wave.size)

    xticks = list()

    xticks.append(np.argmin(np.abs(wave-6562.8)))

    # xticks.append(np.argmin(np.abs(wave - 8542.09)))

    for m in range(2):

        plt.close('all')

        plt.clf()

        plt.cla()

        fig = plt.figure(figsize=(8.27, 11.69))

        subfigs = fig.subfigures(5, 3)

        for i in range(5):

            for j in range(3):

                gs = gridspec.GridSpec(2, 2)

                gs.update(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)

                r = 0

                sys.stdout.write('{}\n'.format(k))

                subfig = subfigs[i][j]

                a = np.where(labels == k)[0]

                for p in range(2):
                    for q in range(2):

                        ax1 = subfig.add_subplot(gs[r])

                        center = rps[k, :, r]

                        # farthest_profile = get_farthest(whole_data, a, center, r)

                        c, f = get_max_min(whole_data, a, r)

                        max_8542, min_8542  = c, f

                        min_8542 = min_8542 * 0.9
                        max_8542 = max_8542 * 1.1

                        in_bins_8542 = np.linspace(min_8542, max_8542, 1000)

                        H1, xedge1, yedge1 = np.histogram2d(
                            np.tile(wave_x, a.shape[0]),
                            whole_data[a, :, r].flatten(),
                            bins=(wave_x, in_bins_8542)
                        )

                        ax1.plot(
                            wave_x,
                            center,
                            color=color,
                            linewidth=0.5,
                            linestyle='solid'
                        )

                        # ax1.plot(
                        #     wave_8542,
                        #     farthest_profile,
                        #     color=color,
                        #     linewidth=0.5,
                        #     linestyle='dotted'
                        # )

                        ymesh = H1.T

                        # ymeshmax = np.max(ymesh, axis=0)

                        ymeshnorm = ymesh / ymesh.max()

                        X1, Y1 = np.meshgrid(xedge1, yedge1)

                        ax1.pcolormesh(X1, Y1, ymeshnorm, cmap=cm)

                        ax1.set_ylim(min_8542, max_8542)

                        if r == 0:
                            ax1.text(
                                0.2,
                                0.6,
                                'n = {} %'.format(
                                    np.round(a.size * 100 / labels.size, 2)
                                ),
                                transform=ax1.transAxes,
                                fontsize=8
                            )

                            ax1.text(
                                0.3,
                                0.8,
                                'RP {}'.format(k),
                                transform=ax1.transAxes,
                                fontsize=8
                            )

                        ax1.set_xticks(xticks)
                        ax1.set_xticklabels([])

                        if r == 0:
                            y_ticks = [
                                np.round(
                                    min_8542 + (max_8542 - min_8542) * 0.1,
                                    2
                                ),
                                np.round(
                                    min_8542 + (max_8542 - min_8542) * 0.8,
                                    2
                                )
                            ]
                        else:
                            y_ticks = [
                                np.round(
                                    min_8542 + (max_8542 - min_8542) * 0.1,
                                    4
                                ),
                                np.round(
                                    min_8542 + (max_8542 - min_8542) * 0.8,
                                    4
                                )
                            ]

                        ax1.set_yticks(y_ticks)
                        ax1.set_yticklabels(y_ticks)

                        ax1.tick_params(axis="y",direction="in", pad=-30)

                        r += 1

                k += 1

        fig.savefig(
            rps_plot_write_dir / 'Ha_RPs_{}.png'.format(k),
            format='png',
            dpi=300
        )

        plt.close('all')

        plt.clf()

        plt.cla()


def get_data_for_label_polarisation_map():

    ind_photosphere = np.array(list(range(0, 58)) + list(range(58, 97)))

    ind_chromosphere = np.array(list(range(97, 306)))

    f = h5py.File(input_file, 'r')

    ind = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    whole_data = f['profiles'][0, :, :, ind, :]

    intensity = whole_data[:, :, 0, 0]

    linpol_p = np.mean(
        np.sqrt(
            np.sum(
                np.square(
                    whole_data[:, :, ind_photosphere, 1:3]
                ),
                3
            )
        ) / whole_data[:, :, ind_photosphere, 0],
        2
    )
    
    linpol_c = np.mean(
        np.sqrt(
            np.sum(
                np.square(
                    whole_data[:, :, ind_chromosphere, 1:3]
                ),
                3
            )
        ) / whole_data[:, :, ind_chromosphere, 0],
        2
    )

    circpol_p = np.mean(
        np.divide(
            np.abs(whole_data[:, :, ind_photosphere, 3]),
            whole_data[:, :, ind_photosphere, 0]
        ),
        2
    )

    circpol_c = np.mean(
        np.divide(
            np.abs(whole_data[:, :, ind_chromosphere, 3]),
            whole_data[:, :, ind_chromosphere, 0]
        ),
        2
    )

    f.close()

    f = h5py.File(kmeans_file, 'r')

    labels = f['final_labels'][()]

    f.close()

    return intensity, linpol_p, linpol_c, circpol_p, circpol_c, labels


def plot_rp_map_fov():

    intensity, linpol_p, linpol_c, circpol_p, circpol_c, labels = get_data_for_label_polarisation_map()

    plt.close('all')

    plt.clf()

    plt.cla()

    fig, axs = plt.subplots(3, 2, figsize=(6, 9))

    im00 = axs[0][0].imshow(intensity, cmap='gray', origin='lower')

    im01 = axs[0][1].imshow(labels, cmap='gray', origin='lower')

    im10 = axs[1][0].imshow(linpol_p, cmap='gray', origin='lower')

    im11 = axs[1][1].imshow(circpol_p, cmap='gray', origin='lower')

    im20 = axs[2][0].imshow(linpol_c, cmap='gray', origin='lower')

    im21 = axs[2][1].imshow(circpol_c, cmap='gray', origin='lower')

    fig.colorbar(im00, ax=axs[0][0], orientation='horizontal')

    fig.colorbar(im01, ax=axs[0][1], orientation='horizontal')

    fig.colorbar(im10, ax=axs[1][0], orientation='horizontal')

    fig.colorbar(im11, ax=axs[1][1], orientation='horizontal')

    fig.colorbar(im20, ax=axs[2][0], orientation='horizontal')

    fig.colorbar(im21, ax=axs[2][1], orientation='horizontal')

    fig.tight_layout()

    fig.savefig(
        rps_plot_write_dir / 'FoV_RPs_pol_map.pdf',
        format='pdf',
        dpi=300
    )

    plt.close('all')

    plt.clf()

    plt.cla()


def make_stic_inversion_files(rps=None):

    wave_indices = [[95, 153], [155, 194], [196, 405]]

    line_indices = [[0, 58], [58, 97], [97, 306]]

    core_indices = [[19, 36], [18, 27], [85, 120]]

    wave_names = ['SiI_8536', 'FeI_8538', 'CaII_8542']

    f = h5py.File(kmeans_file, 'r')

    ca = None

    if rps is None:
        rps = range(f['rps'].shape[0])

    rps = np.array(rps)

    for wave_indice, line_indice, core_indice, wave_name in zip(wave_indices, line_indices, core_indices, wave_names):
        wc8, ic8 = findgrid(wave_8542_orig[wave_indice[0]:wave_indice[1]], (wave_8542_orig[wave_indice[0]:wave_indice[1]][10] - wave_8542_orig[wave_indice[0]:wave_indice[1]][9])*0.25, extra=8)

        ca_8 = sp.profile(nx=rps.size, ny=1, ns=4, nw=wc8.size)

        ca_8.wav[:] = wc8[:]

        ca_8.dat[0, 0, :, ic8, :] = np.transpose(
            f['rps'][rps][:, line_indice[0]:line_indice[1]],
            axes=(1, 0, 2)
        )

        ca_8.weights[:, :] = 1.e16 # Very high value means weight zero
        ca_8.weights[ic8, 0] = 0.004
        ca_8.weights[ic8[core_indice[0]:core_indice[1]], 0] = 0.002
        # ca_8.weights[ic8, 3] = ca_8.weights[ic8, 0] / 2

        if ca is None:
            ca = ca_8
        else:
            ca += ca_8

        broadening_filename = 'gaussian_broadening_{}_pixel_{}.h5'.format(21.7, wave_name)

        lab = "region = {0:10.5f}, {1:8.5f}, {2:3d}, {3:e}, {4}"
        print(" ")
        print("Regions information for the input file:" )
        print(lab.format(ca_8.wav[0], ca_8.wav[1]-ca_8.wav[0], ca_8.wav.size, cont[0],  'spectral, {}'.format(broadening_filename)))
        print("(w0, dw, nw, normalization, degradation_type, instrumental_profile file)")
        print(" ")

    # wha, iha = findgrid(wave_ha, (wave_ha[1] - wave_ha[0]) * 0.25, extra=8)
    #
    # ha = sp.profile(nx=rps.size, ny=1, ns=4, nw=wha.size)
    #
    # ha.wav[:] = wha[:]
    #
    # ha.dat[0, 0, :, iha, :] = np.transpose(
    #     f['rps'][rps][:, 306:],
    #     axes=(1, 0, 2)
    # )
    #
    # ha.weights[:, :] = 1.e16  # Very high value means weight zero
    # ha.weights[iha, 0] = 0.004
    # ha.weights[iha[18:46], 0] = 0.002
    # ha.weights[iha[69:186], 0] = 0.002
    # ha.weights[iha[405:432], 0] = 0.002
    # # ha.weights[iha, 3] = ca_8.weights[iha, 0] / 2

    all_profiles = ca# + ha
    if rps.size != f['rps'].shape[0]:
        writefilename = 'rps_stic_profiles_x_{}_y_1.nc'.format('_'.join([str(_rp) for _rp in rps]))
    else:
        writefilename = 'rps_stic_profiles_x_{}_y_1.nc'.format(rps.size)
    all_profiles.write(
        atmos_rp_write_path / writefilename
    )


def make_stic_inversion_files_halpha_ca_both(rps=None):

    wave_indices = [[95, 153], [155, 194], [196, 405]]

    line_indices = [[0, 58], [58, 97], [97, 306]]

    core_indices = [[19, 36], [18, 27], [85, 120]]

    wave_names = ['SiI_8536', 'FeI_8538', 'CaII_8542']

    f = h5py.File(kmeans_file, 'r')

    ca = None

    if rps is None:
        rps = range(f['rps'].shape[0])

    rps = np.array(rps)

    for wave_indice, line_indice, core_indice, wave_name in zip(wave_indices, line_indices, core_indices, wave_names):
        wc8, ic8 = findgrid(wave_8542_orig[wave_indice[0]:wave_indice[1]], (wave_8542_orig[wave_indice[0]:wave_indice[1]][10] - wave_8542_orig[wave_indice[0]:wave_indice[1]][9])*0.25, extra=8)

        ca_8 = sp.profile(nx=rps.size, ny=1, ns=4, nw=wc8.size)

        ca_8.wav[:] = wc8[:]

        ca_8.dat[0, 0, :, ic8, :] = np.transpose(
            f['rps'][rps][:, line_indice[0]:line_indice[1]],
            axes=(1, 0, 2)
        )

        ca_8.weights[:, :] = 1.e16 # Very high value means weight zero
        ca_8.weights[ic8, 0] = 0.004
        ca_8.weights[ic8[core_indice[0]:core_indice[1]], 0] = 0.002
        # ca_8.weights[ic8, 3] = ca_8.weights[ic8, 0] / 2

        if ca is None:
            ca = ca_8
        else:
            ca += ca_8

        broadening_filename = 'gaussian_broadening_{}_pixel_{}.h5'.format(21.7, wave_name)

        lab = "region = {0:10.5f}, {1:8.5f}, {2:3d}, {3:e}, {4}"
        print(" ")
        print("Regions information for the input file:" )
        print(lab.format(ca_8.wav[0], ca_8.wav[1]-ca_8.wav[0], ca_8.wav.size, cont[0],  'spectral, {}'.format(broadening_filename)))
        print("(w0, dw, nw, normalization, degradation_type, instrumental_profile file)")
        print(" ")

    wha, iha = findgrid(wave_ha, (wave_ha[1] - wave_ha[0]) * 0.25, extra=8)

    ha = sp.profile(nx=rps.size, ny=1, ns=4, nw=wha.size)

    ha.wav[:] = wha[:]

    ha.dat[0, 0, :, iha, :] = np.transpose(
        f['halpha_rps'][rps],
        axes=(1, 0, 2)
    )

    ha.weights[:, :] = 1.e16  # Very high value means weight zero
    ha.weights[iha, 0] = 0.008
    ha.weights[iha[18:46], 0] = 0.004
    ha.weights[iha[69:186], 0] = 0.002
    ha.weights[iha[405:432], 0] = 0.004
    # ha.weights[iha, 3] = ca_8.weights[iha, 0] / 2

    all_profiles = ca + ha
    if rps.size != f['rps'].shape[0]:
        writefilename = 'ha_ca_rps_stic_profiles_x_{}_y_1.nc'.format('_'.join([str(_rp) for _rp in rps]))
    else:
        writefilename = 'ha_ca_rps_stic_profiles_x_{}_y_1.nc'.format(rps.size)
    all_profiles.write(
        atmos_rp_write_path / writefilename
    )


def generate_input_atmos_file(length=30, temp=None, vlos=None, blong=0, name=''):

    f = h5py.File(falc_file_path, 'r')

    m = sp.model(nx=length, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = f['ltau500'][0, 0, 0]

    m.pgas[:, :, :] = 1

    if temp is not None:
        cs = CubicSpline(temp[0], temp[1])
        tp = cs(f['ltau500'][0, 0, 0])
    else:
        tp = f['temp'][0, 0, 0]
    if vlos is not None:
        cs = CubicSpline(vlos[0], vlos[1])
        vl = cs(f['ltau500'][0, 0, 0])
    else:
        vl = f['vlos'][0, 0, 0]

    m.temp[:, :, :] = tp

    m.vlos[:, :, :] = vl

    m.vturb[:, :, :] = 0

    m.Bln[:, :, :] = blong

    m.write(
        atmos_rp_write_path / 'atmos_{}_{}.nc'.format(length, name)
    )


def generate_input_atmos_file_from_previous_result(result_filename=None, rps=None):
    if result_filename is None:
        print ('Give input atmos file')
        sys.exit(1)

    result_filename = Path(result_filename)

    f = h5py.File(result_filename, 'r')

    if not rps:
        rps = range(f['ltau500'].shape[2])

    rps = np.array(rps)

    m = sp.model(nx=rps.size, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = f['ltau500'][0, 0, rps]

    m.pgas[:, :, :] = 1

    m.temp[:, :, :] = f['temp'][0, 0, rps]

    m.vlos[:, :, :] = f['vlos'][0, 0, rps]

    m.vturb[:, :, :] = f['vturb'][0, 0, rps]

    m.Bln[:, :, :] = 100

    m.write(
        atmos_rp_write_path / '{}_{}_blong_100.nc'.format(result_filename.name, '_'.join([str(_rp) for _rp in rps]))
    )


def make_rps_inversion_result_plots(nodes_temp=None, nodes_vlos=None, nodes_vturb=None, nodes_blos=None):

    # rps_atmos_result = Path(
    #     '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/only_Stokes_I/rps_stic_profiles_x_30_y_1_cycle_1_t_6_vl_3_vt_4_atmos.nc'
    # )
    #
    # rps_profs_result = Path(
    #     '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/only_Stokes_I/rps_stic_profiles_x_30_y_1_cycle_1_t_6_vl_3_vt_4_profs.nc'
    # )

    rps_atmos_result = Path(
        '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/new_inversions/rps_stic_profiles_x_30_y_1_cycle_1_t_0_vl_0_vt_0_blong_2_atmos.nc'
    )

    rps_profs_result = Path(
        '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/new_inversions/rps_stic_profiles_x_30_y_1_y_1_cycle_1_t_0_vl_0_vt_0_blong_2_profs.nc'
    )

    rps_input_profs = Path(
        '/home/harsh/SpinorNagaraju/maps_1/stic/rps_stic_profiles_x_30_y_1.nc'
    )
    
    rps_plot_write_dir = Path(
        '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/new_inversions/'
    )

    finputprofs = h5py.File(rps_input_profs, 'r')

    fatmosresult = h5py.File(rps_atmos_result, 'r')

    fprofsresult = h5py.File(rps_profs_result, 'r')

    ind = np.where(finputprofs['profiles'][0, 0, 0, :, 0] != 0)[0]

    for i, k in enumerate(range(30)):
        print(i)
        plt.close('all')

        plt.clf()

        plt.cla()

        fig, axs = plt.subplots(3, 2, figsize=(12, 18))

        axs[0][0].plot(finputprofs['wav'][ind], finputprofs['profiles'][0, 0, i, ind, 0], color='orange', linewidth=0.5)

        axs[0][0].plot(fprofsresult['wav'][ind], fprofsresult['profiles'][0, 0, i, ind, 0], color='brown', linewidth=0.5)

        axs[0][1].plot(
            finputprofs['wav'][ind],
            finputprofs['profiles'][0, 0, i, ind, 3] / finputprofs['profiles'][0, 0, i, ind, 0],
            color='orange',
            linewidth=0.5
        )

        axs[0][1].plot(
            fprofsresult['wav'][ind],
            fprofsresult['profiles'][0, 0, i, ind, 3] / fprofsresult['profiles'][0, 0, i, ind, 0],
            color='brown',
            linewidth=0.5
        )

        axs[1][0].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['temp'][0, 0, i] / 1e3, color='brown')

        axs[1][1].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['vlos'][0, 0, i] / 1e5, color='brown')

        axs[2][0].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['vturb'][0, 0, i] / 1e5, color='brown')

        axs[2][1].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['blong'][0, 0, i], color='brown')

        if nodes_temp is not None:
            for nn in nodes_temp:
                axs[1][0].axvline(nn, linestyle='--', linewidth=0.5)

        if nodes_vlos is not None:
            for nn in nodes_vlos:
                axs[1][1].axvline(nn, linestyle='--', linewidth=0.5)

        if nodes_vturb is not None:
            for nn in nodes_vturb:
                axs[2][0].axvline(nn, linestyle='--', linewidth=0.5)

        if nodes_blos is not None:
            for nn in nodes_blos:
                axs[2][1].axvline(nn, linestyle='--', linewidth=0.5)

        axs[0][0].set_xlabel(r'$\lambda(\AA)$')
        axs[0][0].set_ylabel(r'$I/I_{c}$')

        axs[0][1].set_xlabel(r'$\lambda(\AA)$')
        axs[0][1].set_ylabel(r'$V/I$')

        axs[1][0].set_xlabel(r'$\log(\tau_{500})$')
        axs[1][0].set_ylabel(r'$T[kK]$')

        axs[1][1].set_xlabel(r'$\log(\tau_{500})$')
        axs[1][1].set_ylabel(r'$V_{LOS}[Kms^{-1}]$')

        axs[2][0].set_xlabel(r'$\log(\tau_{500})$')
        axs[2][0].set_ylabel(r'$V_{turb}[Kms^{-1}]$')

        axs[2][1].set_xlabel(r'$\log(\tau_{500})$')
        axs[2][1].set_ylabel(r'$B_{long}[G]$')

        fig.tight_layout()

        fig.savefig(rps_plot_write_dir /'RPs_{}.pdf'.format(k), format='pdf', dpi=300)

        plt.close('all')

        plt.clf()

        plt.cla()
    fprofsresult.close()

    fatmosresult.close()

    finputprofs.close()


def make_pixels_inversion_result_plots():

    # rps_atmos_result = Path(
    #     '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/only_Stokes_I/rps_stic_profiles_x_30_y_1_cycle_1_t_6_vl_3_vt_4_atmos.nc'
    # )
    #
    # rps_profs_result = Path(
    #     '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/only_Stokes_I/rps_stic_profiles_x_30_y_1_cycle_1_t_6_vl_3_vt_4_profs.nc'
    # )

    rps_atmos_result = Path(
        '/home/harsh/SpinorNagaraju/maps_1/stic/run_nagaraju/alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_total_33_cycle_1_t_6_vl_5_vt_4_blong_6_atmos.nc'
    )

    rps_profs_result = Path(
        '/home/harsh/SpinorNagaraju/maps_1/stic/run_nagaraju/alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_total_33_cycle_1_t_6_vl_5_vt_4_blong_6_profs.nc'
    )

    rps_input_profs = Path(
        '/home/harsh/SpinorNagaraju/maps_1/stic/run_nagaraju/alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_total_33.nc'
    )

    rps_plot_write_dir = Path(
        '/home/harsh/SpinorNagaraju/maps_1/stic/run_nagaraju/'
    )

    finputprofs = h5py.File(rps_input_profs, 'r')

    fatmosresult = h5py.File(rps_atmos_result, 'r')

    fprofsresult = h5py.File(rps_profs_result, 'r')

    ind = np.where(finputprofs['profiles'][0, 0, 0, :, 0] != 0)[0]

    # for i, (xx, yy) in enumerate(zip([12, 12], [49, 31])):
    for i in range(33):
        print(i)
        plt.close('all')

        plt.clf()

        plt.cla()

        fig, axs = plt.subplots(3, 2, figsize=(12, 18))

        axs[0][0].plot(finputprofs['wav'][ind], finputprofs['profiles'][0, 0, i, ind, 0], color='orange', linewidth=0.5)

        axs[0][0].plot(fprofsresult['wav'][ind], fprofsresult['profiles'][0, 0, i, ind, 0], color='brown', linewidth=0.5)

        axs[0][1].plot(
            finputprofs['wav'][ind],
            finputprofs['profiles'][0, 0, i, ind, 3] / finputprofs['profiles'][0, 0, i, ind, 0],
            color='orange',
            linewidth=0.5
        )

        axs[0][1].plot(
            fprofsresult['wav'][ind],
            fprofsresult['profiles'][0, 0, i, ind, 3] / fprofsresult['profiles'][0, 0, i, ind, 0],
            color='brown',
            linewidth=0.5
        )

        axs[1][0].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['temp'][0, 0, i] / 1e3, color='brown')

        axs[1][1].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['vlos'][0, 0, i] / 1e5, color='brown')

        axs[2][0].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['vturb'][0, 0, i] / 1e5, color='brown')

        axs[2][1].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['blong'][0, 0, i], color='brown')

        axs[0][0].set_xlabel(r'$\lambda(\AA)$')
        axs[0][0].set_ylabel(r'$I/I_{c}$')

        axs[0][1].set_xlabel(r'$\lambda(\AA)$')
        axs[0][1].set_ylabel(r'$V/I$')

        axs[1][0].set_xlabel(r'$\log(\tau_{500})$')
        axs[1][0].set_ylabel(r'$T[kK]$')

        axs[1][1].set_xlabel(r'$\log(\tau_{500})$')
        axs[1][1].set_ylabel(r'$V_{LOS}[Kms^{-1}]$')

        axs[2][0].set_xlabel(r'$\log(\tau_{500})$')
        axs[2][0].set_ylabel(r'$V_{turb}[Kms^{-1}]$')

        axs[2][1].set_xlabel(r'$\log(\tau_{500})$')
        axs[2][1].set_ylabel(r'$B_{long}[G]$')

        fig.tight_layout()

        # fig.savefig(rps_plot_write_dir /'Pixels_{}_{}.pdf'.format(xx, yy), format='pdf', dpi=300)
        fig.savefig(rps_plot_write_dir / 'Pixels_{}.pdf'.format(i), format='pdf', dpi=300)

        plt.close('all')

        plt.clf()

        plt.cla()
    fprofsresult.close()

    fatmosresult.close()

    finputprofs.close()


def make_ha_ca_rps_inversion_result_plots():

    # rps_atmos_result = Path(
    #     '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/only_Stokes_I/rps_stic_profiles_x_30_y_1_cycle_1_t_6_vl_3_vt_4_atmos.nc'
    # )
    #
    # rps_profs_result = Path(
    #     '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/only_Stokes_I/rps_stic_profiles_x_30_y_1_cycle_1_t_6_vl_3_vt_4_profs.nc'
    # )

    rps_atmos_result = Path(
        '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/ha_ca_rps_stic_profiles_x_0_1_2_4_5_7_9_10_11_13_14_17_18_28_29_y_1_cycle_1_t_8_vl_6_vt_4_blong_0_atmos.nc'
    )

    rps_profs_result = Path(
        '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/ha_ca_rps_stic_profiles_x_0_1_2_4_5_7_9_10_11_13_14_17_18_28_29_y_1_cycle_1_t_8_vl_6_vt_4_blong_0_profs.nc'
    )

    rps_input_profs = Path(
        '/home/harsh/SpinorNagaraju/maps_1/stic/ha_ca_rps_stic_profiles_x_0_1_2_4_5_7_9_10_11_13_14_17_18_28_29_y_1.nc'
    )

    rps_plot_write_dir = Path(
        '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/'
    )

    finputprofs = h5py.File(rps_input_profs, 'r')

    fatmosresult = h5py.File(rps_atmos_result, 'r')

    fprofsresult = h5py.File(rps_profs_result, 'r')

    ind = np.where(finputprofs['profiles'][0, 0, 0, :, 0] != 0)[0]

    for i, k in enumerate([0, 1, 2, 4, 5, 7, 9, 10, 11, 13, 14, 17, 18, 28, 29]):
        print(i)
        plt.close('all')

        plt.clf()

        plt.cla()

        fig, axs = plt.subplots(4, 2, figsize=(12, 18))

        axs[0][0].plot(finputprofs['wav'][ind[0:306]], finputprofs['profiles'][0, 0, i, ind[0:306], 0], color='orange', linewidth=0.5)

        axs[0][0].plot(fprofsresult['wav'][ind[0:306]], fprofsresult['profiles'][0, 0, i, ind[0:306], 0], color='brown', linewidth=0.5)

        axs[0][1].plot(
            finputprofs['wav'][ind[0:306]],
            finputprofs['profiles'][0, 0, i, ind[0:306], 3] / finputprofs['profiles'][0, 0, i, ind[0:306], 0],
            color='orange',
            linewidth=0.5
        )

        axs[0][1].plot(
            fprofsresult['wav'][ind[0:306]],
            fprofsresult['profiles'][0, 0, i, ind[0:306], 3] / fprofsresult['profiles'][0, 0, i, ind[0:306], 0],
            color='brown',
            linewidth=0.5
        )

        axs[1][0].plot(finputprofs['wav'][ind[306:]], finputprofs['profiles'][0, 0, i, ind[306:], 0], color='orange', linewidth=0.5)

        axs[1][0].plot(fprofsresult['wav'][ind[306:]], fprofsresult['profiles'][0, 0, i, ind[306:], 0], color='brown', linewidth=0.5)

        axs[1][1].plot(
            finputprofs['wav'][ind[306:]],
            finputprofs['profiles'][0, 0, i, ind[306:], 3] / finputprofs['profiles'][0, 0, i, ind[306:], 0],
            color='orange',
            linewidth=0.5
        )

        axs[1][1].plot(
            fprofsresult['wav'][ind[306:]],
            fprofsresult['profiles'][0, 0, i, ind[306:], 3] / fprofsresult['profiles'][0, 0, i, ind[306:], 0],
            color='brown',
            linewidth=0.5
        )

        axs[2][0].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['temp'][0, 0, i] / 1e3, color='brown')

        axs[2][1].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['vlos'][0, 0, i] / 1e5, color='brown')

        axs[3][0].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['vturb'][0, 0, i] / 1e5, color='brown')

        axs[3][1].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['blong'][0, 0, i], color='brown')

        axs[0][0].set_xlabel(r'$\lambda(\AA)$')
        axs[0][0].set_ylabel(r'$I/I_{c}$')

        axs[0][1].set_xlabel(r'$\lambda(\AA)$')
        axs[0][1].set_ylabel(r'$V/I$')

        axs[1][0].set_xlabel(r'$\lambda(\AA)$')
        axs[1][0].set_ylabel(r'$I/I_{c}$')

        axs[1][1].set_xlabel(r'$\lambda(\AA)$')
        axs[1][1].set_ylabel(r'$V/I$')

        axs[2][0].set_xlabel(r'$\log(\tau_{500})$')
        axs[2][0].set_ylabel(r'$T[kK]$')

        axs[2][1].set_xlabel(r'$\log(\tau_{500})$')
        axs[2][1].set_ylabel(r'$V_{LOS}[Kms^{-1}]$')

        axs[3][0].set_xlabel(r'$\log(\tau_{500})$')
        axs[3][0].set_ylabel(r'$V_{turb}[Kms^{-1}]$')

        axs[3][1].set_xlabel(r'$\log(\tau_{500})$')
        axs[3][1].set_ylabel(r'$B_{long}[G]$')

        fig.tight_layout()

        fig.savefig(rps_plot_write_dir /'HA_CA_RPs_{}.pdf'.format(k), format='pdf', dpi=300)

        plt.close('all')

        plt.clf()

        plt.cla()
    fprofsresult.close()

    fatmosresult.close()

    finputprofs.close()


def combine_rps_atmos():
    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/full_stokes_blos_2/')

    file_1 = 'rps_stic_profiles_x_30_y_1_cycle_1_t_0_vl_0_vt_0_blos_2_atmos.nc'

    file_2 = 'rps_stic_profiles_x_3_12_25_y_1_cycle_1_t_0_vl_0_vt_0_blos_2_atmos.nc'

    rps_1 = list(set(range(30)) - set([3, 12, 25]))

    # rps_2 = [3, 12, 25]

    m = sp.model(nx=30, ny=1, nt=1, ndep=150)

    f = h5py.File(base_path / file_1, 'r')
    m.ltau[:, :, :] = f['ltau500'][0, 0, 0]
    f.close()

    m.pgas[:, :, :] = 1

    for i in range(30):

        if i in rps_1:
            f = h5py.File(base_path / file_1, 'r')
            file = 1
        else:
            f = h5py.File(base_path / file_2, 'r')
            file = 2

        if file == 1:
            mj = i
        else:
            if i == 3:
                mj = 0
            elif i == 12:
                mj = 1
            else:
                mj = 2

        m.temp[0, 0, i] = f['temp'][0, 0, mj]

        m.vlos[0, 0, i] = f['vlos'][0, 0, mj]

        m.vturb[0, 0, i] = f['vturb'][0, 0, mj]

        m.Bln[0, 0, i] = f['blong'][0, 0, mj]

        f.close()

    m.write(
        base_path / 'combined_rps_stic_profiles_x_30_y_1_cycle_1_t_0_vl_0_vt_0_blos_2_atmos.nc'
    )


def prepare_get_params(param):
    def get_params(rp):
        return param[rp]

    return get_params


def full_map_generate_input_atmos_file_from_previous_result():
    result_filename = Path('/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/full_stokes_6343/rps_stic_profiles_x_30_y_1_cycle_1_t_6_vl_3_vt_4_blos_3_atmos.nc')

    f = h5py.File(result_filename, 'r')

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic')

    fc = h5py.File(base_path / 'chosen_out_30.h5', 'r')

    get_temp = prepare_get_params(f['temp'][0, 0])

    get_vlos = prepare_get_params(f['vlos'][0, 0])

    get_vturb = prepare_get_params(f['vturb'][0, 0])

    get_blos = prepare_get_params(f['blong'][0, 0])

    vec_get_temp = np.vectorize(get_temp, signature='(x,y)->(x,y,z)')

    vec_get_vlos = np.vectorize(get_vlos, signature='(x,y)->(x,y,z)')

    vec_get_vturb = np.vectorize(get_vturb, signature='(x,y)->(x,y,z)')

    vec_get_blos = np.vectorize(get_blos, signature='(x,y)->(x,y,z)')

    labels = fc['final_labels'][()]

    m = sp.model(nx=60, ny=19, nt=1, ndep=150)

    m.ltau[:, :, :] = f['ltau500'][0, 0, 0]

    m.pgas[:, :, :] = 1

    m.temp[0, :, :] = vec_get_temp(labels)

    m.vlos[0, :, :] = vec_get_vlos(labels)

    m.vturb[0, :, :] = vec_get_vturb(labels)

    m.Bln[0, :, :] = vec_get_blos(labels)

    m.write(
        base_path / 'input_atmos_19_60_from_6343_rps.nc'
    )

    fc.close()

    f.close()


'''
Quiet Profiles: 0, 1, 2, 4, 5, 7, 9, 10, 11, 13, 14, 17, 18, 28, 29
Interesting Profiles No Vlos: 3, 6, 8, 12, 15, 16, 19, 20, 22, 23, 25
Emission Red RP: 21, 24
Emission Blue RP: 26, 27
'''


def get_rp_atmos():
    quiet_profiles = [0, 1, 2, 4, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17, 18, 19, 22, 23, 25, 26, 27, 28]
    quiet_profiles_2 = [29]
    spot_profiles = [3, 9, 10, 13, 20, 21]
    emission_profiles = [24]

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/')
    input_path = base_path / 'RPs_plots/inversions/'

    quiet_file = input_path / 'rps_stic_profiles_x_0_1_2_4_5_6_7_8_11_12_14_15_16_17_18_19_22_23_25_26_27_28_y_1_cycle_1_t_5_vl_3_vt_4_blong_3_atmos.nc'
    quiet_file_2 = input_path / 'rps_stic_profiles_x_29_y_1_cycle_1_t_5_vl_3_vt_4_blong_3_atmos.nc'
    spot_file = input_path / 'rps_stic_profiles_x_3_9_10_13_20_21_y_1_cycle_2_t_5_vl_3_vt_4_blong_3_atmos.nc'
    emission_file = input_path / 'rps_stic_profiles_x_24_y_1_cycle_1_t_5_vl_3_vt_4_blong_3_atmos.nc'

    temp = np.zeros((30, 150))
    vlos = np.zeros((30, 150))
    vturb = np.zeros((30, 150))
    blong = np.zeros((30, 150))

    f = h5py.File(quiet_file, 'r')
    for index, i in enumerate(quiet_profiles):
        temp[i] = f['temp'][0, 0, index]
        vlos[i] = f['vlos'][0, 0, index]
        vturb[i] = f['vturb'][0, 0, index]
        blong[i] = f['blong'][0, 0, index]
    f.close()

    f = h5py.File(spot_file, 'r')
    for index, i in enumerate(spot_profiles):
        temp[i] = f['temp'][0, 0, index]
        vlos[i] = f['vlos'][0, 0, index]
        vturb[i] = f['vturb'][0, 0, index]
        blong[i] = f['blong'][0, 0, index]
    f.close()

    f = h5py.File(quiet_file_2, 'r')
    for index, i in enumerate(quiet_profiles_2):
        temp[i] = f['temp'][0, 0, index]
        vlos[i] = f['vlos'][0, 0, index]
        vturb[i] = f['vturb'][0, 0, index]
        blong[i] = f['blong'][0, 0, index]
    f.close()

    f = h5py.File(emission_file, 'r')
    for index, i in enumerate(emission_profiles):
        temp[i] = f['temp'][0, 0, index]
        vlos[i] = f['vlos'][0, 0, index]
        vturb[i] = f['vturb'][0, 0, index]
        blong[i] = f['blong'][0, 0, index]
    f.close()

    return temp, vlos, vturb, blong


def generate_actual_inversion_files_quiet():

    line_indices = [[0, 58], [58, 97], [97, 306]]

    core_indices = [[19, 36], [18, 27], [85, 120]]

    quiet_profiles = [0, 1, 2, 4, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17, 18, 19, 22, 23, 25, 26, 27, 28, 29]

    temp, vlos, vturb, blong = get_rp_atmos()

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/')
    actual_file = base_path / 'processed_inputs/alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc'
    write_path = base_path / 'fulldata_inversions'

    kmeans_file = base_path / 'chosen_out_30.h5'

    f = h5py.File(kmeans_file, 'r')

    labels = f['final_labels'][()]

    f.close()

    a_list = list()
    b_list = list()
    rp_final = list()

    for profile in quiet_profiles:
        a, b = np.where(labels == profile)
        a_list += list(a)
        b_list += list(b)
        rp_final += list(np.ones(a.shape[0]) * profile)

    a_arr = np.array(a_list)
    b_arr = np.array(b_list)
    rp_final = np.array(rp_final)

    pixel_indices = np.zeros((3, a_arr.size), dtype=np.int64)

    pixel_indices[0] = a_arr
    pixel_indices[1] = b_arr
    pixel_indices[2] = rp_final

    fo = h5py.File(
        write_path / 'pixel_indices_quiet_total_{}.h5'.format(
            rp_final.size
        ), 'w'
    )

    fo['pixel_indices'] = pixel_indices

    fo.close()

    f = h5py.File(actual_file, 'r')

    wc8 = f['wav'][()]

    ic8 = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    ca_8 = sp.profile(nx=rp_final.size, ny=1, ns=4, nw=wc8.size)

    ca_8.wav[:] = wc8[:]

    ca_8.dat[0, 0, :, ic8, :] = np.transpose(f['profiles'][()][0, a_arr, b_arr, :, :][:, ic8], axes=(1, 0, 2))

    ca_8.weights[:, :] = 1.e16
    ca_8.weights[ic8, 0] = 0.004
    for line_indice, core_indice in zip(line_indices, core_indices):
        ca_8.weights[ic8[line_indice[0]+core_indice[0]:line_indice[0]+core_indice[1]], 0] = 0.002

    ca_8.weights[ic8, 3] = ca_8.weights[ic8, 0]

    ca_8.weights[ic8, 3] /= 2

    ca_8.weights[ic8[97+85:97+120], 3] /= 2

    ca_8.write(
        write_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_quiet_total_{}.nc'.format(rp_final.size)
    )

    m = sp.model(nx=a_arr.size, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = ltau

    m.pgas[:, :, :] = 1.0

    m.temp[0, 0] = temp[labels[a_arr, b_arr]]

    m.vlos[0, 0] = vlos[labels[a_arr, b_arr]]

    m.vturb[0, 0] = vturb[labels[a_arr, b_arr]]

    m.Bln[0, 0] = blong[labels[a_arr, b_arr]]

    write_filename = write_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_quiet_total_{}_initial_atmos.nc'.format(rp_final.size)

    m.write(str(write_filename))


def generate_actual_inversion_files_spot():

    line_indices = [[0, 58], [58, 97], [97, 306]]

    core_indices = [[19, 36], [18, 27], [85, 120]]

    spot_profiles = [3, 9, 10, 13, 20, 21]

    temp, vlos, vturb, blong = get_rp_atmos()

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/')
    actual_file = base_path / 'processed_inputs/alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc'
    write_path = base_path / 'fulldata_inversions'

    kmeans_file = base_path / 'chosen_out_30.h5'

    f = h5py.File(kmeans_file, 'r')

    labels = f['final_labels'][()]

    f.close()

    a_list = list()
    b_list = list()
    rp_final = list()

    for profile in spot_profiles:
        a, b = np.where(labels == profile)
        a_list += list(a)
        b_list += list(b)
        rp_final += list(np.ones(a.shape[0]) * profile)

    a_arr = np.array(a_list)
    b_arr = np.array(b_list)
    rp_final = np.array(rp_final)

    pixel_indices = np.zeros((3, a_arr.size), dtype=np.int64)

    pixel_indices[0] = a_arr
    pixel_indices[1] = b_arr
    pixel_indices[2] = rp_final

    fo = h5py.File(
        write_path / 'pixel_indices_spot_total_{}.h5'.format(
            rp_final.size
        ), 'w'
    )

    fo['pixel_indices'] = pixel_indices

    fo.close()

    f = h5py.File(actual_file, 'r')

    wc8 = f['wav'][()]

    ic8 = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    ca_8 = sp.profile(nx=rp_final.size, ny=1, ns=4, nw=wc8.size)

    ca_8.wav[:] = wc8[:]

    ca_8.dat[0, 0, :, ic8, :] = np.transpose(f['profiles'][()][0, a_arr, b_arr, :, :][:, ic8], axes=(1, 0, 2))

    ca_8.weights[:, :] = 1.e16
    ca_8.weights[ic8, 0] = 0.004
    for line_indice, core_indice in zip(line_indices, core_indices):
        ca_8.weights[ic8[line_indice[0]+core_indice[0]:line_indice[0]+core_indice[1]], 0] = 0.002

    ca_8.weights[ic8[97+85:97+120], 0] /= 4

    ca_8.weights[ic8, 3] = ca_8.weights[ic8, 0]

    ca_8.weights[ic8, 3] /= 2

    # ca_8.weights[ic8[line_indices[2][0]+core_indices[2][0]:line_indices[2][0]+core_indices[2][1]], 3] /= 2

    ca_8.write(
        write_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_spot_total_{}.nc'.format(rp_final.size)
    )

    m = sp.model(nx=a_arr.size, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = ltau

    m.pgas[:, :, :] = 1.0

    m.temp[0, 0] = temp[labels[a_arr, b_arr]]

    m.vlos[0, 0] = vlos[labels[a_arr, b_arr]]

    m.vturb[0, 0] = vturb[labels[a_arr, b_arr]]

    m.Bln[0, 0] = blong[labels[a_arr, b_arr]]

    write_filename = write_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_spot_total_{}_initial_atmos.nc'.format(rp_final.size)

    m.write(str(write_filename))


def generate_actual_inversion_files_emission():

    line_indices = [[0, 58], [58, 97], [97, 306]]

    core_indices = [[19, 36], [18, 27], [85, 120]]

    # emission_profiles = [24]

    emission_profiles = [20, 21, 24]

    temp, vlos, vturb, blong = get_rp_atmos()

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/')
    actual_file = base_path / 'processed_inputs/alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc'
    write_path = base_path / 'fulldata_inversions'

    kmeans_file = base_path / 'chosen_out_30.h5'

    f = h5py.File(kmeans_file, 'r')

    labels = f['final_labels'][()]

    f.close()

    a_list = list()
    b_list = list()
    rp_final = list()

    for profile in emission_profiles:
        a, b = np.where(labels == profile)
        a_list += list(a)
        b_list += list(b)
        rp_final += list(np.ones(a.shape[0]) * profile)

    a_arr = np.array(a_list)
    b_arr = np.array(b_list)
    rp_final = np.array(rp_final)

    pixel_indices = np.zeros((3, a_arr.size), dtype=np.int64)

    pixel_indices[0] = a_arr
    pixel_indices[1] = b_arr
    pixel_indices[2] = rp_final

    fo = h5py.File(
        write_path / 'pixel_indices_emission_total_{}.h5'.format(
            rp_final.size
        ), 'w'
    )

    fo['pixel_indices'] = pixel_indices

    fo.close()

    f = h5py.File(actual_file, 'r')

    wc8 = f['wav'][()]

    ic8 = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    ca_8 = sp.profile(nx=rp_final.size, ny=1, ns=4, nw=wc8.size)

    ca_8.wav[:] = wc8[:]

    ca_8.dat[0, 0, :, ic8, :] = np.transpose(f['profiles'][()][0, a_arr, b_arr, :, :][:, ic8], axes=(1, 0, 2))

    ca_8.weights[:, :] = 1.e16
    ca_8.weights[ic8, 0] = 0.004
    for line_indice, core_indice in zip(line_indices, core_indices):
        ca_8.weights[ic8[line_indice[0]+core_indice[0]:line_indice[0]+core_indice[1]], 0] = 0.002

    # ca_8.weights[ic8[line_indices[2][0]+core_indices[2][0]:line_indices[2][0]+core_indices[2][1]], 0] /= 2

    ca_8.weights[ic8[97+85:97+120], 0] /= 2

    ca_8.weights[ic8, 3] = ca_8.weights[ic8, 0]

    ca_8.weights[ic8, 3] /= 2

    ca_8.weights[ic8[97+85:97+120], 3] /= 2

    # ca_8.weights[ic8[line_indices[2][0]+core_indices[2][0]:line_indices[2][0]+core_indices[2][1]], 3] /= 2

    ca_8.write(
        write_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_total_{}.nc'.format(rp_final.size)
    )

    m = sp.model(nx=a_arr.size, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = ltau

    m.pgas[:, :, :] = 1.0

    m.temp[0, 0] = temp[labels[a_arr, b_arr]]

    m.vlos[0, 0] = vlos[labels[a_arr, b_arr]]

    m.vturb[0, 0] = vturb[labels[a_arr, b_arr]]

    m.Bln[0, 0] = blong[labels[a_arr, b_arr]]

    write_filename = write_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_total_{}_initial_atmos.nc'.format(rp_final.size)

    m.write(str(write_filename))


def generate_actual_inversion_pixels(pixels):

    '''
    pixels a tuple of two 1D numpy arrays
    indicating the pixel location
    '''

    line_indices = [[0, 58], [58, 97], [97, 306]]

    core_indices = [[19, 36], [18, 27], [85, 120]]

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/')
    actual_file = base_path / 'processed_inputs/alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc'
    write_path = base_path / 'fulldata_inversions'

    f = h5py.File(actual_file, 'r')

    wc8 = f['wav'][()]

    ic8 = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    ca_8 = sp.profile(nx=pixels[0].size, ny=1, ns=4, nw=wc8.size)

    ca_8.wav[:] = wc8[:]

    ca_8.dat[0, 0, :, ic8, :] = np.transpose(f['profiles'][()][0, pixels[0], pixels[1], :, :][:, ic8], axes=(1, 0, 2))

    ca_8.weights[:, :] = 1.e16
    ca_8.weights[ic8, 0] = 0.004
    for line_indice, core_indice in zip(line_indices, core_indices):
        ca_8.weights[ic8[line_indice[0]+core_indice[0]:line_indice[0]+core_indice[1]], 0] = 0.002

    ca_8.weights[ic8[line_indices[2][0]+core_indices[2][0]:line_indices[2][0]+core_indices[2][1]], 0] /= 4

    # ca_8.weights[ic8[97+85:97+120], 0] /= 2

    ca_8.weights[ic8, 3] = ca_8.weights[ic8, 0]

    ca_8.weights[ic8, 3] /= 2

    ca_8.weights[ic8[97+85:97+120], 3] /= 2

    # ca_8.weights[ic8[line_indices[2][0]+core_indices[2][0]:line_indices[2][0]+core_indices[2][1]], 3] /= 2

    ca_8.write(
        write_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_x_{}_y_{}_total_{}.nc'.format(
            '_'.join([str(_p) for _p in pixels[0].tolist()]), '_'.join([str(_p) for _p in pixels[1].tolist()]), pixels[0].size
        )
    )


def merge_atmospheres():
    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/fulldata_inversions/')
    pixel_files = [
        base_path / 'pixel_indices_quiet_total_1027.h5',
        base_path / 'pixel_indices_spot_total_107.h5',
        base_path / 'pixel_indices_emission_total_6.h5'
    ]

    atmos_files = [
        base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_quiet_total_1027_cycle_1_t_5_vl_3_vt_4_blong_3_atmos.nc',
        base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_spot_total_107_cycle_1_t_5_vl_3_vt_4_blong_3_atmos.nc',
        base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_total_6_cycle_1_t_5_vl_3_vt_4_blong_3_atmos.nc'
    ]

    keys = [
        'temp',
        'vlos',
        'vturb',
        'blong',
        'nne',
        'z',
        'ltau500',
        'pgas',
        'rho',
    ]

    f = h5py.File(base_path / 'combined_output.nc', 'w')
    outs = dict()
    for key in keys:
        outs[key] = np.zeros((1, 19, 60, 150), dtype=np.float64)

    for pixel_file, atmos_file in zip(pixel_files, atmos_files):
        pf = h5py.File(pixel_file, 'r')
        af = h5py.File(atmos_file, 'r')
        for key in keys:
            a, b, rp = pf['pixel_indices'][0], pf['pixel_indices'][1], pf['pixel_indices'][2]
            outs[key][0, a, b] = af[key][0, 0]
        pf.close()
        af.close()

    for key in keys:
        f[key] = outs[key]
    f.close()


def merge_output_profiles():
    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/fulldata_inversions/')
    pixel_files = [
        base_path / 'pixel_indices_quiet_total_1027.h5',
        base_path / 'pixel_indices_spot_total_107.h5',
        base_path / 'pixel_indices_emission_total_6.h5'
    ]

    atmos_files = [
        base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_quiet_total_1027_cycle_1_t_5_vl_3_vt_4_blong_3_profs.nc',
        base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_spot_total_107_cycle_1_t_5_vl_3_vt_4_blong_3_profs.nc',
        base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_total_6_cycle_1_t_5_vl_3_vt_4_blong_3_profs.nc'
    ]

    keys = [
        'profiles'
    ]

    f = h5py.File(base_path / 'combined_output_profs.nc', 'w')
    outs = dict()
    for key in keys:
        outs[key] = np.zeros((1, 19, 60, 1236, 4), dtype=np.float64)

    for pixel_file, atmos_file in zip(pixel_files, atmos_files):
        pf = h5py.File(pixel_file, 'r')
        af = h5py.File(atmos_file, 'r')
        for key in keys:
            a, b, rp = pf['pixel_indices'][0], pf['pixel_indices'][1], pf['pixel_indices'][2]
            outs[key][0, a, b] = af[key][0, 0]
        pf.close()
        af.close()

    for key in keys:
        f[key] = outs[key]

    af = h5py.File(atmos_file, 'r')
    f['wav'] = af['wav'][()]
    af.close()

    f.close()


def generate_init_atmos_from_previous_result():
    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/')
    write_path = base_path / 'fulldata_inversions'

    pixel_file = write_path / 'pixel_indices_emission_total_33.h5'

    prev_output = h5py.File(write_path / 'combined_output.nc', 'r')

    fp = h5py.File(pixel_file, 'r')

    x, y = fp['pixel_indices'][0:2]

    m = sp.model(nx=x.size, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = ltau

    m.pgas[:, :, :] = 1.0

    m.temp[0, 0] = prev_output['temp'][()][0, x, y]

    m.vlos[0, 0] = prev_output['vlos'][()][0, x, y]

    m.vturb[0, 0] = prev_output['vturb'][()][0, x, y]

    m.Bln[0, 0] = prev_output['blong'][()][0, x, y]

    write_filename = write_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_total_{}_initial_atmos.nc'.format(
        x.size)

    m.write(write_filename)

    fp.close()

    prev_output.close()


def generate_file_for_rp_response_function():
    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/')
    write_path = base_path / 'fulldata_inversions'

    temp, vlos, vturb, blong = get_rp_atmos()

    m = sp.model(nx=30, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = ltau

    m.pgas[:, :, :] = 1.0

    m.temp[0, 0] = temp

    m.vlos[0, 0] = vlos

    m.vturb[0, 0] = vturb

    m.Bln[0, 0] = blong

    write_filename = write_path / 'rps_atmos_30'

    m.write(str(write_filename))


if __name__ == '__main__':
    # make_rps()
    # make_halpha_rps()
    # plot_rp_map_fov()
    # make_rps_plots()
    # make_halpha_rps_plots()
    # make_stic_inversion_files(rps=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
    # make_stic_inversion_files(rps=[3, 9, 10, 13, 20, 21])
    # make_stic_inversion_files(rps=[24])
    # make_stic_inversion_files(rps=[29])
    # make_stic_inversion_files(rps=[3, 20, 29])
    # make_stic_inversion_files_halpha_ca_both(rps=[0, 1, 2, 4, 5, 7, 9, 10, 11, 13, 14, 17, 18, 28, 29])
    # make_stic_inversion_files_halpha_ca_both(rps=[3, 6, 8, 12, 15, 16, 19, 20, 22, 23, 25])
    # make_stic_inversion_files_halpha_ca_both(rps=[21, 24])
    # make_stic_inversion_files_halpha_ca_both(rps=[26, 27])
    # generate_input_atmos_file(length=1, temp=[[-8, -6, -4, -2, 0, 2], [7000, 5000, 4000, 5000, 7000, 10000]], vlos=[[-8, -6, -4, -2, 0, 2], [0, 0, 0, 0, 0, 0]], blong=-100, name='quiet')
    # generate_input_atmos_file(length=30, temp=[[-8, -6, -4, -2, 0, 2], [9000, 7000, 5000, 6000, 8000, 10000]], vlos=[[-8, -6, -4, -2, 0, 2], [2e5, 2e5, 2e5, 2e5, 2e5, 2e5]], blong=0, name='interesting')
    # generate_input_atmos_file(length=1, temp=[[-8, -6, -4, -2, 0, 2], [11000, 7000, 5000, 6000, 8000, 10000]], vlos=[[-8, -6, -4, -2, 0, 2], [-10e5, -5e5, -3e3, 1e5, 0, 0]], blong=-450, name='red')
    # generate_input_atmos_file(length=3, temp=[[-8, -6, -4, -2, 0, 2], [11000, 7000, 5000, 6000, 8000, 10000]], vlos=[[-8, -6, -4, -2, 0, 2], [10e5, 5e5, 3e3, -1e5, 0, 0]], blong=-450, name='blue')
    # generate_input_atmos_file_from_previous_result(result_filename='/home/harsh/SpinorNagaraju/maps_1/stic/run_nagaraju/rps_stic_profiles_x_30_y_1_cycle_1_t_6_vl_3_vt_4_atmos.nc', rps=[3, 12, 25])
    make_rps_inversion_result_plots(nodes_temp=[-4.5, -3.8, -2.9, -1.8, -0.9, 0], nodes_vlos=[-5.5,-3.8, -2.5, -1], nodes_vturb=[-5, -4, -3, -1], nodes_blos=[-4.5, -1])
    # make_ha_ca_rps_inversion_result_plots()
    # combine_rps_atmos()
    # full_map_generate_input_atmos_file_from_previous_result()
    # generate_actual_inversion_files_quiet()
    # generate_actual_inversion_files_emission()
    # generate_init_atmos_from_previous_result()
    # generate_actual_inversion_files_spot()
    # merge_atmospheres()
    # merge_output_profiles()
    # generate_actual_inversion_pixels((np.array([12, 12]), np.array([49, 31])))
    # generate_actual_inversion_pixels((np.array([12]), np.array([40])))
    # generate_input_atmos_file(length=2, temp=[[-8, -6, -4, -2, 0, 2], [11000, 7000, 5000, 6000, 8000, 10000]], vlos=[[-8, -6, -4, -2, 0, 2], [-10e5, -5e5, -3e3, 1e5, 0, 0]], blong=-200, name='red')
    # generate_input_atmos_file(length=1, temp=[[-8, -6, -4, -2, 0, 2], [11000, 7000, 5000, 6000, 8000, 10000]], vlos=[[-8, -6, -4, -2, 0, 2], [20e5, -6e5, -3e5, -1e5, 0, 0]], blong=-200, name='blue')
    # make_pixels_inversion_result_plots()
    # generate_file_for_rp_response_function()
