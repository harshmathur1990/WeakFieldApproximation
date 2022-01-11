import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/stic/example')
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.gridspec as gridspec
from prepare_data import *


kmeans_output_dir = Path(
    '/home/harsh/SpinorNagaraju/maps_1/stic/kmeans_output'
)

atmos_rp_write_path = Path(
    '/home/harsh/SpinorNagaraju/maps_1/stic/'
)

input_file = Path(
    '/home/harsh/SpinorNagaraju/maps_1/stic/processed_inputs/alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc'
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

wave_8542 = np.array(
    list(wave_8542_orig[95:153]) + list(wave_8542_orig[155:194]) + list(wave_8542_orig[196:405])
)

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
                            np.tile(wave_8542, a.shape[0]),
                            whole_data[a, :, r].flatten(),
                            bins=(wave_8542, in_bins_8542)
                        )

                        ax1.plot(
                            wave_8542,
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

                        ax1.set_xticks([8542.09])
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


def make_stic_inversion_files():

    # ind_photosphere = np.array(list(range(0, 58)) + list(range(58, 97)))

    # outer_core = np.array(list(range(97, 306)))

    # inner_core = np.array(list(range(285, 310)))

    f = h5py.File(kmeans_file, 'r')

    wc8, ic8 = findgrid(wave_8542, (wave_8542[10] - wave_8542[9])*0.25, extra=8)

    ca_8 = sp.profile(nx=30, ny=1, ns=4, nw=wc8.size)

    ca_8.wav[:] = wc8[:]

    ca_8.dat[0, 0, :, ic8, :] = np.transpose(
        f['rps'][()],
        axes=(1, 0, 2)
    )

    ca_8.weights[:,:] = 1.e16 # Very high value means weight zero
    ca_8.weights[ic8, 0] = 0.004
    # ca_8.weights[ic8, 3] = 0.004
    ca_8.weights[ic8[ind_photosphere], 0] /= 4.0
    # ca_8.weights[ic8[ind_photosphere], 3] /= 2.0
    ca_8.weights[ic8[outer_core], 0] /= 2.0
    # ca_8.weights[ic8[outer_core], 3] /= 2.0
    ca_8.weights[ic8[inner_core], 0] /= 2.0
    # ca_8.weights[ic8[inner_core], 3] /= 4.0
    # ca_8.weights[ic8, 3] /= 2.0
    

    ca_8.write(
        atmos_rp_write_path / 'rps_stic_profiles_x_30_y_1.nc'
    )

    lab = "region = {0:10.5f}, {1:8.5f}, {2:3d}, {3:e}, {4}"
    print(" ")
    print("Regions information for the input file:" )
    print(lab.format(ca_8.wav[0], ca_8.wav[1]-ca_8.wav[0], ca_8.wav.size, cont[0],  'none, none'))
    print("(w0, dw, nw, normalization, degradation_type, instrumental_profile file)")
    print(" ")


def generate_input_atmos_file():

    f = h5py.File(falc_file_path, 'r')

    m = sp.model(nx=30, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = f['ltau500'][0, 0, 0]

    m.pgas[:, :, :] = 1

    m.temp[:, :, :] = f['temp'][0, 0, 0]

    m.vlos[:, :, :] = f['vlos'][0, 0, 0]

    m.vturb[:, :, :] = 0

    m.Bln[:, :, :] = 100

    m.write(
        atmos_rp_write_path / 'falc_30_1_blong_100.nc'
    )


def make_rps_inversion_result_plots():

    rps_atmos_result = Path(
        '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/only_Stokes_I/rps_stic_profiles_x_30_y_1_cycle_1_t_6_vl_3_vt_4_atmos.nc'
    )

    rps_profs_result = Path(
        '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/only_Stokes_I/rps_stic_profiles_x_30_y_1_cycle_1_t_6_vl_3_vt_4_profs.nc'
    )

    rps_input_profs = Path(
        '/home/harsh/SpinorNagaraju/maps_1/stic/rps_stic_profiles_x_30_y_1.nc'
    )
    
    rps_plot_write_dir = Path(
        '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/only_Stokes_I'
    )

    finputprofs = h5py.File(rps_input_profs, 'r')

    fatmosresult = h5py.File(rps_atmos_result, 'r')

    fprofsresult = h5py.File(rps_profs_result, 'r')

    ind = np.where(finputprofs['profiles'][0, 0, 0, :, 0] != 0)[0]

    for i in range(30):
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

        fig.savefig(rps_plot_write_dir /'RPs_{}.pdf'.format(i), format='pdf', dpi=300)

        plt.close('all')

        plt.clf()

        plt.cla()
    fprofsresult.close()

    fatmosresult.close()

    finputprofs.close()


if __name__ == '__main__':
    make_rps()
    plot_rp_map_fov()
    make_rps_plots()
    make_stic_inversion_files()
    # make_rps_inversion_result_plots()
