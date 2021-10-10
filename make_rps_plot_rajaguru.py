import sys
sys.path.insert(1, '/home/harsh/stic/example')
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.gridspec as gridspec
from prepare_data import *
import sunpy.io.fits

kmeans_file = Path(
    '/data/harsh1/data_to_harsh/chosen_out_30.h5'
)

straylight_factor_6173 = 20
correction_factor_6173 = np.array(
    [
        1.        , 0.99994698, 0.99989396, 0.99984095, 0.99978793,
        0.99973491, 0.99968188, 0.99962886, 0.99957584, 0.99952282,
        0.99946979, 0.99941677, 0.99936374, 0.99931072, 0.99925769,
        0.99920466, 0.99915164, 0.99909861, 0.99904558, 0.99899255,
        0.99893952, 0.99888649, 0.99883346
    ]
)

straylight_factor_7090 = 3
correction_factor_7090 = np.array(
    [
        1.        , 0.9962398 , 0.99250681, 0.98880074, 0.9851213 ,
        0.98146821, 0.97784117, 0.97423992, 0.97066417, 0.96711367,
        0.96358814, 0.96008731, 0.95661094, 0.95315877, 0.94973053,
        0.946326  , 0.94294491, 0.93958704, 0.93625214, 0.93293997,
        0.92965031, 0.92638292, 0.92313759
    ]
)

wave_6173 = np.array(
    [
        6172.763 , 6173.087 , 6173.11  , 6173.133 , 6173.156 , 6173.179 ,
        6173.202 , 6173.225 , 6173.248 , 6173.271 , 6173.294 , 6173.317 ,
        6173.34  , 6173.363 , 6173.3857, 6173.4087, 6173.4316, 6173.4546,
        6173.4775, 6173.5005, 6173.5234, 6173.5464, 6173.5693
    ]
)

wave_7090 = np.array(
    [
        7089.7646, 7090.1104, 7090.1367, 7090.163 , 7090.1895, 7090.216 ,
        7090.242 , 7090.2686, 7090.295 , 7090.321 , 7090.347 , 7090.3735,
        7090.4   , 7090.4263, 7090.4526, 7090.479 , 7090.5054, 7090.5317,
        7090.5576, 7090.584 , 7090.6104, 7090.6367, 7090.663
    ]
)

falc_file_path = Path(
    '/home/harsh/stic/model_atmos/falc_nicole_for_stic.nc'
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
stic_cgs_calib_factor_6173 = 7842.25
stic_cgs_calib_factor_7090 = 8702.67


cw = np.asarray([6173., 7090.])
cont = []
for ii in cw:
    cont.append(getCont(ii))


def get_input_profiles():

    input_file_format = '/data/harsh1/data_to_harsh/fe00{}.fit'
        
    file_numbers = range(1, 10)

    profiles = None
    for index, file_number in enumerate(file_numbers):
        data, header = sunpy.io.fits.read(
            input_file_format.format(file_number),
            memmap=True
        )[0]

        data = np.transpose(data, axes=(2, 3, 1, 0))

        fr = data[150:350, 150:350, :, :]
        fr = fr.astype(np.float64)

        if profiles is None:
            profiles = np.zeros((9, 200, 200, fr.shape[2], fr.shape[3]), dtype=np.float64)

        profiles[index] = fr

    return profiles


def make_rps():

    profiles = get_input_profiles()

    f = h5py.File(kmeans_file, 'r+')

    keys = ['rps', 'final_labels']

    for key in keys:
        if key in list(f.keys()):
            del f[key]

    labels = f['labels_'][()]

    final_labels = np.zeros((9, 200, 200), dtype=np.int64)

    for i in range(0, 9):
        final_labels[i] = labels[i * 200 * 200: (i + 1) * 200 * 200].reshape(200, 200)

    f['final_labels'] = final_labels

    total_labels = labels.max() + 1

    rps = np.zeros(
        (total_labels, profiles.shape[3], profiles.shape[4]),
        dtype=np.float64
    )

    for i in range(total_labels):
        a, b, c = np.where(final_labels == i)
        rps[i] = np.mean(profiles[a, b, c], axis=0)

    f['rps'] = rps

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
        input_file_format = '/data/harsh1/data_to_harsh/fe00{}.fit'
        
        file_numbers = range(1, 10)

        for file_number in file_numbers:
            data, header = sunpy.io.fits.read(
                input_file_format.format(file_number),
                memmap=True
            )[0]

            data = np.transpose(data, axes=(2, 3, 1, 0))

            fr = data[150:350, 150:350, :, :]
            fr = fr.astype(np.float64)

            fr[:, :, :, 1:4] = fr[:, :, :, 1:4] / fr[:, :, :, 0][:, :, :, np.newaxis]

            fr[:, :, :, 0] = fr[:, :, :, 0] / stic_cgs_calib_factor_6173

            fr = fr.reshape(fr.shape[0] * fr.shape[1], fr.shape[2], fr.shape[3])

            if whole_data is None:
                whole_data = fr
            else:
                whole_data = np.vstack((whole_data, fr))

    f = h5py.File(kmeans_file, 'r')

    if get_labels:
        labels = f['labels_'][()]

    if get_rps:
        rps = f['rps'][()]

    rps[:, :, 1:4] = rps[:, :, 1:4] / rps[:, :, 0][:, :, np.newaxis]

    rps[:, :, 0] = rps[:, :, 0] / stic_cgs_calib_factor_6173

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

                        max_6173, min_6173  = c, f

                        min_6173 = min_6173 * 0.9
                        max_6173 = max_6173 * 1.1

                        in_bins_6173 = np.linspace(min_6173, max_6173, 1000)

                        H1, xedge1, yedge1 = np.histogram2d(
                            np.tile(wave_6173, a.shape[0]),
                            whole_data[a, :, r].flatten(),
                            bins=(wave_6173, in_bins_6173)
                        )

                        ax1.plot(
                            wave_6173,
                            center,
                            color=color,
                            linewidth=0.5,
                            linestyle='solid'
                        )

                        ymesh = H1.T

                        # ymeshmax = np.max(ymesh, axis=0)

                        ymeshnorm = ymesh / ymesh.max()

                        X1, Y1 = np.meshgrid(xedge1, yedge1)

                        ax1.pcolormesh(X1, Y1, ymeshnorm, cmap=cm)

                        ax1.set_ylim(min_6173, max_6173)

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

                        ax1.set_xticks([6173.334])
                        ax1.set_xticklabels([])

                        if r == 0:
                            y_ticks = [
                                np.round(
                                    min_6173 + (max_6173 - min_6173) * 0.1,
                                    2
                                ),
                                np.round(
                                    min_6173 + (max_6173 - min_6173) * 0.8,
                                    2
                                )
                            ]
                        else:
                            y_ticks = [
                                np.round(
                                    min_6173 + (max_6173 - min_6173) * 0.1,
                                    4
                                ),
                                np.round(
                                    min_6173 + (max_6173 - min_6173) * 0.8,
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

        plt.show()

        plt.close('all')

        plt.clf()

        plt.cla()


def get_data_for_label_polarisation_map(time_step):

    ind_photosphere = np.array(range(23))

    whole_data = get_input_profiles()

    intensity = whole_data[time_step, :, :, 0, 0]

    linpol_p = np.mean(
        np.sqrt(
            np.sum(
                np.square(
                    whole_data[time_step, :, :, ind_photosphere, 1:3]
                ),
                3
            )
        ) / whole_data[time_step, :, :, ind_photosphere, 0],
        0
    )

    circpol_p = np.mean(
        np.divide(
            np.abs(whole_data[time_step, :, :, ind_photosphere, 3]),
            whole_data[time_step, :, :, ind_photosphere, 0]
        ),
        0
    )

    f = h5py.File(kmeans_file, 'r')

    labels = f['final_labels'][time_step]

    f.close()

    return intensity, linpol_p, circpol_p, labels


def correct_for_straylight(data, straylight_factor, multiplicative_factor=None):
    
    #data must be of shape (t, x, y, lambda, stokes) with t, x, y optional
    # straylight_factor must be between 0-1

    result = data.copy()

    if multiplicative_factor is not None:
        if result.ndim == 5:
            result[:, :, :, :, 0] = result[:, :, :, :, 0] * multiplicative_factor
        elif result.ndim == 4:
            result[:, :, :, 0] = result[:, :, :, 0] * multiplicative_factor
        elif result.ndim == 3:
            result[:, :, 0] = result[:, :, 0] * multiplicative_factor
        elif result.ndim == 2:
            result[:, 0] = result[:, 0] * multiplicative_factor
        elif result.ndim == 1:
            result = result * multiplicative_factor

    if result.ndim == 5:
            result[:, :, :, :, 0] = (result[:, :, :, :, 0] - straylight_factor * result[:, :, :, 0, 0][:, :, :, np.newaxis]) / (1 - straylight_factor)
    elif result.ndim == 4:
        result[:, :, :, 0] = (result[:, :, :, 0] - straylight_factor * result[:, :, 0, 0][:, :, np.newaxis]) / (1 - straylight_factor)
    elif result.ndim == 3:
        result[:, :, 0] = (result[:, :, 0] - straylight_factor * result[:, 0, 0][:, np.newaxis]) / (1 - straylight_factor)
    elif result.ndim == 2:
        result[:, 0] = (result[:, 0] - straylight_factor * result[0, 0]) / (1 - straylight_factor)
    elif result.ndim == 1:
        result = (result - straylight_factor * result[0]) / (1 - straylight_factor)

    return result


def plot_rp_map_fov(time_step):

    intensity, linpol_p, circpol_p, labels = get_data_for_label_polarisation_map(time_step)

    plt.close('all')

    plt.clf()

    plt.cla()

    fig, axs = plt.subplots(2, 2, figsize=(6, 9))

    im00 = axs[0][0].imshow(intensity, cmap='gray', origin='lower')

    im01 = axs[0][1].imshow(labels, cmap='gray', origin='lower')

    im10 = axs[1][0].imshow(linpol_p, cmap='gray', origin='lower')

    im11 = axs[1][1].imshow(circpol_p, cmap='gray', origin='lower')

    fig.colorbar(im00, ax=axs[0][0], orientation='horizontal')

    fig.colorbar(im01, ax=axs[0][1], orientation='horizontal')

    fig.colorbar(im10, ax=axs[1][0], orientation='horizontal')

    fig.colorbar(im11, ax=axs[1][1], orientation='horizontal')

    fig.tight_layout()

    fig.savefig(
        rps_plot_write_dir / 'FoV_RPs_pol_map_{}.pdf'.format(time_step),
        format='pdf',
        dpi=300
    )

    plt.show()

    plt.close('all')

    plt.clf()

    plt.cla()


def make_actual_inversion_files(time_step):

    rps_plot_write_dir = Path(
        '/data/harsh1/data_to_harsh/RPsPlots/same_weight'
    )

    wfe1, ife1 = findgrid(wave_6173, (wave_6173[10] - wave_6173[9]) * 0.25, extra=8)

    wfe2, ife2 = findgrid(wave_7090, (wave_7090[10] - wave_7090[9]) * 0.25, extra=8)

    fe1 = sp.profile(nx=200, ny=200, ns=4, nw=wfe1.size)

    fe2 = sp.profile(nx=200, ny=200, ns=4, nw=wfe2.size)

    fe1.wav[:] = wfe1[:]

    fe2.wav[:] = wfe2[:]

    profiles = get_input_profiles()

    profiles = profiles[time_step]

    fe1.dat[0, :, :, ife1, :] = np.transpose(
        correct_for_straylight(
            profiles[:, :, :, 0:4],
            straylight_factor_6173  / 100,
            correction_factor_6173
        ) / stic_cgs_calib_factor_6173,
        axes=(2, 0, 1, 3)
    )

    fe2.dat[0, :, :, ife2, 0:1] = np.transpose(
        correct_for_straylight(
            profiles[:, :, :, 4][:, :, :, np.newaxis],
            straylight_factor_7090  / 100,
            correction_factor_7090
        ) / stic_cgs_calib_factor_7090,
        axes=(2, 0, 1, 3)
    )

    fe1.weights[:, :] = 1e16

    fe1.weights[ife1, :] = 0.004

    fe2.weights[:, :] = 1e16

    fe2.weights[ife2, :] = 0.004

    fe = fe1 + fe2

    fe.write(
        rps_plot_write_dir / 'map_{}_stic_profiles.nc'.format(time_step)
    )

    lab = "region = {0:10.5f}, {1:8.5f}, {2:3d}, {3:e}, {4}"
    print(" ")
    print("Regions information for the input file:" )
    print(lab.format(fe1.wav[0], fe1.wav[1]-fe1.wav[0], fe1.wav.size, cont[0],  'none, none'))
    print(lab.format(fe2.wav[0], fe2.wav[1]-fe2.wav[0], fe2.wav.size, cont[1],  'none, none'))
    print("(w0, dw, nw, normalization, degradation_type, instrumental_profile file)")
    print(" ")
    
    generate_input_atmos_file_for_map(time_step)


def prepare_get_parameter(param):

    def get_parameter(rp):
        return param[rp]

    return get_parameter


def generate_input_atmos_file_for_map(time_step):

    rps_plot_write_dir = Path(
        '/data/harsh1/data_to_harsh/RPsPlots/same_weight'
    )

    out_atmos_all = rps_plot_write_dir / 'rps_stic_profiles_cycle_1_t_4_vl_4_vt_0_blos_4_bhor_4_azi_4_atmos.nc'

    out_atmos_16 = rps_plot_write_dir / 'rps_16_stic_profiles_cycle_1_t_4_vl_4_vt_0_blos_4_bhor_4_azi_4_atmos.nc'

    f_all = h5py.File(out_atmos_all, 'r')

    f_16 = h5py.File(out_atmos_16, 'r')

    temp = f_all['temp'][0, 0, :]

    vlos = f_all['vlos'][0, 0, :]

    vturb = f_all['vturb'][0, 0, :]

    blong = f_all['blong'][0, 0, :]

    bhor = f_all['bhor'][0, 0, :]

    azi = f_all['azi'][0, 0, :]

    temp[16] = f_16['temp'][0, 0, 0]

    vlos[16] = f_16['vlos'][0, 0, 0]

    vturb[16] = f_16['vturb'][0, 0, 0]

    blong[16] = f_16['blong'][0, 0, 0]

    bhor[16] = f_16['bhor'][0, 0, 0]

    azi[16] = f_16['azi'][0, 0, 0]

    get_temp = prepare_get_parameter(temp)

    get_vlos = prepare_get_parameter(vlos)

    get_vturb = prepare_get_parameter(vturb)

    get_blong = prepare_get_parameter(blong)

    get_bhor = prepare_get_parameter(bhor)

    get_azi = prepare_get_parameter(azi)

    f_16.close()

    f_all.close()

    f = h5py.File(kmeans_file, 'r')

    m = sp.model(nx=200, ny=200, nt=1, ndep=150)

    m.ltau[:, :, :] = ltau

    m.pgas[:, :, :] = 1

    m.temp[0, :, :] = get_temp(f['final_labels'][time_step])

    m.vlos[0, :, :] = get_vlos(f['final_labels'][time_step])

    m.vturb[0, :, :] = get_vturb(f['final_labels'][time_step])

    m.Bln[0, :, :] = get_blong(f['final_labels'][time_step])

    m.Bho[0, :, :] = get_bhor(f['final_labels'][time_step])

    m.azi[0, :, :] = get_azi(f['final_labels'][time_step])

    m.write(
        rps_plot_write_dir / 'time_step_{}_initial_atmos.nc'.format(
            time_step
        )
    )


def make_stic_inversion_files():

    wfe1, ife1 = findgrid(wave_6173, (wave_6173[10] - wave_6173[9]) * 0.25, extra=8)

    wfe2, ife2 = findgrid(wave_7090, (wave_7090[10] - wave_7090[9]) * 0.25, extra=8)

    fe1 = sp.profile(nx=30, ny=1, ns=4, nw=wfe1.size)

    fe2 = sp.profile(nx=30, ny=1, ns=4, nw=wfe2.size)

    fe1.wav[:] = wfe1[:]

    fe2.wav[:] = wfe2[:]

    f = h5py.File(kmeans_file, 'r')

    fe1.dat[0, 0, :, ife1, :] = np.transpose(
        correct_for_straylight(
            f['rps'][:, :, 0:4],
            straylight_factor_6173  / 100,
            correction_factor_6173
        ) / stic_cgs_calib_factor_6173,
        axes=(1, 0, 2)
    )

    fe2.dat[0, 0, :, ife2, 0:1] = np.transpose(
        correct_for_straylight(
            f['rps'][:, :, 4][:, :, np.newaxis],
            straylight_factor_7090  / 100,
            correction_factor_7090
        ) / stic_cgs_calib_factor_7090,
        axes=(1, 0, 2)
    )

    f.close()

    fe1.weights[:, :] = 1e16

    fe1.weights[ife1, :] = 0.004

    fe2.weights[:, :] = 1e16

    fe2.weights[ife2, :] = 0.004

    fe1.weights[ife1,1:3] /= 4.5 # Some more weight for Q&U
    fe1.weights[ife1,3] /= 3.5    # Some more weight for V

    fe2.weights[ife2,1:3] /= 4.5 # Some more weight for Q&U
    fe2.weights[ife2,3] /= 3.5    # Some more weight for V

    fe = fe1 + fe2

    fe.write(
        rps_plot_write_dir / 'rps_stic_profiles_more_weights.nc'
    )

    lab = "region = {0:10.5f}, {1:8.5f}, {2:3d}, {3:e}, {4}"
    print(" ")
    print("Regions information for the input file:" )
    print(lab.format(fe1.wav[0], fe1.wav[1]-fe1.wav[0], fe1.wav.size, cont[0],  'none, none'))
    print(lab.format(fe2.wav[0], fe2.wav[1]-fe2.wav[0], fe2.wav.size, cont[1],  'none, none'))
    print("(w0, dw, nw, normalization, degradation_type, instrumental_profile file)")
    print(" ")

    """
    Regions information for the input file:
    region = 6172.74000,  0.00575, 148, 4.014861e-05, none, none
    region = 7089.73860,  0.00650, 146, 4.144302e-05, none, none
    (w0, dw, nw, normalization, degradation_type, instrumental_profile file)
    """


def make_sel_rps_stic_files(rps_list):

    rps_list = np.array(rps_list)

    wfe1, ife1 = findgrid(wave_6173, (wave_6173[10] - wave_6173[9]) * 0.25, extra=8)

    wfe2, ife2 = findgrid(wave_7090, (wave_7090[10] - wave_7090[9]) * 0.25, extra=8)

    fe1 = sp.profile(nx=rps_list.size, ny=1, ns=4, nw=wfe1.size)

    fe2 = sp.profile(nx=rps_list.size, ny=1, ns=4, nw=wfe2.size)

    fe1.wav[:] = wfe1[:]

    fe2.wav[:] = wfe2[:]

    f = h5py.File(kmeans_file, 'r')

    fe1.dat[0, 0, :, ife1, :] = np.transpose(
        correct_for_straylight(
            f['rps'][rps_list, :, 0:4],
            straylight_factor_6173  / 100,
            correction_factor_6173
        ) / stic_cgs_calib_factor_6173,
        axes=(1, 0, 2)
    )

    fe2.dat[0, 0, :, ife2, 0:1] = np.transpose(
        correct_for_straylight(
            f['rps'][rps_list, :, 4][:, :, np.newaxis],
            straylight_factor_7090  / 100,
            correction_factor_7090
        ) / stic_cgs_calib_factor_7090,
        axes=(1, 0, 2)
    )

    f.close()

    fe1.weights[:, :] = 1e16

    fe1.weights[ife1, :] = 0.004

    fe2.weights[:, :] = 1e16

    fe2.weights[ife2, :] = 0.004

    fe = fe1 + fe2

    fe.write(
        rps_plot_write_dir / 'rps_{}_stic_profiles.nc'.format(
            '_'.join([str(arp) for arp in list(rps_list)])
        )
    )

    lab = "region = {0:10.5f}, {1:8.5f}, {2:3d}, {3:e}, {4}"
    print(" ")
    print("Regions information for the input file:" )
    print(lab.format(fe1.wav[0], fe1.wav[1]-fe1.wav[0], fe1.wav.size, cont[0],  'none, none'))
    print(lab.format(fe2.wav[0], fe2.wav[1]-fe2.wav[0], fe2.wav.size, cont[1],  'none, none'))
    print("(w0, dw, nw, normalization, degradation_type, instrumental_profile file)")
    print(" ")

    generate_input_atmos_file(rps_list.size)


def generate_input_atmos_file(num=30):

    f = h5py.File(falc_file_path, 'r')

    m = sp.model(nx=num, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = f['ltau500'][0, 0, 0]

    m.pgas[:, :, :] = 1

    m.temp[:, :, :] = f['temp'][0, 0, 0]

    m.vlos[:, :, :] = f['vlos'][0, 0, 0]

    m.vturb[:, :, :] = 0

    m.Bln[:, :, :] = 2000

    m.Bho[:, :, :] = 1000

    m.azi[:, :, :] = 45. * 3.14159 / 180.

    m.write(
        rps_plot_write_dir / 'falc_{}_1_blong_100_bhor_100_azi_45.nc'.format(
            num
        )
    )


def make_rps_inversion_result_plots():

    rps_plot_write_dir = Path(
        '/data/harsh1/data_to_harsh/RPsPlots/more_weight'
    )

    rps_input_profs = rps_plot_write_dir / 'rps_stic_profiles_more_weights.nc'

    rps_atmos_result = rps_plot_write_dir / 'rps_stic_profiles_more_weights_cycle_1_t_4_vl_4_vt_0_blos_4_bhor_4_azi_4_atmos.nc'

    rps_profs_result = rps_plot_write_dir / 'rps_stic_profiles_more_weights_cycle_1_t_4_vl_4_vt_0_blos_4_bhor_4_azi_4_profs.nc'
    
    finputprofs = h5py.File(rps_input_profs, 'r')

    fatmosresult = h5py.File(rps_atmos_result, 'r')

    fprofsresult = h5py.File(rps_profs_result, 'r')

    ind_all = np.where(finputprofs['profiles'][0, 0, 0, :, 0] != 0)[0]

    ind_6173 = ind_all[np.where(ind_all < 148)[0]]

    ind_7090 = ind_all[np.where(ind_all >= 148)[0]]

    for i in range(30):
        plt.close('all')

        plt.clf()

        plt.cla()

        fig, axs = plt.subplots(5, 2, figsize=(6, 9))

        axs[0][0].plot(finputprofs['wav'][ind_6173] - 6173.34, finputprofs['profiles'][0, 0, i, ind_6173, 0], color='orange')

        axs[0][0].plot(finputprofs['wav'][ind_7090] - 7090.4, finputprofs['profiles'][0, 0, i, ind_7090, 0], color='orange')

        axs[0][0].plot(fprofsresult['wav'][ind_6173] - 6173.34, fprofsresult['profiles'][0, 0, i, ind_6173, 0], color='brown')

        axs[0][0].plot(fprofsresult['wav'][ind_7090] - 7090.4, fprofsresult['profiles'][0, 0, i, ind_7090, 0], color='brown')

        axs[0][1].plot(
            finputprofs['wav'][ind_6173] - 6173.34,
            finputprofs['profiles'][0, 0, i, ind_6173, 1] / finputprofs['profiles'][0, 0, i, ind_6173, 0],
            color='orange'
        )

        axs[0][1].plot(
            fprofsresult['wav'][ind_6173] - 6173.34,
            fprofsresult['profiles'][0, 0, i, ind_6173, 1] / fprofsresult['profiles'][0, 0, i, ind_6173, 0],
            color='brown'
        )

        axs[1][0].plot(
            finputprofs['wav'][ind_6173] - 6173.34,
            finputprofs['profiles'][0, 0, i, ind_6173, 2] / finputprofs['profiles'][0, 0, i, ind_6173, 0],
            color='orange'
        ) 

        axs[1][0].plot(
            fprofsresult['wav'][ind_6173] - 6173.34,
            fprofsresult['profiles'][0, 0, i, ind_6173, 2] / fprofsresult['profiles'][0, 0, i, ind_6173, 0],
            color='brown'
        )


        axs[1][1].plot(
            finputprofs['wav'][ind_6173] - 6173.34,
            finputprofs['profiles'][0, 0, i, ind_6173, 3] / finputprofs['profiles'][0, 0, i, ind_6173, 0],
            color='orange'
        )

        axs[1][1].plot(
            fprofsresult['wav'][ind_6173] - 6173.34,
            fprofsresult['profiles'][0, 0, i, ind_6173, 3] / fprofsresult['profiles'][0, 0, i, ind_6173, 0],
            color='brown'
        )


        axs[2][0].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['temp'][0, 0, i] / 1e3, color='brown')

        axs[2][1].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['vlos'][0, 0, i] / 1e5, color='brown')

        axs[3][0].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['vturb'][0, 0, i] / 1e5, color='brown')

        axs[3][1].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['blong'][0, 0, i], color='brown')

        axs[4][0].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['bhor'][0, 0, i], color='brown')

        axs[4][1].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['azi'][0, 0, i] * 180 / 3.14159, color='brown')

        axs[0][0].set_xlabel(r'$\lambda(\AA)$')
        axs[0][0].set_ylabel(r'$I/I_{c}$')

        axs[0][1].set_xlabel(r'$\lambda(\AA)$')
        axs[0][1].set_ylabel(r'$Q/I$')

        axs[1][0].set_xlabel(r'$\lambda(\AA)$')
        axs[1][0].set_ylabel(r'$U/I$')

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

        axs[4][0].set_xlabel(r'$\log(\tau_{500})$')
        axs[4][0].set_ylabel(r'$B_{hor}[G]$')

        axs[4][1].set_xlabel(r'$\log(\tau_{500})$')
        axs[4][1].set_ylabel(r'$Azi[rad]$')

        fig.tight_layout()

        fig.savefig(rps_plot_write_dir / 'RPs_{}.pdf'.format(i), format='pdf', dpi=300)

        plt.close('all')

        plt.clf()

        plt.cla()
    fprofsresult.close()

    fatmosresult.close()

    finputprofs.close()


def make_rps_inversion_result_plots_sel_rps(rps_list):

    rps_list = np.array(rps_list)

    rps_input_profs = rps_plot_write_dir / 'rps_{}_stic_profiles.nc'.format(
        '_'.join([str(arp) for arp in list(rps_list)])
    )

    rps_atmos_result = rps_plot_write_dir / 'rps_{}_stic_profiles_cycle_1_t_4_vl_4_vt_0_blos_4_bhor_4_azi_4_atmos.nc'.format(
        '_'.join([str(arp) for arp in list(rps_list)])
    )

    rps_profs_result = rps_plot_write_dir / 'rps_{}_stic_profiles_cycle_1_t_4_vl_4_vt_0_blos_4_bhor_4_azi_4_profs.nc'.format(
        '_'.join([str(arp) for arp in list(rps_list)])
    )
    
    finputprofs = h5py.File(rps_input_profs, 'r')

    fatmosresult = h5py.File(rps_atmos_result, 'r')

    fprofsresult = h5py.File(rps_profs_result, 'r')

    ind_all = np.where(finputprofs['profiles'][0, 0, 0, :, 0] != 0)[0]

    ind_6173 = ind_all[np.where(ind_all < 148)[0]]

    ind_7090 = ind_all[np.where(ind_all >= 148)[0]]

    for i, rp in enumerate(list(rps_list)):
        plt.close('all')

        plt.clf()

        plt.cla()

        fig, axs = plt.subplots(5, 2, figsize=(6, 9))

        axs[0][0].plot(finputprofs['wav'][ind_6173] - 6173.34, finputprofs['profiles'][0, 0, i, ind_6173, 0], color='orange')

        axs[0][0].plot(finputprofs['wav'][ind_7090] - 7090.4, finputprofs['profiles'][0, 0, i, ind_7090, 0], color='orange')

        axs[0][0].plot(fprofsresult['wav'][ind_6173] - 6173.34, fprofsresult['profiles'][0, 0, i, ind_6173, 0], color='brown')

        axs[0][0].plot(fprofsresult['wav'][ind_7090] - 7090.4, fprofsresult['profiles'][0, 0, i, ind_7090, 0], color='brown')

        axs[0][1].plot(
            finputprofs['wav'][ind_6173] - 6173.34,
            finputprofs['profiles'][0, 0, i, ind_6173, 1] / finputprofs['profiles'][0, 0, i, ind_6173, 0],
            color='orange'
        )

        axs[0][1].plot(
            fprofsresult['wav'][ind_6173] - 6173.34,
            fprofsresult['profiles'][0, 0, i, ind_6173, 1] / fprofsresult['profiles'][0, 0, i, ind_6173, 0],
            color='brown'
        )

        axs[1][0].plot(
            finputprofs['wav'][ind_6173] - 6173.34,
            finputprofs['profiles'][0, 0, i, ind_6173, 2] / finputprofs['profiles'][0, 0, i, ind_6173, 0],
            color='orange'
        ) 

        axs[1][0].plot(
            fprofsresult['wav'][ind_6173] - 6173.34,
            fprofsresult['profiles'][0, 0, i, ind_6173, 2] / fprofsresult['profiles'][0, 0, i, ind_6173, 0],
            color='brown'
        )


        axs[1][1].plot(
            finputprofs['wav'][ind_6173] - 6173.34,
            finputprofs['profiles'][0, 0, i, ind_6173, 3] / finputprofs['profiles'][0, 0, i, ind_6173, 0],
            color='orange'
        )

        axs[1][1].plot(
            fprofsresult['wav'][ind_6173] - 6173.34,
            fprofsresult['profiles'][0, 0, i, ind_6173, 3] / fprofsresult['profiles'][0, 0, i, ind_6173, 0],
            color='brown'
        )


        axs[2][0].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['temp'][0, 0, i] / 1e3, color='brown')

        axs[2][1].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['vlos'][0, 0, i] / 1e5, color='brown')

        axs[3][0].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['vturb'][0, 0, i] / 1e5, color='brown')

        axs[3][1].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['blong'][0, 0, i], color='brown')

        axs[4][0].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['bhor'][0, 0, i], color='brown')

        axs[4][1].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['azi'][0, 0, i] * 180 / 3.14159, color='brown')

        axs[0][0].set_xlabel(r'$\lambda(\AA)$')
        axs[0][0].set_ylabel(r'$I/I_{c}$')

        axs[0][1].set_xlabel(r'$\lambda(\AA)$')
        axs[0][1].set_ylabel(r'$Q/I$')

        axs[1][0].set_xlabel(r'$\lambda(\AA)$')
        axs[1][0].set_ylabel(r'$U/I$')

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

        axs[4][0].set_xlabel(r'$\log(\tau_{500})$')
        axs[4][0].set_ylabel(r'$B_{hor}[G]$')

        axs[4][1].set_xlabel(r'$\log(\tau_{500})$')
        axs[4][1].set_ylabel(r'$Azi[rad]$')

        fig.tight_layout()

        fig.savefig(rps_plot_write_dir / 'RPs_{}.pdf'.format(rp), format='pdf', dpi=300)

        plt.close('all')

        plt.clf()

        plt.cla()
    fprofsresult.close()

    fatmosresult.close()

    finputprofs.close()
