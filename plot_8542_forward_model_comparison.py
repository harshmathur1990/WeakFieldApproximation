import os
import sys
sys.path.insert(2, '/home/harsh/CourseworkRepo/WFAComparison')
sys.path.insert(3, '/home/harsh/CourseworkRepo/rh/rhv2src/python')
import rh
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from stray_light_approximation import normalise_profiles, prepare_get_indice
from helita.sim import rh15d

catalog_base = Path('/home/harsh/CourseworkRepo/WFAComparison')
write_path_nicole = Path('/home/harsh/Spinor Inversions Nagaraju/forward_modelling/nicole')
write_path_rh15d = Path('/home/harsh/Spinor Inversions Nagaraju/forward_modelling/rh15d')

wave_8542 = np.array(
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

def plot_comparison_nicole(filename):

    f = h5py.File(filename, 'r')

    wave = f['wav'][()]

    profiles = f['profiles'][0, 0, 0, :, 0]

    f.close()

    catalog = np.loadtxt(catalog_base / 'catalog_8542.txt')

    norm_line, norm_atlas, _ = normalise_profiles(
        profiles,
        wave,
        catalog[:, 1],
        catalog[:, 0],
        wave[0]
    )

    plt.close('all')

    plt.clf()

    plt.cla()

    fig = plt.figure(figsize=(6, 4))

    gs = gridspec.GridSpec(1, 1)

    # gs.update(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)

    axs = fig.add_subplot(gs[0])

    axs.plot(wave, norm_line, label='Synthesized', color='brown')

    axs.plot(wave, norm_atlas, label='BASS 2000', color='orange')

    axs.set_xlabel(r'$\lambda (\AA)$')

    axs.set_ylabel(r'$I/I_{c}$')

    axs.set_title('Using NICOLE')

    plt.legend()

    plt.show()

    fig.tight_layout()

    fig.savefig(
        write_path_nicole / '{}.pdf'.format(
            filename
        ),
        dpi=300,
        format='pdf'
    )

    plt.close('all')

    plt.clf()

    plt.cla()


def plot_comparison_rh(out_dir, filename):

    out = rh15d.Rh15dout(fdir=out_dir)

    wave_in = out.ray.wavelength.data * 10

    wave_needed = np.array(range(1860)) * 8.4325e-3 + 8531.92775

    get_indice = prepare_get_indice(wave_in)

    vec_get_indice = np.vectorize(get_indice)

    wave_needed_indice = vec_get_indice(wave_needed)

    wave = wave_in[wave_needed_indice]

    profiles = out.ray.intensity.data[0, 0, wave_needed_indice]

    catalog = np.loadtxt(catalog_base / 'catalog_8542.txt')

    norm_line, norm_atlas, _ = normalise_profiles(
        profiles,
        wave,
        catalog[:, 1],
        catalog[:, 0],
        wave[0]
    )

    plt.close('all')

    plt.clf()

    plt.cla()

    fig = plt.figure(figsize=(6, 4))

    gs = gridspec.GridSpec(1, 1)

    # gs.update(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)

    axs = fig.add_subplot(gs[0])

    axs.plot(wave, norm_line, label='Synthesized', color='brown')

    axs.plot(wave, norm_atlas, label='BASS 2000', color='orange')

    axs.set_xlabel(r'$\lambda (\AA)$')

    axs.set_ylabel(r'$I/I_{c}$')

    axs.set_title('Using RH15D / STiC')

    plt.legend()

    fig.tight_layout()

    plt.show()

    fig.savefig(
        write_path_rh15d / '{}.pdf'.format(
            filename
        ),
        dpi=300,
        format='pdf'
    )

    plt.close('all')

    plt.clf()

    plt.cla()


def plot_rhf1d_cmparison_with_stic(outdir, stokes=False, stic_file=None):
    curdir = Path(os.getcwd())

    os.chdir(outdir)

    out = rh.readOutFiles(stokes=stokes)

    os.chdir(curdir)

    wave = np.array(out.spect.lambda0) * 10

    profiles = np.array(out.ray.I)

    get_indice = prepare_get_indice(wave)

    vec_get_indice = np.vectorize(get_indice)

    rh_indice = vec_get_indice(wave_8542)

    catalog = np.loadtxt(catalog_base / 'catalog_8542.txt')

    norm_line, norm_atlas, _ = normalise_profiles(
        profiles[rh_indice],
        wave[rh_indice],
        catalog[:, 1],
        catalog[:, 0],
        wave[rh_indice][0]
    )

    if stic_file:

        f_stic_file = h5py.File(stic_file, 'r')

        get_indice_1 = prepare_get_indice(f_stic_file['wav'][()])

        vec_get_indice_1 = np.vectorize(get_indice_1)

        rh_indice_1 = vec_get_indice_1(wave_8542)

        norm_line_stic, _, _ = normalise_profiles(
            f_stic_file['profiles'][0, 0, 0, rh_indice_1, 0],
            f_stic_file['wav'][rh_indice_1],
            catalog[:, 1],
            catalog[:, 0],
            f_stic_file['wav'][rh_indice_1][0]
        )

        f_stic_file.close()

    plt.close('all')

    plt.clf()

    plt.cla()

    if stokes:

        if not stic_file:
            return False

        f_stic_file = h5py.File(stic_file, 'r')

        fig, axs = plt.subplots(2, 2, figsize=(9, 6))

        Q = np.array(out.ray.Q)[rh_indice]

        U = np.array(out.ray.U)[rh_indice]

        V = np.array(out.ray.V)[rh_indice]

        axs[0][0].plot(wave[rh_indice], norm_line, label='RH', color='brown')

        axs[0][0].plot(wave[rh_indice], norm_atlas, label='BASS 2000', color='orange')

        axs[0][0].plot(f_stic_file['wav'][rh_indice_1], norm_line_stic, label='STiC', color='darkslategray')

        axs[0][0].set_xlabel(r'$\lambda (\AA)$')

        axs[0][0].set_ylabel(r'$I/I_{c}$')

        handles, labels = axs[0][0].get_legend_handles_labels()

        axs[0][0].legend(
            handles,
            labels,
            ncol=3,
            bbox_to_anchor=(0., 1.02, 2.3, .102),
            loc='lower left',
            mode="expand",
            borderaxespad=0.
        )

        axs[0][1].plot(wave[rh_indice], Q / profiles[rh_indice], label='RH', color='brown')

        axs[0][1].plot(f_stic_file['wav'][rh_indice_1], f_stic_file['profiles'][0, 0, 0, rh_indice_1, 1] / f_stic_file['profiles'][0, 0, 0, rh_indice_1, 0], label='STiC', color='darkslategray')

        axs[0][1].set_xlabel(r'$\lambda (\AA)$')

        axs[0][1].set_ylabel(r'$Q/I$')

        axs[1][0].plot(wave[rh_indice], U / profiles[rh_indice], label='RH', color='brown')

        axs[1][0].plot(f_stic_file['wav'][rh_indice_1], f_stic_file['profiles'][0, 0, 0, rh_indice_1, 2] / f_stic_file['profiles'][0, 0, 0, rh_indice_1, 0], label='STiC', color='darkslategray')

        axs[1][0].set_xlabel(r'$\lambda (\AA)$')

        axs[1][0].set_ylabel(r'$U/I$')

        axs[1][1].plot(wave[rh_indice], V / profiles[rh_indice], label='RH', color='brown')

        axs[1][1].plot(f_stic_file['wav'][rh_indice_1], f_stic_file['profiles'][0, 0, 0, rh_indice_1, 3] / f_stic_file['profiles'][0, 0, 0, rh_indice_1, 0], label='STiC', color='darkslategray')

        axs[1][1].set_xlabel(r'$\lambda (\AA)$')

        axs[1][1].set_ylabel(r'$V/I$')

        f_stic_file.close()

        fig.suptitle(r'$JK\;Coupling\;in\;SiI\;8536\;\AA\;line\;(JK\;coupling\;implemented\;in\;RH)$')

        fig.tight_layout()

        plt.show()

        fig.savefig(
            curdir / 'FALCvsBASS2000.pdf'.format(
            ),
            dpi=300,
            format='pdf'
        )

        fig.savefig(
            curdir / 'FALCvsBASS2000.png'.format(
            ),
            dpi=300,
            format='png'
        )

        plt.close('all')

        plt.clf()

        plt.cla()
    else:

        fig = plt.figure(figsize=(6, 4))

        gs = gridspec.GridSpec(1, 1)

    # gs.update(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)

        axs = fig.add_subplot(gs[0])

        axs.plot(wave[rh_indice], norm_line, label='Synthesized', color='brown')

        axs.plot(wave[rh_indice], norm_atlas, label='BASS 2000', color='orange')

        axs.set_xlabel(r'$\lambda (\AA)$')

        axs.set_ylabel(r'$I/I_{c}$')

        axs.set_title('Using RH')

        plt.legend()

        fig.tight_layout()

        plt.show()

        fig.savefig(
            curdir / 'FALCvsBASS2000.pdf'.format(
            ),
            dpi=300,
            format='pdf'
        )

        plt.close('all')

        plt.clf()

        plt.cla()


def plot_stic_with_without_jk_coupling(stic_ls_approx, stic_jk):

    f_stic_ls_approx = h5py.File(stic_ls_approx, 'r')
    f_stic_jk = h5py.File(stic_jk, 'r')

    plt.close('all')

    plt.clf()

    plt.cla()

    fig, axs = plt.subplots(2, 2, figsize=(19.2, 10.8))

    axs[0][0].plot(f_stic_ls_approx['wav'][()], f_stic_ls_approx['profiles'][0, 0, 0, :, 0], label='LS', color='blue', linewidth=0.5)

    axs[0][0].plot(f_stic_jk['wav'][()], f_stic_jk['profiles'][0, 0, 0, :, 0], label='JK', color='green', linewidth=0.5)

    axs[0][0].set_xlabel(r'$\lambda (\AA)$')

    axs[0][0].set_ylabel(r'$I/I_{c}$')

    axs[0][0].legend(loc="upper right")

    # handles, labels = axs[0][0].get_legend_handles_labels()

    # axs[0][0].legend(
    #     handles,
    #     labels,
    #     ncol=3,
    #     bbox_to_anchor=(0., 1.02, 2.3, .102),
    #     loc='lower left',
    #     mode="expand",
    #     borderaxespad=0.
    # )

    axs[0][1].plot(f_stic_ls_approx['wav'][()], f_stic_ls_approx['profiles'][0, 0, 0, :, 1] / f_stic_ls_approx['profiles'][0, 0, 0, :, 0], color='blue', linewidth=0.5)

    axs[0][1].plot(f_stic_jk['wav'][()], f_stic_jk['profiles'][0, 0, 0, :, 1] / f_stic_jk['profiles'][0, 0, 0, :, 0], color='green', linewidth=0.5)

    axs[0][1].set_xlabel(r'$\lambda (\AA)$')

    axs[0][1].set_ylabel(r'$Q/I$')

    axs[1][0].plot(f_stic_ls_approx['wav'][()], f_stic_ls_approx['profiles'][0, 0, 0, :, 2] / f_stic_ls_approx['profiles'][0, 0, 0, :, 0], color='blue', linewidth=0.5)

    axs[1][0].plot(f_stic_jk['wav'][()], f_stic_jk['profiles'][0, 0, 0, :, 2] / f_stic_jk['profiles'][0, 0, 0, :, 0], color='green', linewidth=0.5)

    axs[1][0].set_xlabel(r'$\lambda (\AA)$')

    axs[1][0].set_ylabel(r'$U/I$')

    axs[1][1].plot(f_stic_ls_approx['wav'][()], f_stic_ls_approx['profiles'][0, 0, 0, :, 3] / f_stic_ls_approx['profiles'][0, 0, 0, :, 0], color='blue', linewidth=0.5)

    axs[1][1].plot(f_stic_jk['wav'][()], f_stic_jk['profiles'][0, 0, 0, :, 3] / f_stic_jk['profiles'][0, 0, 0, :, 0], color='green', linewidth=0.5)

    axs[1][1].set_xlabel(r'$\lambda (\AA)$')

    axs[1][1].set_ylabel(r'$U/I$')

    fig.suptitle(r'$JK\;vs\;LS\;Coupling\;in\;SiI\;8536\;\AA\;line$')

    fig.tight_layout()

    fig.savefig(
        'JKvsLS.pdf'.format(
        ),
        dpi=300,
        format='pdf'
    )

    fig.savefig(
        'JKvsLS.png'.format(
        ),
        dpi=300,
        format='png'
    )

    plt.show()

    plt.close('all')

    plt.clf()

    plt.cla()

    f_stic_jk.close()
    f_stic_ls_approx.close()
