import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/pyMilne')
sys.path.insert(1, '/home/harsh/CourseworkRepo/pyMilne/example_CRISP')
import numpy as np
import matplotlib.pyplot as plt
import MilneEddington as ME
import imtools as im
import time
from pathlib import Path
import h5py
import scipy.ndimage
from weak_field_approx import *


def findgrid(w, dw, extra=5):
    """
    Findgrid creates a regular wavelength grid
    with a step of dw that includes all points in
    input array w. It adds extra points at the edges
    for convolution purposes

    Returns the new array and the positions of the
    wavelengths points from w in the new array
    """
    nw = np.int32(np.rint(w / dw))
    nnw = nw[-1] - nw[0] + 1 + 2 * extra

    iw = np.arange(nnw, dtype='float64') * dw - extra * dw + w[0]

    idx = np.arange(w.size, dtype='int32')
    for ii in range(w.size):
        idx[ii] = np.argmin(np.abs(iw - w[ii]))

    return iw, idx


def do_me_inversion():
    dtype = 'float32'

    processed_inputs = Path('/home/harsh/SpinorNagaraju/maps_1/stic/processed_inputs/')

    profile_file = processed_inputs / 'aligned_Ca_Ha_stic_profiles.nc'

    center_wave = 6569.216
    del_lambda = 0.2
    f = h5py.File(profile_file, 'r')
    ind = np.where((f['wav'][()] >= (center_wave - del_lambda)) & (f['wav'][()] <= (center_wave + del_lambda)) & (
                f['profiles'][0, 0, 0, :, 0] != 0))[0]

    iw, idx = findgrid(f['wav'][ind], (f['wav'][ind][ind.size // 2 + 1] - f['wav'][ind][ind.size // 2]) * 0.25,
                       extra=8)  # Fe I 6569

    ny, nx = f['profiles'].shape[1], f['profiles'].shape[2]
    obs = np.zeros((ny, nx, 4, iw.size), dtype=dtype, order='c')

    obs[:, :, :, idx] = np.transpose(f['profiles'][:, :, :, ind, :], axes=(0, 1, 2, 4, 3))[0]

    f.close()
    #
    # Create sigma array with the estimate of the noise for
    # each Stokes parameter at all wavelengths. The extra
    # non-observed points will have a very large noise (1.e34)
    # (zero weight) compared to the observed ones (3.e-3)
    #
    sig = np.zeros((4, iw.size), dtype=dtype) + 1.e32
    sig[0, idx] = 3.e-3
    sig[3, idx] = 3.e-3

    #
    # Since the amplitudes of Stokes Q,U and V are very small
    # they have a low imprint in Chi2. We can artificially
    # give them more weight by lowering the noise estimate.
    #
    # sig[1:3, idx] /= 10
    sig[3, idx] /= 3.5

    if iw.size % 2 == 0:
        kernel_size = iw.size - 1
    else:
        kernel_size = iw.size - 2
    rev_kernel = np.zeros(kernel_size)
    rev_kernel[kernel_size // 2] = 1
    kernel = scipy.ndimage.gaussian_filter1d(rev_kernel, sigma=2 * 4 / 2.355)

    regions = [[iw, kernel]]

    lines = [6569]
    me = ME.MilneEddington(regions, lines, nthreads=8, precision=dtype)

    #
    # Init model parameters
    #
    labels = ['B [G]', 'inc [rad]', 'azi [rad]', 'Vlos [km/s]', 'vDop [Angstroms]', 'lineop', 'damp', 'S0', 'S1']
    iPar = np.float64([1500, 1, 0, -0.5, 0.035, 50., 0.1, 0.24, 0.7])
    Imodel = me.repeat_model(iPar, ny, nx)

    #
    # Run a first cycle with 4 inversions of each pixel (1 + 3 randomizations)
    #
    t0 = time.time()
    mo, syn, chi2 = me.invert_spatially_regularized(Imodel, obs, sig, nIter=100, chi2_thres=1e-3, mu=0.8, alpha=30.,
                                                    alphas=np.float32([1, 1, 1, 0.01, 0.1, 1.0, 0.1, 0.1, 0.1]),
                                                    method=1, delay_bracket=3)
    t1 = time.time()
    print("dT = {0}s -> <Chi2> (including regularization) = {1}".format(t1 - t0, chi2))

    f = h5py.File(processed_inputs / 'me_results_6569.nc', 'w')
    f['B_abs'] = mo[:, :, 0]
    f['inclination_rad'] = mo[:, :, 1]
    f['azi_rad'] = mo[:, :, 2]
    f['vlos_kms'] = mo[:, :, 3]
    f['vdoppler_angstrom'] = mo[:, :, 4]
    f['line_opacity'] = mo[:, :, 5]
    f['damping'] = mo[:, :, 6]
    f['S0'] = mo[:, :, 7]
    f['S1'] = mo[:, :, 8]
    f['syn'] = syn
    f.close()


def do_wfa():
    processed_inputs = Path('/home/harsh/SpinorNagaraju/maps_1/stic/processed_inputs/')

    profile_file = processed_inputs / 'aligned_Ca_Ha_stic_profiles.nc'
    center_wave = 8542.09
    del_lambda = 0.25

    f = h5py.File(profile_file, 'r')
    ind = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    actual_calculate_blos = prepare_calculate_blos(
        f['profiles'][:, :, :, ind, :],
        f['wav'][ind] / 10,
        center_wave / 10,
        (center_wave - del_lambda) / 10,
        (center_wave + del_lambda) / 10,
        1.1
    )

    vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)

    blos = np.fromfunction(vec_actual_calculate_blos, shape=(f['profiles'].shape[1], f['profiles'].shape[2]))

    f.close()
    # plt.imshow(blos, cmap='gray', origin='lower')
    #
    # plt.colorbar()
    #
    # plt.show()

    f = h5py.File(processed_inputs / 'wfa_8542.nc', 'w')
    f['del_lambda_nm'] = del_lambda / 10
    f['geff'] = 1.1
    f['center_wave_nm'] = center_wave / 10
    f['blos_gauss'] = blos
    f.close()


if __name__ == "__main__":
    do_wfa()
