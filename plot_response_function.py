import matplotlib.pyplot as plt
import numpy as np
import h5py
import matplotlib.gridspec as gridspec
from pathlib import Path


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


def make_response_functions():

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/full_stokes_blos_2/')

    filename = 'combined_rps_stic_profiles_x_30_y_1_cycle_1_t_0_vl_0_vt_0_blos_2_atmos_response.nc'

    f = h5py.File(base_path / filename)

    X, Y = np.meshgrid(f['wav'][()], ltau)

    for i in range(30):

        plt.close('all')

        plt.clf()

        plt.cla()

        fig = plt.figure(figsize=(7, 7))

        gs = gridspec.GridSpec(2, 2)

        for j in range(4):

            axs = fig.add_subplot(gs[j])

            if j == 3:
                k = 3
            else:
                k = 0

            im = axs.pcolormesh(
                X, Y,
                f['derivatives'][0, 0, i, j, :, :, k],
                shading='gouraud',
                cmap='gray'
            )

            fig.colorbar(im, ax=axs)

        fig.tight_layout()

        fig.savefig(
            base_path / 'RPs_{}_response.pdf'.format(
                i
            ),
            format='pdf',
            dpi=300
        )

        plt.close('all')

        plt.clf()

        plt.cla()

        fig = plt.figure(figsize=(7, 7))

        gs = gridspec.GridSpec(2, 2)

        for j in range(4):

            axs = fig.add_subplot(gs[j])

            if j == 3:
                k = 3
            else:
                k = 0

            axs.plot(
                ltau,
                np.abs(np.sum(f['derivatives'][0, 0, i, j, :, :, k], 1))
            )

        fig.tight_layout()

        fig.savefig(
            base_path / 'RPs_{}_wave_integrated_response.pdf'.format(
                i
            ),
            format='pdf',
            dpi=300
        )

        print(i)

    f.close()


if __name__ == '__main__':
    make_response_functions()
