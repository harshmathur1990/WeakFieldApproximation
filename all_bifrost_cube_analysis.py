import sys

import numpy as np
import h5py
import matplotlib.pyplot as plt
from weak_field_approx import prepare_calculate_blos_rh15d, prepare_calculate_blos
from pathlib import Path
from tqdm import tqdm
from lightweaver.utils import vac_to_air
from scipy.interpolate import CubicSpline


base_path = Path('/home/harsh/BifrostRun_fast_Access')

suppl_output_file_3000 = base_path / 'BIFROST_en024048_hion_snap_385_0_504_0_504_-500000.0_3000000.0_supplementary_outputs.nc'

suppl_output_file_full = base_path / 'MULTI3D_BIFROST_en024048_hion_snap_385_0_504_0_504_-1020996.0_15000000.0_supplementary_outputs.nc'

atmos_file_3000 = base_path / 'BIFROST_en024048_hion_snap_385_0_504_0_504_-500000.0_3000000.0.nc'

atmos_file_full = base_path / 'BIFROST_en024048_hion_snap_385_0_504_0_504_-1020996.0_15000000.0.nc'

multi_3d_out_file = base_path / 'Multi-3D-H_6_level_populations' / 'output_ray.hdf5'


def prepare_get_value_at_ltau(fs, fa, var_name, ltau):

    def get_val_at_ltau(i, j):
        i = int(i)
        j = int(j)
        ltau500 = fs['ltau500'][0, i, j]
        return np.interp(ltau, ltau500, fa[var_name][0, i, j])

    return get_val_at_ltau



def get_var_at_ltau(var_name='B_z'):

    grid = np.arange(-16, 2, 0.1)

    fs = h5py.File(suppl_output_file_full, 'r')

    keys = list(fs.keys())

    fs.close()

    if '{}_ltau_strata'.format(var_name) not in keys:

        fa = h5py.File(atmos_file_full, 'r')

        fs = h5py.File(suppl_output_file_full, 'r')

        var_ltau_strata = np.zeros((1, 504, 504, grid.size), dtype=np.float64)

        t = tqdm(total=grid.size)

        for index, grid_val in enumerate(grid):
            get_val_at_ltau = prepare_get_value_at_ltau(fs, fa, var_name, grid_val)

            vec_get_val_at_ltau = np.vectorize(get_val_at_ltau)

            val = np.fromfunction(vec_get_val_at_ltau, shape=(504, 504))

            var_ltau_strata[0, :, :, index] = val

            t.update(1)

        fs.close()

        fs = h5py.File(suppl_output_file_full, 'r+')

        fs['{}_ltau_strata'.format(var_name)] = var_ltau_strata

        fs.close()

        fa.close()

    else:

        fs = h5py.File(suppl_output_file_full, 'r')

        var_ltau_strata = fs['{}_ltau_strata'.format(var_name)][()]

        fs.close()

    return var_ltau_strata


def prepare_rotate_blong(blong):
    def rotate_blong(i, j):
        i = int(i)
        j = int(j)

        x = j
        y = blong.shape[0] - i - 1
        return blong[x, y]

    return rotate_blong


def get_blong_wfa_ha_rh15d():

    fm = h5py.File(multi_3d_out_file, 'r')

    wave = vac_to_air(fm['wavelength'][()])

    actual_calculate_blos = prepare_calculate_blos_rh15d(
        fm,
        wavelength_arr=wave,
        lambda0=656.28,
        lambda_range_min=(6562.8 - 0.35) / 10,
        lambda_range_max=(6562.8 + 0.35) / 10,
        g_eff=1.048,
        transition_skip_list=None
    )

    vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)

    blong = np.fromfunction(vec_actual_calculate_blos, shape=(504, 504))

    rotate_blong = prepare_rotate_blong(blong)

    vec_rotate_blong = np.vectorize(rotate_blong)

    rotated_blong = np.fromfunction(vec_rotate_blong, shape=blong.shape)

    return rotated_blong


def get_blong_wfa_ha_rh(deltal=0.35):

    fm = h5py.File(suppl_output_file_full, 'r')

    wave = fm['wave_H'][()] / 10

    actual_calculate_blos = prepare_calculate_blos(
        fm['profiles_H'],
        wavelength_arr=wave,
        lambda0=656.28,
        lambda_range_min=(6562.8 - deltal) / 10,
        lambda_range_max=(6562.8 + deltal) / 10,
        g_eff=1.048,
        transition_skip_list=None
    )

    vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)

    blong = np.fromfunction(vec_actual_calculate_blos, shape=(504, 504))

    return blong


def get_blong_wfa_ca():

    fm = h5py.File(suppl_output_file_3000, 'r')

    wave = fm['wave_CaIR'][()]

    actual_calculate_blos = prepare_calculate_blos(
        fm['profiles_CaIR'],
        wavelength_arr=wave/10,
        lambda0=854.209,
        lambda_range_min=(8542.09 - 0.4) / 10,
        lambda_range_max=(8542.09 + 0.4) / 10,
        g_eff=1.1,
        transition_skip_list=None
    )

    vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)

    blong = np.fromfunction(vec_actual_calculate_blos, shape=(504, 504))

    return blong


def compare_blong_ca_ha():
    blong_ha = get_blong_wfa_ha_rh()

    blong_ca = get_blong_wfa_ca()

    b_z_ltau_strata = get_blong_at_ltau()

    # ha_indice = np.argmin(
    #     np.sum(
    #         np.abs(
    #             b_z_ltau_strata * 1e4 - blong_ha[np.newaxis, :, :, np.newaxis]
    #         ),
    #         axis=(0, 1, 2)
    #     )
    # )
    #
    # ca_indice = np.argmin(
    #     np.sum(
    #         np.abs(
    #             b_z_ltau_strata * 1e4 - blong_ca[np.newaxis, :, :, np.newaxis]
    #         ),
    #         axis=(0, 1, 2)
    #     )
    # )

    grid = np.arange(-16, 2, 0.1)

    ha_indice = np.argmin(np.abs(grid + 11))

    ca_indice = np.argmin(np.abs(grid + 10))

    fig, axs = plt.subplots(2, 2, figsize=(7, 3.5))

    im00 = axs[0][0].imshow(blong_ha, cmap='gray', origin='lower')

    im01 = axs[0][1].imshow(blong_ca, cmap='gray', origin='lower')

    im10 = axs[1][0].imshow(b_z_ltau_strata[0, :, :, ha_indice] * 1e4, cmap='gray', origin='lower')

    im11 = axs[1][1].imshow(b_z_ltau_strata[0, :, :, ca_indice] * 1e4, cmap='gray', origin='lower')

    axs[0][0].set_title(r'WFA H$\alpha\pm$0.35$\mathrm{\AA}$')

    axs[0][1].set_title(r'WFA Ca II 8542 $\pm$0.4$\mathrm{\AA}$')

    axs[1][0].set_title(r'$B_{\mathrm{LOS}}$ at $\log\tau_{500}$ = -11')

    axs[1][1].set_title(r'$B_{\mathrm{LOS}}$ at $\log\tau_{500}$ = -10')

    fig.colorbar(im00, ax=axs[0][0])

    fig.colorbar(im01, ax=axs[0][1])

    fig.colorbar(im10, ax=axs[1][0])

    fig.colorbar(im11, ax=axs[1][1])

    # fig.tight_layout()

    plt.show()


def compare_porta_1d_with_multi3d():

    base_path = Path('/run/media/harsh/5de85c60-8e85-4cc8-89e6-c74a83454760/BifrostRun/')
    f = h5py.File(base_path / 's_385_x_180_y_60_dimx_180_dim_y_180_step_x_3_step_y_3/combined_output_profs.h5', 'r')

    fm = h5py.File(base_path / 'MULTI3D_BIFROST_en024048_hion_snap_385_0_504_0_504_-1020996.0_15000000.0_supplementary_outputs.nc', 'r')

    wave_ha = np.loadtxt(base_path / 'wave_ha.txt')

    wave = vac_to_air(wave_ha / 10)[::-1] * 10

    mean_prof_subs = np.mean(fm['profiles_H'][0, :, :, :, 0], (0, 1))

    si_const = mean_prof_subs[0]

    cgs_const = si_const * 1e3

    fig, axs = plt.subplots(2, 2, figsize=(7, 3.5))

    im00 = axs[0][0].imshow(f['stokes_I'][528, :, :] / cgs_const, cmap='gray', origin='lower')

    im01 = axs[0][1].imshow(fm['profiles_H'][0, 180:180 + 183][:, 60:60 + 183][:, :, 400, 0] / si_const, cmap='gray', origin='lower')

    im10 = axs[1][0].imshow(f['stokes_V'][528] / f['stokes_I'][528], cmap='bwr', origin='lower', vmin=-0.0035, vmax=0.0035)

    im11 = axs[1][1].imshow(fm['profiles_H'][0, 180:180 + 183][:, 60:60 + 183][:, :, 397, 3] / fm['profiles_H'][0, 180:180 + 183][:, 60:60 + 183][:, :, 397, 0], cmap='bwr', origin='lower', vmin=-0.0035, vmax=0.0035)

    fig.colorbar(im00, ax=axs[0][0])
    fig.colorbar(im01, ax=axs[0][1])
    fig.colorbar(im10, ax=axs[1][0])
    fig.colorbar(im11, ax=axs[1][1])

    axs[0][0].set_title(r'Halpha line core PORTA')

    axs[0][1].set_title(r'Halpha line core MULTI3D')

    fig.tight_layout()

    fig.savefig(base_path / 'PORTS_RH15D_vs_MULTI3Dpops_RH1D.pdf', format='pdf', dpi=300)

    fm.close()

    f.close()


def prepare_reinterpolate_to_177_grid(f467, var_name, f177):

    def reinterpolate_to_177_grid(i, j):
        i = int(i)
        j = int(j)
        var = f467[var_name][:, :, i, j]

        cs = CubicSpline(f467['ltau500'][0, i, j], var, axis=2)

        return cs(f177['ltau500'][0, i, j])

    return reinterpolate_to_177_grid


def reinterpolate_multi_3d_supplimentary_outputs():

    new_z = np.array(
        [
            14367.05      , 14076.769     , 13885.17      , 13600.63      ,
        13319.47      , 13133.9       , 12858.3       , 12676.389     ,
        12406.23      , 12139.299     , 11963.11      , 11701.44      ,
        11442.899     , 11272.24      , 11018.811     , 10851.53      ,
        10603.1       , 10357.63      , 10195.6       ,  9955.207     ,
        9721.841     ,  9570.84      ,  9350.985     ,  9208.725     ,
        9001.599     ,  8801.756     ,  8672.444     ,  8484.171     ,
        8302.516     ,  8184.973     ,  8013.836     ,  7848.7135    ,
        7741.872     ,  7586.311     ,  7485.653     ,  7339.1       ,
        7197.6955    ,  7106.2       ,  6972.985     ,  6844.452     ,
        6761.283     ,  6640.194     ,  6561.841     ,  6447.761     ,
        6337.691     ,  6266.47      ,  6162.773     ,  6062.7215    ,
        5997.9825    ,  5903.726     ,  5812.781     ,  5753.935     ,
        5668.256     ,  5612.816     ,  5532.0965    ,  5454.2165    ,
        5403.823     ,  5330.451     ,  5259.659     ,  5213.851     ,
        5147.158     ,  5104.003     ,  5041.171     ,  4980.514     ,
        4940.515     ,  4880.515     ,  4820.516     ,  4780.517     ,
        4720.518     ,  4680.518     ,  4620.519     ,  4560.52      ,
        4520.521     ,  4460.521     ,  4400.522     ,  4360.522     ,
        4300.523     ,  4240.524     ,  4200.526     ,  4140.52675   ,
        4100.52675   ,  4040.52775   ,  3980.527     ,  3940.52875   ,
        3880.53      ,  3820.53      ,  3780.53      ,  3720.53075   ,
        3680.53175   ,  3620.533     ,  3560.533     ,  3520.534     ,
        3460.535     ,  3400.536     ,  3360.536     ,  3300.536     ,
        3260.537     ,  3200.538     ,  3140.539     ,  3100.539     ,
        3040.53875   ,  2980.54      ,  2940.541     ,  2880.543     ,
        2820.544     ,  2780.544     ,  2720.545     ,  2680.545     ,
        2620.546     ,  2560.547     ,  2520.547     ,  2460.547     ,
        2400.548     ,  2360.549     ,  2300.55      ,  2260.55      ,
        2200.551     ,  2140.552     ,  2100.552     ,  2040.553125  ,
        1980.552875  ,  1940.553     ,  1880.554     ,  1840.556     ,
        1780.556     ,  1720.557     ,  1680.556875  ,  1620.557     ,
        1560.558     ,  1520.557875  ,  1460.559     ,  1400.558875  ,
        1360.56      ,  1300.561     ,  1260.561     ,  1200.562     ,
        1140.562     ,  1100.562     ,  1040.563     ,   980.5633125 ,
         940.564     ,   880.5649375 ,   840.5655    ,   780.5664375 ,
         720.594     ,   680.7274375 ,   621.27      ,   561.8346875 ,
         522.242     ,   463.05159375,   423.8785    ,   365.34728125,
         306.90690625,   267.9375    ,   209.40779688,   150.8986875 ,
         112.50460156,    56.37054297,   -38.05236719,   -95.25450781,
        -133.38929688,  -190.49990625,  -247.52040625,  -285.48740625,
        -342.35234375,  -380.22921875,  -437.13378125,  -494.41065625,
        -532.9193125 ,  -591.4046875 ,  -631.07      ,  -691.9981875 ,
        -755.278     ,  -799.137875  ,  -868.0465625 ,  -916.4961875 ,
        -993.7958125
        ]
    )

    base_path = Path('/home/harsh/BifrostRun')

    write_path = Path('/home/harsh/BifrostRun_fast_Access')

    var_names = ['populations', 'Cul', 'a_voigt', 'eta_c', 'eps_c']

    suppl_out_path = base_path / 'MULTI3D_BIFROST_en024048_hion_snap_385_0_504_0_504_-1020996.0_15000000.0_supplementary_outputs.nc'
    suppl_out_path_177 = base_path / 'BIFROST_en024048_hion_snap_385_0_504_0_504_-500000.0_3000000.0_supplementary_outputs.nc'

    fsuppl = h5py.File(suppl_out_path, 'r')
    fsuppl_177 = h5py.File(suppl_out_path_177, 'r')

    fout = h5py.File(write_path / 'MULTI3D_BIFROST_en024048_hion_snap_385_0_504_0_504_-1020996.0_15000000.0_supplementary_outputs_reinterp_177.nc', 'w')

    for var_name in var_names:
        reinterpolate_to_177_grid_param = prepare_reinterpolate_to_177_grid(fsuppl, var_name, fsuppl_177)
        vec_reinterpolate_to_177_grid_param = np.vectorize(reinterpolate_to_177_grid_param, signature='(),()->(m,n,o)')
        param_interp = np.transpose(
            np.fromfunction(
                vec_reinterpolate_to_177_grid_param,
                shape=(504, 504)
            ),
            axes=(2, 3, 0, 1, 4)
        )

        fout[var_name] = param_interp

        sys.stdout.write('{} Done\n'.format(var_name))

    fout.close()

    fsuppl_177.close()

    fsuppl.close()


if __name__ == '__main__':
    # get_var_at_ltau('temperature')
    compare_porta_1d_with_multi3d()
    # reinterpolate_multi_3d_supplimentary_outputs()
