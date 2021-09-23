import numpy as np
import sunpy.io.fits
from helita.sim import rh15d
from pathlib import Path


def make_atmosphere(
    foldername,
    simulation_name,
    snap,
    start_x, end_x,
    start_y, end_y,
    height_min_in_m,
    height_max_in_m
):
    if isinstance(foldername, str):
        foldername = Path(foldername)

    temp_file = 'BIFROST_{}_lgtg_{}.fits'.format(
        simulation_name,
        snap
    )

    data, header = sunpy.io.fits.read(
        foldername / temp_file
    )[0]

    height, _ = sunpy.io.fits.read(
        foldername / temp_file
    )[1]

    height = height * 1e6

    ind = np.where((height >= height_min_in_m) & (height <= height_max_in_m))[0]

    T = np.zeros((1, end_x - start_x, end_y - start_y, ind.size))

    T[0] = np.power(
        10,
        np.transpose(
            data[ind, start_x:end_x, start_y:end_y],
            axes=(1, 2, 0)
        )
    )

    vz_file = 'BIFROST_{}_uz_{}.fits'.format(
        simulation_name, snap
    )

    data, header = sunpy.io.fits.read(
        foldername / vz_file
    )[0]

    vz = np.zeros((1, end_x - start_x, end_y - start_y, ind.size))

    vz[0] = np.transpose(
        data[ind, start_x:end_x, start_y:end_y],
        axes=(1, 2, 0)
    )

    z = np.zeros((1, end_x - start_x, end_y - start_y, ind.size))

    z[:, :, :, :] = height[ind]

    nH = np.zeros((1, 6, end_x - start_x, end_y - start_y, ind.size))

    for i in range(1, 7):
        nhfile = 'BIFROST_{}_lgn{}_{}.fits'.format(
            simulation_name, i, snap
        )

        data, header = sunpy.io.fits.read(
            foldername / nhfile
        )[0]

        nH[0, i-1] = np.power(
            10,
            np.transpose(
                data[ind, start_x:end_x, start_y:end_y],
                axes=(1, 2, 0)
            )
        )

    nefile = 'BIFROST_{}_lgne_{}.fits'.format(
        simulation_name, snap
    )

    data, header = sunpy.io.fits.read(
            foldername / nefile
        )[0]

    ne = np.zeros((1, end_x - start_x, end_y - start_y, ind.size))

    ne[0] = np.power(
        10,
        np.transpose(
            data[ind, start_x:end_x, start_y:end_y],
            axes=(1, 2, 0)
        )
    )

    bxfile = 'BIFROST_{}_bx_{}.fits'.format(
        simulation_name, snap
    )

    data, header = sunpy.io.fits.read(
            foldername / bxfile
        )[0]

    Bx = np.zeros((1, end_x - start_x, end_y - start_y, ind.size))

    Bx[0] = np.transpose(
        data[ind, start_x:end_x, start_y:end_y],
        axes=(1, 2, 0)
    )

    byfile = 'BIFROST_{}_by_{}.fits'.format(
        simulation_name, snap
    )

    data, header = sunpy.io.fits.read(
            foldername / byfile
        )[0]

    By = np.zeros((1, end_x - start_x, end_y - start_y, ind.size))

    By[0] = np.transpose(
        data[ind, start_x:end_x, start_y:end_y],
        axes=(1, 2, 0)
    )

    bzfile = 'BIFROST_{}_bz_{}.fits'.format(
        simulation_name, snap
    )

    data, header = sunpy.io.fits.read(
            foldername / bzfile
        )[0]

    Bz = np.zeros((1, end_x - start_x, end_y - start_y, ind.size))

    Bz[0] = np.transpose(
        data[ind, start_x:end_x, start_y:end_y],
        axes=(1, 2, 0)
    )

    T = T[:, :, :, ::-1]
    vz = vz[:, :, :, ::-1]
    z = z[:, :, :, ::-1]
    nH = nH[:, :, :, :, ::-1]
    ne = ne[:, :, :, ::-1]
    Bx = Bx[:, :, :, ::-1]
    By = By[:, :, :, ::-1]
    Bz = Bz[:, :, :, ::-1]

    outfile = 'bifrost_{}_{}_{}_{}_{}_{}_{}.nc'.format(
        simulation_name, start_x, end_x, start_y, end_y, height_min_in_m, height_max_in_m
    )

    rh15d.make_xarray_atmos(
        outfile, T, vz, z,
        nH=nH, Bz=Bz, By=By,
        Bx=Bx, ne=ne,
        desc='Bifrost Simulation (simulation_name, start_x, end_x, start_y, end_y, height_min_in_m, height_max_in_m) - {} - {} - {} - {} - {} - {} - {}'.format(
            simulation_name, start_x, end_x, start_y, end_y, height_min_in_m, height_max_in_m
        ),
        snap=snap
    )


if __name__ == '__main__':
    make_atmosphere(
        '/data/harsh/en024048_hion/atmos',
        'en024048_hion',
        412,
        250, 251,
        250, 251,
        -500 * 1e3,
        2000 * 1e3
    )
