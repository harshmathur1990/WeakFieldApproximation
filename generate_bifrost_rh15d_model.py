import sys
# sys.path.insert(1, '/home/harsh/CourseworkRepo/stic/example')
sys.path.insert(1, '/home/harsh/stic/example')
import numpy as np
import sunpy.io.fits
from helita.sim import rh15d
from pathlib import Path
from witt import witt


def make_atmosphere(
    foldername,
    simulation_name,
    snap,
    start_x, end_x,
    start_y, end_y,
    height_min_in_m,
    height_max_in_m,
    simulation_code_name='BIFROST',
    lte=False
):
    if isinstance(foldername, str):
        foldername = Path(foldername)

    temp_file = '{}_{}_lgtg_{}.fits'.format(
        simulation_code_name,
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

    vz_file = '{}_{}_uz_{}.fits'.format(
        simulation_code_name,
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

    ne = np.zeros((1, end_x - start_x, end_y - start_y, ind.size))

    if lte is False:
        for i in range(1, 7):
            nhfile = '{}_{}_lgn{}_{}.fits'.format(
                simulation_code_name,
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

        nefile = '{}_{}_lgne_{}.fits'.format(
            simulation_code_name,
            simulation_name, snap
        )

        data, header = sunpy.io.fits.read(
                foldername / nefile
            )[0]

        ne[0] = np.power(
            10,
            np.transpose(
                data[ind, start_x:end_x, start_y:end_y],
                axes=(1, 2, 0)
            )
        )

    else:
        w = witt()

        pe_frpm_pg = np.vectorize(
            w.get_pe_from_pg,
            signature='(z,x,y),(z,x,y)->(z,x,y)'
        )

        pg_file = '{}_{}_lgp_{}.fits'.format(
            simulation_code_name,
            simulation_name, snap
        )

        data, header = sunpy.io.fits.read(
            foldername / pg_file
        )[0]

        pe = pe_frpm_pg(
            np.transpose(
                T[0],
                axes=(2, 1, 0)
            ),
            np.power(
                10,
                data[ind, start_x:end_x, start_y:end_y],
            ) * 10
        )

        h6tpgpe = np.vectorize(
            w.getH6pop,
            signature='(z,x,y),(z,x,y),(z,x,y)->(z,x,y,6)'
        )

        h6pop = h6tpgpe(
            np.transpose(
                T[0],
                axes=(2, 1, 0)
            ),
            data * 10,
            pe
        )

        nH[0] = np.transpose(
            h6pop,
            axes=(1, 2, 0)
        ) / 1e6

    bxfile = '{}_{}_bx_{}.fits'.format(
        simulation_code_name,
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

    byfile = '{}_{}_by_{}.fits'.format(
        simulation_code_name,
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

    bzfile = '{}_{}_bz_{}.fits'.format(
        simulation_code_name,
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

    outfile = str(
        foldername / '{}_{}_{}_{}_{}_{}_{}_{}.nc'.format(
            simulation_code_name,
            simulation_name, start_x, end_x, start_y, end_y, height_min_in_m, height_max_in_m
        )
    )

    rh15d.make_xarray_atmos(
        outfile, T, vz, z,
        nH=nH, Bz=Bz, By=By,
        Bx=Bx, ne=ne,
        desc='{} Simulation (simulation_name, start_x, end_x, start_y, end_y, height_min_in_m, height_max_in_m) - {} - {} - {} - {} - {} - {} - {}'.format(
            simulation_code_name,
            simulation_name, start_x, end_x, start_y, end_y, height_min_in_m, height_max_in_m
        ),
        snap=snap
    )


if __name__ == '__main__':
    # make_atmosphere(
    #     '/data/harsh/en024048_hion/atmos',
    #     'en024048_hion',
    #     412,
    #     250, 251,
    #     250, 251,
    #     -500 * 1e3,
    #     2000 * 1e3
    # )

    make_atmosphere(
        '/data/harsh/ar098192/atmos',
        'ar098192',
        294000,
        0, 256,
        0, 512,
        -500 * 1e3,
        3000 * 1e3,
        simulation_code_name='MURaM',
        lte=True
    )
