import sys
import numpy as np
import sunpy.io.fits


variable_to_file = {
    'Bx': 'BIFROST_en024048_hion_bx_404.fits',
    'By': 'BIFROST_en024048_hion_by_404.fits',
    'Bz': 'BIFROST_en024048_hion_bz_404.fits',
    'nH': [
        'BIFROST_en024048_hion_lgn1_404.fits',
        'BIFROST_en024048_hion_lgn2_404.fits',
        'BIFROST_en024048_hion_lgn3_404.fits',
        'BIFROST_en024048_hion_lgn4_404.fits',
        'BIFROST_en024048_hion_lgn5_404.fits',
        'BIFROST_en024048_hion_lgn6_404.fits'
    ],
    'ne': 'BIFROST_en024048_hion_lgne_404.fits',
    'vz': 'BIFROST_en024048_hion_uz_404.fits',
    'cmass': 'BIFROST_en024048_hion_lgr_404.fits',
    'T': 'BIFROST_en024048_hion_lgtg_404.fits'
}


def generate(base_path, variable_to_file):
    result_dict = dict()

    for variable, filename in variable_to_file.items():

        sys.stdout.write('Working on variable: {}\n'.format(variable))

        if not isinstance(filename, list):
            filepath = base_path / filename

            data, _ = sunpy.io.fits.read(filepath)[0]

            strides = data.strides

            shape = data.shape

            data_rh = np.lib.stride_tricks.as_strided(
                data,
                shape=(
                    shape[1],
                    shape[2],
                    shape[0]
                ),
                strides=(
                    strides[1],
                    strides[2],
                    strides[0]
                ),
            )

            if variable in ['ne', 'cmass', 'T']:
                data_rh = np.power(10, data_rh)

            result_dict[variable] = data_rh[:, :, ::-1]
        else:
            file_list = filename

            variable_res = None

            for index, a_file in enumerate(file_list):

                sys.stdout.write('Working on Index: {}\n'.format(index))

                filepath = base_path / a_file

                data, _ = sunpy.io.fits.read(filepath)[0]

                strides = data.strides

                shape = data.shape

                data_rh = np.lib.stride_tricks.as_strided(
                    data,
                    shape=(
                        shape[1],
                        shape[2],
                        shape[0]
                    ),
                    strides=(
                        strides[1],
                        strides[2],
                        strides[0]
                    ),
                )

                data_rh = np.power(10, data_rh)

                if variable_res is None:
                    variable_res = np.zeros(
                        shape=(len(file_list), shape[1], shape[2], shape[0]),
                        dtype=data_rh.dtype
                    )

                variable_res[index] = data_rh[:, :, ::-1]

            result_dict[variable] = variable_res

    return result_dict
