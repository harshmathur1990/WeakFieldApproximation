import numpy as np


def generate_constant_values(length, in_value_list):
    return [np.ones(shape=length) * value for value in in_value_list]


def generate_linear_with_log_optical_depth_values(
    log_tau,
    value_top_list,
    value_bot_list,
    mode='topgreater'
):

    out_list = list()

    for value_top in value_top_list:
        for value_bot in value_bot_list:
            if mode == 'topgreater':
                if value_top < value_bot:
                    continue
            if mode == 'botgreater':
                if value_top > value_bot:
                    continue

            slope = (value_top - value_bot) / (log_tau[0] - log_tau[-1])

            values = slope * (np.array(log_tau) - log_tau[0]) + value_top

            out_list.append(values)

    return out_list


def generate_constant_values_with_discontinuity(
    log_tau,
    value_top,
    value_bot,
    discontinuity
):
    indices_top = np.where(log_tau <= discontinuity)[0]
    indices_bot = np.where(log_tau > discontinuity)[0]

    out_array = np.ones(shape=len(log_tau))

    out_array[indices_top] *= value_top

    out_array[indices_bot] *= value_bot

    return out_array


def generate_magnetic_field_with_inclination(
    length,
    magnitude_list,
    inclination_list,
    azimuth_list
):

    out_list = list()

    for magnitude in magnitude_list:
        for inclination in inclination_list:
            for azimuth in azimuth_list:
                bz = np.ones(shape=(length)) * \
                    np.cos(np.deg2rad(inclination)) * magnitude
                bx = np.ones(shape=(length)) * magnitude * \
                    np.sin(np.deg2rad(inclination)) * \
                    np.cos(np.deg2rad(azimuth))
                by = np.ones(shape=(length)) * magnitude * \
                    np.sin(np.deg2rad(inclination)) * \
                    np.sin(np.deg2rad(azimuth))
                out_list.append((bz, bx, by))

    return out_list
