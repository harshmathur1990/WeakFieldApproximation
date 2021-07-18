import sys


l_dict = {
    's': 0,
    'p': 1,
    'd': 2,
    'f': 3,
    'g': 4,
    'h': 5,
    'i': 6
}


class Level(object):
    def __init__(self, slp, slv, config, j, e, ryb_e, g):
        self.slp = slp
        self.slv = slv
        self.config = config
        self.j = j
        self.e = e
        self.ryb_e = ryb_e
        self.g = g


class Line(object):
    def __init__(self, i_slp, i_slv, i_j, j_slp, j_slv, j_j, gf):
        self.i_slp = i_slp
        self.i_slv = i_slv
        self.i_j = i_j
        self.j_slp = j_slp
        self.j_slv = j_slv
        self.j_j = j_j
        self.gf = gf


def print_level_list(filename, levels_list):
    f = open(filename, 'w')

    fw = '  {}  {}  {}  {}  {}  {}  {}\n'

    for level in levels_list:
        f.write(
            fw.format(
                level.slp, level.slv, level.config,
                level.j, level.e, level.ryb_e, level.g
            )
        )

    f.close()


def print_line_list(filename, lines_list):
    f = open(filename, 'w')

    fw = '  {}  {}  {}  {}  {}  {}  {}\n'

    for line in lines_list:
        f.write(
            fw.format(
                line.i_slp, line.i_slv, line.i_j,
                line.j_slp, line.j_slv, line.j_j, line.gf
            )
        )

    f.close()


def get_level_config_map(levels_list):
    result_dict = dict()

    for level in levels_list:
        result_dict[level.config] = level

    return result_dict


def generate_energy_level_lines_file_for_rh(
    pqn,
    nist_level_fiename,
    nist_lines_filename,
    topbase_lines_filename,
    write_level_filename,
    write_lines_filename
):
    # do levels

    f = open(nist_level_fiename)
    lines = f.readlines()

    levels_list = list()

    l_nums = {
        's': {'count': 0, 'n': 0},
        'p': {'count': 0, 'n': 0},
        'd': {'count': 0, 'n': 0},
        'f': {'count': 0, 'n': 0},
        'g': {'count': 0, 'n': 0},
        'h': {'count': 0, 'n': 0},
        'i': {'count': 0, 'n': 0}
    }

    for line in lines:
        splitlines = line.split('\t')

        if splitlines[0][1:-1].isnumeric():
            continue

        pn = int(splitlines[0][1:-1][:-1])

        if pn > pqn:
            continue

        l = splitlines[0][1:-1][-1]

        slp = 200 + l_dict[l] * 10 + l_dict[l]%2

        slv_dict = l_nums[l]

        if pn == slv_dict['n']:
            slv = slv_dict['count']
        else:
            slv = slv_dict['count'] + 1
            slv_dict['n'] = pn
            slv_dict['count'] = slv

        j = int(splitlines[2][1:-1][:-2])

        ryb_es = splitlines[4][1:-1].strip()

        if ryb_es[0] == '[' or ryb_es[0] == '(':
            ryd_e = float(splitlines[4][1:-1][1:-1])
        else:
            ryb_e = float(splitlines[4][1:-1])

        g = int(splitlines[3])

        levels_list.append(
            Level(
                slp=slp,
                slv=slv,
                config=splitlines[0][1:-1],
                j=j,
                e=-1+ryb_e,
                ryb_e=ryb_e,
                g=g
            )
        )

    print_level_list(write_level_filename, levels_list)

    f.close()

    #do lines

    level_config_map = get_level_config_map(levels_list)

    f = open(topbase_lines_filename, 'r')

    lines = f.readlines()

    lines_list = list()

    f2 = open(nist_lines_filename, 'r')

    nist_lines = f2.readlines()

    for line in lines:
        splitlines = line.split('\t')

        i_slp = int(splitlines[3])

        i_slv = int(splitlines[5])

        j_slp = int(splitlines[4])

        j_slv = int(splitlines[6])

        gf = float(splitlines[7])

        for nist_line in nist_lines:

            import ipdb;ipdb.set_trace()

            nist_splitlines = nist_line.split('\t')

            if len(nist_splitlines) != 9:
                continue

            if not nist_splitlines[1] or not nist_splitlines[3] or not nist_splitlines[4] or not nist_splitlines[5] or not nist_splitlines[6] or not nist_splitlines[7] or not nist_splitlines[8]:
                continue

            if gf > 0:
                low_slp = j_slp

                low_slv = j_slv

                up_slp = i_slp

                up_slv = i_slv

            else:
                low_slp = i_slp

                low_slv = i_slv

                up_slp = j_slp

                up_slv = j_slv

            low_level_str = nist_splitlines[3][1:-1]

            up_level_str = nist_splitlines[6][1:-1]

            low_level = level_config_map[low_level_str]

            up_level = level_config_map[up_level_str]

            low_j = int(nist_splitlines[5][1:-1][:-2])

            up_j = int(nist_splitlines[8][1:-1][:-2])

            low_n = int(low_level.config[:-1])

            up_n = int(up_level.config[:-1])

            f_value = float(nist_splitlines[1][1:-1])

            if low_n > pqn or up_n > pqn or low_slp != low_level.slp or low_slv != low_level.slv or up_slp != up_level.slp or up_slv != up_level.slv:
                continue

            lines_list.append(
                Line(
                    i_slp=i_slp,
                    i_slv=i_slv,
                    i_j=up_j if gf > 0 else low_j,
                    j_slp=j_slp,
                    j_slv=j_slv,
                    j_j=low_j if gf > 0 else up_j,
                    gf=low_level.g * f_value * (1 if gf > 0 else -1)
                )
            )

    print_line_list(write_lines_filename, lines_list)


if __name__ == '__main__':
    pass