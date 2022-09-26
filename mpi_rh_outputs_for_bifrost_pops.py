import sys
# sys.path.insert(1, '/home/harsh/CourseworkRepo/rh/RH-uitenbroek/python/')
sys.path.insert(1, '/home/harsh/rh-uitenbroek/python/')
# sys.path.insert(2, '/home/harsh/CourseworkRepo/rh/rhv2src/python/')
sys.path.insert(2, '/home/harsh/RH-Old/python/')
import enum
import os
import numpy as np
import h5py
import xdrlib
from mpi4py import MPI
from pathlib import Path
import shutil
from helita.sim import multi
import subprocess
import rhanalyze
from xdr_tools import XDR_Reader, XDR_Specs, XDR_Struct


xdr_pops = XDR_Specs(
    [
        ['atmosID', 'cstr', None],
        ['Nlevel', int, None],
        ['Nspace', int, None],
        ['n', float, '(Nspace,Nlevel)'],  # Stop if not n
        ['nstar', float, '(Nspace,Nlevel)']
    ]
)

atmos_file = Path(
    '/data/harsh/merge_bifrost_output/BIFROST_en024048_hion_snap_385_0_504_0_504_-1020996.0_15000000.0.nc'
)

# atmos_file = Path(
#     '/home/harsh/BifrostRun/BIFROST_en024048_hion_snap_385_0_504_0_504_-1020996.0_15000000.0.nc'
# )

ltau_out_file = Path(
    '/data/harsh/merge_bifrost_output/MULTI3D_BIFROST_en024048_hion_snap_385_0_504_0_504_-1020996.0_15000000.0_supplementary_outputs.nc'
)

# ltau_out_file = Path(
#     '/home/harsh/BifrostRun/MULTI3D_BIFROST_en024048_hion_snap_385_0_504_0_504_-1020996.0_15000000.0_supplementary_outputs.nc'
# )

# bifrost_out_file = Path('/home/harsh/BifrostRun/Multi-3D-H_6_level_populations/output_aux.hdf5')

bifrost_out_file = Path('/data/harsh/merge_bifrost_output/Multi-3D-H_6_level_populations/output_aux.hdf5')

rh_run_base_dirs = Path('/data/harsh/run_bifrost_dirs')

# rh_run_base_dirs = Path('/home/harsh/BifrostRun/run_bifrost_dirs')

stop_file = rh_run_base_dirs / 'stop'

# rh_base_path = Path('/home/harsh/CourseworkRepo/rh/RH-uitenbroek/')

rh_base_path = Path('/home/harsh/rh-uitenbroek/')

sub_dir_format = 'process_{}'

input_filelist = [
    'molecules.input',
    'kurucz.input',
    'atoms.input',
    'keyword.input',
    'ray.input',
    'contribute.input',
    'wavefile.wave'
]

wave_H = np.arange(6562.8 - 4, 6562.8 + 4, 0.01)


bifrost_indice = np.array(
    [
        237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249,
        250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262,
        263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275,
        276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288,
        289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301,
        302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314,
        315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327,
        328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340,
        341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353,
        354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366,
        367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379,
        380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392,
        393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405,
        406, 407, 408, 409, 410, 411, 412, 413
    ]
)


def generate_radiative_transitions():
    transitions = [3, 0, 1, 0, 5, 1, 5, 3, 7, 0, 4, 0, 7, 2, 4, 2, 6, 1, 8, 3, 6, 3]

    return transitions


def collisional_transitions():
    collisional_transitions = list()

    radiative_transitions = generate_radiative_transitions()
    for i in range(9):
        for j in range(9):
            if i <= j:
                continue

            rad_p = False

            k = 0
            while k < len(radiative_transitions):
                if radiative_transitions[k] == i and radiative_transitions[k + 1] == j:
                    rad_p = True
                    break
                k += 2

            if rad_p is False:
                collisional_transitions.append(i)
                collisional_transitions.append(j)

    return collisional_transitions


def create_mag_file(
    Bx, By, Bz,
    write_path,
    height_len
):
    b_filename = 'MAG_FIELD.B'
    xdr = xdrlib.Packer()

    Babs = np.sqrt(
        np.add(
            np.square(Bx),
            np.add(
                np.square(By),
                np.square(Bz)
            )
        )
    )

    Binc = np.arccos(np.divide(Bz, Babs))

    Bazi = np.arctan2(By, Bx)

    shape = (height_len,)

    xdr.pack_farray(
        np.prod(shape),
        Babs.flatten(),
        xdr.pack_double
    )
    xdr.pack_farray(
        np.prod(shape),
        Binc,
        xdr.pack_double
    )
    xdr.pack_farray(
        np.prod(shape),
        Bazi,
        xdr.pack_double
    )

    with open(write_path / b_filename, 'wb') as f:
        f.write(xdr.get_buffer())


def write_atmos_files(write_path, x, y, height_len):
    atmos_filename = 'Atmos1D.atmos'
    f = h5py.File(atmos_file, 'r')
    multi.watmos_multi(
        str(write_path / atmos_filename),
        f['temperature'][0, x, y],
        f['electron_density'][0, x, y] / 1e6,
        z=f['z'][0, x, y] / 1e3,
        vz=f['velocity_z'][0, x, y] / 1e3,
        # vturb=f['velocity_turbulent'][0, x, y] / 1e3,
        nh=f['hydrogen_populations'][0, :, x, y] / 1e6,
        id='BIFROST {} {}'.format(x, y),
        scale='height'
    )
    create_mag_file(
        Bx=f['B_x'][0, x, y],
        By=f['B_y'][0, x, y],
        Bz=f['B_z'][0, x, y],
        write_path=write_path,
        height_len=height_len
    )

    write_populations(
        write_path=write_path,
        x=x,
        y=y,
        height_len=height_len
    )

    f.close()


def write_populations(write_path, x, y, height_len):
    reader = XDR_Reader(XDR_Specs(xdr_pops))

    struct = XDR_Struct(XDR_Specs(xdr_pops))
    struct['atmosID'] = 'BIFROST {} {}'.format(x, y).encode('utf-8')
    struct['Nspace'] = height_len
    struct['Nlevel'] = 10

    bifrost_x = y
    bifrost_y = 504 - x - 1

    fb = h5py.File(bifrost_out_file, 'r')

    pops = fb['atom_H']['populations'][:, bifrost_x, bifrost_y]

    pops_sublevel = np.zeros((height_len, 10), dtype=np.float64)

    pops_sublevel[:, 0] = pops[0]

    pops_sublevel[:, 1] = pops[1] * 2 / 8

    pops_sublevel[:, 2] = pops[1] * 2 / 8

    pops_sublevel[:, 3] = pops[1] * 4 / 8

    pops_sublevel[:, 4] = pops[2] * 2 / 18

    pops_sublevel[:, 5] = pops[2] * 2 / 18

    pops_sublevel[:, 6] = pops[2] * 4 / 18

    pops_sublevel[:, 7] = pops[2] * 4 / 18

    pops_sublevel[:, 8] = pops[2] * 6 / 18

    pops_sublevel[:, 9] = pops[5]

    pops_lte = fb['atom_H']['populations_LTE'][:, bifrost_x, bifrost_y]

    pops_sublevel_LTE = np.zeros((height_len, 10), dtype=np.float64)

    pops_sublevel_LTE[:, 0] = pops_lte[0]

    pops_sublevel_LTE[:, 1] = pops_lte[1] * 2 / 8

    pops_sublevel_LTE[:, 2] = pops_lte[1] * 2 / 8

    pops_sublevel_LTE[:, 3] = pops_lte[1] * 4 / 8

    pops_sublevel_LTE[:, 4] = pops_lte[2] * 2 / 18

    pops_sublevel_LTE[:, 5] = pops_lte[2] * 2 / 18

    pops_sublevel_LTE[:, 6] = pops_lte[2] * 4 / 18

    pops_sublevel_LTE[:, 7] = pops_lte[2] * 4 / 18

    pops_sublevel_LTE[:, 8] = pops_lte[2] * 6 / 18

    pops_sublevel_LTE[:, 9] = pops_lte[5]

    struct['n'] = pops_sublevel

    struct['nstar'] = pops_sublevel_LTE

    reader.write(write_path / 'pops.H.out', struct)

    fb.close()

class Status(enum.Enum):
    Requesting_work = 0
    Work_assigned = 1
    Work_done = 2
    Work_failure = 3


def do_work(read_path):
    cwd = os.getcwd()

    os.chdir(read_path)

    out = rhanalyze.rhout()

    ltau500 = np.log(np.array(out.geometry.tau500))

    radiative_transition_as_list = generate_radiative_transitions()

    radiative_transitions = np.array(radiative_transition_as_list).reshape(len(radiative_transition_as_list) // 2, 2)

    col_transitions_as_list = collisional_transitions()

    total_transitions_as_list = radiative_transition_as_list + col_transitions_as_list

    total_transitions = np.array(total_transitions_as_list).reshape(len(total_transitions_as_list) // 2, 2)

    adamp = np.zeros((radiative_transitions.shape[0], ltau500.size))

    cularr = np.zeros((total_transitions.shape[0], ltau500.size))

    populations = np.zeros((ltau500.size, nH), dtype=np.float64)

    for indd, atom in out.atoms.items():
        if atom.atomID == 'H':
            for index1, trans in atom.transition.items():
                if trans.type == 'ATOMIC_LINE':
                    i = trans.i
                    j = trans.j

                    intersection = np.intersect1d(np.where(radiative_transitions[:, 0] == j), np.where(radiative_transitions[:, 1] == i))

                    if intersection.size == 1:
                        adamp[intersection[0]] = trans.adamp

            populations[:, :] = atom.n[:, 0:9]

            for index2 in range(total_transitions.shape[0]):
                j = total_transitions[index2][0]
                i = total_transitions[index2][1]

                cularr[index2] = atom.Cij[:, i, j]

    interested_wave = [121.5668237310, 121.5673644608, 656.275181, 656.290944, 102.572182505, 102.572296565, 656.272483,
                       656.277153, 656.270970, 656.285177, 656.286734]

    eta_c = np.zeros((len(interested_wave), ltau500.size))
    eps_c = np.zeros((len(interested_wave), ltau500.size))

    for index3, wave in enumerate(interested_wave):
        ind = np.argmin(np.abs(out.spectrum.waves - wave))

        out.opacity.read(ind, 4)

        eta_c[index3] = out.opacity.chi_c
        eps_c[index3] = out.opacity.eta_c

    intensity_list = list()

    for w in [wave_H]:
        indd = list()
        for ww in w:
            indd.append(np.argmin(np.abs(out.spectrum.waves - ww / 10)))
        indd = np.array(indd)
        spect = np.zeros((indd.size, 4))
        spect[:, 0] = out.rays[0].I[indd]
        spect[:, 1] = out.rays[0].Q[indd]
        spect[:, 2] = out.rays[0].U[indd]
        spect[:, 3] = out.rays[0].V[indd]
        intensity_list.append(spect)

    os.chdir(cwd)

    if np.array_equal(adamp, np.zeros_like(adamp)):
        comm.Abort(-1)
    if np.array_equal(cularr, np.zeros_like(cularr)):
        comm.Abort(-2)
    if np.array_equal(populations, np.zeros_like(populations)):
        comm.Abort(-3)
    if np.array_equal(eta_c, np.zeros_like(eta_c)):
        comm.Abort(-4)
    if np.array_equal(eps_c, np.zeros_like(eps_c)):
        comm.Abort(-5)
    return ltau500, adamp, cularr, populations, eta_c, eps_c, intensity_list, Status.Work_done


if __name__ == '__main__':

    nH = 9
    n_transitions = 36
    n_rad_transitions = 11

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    stop_work = False

    if rank == 0:
        status = MPI.Status()
        waiting_queue = set()
        running_queue = set()
        finished_queue = set()
        failure_queue = set()

        sys.stdout.write('Making Output File.\n')

        f = h5py.File(atmos_file, 'r')
        nx = f['temperature'].shape[1]
        ny = f['temperature'].shape[2]
        height_len = f['temperature'].shape[3]
        f.close()

        # try:
        #     os.remove(ltau_out_file)
        # except:
        #     pass

        if not os.path.exists(ltau_out_file):
            fo = h5py.File(ltau_out_file, 'w')
            fo['ltau500'] = np.zeros((1, nx, ny, height_len), dtype=np.float64)
            fo['a_voigt'] = np.zeros((1, n_rad_transitions, nx, ny, height_len), dtype=np.float64)
            fo['populations'] = np.zeros((1, nH, nx, ny, height_len), dtype=np.float64)
            fo['Cul'] = np.zeros((1, n_transitions, nx, ny, height_len), dtype=np.float64)
            fo['eta_c'] = np.zeros((1, n_rad_transitions, nx, ny, height_len), dtype=np.float64)
            fo['eps_c'] = np.zeros((1, n_rad_transitions, nx, ny, height_len), dtype=np.float64)
            fo['profiles_H'] = np.zeros((1, nx, ny, wave_H.size, 4), dtype=np.float64)
            fo['wave_H'] = wave_H
            fo.close()

        sys.stdout.write('Made Output File.\n')

        job_matrix = np.zeros((nx, ny), dtype=np.int64)

        fo = h5py.File(ltau_out_file, 'r')
        a, b, c = np.where(fo['ltau500'][:, :, :, 0] != 0)
        job_matrix[b, c] = 1
        fo.close()

        x, y = np.where(job_matrix == 0)

        for i in range(x.size):
            waiting_queue.add(i)

        for worker in range(1, size):
            if len(waiting_queue) == 0:
                break
            item = waiting_queue.pop()
            work_type = {
                'job': 'work',
                'item': (item, x[item], y[item], height_len)
            }
            comm.send(work_type, dest=worker, tag=1)
            running_queue.add(item)

        sys.stdout.write('Finished First Phase\n')

        while len(running_queue) != 0 or len(waiting_queue) != 0:

            if stop_work == False and stop_file.exists():
                stop_work = True
                waiting_queue = set()
                stop_file.unlink()
                sys.stdout.write('\nStop requested.\n')

            status_dict = comm.recv(
                source=MPI.ANY_SOURCE,
                tag=2,
                status=status
            )
            sender = status.Get_source()
            jobstatus = status_dict['status']
            item, xx, yy, ltau500, adamp, cularr, populations, eta_c, eps_c, intensity_list = status_dict['item']
            fo = h5py.File(ltau_out_file, 'r+')
            fo['ltau500'][0, xx, yy] = ltau500
            fo['a_voigt'][0, :, xx, yy] = adamp
            fo['populations'][0, :, xx, yy] = populations.T
            fo['Cul'][0, :, xx, yy] = cularr
            fo['eta_c'][0, :, xx, yy] = eta_c
            fo['eps_c'][0, :, xx, yy] = eps_c
            fo['profiles_H'][0, xx, yy] = intensity_list[0]
            fo.close()
            sys.stdout.write(
                'Sender: {} x: {} y: {} Status: {}\n'.format(
                    sender, xx, yy, jobstatus.value
                )
            )
            running_queue.discard(item)
            if jobstatus == Status.Work_done:
                finished_queue.add(item)
            else:
                failure_queue.add(item)

            if len(waiting_queue) != 0:
                new_item = waiting_queue.pop()
                work_type = {
                    'job': 'work',
                    'item': (new_item, x[new_item], y[new_item], height_len)
                }
                comm.send(work_type, dest=sender, tag=1)
                running_queue.add(new_item)

        for worker in range(1, size):
            work_type = {
                'job': 'stopwork'
            }
            comm.send(work_type, dest=worker, tag=1)

    if rank > 0:
        sub_dir_path = rh_run_base_dirs / 'runs' / 'process_{}'.format(rank)
        sub_dir_path.mkdir(parents=True, exist_ok=True)
        for input_file in input_filelist:
            shutil.copy(
                rh_run_base_dirs / input_file,
                sub_dir_path / input_file
            )

        while 1:
            work_type = comm.recv(source=0, tag=1)

            if work_type['job'] != 'work':
                break

            item, x, y, height_len = work_type['item']

            sys.stdout.write(
                'Rank: {} x: {} y: {} start\n'.format(
                    rank, x, y
                )
            )

            commands = [
                'rm -rf *.dat',
                'rm -rf *.out',
                'rm -rf spectrum*',
                'rm -rf background.ray',
                'rm -rf Atmos1D.atmos',
                'rm -rf MAG_FIELD.B'
            ]

            for cmd in commands:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(sub_dir_path),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True
                )
                process.communicate()

            write_atmos_files(sub_dir_path, x, y, height_len)

            cmdstr = rh_base_path / 'rhf1d/rhf1d'

            command = '{} 2>&1 | tee output.txt'.format(
                cmdstr
            )

            process = subprocess.Popen(
                command,
                cwd=str(sub_dir_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )

            process.communicate()

            cmdstr = rh_base_path / 'rhf1d/solveray'

            command = '{} 2>&1 | tee output.txt'.format(
                cmdstr
            )

            process = subprocess.Popen(
                command,
                cwd=str(sub_dir_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )

            process.communicate()

            ltau500, adamp, cularr, populations, eta_c, eps_c, intensity_list, status_work = do_work(sub_dir_path)

            comm.send({'status': status_work, 'item': (item, x, y, ltau500, adamp, cularr, populations, eta_c, eps_c, intensity_list)}, dest=0, tag=2)
