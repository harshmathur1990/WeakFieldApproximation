import sys
# sys.path.insert(1, '/home/harsh/CourseworkRepo/rh/RH-uitenbroek/python/')
sys.path.insert(1, '/home/harsh/rh-uitenbroek/python/')
import enum
import os
import numpy as np
import h5py
import xdrlib
from mpi4py import MPI
from pathlib import Path
import tables as tb
import shutil
from helita.sim import multi
import subprocess
import rhanalyze
import time


atmos_file = Path(
    '/data/harsh/merge_bifrost_output/bifrost_en024048_hion_0_504_0_504_-500000.0_3000000.0.nc'
)

# atmos_file = Path(
#     '/home/harsh/BifrostRun/bifrost_en024048_hion_0_504_0_504_-500000.0_3000000.0.nc'
# )

ltau_out_file = Path(
    '/data/harsh/merge_bifrost_output/bifrost_en024048_hion_0_504_0_504_-500000.0_3000000.0_supplementary_outputs.nc'
)

# ltau_out_file = Path(
#     '/home/harsh/BifrostRun/bifrost_en024048_hion_0_504_0_504_-500000.0_3000000.0_supplementary_outputs.nc'
# )

rh_run_base_dirs = Path('/data/harsh/run_bifrost_dirs')

# rh_run_base_dirs = Path('/home/harsh/BifrostRun/run_bifrost_dirs')

stop_file = rh_run_base_dirs / 'stop'

sub_dir_format = 'process_{}'

input_filelist = [
    'molecules.input',
    'kurucz.input',
    'atoms.input',
    'keyword.input',
    'ray.input',
    'contribute.input'
]


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
    f.close()


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
    return ltau500, adamp, cularr, populations, eta_c, eps_c, Status.Work_done


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

        try:
            os.remove(ltau_out_file)
        except:
            pass

        fo = h5py.File(ltau_out_file, 'w')
        fo['ltau500'] = np.zeros((1, nx, ny, height_len), dtype=np.float64)
        fo['a_voigt'] = np.zeros((1, n_rad_transitions, nx, ny, height_len), dtype=np.float64)
        fo['populations'] = np.zeros((1, nH, nx, ny, height_len), dtype=np.float64)
        fo['Cul'] = np.zeros((1, n_transitions, nx, ny, height_len), dtype=np.float64)
        fo['eta_c'] = np.zeros((1, n_rad_transitions, nx, ny, height_len), dtype=np.float64)
        fo['eps_c'] = np.zeros((1, n_rad_transitions, nx, ny, height_len), dtype=np.float64)
        fo.close()

        sys.stdout.write('Made Output File.\n')

        job_matrix = np.zeros((nx, ny), dtype=np.int64)

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

            status_dict = comm.recv(
                source=MPI.ANY_SOURCE,
                tag=2,
                status=status
            )
            sender = status.Get_source()
            jobstatus = status_dict['status']
            item, xx, yy, ltau500, adamp, cularr, populations, eta_c, eps_c = status_dict['item']
            fo = h5py.File(ltau_out_file, 'r+')
            fo['ltau500'][0, xx, yy] = ltau500
            fo['a_voigt'][0, :, xx, yy] = adamp
            fo['populations'][0, :, xx, yy] = populations.T
            fo['Cul'][0, :, xx, yy] = cularr
            fo['eta_c'][0, :, xx, yy] = eta_c
            fo['eps_c'][0, :, xx, yy] = eps_c
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

            cmdstr = '/home/harsh/rh-uitenbroek/rhf1d/rhf1d'

            # cmdstr = '/home/harsh/CourseworkRepo/rh/RH-uitenbroek/rhf1d/rhf1d'

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

            ltau500, adamp, cularr, populations, eta_c, eps_c, status_work = do_work(sub_dir_path)

            comm.send({'status': status_work, 'item': (item, x, y, ltau500, adamp, cularr, populations, eta_c, eps_c)}, dest=0, tag=2)
