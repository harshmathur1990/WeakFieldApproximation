import sys
# sys.path.insert(1, '/home/harsh/CourseworkRepo/rh/rhv2src/python/')
sys.path.insert(1, '/home/harsh/rh/python/')
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
import rh
import time


atmos_file = Path(
    '/data/harsh/run_bifrost/Atmos/bifrost_en024048_hion_0_504_0_504_-500000.0_3000000.0.nc'
)

ltau_out_file = Path(
    '/data/harsh/run_bifrost/Atmos/bifrost_en024048_hion_0_504_0_504_-500000.0_3000000.0_copy.nc'
)

# atmos_file = Path(
#     '/home/harsh/BifrostRun/bifrost_en024048_hion_0_504_0_504_-500000.0_3000000.0.nc'
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


def create_mag_file(
    Bx, By, Bz,
    write_path,
    shape=(177, )
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


def write_atmos_files(write_path, x, y):
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
        id='Bifrost {} {}'.format(x, y),
        scale='height'
    )
    create_mag_file(
        Bx=f['B_x'][0, x, y],
        By=f['B_y'][0, x, y],
        Bz=f['B_z'][0, x, y],
        write_path=write_path
    )
    f.close()


class Status(enum.Enum):
    Requesting_work = 0
    Work_assigned = 1
    Work_done = 2
    Work_failure = 3


def reverse_transitions(transitions):

    rev = transitions.copy()
    i = 0
    while i < len(rev):
        rev[i] = transitions[i+1]
        rev[i+1] = transitions[i]
        i += 2

    return rev


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


def do_work(read_path):
    cwd = os.getcwd()

    os.chdir(read_path)

    out = rh.readOutFiles()

    ltau500 = np.array(out.geometry.tau_ref)

    os.chdir(cwd)

    return ltau500, Status.Work_done


def make_ray_file():

    os.chdir('/home/harsh/CourseworkRepo/rh/rhv2src/rhf1d/run_3')

    out = rh.readOutFiles()

    wave = np.array(out.spect.lambda0)

    indices = list()

    interesting_waves = [121.5668237310, 121.5673644608, 656.275181, 656.290944, 102.572182505, 102.572296565, 656.272483, 656.277153, 	656.270970, 656.285177, 656.286734]

    for w in interesting_waves:
        indices.append(
            np.argmin(np.abs(wave-w))
        )

    f = open('ray.input', 'w')

    f.write('1.00\n')
    f.write(
        '{} {}'.format(
            len(indices),
            ' '.join([str(indice) for indice in indices])
        )
    )
    f.close()


if __name__ == '__main__':

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

        job_matrix = np.zeros((504, 504), dtype=np.int64)

        x, y = np.where(job_matrix == 0)

        for i in range(x.size):
            waiting_queue.add(i)

        for worker in range(1, size):
            if len(waiting_queue) == 0:
                break
            item = waiting_queue.pop()
            work_type = {
                'job': 'work',
                'item': (item, x[item], y[item])
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
            item, xx, yy, ltau500 = status_dict['item']
            fo = h5py.File(ltau_out_file, 'r+')
            fo['ltau500'][0, xx, yy] = ltau500
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
                    'item': (new_item, x[new_item], y[new_item])
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

            item, x, y = work_type['item']

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

            # start_time = time.time()
            for cmd in commands:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(sub_dir_path),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True
                )
                process.communicate()

            # sys.stdout.write(
            #     'Rank: {} RH Remove Files Time: {}\n'.format(
            #         rank, time.time() - start_time
            #     )
            # )

            # start_time = time.time()
            write_atmos_files(sub_dir_path, x, y)
            # sys.stdout.write(
            #     'Rank: {} RH Make Atmosphere Files Time: {}\n'.format(
            #         rank, time.time() - start_time
            #     )
            # )

            cmdstr = '/home/harsh/RH-Old/rhf1d/rhf1d'

            # cmdstr = '/home/harsh/CourseworkRepo/rh/rhv2src/rhf1d/rhf1d'

            command = '{} 2>&1 | tee output.txt'.format(
                cmdstr
            )

            # start_time = time.time()
            process = subprocess.Popen(
                command,
                cwd=str(sub_dir_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )

            process.communicate()

            # sys.stdout.write(
            #     'Rank: {} RH Run Time: {}\n'.format(
            #         rank, time.time() - start_time
            #     )
            # )

            # start_time = time.time()
            ltau500, status = do_work(sub_dir_path)
            # sys.stdout.write(
            #     'Rank: {} RH Save Time: {}\n'.format(
            #         rank, time.time() - start_time
            #     )
            # )
            comm.send({'status': Status.Work_done, 'item': (item, x, y, ltau500)}, dest=0, tag=2)
