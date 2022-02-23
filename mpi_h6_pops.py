import sys
# sys.path.insert(1, '/home/harsh/CourseworkRepo/rh/rhv2src/python/')
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
    '/data/harsh/ar098192/atmos/MURaM_ar098192_0_256_0_512_-500000.0_3000000.0.nc'
)

ltau_out_file = Path(
    '/data/harsh/ar098192/atmos/MURaM_ar098192_0_256_0_512_-500000.0_3000000.0_ltau_h6pops_n_elec.nc'
)

rh_run_base_dirs = Path('/data/harsh/run_muram_dirs')

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
    shape=(55, )
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
        id='MURAM {} {}'.format(x, y),
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


def do_work(read_path):
    cwd = os.getcwd()

    os.chdir(read_path)

    out = rhanalyze.rhout()

    ltau500 = np.log(np.array(out.geometry.tau500))

    ne = out.atmos.n_elec

    h6pops = None

    for ind, atom in out.atoms.items():
        if atom.atomID == 'H':
            h6pops = atom.n

    os.chdir(cwd)

    if h6pops is None:
        sys.stdout.write('H6Pops is Empty.\n')

        comm.Abort(-1)
    return ltau500, ne, h6pops, Status.Work_done


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

        sys.stdout.write('Making Output File.\n')

        os.remove(ltau_out_file)

        fo = h5py.File(ltau_out_file, 'w')
        if 'ltau500' in list(fo.keys()):
            del fo['ltau500']
        if 'electron_density' in list(fo.keys()):
            del fo['electron_density']
        if 'hydrogen_populations' in list(fo.keys()):
            del fo['hydrogen_populations']
        fo['ltau500'] = np.zeros((1, 256, 512, 55), dtype=np.float64)
        fo['electron_density'] = np.zeros((1, 256, 512, 55), dtype=np.float64)
        fo['hydrogen_populations'] = np.zeros((1, 6, 256, 512, 55), dtype=np.float64)
        fo.close()

        sys.stdout.write('Made Output File.\n')

        job_matrix = np.zeros((256, 512), dtype=np.int64)

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
            item, xx, yy, ltau500, ne, h6pops = status_dict['item']
            fo = h5py.File(ltau_out_file, 'r+')
            fo['ltau500'][0, xx, yy] = ltau500
            fo['electron_density'][0, xx, yy] = ne
            fo['hydrogen_populations'][0, :, xx, yy] = h6pops.T
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

            for cmd in commands:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(sub_dir_path),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True
                )
                process.communicate()

            write_atmos_files(sub_dir_path, x, y)

            cmdstr = '/home/harsh/rh-uitenbroek/rhf1d/rhf1d'

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

            ltau500, ne, h6pops, status_work = do_work(sub_dir_path)

            comm.send({'status': status_work, 'item': (item, x, y, ltau500, ne, h6pops)}, dest=0, tag=2)
