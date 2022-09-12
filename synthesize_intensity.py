import enum
import os
import sys
import traceback

import h5py as h5py
import lightweaver
from lightweaver import Atmosphere, ScaleType

import lightweaver as lw

import numpy as np
from pathlib import Path
from lightweaver_extension import conv_atom
from tqdm import tqdm
from mpi4py import MPI
from lightweaver.utils import air_to_vac


base_path = Path(
    '/home/harsh/CourseworkRepo/rh/RH-uitenbroek/Atoms'
)

# base_path = Path('/home/harsh/rh-uitenbroek/Atoms')

atoms_with_substructure_list = [
    'H_6.atom',
    'C.atom',
    'O.atom',
    'Si.atom',
    'Al.atom',
    'CaII.atom',
    'Fe.atom',
    'He.atom',
    'Mg.atom',
    'N.atom',
    'Na.atom',
    'S.atom'
]

# write_path = Path('/data/harsh/merge_bifrost_output')

write_path = Path('/home/harsh/BifrostRun/')

# atmos_file = Path(
#     '/data/harsh/merge_bifrost_output/bifrost_en024048_hion_0_504_0_504_-500000.0_3000000.0.nc'
# )

atmos_file = Path(
    '/home/harsh/BifrostRun/bifrost_en024048_hion_0_504_0_504_-500000.0_3000000.0.nc'
)

out_file = write_path / 'intensity_out.h5'

stop_file = write_path / 'stop'

wave_H = np.arange(6562.8 - 4, 6562.8 + 4, 0.01)
wave_CaIR = np.arange(8542.09 - 4, 8542.09 + 4, 0.01)
wave_CaK = np.arange(3933.68 - 4, 3933.68 + 4, 0.01)

wave1 = air_to_vac(wave_H / 10)
wave2 = air_to_vac(wave_CaIR / 10)
wave3 = air_to_vac(wave_CaK / 10)


class Status(enum.Enum):
    Requesting_work = 0
    Work_assigned = 1
    Work_done = 2
    Work_failure = 3


def synthesize_line(atoms, atmos, conserve, useNe, wave, threads=1):
    # Configure the atmospheric angular quadrature
    atmos.quadrature(5)
    # Configure the set of atomic models to use.
    aSet = lw.RadiativeSet(atoms)
    # Set H and Ca to "active" i.e. NLTE, everything else participates as an
    # LTE background.
    aSet.set_active('H', 'Ca')
    # Compute the necessary wavelength dependent information (SpectrumConfiguration).
    spect = aSet.compute_wavelength_grid()

    # Either compute the equilibrium populations at the fixed electron density
    # provided in the model, or iterate an LTE electron density and compute the
    # corresponding equilibrium populations (SpeciesStateTable).
    if useNe:
        eqPops = aSet.compute_eq_pops(atmos)
    else:
        eqPops = aSet.iterate_lte_ne_eq_pops(atmos)

    # Configure the Context which holds the state of the simulation for the
    # backend, and provides the python interface to the backend.
    # Feel free to increase Nthreads to increase the number of threads the
    # program will use.
    ctx = lw.Context(atmos, spect, eqPops, conserveCharge=conserve, Nthreads=threads)
    # Iterate the Context to convergence (using the iteration function now
    # provided by Lightweaver)
    try:
        lw.iterate_ctx_se(ctx)
    except Exception as e:
        sys.stdout.write('exception\n')
        sys.stdout.write(traceback.format_exc())
        return None, [None, None, None]
    # Update the background populations based on the converged solution and
    # compute the final intensity for mu=1 on the provided wavelength grid.
    eqPops.update_lte_atoms_Hmin_pops(atmos)
    if isinstance(wave, np.ndarray):
        Iwave = ctx.compute_rays(wave, [atmos.muz[-1]], stokes=True)
    elif isinstance(wave, list):
        Iwave = list()
        for w in wave:
            Iwave.append(ctx.compute_rays(w, [atmos.muz[-1]], stokes=True).T)
    else:
        comm.Abort(-1)

    return ctx, Iwave


def get_mag_field(
    Bx, By, Bz
):

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

    return Babs, Binc, Bazi


def make_atmos_structure(x, y):

    f = h5py.File(atmos_file, 'r')

    Babs, Binc, Bazi = get_mag_field(
        Bx=f['B_x'][0, x, y],
        By=f['B_y'][0, x, y],
        Bz=f['B_z'][0, x, y],
    )

    atmos = Atmosphere.make_1d(
        ScaleType.Geometric,
        depthScale=f['z'][0, x, y],
        temperature=f['temperature'][0, x, y],
        vlos=f['velocity_z'][0, x, y],
        vturb=np.zeros_like(f['velocity_z'][0, x, y]),
        hydrogenPops=f['hydrogen_populations'][0, :, x, y],
        B=Babs,
        gammaB=Binc,
        chiB=Bazi
    )

    f.close()

    return atmos


def get_atomic_structure():
    atoms_with_substructure = list()

    for at in atoms_with_substructure_list:
        atoms_with_substructure.append(conv_atom(base_path / at))

    return atoms_with_substructure


if __name__ == '__main__':

    # lightweaver.benchmark()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    stop_work = False

    if rank == 0:
        status = MPI.Status()
        waiting_queue = set()
        running_queue = set()

        f = h5py.File(atmos_file, 'r')
        nx = f['temperature'].shape[1]
        ny = f['temperature'].shape[2]
        height_len = f['temperature'].shape[3]
        f.close()


        if not os.path.exists(out_file):
            sys.stdout.write('Creating output file.\n')
            fo = h5py.File(out_file, 'w')
            fo['profiles_H'] = np.zeros((1, nx, ny, wave1.size, 4), dtype=np.float64)
            fo['profiles_CaIR'] = np.zeros((1, nx, ny, wave2.size, 4), dtype=np.float64)
            fo['profiles_CaK'] = np.zeros((1, nx, ny, wave3.size, 4), dtype=np.float64)
            fo['wave_H'] = wave_H
            fo['wave_CaIR'] = wave_CaIR
            fo['wave_CaK'] = wave_CaK
            fo.close()
            sys.stdout.write('Output file created.\n')

        job_matrix = np.zeros((nx, ny), dtype=np.int64)

        fo = h5py.File(out_file, 'r')
        b, c = np.where(fo['profiles_H'][0, :, :, 0, 0] != 0)
        job_matrix[b, c] = 1
        fo.close()

        x, y = np.where(job_matrix == 0)

        for i in range(x.size):
            waiting_queue.add(i)

        t = tqdm(total=x.size)

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
            item, xx, yy, intensity_1, intensity_2, intensity_3 = status_dict['item']
            if jobstatus is Status.Work_done:
                fo = h5py.File(out_file, 'r+')
                fo['profiles_H'][0, xx, yy] = intensity_1
                fo['profiles_CaIR'][0, xx, yy] = intensity_2
                fo['profiles_CaK'][0, xx, yy] = intensity_3
                fo.close()
            running_queue.discard(item)
            t.update(1)
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

        while 1:
            work_type = comm.recv(source=0, tag=1)

            if work_type['job'] != 'work':
                break

            item, x, y = work_type['item']

            x, y = 73, 72

            atoms = get_atomic_structure()

            atmos = make_atmos_structure(x, y)

            ctx, intensities = synthesize_line(
                atoms=atoms,
                atmos=atmos,
                conserve=False,
                useNe=True,
                wave=[wave1, wave2, wave3],
                threads=1
            )

            intensity_1, intensity_2, intensity_3 = tuple(intensities)

            if ctx is None:
                comm.send({'status': Status.Work_failure, 'item': (item, x, y, intensity_1, intensity_2, intensity_3)}, dest=0, tag=2)
            else:
                comm.send({'status': Status.Work_done, 'item': (item, x, y, intensity_1, intensity_2, intensity_3)}, dest=0, tag=2)
