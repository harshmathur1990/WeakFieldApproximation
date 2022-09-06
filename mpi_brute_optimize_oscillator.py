import os
import sys

import h5py as h5py
import scipy.optimize
from lightweaver.fal import Falc82
import lightweaver as lw
import matplotlib.pyplot as plt
import time
import numpy as np
from pathlib import Path
from lightweaver_extension import conv_atom
import multiprocessing
from lightweaver.utils import vac_to_air, air_to_vac
from lmfit import Parameters, minimize
from mpi4py import MPI


base_path = Path(
    '/home/harsh/CourseworkRepo/rh/RH-uitenbroek/Atoms'
)
atoms_no_substructure_list = [
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

atoms_with_substructure_list = [
    'H_6_sublevel_strict.atom',
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

out_file = 'f_values_out.h5'


wave = np.arange(6564.5-4, 6564.5 + 4, 0.01) / 10

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
    lw.iterate_ctx_se(ctx)
    # Update the background populations based on the converged solution and
    # compute the final intensity for mu=1 on the provided wavelength grid.
    eqPops.update_lte_atoms_Hmin_pops(atmos)
    Iwave = ctx.compute_rays(wave, [atmos.muz[-1]], stokes=False)
    # return ctx, Iwave

    return ctx, Iwave


def get_observation():
    catalog = np.loadtxt('catalog_6563.txt')

    obs_wave, intensity = catalog[:, 0], catalog[:, 1]

    indices = list()

    for w in vac_to_air(wave):
        indices.append(np.argmin(np.abs(obs_wave - w * 10)))

    indices = np.array(indices)

    return intensity[indices]


def synthesize(f_values, waver, threads=1):

    line_indices = [
        (5, 1),
        (5, 3),
        (7, 2),
        (4, 2),
        (6, 1),
        (8, 3),
        (6, 3)
    ]

    # atoms_no_substructure = list()

    atoms_with_substructure = list()

    # for at in atoms_no_substructure_list:
    #     atoms_no_substructure.append(conv_atom(base_path / at))

    for at in atoms_with_substructure_list:
        atoms_with_substructure.append(conv_atom(base_path / at))

    # total_gf = 0

    h_with_substructure = atoms_with_substructure[0]

    index = 0

    for line_indice in line_indices:
        for line in h_with_substructure.lines:
            if line.j == line_indice[0] and line.i == line_indice[1]:
                line.f = f_values[index]
                # total_gf += f_values[index] * line.iLevel.g
                index += 1
                break

    h_with_substructure.recompute_radiative_broadening()
    h_with_substructure.recompute_collisional_rates()

    # total_gf /= 8
    #
    # h_without_substructure = atoms_no_substructure[0]
    #
    # h_without_substructure.lines[4].f = total_gf
    # h_without_substructure.recompute_radiative_broadening()
    # h_without_substructure.recompute_collisional_rates()

    h_with_substructure.__post_init__()
    # h_without_substructure.__post_init__()

    fal = Falc82()
    _, i_obs_1 = synthesize_line(
        atoms=atoms_with_substructure,
        atmos=fal,
        conserve=False,
        useNe=True,
        wave=waver,
        threads=threads
    )


    return i_obs_1, None


def cost_function(synth1, synth2=None, observation=None, weights=None):

    if observation is None:
        observation = np.zeros_like(synth1)
    else:
        observation /= observation[0]

    if weights is None:
        weights = np.ones_like(synth1)

    synth1 /= synth1[0]

    diff = np.subtract(
        synth1,
        observation
    )

    diff /= weights

    return diff


def minimization_func(params, waver, observation=None, weights=None, threads=1):

    f_values = np.zeros(7, dtype=np.float64)
    f_values[0] = params['f0']
    f_values[1] = params['f1']
    f_values[2] = params['f2']
    f_values[3] = params['f3']
    f_values[4] = params['f4']
    f_values[5] = params['f5']
    f_values[6] = params['f6']

    f_this = f_values

    i_obs_1, i_obs_2 = synthesize(f_this, waver, threads=threads)

    return cost_function(synth1=i_obs_1, synth2=i_obs_2, observation=observation, weights=weights)


if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    stop_work = False

    f_values_init = np.array([0.75, 0.75, 0.001, 0.001, 0.75, 0.001, 0.75])

    range_val = 0.25

    step = 0.1

    if rank == 0:
        status = MPI.Status()
        waiting_queue = set()
        running_queue = set()
        finished_queue = set()
        failure_queue = set()

        sys.stdout.write('Making Output File.\n')

        if not os.path.exists(out_file):
            fo = h5py.File(out_file, 'w')
            fo = np.zeros((7, ))
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