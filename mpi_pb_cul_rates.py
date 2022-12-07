import sys
import enum
import os
import numpy as np
import h5py
from mpi4py import MPI
from pathlib import Path
import shutil
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import tables as tb


atmos_file = Path(
    '/home/harsh/BifrostRun_fast_Access/BIFROST_en024048_hion_snap_385_0_504_0_504_-1020996.0_15000000.0.nc'
)

ltau_out_file = Path(
    '/home/harsh/BifrostRun_fast_Access/MULTI3D_BIFROST_en024048_hion_snap_385_0_504_0_504_-1020996.0_15000000.0_pb_rates.nc'
)

stop_file = Path('/home/harsh/CourseworkRepo/WFAComparison/stop')

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


class Status(enum.Enum):
    Requesting_work = 0
    Work_assigned = 1
    Work_done = 2
    Work_failure = 3


def prepare_get_pb_rates_for_rh():
    atomic_level_to_n_mapper = np.array([1, 2, 2, 2, 3, 3, 3, 3, 3, 3])

    atomic_level_to_g_mapper = np.array([2, 2, 2, 4, 2, 2, 4, 4, 6])

    n_to_g_mapper = np.array([-1, 2, 8, 18])

    PB04 = [
        [6.40e-1, 6.98e-1, 7.57e-1, 8.09e-1, 8.97e-1, 9.78e-1, 1.06e+0, 1.15e+0, 1.32e+0, 1.51e+0, 1.68e+0, 2.02e+0,
         2.33e+0, 2.97e+0, 3.50e+0, 3.95e+0],
        [2.20e-1, 2.40e-1, 2.50e-1, 2.61e-1, 2.88e-1, 3.22e-1, 3.59e-1, 3.96e-1, 4.64e-1, 5.26e-1, 5.79e-1, 6.70e-1,
         7.43e-1, 8.80e-1, 9.79e-1, 1.06e+0],
        [9.93e-2, 1.02e-1, 1.10e-1, 1.22e-1, 1.51e-1, 1.80e-1, 2.06e-1, 2.28e-1, 2.66e-1, 2.95e-1, 3.18e-1, 3.55e-1,
         3.83e-1, 4.30e-1, 4.63e-1, 4.88e-1],
        [4.92e-2, 5.84e-2, 7.17e-2, 8.58e-2, 1.12e-1, 1.33e-1, 1.50e-1, 1.64e-1, 1.85e-1, 2.01e-1, 2.12e-1, 2.29e-1,
         2.39e-1, 2.59e-1, 2.71e-1, 2.81e-1],
        [2.97e-2, 4.66e-2, 6.28e-2, 7.68e-2, 9.82e-2, 1.14e-1, 1.25e-1, 1.33e-1, 1.45e-1, 1.53e-1, 1.58e-1, 1.65e-1,
         1.70e-1, 1.77e-1, 1.82e-1, 1.85e-1],
        [5.03e-2, 6.72e-2, 7.86e-2, 8.74e-2, 1.00e-1, 1.10e-1, 1.16e-1, 1.21e-1, 1.27e-1, 1.31e-1, 1.34e-1, 1.36e-1,
         1.37e-1, 1.39e-1, 1.39e-1, 1.40e-1],
        [2.35e+1, 2.78e+1, 3.09e+1, 3.38e+1, 4.01e+1, 4.71e+1, 5.45e+1, 6.20e+1, 7.71e+1, 9.14e+1, 1.05e+2, 1.29e+2,
         1.51e+2, 1.93e+2, 2.26e+2, 2.52e+2],
        [1.07e+1, 1.15e+1, 1.23e+1, 1.34e+1, 1.62e+1, 1.90e+1, 2.18e+1, 2.44e+1, 2.89e+1, 3.27e+1, 3.60e+1, 4.14e+1,
         4.56e+1, 5.31e+1, 5.83e+1, 6.23e+1],
        [5.22e+0, 5.90e+0, 6.96e+0, 8.15e+0, 1.04e+1, 1.23e+1, 1.39e+1, 1.52e+1, 1.74e+1, 1.90e+1, 2.03e+1, 2.23e+1,
         2.37e+1, 2.61e+1, 2.78e+1, 2.89e+1],
        [2.91e+0, 4.53e+0, 6.06e+0, 7.32e+0, 9.17e+0, 1.05e+1, 1.14e+1, 1.21e+1, 1.31e+1, 1.38e+1, 1.44e+1, 1.51e+1,
         1.56e+1, 1.63e+1, 1.68e+1, 1.71e+1],
        [5.25e+0, 7.26e+0, 8.47e+0, 9.27e+0, 1.03e+1, 1.08e+1, 1.12e+1, 1.14e+1, 1.17e+1, 1.18e+1, 1.19e+1, 1.19e+1,
         1.20e+1, 1.19e+1, 1.19e+1, 1.19e+1],
        [1.50e+2, 1.90e+2, 2.28e+2, 2.70e+2, 3.64e+2, 4.66e+2, 5.70e+2, 6.72e+2, 8.66e+2, 1.04e+3, 1.19e+3, 1.46e+3,
         1.67e+3, 2.08e+3, 2.39e+3, 2.62e+3],
        [7.89e+1, 9.01e+1, 1.07e+2, 1.26e+2, 1.66e+2, 2.03e+2, 2.37e+2, 2.68e+2, 3.19e+2, 3.62e+2, 3.98e+2, 4.53e+2,
         4.95e+2, 5.68e+2, 6.16e+2, 6.51e+2],
        [4.13e+1, 6.11e+1, 8.21e+1, 1.01e+2, 1.31e+2, 1.54e+2, 1.72e+2, 1.86e+2, 2.08e+2, 2.24e+2, 2.36e+2, 2.53e+2,
         2.65e+2, 2.83e+2, 2.94e+2, 3.02e+2],
        [7.60e+1, 1.07e+2, 1.25e+2, 1.37e+2, 1.52e+2, 1.61e+2, 1.68e+2, 1.72e+2, 1.78e+2, 1.81e+2, 1.83e+2, 1.85e+2,
         1.86e+2, 1.87e+2, 1.86e+2, 1.87e+2],
        [5.90e+2, 8.17e+2, 1.07e+3, 1.35e+3, 1.93e+3, 2.47e+3, 2.96e+3, 3.40e+3, 4.14e+3, 4.75e+3, 5.25e+3, 6.08e+3,
         6.76e+3, 8.08e+3, 9.13e+3, 1.00e+4],
        [2.94e+2, 4.21e+2, 5.78e+2, 7.36e+2, 1.02e+3, 1.26e+3, 1.46e+3, 1.64e+3, 1.92e+3, 2.15e+3, 2.33e+3, 2.61e+3,
         2.81e+3, 3.15e+3, 3.36e+3, 3.51e+3],
        [4.79e+2, 7.06e+2, 8.56e+2, 9.66e+2, 1.11e+3, 1.21e+3, 1.29e+3, 1.34e+3, 1.41e+3, 1.46e+3, 1.50e+3, 1.55e+3,
         1.57e+3, 1.61e+3, 1.62e+3, 1.63e+3],
        [1.93e+3, 2.91e+3, 4.00e+3, 5.04e+3, 6.81e+3, 8.20e+3, 9.29e+3, 1.02e+4, 1.15e+4, 1.26e+4, 1.34e+4, 1.49e+4,
         1.63e+4, 1.97e+4, 2.27e+4, 2.54e+4],
        [1.95e+3, 3.24e+3, 4.20e+3, 4.95e+3, 6.02e+3, 6.76e+3, 7.29e+3, 7.70e+3, 8.26e+3, 8.63e+3, 8.88e+3, 9.21e+3,
         9.43e+3, 9.78e+3, 1.00e+4, 1.02e+4],
        [6.81e+1, 1.17e+4, 1.50e+4, 1.73e+4, 2.03e+4, 2.21e+4, 2.33e+4, 2.41e+4, 2.52e+4, 2.60e+4, 2.69e+4, 2.90e+4,
         3.17e+4, 3.94e+4, 4.73e+4, 5.50e+4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    PB04 = np.array(PB04)

    PB04_temp = np.array(np.array([0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 15, 20, 25]) * 10000).astype(
        np.int64)

    PB_index = np.array(
        [
            [-1, 0, 1, 2, 3, 4, 5],
            [-1, -1, 6, 7, 8, 9, 10],
            [-1, -1, -1, 11, 12, 13, 14],
            [-1, -1, -1, -1, 15, 16, 17],
            [-1, -1, -1, -1, -1, 18, 19],
            [-1, -1, -1, -1, -1, -1, 20]
        ]
    )

    total_transitions = np.array(generate_radiative_transitions() + collisional_transitions())
    total_transitions = total_transitions.reshape(total_transitions.shape[0] // 2, 2)

    ni = atomic_level_to_n_mapper[total_transitions[:, 0]]
    nj = atomic_level_to_n_mapper[total_transitions[:, 1]]

    indices = PB_index[nj - 1, ni - 1]

    ngi = n_to_g_mapper[ni]

    cs = CubicSpline(PB04_temp, PB04[indices], axis=1)

    gj = atomic_level_to_g_mapper[total_transitions[:, 1]]

    ngj = n_to_g_mapper[nj]

    def get_pb_rates_for_rh(temp, electron_density_si):

        return cs(temp) * electron_density_si * 1e-6 * 8.63e-6 * gj / (ngi * np.sqrt(temp) * ngj)

    return get_pb_rates_for_rh


if __name__ == '__main__':

    nH = 9
    n_transitions = 36
    n_rad_transitions = 11

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    stop_work = False

    batch_size = 36288

    if rank == 0:
        status = MPI.Status()
        waiting_queue = set()
        running_queue = set()
        finished_queue = set()
        failure_queue = set()

        f = h5py.File(atmos_file, 'r')
        nx = f['temperature'].shape[1]
        ny = f['temperature'].shape[2]
        height_len = f['temperature'].shape[3]
        f.close()

        if not os.path.exists(ltau_out_file):
            fo = tb.open_file(ltau_out_file, mode='w', title='PB rates')
            fo.create_array(fo.root, 'Cul_pb_rates', np.zeros((nx, ny, height_len, n_transitions)), 'PB Collisional rates')
            fo.close()

        job_matrix = np.zeros((nx, ny, height_len), dtype=np.int64)

        fo = h5py.File(ltau_out_file, 'r')
        a, b, c = np.where(fo['Cul_pb_rates'][:, :, :, 0] != 0)
        job_matrix[a, b, c] = 1
        fo.close()

        x, y, z = np.where(job_matrix == 0)

        t = tqdm(total=x.shape[0])

        for i in range(x.size // batch_size):
            waiting_queue.add(i)

        for worker in range(1, size):
            if len(waiting_queue) == 0:
                break
            item = waiting_queue.pop()
            work_type = {
                'job': 'work',
                'item': (item, x[item * batch_size: item * batch_size + batch_size], y[item * batch_size: item * batch_size + batch_size], z[item * batch_size: item * batch_size + batch_size])
            }
            comm.send(work_type, dest=worker, tag=1)
            running_queue.add(item)

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
            item, xx, yy, zz, cularr = status_dict['item']
            fo = tb.open_file(ltau_out_file, mode='r+')
            print(fo.root.Cul_pb_rates.shape)
            print(cularr.shape)
            import ipdb;ipdb.set_trace()
            fo.root.Cul_pb_rates[xx][:, yy][:, :, zz] = cularr
            t.update(xx.shape[0])
            fo.close()
            running_queue.discard(item)
            if jobstatus == Status.Work_done:
                finished_queue.add(item)
            else:
                failure_queue.add(item)

            if len(waiting_queue) != 0:
                new_item = waiting_queue.pop()
                work_type = {
                    'job': 'work',
                    'item': (new_item, x[new_item * batch_size: new_item * batch_size + batch_size],
                             y[new_item * batch_size: new_item * batch_size + batch_size],
                             z[new_item * batch_size: new_item * batch_size + batch_size])
                }
                comm.send(work_type, dest=sender, tag=1)
                running_queue.add(new_item)

        for worker in range(1, size):
            work_type = {
                'job': 'stopwork'
            }
            comm.send(work_type, dest=worker, tag=1)

    if rank > 0:

        get_pb_rates_for_rh = prepare_get_pb_rates_for_rh()

        vec_get_pb_rates_for_rh = np.vectorize(get_pb_rates_for_rh, signature='(),()->(n)')

        while 1:
            work_type = comm.recv(source=0, tag=1)

            if work_type['job'] != 'work':
                break

            item, x, y, z = work_type['item']

            f = h5py.File(atmos_file, 'r')

            temp = f['temperature'][()][0, x, y, z]
            electron_density_si = f['electron_density'][()][0, x, y, z]

            f.close()

            rates = vec_get_pb_rates_for_rh(temp, electron_density_si)

            comm.send({'status': Status.Work_done, 'item': (item, x, y, z, rates)}, dest=0, tag=2)
