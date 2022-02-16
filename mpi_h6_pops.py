import sys
import enum
import numpy as np
import h5py
from mpi4py import MPI
from pathlib import Path
import shutil
import sunpy.io


output_path = Path('/home/harsh/BifrostRun/bifrost_supplementary_outputs_using_RH/')


class Status(enum.Enum):
    Requesting_work = 0
    Work_assigned = 1
    Work_done = 2
    Work_failure = 3


def do_work(x, y, read_path):


    return Status.Work_done


if __name__ == '__main__':

    foldername = '/data/harsh/ar098192/atmos',
    simulation_name = 'ar098192',
    snap = 294000,
    start_x = 0
    end_x = 256,
    start_y = 0
    end_y = 512,
    height_min_in_m = -500 * 1e-3,
    height_max_in_m = 3000 * 1e3,
    simulation_code_name = 'MURaM',

    temp_file = '{}_{}_lgtg_{}.fits'.format(
        simulation_code_name,
        simulation_name,
        snap
    )

    height, _ = sunpy.io.fits.read(
        foldername / temp_file
    )[1]

    height = height * 1e6

    ind = np.where((height >= height_min_in_m) & (height <= height_max_in_m))[0]

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
            item, xx, yy = status_dict['item']
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

            # cmdstr = '/home/harsh/RH-Old/rhf1d/rhf1d'

            cmdstr = '/home/harsh/CourseworkRepo/rh/rhv2src/rhf1d/rhf1d'

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
            status = do_work(x, y, sub_dir_path)
            # sys.stdout.write(
            #     'Rank: {} RH Save Time: {}\n'.format(
            #         rank, time.time() - start_time
            #     )
            # )
            comm.send({'status': Status.Work_done, 'item': (item, x, y)}, dest=0, tag=2)
