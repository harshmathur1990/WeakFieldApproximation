import sys
import enum
import traceback
import numpy as np
import os
import os.path
import sunpy.io
import h5py
from mpi4py import MPI
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path


kmeans_output_dir = '/data/harsh1/data_to_harsh/kmeans_6173_7090'
input_file_format = '/data/harsh1/data_to_harsh/fe00{}.fit'
file_numbers = range(1, 10)

framerows = None
for file_number in file_numbers:
    data, header = sunpy.io.fits.read(
        input_file_format.format(file_number),
        memmap=True
    )[0]

    data = np.transpose(data, axes=(2, 3, 1, 0))
    fr = data[150:350, 150:350, :, :]
    fr = fr.astype(np.float64)
    fr[:, :, :, 1:4] = fr[:, :, :, 1:4] / fr[:, :, :, 0][:, :, :, np.newaxis]

    fr = fr.reshape(fr.shape[0] * fr.shape[1], fr.shape[2] * fr.shape[3])
    if framerows is None:
        framerows = fr
    else:
        framerows = np.vstack((framerows, fr))

mn = np.mean(framerows, axis=0)
sd = np.std(framerows, axis=0)
weights = np.ones(framerows.shape[1])  # * 0.025
# weights[10:20] = 0.05


class Status(enum.Enum):
    Requesting_work = 0
    Work_assigned = 1
    Work_done = 2
    Work_failure = 3


def do_work(num_clusters):
    global framerows
    global mn
    global sd
    global weights

    sys.stdout.write('Processing for Num Clusters: {}\n'.format(num_clusters))

    try:

        sys.stdout.write('Process: {} Read from File\n'.format(num_clusters))

        framerows = (framerows - mn) / sd
        framerows *= weights

        model = KMeans(
            n_clusters=num_clusters,
            max_iter=10000,
            tol=1e-6
        )

        sys.stdout.write('Process: {} Before KMeans\n'.format(num_clusters))

        model.fit(framerows)

        sys.stdout.write('Process: {} Fitted KMeans\n'.format(num_clusters))

        fout = h5py.File(
            '{}/out_{}.h5'.format(kmeans_output_dir, num_clusters), 'w'
        )
        sys.stdout.write(
            'Process: {} Open file for writing\n'.format(num_clusters)
        )
        fout['cluster_centers_'] = model.cluster_centers_
        fout['labels_'] = model.labels_
        fout['inertia_'] = model.inertia_
        fout['n_iter_'] = model.n_iter_

        rps = np.zeros((num_clusters, 23, 5), dtype=np.float64)

        framerows /= weights
        framerows = (framerows * sd) - mn

        fdr = framerows.reshape(framerows.shape[0], 23, 5)
        for i in range(num_clusters):
            a = np.where(model.labels_ == i)
            rps[i] = np.mean(fdr[a], axis=0)

        fout['rps'] = rps

        sys.stdout.write('Process: {} Wrote to file\n'.format(num_clusters))
        fout.close()
        sys.stdout.write('Success for Num Clusters: {}\n'.format(num_clusters))
        return Status.Work_done
    except Exception:
        sys.stdout.write('Failed for {}\n'.format(num_clusters))
        exc = traceback.format_exc()
        sys.stdout.write(exc)
        return Status.Work_failure


def plot_inertia():

    base_path = Path(kmeans_output_dir)

    k = range(2, 100, 1)

    inertia = list()

    for k_value in k:
        f = h5py.File(
            base_path / 'out_{}.h5'.format(
                k_value
            )
        )

        inertia.append(f['inertia_'][()])
        f.close()

    inertia = np.array(inertia)

    diff_inertia = inertia[:-1]  - inertia[1:]

    plt.close('all')
    plt.clf()
    plt.cla()

    fig, axs = plt.subplots(1, 1, figsize=(5.845, 4.135,))

    axs.plot(k, inertia / 1e5, color='#364f6B')

    axs.set_xlabel('Number of Clusters')

    axs.set_ylabel(r'$\sigma_{k}\;*\;1e5$')

    axs.grid()

    axs.axvline(x=30, linestyle='--')

    axs.set_xticks([0, 20, 30, 40, 60, 80, 100])

    axs.set_xticklabels([0, 20, 30, 40, 60, 80, 100])

    ax2 = axs.twinx()

    ax2.plot(k[1:], diff_inertia / 1e5, color='#3fC1C9')

    ax2.set_ylabel(r'$\sigma_{k} - \sigma_{k+1}\;*\;1e5$')

    axs.yaxis.label.set_color('#364f6B')

    ax2.yaxis.label.set_color('#3fC1C9')

    axs.tick_params(axis='y', colors='#364f6B')

    ax2.tick_params(axis='y', colors='#3fC1C9')

    fig.tight_layout()

    fig.savefig(base_path / 'KMeansInertia.pdf', format='pdf', dpi=300)

    plt.close('all')

    plt.clf()

    plt.cla()


if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        status = MPI.Status()
        waiting_queue = set()
        running_queue = set()
        finished_queue = set()
        failure_queue = set()

        for i in range(2, 100, 1):
            waiting_queue.add(i)

        filepath = '{}/status_job.h5'.format(kmeans_output_dir)
        if os.path.exists(filepath):
            mode = 'r+'
        else:
            mode = 'w'

        f = h5py.File(filepath, mode)

        if 'finished' in list(f.keys()):
            finished = list(f['finished'][()])
        else:
            finished = list()

        for index in finished:
            waiting_queue.discard(index)

        for worker in range(1, size):
            if len(waiting_queue) == 0:
                break
            item = waiting_queue.pop()
            work_type = {
                'job': 'work',
                'item': item
            }
            comm.send(work_type, dest=worker, tag=1)
            running_queue.add(item)

        sys.stdout.write('Finished First Phase\n')

        while len(running_queue) != 0 or len(waiting_queue) != 0:
            try:
                status_dict = comm.recv(
                    source=MPI.ANY_SOURCE,
                    tag=2,
                    status=status
                )
            except Exception:
                sys.stdout.write('Failed to get\n')
                sys.exit(1)

            sender = status.Get_source()
            jobstatus = status_dict['status']
            item = status_dict['item']
            sys.stdout.write(
                'Sender: {} item: {} Status: {}\n'.format(
                    sender, item, jobstatus.value
                )
            )
            running_queue.discard(item)
            if jobstatus == Status.Work_done:
                finished_queue.add(item)
                if 'finished' in list(f.keys()):
                    del f['finished']
                finished.append(item)
                f['finished'] = finished
            else:
                failure_queue.add(item)

            if len(waiting_queue) != 0:
                new_item = waiting_queue.pop()
                work_type = {
                    'job': 'work',
                    'item': new_item
                }
                comm.send(work_type, dest=sender, tag=1)
                running_queue.add(new_item)

        f.close()

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

            item = work_type['item']

            status = do_work(item)

            comm.send({'status': status, 'item': item}, dest=0, tag=2)
