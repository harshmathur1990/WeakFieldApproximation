import sys
# sys.path.insert(1, '/home/harsh/rh-uitenbroek/python/')
sys.path.insert(1, '/home/harsh/CourseworkRepo/rh/RH-uitenbroek/python')
import enum
import os
import numpy as np
import h5py
import xdrlib
from mpi4py import MPI
from pathlib import Path
import tables as tb
import shutil
from helita.sim import multi, rh15d
import subprocess
import rhanalyze
import time
from specutils.utils.wcs_utils import air_to_vac
from astropy import units as u


# rh_base_dir = Path('/home/harsh/rh-uitenbroek/')

rh_base_dir = Path('/home/harsh/CourseworkRepo/rh/RH-uitenbroek/')

# falc_path = Path('/home/harsh/rh/Atmos/FALC_82_5x5.hdf5')

falc_path = Path('/home/harsh/CourseworkRepo/rh/rh/Atmos/FALC_82_5x5.hdf5')

# response_function_out_file = Path(
#     '/data/harsh/run_vishnu/FALC_response.nc'
# )

response_function_out_file = Path(
    '/home/harsh/run_vishnu/FALC_response.nc'
)

rh_run_base_dirs = Path('/home/harsh/run_vishnu/')

# rh_run_base_dirs = Path('/data/harsh/run_vishnu/')

stop_file = rh_run_base_dirs / 'stop'

sub_dir_format = 'process_{}'

input_filelist = [
    'molecules.input',
    'kurucz.input',
    'atoms.input',
    'keyword.input',
    'ray.input',
    'contribute.input',
    'kurucz_6173.input',
    'VishnuWave.wave'
]

b_list = list(np.array([50, -50, 200, -200, 500, -500, 1000, -1000, 2000, -2000, 3000, -3000]) * 1e-4)

regions = [
    # [8534.5900, 0.01068, 1404, 4.227743e-08, 4],
    # [6294.5000, 0.00788, 1904, 4.054384e-08, 4],
    # [5242.7086, 0.00656, 2286, 3.548769e-08, 4],
    [6165.8352, 0.00772, 1943, 4.014861e-08, 4]
]

total_wave = regions[0][2]# + regions[1][2] + regions[2][2]


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


def write_atmos_files(write_path, b_val, height_len, height_index=-1, multiplicative_factor=0):
    atmos_filename = 'Atmos1D.atmos'
    f = h5py.File(falc_path, 'r')
    multi.watmos_multi(
        str(write_path / atmos_filename),
        f['temperature'][0, 0, 0],
        f['electron_density'][0, 0, 0] / 1e6,
        z=f['z'][0] / 1e3,
        vz=f['velocity_z'][0, 0, 0] / 1e3,
        vturb=f['velocity_turbulent'][0, 0, 0] / 1e3,
        nh=f['hydrogen_populations'][0, :, 0, 0] / 1e6,
        id='FALC',
        scale='height'
    )
    Bz = np.ones(height_len) * b_val
    perturbation = None
    if height_index != -1:
        perturbation = Bz[height_index] * multiplicative_factor
        Bz[height_index] += perturbation
    create_mag_file(
        Bx=np.zeros(height_len),
        By=np.zeros(height_len),
        Bz=Bz,
        write_path=write_path,
        height_len=height_len
    )
    f.close()
    return perturbation


class Status(enum.Enum):
    Requesting_work = 0
    Work_assigned = 1
    Work_done = 2
    Work_failure = 3


def get_original_profile(read_path):
    cwd = os.getcwd()

    os.chdir(read_path)

    try:
        out = rhanalyze.rhout()
    except Exception as e:
        sys.stdout.write('Failure at Rank: {}'.format(rank))
        comm.Abort(-1)
        # return np.zeros((total_wave, 4)), Status.Work_failure

    I = list()
    Q = list()
    U = list()
    V = list()

    for index, region in enumerate(regions):
        waves = np.linspace(region[0] / 10, (region[0] + (region[1] * region[2])) / 10, region[2])

        ind_list = list()

        for wave in waves:
            ind_list.append(np.argmin(np.abs(out.spectrum.waves - wave)))

        ind = np.array(ind_list)

        for idx, ray in out.rays.items():
            if ray.muz == 1:
                I += list(ray.I[ind] / region[-2])
                Q += list(ray.Q[ind] / region[-2])
                U += list(ray.U[ind] / region[-2])
                V += list(ray.V[ind] / region[-2])

    profiles = np.zeros((len(I), 4), dtype=np.float64)

    profiles[:, 0] = I
    profiles[:, 1] = Q
    profiles[:, 2] = U
    profiles[:, 3] = V

    ltau500 = np.log(np.array(out.geometry.tau500))

    os.chdir(cwd)

    return ltau500, profiles, Status.Work_done


def create_wave_file(wave_array_in_nm, outfile):
    new_wave = air_to_vac(wave_array_in_nm * u.nm, method='edlen1966',
                          scheme='iteration').value
    p = xdrlib.Packer()
    nw = len(new_wave)
    p.pack_int(nw)
    p.pack_farray(nw, new_wave.astype('d'), p.pack_double)
    f = open(outfile, 'wb')
    f.write(p.get_buffer())
    f.close()
    print(("Wrote %i wavelengths to file." % nw))


def create_wave_file_for_regions():
    wave_list = list()

    for region in regions:
        wave_list += list(np.linspace(region[0], region[0] + (region[1] * region[2]), region[2]))

    wave = np.array(wave_list)

    wave /= 10

    create_wave_file(wave, str(rh_run_base_dirs / 'VishnuWave.wave'))


def create_atmosphere_file():
    falc = h5py.File(falc_path, 'r')
    outfile = str(rh_run_base_dirs / 'FALC_Atmosphere_response.nc')
    T = np.zeros((1, 1, len(b_list), falc['z'][0].size), np.float64)
    vz = np.zeros((1, 1, len(b_list), falc['z'][0].size), np.float64)
    z = np.zeros((1, 1, len(b_list), falc['z'][0].size), np.float64)
    nH = np.zeros((1, falc['hydrogen_populations'].shape[1], 1, len(b_list), falc['z'][0].size), np.float64)
    ne = np.zeros((1, 1, len(b_list), falc['z'][0].size), np.float64)
    vturb = np.zeros((1, 1, len(b_list), falc['z'][0].size), np.float64)
    Bz = np.zeros((1, 1, len(b_list), falc['z'][0].size), np.float64)
    desc = 'FALC'
    T[0, 0, :] = falc['temperature'][0, 0, 0]
    vz[0, 0, :] = falc['velocity_z'][0, 0, 0]
    z[0, 0, :] = falc['z'][0]
    nH[0, :, 0] = falc['hydrogen_populations'][0, :, 0, 0][:, np.newaxis, :]
    ne[0, 0, :] = falc['electron_density'][0, 0, 0]
    vturb[0, 0, :] = falc['velocity_turbulent'][0, 0, 0]
    for index, b_val in enumerate(b_list):
        Bz[0, 0, index] = np.ones(falc['z'][0].size) * b_val

    rh15d.make_xarray_atmos(
        outfile=outfile,
        T=T,
        vz=vz,
        z=z,
        nH=nH,
        ne=ne,
        vturb=vturb,
        Bz=Bz,
        desc=desc
    )

    falc.close()

    fo = h5py.File(response_function_out_file, 'r')

    f = h5py.File(outfile, 'r+')

    for key in list(fo.keys()):
        f[key] = fo[key][()]

    wave_list = list()

    for region in regions:
        wave_list += list(np.linspace(region[0], region[0] + (region[1] * region[2]), region[2]))

    wave = np.array(wave_list)

    wave /= 10

    f['wav'] = wave

    f.close()


if __name__ == '__main__':
#     create_wave_file_for_regions()
    create_atmosphere_file()

'''
if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    stop_work = False

    if rank == 0:

        status = MPI.Status()

        sys.stdout.write('Making Output File.\n')

        f = h5py.File(falc_path, 'r')
        height_len = f['temperature'].shape[3]
        f.close()

        try:
            os.remove(response_function_out_file)
        except:
            pass

        fo = h5py.File(response_function_out_file, 'w')
        fo['profiles'] = np.zeros((1, 1, len(b_list), total_wave, 4), dtype=np.float64)
        fo['derivatives'] = np.zeros((1, 1, len(b_list), height_len, total_wave, 4), dtype=np.float64)
        fo['ltau500'] = np.zeros((1, 1, len(b_list), height_len), dtype=np.float64)
        fo.close()

        sys.stdout.write('Made Output File.\n')

        for index, b_val in enumerate(b_list):

            waiting_queue = set()
            running_queue = set()
            finished_queue = set()
            failure_queue = set()

            heights = np.arange(height_len)

            for i in range(heights.size):
                waiting_queue.add(i)

            for worker in range(1, size):
                if len(waiting_queue) == 0:
                    break
                item = waiting_queue.pop()
                work_type = {
                    'job': 'work',
                    'item': (item, b_val, heights[item], height_len)
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
                item, b_val, heights[item], height_len, stokes_profiles, response, ltau500 = status_dict['item']
                fo = h5py.File(response_function_out_file, 'r+')
                fo['profiles'][0, 0, index] = stokes_profiles
                fo['derivatives'][0, 0, index, heights[item]] = response
                fo['ltau500'][0, 0, index] = ltau500
                fo.close()
                sys.stdout.write(
                    'Sender: {} b_val: {} heights[item]: {} Status: {}\n'.format(
                        sender, b_val, heights[item], jobstatus.value
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
                        'item': (new_item, b_val, heights[new_item], height_len)
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

            item, b_val, height_index, height_len = work_type['item']

            sys.stdout.write(
                'Rank: {} b_val: {} height_index: {} Generating Profiles\n'.format(
                    rank, b_val, height_index
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

                process.wait()

            write_atmos_files(sub_dir_path, b_val, height_len)

            cmdstr_1 = str(rh_base_dir / 'rhf1d/rhf1d')

            cmdstr_2 = str(rh_base_dir / 'rhf1d/solveray')

            command = '{} 2>&1 | tee output.txt && {} 2>&1 | tee -a output.txt'.format(
                cmdstr_1, cmdstr_2
            )

            process = subprocess.Popen(
                command,
                cwd=str(sub_dir_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )

            process.communicate()

            process.wait()

            ltau500, profiles, status_work = get_original_profile(sub_dir_path)

            sys.stdout.write(
                'Rank: {} bval: {} height_index: {} Generating Plus Profiles\n'.format(
                    rank, b_val, height_index
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

                process.wait()

            plus_perturbation = write_atmos_files(sub_dir_path, b_val, height_len, height_index=height_index, multiplicative_factor=0.01)

            cmdstr_1 = str(rh_base_dir / 'rhf1d/rhf1d')

            cmdstr_2 = str(rh_base_dir / 'rhf1d/solveray')

            command = '{} 2>&1 | tee output.txt && {} 2>&1 | tee -a output.txt'.format(
                cmdstr_1, cmdstr_2
            )

            process = subprocess.Popen(
                command,
                cwd=str(sub_dir_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )

            process.communicate()

            process.wait()

            _, plus_profiles, status_work = get_original_profile(sub_dir_path)

            sys.stdout.write(
                'Rank: {} b_val: {} height_index: {} Generating Negative Profiles\n'.format(
                    rank, b_val, height_index
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

                process.wait()

            negative_perturbation = write_atmos_files(
                sub_dir_path, b_val, height_len,
                height_index=height_index,
                multiplicative_factor=-0.01
            )

            cmdstr_1 = str(rh_base_dir / 'rhf1d/rhf1d')

            cmdstr_2 = str(rh_base_dir / 'rhf1d/solveray')

            command = '{} 2>&1 | tee output.txt && {} 2>&1 | tee -a output.txt'.format(
                cmdstr_1, cmdstr_2
            )

            process = subprocess.Popen(
                command,
                cwd=str(sub_dir_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )

            process.communicate()

            process.wait()

            _, negative_profiles, status_work = get_original_profile(sub_dir_path)

            response = (plus_profiles - negative_profiles) / (plus_perturbation - negative_perturbation)

            comm.send({'status': status_work, 'item': (item, b_val, height_index, height_len, profiles, response, ltau500)}, dest=0, tag=2)
'''