import numpy as np
import sys
from scipy.optimize import least_squares


def calculate_initial_wavelengths(
	valid_profile_indices,
	stokes_v,
	intensities,
	interp_fn
):

	global interp_fn2
	bisectors = list()
	for intensity in intensities:
		sys.stdout.write('{}\n'.format(intensity))
		interp_fn2 = lambda x: interp_fn(x) - intensity
		f1 = stokes_v[valid_profile_indices]
		g1 = np.ones_like(f1) * intensity
		start_index, end_index = np.argwhere(
			np.diff(np.sign(f1 - g1))
		).flatten()
		start = f['wav'][valid_profile_indices][start_index]
		end = f['wav'][valid_profile_indices][end_index]
		# sys.stdout.write(
		# 	'least_squares(interp_fn2, {}).x, least_squares(interp_fn2, {}).x\n'.format(
		# 		start, end
		# 	)
		# )
		# a, b = least_squares(interp_fn2, start).x, least_squares(interp_fn2, end).x
		bisectors.append(
			{
				'start': start,
				'end': end,
			}
		)
	return bisectors
