import h5py
import sunpy.io
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
from astropy import units as u
import astropy.coordinates
import scipy.ndimage
from sunpy.image.coalignment import mapsequence_coalign_by_match_template as mc_coalign
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicStonyhurst

# def do_align_hmi_with_hifi(wfa_image, gong_image_path, angle=180):
#
#     data, header = sunpy.io.fits.read(gong_image_path)[1]
#
#     # header['CDELT1'] = 2.44
#     #
#     # header['CDELT2'] = 2.44
#     #
#     # header['CUNIT1'] = 'arcsec'
#     #
#     # header['CUNIT2'] = 'arcsec'
#     #
#     # header['CTYPE1'] = 'HPLT-TAN'
#     #
#     # header['CTYPE2'] = 'HPLN-TAN'
#     #
#     # header['CNAME1'] = 'HPC lat'
#     #
#     # header['CNAME2'] = 'HPC lon'
#
#     gong_map = sunpy.map.Map(data, header)
#
#     spread = 80
#
#     init = (182 - spread / 2, 308 - spread / 2)
#
#     final = (182 + spread / 2, 308 + spread / 2)
#
#     y0 = init[1] * u.arcsec
#
#     x0 = init[0] * u.arcsec
#
#     xf = final[0] * u.arcsec
#
#     yf = final[1] * u.arcsec
#
#     bottom_left1 = astropy.coordinates.SkyCoord(
#         x0, y0, frame=gong_map.coordinate_frame
#     )
#
#     top_right1 = astropy.coordinates.SkyCoord(
#         xf, yf, frame=gong_map.coordinate_frame
#     )
#
#     submap = gong_map.submap(bottom_left=bottom_left1, top_right=top_right1)
#
#     resampled_gong_image = sunpy.image.resample.resample(
#         orig=submap.data,
#         dimensions=(
#             submap.data.shape[0] * submap.meta['cdelt1'] / 0.38,
#             submap.data.shape[1] * submap.meta['cdelt2'] / 0.38
#         ),
#         method='spline',
#         minusone=False
#     )
#
#     new_meta = submap.meta.copy()
#
#     new_meta['naxis1'] = resampled_gong_image.shape[0]
#     new_meta['naxis2'] = resampled_gong_image.shape[1]
#     new_meta['crpix1'] = new_meta['crpix1'] * new_meta['cdelt1'] / 0.38
#     new_meta['crpix2'] = new_meta['crpix2'] * new_meta['cdelt2'] / 0.38
#     new_meta['cdelt1'] = 0.38
#     new_meta['cdelt2'] = 0.38
#
#     new_submap = sunpy.map.Map(resampled_gong_image, new_meta)
#
#     rotated_wfa_data = scipy.ndimage.rotate(
#         wfa_image,
#         angle=angle,
#         order=3,
#         prefilter=False,
#         reshape=False,
#         cval=np.nan
#     )
#
#     rotated_wfa_data[np.where(np.isnan(rotated_wfa_data))] = 0.0
#
#     rotated_meta = new_meta.copy()
#
#     rotated_meta['naxis1'] = rotated_wfa_data.shape[0]
#
#     rotated_meta['naxis2'] = rotated_wfa_data.shape[1]
#
#     rotated_map = sunpy.map.Map(rotated_wfa_data, rotated_meta)
#
#     map_sequence = sunpy.map.Map((new_submap, rotated_map), sequence=True)
#
#     return mc_coalign(map_sequence)


if __name__ == '__main__':
    fme = h5py.File('/home/harsh/SpinorNagaraju/maps_1/stic/processed_inputs/me_results_6569.nc', 'r')

    bme = fme['B_abs'][()] * np.cos(fme['inclination_rad'][()])

    gong_image_path = '/home/harsh/SpinorNagaraju/mdi.fd_M_96m_lev182.20081204_160000_TAI.data.fits'

    data, header = sunpy.io.fits.read(gong_image_path)[1]

    gong_map = sunpy.map.Map(data, header)

    rotated_map = gong_map

    # rotated_map = gong_map.rotate(recenter=True, scale=gong_map.meta['cdelt1'] / 0.38, order=3)

    bottom_left = SkyCoord(10 * u.deg, 25 * u.deg,
                           frame=HeliographicStonyhurst, obstime=rotated_map.date)

    fig = plt.figure()
    ax = fig.add_subplot(projection=rotated_map)
    rotated_map.plot(axes=ax)
    rotated_map.draw_grid(axes=ax)
    rotated_map.draw_quadrangle(bottom_left, axes=ax, width=10 * u.deg, height=10 * u.deg,
                        edgecolor='blue', linewidth=2)
    ax.plot_coord(bottom_left, 'x', color='red')

    plt.show()
