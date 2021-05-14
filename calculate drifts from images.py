import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops


def get_drift_data(
    data,
    ref_centroid_y,
    ref_centroid_x,
    arcsec_per_mm=3.3,
    drift_y=None,
    drift_x=None
):

    k = -2

    if drift_y is None:
        drift_y = list()

    if drift_x is None:
        drift_x = list()

    for i in range(data.shape[0]):
        mean = data[i].mean()
        std = data[i].std()
        label_image = data[i] < (mean +(k *std))
        regions = regionprops(label_image.astype(np.int64), intensity_image=data[i])
        centroid_y, centroid_x = regions[0].centroid
        drift_y.append(centroid_y - ref_centroid_y)
        drift_x.append(centroid_x - ref_centroid_x)

    return drift_y, drift_x


def get_all_drifts(data_list):
    k = -2
    arcsec_per_mm=3.3
    mean = data_list[0][0].mean()
    std = data_list[0][0].std()

    ref_label_image = data_list[0][0] < (mean + (k *std))

    regions = regionprops(ref_label_image.astype(np.int64), intensity_image=data_list[0][0])

    centroid_y, centroid_x = regions[0].centroid

    drift_y = list()

    drift_x = list()

    for data in data_list:
        get_drift_data(data, centroid_y, centroid_x, drift_y=drift_y, drift_x=drift_x)

    drift_y, drift_x = np.array(drift_y), np.array(drift_x)

    drift_in_mm_y = drift_y * 6.5e-3

    drift_in_mm_x = drift_x * 6.5e-3

    drift_in_arcsec_y = drift_in_mm_y * arcsec_per_mm

    drift_in_arcsec_x = drift_in_mm_x * arcsec_per_mm

    drift_in_mm_y -= drift_in_mm_y.mean()

    drift_in_mm_x -= drift_in_mm_x.mean()

    drift_in_arcsec_y -= drift_in_arcsec_y.mean()

    drift_in_arcsec_x -= drift_in_arcsec_y.mean()

    return drift_in_mm_y, drift_in_mm_x, drift_in_arcsec_y, drift_in_arcsec_x