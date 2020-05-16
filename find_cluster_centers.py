import sys
import numpy as np
from sklearn.cluster import KMeans


def cluster_with_mean_std(dataset, number_of_cluster, ignore_min=-1):

    result_dict = dict()

    for _cluster_no in np.arange(number_of_cluster):

        if _cluster_no < ignore_min:
            continue

        cluster_number = _cluster_no + 1

        sys.stdout.write(
            'Processing for Total Clusters: {}\n'.format(
                cluster_number
            )
        )

        kmeans = KMeans(n_clusters=cluster_number)

        kresult = kmeans.fit(dataset)

        std_list = list()
        for this_cluster in np.arange(cluster_number):
            std_of_this_cluster = 0
            indices = np.where(kresult.labels_ == this_cluster)[0]
            subtract_mean = np.subtract(
                dataset[indices],
                kresult.cluster_centers_[this_cluster]
            )
            squarred_subtract = np.power(
                2,
                subtract_mean
            )
            std_of_this_cluster = np.sqrt(np.sum(squarred_subtract))

            std_list.append(std_of_this_cluster)

        result_dict[cluster_number] = np.mean(std_list)

    return result_dict
