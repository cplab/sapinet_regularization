""" Distance utility functions. """
import numpy as np
from scipy.stats import wasserstein_distance


def pairwise_wasserstein_distances(arr):
    """
    Compute pairwise Wasserstein distances between histograms represented by rows in arr, which may be
    of variable length.

    Parameters:
    - arr: numpy array with dtype=object and shape (n_classes,).

    Returns:
    - distance_matrix: numpy array of shape (n_classes, n_classes)
                       where distance_matrix[i, j] is the Wasserstein distance between histograms
                       of class i and class j.
    """
    n_classes = len(arr)

    # initialize distance matrix.
    distance_matrix = np.zeros((n_classes, n_classes))

    # compute pairwise Wasserstein distances.
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            # extract histograms for class i and class j.
            hist_i = arr[i]
            hist_j = arr[j]

            # compute Wasserstein distance between hist_i and hist_j.
            distance = wasserstein_distance(hist_i.flatten(), hist_j.flatten())

            # store the computed distance in both directions.
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix
