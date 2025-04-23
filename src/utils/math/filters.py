from numpy.typing import NDArray

import numpy as np
from scipy import ndimage


def cluster_size_thresholding(array, min_size: int = None, percentile: float = None):
    """
    Zero out islands in the array that are smaller than min_size or not in the top percentile_threshold.

    Parameters
    ----------
    array: NDArray
        2D array containing data.

    min_size: int
        Minimum size of islands to retain.

    percentile: float, optional
        Percentile threshold to retain top islands by size.

    Returns
    -------
    NDArray: 2D array with small islands zeroed out.

    """
    if min_size is None and percentile is None:
        raise ValueError("Either min_size or percentile_threshold must be provided.")

    # Create a copy of the array to avoid modifying the original
    array_copy = np.copy(array)

    # Define the structure for 8-connectivity
    structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    # Label connected components using 8-connectivity
    labeled_array, num_features = ndimage.label(array_copy, structure=structure)

    # Calculate the size of each island
    island_sizes = []
    for i in range(1, num_features + 1):
        island_sizes.append((i, np.sum(labeled_array == i)))

    # Determine the size threshold if percentile_threshold is provided
    if percentile is not None:
        if not (0 < percentile <= 100):
            raise ValueError("percentile_threshold must be between 0 and 100.")

        sizes = [size for _, size in island_sizes]
        size_threshold = np.percentile(sizes, percentile)
    else:
        size_threshold = 0  # Default to zero if percentile_threshold is not provided

    # Iterate over the labeled components
    for i, size in island_sizes:
        # Zero out the component if it's smaller than the size threshold
        if size < max(min_size or 0, size_threshold):
            array_copy[labeled_array == i] = 0

    return array_copy, size_threshold


def gaussian(arr: NDArray, sigma: float):
    """Gaussian filtering, allowing intensity to leak into NaN regions.

    See Also
    --------
    https://stackoverflow.com/a/36307291/7128154

    """
    gauss = arr.copy()
    gauss[np.isnan(gauss)] = 0
    gauss = ndimage.gaussian_filter(gauss, sigma=sigma, mode="constant", cval=0)

    norm = np.ones(shape=arr.shape)
    norm[np.isnan(arr)] = 0
    norm = ndimage.gaussian_filter(norm, sigma=sigma, mode="constant", cval=0)

    # avoid RuntimeWarning: invalid value encountered in true_divide
    norm = np.where(norm == 0, 1, norm)
    gauss = gauss / norm
    gauss[np.isnan(arr)] = np.nan

    return gauss
