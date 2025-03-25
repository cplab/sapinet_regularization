""" Mathematical and computational utility functions. """
import itertools

import numpy as np

import torch
from torch import Tensor

from jenkspy import jenks_breaks
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt


def count_duplicates(tensor1, tensor2, tol=1e-5):
    """
    Compute the percentage of duplicated 'rows' between two nD tensors.
    A row is defined as a slice along the first dimension.
    Handles NaN values and performs 'close' comparison for floating-point values.

    :param tensor1: First nD tensor.
    :param tensor2: Second nD tensor.
    :param tol: Tolerance for float comparisons.
    :return: Percentage of slices in tensor1 that have duplicates in tensor2.
    """
    # Ensure both tensors have the same shape except the first dimension
    assert tensor1.shape[1:] == tensor2.shape[1:], "Tensors must have the same shape except for the first dimension."

    def rows_are_close(row1, row2, tol):
        """Helper function to compare two rows, handling NaNs and float precision."""
        nan_mask1 = torch.isnan(row1)
        nan_mask2 = torch.isnan(row2)

        # Check that NaN positions are the same in both rows
        if not torch.equal(nan_mask1, nan_mask2):
            return False

        # Check that non-NaN values are close within tolerance
        return torch.allclose(row1[~nan_mask1], row2[~nan_mask2], atol=tol)

    # Count the number of duplicate rows
    duplicated_count = 0
    for row1 in tensor1:
        for row2 in tensor2:
            if rows_are_close(row1, row2, tol):
                duplicated_count += 1
                break  # Move to the next row in tensor1 once a match is found

    # Calculate the percentage of duplicated rows in tensor1
    percentage = (duplicated_count / len(tensor1)) * 100
    return percentage


def map_words_to_integers(arr):
    unique_words = np.unique(arr)  # Get unique words
    word_to_int = {word: idx + 1 for idx, word in enumerate(unique_words)}  # Create mapping
    mapped_array = np.vectorize(word_to_int.get)(arr)  # Map words to integers

    return mapped_array


def density_from_breaks(
    arr: Tensor,
    factor: int,
    smoothing: float = 0,
    inf_replacement: float = 1e-3,
    uniform: bool = True,
    kde: bool = False,
) -> Tensor:
    """Computes desired densities from observations, knowing the duplication factor.

    Smoothing should be used to flatten the prior, reflecting decreased confidence in the sample.

    Parameters
    ----------
    arr: Tensor
        Array of observations.

    factor: int
        Duplication factor.

    smoothing: float
        Smoothing factor between 0 and 1. Higher values flatten the prior and
        reflect decreased confidence in the sample.

    inf_replacement: float
        If 0s exist in observations, replace them with this value to prevent division by infinity.
        Defaults to 1e+3, but may be set to False, None, or zero to use the maximal numeric value instead.

    uniform: bool
        Whether interpolated weights should follow a 1/x distribution, corresponding to a uniform prior.
        If set to False, weights are linearly spaced, corresponding to an exponential prior.

    kde: bool
        Whether to use KDE method rather than Jenks natural breaks (latter is default).

    """
    if not uniform:
        arr = 1.0 / arr

    # force not to cover the subthreshold value range [0, 1/limit] if some observations are 0.
    arr[arr < inf_replacement] = inf_replacement

    # add a 1s observation to ensure that, across sensors, the value range covered would be locked to [1/limit, 1].
    # stabilizes utilization with automated parameter selection and makes comparisons easier.
    arr = torch.cat((arr, torch.ones(1, device=arr.device)))

    # fix smoothing value if too small given duplication factor.
    smoothing_corrected = int(max([smoothing * factor, 1]))

    # find out what the maximal value in the array is.
    max_numeric = torch.max(arr[arr < torch.inf])

    # take care of potential inf values, e.g., if input is 1/x and there were 0s in the original observation vector.
    if inf_replacement:
        arr[arr == torch.inf] = inf_replacement
    else:
        # if inf_replacement was set to False, None, or zero, use maximal numeric value.
        if max_numeric:
            arr[arr == torch.inf] = max_numeric

    # if we only have identical observations (which can happen with many 0s, or just a single observation).
    if len(torch.unique(arr)) == 1:
        # return a default linspace from that value to some cap.
        return torch.linspace(arr[0], inf_replacement if inf_replacement else max_numeric, factor)

    sm = factor // smoothing_corrected
    if sm == 1:
        # the regularized case, completely smooth (linear coverage), but scaled to the observed range of each sensor.
        n_classes = 2
    else:
        n_classes = min(len(arr), factor) if (len(arr) < sm or sm < 1) else sm
        n_classes = len(torch.unique(arr)) if len(torch.unique(arr)) < n_classes else n_classes

    breaks = torch.as_tensor(jenks_breaks(torch.unique(arr), n_classes), device=arr.device)[1:]

    # gradient method with KDE.
    if kde:
        kde = gaussian_kde(arr, bw_method=smoothing + (0.01 if smoothing == 0 else 0), weights=None)
        interp_range = np.arange(inf_replacement, 1, (1 - inf_replacement) / (factor * 100))

        # find inflection points.
        grad = np.gradient(kde.evaluate(interp_range))
        breaks = np.hstack((inf_replacement, interp_range[np.where(np.diff(np.sign(grad)))[0]], 1))

        # prune if smoothing factor yields excessive number of bisection points.
        excess = len(breaks) - factor
        if excess > 0:
            segment_widths = torch.as_tensor(
                [breaks[i + 1] - breaks[i] for i in range(len(breaks) - 1)], device=arr.device
            )
            ranked_segments = torch.argsort(segment_widths, descending=False)
            breaks = np.delete(breaks, ranked_segments[-excess:])

        # number of classes now guaranteed to be < factor.
        n_classes = len(breaks)

        # back to a tensor.
        breaks = torch.as_tensor(breaks, device=arr.device)

    initial_allocation = factor // (n_classes - 1)
    difference = factor - initial_allocation * (n_classes - 1)

    # allocate initial densities equally among variable-width segments.
    densities = torch.zeros(n_classes - 1, device=arr.device) + initial_allocation

    # compute segment widths and rank them for intelligent (de)allocation of the difference from duplication factor.
    segment_widths = torch.as_tensor([breaks[i + 1] - breaks[i] for i in range(n_classes - 1)], device=arr.device)

    sign = -1 if difference < 0 else 1
    ranked_segments = torch.argsort(segment_widths, descending=sign < 0)

    for i in range(int(difference)):
        densities[ranked_segments[i % (n_classes - 1)]] += sign

    if sum(densities) != factor:
        raise ValueError(
            f"Densities ({densities}) didn't sum to duplication factor ({factor}). " f"Check implementation."
        )

    # linspace with the densities found.
    weights = []
    for i in range(n_classes - 1):
        segment = torch.linspace(breaks[i], breaks[i + 1], int(densities[i]) + 1)
        if uniform:
            # [:-1] or [1:] prevent double sampling at middle segment borders.
            # [:-1] retains 1/limit and discards 1 (to reverse this, use [1:]).
            segment = 1 / segment[1:]
        weights.extend(segment)

    weights = torch.as_tensor(weights, device=arr.device)

    # plt.vlines(1 / weights, ymin=.999, ymax=1.001, color="crimson")
    # plt.scatter(arr, [1] * len(arr), color="black", s=.5)
    # plt.xlim((inf_replacement, 1))
    # plt.show()

    return weights


def tensor_linspace(start: Tensor, end: Tensor, steps: int, device: str = "cpu") -> Tensor:
    """Pytorch's version of linspace does not support matrices."""
    result = torch.zeros((steps, start.shape[start.dim() - 1]), device=device)
    for i in range(start.shape[start.dim() - 1]):
        result[:, i] = torch.linspace(start[i], end[i], steps)

    return result


def k_bits(n, k):
    """Generate all binary integers of length 'n' with exactly 'k' 1-bits."""
    result = []

    for bits in itertools.combinations(range(n), k):
        s = ["0"] * n
        for bit in bits:
            s[bit] = "1"
        result.append("".join(s))

    return result
